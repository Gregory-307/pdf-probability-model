"""M5 Pipeline Comparison: Point Prediction vs Distribution Prediction.

Pipeline 1: Data → XGBoost → Y (point)
Pipeline 2: Data → XGBoost → Distribution Parameters → Distribution → Y + uncertainty
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import temporalpdf as tpdf
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("M5 PIPELINE COMPARISON")
print("="*70)

# =============================================================================
# LOAD DATA
# =============================================================================
print("\n[1/6] Loading M5 data...")
m5 = load_dataset('autogluon/fev_datasets', 'm5_1W', split='train')

rows = []
for i in range(500):
    item = m5[i]
    n_weeks = len(item['target'])
    for t_idx in range(n_weeks):
        rows.append({
            'item_id': item['item_id'],
            'store_id': item['store_id'],
            'state_id': item['state_id'],
            'week': t_idx,
            'sales': item['target'][t_idx],
            'sell_price': item['sell_price'][t_idx],
            'snap': item[f'snap_{item["state_id"]}'][t_idx],
        })

df = pd.DataFrame(rows)
print(f"    Loaded {len(df):,} observations")

# =============================================================================
# CREATE FEATURES AND TARGETS
# =============================================================================
print("\n[2/6] Creating features...")

def create_samples(group, window=8):
    """Create samples with features and distribution parameter targets."""
    group = group.sort_values('week')
    sales = group['sales'].values
    prices = group['sell_price'].values
    snap = group['snap'].values

    records = []
    lookback = 4

    for i in range(lookback + window, len(sales)):
        # Features
        feat = {
            'sales_lag1': sales[i-1],
            'sales_lag2': sales[i-2],
            'sales_mean4': np.mean(sales[i-lookback:i]),
            'sales_std4': np.std(sales[i-lookback:i]) + 0.1,
            'price': prices[i],
            'snap': snap[i],
            'was_active': 1 if sales[i-1] > 0 else 0,
        }

        # Target for Pipeline 1: just Y
        y_target = sales[i]

        # Targets for Pipeline 2: distribution parameters
        # Fit Student-t to recent window to get location and scale
        window_vals = sales[i-window:i]
        if np.std(window_vals) > 0:
            # Fit Student-t to window
            nu, loc, scale = stats.t.fit(window_vals)
            nu = np.clip(nu, 1.5, 30)  # Reasonable bounds
            scale = max(scale, 0.1)
        else:
            nu, loc, scale = 5.0, np.mean(window_vals), 1.0

        records.append({
            **feat,
            'y': y_target,
            'dist_mu': loc,
            'dist_sigma': scale,
            'dist_nu': nu,
            'item_id': group['item_id'].iloc[0],
        })

    return pd.DataFrame(records)

all_records = []
for (item_id, store_id), group in df.groupby(['item_id', 'store_id']):
    recs = create_samples(group)
    if len(recs) > 0:
        all_records.append(recs)

data = pd.concat(all_records, ignore_index=True)
print(f"    Dataset: {len(data):,} samples")

# Train/test split
unique_items = data['item_id'].unique()
n_train = int(len(unique_items) * 0.8)
train_items, test_items = unique_items[:n_train], unique_items[n_train:]
train = data[data['item_id'].isin(train_items)]
test = data[data['item_id'].isin(test_items)]
print(f"    Train: {len(train):,} | Test: {len(test):,}")

feature_cols = ['sales_lag1', 'sales_lag2', 'sales_mean4', 'sales_std4', 'price', 'snap', 'was_active']

X_train = train[feature_cols].values
X_test = test[feature_cols].values
y_train = train['y'].values
y_test = test['y'].values

# Distribution parameter targets for Pipeline 2
dist_targets_train = train[['dist_mu', 'dist_sigma', 'dist_nu']].values
dist_targets_test = test[['dist_mu', 'dist_sigma', 'dist_nu']].values

# =============================================================================
# PIPELINE 1: Data → Model → Y
# =============================================================================
print("\n" + "="*70)
print("PIPELINE 1: Data → XGBoost → Y (point prediction)")
print("="*70)

print("\n[3/6] Training Pipeline 1...")
model_p1 = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model_p1.fit(X_train, y_train)

y_pred_p1 = model_p1.predict(X_test)
residuals_p1 = y_test - y_pred_p1

rmse_p1 = np.sqrt(np.mean(residuals_p1**2))
mae_p1 = np.mean(np.abs(residuals_p1))

print(f"    RMSE: {rmse_p1:.3f}")
print(f"    MAE:  {mae_p1:.3f}")

# =============================================================================
# PIPELINE 2: Data → Model → Distribution Parameters → Distribution → Y
# =============================================================================
print("\n" + "="*70)
print("PIPELINE 2: Data → XGBoost → Distribution Parameters → Y + uncertainty")
print("="*70)

print("\n[4/6] Training Pipeline 2 (multi-output for distribution params)...")

# Train model to predict distribution parameters (mu, sigma, nu)
model_p2 = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
)
model_p2.fit(X_train, dist_targets_train)

# Predict distribution parameters
dist_params_pred = model_p2.predict(X_test)
pred_mu = dist_params_pred[:, 0]
pred_sigma = np.clip(dist_params_pred[:, 1], 0.1, None)  # Ensure positive
pred_nu = np.clip(dist_params_pred[:, 2], 1.5, 30)  # Reasonable bounds

# Point prediction from distribution = mean of Student-t = mu (when nu > 1)
y_pred_p2 = pred_mu

residuals_p2 = y_test - y_pred_p2

rmse_p2 = np.sqrt(np.mean(residuals_p2**2))
mae_p2 = np.mean(np.abs(residuals_p2))

print(f"    RMSE: {rmse_p2:.3f}")
print(f"    MAE:  {mae_p2:.3f}")

# Compute intervals from predicted distributions
intervals_80 = []
intervals_95 = []
for i in range(len(y_test)):
    mu, sigma, nu = pred_mu[i], pred_sigma[i], pred_nu[i]
    q10 = stats.t.ppf(0.10, nu, mu, sigma)
    q90 = stats.t.ppf(0.90, nu, mu, sigma)
    q025 = stats.t.ppf(0.025, nu, mu, sigma)
    q975 = stats.t.ppf(0.975, nu, mu, sigma)
    intervals_80.append((q10, q90, q10 <= y_test[i] <= q90))
    intervals_95.append((q025, q975, q025 <= y_test[i] <= q975))

coverage_80 = np.mean([x[2] for x in intervals_80])
coverage_95 = np.mean([x[2] for x in intervals_95])

print(f"    80% Coverage: {coverage_80*100:.1f}%")
print(f"    95% Coverage: {coverage_95*100:.1f}%")

# =============================================================================
# COMPARISON
# =============================================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print(f"""
                    Pipeline 1          Pipeline 2
                    (Point only)        (Distribution)
    ────────────────────────────────────────────────────
    RMSE            {rmse_p1:.3f}              {rmse_p2:.3f}
    MAE             {mae_p1:.3f}               {mae_p2:.3f}
    80% Coverage    N/A                 {coverage_80*100:.1f}%
    95% Coverage    N/A                 {coverage_95*100:.1f}%
    ────────────────────────────────────────────────────

    Pipeline 1 output: {y_pred_p1[0]:.1f}
    Pipeline 2 output: {y_pred_p2[0]:.1f} (mu), {pred_sigma[0]:.1f} (sigma), {pred_nu[0]:.1f} (nu)
""")

# =============================================================================
# VISUALIZATION - TWO RESIDUAL PLOTS
# =============================================================================
print("[5/6] Generating comparison plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Pipeline 1 Residuals
ax = axes[0, 0]
ax.hist(residuals_p1, bins=80, alpha=0.7, color='steelblue', edgecolor='white', density=True)
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.axvline(np.mean(residuals_p1), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(residuals_p1):.2f}')
ax.set_xlabel('Residual (Actual - Predicted)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'PIPELINE 1 Residuals\nRMSE={rmse_p1:.2f}, MAE={mae_p1:.2f}', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-60, 60)

# Panel 2: Pipeline 2 Residuals
ax = axes[0, 1]
ax.hist(residuals_p2, bins=80, alpha=0.7, color='coral', edgecolor='white', density=True)
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.axvline(np.mean(residuals_p2), color='orange', linestyle='-', linewidth=2, label=f'Mean: {np.mean(residuals_p2):.2f}')
ax.set_xlabel('Residual (Actual - Distribution Mean)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'PIPELINE 2 Residuals\nRMSE={rmse_p2:.2f}, MAE={mae_p2:.2f}', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-60, 60)

# Panel 3: Overlay comparison
ax = axes[1, 0]
ax.hist(residuals_p1, bins=80, alpha=0.5, color='steelblue', edgecolor='white', density=True, label=f'Pipeline 1 (RMSE={rmse_p1:.2f})')
ax.hist(residuals_p2, bins=80, alpha=0.5, color='coral', edgecolor='white', density=True, label=f'Pipeline 2 (RMSE={rmse_p2:.2f})')
ax.axvline(0, color='red', linestyle='--', linewidth=2)
ax.set_xlabel('Residual', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Residual Comparison (Overlay)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_xlim(-60, 60)

# Panel 4: Pipeline 2 additional output - predicted distributions for a sample
ax = axes[1, 1]
# Show a few example predicted distributions
sample_indices = [0, 100, 500, 1000]
colors = ['blue', 'green', 'orange', 'red']
x_range = np.linspace(-20, 80, 200)

for idx, sample_idx in enumerate(sample_indices):
    if sample_idx < len(pred_mu):
        mu, sigma, nu = pred_mu[sample_idx], pred_sigma[sample_idx], pred_nu[sample_idx]
        pdf = stats.t.pdf(x_range, nu, mu, sigma)
        ax.plot(x_range, pdf, color=colors[idx], linewidth=2,
                label=f'Sample {sample_idx}: mu={mu:.1f}, sigma={sigma:.1f}')
        ax.axvline(y_test[sample_idx], color=colors[idx], linestyle='--', alpha=0.5)

ax.set_xlabel('Sales', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Pipeline 2: Predicted Distributions\n(dashed = actual value)', fontsize=14, fontweight='bold')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('showcase/m5_comparison.png', dpi=150, bbox_inches='tight')
print("    Saved: showcase/m5_comparison.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================
print("\n[6/6] Summary")
print("="*70)
print(f"""
PIPELINE 1: Data → XGBoost → Y
------------------------------
- Model predicts Y directly
- Output: single number
- RMSE: {rmse_p1:.3f}, MAE: {mae_p1:.3f}

PIPELINE 2: Data → XGBoost → (mu, sigma, nu) → Student-t → Y
------------------------------------------------------------
- Model predicts distribution parameters
- Distribution gives: point estimate (mu) + uncertainty (sigma) + shape (nu)
- RMSE: {rmse_p2:.3f}, MAE: {mae_p2:.3f}
- 80% Coverage: {coverage_80*100:.1f}%, 95% Coverage: {coverage_95*100:.1f}%

KEY DIFFERENCE:
- Both have residuals (both produce point predictions)
- Pipeline 2's point prediction comes FROM the distribution
- Pipeline 2 ALSO gives you uncertainty, intervals, full distribution shape
""")

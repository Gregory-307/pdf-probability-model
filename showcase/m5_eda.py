"""M5 Dataset - Exploratory Data Analysis.

Complete walkthrough with properly fitted temporal distribution.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import temporalpdf as tpdf
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# LOAD DATA WITH ALL FEATURES
# =============================================================================
print("Loading M5 data...")
m5 = load_dataset('autogluon/fev_datasets', 'm5_1W', split='train')

# Flatten to DataFrame with all features
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
            'event_sporting': item['event_Sporting'][t_idx],
            'event_cultural': item['event_Cultural'][t_idx],
            'event_national': item['event_National'][t_idx],
        })

df = pd.DataFrame(rows)
print(f"Loaded {len(df):,} observations, {df['item_id'].nunique()} products")

# =============================================================================
# CREATE FEATURES FOR MULTI-HORIZON FORECASTING
# =============================================================================
print("\nCreating features for multi-horizon forecasting...")

def create_features_multihorizon(group, max_horizon=8):
    """Create features and targets for multiple forecast horizons."""
    group = group.sort_values('week')
    sales = group['sales'].values
    prices = group['sell_price'].values
    snap = group['snap'].values

    records = []
    lookback = 4

    for i in range(lookback, len(sales) - max_horizon):
        # Features from current position
        feat = {
            'sales_lag1': sales[i-1],
            'sales_lag2': sales[i-2],
            'sales_mean4': np.mean(sales[i-lookback:i]),
            'sales_std4': np.std(sales[i-lookback:i]) + 0.1,
            'price': prices[i],
            'snap': snap[i],
            'was_active': 1 if sales[i-1] > 0 else 0,
        }

        # Targets at different horizons
        for h in range(1, max_horizon + 1):
            records.append({
                **feat,
                'horizon': h,
                'target': sales[i + h - 1],
                'item_id': group['item_id'].iloc[0],
            })

    return pd.DataFrame(records)

# Build dataset
all_records = []
for (item_id, store_id), group in df.groupby(['item_id', 'store_id']):
    recs = create_features_multihorizon(group, max_horizon=8)
    if len(recs) > 0:
        all_records.append(recs)

data = pd.concat(all_records, ignore_index=True)
print(f"Multi-horizon dataset: {len(data):,} samples")
print(f"Horizons: 1 to {data['horizon'].max()} weeks ahead")

# =============================================================================
# TRAIN XGBOOST AND MEASURE ERROR GROWTH BY HORIZON
# =============================================================================
print("\nTraining XGBoost and measuring error growth by horizon...")

feature_cols = ['sales_lag1', 'sales_lag2', 'sales_mean4', 'sales_std4', 'price', 'snap', 'was_active']

# Train/test split (by product - 80/20)
unique_items = data['item_id'].unique()
n_train_items = int(len(unique_items) * 0.8)
train_items = unique_items[:n_train_items]
test_items = unique_items[n_train_items:]
train = data[data['item_id'].isin(train_items)]
test = data[data['item_id'].isin(test_items)]
print(f"Train items: {len(train_items)}, Test items: {len(test_items)}")

# Train one model (horizon as feature)
X_train = train[feature_cols + ['horizon']].values
y_train = train['target'].values
X_test = test[feature_cols + ['horizon']].values
y_test = test['target'].values

print("Training GradientBoostingRegressor...")
model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute residuals by horizon
test = test.copy()
test['pred'] = y_pred
test['residual'] = test['target'] - test['pred']

print("\nResidual statistics by horizon:")
horizon_stats = []
for h in range(1, 9):
    h_residuals = test[test['horizon'] == h]['residual'].values
    std = np.std(h_residuals)
    horizon_stats.append({'horizon': h, 'std': std, 'n': len(h_residuals)})
    print(f"  h={h}: std={std:.2f}, n={len(h_residuals)}")

horizon_df = pd.DataFrame(horizon_stats)

# =============================================================================
# FIT VOLATILITY GROWTH MODEL
# =============================================================================
print("\nFitting volatility growth model...")

# Model: std(h) = std_0 * (1 + growth_rate)^h  or  std(h) = std_0 * sqrt(h)
# Try square root model (common for random walks)
def sqrt_model(h, std_0, scale):
    return std_0 * np.sqrt(1 + scale * (h - 1))

# Try exponential growth model
def exp_model(h, std_0, growth):
    return std_0 * (1 + growth) ** (h - 1)

# Fit both
h_vals = horizon_df['horizon'].values
std_vals = horizon_df['std'].values

try:
    popt_sqrt, _ = curve_fit(sqrt_model, h_vals, std_vals, p0=[std_vals[0], 0.5], maxfev=5000)
    std_0_sqrt, scale_sqrt = popt_sqrt
    fitted_sqrt = sqrt_model(h_vals, *popt_sqrt)
    rmse_sqrt = np.sqrt(np.mean((std_vals - fitted_sqrt)**2))
    print(f"  Sqrt model: std_0={std_0_sqrt:.2f}, scale={scale_sqrt:.3f}, RMSE={rmse_sqrt:.3f}")
except:
    rmse_sqrt = np.inf
    std_0_sqrt, scale_sqrt = std_vals[0], 0.5

try:
    popt_exp, _ = curve_fit(exp_model, h_vals, std_vals, p0=[std_vals[0], 0.1], maxfev=5000)
    std_0_exp, growth_exp = popt_exp
    fitted_exp = exp_model(h_vals, *popt_exp)
    rmse_exp = np.sqrt(np.mean((std_vals - fitted_exp)**2))
    print(f"  Exp model:  std_0={std_0_exp:.2f}, growth={growth_exp:.3f}, RMSE={rmse_exp:.3f}")
except:
    rmse_exp = np.inf
    std_0_exp, growth_exp = std_vals[0], 0.1

# Use better model
if rmse_sqrt < rmse_exp:
    print(f"\n  Using SQRT model (better fit)")
    fitted_growth = 'sqrt'
    delta_0 = std_0_sqrt
    growth_param = scale_sqrt
else:
    print(f"\n  Using EXP model (better fit)")
    fitted_growth = 'exp'
    delta_0 = std_0_exp
    growth_param = growth_exp

# =============================================================================
# FIT DISTRIBUTION TO RESIDUALS (h=1 for base distribution)
# =============================================================================
print("\nFitting distribution to h=1 residuals...")
h1_residuals = test[test['horizon'] == 1]['residual'].values

result = tpdf.compare_distributions(
    h1_residuals,
    distributions=("normal", "student_t", "nig"),
    n_folds=5,
    metric="crps"
)

print("\nCRPS by distribution (lower = better):")
for dist, score in sorted(result['mean_scores'].items(), key=lambda x: x[1]):
    marker = " <-- BEST" if dist == result['winner'] else ""
    print(f"  {dist:12s}: {score:.4f}{marker}")

# Fit the winning distribution
params_nig = tpdf.fit_nig(h1_residuals)
params_t = tpdf.fit_student_t(h1_residuals)
params_normal = tpdf.fit_normal(h1_residuals)

print(f"\nWinner: {result['winner']}")

# =============================================================================
# EDA FIGURE
# =============================================================================
print("\nGenerating EDA plot...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Panel 1: Time series with features
ax = axes[0, 0]
sample_keys = list(df.groupby(['item_id', 'store_id']).groups.keys())[:3]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for idx, key in enumerate(sample_keys):
    subset = df[(df['item_id'] == key[0]) & (df['store_id'] == key[1])].sort_values('week')
    ax.plot(subset['week'], subset['sales'], color=colors[idx], alpha=0.7, linewidth=1, label=f'{key[0][:8]}')
    zero_mask = subset['sales'].values == 0
    ax.scatter(subset['week'].values[zero_mask], subset['sales'].values[zero_mask],
               color='red', s=15, zorder=5, marker='x')
ax.scatter([], [], color='red', marker='x', label='Zero (inactive)')
ax.set_xlabel('Week')
ax.set_ylabel('Weekly Sales')
ax.set_title('Time Series (red X = inactive)')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

# Panel 2: X vs Y
ax = axes[0, 1]
h1_data = test[test['horizon'] == 1]
idx = np.random.choice(len(h1_data), size=min(3000, len(h1_data)), replace=False)
x_vals = h1_data['sales_lag1'].values[idx]
y_vals = h1_data['target'].values[idx]
active = h1_data['was_active'].values[idx]
colors_scatter = ['red' if a == 0 else 'steelblue' for a in active]
ax.scatter(x_vals, y_vals, c=colors_scatter, alpha=0.4, s=10)
ax.plot([0, np.percentile(x_vals, 99)], [0, np.percentile(x_vals, 99)], 'k--', linewidth=1, label='Y = X')
ax.scatter([], [], c='steelblue', label='Active')
ax.scatter([], [], c='red', label='Inactive')
ax.set_xlabel('X: Last Week Sales')
ax.set_ylabel('Y: This Week Sales')
ax.set_title('Prediction Task (h=1)')
ax.legend(loc='upper left', fontsize=8)
ax.set_xlim(0, np.percentile(x_vals, 99))
ax.set_ylim(0, np.percentile(y_vals, 99))
ax.grid(True, alpha=0.3)

# Panel 3: Residual distribution
ax = axes[0, 2]
ax.hist(h1_residuals, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='white', label='Residuals')
x_range = np.linspace(np.percentile(h1_residuals, 0.5), np.percentile(h1_residuals, 99.5), 200)
ax.plot(x_range, stats.norm.pdf(x_range, 0, np.std(h1_residuals)), 'g-', linewidth=2, label='Normal')
ax.set_xlabel('Residual')
ax.set_ylabel('Density')
ax.set_title(f'Residuals (h=1): skew={stats.skew(h1_residuals):.2f}, kurt={stats.kurtosis(h1_residuals):.0f}')
ax.legend()
ax.set_xlim(np.percentile(h1_residuals, 1), np.percentile(h1_residuals, 99))

# Panel 4: ERROR GROWTH BY HORIZON (key plot!)
ax = axes[1, 0]
ax.scatter(horizon_df['horizon'], horizon_df['std'], s=100, c='steelblue', zorder=5, label='Observed std')
h_fit = np.linspace(1, 8, 50)
if fitted_growth == 'sqrt':
    std_fit = sqrt_model(h_fit, std_0_sqrt, scale_sqrt)
    ax.plot(h_fit, std_fit, 'r-', linewidth=2, label=f'Sqrt fit: σ={std_0_sqrt:.1f}√(1+{scale_sqrt:.2f}(h-1))')
else:
    std_fit = exp_model(h_fit, std_0_exp, growth_exp)
    ax.plot(h_fit, std_fit, 'r-', linewidth=2, label=f'Exp fit: σ={std_0_exp:.1f}(1+{growth_exp:.2f})^(h-1)')
ax.set_xlabel('Forecast Horizon (weeks)')
ax.set_ylabel('Residual Std Dev')
ax.set_title('FITTED: Error Grows with Horizon')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 5: Log scale residuals
ax = axes[1, 1]
ax.hist(h1_residuals, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='white', label='Residuals')
ax.plot(x_range, stats.norm.pdf(x_range, 0, np.std(h1_residuals)), 'g-', linewidth=2, label='Normal')
ax.set_yscale('log')
ax.set_ylim(1e-5, 1)
ax.set_xlabel('Residual')
ax.set_ylabel('Log Density')
ax.set_title('Log Scale: Heavy Tails')
ax.legend()

# Panel 6: Summary
ax = axes[1, 2]
ax.axis('off')
summary = f"""M5 FORECASTING TASK
-------------------
Features: lag sales, price, SNAP, events
Target: weekly sales at horizon h

XGBOOST MODEL
-------------------
Trained on {len(train_items)} products
Tested on {len(test_items)} products

ERROR GROWTH (FITTED)
-------------------
Model: {'sqrt' if fitted_growth == 'sqrt' else 'exponential'}
Base std (h=1): {delta_0:.2f}
Growth param: {growth_param:.3f}

DISTRIBUTION SELECTION
-------------------
Winner: {result['winner']}
Normal CRPS: {result['mean_scores']['normal']:.3f}
Student-t CRPS: {result['mean_scores']['student_t']:.3f}
NIG CRPS: {result['mean_scores']['nig']:.3f}"""

ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))

plt.tight_layout()
plt.savefig('showcase/m5_eda.png', dpi=150, bbox_inches='tight')
print("Saved: showcase/m5_eda.png")
plt.close()

# =============================================================================
# DISTRIBUTION FITS PLOT
# =============================================================================
print("\nGenerating distribution fits plot...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
nig = tpdf.NIG()

# Panel 1: PDF fits
ax = axes[0]
ax.hist(h1_residuals, bins=80, density=True, alpha=0.5, color='gray', edgecolor='white', label='Data')
ax.plot(x_range, stats.norm.pdf(x_range, params_normal.mu_0, params_normal.sigma_0), 'g-', linewidth=2, label='Normal')
ax.plot(x_range, stats.t.pdf(x_range, params_t.nu, params_t.mu_0, params_t.sigma_0), 'orange', linewidth=2, label=f'Student-t (ν={params_t.nu:.1f})')
ax.plot(x_range, nig.pdf(x_range, 0, params_nig), 'b-', linewidth=2, label='NIG')
ax.set_xlabel('Residual')
ax.set_ylabel('Density')
ax.set_title('Distribution Fits')
ax.legend()

# Panel 2: Log scale
ax = axes[1]
ax.hist(h1_residuals, bins=80, density=True, alpha=0.5, color='gray', edgecolor='white', label='Data')
ax.plot(x_range, stats.norm.pdf(x_range, params_normal.mu_0, params_normal.sigma_0), 'g-', linewidth=2, label='Normal')
ax.plot(x_range, stats.t.pdf(x_range, params_t.nu, params_t.mu_0, params_t.sigma_0), 'orange', linewidth=2, label='Student-t')
ax.plot(x_range, nig.pdf(x_range, 0, params_nig), 'b-', linewidth=2, label='NIG')
ax.set_yscale('log')
ax.set_ylim(1e-5, 1)
ax.set_xlabel('Residual')
ax.set_ylabel('Log Density')
ax.set_title('Log Scale (tails)')
ax.legend()

# Panel 3: CRPS
ax = axes[2]
dists = ['Normal', 'Student-t', 'NIG']
scores = [result['mean_scores']['normal'], result['mean_scores']['student_t'], result['mean_scores']['nig']]
stds = [result['std_scores']['normal'], result['std_scores']['student_t'], result['std_scores']['nig']]
colors = ['green', 'orange', 'blue']
bars = ax.bar(dists, scores, yerr=stds, color=colors, alpha=0.7, capsize=5, edgecolor='black')
ax.set_ylabel('CRPS (lower = better)')
ax.set_title(f'Distribution Selection\nWinner: {result["winner"]}')
for i, (bar, score, std) in enumerate(zip(bars, scores, stds)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1,
            f'{score:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
winner_map = {'normal': 0, 'student_t': 1, 'nig': 2}
bars[winner_map[result['winner']]].set_edgecolor('red')
bars[winner_map[result['winner']]].set_linewidth(3)

plt.tight_layout()
plt.savefig('showcase/m5_distribution_fits.png', dpi=150, bbox_inches='tight')
print("Saved: showcase/m5_distribution_fits.png")
plt.close()

# =============================================================================
# TEMPORAL 3D PLOT - PROPERLY FITTED
# =============================================================================
print("\nGenerating FITTED temporal 3D plot...")

# Create NIG params with FITTED growth rate
if fitted_growth == 'sqrt':
    # For sqrt model, convert to delta_growth approximation
    # std(h) = std_0 * sqrt(1 + scale*(h-1))
    # At h=8: std(8) = std_0 * sqrt(1 + scale*7)
    # Effective growth per step ≈ (std(8)/std(1) - 1) / 7
    effective_growth = (sqrt_model(8, std_0_sqrt, scale_sqrt) / std_0_sqrt - 1) / 7
else:
    effective_growth = growth_exp

print(f"Using fitted delta_growth = {effective_growth:.4f}")

params_temporal = tpdf.NIGParameters(
    mu=params_nig.mu,
    delta=params_nig.delta,  # Base volatility from h=1 fit
    alpha=params_nig.alpha,
    beta=params_nig.beta,
    delta_growth=effective_growth,  # FITTED from multi-horizon errors
)

# Evaluate
pdf_result = tpdf.evaluate(
    tpdf.NIG(),
    params_temporal,
    value_range=(np.percentile(h1_residuals, 1), np.percentile(h1_residuals, 99)),
    time_range=(1, 8),  # Match our horizon range
    value_points=100,
    time_points=50,
)

# Plot
fig = plt.figure(figsize=(16, 5))

Z = pdf_result.pdf_matrix
X = pdf_result.value_grid
T = pdf_result.time_grid

# Panel 1: 3D surface
ax1 = fig.add_subplot(131, projection='3d')
X_mesh, T_mesh = np.meshgrid(X, T)
ax1.plot_surface(X_mesh, T_mesh, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('Residual')
ax1.set_ylabel('Horizon (weeks)')
ax1.set_zlabel('Density')
ax1.set_title('3D: Distribution Evolves (FITTED)')
ax1.view_init(elev=25, azim=45)

# Panel 2: Heatmap
ax2 = fig.add_subplot(132)
im = ax2.imshow(Z, aspect='auto', origin='lower', extent=[X[0], X[-1], T[0], T[-1]], cmap='viridis')
ax2.set_xlabel('Residual')
ax2.set_ylabel('Horizon (weeks)')
ax2.set_title(f'Heatmap (growth={effective_growth:.3f}/week)')
plt.colorbar(im, ax=ax2, label='Density')

# Panel 3: Slices + observed std
ax3 = fig.add_subplot(133)
time_indices = [0, len(T)//3, 2*len(T)//3, -1]
slice_colors = ['blue', 'green', 'orange', 'red']
for i, t_idx in enumerate(time_indices):
    t_val = T[t_idx]
    pdf_slice = Z[t_idx, :]
    ax3.plot(X, pdf_slice, color=slice_colors[i], linewidth=2, label=f'h={t_val:.0f}w')
ax3.set_xlabel('Residual')
ax3.set_ylabel('Density')
ax3.set_title('Distribution by Horizon (FITTED)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('showcase/m5_temporal_3d.png', dpi=150, bbox_inches='tight')
print("Saved: showcase/m5_temporal_3d.png")
plt.close()

# =============================================================================
# VALIDATION: Compare fitted vs observed std at each horizon
# =============================================================================
print("\n" + "="*60)
print("VALIDATION: Fitted vs Observed Error Growth")
print("="*60)
print(f"\nHorizon | Observed Std | Fitted Std | Error")
print("-" * 50)
for h in range(1, 9):
    obs_std = horizon_df[horizon_df['horizon'] == h]['std'].values[0]
    if fitted_growth == 'sqrt':
        fit_std = sqrt_model(h, std_0_sqrt, scale_sqrt)
    else:
        fit_std = exp_model(h, std_0_exp, growth_exp)
    error = abs(obs_std - fit_std) / obs_std * 100
    print(f"   {h}    |    {obs_std:.2f}     |    {fit_std:.2f}    |  {error:.1f}%")

print("\n" + "="*60)
print("COMPLETE - All plots use FITTED parameters")
print("="*60)

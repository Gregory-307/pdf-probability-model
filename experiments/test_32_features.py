"""
Test with 32 features - extended feature set with shape-aware features.

Hypothesis: More features (especially kurtosis, skewness) will help Pipeline 2
predict distribution parameters better.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

import temporalpdf as tpdf

np.random.seed(42)

# Configuration
BARRIER = 3.0
HORIZON = 10
LOOKBACK = 20

print(f"Testing {BARRIER}% barrier, {HORIZON}-day horizon, 32 features")
print("=" * 60)

# Load data
df = pd.read_csv('data/equity_returns.csv', parse_dates=['date'])
returns = df['return_pct'].values
print(f"Loaded {len(returns):,} daily returns")


def extract_32_features(window):
    """Extract 32 features including shape-aware features."""
    n = len(window)

    def safe_corr(a, b):
        if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return c if np.isfinite(c) else 0.0

    half = n // 2
    vol_first = np.std(window[:half]) if half > 1 else 0.01
    vol_second = np.std(window[half:]) if half > 1 else 0.01

    features = [
        # Original 8
        np.mean(window),                          # 1. Mean return
        np.std(window),                           # 2. Volatility
        window[-1],                               # 3. Yesterday's return
        window[-2],                               # 4. Day before yesterday
        np.min(window),                           # 5. Min return
        np.max(window),                           # 6. Max return
        np.sum(window > 0) / n,                   # 7. Positive day ratio
        np.max(window) - np.min(window),          # 8. Range

        # Shape features (9-12)
        stats.skew(window),                       # 9. Skewness
        stats.kurtosis(window),                   # 10. Kurtosis (KEY for tail modeling!)
        np.percentile(window, 5),                 # 11. 5th percentile
        np.percentile(window, 95),                # 12. 95th percentile

        # Volatility features (13-16)
        vol_first,                                # 13. First half vol
        vol_second,                               # 14. Second half vol
        vol_second / max(vol_first, 0.001),       # 15. Vol ratio
        np.abs(window).mean(),                    # 16. Mean absolute return

        # Trend features (17-20)
        window[-5:].mean() - window[:5].mean() if n >= 10 else 0,  # 17. Momentum
        safe_corr(np.arange(n), window),          # 18. Trend strength
        (window > window.mean()).sum() / n,       # 19. Above mean ratio
        window[-1] - window[0],                   # 20. Period return

        # Autocorrelation (21-24)
        safe_corr(window[:-1], window[1:]),       # 21. Lag-1 autocorr
        safe_corr(window[:-2], window[2:]) if n > 3 else 0,  # 22. Lag-2 autocorr
        safe_corr(window[:-5], window[5:]) if n > 6 else 0,  # 23. Lag-5 autocorr
        abs(safe_corr(window[:-1], window[1:])),  # 24. Abs lag-1 autocorr

        # Tail features (25-28)
        (window < np.percentile(window, 10)).sum(),  # 25. Low tail events
        (window > np.percentile(window, 90)).sum(),  # 26. High tail events
        np.abs(window).max(),                        # 27. Max abs return
        (np.abs(window) > 2 * np.std(window)).sum() if np.std(window) > 0 else 0,  # 28. Outliers

        # Recent features (29-32)
        window[-3:].mean() if n >= 3 else window[-1],  # 29. Last 3 days mean
        window[-3:].std() if n >= 3 else 0,            # 30. Last 3 days vol
        window[-1] / max(np.std(window), 0.001),       # 31. Standardized last
        (window[-1] - np.mean(window)) / max(np.std(window), 0.001),  # 32. Z-score
    ]

    # Handle NaN/Inf
    return [0.0 if not np.isfinite(f) else f for f in features]


# Prepare data
X, y_barrier, volatility_at_pred = [], [], []

for i in range(LOOKBACK, len(returns) - HORIZON):
    window = returns[i - LOOKBACK:i]
    future = returns[i:i + HORIZON]

    features = extract_32_features(window)
    X.append(features)
    volatility_at_pred.append(np.std(window))

    max_cumsum = np.max(np.cumsum(future))
    y_barrier.append(max_cumsum >= BARRIER)

X = np.array(X)
y_barrier = np.array(y_barrier)
volatility_at_pred = np.array(volatility_at_pred)

print(f"Feature matrix shape: {X.shape}")

# Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_barrier[:split], y_barrier[split:]
vol_test = volatility_at_pred[split:]

print(f"Train: {len(y_train):,} | Test: {len(y_test):,}")
print(f"Barrier hit rate - Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

# ========== PIPELINE 1 ==========
print("\n" + "=" * 60)
print("PIPELINE 1: XGBoost Classifier (32 features)")
print("=" * 60)

clf_p1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_p1.fit(X_train, y_train)
p_barrier_p1 = clf_p1.predict_proba(X_test)[:, 1]

print(f"Mean P(hit): {p_barrier_p1.mean():.3f}")

# Feature importance
importances = clf_p1.feature_importances_
feature_names = [
    'mean', 'std', 'ret_1', 'ret_2', 'min', 'max', 'pos_ratio', 'range',
    'skew', 'kurtosis', 'p5', 'p95',
    'vol_1', 'vol_2', 'vol_ratio', 'abs_mean',
    'momentum', 'trend', 'above_mean', 'period_ret',
    'ac1', 'ac2', 'ac5', 'abs_ac1',
    'low_tail', 'high_tail', 'max_abs', 'outliers',
    'last3_mean', 'last3_std', 'std_last', 'zscore'
]
top_features = sorted(zip(feature_names, importances), key=lambda x: -x[1])[:10]
print("\nTop 10 features for P1:")
for name, imp in top_features:
    print(f"  {name:15s}: {imp:.3f}")

# ========== PIPELINE 2 ==========
print("\n" + "=" * 60)
print("PIPELINE 2: Features -> Distribution Params -> Simulate (32 features)")
print("=" * 60)

# Discovery
train_returns = returns[LOOKBACK:LOOKBACK + split]
discovery = tpdf.discover(
    train_returns,
    candidates=['normal', 'student_t', 'nig'],
    cv_folds=5,
)
print(f"Best distribution: {discovery.best.upper()}")

# Create parameter targets
y_params_list = []
for i in range(LOOKBACK, len(returns) - HORIZON):
    future = returns[i:i + HORIZON]

    if discovery.best == 'nig':
        params = tpdf.fit_nig(future)
        y_params_list.append([params.mu, params.delta, params.alpha, params.beta])
    elif discovery.best == 'student_t':
        params = tpdf.fit_student_t(future)
        y_params_list.append([params.mu_0, params.sigma_0, params.nu])
    else:
        params = tpdf.fit_normal(future)
        y_params_list.append([params.mu_0, params.sigma_0])

y_params_all = np.array(y_params_list)
y_params_train = y_params_all[:split]
y_params_test = y_params_all[split:]
n_params = y_params_train.shape[1]

# Train multi-output model
model_p2 = MultiOutputRegressor(
    GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
)
model_p2.fit(X_train, y_params_train)
y_params_pred = model_p2.predict(X_test)

print(f"\nParameter prediction quality (32 features):")
param_names = ['mu', 'sigma', 'nu/alpha', 'beta'][:n_params]
for i, name in enumerate(param_names):
    corr = np.corrcoef(y_params_test[:, i], y_params_pred[:, i])[0, 1]
    print(f"  {name}: correlation = {corr:.3f}")

# Simulate
if discovery.best == 'nig':
    dist = tpdf.NIG()
elif discovery.best == 'student_t':
    dist = tpdf.StudentT()
else:
    dist = tpdf.Normal()

N_SIMS = 3000
rng = np.random.default_rng(42)

p_barrier_p2 = []
for i in range(len(y_test)):
    pred_params = y_params_pred[i]

    if discovery.best == 'nig':
        params = tpdf.NIGParameters(
            mu=pred_params[0],
            delta=max(pred_params[1], 0.01),
            alpha=max(pred_params[2], 0.1),
            beta=pred_params[3] if n_params > 3 else 0
        )
    elif discovery.best == 'student_t':
        params = tpdf.StudentTParameters(
            mu_0=pred_params[0],
            sigma_0=max(pred_params[1], 0.01),
            nu=max(pred_params[2], 2.1)
        )
    else:
        params = tpdf.NormalParameters(
            mu_0=pred_params[0],
            sigma_0=max(pred_params[1], 0.01)
        )

    all_returns = dist.sample(N_SIMS * HORIZON, 0.0, params, rng=rng)
    paths = all_returns.reshape(N_SIMS, HORIZON)
    max_cumsum = np.max(np.cumsum(paths, axis=1), axis=1)
    p_barrier_p2.append(np.mean(max_cumsum >= BARRIER))

p_barrier_p2 = np.array(p_barrier_p2)

print(f"\nMean P(hit): {p_barrier_p2.mean():.3f}")
print(f"Std P(hit):  {p_barrier_p2.std():.3f}")

# ========== RESULTS ==========
print("\n" + "=" * 60)
print("RESULTS: 32 FEATURES")
print("=" * 60)

brier_p1 = np.mean((p_barrier_p1 - y_test.astype(float)) ** 2)
brier_p2 = np.mean((p_barrier_p2 - y_test.astype(float)) ** 2)

print(f"\n{'Metric':>25} {'Pipeline 1':>15} {'Pipeline 2':>15}")
print("-" * 57)
print(f"{'Brier Score':>25} {brier_p1:>15.4f} {brier_p2:>15.4f}")
print(f"{'Mean Prediction':>25} {p_barrier_p1.mean():>15.3f} {p_barrier_p2.mean():>15.3f}")
print(f"{'Prediction Std':>25} {p_barrier_p1.std():>15.3f} {p_barrier_p2.std():>15.3f}")
print(f"{'Actual Rate':>25} {y_test.mean():>15.3f} {y_test.mean():>15.3f}")
print("-" * 57)

winner = "1" if brier_p1 < brier_p2 else "2"
print(f"\nWinner (Brier Score): Pipeline {winner}")
print(f"Brier difference: {brier_p2 - brier_p1:.4f} ({'P2 better' if brier_p2 < brier_p1 else 'P1 better'})")

# Compare to 8-feature baseline
print("\n" + "=" * 60)
print("KEY QUESTION: Did 32 features help?")
print("=" * 60)
print("Compare these results to the 8-feature baseline (v3_barrier_showcase.ipynb)")
print("Look for:")
print("  1. Did parameter correlations improve?")
print("  2. Did Brier score improve for P2?")
print("  3. Which new features are most important?")

# Save plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
prob_true_p1, prob_pred_p1 = calibration_curve(y_test, p_barrier_p1, n_bins=10)
prob_true_p2, prob_pred_p2 = calibration_curve(y_test, p_barrier_p2, n_bins=10)

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect')
ax.plot(prob_pred_p1, prob_true_p1, 'ro-', lw=2, markersize=8, label=f'P1 (Brier={brier_p1:.3f})')
ax.plot(prob_pred_p2, prob_true_p2, 'bs-', lw=2, markersize=8, label=f'P2 (Brier={brier_p2:.3f})')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Observed Hit Rate')
ax.set_title(f'Calibration: 32 Features')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.hist(p_barrier_p1, bins=30, alpha=0.5, color='red', label='Pipeline 1', density=True)
ax.hist(p_barrier_p2, bins=30, alpha=0.5, color='blue', label='Pipeline 2', density=True)
ax.axvline(y_test.mean(), color='black', ls='--', lw=2, label=f'Actual: {y_test.mean():.1%}')
ax.set_xlabel('Predicted P(barrier hit)')
ax.set_ylabel('Density')
ax.set_title('Prediction Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/test_32_features.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print(f"\nPlot saved to experiments/test_32_features.png")

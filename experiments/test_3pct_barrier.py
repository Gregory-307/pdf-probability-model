"""
Test 3% barrier specifically - the exact same test as v3_barrier_showcase but with 3% barrier.
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
BARRIER = 3.0  # Changed from 5%
HORIZON = 10
LOOKBACK = 20
N_FEATURES = 8

print(f"Testing {BARRIER}% barrier, {HORIZON}-day horizon, {N_FEATURES} features")
print("=" * 60)

# Load data
df = pd.read_csv('data/equity_returns.csv', parse_dates=['date'])
returns = df['return_pct'].values
print(f"Loaded {len(returns):,} daily returns")

# Feature extraction (same as v3 showcase)
def extract_features(window):
    n = len(window)
    return [
        np.mean(window),
        np.std(window),
        window[-1],
        window[-2],
        np.min(window),
        np.max(window),
        np.sum(window > 0) / n,
        np.max(window) - np.min(window),
    ]

# Prepare data
X, y_barrier, volatility_at_pred = [], [], []

for i in range(LOOKBACK, len(returns) - HORIZON):
    window = returns[i - LOOKBACK:i]
    future = returns[i:i + HORIZON]

    features = extract_features(window)
    X.append(features)
    volatility_at_pred.append(np.std(window))

    max_cumsum = np.max(np.cumsum(future))
    y_barrier.append(max_cumsum >= BARRIER)

X = np.array(X)
y_barrier = np.array(y_barrier)
volatility_at_pred = np.array(volatility_at_pred)

# Split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y_barrier[:split], y_barrier[split:]
vol_test = volatility_at_pred[split:]

print(f"\nTrain: {len(y_train):,} | Test: {len(y_test):,}")
print(f"Barrier hit rate - Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

# ========== PIPELINE 1 ==========
print("\n" + "=" * 60)
print("PIPELINE 1: XGBoost Classifier")
print("=" * 60)

clf_p1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
clf_p1.fit(X_train, y_train)
p_barrier_p1 = clf_p1.predict_proba(X_test)[:, 1]

print(f"Mean P(hit): {p_barrier_p1.mean():.3f}")
print(f"Actual rate: {y_test.mean():.3f}")

# ========== PIPELINE 2 ==========
print("\n" + "=" * 60)
print("PIPELINE 2: Features -> Distribution Params -> Simulate")
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

print(f"\nParameter prediction quality:")
for i, name in enumerate(['mu', 'sigma', 'nu/alpha'][:n_params]):
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
print("RESULTS")
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

# Regime analysis
vol_median = np.median(vol_test)
high_vol_mask = vol_test > vol_median
low_vol_mask = ~high_vol_mask

print("\n" + "=" * 60)
print("REGIME ANALYSIS")
print("=" * 60)

for regime, mask in [('HIGH VOLATILITY', high_vol_mask), ('LOW VOLATILITY', low_vol_mask)]:
    actual = y_test[mask].mean()
    p1_pred = p_barrier_p1[mask].mean()
    p2_pred = p_barrier_p2[mask].mean()

    p1_error = abs(p1_pred - actual)
    p2_error = abs(p2_pred - actual)

    print(f"\n{regime}:")
    print(f"  Actual: {actual:.1%}")
    print(f"  P1: {p1_pred:.1%} (error: {p1_error:.1%})")
    print(f"  P2: {p2_pred:.1%} (error: {p2_error:.1%})")
    print(f"  Better: Pipeline {'1' if p1_error < p2_error else '2'}")

# Save plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Calibration
ax = axes[0]
prob_true_p1, prob_pred_p1 = calibration_curve(y_test, p_barrier_p1, n_bins=10)
prob_true_p2, prob_pred_p2 = calibration_curve(y_test, p_barrier_p2, n_bins=10)

ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect')
ax.plot(prob_pred_p1, prob_true_p1, 'ro-', lw=2, markersize=8, label=f'P1 (Brier={brier_p1:.3f})')
ax.plot(prob_pred_p2, prob_true_p2, 'bs-', lw=2, markersize=8, label=f'P2 (Brier={brier_p2:.3f})')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Observed Hit Rate')
ax.set_title(f'Calibration: {BARRIER}% Barrier, {HORIZON}-day Horizon')
ax.legend()
ax.grid(True, alpha=0.3)

# Histograms
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
plt.savefig(f'experiments/test_{int(BARRIER)}pct_barrier.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print(f"\nPlot saved to experiments/test_{int(BARRIER)}pct_barrier.png")

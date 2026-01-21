"""
Test longer horizons: 15, 20, 30 days.

Hypothesis: The distributional approach (Pipeline 2) should perform better at longer
horizons where uncertainty accumulates and having a proper distribution matters more.
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
LOOKBACK = 20
N_FEATURES = 8
HORIZONS = [5, 10, 15, 20, 30]

print("Testing multiple horizons")
print("=" * 60)

# Load data
df = pd.read_csv('data/equity_returns.csv', parse_dates=['date'])
returns = df['return_pct'].values
print(f"Loaded {len(returns):,} daily returns")


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


def run_horizon_test(returns, horizon):
    """Run the full P1 vs P2 test for a given horizon."""
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Prepare data
    X, y_barrier, volatility_at_pred = [], [], []

    for i in range(LOOKBACK, len(returns) - horizon):
        window = returns[i - LOOKBACK:i]
        future = returns[i:i + horizon]

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

    actual_rate = y_test.mean()

    # Pipeline 1
    clf_p1 = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf_p1.fit(X_train, y_train)
    p_barrier_p1 = clf_p1.predict_proba(X_test)[:, 1]

    # Pipeline 2
    train_returns = returns[LOOKBACK:LOOKBACK + split]
    discovery = tpdf.discover(train_returns, candidates=['normal', 'student_t', 'nig'], cv_folds=5)

    y_params_list = []
    for i in range(LOOKBACK, len(returns) - horizon):
        future = returns[i:i + horizon]

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

    model_p2 = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    )
    model_p2.fit(X_train, y_params_train)
    y_params_pred = model_p2.predict(X_test)

    # Compute correlations
    corrs = []
    for i in range(n_params):
        c = np.corrcoef(y_params_test[:, i], y_params_pred[:, i])[0, 1]
        corrs.append(c if np.isfinite(c) else 0.0)

    # Simulate
    if discovery.best == 'nig':
        dist = tpdf.NIG()
    elif discovery.best == 'student_t':
        dist = tpdf.StudentT()
    else:
        dist = tpdf.Normal()

    N_SIMS = 1000  # Reduced for speed

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

        all_returns = dist.sample(N_SIMS * horizon, 0.0, params, rng=rng)
        paths = all_returns.reshape(N_SIMS, horizon)
        max_cumsum = np.max(np.cumsum(paths, axis=1), axis=1)
        p_barrier_p2.append(np.mean(max_cumsum >= BARRIER))

    p_barrier_p2 = np.array(p_barrier_p2)

    # Brier scores
    brier_p1 = np.mean((p_barrier_p1 - y_test.astype(float)) ** 2)
    brier_p2 = np.mean((p_barrier_p2 - y_test.astype(float)) ** 2)

    return {
        'horizon': horizon,
        'actual_rate': actual_rate,
        'brier_p1': brier_p1,
        'brier_p2': brier_p2,
        'brier_diff': brier_p2 - brier_p1,
        'mean_pred_p1': p_barrier_p1.mean(),
        'mean_pred_p2': p_barrier_p2.mean(),
        'std_pred_p1': p_barrier_p1.std(),
        'std_pred_p2': p_barrier_p2.std(),
        'discovery_best': discovery.best,
        'param_corr_sigma': corrs[1] if len(corrs) > 1 else 0.0,
        'winner': 'P1' if brier_p1 < brier_p2 else 'P2',
    }


# Run tests for each horizon
results = []

for horizon in HORIZONS:
    print(f"\n{'=' * 60}")
    print(f"HORIZON: {horizon} days")
    print('=' * 60)

    result = run_horizon_test(returns, horizon)
    results.append(result)

    print(f"  Actual barrier hit rate: {result['actual_rate']:.1%}")
    print(f"  Best distribution: {result['discovery_best']}")
    print(f"  Brier P1: {result['brier_p1']:.4f}")
    print(f"  Brier P2: {result['brier_p2']:.4f}")
    print(f"  Winner: {result['winner']}")

# Create summary DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('experiments/horizon_comparison_results.csv', index=False)

# Print summary table
print("\n" + "=" * 60)
print("HORIZON COMPARISON SUMMARY")
print("=" * 60)
print(f"\n{'Horizon':>8} {'Actual%':>8} {'Brier P1':>10} {'Brier P2':>10} {'Diff':>10} {'Winner':>8}")
print("-" * 60)
for r in results:
    diff_str = f"{r['brier_diff']:+.4f}"
    print(f"{r['horizon']:>8} {r['actual_rate']*100:>7.1f}% {r['brier_p1']:>10.4f} {r['brier_p2']:>10.4f} {diff_str:>10} {r['winner']:>8}")

# Key insight
p1_wins = sum(1 for r in results if r['winner'] == 'P1')
p2_wins = sum(1 for r in results if r['winner'] == 'P2')

print(f"\nP1 wins: {p1_wins} | P2 wins: {p2_wins}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

horizons = [r['horizon'] for r in results]
brier_p1 = [r['brier_p1'] for r in results]
brier_p2 = [r['brier_p2'] for r in results]
actual_rates = [r['actual_rate'] for r in results]

ax = axes[0]
width = 0.35
x = np.arange(len(horizons))
bars1 = ax.bar(x - width/2, brier_p1, width, label='Pipeline 1', color='red', alpha=0.7)
bars2 = ax.bar(x + width/2, brier_p2, width, label='Pipeline 2', color='blue', alpha=0.7)
ax.set_xlabel('Horizon (days)')
ax.set_ylabel('Brier Score (lower = better)')
ax.set_title('Brier Score by Horizon')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Mark winners
for i, r in enumerate(results):
    if r['winner'] == 'P2':
        ax.annotate('*', xy=(i + width/2, r['brier_p2']), ha='center', fontsize=14, color='blue')

ax = axes[1]
ax.plot(horizons, actual_rates, 'ko-', lw=2, markersize=10, label='Actual barrier hit rate')
ax.plot(horizons, [r['mean_pred_p1'] for r in results], 'r^--', lw=2, markersize=8, label='P1 mean prediction')
ax.plot(horizons, [r['mean_pred_p2'] for r in results], 'bs--', lw=2, markersize=8, label='P2 mean prediction')
ax.set_xlabel('Horizon (days)')
ax.set_ylabel('Probability')
ax.set_title(f'Predictions vs Actual Rate ({BARRIER}% barrier)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('experiments/horizon_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

print(f"\nResults saved to experiments/horizon_comparison_results.csv")
print(f"Plot saved to experiments/horizon_comparison.png")

print("\n" + "=" * 60)
print("KEY INSIGHT")
print("=" * 60)
print("""
At longer horizons, barrier hit rates increase (more time to hit barrier).
The question is: Does P2's distributional approach handle this better?

Look for:
1. Does P2 win more at longer horizons (where uncertainty matters more)?
2. Does P2's prediction spread (std) increase appropriately with horizon?
3. Does P1 become overconfident at longer horizons?
""")

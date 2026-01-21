"""
Run experiments using pre-cached data.

First run: python experiments/prepare_data.py
Then run: python experiments/run_experiments.py
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import temporalpdf as tpdf

# Load cached data
cache_path = Path('experiments/data_cache.pkl')
if not cache_path.exists():
    print("ERROR: Run 'python experiments/prepare_data.py' first!")
    sys.exit(1)

print("Loading cached data...", flush=True)
with open(cache_path, 'rb') as f:
    data_cache = pickle.load(f)
print("Loaded.", flush=True)


def run_experiment(horizon, barrier, n_features, distribution='student_t', test_frac=0.2, n_sims=1000):
    """Run P1 vs P2 experiment with cached data."""
    data = data_cache[horizon]

    # Select features
    X = data['X_8'] if n_features == 8 else data['X_32']
    y_barrier = data['y_barrier'][barrier]

    # Select distribution params
    if distribution == 'normal':
        y_params = data['y_params_normal']
        dist = tpdf.Normal()
        def make_params(p):
            return tpdf.NormalParameters(mu_0=p[0], sigma_0=max(p[1], 0.01))
    elif distribution == 'student_t':
        y_params = data['y_params_student_t']
        dist = tpdf.StudentT()
        def make_params(p):
            return tpdf.StudentTParameters(mu_0=p[0], sigma_0=max(p[1], 0.01), nu=max(p[2], 2.1))
    else:  # nig
        y_params = data['y_params_nig']
        dist = tpdf.NIG()
        def make_params(p):
            return tpdf.NIGParameters(mu=p[0], delta=max(p[1], 0.01), alpha=max(p[2], 0.1), beta=p[3])

    # Split
    split = int(len(X) * (1 - test_frac))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_barrier[:split], y_barrier[split:]
    y_params_train, y_params_test = y_params[:split], y_params[split:]

    # P1: Direct classification
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    p1 = clf.predict_proba(X_test)[:, 1]
    brier_p1 = np.mean((p1 - y_test.astype(float)) ** 2)

    # P2: Predict params -> simulate
    model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    )
    model.fit(X_train, y_params_train)
    y_pred = model.predict(X_test)

    rng = np.random.default_rng(42)
    p2 = []
    for i in range(len(y_test)):
        params = make_params(y_pred[i])
        samples = dist.sample(n_sims * horizon, 0.0, params, rng=rng)
        paths = samples.reshape(n_sims, horizon)
        p2.append(np.mean(np.max(np.cumsum(paths, axis=1), axis=1) >= barrier))
    p2 = np.array(p2)
    brier_p2 = np.mean((p2 - y_test.astype(float)) ** 2)

    return {
        'horizon': horizon,
        'barrier': barrier,
        'n_features': n_features,
        'distribution': distribution,
        'actual_rate': y_test.mean(),
        'brier_p1': brier_p1,
        'brier_p2': brier_p2,
        'brier_diff': brier_p2 - brier_p1,
        'mean_p1': p1.mean(),
        'mean_p2': p2.mean(),
        'winner': 'P1' if brier_p1 < brier_p2 else 'P2',
    }


if __name__ == '__main__':
    print("\nRUNNING EXPERIMENTS")
    print("=" * 70)

    results = []

    # Test matrix
    horizons = [10, 20, 30]
    barriers = [3.0, 5.0]
    feature_counts = [8, 32]

    total = len(horizons) * len(barriers) * len(feature_counts)
    i = 0

    for horizon in horizons:
        for barrier in barriers:
            for n_features in feature_counts:
                i += 1
                print(f"\n[{i}/{total}] H={horizon}d, B={barrier}%, F={n_features}", flush=True)

                result = run_experiment(horizon, barrier, n_features)
                results.append(result)

                print(f"  Actual: {result['actual_rate']:.1%}")
                print(f"  P1: {result['brier_p1']:.4f}, P2: {result['brier_p2']:.4f}")
                print(f"  Winner: {result['winner']} (diff: {result['brier_diff']:+.4f})")

    # Summary
    df = pd.DataFrame(results)
    df.to_csv('experiments/experiment_results.csv', index=False)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nP1 wins: {(df['winner'] == 'P1').sum()}")
    print(f"P2 wins: {(df['winner'] == 'P2').sum()}")

    print("\nAll results:")
    print(df[['horizon', 'barrier', 'n_features', 'actual_rate', 'brier_p1', 'brier_p2', 'winner']].to_string(index=False))

    print("\nP2 wins when:")
    p2_wins = df[df['winner'] == 'P2']
    if len(p2_wins) > 0:
        print(p2_wins[['horizon', 'barrier', 'n_features', 'brier_diff']].to_string(index=False))
    else:
        print("  (no P2 wins)")

    print(f"\nResults saved to experiments/experiment_results.csv")

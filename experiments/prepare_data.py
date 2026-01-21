"""
Prepare experiment data ONCE. Cache fitted parameters for all horizons.

Run this first, then experiments load the cached data.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from scipy import stats
import temporalpdf as tpdf
import pickle
from pathlib import Path
import time

print("PREPARING EXPERIMENT DATA")
print("=" * 60)

# Load returns
df = pd.read_csv('data/equity_returns.csv', parse_dates=['date'])
returns = df['return_pct'].values
print(f"Loaded {len(returns):,} daily returns")

# Configuration
LOOKBACK = 20
HORIZONS = [5, 10, 15, 20, 30]
BARRIERS = [2.0, 3.0, 5.0, 7.0]

def extract_8_features(window):
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

def extract_32_features(window):
    n = len(window)
    half = n // 2

    def safe_corr(a, b):
        if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
            return 0.0
        c = np.corrcoef(a, b)[0, 1]
        return c if np.isfinite(c) else 0.0

    vol_first = np.std(window[:half]) if half > 1 else 0.01
    vol_second = np.std(window[half:]) if half > 1 else 0.01

    features = [
        np.mean(window), np.std(window), window[-1], window[-2],
        np.min(window), np.max(window), np.sum(window > 0) / n,
        np.max(window) - np.min(window),
        stats.skew(window), stats.kurtosis(window),
        np.percentile(window, 5), np.percentile(window, 95),
        vol_first, vol_second, vol_second / max(vol_first, 0.001),
        np.abs(window).mean(),
        window[-5:].mean() - window[:5].mean() if n >= 10 else 0,
        safe_corr(np.arange(n), window),
        (window > window.mean()).sum() / n, window[-1] - window[0],
        safe_corr(window[:-1], window[1:]),
        safe_corr(window[:-2], window[2:]) if n > 3 else 0,
        safe_corr(window[:-5], window[5:]) if n > 6 else 0,
        abs(safe_corr(window[:-1], window[1:])),
        (window < np.percentile(window, 10)).sum(),
        (window > np.percentile(window, 90)).sum(),
        np.abs(window).max(),
        (np.abs(window) > 2 * np.std(window)).sum() if np.std(window) > 0 else 0,
        window[-3:].mean() if n >= 3 else window[-1],
        window[-3:].std() if n >= 3 else 0,
        window[-1] / max(np.std(window), 0.001),
        (window[-1] - np.mean(window)) / max(np.std(window), 0.001),
    ]
    return [0.0 if not np.isfinite(f) else f for f in features]

# Prepare data for each horizon
data_cache = {}

for horizon in HORIZONS:
    print(f"\nPreparing horizon={horizon}d...", flush=True)
    start = time.time()

    X_8, X_32, volatility = [], [], []
    y_barrier = {b: [] for b in BARRIERS}
    y_params_normal, y_params_student_t, y_params_nig = [], [], []

    n_samples = len(returns) - LOOKBACK - horizon

    for i in range(LOOKBACK, len(returns) - horizon):
        window = returns[i - LOOKBACK:i]
        future = returns[i:i + horizon]

        # Features (fast)
        X_8.append(extract_8_features(window))
        X_32.append(extract_32_features(window))
        volatility.append(np.std(window))

        # Barrier targets (fast)
        max_cumsum = np.max(np.cumsum(future))
        for b in BARRIERS:
            y_barrier[b].append(max_cumsum >= b)

        # Distribution parameters (slow - but only done once!)
        params_n = tpdf.fit_normal(future)
        y_params_normal.append([params_n.mu_0, params_n.sigma_0])

        params_t = tpdf.fit_student_t(future)
        y_params_student_t.append([params_t.mu_0, params_t.sigma_0, params_t.nu])

        params_nig = tpdf.fit_nig(future)
        y_params_nig.append([params_nig.mu, params_nig.delta, params_nig.alpha, params_nig.beta])

        # Progress
        if (i - LOOKBACK + 1) % 500 == 0:
            print(f"  {i - LOOKBACK + 1}/{n_samples}", flush=True)

    data_cache[horizon] = {
        'X_8': np.array(X_8),
        'X_32': np.array(X_32),
        'volatility': np.array(volatility),
        'y_barrier': {b: np.array(v) for b, v in y_barrier.items()},
        'y_params_normal': np.array(y_params_normal),
        'y_params_student_t': np.array(y_params_student_t),
        'y_params_nig': np.array(y_params_nig),
    }

    elapsed = time.time() - start
    print(f"  Done in {elapsed:.1f}s ({n_samples} samples)")

# Save cache
cache_path = Path('experiments/data_cache.pkl')
with open(cache_path, 'wb') as f:
    pickle.dump(data_cache, f)

print(f"\n{'=' * 60}")
print(f"Data cached to {cache_path}")
print(f"Horizons: {HORIZONS}")
print(f"Barriers: {BARRIERS}")
print(f"Feature sets: 8, 32")
print(f"Distributions: normal, student_t, nig")
print(f"\nNow run experiments with: python experiments/run_experiments.py")

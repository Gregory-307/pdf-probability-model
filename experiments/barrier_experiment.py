"""
Barrier Probability Experiment: Pipeline 1 vs Pipeline 2

Tests different configurations:
- Barrier levels: 2%, 3%, 5%, 7%
- Horizons: 5, 10, 15, 20, 30 days
- Feature counts: 8, 16, 32
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.calibration import calibration_curve
from dataclasses import dataclass
from typing import Literal
import warnings
warnings.filterwarnings('ignore')

import temporalpdf as tpdf


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    barrier: float = 3.0
    horizon: int = 10
    n_features: int = 8
    lookback: int = 20
    n_estimators: int = 100
    max_depth: int = 3
    n_sims: int = 3000
    test_fraction: float = 0.2
    random_seed: int = 42


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    config: ExperimentConfig
    brier_p1: float
    brier_p2: float
    mean_pred_p1: float
    mean_pred_p2: float
    std_pred_p1: float
    std_pred_p2: float
    actual_rate: float
    discovery_best: str
    param_corr_mu: float
    param_corr_sigma: float
    param_corr_nu: float
    high_vol_error_p1: float
    high_vol_error_p2: float
    low_vol_error_p1: float
    low_vol_error_p2: float
    winner: Literal['P1', 'P2', 'TIE']

    def to_dict(self):
        return {
            'barrier': self.config.barrier,
            'horizon': self.config.horizon,
            'n_features': self.config.n_features,
            'brier_p1': self.brier_p1,
            'brier_p2': self.brier_p2,
            'brier_diff': self.brier_p2 - self.brier_p1,
            'mean_pred_p1': self.mean_pred_p1,
            'mean_pred_p2': self.mean_pred_p2,
            'std_pred_p1': self.std_pred_p1,
            'std_pred_p2': self.std_pred_p2,
            'actual_rate': self.actual_rate,
            'discovery_best': self.discovery_best,
            'param_corr_mu': self.param_corr_mu,
            'param_corr_sigma': self.param_corr_sigma,
            'param_corr_nu': self.param_corr_nu,
            'high_vol_error_p1': self.high_vol_error_p1,
            'high_vol_error_p2': self.high_vol_error_p2,
            'low_vol_error_p1': self.low_vol_error_p1,
            'low_vol_error_p2': self.low_vol_error_p2,
            'winner': self.winner,
        }


def extract_features(window: np.ndarray, n_features: int = 8) -> list:
    """
    Extract features from a lookback window.

    Args:
        window: Array of returns in the lookback window
        n_features: 8, 16, or 32 features

    Returns:
        List of features
    """
    n = len(window)

    # Base 8 features
    features = [
        np.mean(window),                          # 1. Mean return
        np.std(window),                           # 2. Volatility
        window[-1],                               # 3. Yesterday's return
        window[-2],                               # 4. Day before yesterday
        np.min(window),                           # 5. Min return
        np.max(window),                           # 6. Max return
        np.sum(window > 0) / n,                   # 7. Positive day ratio
        np.max(window) - np.min(window),          # 8. Range
    ]

    if n_features >= 16:
        # Shape features (9-12)
        features.extend([
            stats.skew(window),                   # 9. Skewness
            stats.kurtosis(window),               # 10. Kurtosis
            np.percentile(window, 5),             # 11. 5th percentile
            np.percentile(window, 95),            # 12. 95th percentile
        ])

        # Volatility features (13-16)
        half = n // 2
        vol_first = np.std(window[:half]) if half > 1 else 0.01
        vol_second = np.std(window[half:]) if half > 1 else 0.01
        features.extend([
            vol_first,                            # 13. First half vol
            vol_second,                           # 14. Second half vol
            vol_second / max(vol_first, 0.001),   # 15. Vol ratio
            np.abs(window).mean(),                # 16. Mean absolute return
        ])

    if n_features >= 32:
        # Trend features (17-20)
        features.extend([
            window[-5:].mean() - window[:5].mean() if n >= 10 else 0,  # 17. Momentum
            np.corrcoef(np.arange(n), window)[0, 1] if n > 2 else 0,   # 18. Trend strength
            (window > window.mean()).sum() / n,                        # 19. Above mean ratio
            window[-1] - window[0],                                    # 20. Period return
        ])

        # Autocorrelation (21-24)
        def safe_corr(a, b):
            if len(a) < 3 or np.std(a) < 1e-10 or np.std(b) < 1e-10:
                return 0.0
            return np.corrcoef(a, b)[0, 1]

        features.extend([
            safe_corr(window[:-1], window[1:]),                        # 21. Lag-1 autocorr
            safe_corr(window[:-2], window[2:]) if n > 3 else 0,        # 22. Lag-2 autocorr
            safe_corr(window[:-5], window[5:]) if n > 6 else 0,        # 23. Lag-5 autocorr
            abs(safe_corr(window[:-1], window[1:])),                   # 24. Abs lag-1 autocorr
        ])

        # Tail features (25-28)
        features.extend([
            (window < np.percentile(window, 10)).sum(),                # 25. Low tail events
            (window > np.percentile(window, 90)).sum(),                # 26. High tail events
            np.abs(window).max(),                                      # 27. Max abs return
            (np.abs(window) > 2 * np.std(window)).sum() if np.std(window) > 0 else 0,  # 28. Outliers
        ])

        # Recent features (29-32)
        features.extend([
            window[-3:].mean() if n >= 3 else window[-1],              # 29. Last 3 days mean
            window[-3:].std() if n >= 3 else 0,                        # 30. Last 3 days vol
            window[-1] / max(np.std(window), 0.001),                   # 31. Standardized last
            (window[-1] - np.mean(window)) / max(np.std(window), 0.001),  # 32. Z-score
        ])

    # Handle NaN/Inf
    features = [0.0 if not np.isfinite(f) else f for f in features]

    return features[:n_features]


def run_experiment(config: ExperimentConfig, returns: np.ndarray) -> ExperimentResult:
    """
    Run a single experiment with given configuration.

    Args:
        config: Experiment configuration
        returns: Array of daily returns

    Returns:
        ExperimentResult with all metrics
    """
    np.random.seed(config.random_seed)
    rng = np.random.default_rng(config.random_seed)

    # Prepare data
    X, y_barrier, volatility_at_pred = [], [], []

    for i in range(config.lookback, len(returns) - config.horizon):
        window = returns[i - config.lookback:i]
        future = returns[i:i + config.horizon]

        # Extract features
        features = extract_features(window, config.n_features)
        X.append(features)

        # Volatility for regime analysis
        volatility_at_pred.append(np.std(window))

        # Target: Did barrier get hit?
        max_cumsum = np.max(np.cumsum(future))
        y_barrier.append(max_cumsum >= config.barrier)

    X = np.array(X)
    y_barrier = np.array(y_barrier)
    volatility_at_pred = np.array(volatility_at_pred)

    # Train/test split
    split = int(len(X) * (1 - config.test_fraction))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_barrier[:split], y_barrier[split:]
    vol_test = volatility_at_pred[split:]

    # ========== PIPELINE 1: XGBoost Classifier ==========
    clf_p1 = GradientBoostingClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        random_state=config.random_seed
    )
    clf_p1.fit(X_train, y_train)
    p_barrier_p1 = clf_p1.predict_proba(X_test)[:, 1]

    # ========== PIPELINE 2: Distribution Parameters ==========

    # Discovery on training returns
    train_returns = returns[config.lookback:config.lookback + split]
    discovery = tpdf.discover(
        train_returns,
        candidates=['normal', 'student_t', 'nig'],
        cv_folds=5,
    )

    # Create parameter targets based on discovered distribution
    y_params_list = []
    for i in range(config.lookback, len(returns) - config.horizon):
        future = returns[i:i + config.horizon]

        if discovery.best == 'nig':
            params = tpdf.fit_nig(future)
            y_params_list.append([params.mu, params.delta, params.alpha, params.beta])
        elif discovery.best == 'student_t':
            params = tpdf.fit_student_t(future)
            y_params_list.append([params.mu_0, params.sigma_0, params.nu])
        else:  # normal
            params = tpdf.fit_normal(future)
            y_params_list.append([params.mu_0, params.sigma_0])

    y_params_all = np.array(y_params_list)
    y_params_train = y_params_all[:split]
    y_params_test = y_params_all[split:]
    n_params = y_params_train.shape[1]

    # Train multi-output regressor
    model_p2 = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_seed
        )
    )
    model_p2.fit(X_train, y_params_train)
    y_params_pred = model_p2.predict(X_test)

    # Compute parameter correlations
    param_corrs = []
    for i in range(min(3, n_params)):
        corr = np.corrcoef(y_params_test[:, i], y_params_pred[:, i])[0, 1]
        param_corrs.append(corr if np.isfinite(corr) else 0.0)
    while len(param_corrs) < 3:
        param_corrs.append(0.0)

    # Simulate barrier probabilities
    if discovery.best == 'nig':
        dist = tpdf.NIG()
    elif discovery.best == 'student_t':
        dist = tpdf.StudentT()
    else:
        dist = tpdf.Normal()

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
                nu=max(pred_params[2], 2.1) if n_params > 2 else 4.0
            )
        else:
            params = tpdf.NormalParameters(
                mu_0=pred_params[0],
                sigma_0=max(pred_params[1], 0.01)
            )

        # Simulate paths
        all_returns = dist.sample(config.n_sims * config.horizon, 0.0, params, rng=rng)
        paths = all_returns.reshape(config.n_sims, config.horizon)
        max_cumsum = np.max(np.cumsum(paths, axis=1), axis=1)
        p_hit = np.mean(max_cumsum >= config.barrier)
        p_barrier_p2.append(p_hit)

    p_barrier_p2 = np.array(p_barrier_p2)

    # ========== Compute Metrics ==========

    # Brier scores
    brier_p1 = np.mean((p_barrier_p1 - y_test.astype(float)) ** 2)
    brier_p2 = np.mean((p_barrier_p2 - y_test.astype(float)) ** 2)

    # Regime analysis
    vol_median = np.median(vol_test)
    high_vol_mask = vol_test > vol_median
    low_vol_mask = ~high_vol_mask

    def regime_error(predictions, actuals, mask):
        if mask.sum() == 0:
            return 0.0
        return abs(predictions[mask].mean() - actuals[mask].mean())

    high_vol_error_p1 = regime_error(p_barrier_p1, y_test, high_vol_mask)
    high_vol_error_p2 = regime_error(p_barrier_p2, y_test, high_vol_mask)
    low_vol_error_p1 = regime_error(p_barrier_p1, y_test, low_vol_mask)
    low_vol_error_p2 = regime_error(p_barrier_p2, y_test, low_vol_mask)

    # Winner
    if abs(brier_p1 - brier_p2) < 0.001:
        winner = 'TIE'
    elif brier_p1 < brier_p2:
        winner = 'P1'
    else:
        winner = 'P2'

    return ExperimentResult(
        config=config,
        brier_p1=brier_p1,
        brier_p2=brier_p2,
        mean_pred_p1=p_barrier_p1.mean(),
        mean_pred_p2=p_barrier_p2.mean(),
        std_pred_p1=p_barrier_p1.std(),
        std_pred_p2=p_barrier_p2.std(),
        actual_rate=y_test.mean(),
        discovery_best=discovery.best,
        param_corr_mu=param_corrs[0],
        param_corr_sigma=param_corrs[1],
        param_corr_nu=param_corrs[2],
        high_vol_error_p1=high_vol_error_p1,
        high_vol_error_p2=high_vol_error_p2,
        low_vol_error_p1=low_vol_error_p1,
        low_vol_error_p2=low_vol_error_p2,
        winner=winner,
    )


def run_experiment_suite(
    returns: np.ndarray,
    barriers: list = [3.0, 5.0],
    horizons: list = [10, 20],
    feature_counts: list = [8, 32],
) -> pd.DataFrame:
    """
    Run a suite of experiments with different configurations.

    Returns:
        DataFrame with all results
    """
    results = []
    total = len(barriers) * len(horizons) * len(feature_counts)

    print(f"Running {total} experiments...")
    print("=" * 60)

    for barrier in barriers:
        for horizon in horizons:
            for n_features in feature_counts:
                config = ExperimentConfig(
                    barrier=barrier,
                    horizon=horizon,
                    n_features=n_features,
                )

                print(f"\nBarrier={barrier}%, Horizon={horizon}d, Features={n_features}")

                try:
                    result = run_experiment(config, returns)
                    results.append(result.to_dict())

                    print(f"  Brier P1: {result.brier_p1:.4f}")
                    print(f"  Brier P2: {result.brier_p2:.4f}")
                    print(f"  Winner: {result.winner}")
                    print(f"  Actual rate: {result.actual_rate:.1%}")
                except Exception as e:
                    print(f"  ERROR: {e}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Load data
    print("Loading S&P 500 returns...")
    df = pd.read_csv('data/equity_returns.csv', parse_dates=['date'])
    returns = df['return_pct'].values
    print(f"Loaded {len(returns):,} daily returns")

    # Run experiment suite
    results_df = run_experiment_suite(
        returns,
        barriers=[2.0, 3.0, 5.0],
        horizons=[5, 10, 15, 20],
        feature_counts=[8, 16, 32],
    )

    # Save results
    results_df.to_csv('experiments/barrier_experiment_results.csv', index=False)
    print("\n" + "=" * 60)
    print("Results saved to experiments/barrier_experiment_results.csv")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nP1 wins: {(results_df['winner'] == 'P1').sum()}")
    print(f"P2 wins: {(results_df['winner'] == 'P2').sum()}")
    print(f"Ties: {(results_df['winner'] == 'TIE').sum()}")

    print("\nBest configurations for P2:")
    p2_wins = results_df[results_df['winner'] == 'P2'].sort_values('brier_diff')
    if len(p2_wins) > 0:
        print(p2_wins[['barrier', 'horizon', 'n_features', 'brier_p1', 'brier_p2', 'brier_diff']].head())
    else:
        print("  No P2 wins found")

    print("\nWorst configurations for P2:")
    p1_wins = results_df[results_df['winner'] == 'P1'].sort_values('brier_diff', ascending=False)
    if len(p1_wins) > 0:
        print(p1_wins[['barrier', 'horizon', 'n_features', 'brier_p1', 'brier_p2', 'brier_diff']].head())

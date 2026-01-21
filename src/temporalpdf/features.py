"""
Feature extraction for distributional regression.

These features are specifically designed to help predict distribution parameters:
- Hill estimator: predicts tail index (nu for Student-t)
- Jarque-Bera: tests normality (high = fat tails)
- Realized moments: skewness, kurtosis
- Volatility features: clustering, regime indicators

Use these as inputs to DistributionalRegressor for better parameter prediction.

Example:
    >>> import temporalpdf as tpdf
    >>>
    >>> # Extract features from rolling windows
    >>> features = tpdf.extract_calibration_features(returns, window=60)
    >>>
    >>> # Use with DistributionalRegressor
    >>> model = tpdf.DistributionalRegressor(distribution="student_t")
    >>> model.fit(features, returns[60:])
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Union


def hill_estimator(data: np.ndarray, k: int = 10) -> float:
    """
    Hill estimator for tail index.

    Estimates the tail exponent from the k largest observations.
    Related to degrees of freedom: lower Hill = fatter tails.

    For Student-t with nu degrees of freedom, Hill estimator converges to nu.

    Args:
        data: Array of observations
        k: Number of upper order statistics to use (default 10)

    Returns:
        Estimated tail index (higher = thinner tails)

    Example:
        >>> # Student-t with nu=5 should give Hill estimate around 5
        >>> data = np.random.standard_t(5, size=1000)
        >>> hill = hill_estimator(data, k=20)
    """
    data = np.asarray(data)
    sorted_abs = np.sort(np.abs(data))[::-1]  # Descending

    if len(sorted_abs) < k + 1:
        return 4.0  # Default for insufficient data

    # Hill estimator: k / sum(log(X_(i) / X_(k+1)))
    log_ratios = np.log(sorted_abs[:k] / sorted_abs[k])

    if np.sum(log_ratios) <= 0:
        return 10.0  # Very thin tails

    return float(k / np.sum(log_ratios))


def jarque_bera_stat(data: np.ndarray) -> float:
    """
    Jarque-Bera test statistic for normality.

    Higher values indicate departure from normality (fat tails or skewness).
    Useful for predicting whether Normal or Student-t is more appropriate.

    Args:
        data: Array of observations

    Returns:
        JB test statistic (higher = less normal)
    """
    data = np.asarray(data)
    n = len(data)

    if n < 4:
        return 0.0

    skew = stats.skew(data)
    kurt = stats.kurtosis(data)  # Excess kurtosis

    # JB = n/6 * (S^2 + K^2/4)
    jb = (n / 6) * (skew**2 + (kurt**2) / 4)

    return float(jb)


def realized_moments(data: np.ndarray) -> Dict[str, float]:
    """
    Compute realized moments of the data.

    Returns mean, std, skewness, and kurtosis.

    Args:
        data: Array of observations

    Returns:
        Dict with 'mean', 'std', 'skewness', 'kurtosis' keys
    """
    data = np.asarray(data)

    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'skewness': float(stats.skew(data)),
        'kurtosis': float(stats.kurtosis(data)),  # Excess kurtosis
    }


def volatility_clustering(data: np.ndarray) -> float:
    """
    Measure of volatility clustering (autocorrelation of absolute returns).

    High clustering suggests GARCH-like dynamics and time-varying sigma.

    Args:
        data: Array of observations (e.g., returns)

    Returns:
        Autocorrelation of |data| at lag 1
    """
    data = np.asarray(data)
    abs_data = np.abs(data)

    if len(abs_data) < 3:
        return 0.0

    # Correlation between |r_t| and |r_{t-1}|
    corr = np.corrcoef(abs_data[:-1], abs_data[1:])[0, 1]

    if np.isnan(corr):
        return 0.0

    return float(corr)


def garch_proxy(data: np.ndarray, short_window: int = 5) -> float:
    """
    GARCH-like volatility forecast proxy.

    Ratio of recent squared returns to overall squared returns.
    Values > 1 indicate elevated volatility regime.

    Args:
        data: Array of observations
        short_window: Recent window size for numerator

    Returns:
        Ratio of recent volatility to overall volatility
    """
    data = np.asarray(data)

    if len(data) < short_window + 1:
        return 1.0

    recent_var = np.mean(data[-short_window:]**2)
    overall_var = np.mean(data**2)

    if overall_var < 1e-10:
        return 1.0

    return float(recent_var / overall_var)


def vol_regime_indicator(data: np.ndarray, short_window: int = 5) -> float:
    """
    Volatility regime indicator.

    Ratio of recent std dev to overall std dev.
    Values > 1 indicate high vol regime, < 1 indicate low vol regime.

    Args:
        data: Array of observations
        short_window: Recent window size

    Returns:
        Ratio of recent to overall volatility
    """
    data = np.asarray(data)

    if len(data) < short_window + 1:
        return 1.0

    recent_std = np.std(data[-short_window:])
    overall_std = np.std(data)

    if overall_std < 1e-10:
        return 1.0

    return float(recent_std / overall_std)


def extreme_event_frequency(data: np.ndarray, threshold: float = 2.0) -> float:
    """
    Frequency of extreme events (observations beyond threshold std devs).

    Higher frequency suggests fat tails.

    Args:
        data: Array of observations
        threshold: Number of standard deviations for "extreme"

    Returns:
        Fraction of observations beyond threshold
    """
    data = np.asarray(data)
    std = np.std(data)

    if std < 1e-10:
        return 0.0

    standardized = np.abs(data - np.mean(data)) / std
    return float(np.mean(standardized > threshold))


def tail_asymmetry(data: np.ndarray, threshold: float = 2.0) -> float:
    """
    Asymmetry between upper and lower tails.

    Values > 1 indicate more positive extremes, < 1 more negative extremes.

    Args:
        data: Array of observations
        threshold: Number of standard deviations for tail

    Returns:
        Ratio of positive to negative tail events
    """
    data = np.asarray(data)
    std = np.std(data)
    mean = np.mean(data)

    if std < 1e-10:
        return 1.0

    upper_tail = np.sum(data > mean + threshold * std)
    lower_tail = np.sum(data < mean - threshold * std)

    if lower_tail == 0:
        return 10.0 if upper_tail > 0 else 1.0

    return float(upper_tail / lower_tail)


def max_drawdown(data: np.ndarray) -> float:
    """
    Maximum drawdown of cumulative sum.

    Useful for understanding tail risk in path-dependent context.

    Args:
        data: Array of observations (e.g., returns)

    Returns:
        Maximum drawdown (positive value)
    """
    data = np.asarray(data)
    cumsum = np.cumsum(data)
    running_max = np.maximum.accumulate(cumsum)
    drawdowns = running_max - cumsum

    return float(np.max(drawdowns))


def calibration_features(data: np.ndarray) -> Dict[str, float]:
    """
    Extract all calibration features from a data window.

    These features are specifically designed to predict distribution parameters:
    - hill_estimator: predicts nu (tail index)
    - jarque_bera: indicates departure from normality
    - kurtosis: directly relates to nu
    - vol_clustering: indicates need for time-varying sigma
    - garch_proxy: current volatility regime

    Args:
        data: Array of observations (e.g., a rolling window of returns)

    Returns:
        Dict of feature name -> value

    Example:
        >>> features = calibration_features(returns[-60:])
        >>> print(f"Tail index estimate: {features['hill_estimator']:.2f}")
    """
    data = np.asarray(data)
    moments = realized_moments(data)

    return {
        # Tail features (predict nu)
        'hill_estimator': hill_estimator(data, k=10),
        'jarque_bera': jarque_bera_stat(data),
        'kurtosis': moments['kurtosis'],
        'extreme_freq': extreme_event_frequency(data, threshold=2.0),

        # Location features (predict mu)
        'mean': moments['mean'],

        # Scale features (predict sigma)
        'std': moments['std'],
        'garch_proxy': garch_proxy(data, short_window=5),
        'vol_regime': vol_regime_indicator(data, short_window=5),
        'vol_clustering': volatility_clustering(data),

        # Asymmetry features (predict beta for NIG)
        'skewness': moments['skewness'],
        'tail_asymmetry': tail_asymmetry(data, threshold=2.0),

        # Risk features
        'max_drawdown': max_drawdown(data),
    }


def extract_calibration_features(
    data: np.ndarray,
    window: int = 60,
    step: int = 1,
) -> np.ndarray:
    """
    Extract calibration features from rolling windows.

    Creates a feature matrix where each row contains features extracted
    from a rolling window ending at that index.

    Args:
        data: Full time series of observations
        window: Rolling window size
        step: Step size between windows (1 = every observation)

    Returns:
        Feature matrix of shape (n_windows, n_features)

    Example:
        >>> returns = np.random.randn(500)
        >>> features = extract_calibration_features(returns, window=60)
        >>> print(f"Feature matrix shape: {features.shape}")
        >>> # Shape will be (440, 12) - one row per valid window
    """
    data = np.asarray(data)
    n = len(data)

    if n < window:
        raise ValueError(f"Data length ({n}) must be >= window ({window})")

    # Get feature names from first window
    sample_features = calibration_features(data[:window])
    feature_names = list(sample_features.keys())
    n_features = len(feature_names)

    # Pre-allocate output
    n_windows = (n - window) // step + 1
    features = np.zeros((n_windows, n_features))

    # Extract features for each window
    for i in range(n_windows):
        start = i * step
        end = start + window
        window_data = data[start:end]

        feat_dict = calibration_features(window_data)
        features[i] = [feat_dict[name] for name in feature_names]

    return features


def get_feature_names() -> List[str]:
    """
    Get the names of calibration features in order.

    Returns:
        List of feature names matching columns of extract_calibration_features()
    """
    # Get from sample call
    sample = calibration_features(np.random.randn(100))
    return list(sample.keys())


__all__ = [
    # Individual feature functions
    'hill_estimator',
    'jarque_bera_stat',
    'realized_moments',
    'volatility_clustering',
    'garch_proxy',
    'vol_regime_indicator',
    'extreme_event_frequency',
    'tail_asymmetry',
    'max_drawdown',
    # Combined extraction
    'calibration_features',
    'extract_calibration_features',
    'get_feature_names',
]

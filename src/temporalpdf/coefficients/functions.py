"""Core statistical functions for coefficient calculation."""

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from scipy.stats import skew as scipy_skew
from sklearn.linear_model import LinearRegression


def calculate_mean(values: NDArray[np.float64] | pd.Series) -> float:
    """
    Calculate the mean of values (mu_0).

    Handles NaN values safely using nanmean.

    Args:
        values: Array or Series of values (typically percentage changes)

    Returns:
        Mean value
    """
    return float(np.nanmean(values))


def calculate_volatility(values: NDArray[np.float64] | pd.Series) -> float:
    """
    Calculate the volatility (sigma_0) as standard deviation.

    Handles NaN values safely using nanstd.

    Args:
        values: Array or Series of values (typically percentage changes)

    Returns:
        Standard deviation
    """
    return float(np.nanstd(values))


def calculate_skewness(values: NDArray[np.float64] | pd.Series) -> float:
    """
    Calculate the skewness (alpha) of the distribution.

    Measures asymmetry in the distribution. Positive skewness indicates
    a longer right tail, negative indicates a longer left tail.

    Args:
        values: Array or Series of values (typically percentage changes)

    Returns:
        Skewness coefficient (Fisher-Pearson)
    """
    if isinstance(values, np.ndarray):
        values = values[~np.isnan(values)]
    elif isinstance(values, pd.Series):
        values = values.dropna().values

    if len(values) < 3:
        return 0.0

    return float(scipy_skew(values))


def calculate_mean_rate(values: NDArray[np.float64] | pd.Series) -> float:
    """
    Calculate the mean rate (delta) as the slope of linear regression over time.

    This represents the trend in the values over the observation window.

    Args:
        values: Array or Series of values (typically raw prices/values)

    Returns:
        Slope of the regression line (trend per time unit)
    """
    if isinstance(values, pd.Series):
        values = values.values

    # Remove NaN values
    valid_mask = ~np.isnan(values)
    values = values[valid_mask]

    if len(values) < 2:
        return 0.0

    time_indices = np.arange(len(values)).reshape(-1, 1)
    model = LinearRegression().fit(time_indices, values)
    return float(model.coef_[0])


def calculate_volatility_growth(
    values: NDArray[np.float64] | pd.Series,
    window: int = 7,
) -> float:
    """
    Calculate the volatility growth (beta) as the slope of rolling volatility.

    This measures how volatility changes over time within the observation window.
    A positive value indicates increasing volatility, negative indicates decreasing.

    Args:
        values: Array or Series of values (typically percentage changes)
        window: Rolling window size for calculating volatility

    Returns:
        Slope of volatility trend (volatility change per time unit)
    """
    if isinstance(values, np.ndarray):
        values = pd.Series(values)

    # Compute rolling volatility
    rolling_volatility = values.rolling(window=window).std().dropna()

    if len(rolling_volatility) < 2:
        return 0.0

    # Perform linear regression on rolling volatility
    time_indices = np.arange(len(rolling_volatility)).reshape(-1, 1)
    model = LinearRegression().fit(time_indices, rolling_volatility.values)
    return float(model.coef_[0])


def calculate_all_coefficients(
    values: NDArray[np.float64] | pd.Series,
    pct_changes: NDArray[np.float64] | pd.Series,
    volatility_window: int = 7,
) -> dict[str, float]:
    """
    Calculate all five distribution coefficients at once.

    This is a convenience function that calculates all coefficients
    in a single call.

    Args:
        values: Raw values (for mean_rate calculation)
        pct_changes: Percentage/fractional changes (for other calculations)
        volatility_window: Window for volatility growth calculation

    Returns:
        Dictionary with keys: mean, volatility, skewness, mean_rate, volatility_growth
    """
    return {
        "mean": calculate_mean(pct_changes),
        "volatility": calculate_volatility(pct_changes),
        "skewness": calculate_skewness(pct_changes),
        "mean_rate": calculate_mean_rate(values),
        "volatility_growth": calculate_volatility_growth(pct_changes, window=volatility_window),
    }

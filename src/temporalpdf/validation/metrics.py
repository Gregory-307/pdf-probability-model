"""Validation metric functions."""

import numpy as np
from numpy.typing import NDArray


def log_likelihood(
    observed: float,
    pdf_values: NDArray[np.float64],
    value_grid: NDArray[np.float64],
    min_prob: float = 1e-10,
) -> float:
    """
    Calculate log-likelihood of an observed value under a PDF.

    Args:
        observed: The observed value
        pdf_values: PDF values at each point in value_grid
        value_grid: The grid of values
        min_prob: Minimum probability to avoid log(0)

    Returns:
        Log-likelihood value
    """
    # Interpolate PDF at observed point
    pdf_at_obs = float(np.interp(observed, value_grid, pdf_values))
    pdf_at_obs = max(pdf_at_obs, min_prob)
    return float(np.log(pdf_at_obs))


def mae(predicted: float, observed: float) -> float:
    """
    Calculate Mean Absolute Error for a single observation.

    Args:
        predicted: Predicted value (e.g., expected value)
        observed: Observed value

    Returns:
        Absolute error
    """
    return float(abs(predicted - observed))


def mse(predicted: float, observed: float) -> float:
    """
    Calculate Mean Squared Error for a single observation.

    Args:
        predicted: Predicted value
        observed: Observed value

    Returns:
        Squared error
    """
    return float((predicted - observed) ** 2)


def r_squared(
    predicted: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> float:
    """
    Calculate R-squared (coefficient of determination).

    R^2 = 1 - SS_res / SS_tot

    where:
        SS_res = sum((observed - predicted)^2)
        SS_tot = sum((observed - mean(observed))^2)

    Args:
        predicted: Array of predicted values
        observed: Array of observed values

    Returns:
        R-squared value (can be negative if model is worse than mean)
    """
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)

    if ss_tot == 0:
        return 0.0

    return float(1 - ss_res / ss_tot)


def rmse(
    predicted: NDArray[np.float64],
    observed: NDArray[np.float64],
) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        predicted: Array of predicted values
        observed: Array of observed values

    Returns:
        RMSE value
    """
    return float(np.sqrt(np.mean((observed - predicted) ** 2)))


def mean_log_likelihood(
    log_likelihoods: NDArray[np.float64],
) -> float:
    """
    Calculate mean log-likelihood.

    Args:
        log_likelihoods: Array of log-likelihood values

    Returns:
        Mean log-likelihood
    """
    return float(np.mean(log_likelihoods))


def probability_integral_transform(
    observed: float,
    pdf_values: NDArray[np.float64],
    value_grid: NDArray[np.float64],
) -> float:
    """
    Calculate Probability Integral Transform (PIT) value.

    The PIT is the CDF value at the observed point. For a well-calibrated
    model, PIT values should be uniformly distributed.

    Args:
        observed: The observed value
        pdf_values: PDF values at each point in value_grid
        value_grid: The grid of values

    Returns:
        PIT value (between 0 and 1)
    """
    # Compute CDF by numerical integration
    cdf = np.cumsum(pdf_values) * (value_grid[1] - value_grid[0])
    cdf = cdf / cdf[-1]  # Normalize to [0, 1]

    # Interpolate CDF at observed point
    pit = float(np.interp(observed, value_grid, cdf))
    return np.clip(pit, 0.0, 1.0)

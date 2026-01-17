"""Probability queries for distributional predictions.

These utilities enable rich probability queries from predicted distributions:
- P(X > threshold)
- P(X < threshold)
- P(a < X < b)
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Distribution(Protocol):
    """Protocol for distributions with CDF."""

    def cdf(
        self, x: NDArray[np.float64], t: float, params: object
    ) -> NDArray[np.float64]: ...


def prob_greater_than(
    dist: Distribution,
    params: object,
    threshold: float,
    t: float = 0.0,
) -> float:
    """
    Compute P(X > threshold).

    Args:
        dist: Distribution with cdf method
        params: Distribution parameters
        threshold: Value to compare against
        t: Time point

    Returns:
        Probability that X exceeds threshold
    """
    x = np.array([threshold])
    cdf_val = dist.cdf(x, t, params)
    return float(1 - cdf_val[0])


def prob_less_than(
    dist: Distribution,
    params: object,
    threshold: float,
    t: float = 0.0,
) -> float:
    """
    Compute P(X < threshold).

    Args:
        dist: Distribution with cdf method
        params: Distribution parameters
        threshold: Value to compare against
        t: Time point

    Returns:
        Probability that X is less than threshold
    """
    x = np.array([threshold])
    cdf_val = dist.cdf(x, t, params)
    return float(cdf_val[0])


def prob_between(
    dist: Distribution,
    params: object,
    lower: float,
    upper: float,
    t: float = 0.0,
) -> float:
    """
    Compute P(lower < X < upper).

    Args:
        dist: Distribution with cdf method
        params: Distribution parameters
        lower: Lower bound
        upper: Upper bound
        t: Time point

    Returns:
        Probability that X is between lower and upper
    """
    if lower >= upper:
        raise ValueError(f"lower must be less than upper, got {lower} >= {upper}")

    x = np.array([lower, upper])
    cdf_vals = dist.cdf(x, t, params)
    return float(cdf_vals[1] - cdf_vals[0])


def prob_loss_exceeds(
    dist: Distribution,
    params: object,
    loss_threshold: float,
    t: float = 0.0,
) -> float:
    """
    Compute P(loss > threshold) = P(X < -threshold).

    For return distributions, this gives the probability of
    losing more than the threshold.

    Args:
        dist: Distribution with cdf method
        params: Distribution parameters
        loss_threshold: Loss amount (positive number)
        t: Time point

    Returns:
        Probability of loss exceeding threshold
    """
    if loss_threshold < 0:
        raise ValueError(f"loss_threshold should be positive, got {loss_threshold}")

    return prob_less_than(dist, params, -loss_threshold, t)

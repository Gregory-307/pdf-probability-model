"""
Scoring functions for distribution evaluation.

Uses proper scoring rules to evaluate predictive distributions.
"""

import numpy as np
from numpy.typing import NDArray


def crps_from_samples(y_true: float, samples: NDArray[np.float64]) -> float:
    """
    Compute CRPS from Monte Carlo samples.

    CRPS = E[|X - y|] - 0.5 * E[|X - X'|]

    where X, X' are independent samples from the forecast distribution.

    Args:
        y_true: Observed value
        samples: Samples from forecast distribution

    Returns:
        CRPS value (lower is better)
    """
    n = len(samples)
    # E[|X - y|]
    term1 = np.mean(np.abs(samples - y_true))

    # E[|X - X'|] via pairwise differences
    # For efficiency, use the identity: E[|X-X'|] = 2 * integral_x F(x)(1-F(x)) dx
    # But for moderate n, direct computation is fine
    if n <= 5000:
        term2 = np.mean(np.abs(samples[:, None] - samples[None, :]))
    else:
        # Subsample for large n
        idx1 = np.random.choice(n, min(n, 2500), replace=False)
        idx2 = np.random.choice(n, min(n, 2500), replace=False)
        term2 = np.mean(np.abs(samples[idx1, None] - samples[None, idx2]))

    return term1 - 0.5 * term2


def crps_normal(y_true: float, mu: float, sigma: float) -> float:
    """
    Closed-form CRPS for Normal distribution.

    CRPS = sigma * (z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi))

    where z = (y - mu) / sigma

    Args:
        y_true: Observed value
        mu: Mean of Normal distribution
        sigma: Standard deviation

    Returns:
        CRPS value
    """
    from scipy import stats

    z = (y_true - mu) / sigma
    return sigma * (
        z * (2 * stats.norm.cdf(z) - 1)
        + 2 * stats.norm.pdf(z)
        - 1 / np.sqrt(np.pi)
    )


def log_score(y_true: float, pdf_value: float) -> float:
    """
    Negative log likelihood (log score).

    Args:
        y_true: Observed value
        pdf_value: PDF evaluated at y_true

    Returns:
        Negative log likelihood (lower is better)
    """
    return -np.log(max(pdf_value, 1e-300))

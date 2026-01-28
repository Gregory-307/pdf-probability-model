"""Proper scoring rules for evaluating distributional predictions.

A scoring rule S(P, y) is **strictly proper** if the expected score is uniquely
optimized when the forecaster reports their true belief.

References:
    Gneiting, T. & Raftery, A.E. (2007). Strictly Proper Scoring Rules,
    Prediction, and Estimation. JASA, 102(477), 359-378.

    Matheson, J.E. & Winkler, R.L. (1976). Scoring Rules for Continuous
    Probability Distributions. Management Science, 22(10), 1087-1096.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy import stats


class Distribution(Protocol):
    """Protocol for distributions that can be scored."""

    def pdf(
        self, x: NDArray[np.float64], t: float, params: object
    ) -> NDArray[np.float64]: ...

    def cdf(
        self, x: NDArray[np.float64], t: float, params: object
    ) -> NDArray[np.float64]: ...

    def ppf(
        self, q: NDArray[np.float64], t: float, params: object
    ) -> NDArray[np.float64]: ...


class ScoringRule(Protocol):
    """Protocol for proper scoring rules."""

    @property
    def name(self) -> str: ...

    @property
    def is_proper(self) -> bool: ...

    def __call__(
        self,
        dist: Distribution,
        params: object,
        y: float | NDArray[np.float64],
        t: float,
    ) -> float | NDArray[np.float64]: ...


class LogScore:
    """
    Log Score (Negative Log-Likelihood).

    S_log(P, y) = -log(p(y))

    Properties:
    - Strictly proper
    - Local: only depends on density at observation
    - Can be -infinity if density is zero at observation
    - Sensitive to calibration errors in tails

    Lower is better (negative log-likelihood).
    """

    @property
    def name(self) -> str:
        return "Log Score"

    @property
    def is_proper(self) -> bool:
        return True

    def __call__(
        self,
        dist: Distribution,
        params: object,
        y: float | NDArray[np.float64],
        t: float = 0.0,
    ) -> float | NDArray[np.float64]:
        """
        Compute log score for observation(s) y.

        Args:
            dist: Distribution object with pdf method
            params: Distribution parameters
            y: Observation(s) to score
            t: Time point (default 0)

        Returns:
            Log score(s) - lower is better
        """
        y = np.atleast_1d(y)
        pdf_vals = dist.pdf(y, t, params)

        # Clip to avoid log(0)
        pdf_vals = np.maximum(pdf_vals, 1e-300)

        scores = -np.log(pdf_vals)

        return float(scores[0]) if len(scores) == 1 else scores


class CRPS:
    """
    Continuous Ranked Probability Score.

    CRPS(F, y) = integral_{-inf}^{inf} (F(x) - 1{x >= y})^2 dx
               = integral_{-inf}^{y} F(x)^2 dx + integral_{y}^{inf} (1 - F(x))^2 dx

    Properties:
    - Strictly proper
    - Global: considers entire distribution
    - Always finite
    - Generalizes MAE to probabilistic forecasts
    - Has same units as the observation

    Lower is better.

    References:
        Gneiting, T. & Raftery, A.E. (2007). Strictly Proper Scoring Rules.
    """

    @property
    def name(self) -> str:
        return "CRPS"

    @property
    def is_proper(self) -> bool:
        return True

    def __call__(
        self,
        dist: Distribution,
        params: object,
        y: float | NDArray[np.float64],
        t: float = 0.0,
    ) -> float | NDArray[np.float64]:
        """
        Compute CRPS using numerical integration of CDF.

        Uses the representation:
            CRPS = integral_{-inf}^{y} F(x)^2 dx + integral_{y}^{inf} (1 - F(x))^2 dx

        Args:
            dist: Distribution object with cdf and ppf methods
            params: Distribution parameters
            y: Observation(s) to score
            t: Time point (default 0)

        Returns:
            CRPS score(s) - lower is better
        """
        from scipy import integrate

        y = np.atleast_1d(y)

        # Determine integration bounds based on distribution spread
        # Use ppf to find practical bounds
        lower_bound = float(dist.ppf(np.array([0.0001]), t, params)[0])
        upper_bound = float(dist.ppf(np.array([0.9999]), t, params)[0])

        scores = np.empty(len(y))
        for i, yi in enumerate(y):
            yi_float = float(yi)

            # Left integral: integral_{lower}^{y} F(x)^2 dx
            def left_integrand(x: float) -> float:
                cdf_val = dist.cdf(np.array([x]), t, params)[0]
                return cdf_val ** 2

            # Right integral: integral_{y}^{upper} (1 - F(x))^2 dx
            def right_integrand(x: float) -> float:
                cdf_val = dist.cdf(np.array([x]), t, params)[0]
                return (1 - cdf_val) ** 2

            left_integral, _ = integrate.quad(
                left_integrand,
                lower_bound,
                yi_float,
                limit=50,
            )

            right_integral, _ = integrate.quad(
                right_integrand,
                yi_float,
                upper_bound,
                limit=50,
            )

            scores[i] = left_integral + right_integral

        return float(scores[0]) if len(scores) == 1 else scores


def log_score(
    dist: Distribution,
    params: object,
    y: float | NDArray[np.float64],
    t: float = 0.0,
) -> float | NDArray[np.float64]:
    """Convenience function for log score."""
    return LogScore()(dist, params, y, t)


def crps(
    dist: Distribution,
    params: object,
    y: float | NDArray[np.float64],
    t: float = 0.0,
) -> float | NDArray[np.float64]:
    """
    Compute CRPS using numerical integration.

    This is the preferred method - exact up to numerical precision.
    For faster approximate results, use crps_mc().
    """
    return CRPS()(dist, params, y, t)


def crps_mc(
    dist: Distribution,
    params: object,
    y: float | NDArray[np.float64],
    t: float = 0.0,
    n_samples: int = 10000,
    rng: np.random.Generator | None = None,
) -> float | NDArray[np.float64]:
    """
    Compute CRPS using Monte Carlo sampling.

    Uses the representation: CRPS = E|X - y| - 0.5 * E|X - X'|

    Use this when numerical integration is too slow or for validation.
    Faster than crps() but has ~1% sampling variance with default n_samples.

    Args:
        dist: Distribution with sample or ppf method
        params: Distribution parameters
        y: Observation(s) to score
        t: Time point (default 0)
        n_samples: Number of Monte Carlo samples
        rng: Random number generator

    Returns:
        CRPS score(s) - lower is better
    """
    y_arr = np.atleast_1d(y)

    if rng is None:
        rng = np.random.default_rng()

    if hasattr(dist, "sample"):
        samples = dist.sample(n_samples, t, params, rng)
    elif hasattr(dist, "ppf"):
        u = rng.uniform(size=n_samples)
        samples = dist.ppf(u, t, params)
    else:
        raise ValueError("Distribution must have 'sample' or 'ppf' method")

    scores = np.empty(len(y_arr))
    for i, yi in enumerate(y_arr):
        term1 = np.mean(np.abs(samples - yi))
        half = n_samples // 2
        term2 = np.mean(np.abs(samples[:half] - samples[half : 2 * half]))
        scores[i] = term1 - 0.5 * term2

    return float(scores[0]) if len(scores) == 1 else scores


def crps_normal(
    y: float | NDArray[np.float64],
    mu: float | NDArray[np.float64],
    sigma: float | NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """
    Closed-form CRPS for Normal distribution.

    CRPS(N(mu, sigma^2), y) = sigma * [z*(2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]

    where z = (y - mu) / sigma.

    This is much faster than Monte Carlo for Normal distributions.

    Args:
        y: Observation(s)
        mu: Mean(s)
        sigma: Standard deviation(s)

    Returns:
        CRPS score(s)
    """
    y = np.atleast_1d(y)
    mu = np.atleast_1d(mu)
    sigma = np.atleast_1d(sigma)

    z = (y - mu) / sigma

    # Standard normal PDF and CDF
    phi_z = stats.norm.pdf(z)
    Phi_z = stats.norm.cdf(z)

    crps_vals = sigma * (z * (2 * Phi_z - 1) + 2 * phi_z - 1 / np.sqrt(np.pi))

    return float(crps_vals[0]) if len(crps_vals) == 1 else crps_vals

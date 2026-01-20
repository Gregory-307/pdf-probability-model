"""Kelly Criterion for optimal position sizing.

The Kelly criterion maximizes the expected logarithmic growth rate of wealth.

References:
    Kelly, J.L. (1956). A New Interpretation of Information Rate.
    Bell System Technical Journal, 35(4), 917-926.

    Thorp, E.O. (2006). The Kelly Criterion in Blackjack, Sports Betting,
    and the Stock Market. Handbook of Asset and Liability Management, 1, 385-428.

    MacLean, L.C., Thorp, E.O., & Ziemba, W.T. (2011). The Kelly Capital Growth
    Investment Criterion. World Scientific.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..core.result import RiskMetric


class Distribution(Protocol):
    """Protocol for distributions."""

    def mean(self, t: float, params: object) -> float: ...

    def variance(self, t: float, params: object) -> float: ...

    def sample(
        self, n: int, t: float, params: object, rng: np.random.Generator | None
    ) -> NDArray[np.float64]: ...


class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing.

    The Kelly fraction maximizes expected log growth:
        f* = argmax_f E[log(1 + f*r)]

    For small returns, this approximates to:
        f* ≈ μ / σ²

    Properties:
    - Maximizes long-run geometric growth rate
    - Can lead to large drawdowns in practice
    - Fractional Kelly (κ * f*) reduces variance significantly

    Fractional Kelly tradeoffs (from Thorp 2006):
    | κ    | Growth vs Full Kelly | Variance vs Full Kelly |
    |------|---------------------|------------------------|
    | 1.0  | 100%                | 100%                   |
    | 0.5  | 75%                 | 25%                    |
    | 0.25 | 44%                 | 6%                     |
    """

    @property
    def name(self) -> str:
        return "Kelly Criterion"

    def __call__(
        self,
        dist: Distribution,
        params: object,
        t: float = 0.0,
        risk_free_rate: float = 0.0,
        method: str = "analytic",
        n_samples: int = 100000,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Compute optimal Kelly fraction.

        Args:
            dist: Distribution with mean/variance methods
            params: Distribution parameters
            t: Time point
            risk_free_rate: Risk-free rate (annualized)
            method: "analytic" uses μ/σ², "numerical" uses grid search
            n_samples: Number of samples for numerical method
            rng: Random number generator

        Returns:
            Optimal Kelly fraction (can be > 1 for leverage)
        """
        if method == "analytic":
            return self._analytic_kelly(dist, params, t, risk_free_rate)
        elif method == "numerical":
            return self._numerical_kelly(
                dist, params, t, risk_free_rate, n_samples, rng
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def _analytic_kelly(
        self,
        dist: Distribution,
        params: object,
        t: float,
        risk_free_rate: float,
    ) -> float:
        """
        Analytic Kelly using continuous approximation.

        f* ≈ (μ - r_f) / σ²

        This is exact for Normal returns and a good approximation
        for other distributions with small expected returns.
        """
        mu = dist.mean(t, params)
        var = dist.variance(t, params)

        if var <= 0:
            return 0.0

        return (mu - risk_free_rate) / var

    def _numerical_kelly(
        self,
        dist: Distribution,
        params: object,
        t: float,
        risk_free_rate: float,
        n_samples: int,
        rng: np.random.Generator | None,
    ) -> float:
        """
        Numerical Kelly using Monte Carlo grid search.

        Maximizes E[log(1 + f*r)] over a grid of fractions.
        More accurate for distributions with significant skewness/kurtosis.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Sample returns
        samples = dist.sample(n_samples, t, params, rng) - risk_free_rate

        # Grid of fractions to try
        fractions = np.linspace(-2.0, 3.0, 1000)

        # Compute expected log growth for each fraction
        growth_rates = np.empty(len(fractions))

        for i, f in enumerate(fractions):
            log_growth = np.log(np.maximum(1 + f * samples, 1e-10))
            growth_rates[i] = np.mean(log_growth)

        # Find optimal
        best_idx = np.argmax(growth_rates)

        return float(fractions[best_idx])


def kelly_fraction(
    dist: Distribution,
    params: object,
    t: float = 0.0,
    risk_free_rate: float = 0.0,
    method: str = "analytic",
) -> float:
    """Convenience function for Kelly fraction."""
    return KellyCriterion()(dist, params, t, risk_free_rate, method)


def fractional_kelly(
    dist: Distribution,
    params: object,
    fraction: float = 0.5,
    t: float = 0.0,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Compute fractional Kelly position size.

    Fractional Kelly (using κ < 1) significantly reduces variance
    while only moderately reducing expected growth.

    Common choices:
    - κ = 0.5 (half-Kelly): 75% growth, 25% variance
    - κ = 0.25 (quarter-Kelly): 44% growth, 6% variance

    Args:
        dist: Distribution
        params: Distribution parameters
        fraction: Kelly fraction κ (default 0.5 for half-Kelly)
        t: Time point
        risk_free_rate: Risk-free rate

    Returns:
        Fractional Kelly position size
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    full_kelly = kelly_fraction(dist, params, t, risk_free_rate)

    return fraction * full_kelly


# =============================================================================
# V2 API - Functions returning RiskMetric with confidence intervals
# =============================================================================


def kelly_with_ci(
    dist: Distribution,
    params: object,
    t: float = 0.0,
    risk_free_rate: float = 0.0,
    confidence_level: float = 0.90,
    n_samples: int = 100000,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> RiskMetric:
    """
    Compute Kelly fraction with confidence interval via bootstrap.

    Args:
        dist: Distribution with sample method
        params: Distribution parameters
        t: Time point
        risk_free_rate: Risk-free rate
        confidence_level: CI level (default 0.90 for 90% CI)
        n_samples: Number of samples from distribution
        n_bootstrap: Number of bootstrap iterations
        rng: Random number generator

    Returns:
        RiskMetric with value and confidence_interval
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample from distribution
    samples = dist.sample(n_samples, t, params, rng) - risk_free_rate

    # Point estimate: μ / σ²
    kelly_val = float(np.mean(samples) / (np.var(samples) + 1e-10))

    # Bootstrap CI
    kelly_boots = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, n_samples, replace=True)
        boot_samples = samples[boot_idx]
        boot_kelly = np.mean(boot_samples) / (np.var(boot_samples) + 1e-10)
        kelly_boots.append(boot_kelly)

    ci_low = (1 - confidence_level) / 2
    ci_high = 1 - ci_low

    return RiskMetric(
        value=kelly_val,
        confidence_interval=(
            float(np.quantile(kelly_boots, ci_low)),
            float(np.quantile(kelly_boots, ci_high))
        ),
        standard_error=float(np.std(kelly_boots)),
    )

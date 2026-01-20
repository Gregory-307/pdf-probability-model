"""Risk measures: Value at Risk (VaR) and Conditional Value at Risk (CVaR).

References:
    Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
    Value-at-Risk. Journal of Risk, 2, 21-41.

    Rockafellar, R.T. & Uryasev, S. (2002). Conditional Value-at-Risk for
    General Loss Distributions. Journal of Banking & Finance, 26(7), 1443-1471.

    Artzner, P., Delbaen, F., Eber, J.M., & Heath, D. (1999). Coherent Measures
    of Risk. Mathematical Finance, 9(3), 203-228.
"""

from typing import Protocol, overload

import numpy as np
from numpy.typing import NDArray

from ..core.result import RiskMetric


class Distribution(Protocol):
    """Protocol for distributions with quantile function."""

    def ppf(
        self, q: NDArray[np.float64], t: float, params: object
    ) -> NDArray[np.float64]: ...

    def pdf(
        self, x: NDArray[np.float64], t: float, params: object
    ) -> NDArray[np.float64]: ...

    def sample(
        self, n: int, t: float, params: object, rng: np.random.Generator | None
    ) -> NDArray[np.float64]: ...


class VaR:
    """
    Value at Risk (VaR).

    VaR_alpha(X) = -quantile(X, alpha) = -F^{-1}(alpha)

    For a return distribution, VaR at confidence level (1-alpha) is the
    alpha-quantile negated (since losses are negative returns).

    Example:
        95% VaR = VaR(alpha=0.05) → maximum loss at 95% confidence

    Properties:
    - NOT a coherent risk measure (fails subadditivity)
    - Easy to interpret: "5% chance of losing more than VaR"
    - Does NOT capture severity of tail losses

    References:
        Jorion, P. (2007). Value at Risk: The New Benchmark for Managing
        Financial Risk (3rd ed.). McGraw-Hill.
    """

    @property
    def name(self) -> str:
        return "Value at Risk"

    def __call__(
        self,
        dist: Distribution,
        params: object,
        alpha: float = 0.05,
        t: float = 0.0,
    ) -> float:
        """
        Compute VaR at confidence level (1 - alpha).

        Args:
            dist: Distribution with ppf method
            params: Distribution parameters
            alpha: Tail probability (default 0.05 for 95% VaR)
            t: Time point

        Returns:
            VaR value (positive = loss)
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        q = np.array([alpha])
        quantile = dist.ppf(q, t, params)

        return -float(quantile[0])


class CVaR:
    """
    Conditional Value at Risk (Expected Shortfall).

    CVaR_alpha(X) = E[X | X <= VaR_alpha(X)]
                  = E[-L | L >= VaR_alpha] for losses L = -X

    CVaR captures the expected loss given that we are in the tail.

    Properties:
    - Coherent risk measure (satisfies subadditivity)
    - Captures tail severity, not just tail probability
    - Always >= VaR
    - Convex, enabling portfolio optimization

    References:
        Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
        Value-at-Risk. Journal of Risk, 2, 21-41.
    """

    @property
    def name(self) -> str:
        return "Conditional Value at Risk"

    def __call__(
        self,
        dist: Distribution,
        params: object,
        alpha: float = 0.05,
        t: float = 0.0,
        n_samples: int = 100000,
        rng: np.random.Generator | None = None,
    ) -> float:
        """
        Compute CVaR at confidence level (1 - alpha).

        Uses Monte Carlo sampling for general distributions.

        Args:
            dist: Distribution with sample method
            params: Distribution parameters
            alpha: Tail probability (default 0.05 for 95% CVaR)
            t: Time point
            n_samples: Number of Monte Carlo samples
            rng: Random number generator

        Returns:
            CVaR value (positive = expected loss in tail)
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        if rng is None:
            rng = np.random.default_rng()

        # Sample from distribution
        samples = dist.sample(n_samples, t, params, rng)

        # Find VaR quantile
        var_quantile = np.percentile(samples, alpha * 100)

        # Expected value in the tail
        tail_samples = samples[samples <= var_quantile]

        if len(tail_samples) == 0:
            # Fallback: use the lowest samples
            n_tail = max(1, int(n_samples * alpha))
            tail_samples = np.sort(samples)[:n_tail]

        return -float(np.mean(tail_samples))


def var(
    dist: Distribution,
    params: object,
    alpha: float = 0.05,
    t: float = 0.0,
) -> float:
    """Convenience function for VaR."""
    return VaR()(dist, params, alpha, t)


def cvar(
    dist: Distribution,
    params: object,
    alpha: float = 0.05,
    t: float = 0.0,
    n_samples: int = 100000,
    rng: np.random.Generator | None = None,
) -> float:
    """Convenience function for CVaR."""
    return CVaR()(dist, params, alpha, t, n_samples, rng)


# =============================================================================
# V2 API - Functions returning RiskMetric with confidence intervals
# =============================================================================


def var_with_ci(
    dist: Distribution,
    params: object,
    alpha: float = 0.05,
    t: float = 0.0,
    confidence_level: float = 0.90,
    n_samples: int = 100000,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> RiskMetric:
    """
    Compute VaR with confidence interval via bootstrap.

    Args:
        dist: Distribution with sample method
        params: Distribution parameters
        alpha: Tail probability (default 0.05 for 95% VaR)
        t: Time point
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
    samples = dist.sample(n_samples, t, params, rng)

    # Point estimate
    var_val = -float(np.quantile(samples, alpha))

    # Bootstrap CI
    var_boots = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, n_samples, replace=True)
        boot_samples = samples[boot_idx]
        var_boots.append(-np.quantile(boot_samples, alpha))

    ci_low = (1 - confidence_level) / 2
    ci_high = 1 - ci_low

    return RiskMetric(
        value=var_val,
        confidence_interval=(
            float(np.quantile(var_boots, ci_low)),
            float(np.quantile(var_boots, ci_high))
        ),
        standard_error=float(np.std(var_boots)),
    )


def cvar_with_ci(
    dist: Distribution,
    params: object,
    alpha: float = 0.05,
    t: float = 0.0,
    confidence_level: float = 0.90,
    n_samples: int = 100000,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> RiskMetric:
    """
    Compute CVaR with confidence interval via bootstrap.

    Args:
        dist: Distribution with sample method
        params: Distribution parameters
        alpha: Tail probability (default 0.05 for 95% CVaR)
        t: Time point
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
    samples = dist.sample(n_samples, t, params, rng)

    # Point estimate
    threshold = np.quantile(samples, alpha)
    tail = samples[samples <= threshold]
    cvar_val = -float(np.mean(tail)) if len(tail) > 0 else -threshold

    # Bootstrap CI
    cvar_boots = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_samples, n_samples, replace=True)
        boot_samples = samples[boot_idx]
        threshold = np.quantile(boot_samples, alpha)
        tail = boot_samples[boot_samples <= threshold]
        cvar_boots.append(-np.mean(tail) if len(tail) > 0 else -threshold)

    ci_low = (1 - confidence_level) / 2
    ci_high = 1 - ci_low

    return RiskMetric(
        value=cvar_val,
        confidence_interval=(
            float(np.quantile(cvar_boots, ci_low)),
            float(np.quantile(cvar_boots, ci_high))
        ),
        standard_error=float(np.std(cvar_boots)),
    )

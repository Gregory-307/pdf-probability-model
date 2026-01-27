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
                  = (1/alpha) * integral_{-inf}^{VaR} x * f(x) dx

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
    ) -> float:
        """
        Compute CVaR at confidence level (1 - alpha).

        Uses numerical integration of x * pdf(x) over the tail.

        Args:
            dist: Distribution with pdf and ppf methods
            params: Distribution parameters
            alpha: Tail probability (default 0.05 for 95% CVaR)
            t: Time point

        Returns:
            CVaR value (positive = expected loss in tail)
        """
        from scipy import integrate

        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        # Get VaR (the alpha-quantile)
        var_quantile = dist.ppf(np.array([alpha]), t, params)[0]

        # CVaR = (1/alpha) * integral_{-inf}^{var_quantile} x * f(x) dx
        # We integrate x * pdf(x) from a practical lower bound to VaR
        def integrand(x: float) -> float:
            pdf_val = dist.pdf(np.array([x]), t, params)[0]
            return x * pdf_val

        # Determine practical lower bound (far into left tail)
        # Use VaR as reference point and go 10 "VaR distances" below
        if var_quantile < 0:
            lower_bound = var_quantile * 10  # For negative VaR, multiply makes more negative
        else:
            lower_bound = var_quantile - 10 * abs(var_quantile) - 0.5  # Ensure we go left

        # Numerical integration
        integral_result, _ = integrate.quad(
            integrand,
            lower_bound,
            var_quantile,
            limit=100,
        )

        # CVaR = -E[X | X <= VaR] (negated because losses are negative returns)
        cvar_value = -integral_result / alpha

        return float(cvar_value)


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
) -> float:
    """
    Compute CVaR using numerical integration.

    This is the preferred method - exact up to numerical precision.
    """
    return CVaR()(dist, params, alpha, t)


def cvar_mc(
    dist: Distribution,
    params: object,
    alpha: float = 0.05,
    t: float = 0.0,
    n_samples: int = 100000,
    rng: np.random.Generator | None = None,
) -> float:
    """
    Compute CVaR using Monte Carlo sampling.

    Use this when numerical integration is too slow or for validation.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if rng is None:
        rng = np.random.default_rng()

    samples = dist.sample(n_samples, t, params, rng)
    var_quantile = np.percentile(samples, alpha * 100)
    tail_samples = samples[samples <= var_quantile]

    if len(tail_samples) == 0:
        n_tail = max(1, int(n_samples * alpha))
        tail_samples = np.sort(samples)[:n_tail]

    return -float(np.mean(tail_samples))


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
    Compute VaR with confidence interval.

    Point estimate uses exact ppf. CI uses parametric bootstrap to quantify
    sampling variability (useful for understanding estimation stability).

    Args:
        dist: Distribution with ppf and sample methods
        params: Distribution parameters
        alpha: Tail probability (default 0.05 for 95% VaR)
        t: Time point
        confidence_level: CI level (default 0.90 for 90% CI)
        n_samples: Number of samples for bootstrap
        n_bootstrap: Number of bootstrap iterations
        rng: Random number generator

    Returns:
        RiskMetric with value and confidence_interval
    """
    # Point estimate: exact via ppf
    var_val = var(dist, params, alpha, t)

    # Bootstrap CI via parametric bootstrap
    if rng is None:
        rng = np.random.default_rng()

    samples = dist.sample(n_samples, t, params, rng)

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
    Compute CVaR with confidence interval.

    Point estimate uses exact numerical integration. CI uses parametric
    bootstrap to quantify sampling variability.

    Args:
        dist: Distribution with pdf, ppf and sample methods
        params: Distribution parameters
        alpha: Tail probability (default 0.05 for 95% CVaR)
        t: Time point
        confidence_level: CI level (default 0.90 for 90% CI)
        n_samples: Number of samples for bootstrap
        n_bootstrap: Number of bootstrap iterations
        rng: Random number generator

    Returns:
        RiskMetric with value and confidence_interval
    """
    # Point estimate: exact via numerical integration
    cvar_val = cvar(dist, params, alpha, t)

    # Bootstrap CI via parametric bootstrap
    if rng is None:
        rng = np.random.default_rng()

    samples = dist.sample(n_samples, t, params, rng)

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

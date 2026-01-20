"""Parameter dataclasses for distribution implementations."""

from dataclasses import dataclass
from .distribution import DistributionParameters


@dataclass(frozen=True)
class GeneralizedLaplaceParameters(DistributionParameters):
    """
    Parameters for the Generalized Laplace distribution with skew and time evolution.

    Mathematical form:
        f(x, t) = C * exp(-|x - mu(t)|^(1+k) / (2*sigma(t)^2)) * (1 + alpha*(x - mu(t))) * exp(-lambda*t)

    where:
        mu(t) = mu_0 + delta * t        (time-dependent mean)
        sigma(t) = sigma_0 * (1 + beta * t)  (time-dependent volatility)

    Attributes:
        mu_0: Initial location parameter (mean at t=0)
        sigma_0: Initial scale parameter (volatility at t=0), must be positive
        alpha: Skewness parameter (asymmetry). Positive values skew right, negative skew left.
        delta: Mean drift rate (trend in location over time)
        beta: Volatility growth rate (scale expansion over time)
        k: Tail sharpness parameter (k=0 approaches Gaussian, k=1 is Laplace-like)
        lambda_decay: Time decay parameter for prediction confidence
    """

    mu_0: float
    sigma_0: float
    alpha: float = 0.0
    delta: float = 0.0
    beta: float = 0.0
    k: float = 1.0
    lambda_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_0 <= 0:
            raise ValueError("sigma_0 must be positive")
        if self.k < 0:
            raise ValueError("k must be non-negative")
        if self.lambda_decay < 0:
            raise ValueError("lambda_decay must be non-negative")


@dataclass(frozen=True)
class NormalParameters(DistributionParameters):
    """
    Parameters for time-evolving Normal (Gaussian) distribution.

    Mathematical form:
        f(x, t) = (1 / (sigma(t) * sqrt(2*pi))) * exp(-(x - mu(t))^2 / (2*sigma(t)^2))

    where:
        mu(t) = mu_0 + delta * t
        sigma(t) = sigma_0 * (1 + beta * t)

    Attributes:
        mu_0: Initial mean (location at t=0)
        sigma_0: Initial standard deviation (scale at t=0), must be positive
        delta: Mean drift rate per unit time
        beta: Volatility growth rate (multiplicative factor)
    """

    mu_0: float
    sigma_0: float
    delta: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_0 <= 0:
            raise ValueError("sigma_0 must be positive")

    def with_mu_0(self, new_mu_0: float) -> "NormalParameters":
        """Return new params with updated mu_0."""
        return NormalParameters(mu_0=new_mu_0, sigma_0=self.sigma_0, delta=self.delta, beta=self.beta)

    def with_sigma_0(self, new_sigma_0: float) -> "NormalParameters":
        """Return new params with updated sigma_0."""
        return NormalParameters(mu_0=self.mu_0, sigma_0=new_sigma_0, delta=self.delta, beta=self.beta)


@dataclass(frozen=True)
class StudentTParameters(DistributionParameters):
    """
    Parameters for time-evolving Student's t distribution.

    The Student's t distribution has heavier tails than the Normal distribution,
    making it useful for modeling data with outliers or fat-tailed behavior.

    Attributes:
        mu_0: Initial location parameter
        sigma_0: Initial scale parameter, must be positive
        nu: Degrees of freedom, must be positive. Lower values = heavier tails.
            nu > 30 approximates Normal distribution.
        delta: Mean drift rate per unit time
        beta: Volatility growth rate (multiplicative factor)
    """

    mu_0: float
    sigma_0: float
    nu: float
    delta: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_0 <= 0:
            raise ValueError("sigma_0 must be positive")
        if self.nu <= 0:
            raise ValueError("nu (degrees of freedom) must be positive")

    def with_mu_0(self, new_mu_0: float) -> "StudentTParameters":
        """Return new params with updated mu_0."""
        return StudentTParameters(mu_0=new_mu_0, sigma_0=self.sigma_0, nu=self.nu, delta=self.delta, beta=self.beta)

    def with_sigma_0(self, new_sigma_0: float) -> "StudentTParameters":
        """Return new params with updated sigma_0."""
        return StudentTParameters(mu_0=self.mu_0, sigma_0=new_sigma_0, nu=self.nu, delta=self.delta, beta=self.beta)

    def with_nu(self, new_nu: float) -> "StudentTParameters":
        """Return new params with updated nu."""
        return StudentTParameters(mu_0=self.mu_0, sigma_0=self.sigma_0, nu=new_nu, delta=self.delta, beta=self.beta)


@dataclass(frozen=True)
class SkewNormalParameters(DistributionParameters):
    """
    Parameters for time-evolving Skew-Normal distribution.

    The Skew-Normal distribution extends the Normal distribution with an
    asymmetry parameter, allowing it to model data with skewed distributions.

    Attributes:
        mu_0: Initial location parameter (called xi in scipy)
        sigma_0: Initial scale parameter (called omega in scipy), must be positive
        alpha: Skewness parameter (called a in scipy). alpha=0 gives Normal distribution.
               Positive values skew right, negative values skew left.
        delta: Mean drift rate per unit time
        beta: Volatility growth rate (multiplicative factor)
    """

    mu_0: float
    sigma_0: float
    alpha: float
    delta: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        if self.sigma_0 <= 0:
            raise ValueError("sigma_0 must be positive")

    def with_mu_0(self, new_mu_0: float) -> "SkewNormalParameters":
        """Return new params with updated mu_0."""
        return SkewNormalParameters(mu_0=new_mu_0, sigma_0=self.sigma_0, alpha=self.alpha, delta=self.delta, beta=self.beta)

    def with_sigma_0(self, new_sigma_0: float) -> "SkewNormalParameters":
        """Return new params with updated sigma_0."""
        return SkewNormalParameters(mu_0=self.mu_0, sigma_0=new_sigma_0, alpha=self.alpha, delta=self.delta, beta=self.beta)

    def with_alpha(self, new_alpha: float) -> "SkewNormalParameters":
        """Return new params with updated alpha (skewness)."""
        return SkewNormalParameters(mu_0=self.mu_0, sigma_0=self.sigma_0, alpha=new_alpha, delta=self.delta, beta=self.beta)

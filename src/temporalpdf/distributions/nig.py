"""Normal Inverse Gaussian (NIG) distribution with time evolution.

References:
    Barndorff-Nielsen, O.E. (1997). Normal Inverse Gaussian Distributions and
    Stochastic Volatility Modelling. Scandinavian Journal of Statistics, 24(1), 1-13.

    Barndorff-Nielsen, O.E. (1998). Processes of Normal Inverse Gaussian Type.
    Finance and Stochastics, 2(1), 41-68.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy import special

from ..core.distribution import TimeEvolvingDistribution

if TYPE_CHECKING:
    from ..core.volatility import VolatilityModel


@dataclass(frozen=True)
class NIGParameters:
    """
    Parameters for the Normal Inverse Gaussian distribution.

    The NIG distribution is parameterized by four quantities:
    - mu: Location parameter (real)
    - delta: Scale parameter (> 0) - initial scale at t=0
    - alpha: Steepness/tail heaviness (> |beta|)
    - beta: Skewness parameter (|beta| < alpha)

    For time evolution, you can use either:
    1. Simple linear growth: delta_growth parameter
    2. Sophisticated models: volatility_model parameter

    If volatility_model is provided, it overrides delta_growth.

    Examples:
        # Simple linear growth (legacy)
        params = NIGParameters(mu=0, delta=0.02, alpha=15, beta=-2, delta_growth=0.05)

        # Mean-reverting volatility
        from temporalpdf import mean_reverting
        params = NIGParameters(
            mu=0, delta=0.03,  # Current elevated vol
            alpha=15, beta=-2,
            volatility_model=mean_reverting(sigma_long=0.02, kappa=0.1)
        )

        # GARCH-style forecast
        from temporalpdf import garch_forecast
        params = NIGParameters(
            mu=0, delta=0.025, alpha=15, beta=-2,
            volatility_model=garch_forecast(omega=0.00001, alpha=0.1, beta=0.85)
        )

    Constraints:
        alpha > 0, delta > 0, |beta| < alpha
    """

    mu: float
    delta: float
    alpha: float
    beta: float
    mu_drift: float = 0.0
    delta_growth: float = 0.0
    volatility_model: "VolatilityModel | None" = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if self.delta <= 0:
            raise ValueError(f"delta must be positive, got {self.delta}")
        if self.alpha <= 0:
            raise ValueError(f"alpha must be positive, got {self.alpha}")
        if abs(self.beta) >= self.alpha:
            raise ValueError(
                f"|beta| must be less than alpha, got |{self.beta}| >= {self.alpha}"
            )

    def with_mu(self, new_mu: float) -> "NIGParameters":
        """Return new params with updated mu."""
        return NIGParameters(
            mu=new_mu, delta=self.delta, alpha=self.alpha, beta=self.beta,
            mu_drift=self.mu_drift, delta_growth=self.delta_growth,
            volatility_model=self.volatility_model
        )

    def with_delta(self, new_delta: float) -> "NIGParameters":
        """Return new params with updated delta (scale)."""
        return NIGParameters(
            mu=self.mu, delta=new_delta, alpha=self.alpha, beta=self.beta,
            mu_drift=self.mu_drift, delta_growth=self.delta_growth,
            volatility_model=self.volatility_model
        )

    def with_alpha(self, new_alpha: float) -> "NIGParameters":
        """Return new params with updated alpha (tail heaviness)."""
        return NIGParameters(
            mu=self.mu, delta=self.delta, alpha=new_alpha, beta=self.beta,
            mu_drift=self.mu_drift, delta_growth=self.delta_growth,
            volatility_model=self.volatility_model
        )

    def with_beta(self, new_beta: float) -> "NIGParameters":
        """Return new params with updated beta (skewness)."""
        return NIGParameters(
            mu=self.mu, delta=self.delta, alpha=self.alpha, beta=new_beta,
            mu_drift=self.mu_drift, delta_growth=self.delta_growth,
            volatility_model=self.volatility_model
        )

    def _delta_at_time(self, t: float) -> float:
        """Compute delta at time t using the configured volatility model."""
        if self.volatility_model is not None:
            return self.volatility_model.at_time(self.delta, t)
        return self.delta * (1 + self.delta_growth * t)

    def _delta_at_times(self, t: np.ndarray) -> np.ndarray:
        """Compute delta at multiple times (vectorized)."""
        if self.volatility_model is not None:
            return self.volatility_model.at_times(self.delta, t)
        return self.delta * (1 + self.delta_growth * t)


class NIGDistribution(TimeEvolvingDistribution[NIGParameters]):
    """
    Normal Inverse Gaussian (NIG) distribution with time evolution.

    The NIG distribution is a four-parameter family that captures:
    - Semi-heavy tails (heavier than Normal, lighter than Pareto)
    - Asymmetry/skewness
    - Excess kurtosis

    It is widely used in financial modeling because:
    1. Closed under convolution (daily -> weekly returns tractable)
    2. Captures stylized facts of financial returns
    3. Has interpretable parameters

    Mathematical formulation:
        f(x; α, β, μ, δ) = (αδ/π) * exp(δγ + β(x-μ)) * K₁(α*q(x)) / q(x)

    where:
        γ = sqrt(α² - β²)
        q(x) = sqrt(δ² + (x - μ)²)
        K₁ is the modified Bessel function of the second kind

    References:
        Barndorff-Nielsen, O.E. (1997). Normal Inverse Gaussian Distributions
        and Stochastic Volatility Modelling.
    """

    @property
    def name(self) -> str:
        return "Normal Inverse Gaussian (NIG)"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu", "delta", "alpha", "beta", "mu_drift", "delta_growth")

    def pdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: NIGParameters,
    ) -> NDArray[np.float64]:
        """
        Evaluate the NIG probability density function.

        Args:
            x: Array of values to evaluate
            t: Time point
            params: Distribution parameters

        Returns:
            Array of probability density values
        """
        x = np.asarray(x, dtype=np.float64)

        # Time-evolved parameters
        mu_t = params.mu + params.mu_drift * t
        delta_t = params._delta_at_time(t)

        alpha = params.alpha
        beta = params.beta

        # Derived quantity
        gamma = np.sqrt(alpha**2 - beta**2)

        # Argument for Bessel function: q(x) = sqrt(delta^2 + (x - mu)^2)
        q_x = np.sqrt(delta_t**2 + (x - mu_t) ** 2)

        # Prevent numerical issues with very small q values
        q_x = np.maximum(q_x, 1e-300)

        # Log-space computation for numerical stability
        log_pdf = (
            np.log(alpha)
            + np.log(delta_t)
            - np.log(np.pi)
            + delta_t * gamma
            + beta * (x - mu_t)
            + np.log(special.kv(1, alpha * q_x))
            - np.log(q_x)
        )

        return np.exp(log_pdf)

    def pdf_matrix(
        self,
        x: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        params: NIGParameters,
    ) -> NDArray[np.float64]:
        """
        Evaluate the PDF over a 2D grid of (time, value).

        Vectorized implementation for performance.

        Args:
            x: Array of values (value axis)
            time_grid: Array of time points (time axis)
            params: Distribution parameters

        Returns:
            2D array of shape (len(time_grid), len(x))
        """
        x = np.asarray(x, dtype=np.float64)
        time_grid = np.asarray(time_grid, dtype=np.float64)

        # Broadcast: time_grid[:, None] and x[None, :]
        t = time_grid[:, np.newaxis]  # (T, 1)

        # Time-evolved parameters (broadcasting)
        mu_t = params.mu + params.mu_drift * t  # (T, 1)
        delta_t = params._delta_at_times(time_grid)[:, np.newaxis]  # (T, 1)

        alpha = params.alpha
        beta = params.beta
        gamma = np.sqrt(alpha**2 - beta**2)

        # Compute q(x) for all (t, x) pairs
        q_x = np.sqrt(delta_t**2 + (x - mu_t) ** 2)  # (T, N)
        q_x = np.maximum(q_x, 1e-300)

        # Log-space computation
        log_pdf = (
            np.log(alpha)
            + np.log(delta_t)
            - np.log(np.pi)
            + delta_t * gamma
            + beta * (x - mu_t)
            + np.log(special.kv(1, alpha * q_x))
            - np.log(q_x)
        )

        return np.exp(log_pdf)

    def cdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: NIGParameters,
    ) -> NDArray[np.float64]:
        """
        Cumulative distribution function (numerical integration).

        Uses scipy.integrate.quad for accurate integration from practical
        -infinity to each query point. This properly captures tail probabilities.

        Args:
            x: Array of values
            t: Time point
            params: Distribution parameters

        Returns:
            Array of CDF values
        """
        from scipy import integrate

        x = np.asarray(x, dtype=np.float64)

        # Time-evolved parameters
        mu_t = params.mu + params.mu_drift * t
        delta_t = params._delta_at_time(t)

        # For each query point, integrate from practical -infinity
        # Use a wide range that captures >99.99% of probability
        lower_bound = mu_t - 20 * delta_t

        def pdf_scalar(z: float) -> float:
            return self.pdf(np.array([z]), t, params)[0]

        results = np.empty_like(x)
        for i, xi in enumerate(x):
            if xi <= lower_bound:
                results[i] = 0.0
            else:
                val, _ = integrate.quad(pdf_scalar, lower_bound, xi, limit=100)
                results[i] = np.clip(val, 0.0, 1.0)

        return results

    def ppf(
        self,
        q: NDArray[np.float64],
        t: float,
        params: NIGParameters,
    ) -> NDArray[np.float64]:
        """
        Percent point function (quantile function / inverse CDF).

        Uses scipy.optimize.brentq for accurate root-finding of CDF(x) = q.
        Falls back to grid interpolation if brentq fails.

        Args:
            q: Array of quantiles (0 < q < 1)
            t: Time point
            params: Distribution parameters

        Returns:
            Array of values corresponding to quantiles
        """
        from scipy import optimize

        q = np.asarray(q, dtype=np.float64)

        # Time-evolved parameters for determining search range
        mu_t = params.mu + params.mu_drift * t
        delta_t = params._delta_at_time(t)

        results = np.empty_like(q)
        for i, qi in enumerate(q):
            # Define function to find root of: CDF(x) - q = 0
            def objective(x: float) -> float:
                cdf_val = self.cdf(np.array([x]), t, params)[0]
                return cdf_val - qi

            # Search in a wide range
            x_low = mu_t - 15 * delta_t
            x_high = mu_t + 15 * delta_t

            try:
                result = optimize.brentq(objective, x_low, x_high)
                results[i] = result
            except ValueError:
                # Fallback to grid search if brentq fails
                x_grid = np.linspace(x_low, x_high, 10000)
                cdf_grid = self.cdf(x_grid, t, params)
                results[i] = np.interp(qi, cdf_grid, x_grid)

        return results

    def mean(self, t: float, params: NIGParameters) -> float:
        """
        Expected value of the distribution.

        E[X] = mu + delta * beta / gamma
        """
        mu_t = params.mu + params.mu_drift * t
        delta_t = params._delta_at_time(t)
        gamma = np.sqrt(params.alpha**2 - params.beta**2)

        return mu_t + delta_t * params.beta / gamma

    def variance(self, t: float, params: NIGParameters) -> float:
        """
        Variance of the distribution.

        Var[X] = delta * alpha^2 / gamma^3
        """
        delta_t = params._delta_at_time(t)
        gamma = np.sqrt(params.alpha**2 - params.beta**2)

        return delta_t * params.alpha**2 / gamma**3

    def skewness(self, t: float, params: NIGParameters) -> float:
        """
        Skewness of the distribution.

        Skew[X] = 3 * beta / (alpha * sqrt(delta * gamma))
        """
        delta_t = params._delta_at_time(t)
        gamma = np.sqrt(params.alpha**2 - params.beta**2)

        return 3 * params.beta / (params.alpha * np.sqrt(delta_t * gamma))

    def kurtosis(self, t: float, params: NIGParameters) -> float:
        """
        Excess kurtosis of the distribution.

        Kurt[X] = 3 * (1 + 4*beta^2/alpha^2) / (delta * gamma)
        """
        delta_t = params._delta_at_time(t)
        gamma = np.sqrt(params.alpha**2 - params.beta**2)

        return 3 * (1 + 4 * params.beta**2 / params.alpha**2) / (delta_t * gamma)

    def sample(
        self,
        n: int,
        t: float,
        params: NIGParameters,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """
        Draw n samples from the NIG distribution.

        Uses the normal-variance mixture representation:
            X = mu + beta*V + sqrt(V)*Z
        where V ~ InverseGaussian(delta, gamma) and Z ~ N(0,1).

        Args:
            n: Number of samples
            t: Time point
            params: Distribution parameters
            rng: Random number generator

        Returns:
            Array of n samples
        """
        if rng is None:
            rng = np.random.default_rng()

        # Time-evolved parameters
        mu_t = params.mu + params.mu_drift * t
        delta_t = params._delta_at_time(t)
        gamma = np.sqrt(params.alpha**2 - params.beta**2)

        # Sample from Inverse Gaussian
        # Using the transformation method
        chi = rng.standard_normal(n) ** 2
        v = delta_t / gamma
        w = v + (v**2 * chi - v * np.sqrt(chi * (4 * v + v**2 * chi))) / 2

        # Uniform for rejection
        u = rng.uniform(size=n)
        mask = u > v / (v + w)
        w[mask] = v**2 / w[mask]

        # Normal component
        z = rng.standard_normal(n)

        # NIG samples
        return mu_t + params.beta * w + np.sqrt(w) * z

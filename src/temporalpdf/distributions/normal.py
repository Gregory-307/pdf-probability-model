"""Time-evolving Normal (Gaussian) distribution."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..core.distribution import TimeEvolvingDistribution
from ..core.parameters import NormalParameters


class NormalDistribution(TimeEvolvingDistribution[NormalParameters]):
    """
    Time-evolving Normal (Gaussian) distribution.

    The classic bell-curve distribution with time-dependent location and scale.

    Mathematical formulation:
        f(x, t) = (1 / (sigma(t) * sqrt(2*pi))) * exp(-(x - mu(t))^2 / (2*sigma(t)^2))

    where:
        mu(t) = mu_0 + delta * t
        sigma(t) = sigma_0 * (1 + beta * t)

    Use cases:
    - When the underlying process is approximately Gaussian
    - When you expect symmetric uncertainty
    - As a baseline for comparison with more complex distributions
    """

    @property
    def name(self) -> str:
        return "Normal (Gaussian)"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu_0", "sigma_0", "delta", "beta")

    def pdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: NormalParameters,
    ) -> NDArray[np.float64]:
        """
        Evaluate the PDF at values x for time t.

        Args:
            x: Array of values to evaluate
            t: Time point
            params: Distribution parameters

        Returns:
            Array of probability density values
        """
        mu_t = params.mu_0 + params.delta * t
        sigma_t = params.sigma_0 * (1 + params.beta * t)
        return norm.pdf(x, loc=mu_t, scale=sigma_t)

    def pdf_matrix(
        self,
        x: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        params: NormalParameters,
    ) -> NDArray[np.float64]:
        """
        Evaluate the PDF over a 2D grid of (time, value).

        Args:
            x: Array of values (value axis)
            time_grid: Array of time points (time axis)
            params: Distribution parameters

        Returns:
            2D array of shape (len(time_grid), len(x))
        """
        time_grid = np.asarray(time_grid)
        pdf_matrix = np.zeros((len(time_grid), len(x)))

        for i, t in enumerate(time_grid):
            pdf_matrix[i, :] = self.pdf(x, float(t), params)

        return pdf_matrix

    def cdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: NormalParameters,
    ) -> NDArray[np.float64]:
        """
        Evaluate the CDF at values x for time t.

        Args:
            x: Array of values to evaluate
            t: Time point
            params: Distribution parameters

        Returns:
            Array of cumulative probability values
        """
        mu_t = params.mu_0 + params.delta * t
        sigma_t = params.sigma_0 * (1 + params.beta * t)
        return norm.cdf(x, loc=mu_t, scale=sigma_t)

    def quantile(
        self,
        p: NDArray[np.float64] | float,
        t: float,
        params: NormalParameters,
    ) -> NDArray[np.float64] | float:
        """
        Evaluate the quantile function (inverse CDF) at probability p for time t.

        Args:
            p: Probability value(s) between 0 and 1
            t: Time point
            params: Distribution parameters

        Returns:
            Quantile value(s)
        """
        mu_t = params.mu_0 + params.delta * t
        sigma_t = params.sigma_0 * (1 + params.beta * t)
        return norm.ppf(p, loc=mu_t, scale=sigma_t)

    # V2 API alias
    ppf = quantile

    def sample(
        self,
        n: int,
        t: float,
        params: NormalParameters,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """
        Draw n samples from the Normal distribution.

        Args:
            n: Number of samples
            t: Time point
            params: Distribution parameters
            rng: Random number generator (optional)

        Returns:
            Array of n samples
        """
        mu_t = params.mu_0 + params.delta * t
        sigma_t = params.sigma_0 * (1 + params.beta * t)
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(loc=mu_t, scale=sigma_t, size=n)

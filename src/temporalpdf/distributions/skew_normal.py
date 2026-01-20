"""Time-evolving Skew-Normal distribution."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import skewnorm

from ..core.distribution import TimeEvolvingDistribution
from ..core.parameters import SkewNormalParameters


class SkewNormalDistribution(TimeEvolvingDistribution[SkewNormalParameters]):
    """
    Time-evolving Skew-Normal distribution.

    The Skew-Normal distribution extends the Normal distribution with an
    asymmetry parameter, allowing it to model data with skewed distributions
    while maintaining most of the tractability of the Normal.

    Mathematical formulation:
        f(x, t) = (2 / sigma(t)) * phi((x - mu(t)) / sigma(t)) * Phi(alpha * (x - mu(t)) / sigma(t))

    where:
        phi(z) = standard normal PDF
        Phi(z) = standard normal CDF
        mu(t) = mu_0 + delta * t
        sigma(t) = sigma_0 * (1 + beta * t)

    Use cases:
    - Data with moderate asymmetry
    - When Normal is too symmetric but full flexibility isn't needed
    - Financial applications with directional bias
    - Biological measurements with natural skew

    Notes on alpha (skewness parameter):
    - alpha = 0: Standard Normal distribution
    - alpha > 0: Right (positive) skew
    - alpha < 0: Left (negative) skew
    - |alpha| -> infinity: approaches half-Normal
    """

    @property
    def name(self) -> str:
        return "Skew-Normal"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu_0", "sigma_0", "alpha", "delta", "beta")

    def pdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: SkewNormalParameters,
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

        # scipy's skewnorm uses (a, loc, scale) parameterization
        # where a is the shape parameter (skewness)
        return skewnorm.pdf(x, a=params.alpha, loc=mu_t, scale=sigma_t)

    def pdf_matrix(
        self,
        x: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        params: SkewNormalParameters,
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
        params: SkewNormalParameters,
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
        return skewnorm.cdf(x, a=params.alpha, loc=mu_t, scale=sigma_t)

    def quantile(
        self,
        p: NDArray[np.float64] | float,
        t: float,
        params: SkewNormalParameters,
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
        return skewnorm.ppf(p, a=params.alpha, loc=mu_t, scale=sigma_t)

    # V2 API alias
    ppf = quantile

    def sample(
        self,
        n: int,
        t: float,
        params: SkewNormalParameters,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """
        Draw n samples from the Skew-Normal distribution.

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
        # scipy's skewnorm.rvs provides sampling
        return skewnorm.rvs(a=params.alpha, loc=mu_t, scale=sigma_t, size=n)

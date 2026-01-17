"""Time-evolving Student's t distribution."""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t

from ..core.distribution import TimeEvolvingDistribution
from ..core.parameters import StudentTParameters


class StudentTDistribution(TimeEvolvingDistribution[StudentTParameters]):
    """
    Time-evolving Student's t distribution.

    The Student's t distribution has heavier tails than the Normal distribution,
    making it useful for modeling data with outliers or fat-tailed behavior.
    As degrees of freedom (nu) increases, the distribution approaches Normal.

    Mathematical formulation:
        f(x, t) = (Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2))) *
                  (1 + ((x - mu(t)) / sigma(t))^2 / nu)^(-(nu+1)/2) / sigma(t)

    where:
        mu(t) = mu_0 + delta * t
        sigma(t) = sigma_0 * (1 + beta * t)

    Use cases:
    - Financial returns (often exhibit fat tails)
    - Data with potential outliers
    - When Normal assumption is too restrictive
    - Small sample inference

    Notes on degrees of freedom (nu):
    - nu = 1: Cauchy distribution (very heavy tails)
    - nu = 2-3: Heavy tails, variance may be infinite
    - nu > 30: Approximates Normal distribution
    """

    @property
    def name(self) -> str:
        return "Student's t"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu_0", "sigma_0", "nu", "delta", "beta")

    def pdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: StudentTParameters,
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

        # scipy's t distribution uses loc and scale parameters
        return student_t.pdf(x, df=params.nu, loc=mu_t, scale=sigma_t)

    def pdf_matrix(
        self,
        x: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        params: StudentTParameters,
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
        params: StudentTParameters,
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
        return student_t.cdf(x, df=params.nu, loc=mu_t, scale=sigma_t)

    def quantile(
        self,
        p: NDArray[np.float64] | float,
        t: float,
        params: StudentTParameters,
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
        return student_t.ppf(p, df=params.nu, loc=mu_t, scale=sigma_t)

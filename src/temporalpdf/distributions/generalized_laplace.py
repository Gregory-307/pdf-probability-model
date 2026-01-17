"""Generalized Exponential Power distribution with skew and time evolution.

This is a custom distribution combining elements from:
- Exponential Power (Generalized Normal) distribution: Nadarajah (2005)
- Asymmetric Laplace distribution: Kotz et al. (2001)

References:
    Nadarajah, S. (2005). A generalized normal distribution. Journal of
    Applied Statistics, 32(7), 685-694.

    Kotz, S., Kozubowski, T.J., & Podgorski, K. (2001). The Laplace
    Distribution and Generalizations. Birkhäuser.

Note: For financial applications, consider using the NIG (Normal Inverse
Gaussian) distribution instead, which has stronger theoretical foundations
and is more widely used in quantitative finance.
"""

import numpy as np
from numpy.typing import NDArray

from ..core.distribution import TimeEvolvingDistribution
from ..core.parameters import GeneralizedLaplaceParameters


class GeneralizedLaplaceDistribution(TimeEvolvingDistribution[GeneralizedLaplaceParameters]):
    """
    Generalized Exponential Power distribution with skew and time evolution.

    This is a flexible custom distribution combining exponential power
    tails with linear skew adjustment. It extends the Laplace/Gaussian
    family with:
    - Time-evolving location (mu) and scale (sigma)
    - Skewness parameter for asymmetry
    - Adjustable tail sharpness via exponent k
    - Optional time decay for prediction confidence

    Mathematical formulation:
        f(x, t) = C(t) * base(x, t) * skew(x, t) * decay(t)

    where:
        base(x, t) = exp(-|x - mu(t)|^(1+k) / (2*sigma(t)^2))
        skew(x, t) = max(0, 1 + alpha*(x - mu(t)))
        decay(t) = exp(-lambda * t)
        mu(t) = mu_0 + delta * t
        sigma(t) = sigma_0 * (1 + beta * t)
        C(t) = normalization constant

    Special cases:
        k=1, alpha=0: Approximately Gaussian
        k=0, alpha=0: Laplace distribution

    Note:
        For financial applications with heavy tails and skewness, the NIG
        (Normal Inverse Gaussian) distribution is recommended instead, as
        it has stronger theoretical properties and academic support.

    See Also:
        NIGDistribution: Preferred for financial return modeling
    """

    @property
    def name(self) -> str:
        return "Generalized Laplace with Skew"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return ("mu_0", "sigma_0", "alpha", "delta", "beta", "k", "lambda_decay")

    def pdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: GeneralizedLaplaceParameters,
    ) -> NDArray[np.float64]:
        """
        Evaluate the PDF at values x for time t.

        Args:
            x: Array of values to evaluate
            t: Time point
            params: Distribution parameters

        Returns:
            Array of probability density values (normalized)
        """
        # Time-dependent parameters
        mu_t = params.mu_0 + params.delta * t
        sigma_t = params.sigma_0 * (1 + params.beta * t)

        # Compute components
        base = np.exp(-np.abs(x - mu_t) ** (1 + params.k) / (2 * sigma_t**2))
        skew = np.maximum(0.0, 1 + params.alpha * (x - mu_t))
        decay = np.exp(-params.lambda_decay * t)

        raw_pdf = base * skew * decay

        # Normalize using trapezoidal integration
        normalization = np.trapezoid(raw_pdf, x)
        if normalization <= 0:
            normalization = 1.0

        return raw_pdf / normalization

    def pdf_matrix(
        self,
        x: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        params: GeneralizedLaplaceParameters,
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

        # Time-dependent parameters (broadcasting)
        mu_t = params.mu_0 + params.delta * t  # (T, 1)
        sigma_t = params.sigma_0 * (1 + params.beta * t)  # (T, 1)

        # Compute components vectorized
        base = np.exp(-np.abs(x - mu_t) ** (1 + params.k) / (2 * sigma_t**2))
        skew = np.maximum(0.0, 1 + params.alpha * (x - mu_t))
        decay = np.exp(-params.lambda_decay * t)

        raw_pdf = base * skew * decay  # (T, N)

        # Normalize each row using trapezoidal integration
        dx = x[1] - x[0] if len(x) > 1 else 1.0
        normalization = np.trapezoid(raw_pdf, x, axis=1, dx=dx)[:, np.newaxis]
        normalization = np.maximum(normalization, 1e-10)

        return raw_pdf / normalization

    def pdf_no_decay(
        self,
        x: NDArray[np.float64],
        t: float,
        params: GeneralizedLaplaceParameters,
    ) -> NDArray[np.float64]:
        """
        Evaluate the PDF without time decay (constant confidence).

        Useful when you want to model the distribution shape without
        the confidence reduction over time.

        Args:
            x: Array of values to evaluate
            t: Time point
            params: Distribution parameters

        Returns:
            Array of probability density values (normalized)
        """
        # Time-dependent parameters
        mu_t = params.mu_0 + params.delta * t
        sigma_t = params.sigma_0 * (1 + params.beta * t)

        # Compute components (no decay)
        base = np.exp(-np.abs(x - mu_t) ** (1 + params.k) / (2 * sigma_t**2))
        skew = np.maximum(0.0, 1 + params.alpha * (x - mu_t))

        raw_pdf = base * skew

        # Normalize
        normalization = np.trapezoid(raw_pdf, x)
        if normalization <= 0:
            normalization = 1.0

        return raw_pdf / normalization

"""Base distribution protocol and abstract class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar, Generic, Protocol, runtime_checkable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray, ArrayLike

if TYPE_CHECKING:
    from ..temporal.weights.base import WeightScheme


@dataclass(frozen=True)
class DistributionParameters:
    """
    Base class for distribution parameters.

    All parameter classes should inherit from this and use frozen=True
    for immutability.
    """

    pass


P = TypeVar("P", bound=DistributionParameters)


# =============================================================================
# V2 PROTOCOL - The minimal interface all distributions MUST implement
# =============================================================================


@runtime_checkable
class Distribution(Protocol):
    """
    Protocol that all distributions must implement (V2 API).

    This is the minimal interface required for integration with the
    temporal modeling pipeline, backtesting, and decision utilities.
    """

    def pdf(self, x: ArrayLike, t: float, params: DistributionParameters) -> ArrayLike:
        """Probability density function."""
        ...

    def cdf(self, x: ArrayLike, t: float, params: DistributionParameters) -> ArrayLike:
        """Cumulative distribution function."""
        ...

    def ppf(self, q: ArrayLike, t: float, params: DistributionParameters) -> ArrayLike:
        """Percent point function (inverse CDF / quantile function)."""
        ...

    def sample(self, n: int, t: float, params: DistributionParameters) -> ArrayLike:
        """Generate random samples."""
        ...


# =============================================================================
# V1 ABC - Kept for backwards compatibility
# =============================================================================


class TimeEvolvingDistribution(ABC, Generic[P]):
    """
    Abstract base class for time-evolving probability distributions.

    A time-evolving distribution models how a probability density function
    changes over a continuous time dimension, capturing dynamics like
    shifting means, growing volatility, and changing shape parameters.

    Type Parameters:
        P: The parameter class for this distribution (must inherit from DistributionParameters)

    Example:
        >>> class MyDistribution(TimeEvolvingDistribution[MyParameters]):
        ...     def pdf(self, x, t, params):
        ...         # implementation
        ...         pass
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable distribution name."""
        ...

    @property
    @abstractmethod
    def parameter_names(self) -> tuple[str, ...]:
        """Names of the distribution parameters."""
        ...

    @abstractmethod
    def pdf(
        self,
        x: NDArray[np.float64],
        t: float,
        params: P,
    ) -> NDArray[np.float64]:
        """
        Evaluate the PDF at values x for time t.

        Args:
            x: Array of values to evaluate
            t: Time point
            params: Distribution parameters

        Returns:
            Array of probability density values (normalized to integrate to 1)
        """
        ...

    @abstractmethod
    def pdf_matrix(
        self,
        x: NDArray[np.float64],
        time_grid: NDArray[np.float64],
        params: P,
    ) -> NDArray[np.float64]:
        """
        Evaluate the PDF over a 2D grid of (time, value).

        Args:
            x: Array of values (value axis)
            time_grid: Array of time points (time axis)
            params: Distribution parameters

        Returns:
            2D array of shape (len(time_grid), len(x)) where each row
            is the PDF at that time point
        """
        ...

    def expected_value(
        self,
        x: NDArray[np.float64],
        t: float,
        params: P,
    ) -> float:
        """
        Calculate expected value E[X] at time t.

        Args:
            x: Array of values for integration
            t: Time point
            params: Distribution parameters

        Returns:
            Expected value at time t
        """
        pdf_values = self.pdf(x, t, params)
        return float(np.trapezoid(x * pdf_values, x))

    def variance(
        self,
        x: NDArray[np.float64],
        t: float,
        params: P,
    ) -> float:
        """
        Calculate variance Var[X] at time t.

        Args:
            x: Array of values for integration
            t: Time point
            params: Distribution parameters

        Returns:
            Variance at time t
        """
        pdf_values = self.pdf(x, t, params)
        mean = self.expected_value(x, t, params)
        return float(np.trapezoid((x - mean) ** 2 * pdf_values, x))

    def std(
        self,
        x: NDArray[np.float64],
        t: float,
        params: P,
    ) -> float:
        """
        Calculate standard deviation at time t.

        Args:
            x: Array of values for integration
            t: Time point
            params: Distribution parameters

        Returns:
            Standard deviation at time t
        """
        return float(np.sqrt(self.variance(x, t, params)))

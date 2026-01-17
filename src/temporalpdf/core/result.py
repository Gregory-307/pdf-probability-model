"""Result container types for PDF evaluation and validation."""

from dataclasses import dataclass, field
from typing import Any
from datetime import datetime
import numpy as np
from numpy.typing import NDArray


@dataclass
class PDFResult:
    """
    Container for PDF evaluation results.

    Immutable result object that captures both the computed PDF
    and the metadata about how it was computed.

    Attributes:
        pdf_matrix: 2D array of shape (time_points, value_points)
        value_grid: 1D array of values (x-axis)
        time_grid: 1D array of time points (t-axis)
        distribution_name: Name of the distribution used
        parameters: Dictionary of parameter values
        computed_at: Timestamp of computation

    Example:
        >>> result = distribution.evaluate(grid, params)
        >>> print(f"E[X] over time: {result.cumulative_expected_value:.4f}")
    """

    pdf_matrix: NDArray[np.float64]
    value_grid: NDArray[np.float64]
    time_grid: NDArray[np.float64]
    distribution_name: str
    parameters: dict[str, Any]
    computed_at: datetime = field(default_factory=datetime.now)

    @property
    def expected_values(self) -> NDArray[np.float64]:
        """
        Expected value E[X] at each time step.

        Returns:
            1D array of expected values, one per time point
        """
        pdf_sum = np.sum(self.pdf_matrix, axis=1)
        pdf_sum = np.where(pdf_sum == 0, 1, pdf_sum)
        return np.dot(self.pdf_matrix, self.value_grid) / pdf_sum

    @property
    def cumulative_expected_value(self) -> float:
        """
        Cumulative expected value over the time range.

        Integrates E[X] over time using trapezoidal rule.

        Returns:
            Total expected value weighted by time
        """
        time_intervals = np.diff(self.time_grid, prepend=0)
        return float(np.sum(self.expected_values * time_intervals))

    def slice_at_time(self, t: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get PDF slice at a specific time.

        Args:
            t: Time point to slice at

        Returns:
            Tuple of (value_grid, pdf_values) at the nearest time point
        """
        idx = int(np.abs(self.time_grid - t).argmin())
        return self.value_grid, self.pdf_matrix[idx, :]

    def expected_value_at_time(self, t: float) -> float:
        """
        Get expected value at a specific time.

        Args:
            t: Time point

        Returns:
            E[X] at the nearest time point
        """
        idx = int(np.abs(self.time_grid - t).argmin())
        return float(self.expected_values[idx])

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the PDF matrix (time_points, value_points)."""
        return (self.pdf_matrix.shape[0], self.pdf_matrix.shape[1])


@dataclass
class ValidationResult:
    """
    Container for model validation results.

    Holds computed validation metrics comparing predicted distributions
    against observed data.

    Attributes:
        log_likelihood: Average log-likelihood of observed values under the PDF
        mae: Mean Absolute Error between expected and observed values
        mse: Mean Squared Error between expected and observed values
        r_squared: Coefficient of determination (R^2)
        total_samples: Number of samples used in validation
        per_sample_metrics: Optional array of per-sample metrics for analysis
    """

    log_likelihood: float
    mae: float
    mse: float
    r_squared: float
    total_samples: int
    per_sample_metrics: NDArray[np.float64] | None = None

    def summary(self) -> str:
        """
        Return human-readable summary of validation results.

        Returns:
            Formatted string with all metrics
        """
        return (
            f"Validation Results (n={self.total_samples}):\n"
            f"  Log-Likelihood: {self.log_likelihood:.4f}\n"
            f"  MAE: {self.mae:.6f}\n"
            f"  MSE: {self.mse:.6f}\n"
            f"  RMSE: {np.sqrt(self.mse):.6f}\n"
            f"  R-squared: {self.r_squared:.4f}"
        )

    @property
    def rmse(self) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(self.mse))

    def to_dict(self) -> dict[str, Any]:
        """
        Convert results to dictionary.

        Returns:
            Dictionary with all metrics
        """
        return {
            "log_likelihood": self.log_likelihood,
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
            "r_squared": self.r_squared,
            "total_samples": self.total_samples,
        }

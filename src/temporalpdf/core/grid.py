"""Grid management for PDF evaluation."""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


@dataclass
class EvaluationGrid:
    """
    Manages the value and time grids for PDF evaluation.

    Provides a clean interface for defining the domain over which
    distributions are evaluated. The grid defines both the range of
    values (x-axis) and time points (t-axis) for the PDF.

    Attributes:
        value_grid: 1D array of values (x-axis)
        time_grid: 1D array of time points (t-axis)

    Example:
        >>> grid = EvaluationGrid.from_ranges(
        ...     value_range=(-0.2, 0.2),
        ...     time_range=(0, 60),
        ...     value_points=200,
        ...     time_points=100
        ... )
        >>> print(grid.shape)
        (100, 200)
    """

    value_grid: NDArray[np.float64]
    time_grid: NDArray[np.float64]

    @classmethod
    def from_ranges(
        cls,
        value_range: tuple[float, float] = (-0.2, 0.2),
        time_range: tuple[float, float] = (0.0, 60.0),
        value_points: int = 200,
        time_points: int = 100,
    ) -> "EvaluationGrid":
        """
        Create a grid from range specifications.

        Args:
            value_range: (min, max) for value grid
            time_range: (min, max) for time grid
            value_points: Number of points in value grid
            time_points: Number of points in time grid

        Returns:
            EvaluationGrid with linearly spaced grids
        """
        return cls(
            value_grid=np.linspace(value_range[0], value_range[1], value_points),
            time_grid=np.linspace(time_range[0], time_range[1], time_points),
        )

    @property
    def meshgrid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Return meshgrid for 3D plotting.

        Returns:
            Tuple of (X, T) meshgrids suitable for matplotlib's plot_surface
        """
        return np.meshgrid(self.value_grid, self.time_grid)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Shape of the resulting PDF matrix.

        Returns:
            (time_points, value_points) tuple
        """
        return (len(self.time_grid), len(self.value_grid))

    @property
    def value_range(self) -> tuple[float, float]:
        """Return the (min, max) of the value grid."""
        return (float(self.value_grid.min()), float(self.value_grid.max()))

    @property
    def time_range(self) -> tuple[float, float]:
        """Return the (min, max) of the time grid."""
        return (float(self.time_grid.min()), float(self.time_grid.max()))

    @property
    def value_step(self) -> float:
        """Return the step size of the value grid."""
        if len(self.value_grid) < 2:
            return 0.0
        return float(self.value_grid[1] - self.value_grid[0])

    @property
    def time_step(self) -> float:
        """Return the step size of the time grid."""
        if len(self.time_grid) < 2:
            return 0.0
        return float(self.time_grid[1] - self.time_grid[0])

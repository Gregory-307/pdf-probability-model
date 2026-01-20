"""Base protocol for parameter dynamics models."""

from typing import Protocol
import numpy as np
from numpy.typing import NDArray


class DynamicsModel(Protocol):
    """
    Protocol for parameter dynamics models.

    Dynamics models describe how a parameter evolves over time.
    They can be fit to historical parameter series and used to
    project parameters forward with uncertainty.
    """

    def fit(self, param_series: NDArray[np.float64]) -> "DynamicsModel":
        """
        Fit model to historical parameter series.

        Args:
            param_series: Array of parameter values over time

        Returns:
            Self (for method chaining)
        """
        ...

    def project(
        self,
        current_value: float,
        horizon: int,
        n_paths: int = 1000,
    ) -> NDArray[np.float64]:
        """
        Project parameter forward using Monte Carlo simulation.

        Args:
            current_value: Current parameter value
            horizon: Number of steps to project
            n_paths: Number of simulation paths

        Returns:
            Array of shape (n_paths, horizon) with simulated paths
        """
        ...

    def summary(self) -> dict[str, float]:
        """
        Return fitted model parameters as a dictionary.

        Returns:
            Dictionary of model parameters
        """
        ...

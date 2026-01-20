"""Base protocol for weighting schemes."""

from typing import Protocol
import numpy as np
from numpy.typing import NDArray


class WeightScheme(Protocol):
    """
    Protocol for weighting schemes used in parameter estimation.

    Weighting schemes determine how much influence each observation has
    when fitting distribution parameters. This enables recent observations
    to have more influence than older ones.

    Convention:
        - Index 0 = most recent observation
        - Index n-1 = oldest observation
        - Weights always sum to 1
    """

    def get_weights(self, n: int) -> NDArray[np.float64]:
        """
        Return array of weights for n observations.

        Args:
            n: Number of observations

        Returns:
            Array of weights where index 0 is most recent.
            Weights sum to 1.
        """
        ...

    def effective_sample_size(self, n: int) -> float:
        """
        Return effective sample size given n observations.

        The effective sample size accounts for the concentration of weights.
        Equal weights give ESS = n, while concentrated weights give ESS < n.

        Formula: ESS = 1 / sum(w_i^2)

        Args:
            n: Number of observations

        Returns:
            Effective sample size (always <= n)
        """
        ...

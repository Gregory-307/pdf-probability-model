"""
Weighting scheme implementations.

All schemes follow the convention:
- Index 0 = most recent observation
- Index n-1 = oldest observation
- Weights sum to 1
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray


@dataclass
class SMA:
    """
    Simple Moving Average - equal weights over a window.

    All observations within the window get equal weight.
    Observations outside the window get zero weight.

    Args:
        window: Number of observations to include

    Example:
        >>> sma = SMA(window=20)
        >>> weights = sma.get_weights(100)
        >>> assert weights[:20].sum() == 1.0
        >>> assert weights[20:].sum() == 0.0
    """
    window: int

    def get_weights(self, n: int) -> NDArray[np.float64]:
        """Return equal weights for observations within window."""
        effective_n = min(n, self.window)
        weights = np.zeros(n)
        weights[:effective_n] = 1.0 / effective_n
        return weights

    def effective_sample_size(self, n: int) -> float:
        """ESS equals the window size (or n if smaller)."""
        return float(min(n, self.window))


@dataclass
class EMA:
    """
    Exponential Moving Average.

    Weights decay exponentially with age. The halflife parameter
    controls how quickly old observations lose influence.

    Formula: weight[i] = (1 - alpha) * alpha^i
    Where: alpha = exp(-ln(2) / halflife)

    Args:
        halflife: Number of observations for weight to decay by 50%

    Example:
        >>> ema = EMA(halflife=20)
        >>> weights = ema.get_weights(100)
        >>> assert weights[0] > weights[1] > weights[2]  # Decaying
        >>> assert abs(weights.sum() - 1.0) < 1e-10
    """
    halflife: float

    def get_weights(self, n: int) -> NDArray[np.float64]:
        """Return exponentially decaying weights."""
        alpha = np.exp(-np.log(2) / self.halflife)
        raw_weights = (1 - alpha) * (alpha ** np.arange(n))
        return raw_weights / raw_weights.sum()

    def effective_sample_size(self, n: int) -> float:
        """Compute ESS from sum of squared weights."""
        weights = self.get_weights(n)
        return 1.0 / np.sum(weights ** 2)


@dataclass
class Linear:
    """
    Linear decay weights.

    Weight decreases linearly from the most recent observation.
    Observations outside the window get zero weight.

    Formula: weight[i] = max(window - i, 0)
    Normalized to sum to 1.

    Args:
        window: Number of observations with non-zero weight

    Example:
        >>> linear = Linear(window=10)
        >>> weights = linear.get_weights(20)
        >>> # First observation has weight 10, second has 9, etc.
    """
    window: int

    def get_weights(self, n: int) -> NDArray[np.float64]:
        """Return linearly decaying weights."""
        raw_weights = np.maximum(self.window - np.arange(n), 0).astype(float)
        return raw_weights / raw_weights.sum()

    def effective_sample_size(self, n: int) -> float:
        """Compute ESS from sum of squared weights."""
        weights = self.get_weights(n)
        return 1.0 / np.sum(weights ** 2)


@dataclass
class PowerDecay:
    """
    Power decay weights.

    Weights decay as a power of the observation age.

    Formula: weight[i] = 1 / (i + 1)^power

    Args:
        power: Decay exponent. Higher = faster decay.
            - power=0.5: sqrt decay (slow)
            - power=1.0: 1/n decay (moderate)
            - power=2.0: 1/n^2 decay (fast)
        window: Optional window size (None = use all observations)

    Example:
        >>> power = PowerDecay(power=1.0)
        >>> weights = power.get_weights(100)
        >>> # weight[0] = 1, weight[1] = 0.5, weight[2] = 0.33, etc.
    """
    power: float
    window: int | None = None

    def get_weights(self, n: int) -> NDArray[np.float64]:
        """Return power-decaying weights."""
        raw_weights = 1.0 / ((np.arange(n) + 1) ** self.power)
        if self.window is not None:
            raw_weights[self.window:] = 0
        return raw_weights / raw_weights.sum()

    def effective_sample_size(self, n: int) -> float:
        """Compute ESS from sum of squared weights."""
        weights = self.get_weights(n)
        return 1.0 / np.sum(weights ** 2)


@dataclass
class Gaussian:
    """
    Gaussian decay weights.

    Weights follow a Gaussian curve centered at the most recent observation.

    Formula: weight[i] = exp(-0.5 * (i / sigma)^2)

    Args:
        sigma: Standard deviation of the Gaussian (in observation units)

    Example:
        >>> gauss = Gaussian(sigma=10)
        >>> weights = gauss.get_weights(50)
        >>> # Bell curve centered at index 0
    """
    sigma: float

    def get_weights(self, n: int) -> NDArray[np.float64]:
        """Return Gaussian-shaped weights."""
        raw_weights = np.exp(-0.5 * (np.arange(n) / self.sigma) ** 2)
        return raw_weights / raw_weights.sum()

    def effective_sample_size(self, n: int) -> float:
        """Compute ESS from sum of squared weights."""
        weights = self.get_weights(n)
        return 1.0 / np.sum(weights ** 2)


@dataclass
class Custom:
    """
    Custom weight function.

    Allows arbitrary weight functions defined by the user.

    Args:
        func: Callable (i, n) -> weight for observation i out of n total.
              i=0 is most recent. Weights will be normalized to sum to 1.

    Example:
        >>> # Triangular weights
        >>> custom = Custom(func=lambda i, n: max(0, 1 - i/50))
        >>> weights = custom.get_weights(100)
    """
    func: Callable[[int, int], float]

    def get_weights(self, n: int) -> NDArray[np.float64]:
        """Return custom weights based on user function."""
        raw_weights = np.array([self.func(i, n) for i in range(n)])
        return raw_weights / raw_weights.sum()

    def effective_sample_size(self, n: int) -> float:
        """Compute ESS from sum of squared weights."""
        weights = self.get_weights(n)
        return 1.0 / np.sum(weights ** 2)

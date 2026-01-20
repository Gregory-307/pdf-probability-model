"""
Weighting schemes for parameter estimation.

All schemes:
- Use index 0 = most recent observation
- Normalize weights to sum to 1
- Provide effective sample size calculation
"""

from .base import WeightScheme
from .schemes import (
    SMA,
    EMA,
    Linear,
    PowerDecay,
    Gaussian,
    Custom,
)

__all__ = [
    "WeightScheme",
    "SMA",
    "EMA",
    "Linear",
    "PowerDecay",
    "Gaussian",
    "Custom",
]

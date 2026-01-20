"""
Temporal modeling module for time-evolving parameter dynamics.

This module provides:
- Weighting schemes for parameter estimation (SMA, EMA, Linear, etc.)
- Parameter tracking over rolling windows
- Dynamics models (Constant, RandomWalk, MeanReverting, AR, GARCH)
- Projection and predictive distribution classes
- The central TemporalModel class that combines everything
"""

from .weights import (
    WeightScheme,
    SMA,
    EMA,
    Linear,
    PowerDecay,
    Gaussian,
    Custom,
)

from .tracking import ParameterTracker

from .dynamics import (
    DynamicsModel,
    Constant,
    RandomWalk,
    MeanReverting,
    AR,
    GARCH,
)

from .projection import Projection, ParamDistribution
from .predictive import PredictiveDistribution
from .model import TemporalModel

__all__ = [
    # Weighting schemes
    "WeightScheme",
    "SMA",
    "EMA",
    "Linear",
    "PowerDecay",
    "Gaussian",
    "Custom",
    # Tracking
    "ParameterTracker",
    # Dynamics
    "DynamicsModel",
    "Constant",
    "RandomWalk",
    "MeanReverting",
    "AR",
    "GARCH",
    # Projection
    "Projection",
    "ParamDistribution",
    # Predictive
    "PredictiveDistribution",
    # Main model
    "TemporalModel",
]

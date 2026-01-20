"""
High-level API facade for temporalpdf.

This module provides the simplest entry points for common use cases.
For advanced usage, import from submodules directly.

Quick Start:
    >>> import temporalpdf as tpdf
    >>>
    >>> # Discover best distribution for your data
    >>> result = tpdf.discover(returns)
    >>> print(f"Best: {result.best} (confidence: {result.confidence})")
    >>>
    >>> # Fit a distribution
    >>> params = tpdf.fit(returns, distribution='nig')
    >>>
    >>> # Build a temporal model
    >>> model = tpdf.temporal_model(returns, distribution='nig')
    >>> decision = model.decision(t=5, alpha=0.05)
    >>> print(decision)
    >>>
    >>> # Run a backtest
    >>> bt_result = tpdf.backtest(returns, distribution='nig')
    >>> print(bt_result.summary())
"""

from typing import Literal, Sequence
import numpy as np
from numpy.typing import NDArray

# Re-export key classes for convenience
from .discovery import discover, DiscoveryResult
from .utilities import fit, fit_nig, fit_student_t, fit_normal
from .temporal import (
    TemporalModel,
    ParameterTracker,
    SMA,
    EMA,
    Linear,
    PowerDecay,
    Gaussian,
    Custom,
    Constant,
    RandomWalk,
    MeanReverting,
    AR,
    GARCH,
)
from .backtest import Backtest, BacktestResult


def temporal_model(
    data: NDArray[np.float64],
    distribution: Literal["nig", "normal", "student_t"] = "nig",
    weighting: str = "ema",
    halflife: float = 20,
    window: int = 60,
    track_params: bool = True,
) -> TemporalModel:
    """
    Create and fit a temporal model with sensible defaults.

    This is the recommended entry point for most users.

    Args:
        data: Array of returns
        distribution: Which distribution to use
        weighting: 'ema', 'sma', or 'linear'
        halflife: Halflife for EMA weighting
        window: Window size for SMA/tracking
        track_params: Whether to track parameters over time

    Returns:
        Fitted TemporalModel ready for prediction

    Example:
        >>> model = tpdf.temporal_model(returns)
        >>> decision = model.decision(t=5)
        >>> print(f"VaR: {decision.var.value:.2%}")
    """
    # Set up weighting
    if weighting == "ema":
        weight_scheme = EMA(halflife=halflife)
    elif weighting == "sma":
        weight_scheme = SMA(window=window)
    elif weighting == "linear":
        weight_scheme = Linear(window=window)
    else:
        raise ValueError(f"Unknown weighting: {weighting}")

    # Set up tracking
    tracker = None
    if track_params:
        tracker = ParameterTracker(
            distribution=distribution,
            window=window,
            step=1,
        )

    model = TemporalModel(
        distribution=distribution,
        tracking=tracker,
        weighting=weight_scheme,
    )

    model.fit(data)
    return model


def backtest(
    data: NDArray[np.float64],
    distribution: Literal["nig", "normal", "student_t"] = "nig",
    lookback: int = 252,
    alpha: float = 0.05,
    **kwargs,
) -> BacktestResult:
    """
    Run a VaR backtest on historical data.

    Args:
        data: Array of returns
        distribution: Which distribution to use
        lookback: Lookback window for fitting
        alpha: VaR confidence level (0.05 = 95% VaR)
        **kwargs: Additional arguments to Backtest

    Returns:
        BacktestResult with statistics and test results

    Example:
        >>> result = tpdf.backtest(returns, distribution='nig')
        >>> print(result.summary())
        >>> if result.status == 'PASS':
        ...     print("Model is well-calibrated")
    """
    bt = Backtest(
        distribution=distribution,
        lookback=lookback,
        alpha=alpha,
        **kwargs,
    )
    return bt.run(data)


def compare_distributions(
    data: NDArray[np.float64],
    candidates: Sequence[str] = ("normal", "student_t", "nig"),
    n_folds: int = 5,
) -> dict:
    """
    Compare multiple distributions via cross-validation.

    Args:
        data: Array of returns
        candidates: List of distribution names to compare
        n_folds: Number of cross-validation folds

    Returns:
        Dictionary with comparison results

    Example:
        >>> result = tpdf.compare_distributions(returns)
        >>> best = min(result['scores'], key=result['scores'].get)
        >>> print(f"Best distribution: {best}")
    """
    result = discover(
        data,
        candidates=candidates,
        cv_folds=n_folds,
    )

    return {
        "best": result.best,
        "confidence": result.confidence,
        "scores": result.scores,
        "std_scores": result.std_scores,
    }


__all__ = [
    # Main functions
    "discover",
    "fit",
    "temporal_model",
    "backtest",
    "compare_distributions",
    # Fitting
    "fit_nig",
    "fit_student_t",
    "fit_normal",
    # Result types
    "DiscoveryResult",
    "BacktestResult",
    # Model classes
    "TemporalModel",
    "ParameterTracker",
    "Backtest",
    # Weighting
    "SMA",
    "EMA",
    "Linear",
    "PowerDecay",
    "Gaussian",
    "Custom",
    # Dynamics
    "Constant",
    "RandomWalk",
    "MeanReverting",
    "AR",
    "GARCH",
]

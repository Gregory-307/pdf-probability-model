"""
Backtesting module for evaluating temporal models.

Provides:
- Rolling backtest framework
- Statistical tests (Kupiec, Christoffersen)
- Model comparison utilities
"""

from .runner import Backtest, BacktestResult
from .tests import kupiec_test, christoffersen_test

__all__ = [
    "Backtest",
    "BacktestResult",
    "kupiec_test",
    "christoffersen_test",
]

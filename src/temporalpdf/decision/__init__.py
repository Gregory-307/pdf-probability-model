"""Decision utilities for risk management and position sizing.

References:
    Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
    Value-at-Risk. Journal of Risk, 2, 21-41.

    Kelly, J.L. (1956). A New Interpretation of Information Rate.
    Bell System Technical Journal, 35(4), 917-926.
"""

from .risk import VaR, CVaR, var, cvar
from .kelly import KellyCriterion, kelly_fraction, fractional_kelly
from .probability import prob_greater_than, prob_less_than, prob_between

__all__ = [
    "VaR",
    "CVaR",
    "var",
    "cvar",
    "KellyCriterion",
    "kelly_fraction",
    "fractional_kelly",
    "prob_greater_than",
    "prob_less_than",
    "prob_between",
]

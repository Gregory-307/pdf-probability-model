"""Proper scoring rules for distributional predictions.

References:
    Gneiting, T. & Raftery, A.E. (2007). Strictly Proper Scoring Rules,
    Prediction, and Estimation. JASA, 102(477), 359-378.
"""

from .rules import CRPS, LogScore, crps, crps_mc, log_score, crps_normal

__all__ = [
    "CRPS",
    "LogScore",
    "crps",
    "crps_mc",  # Monte Carlo alternative
    "log_score",
    "crps_normal",
]

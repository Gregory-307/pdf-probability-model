"""
Discovery module for automatic distribution selection.

Provides tools for:
- Comparing distributions via proper scoring rules
- Statistical significance testing
- Automatic best distribution selection
"""

from .selection import discover, DiscoveryResult
from .significance import paired_t_test, determine_confidence
from .scoring import crps_from_samples, log_score

__all__ = [
    "discover",
    "DiscoveryResult",
    "paired_t_test",
    "determine_confidence",
    "crps_from_samples",
    "log_score",
]

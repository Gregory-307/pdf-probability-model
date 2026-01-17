"""Distribution implementations for temporalpdf.

Includes both custom distributions and well-cited financial distributions:

- Normal: Baseline, efficient computation
- Student-t: Heavy tails, outlier robustness
- Skew-Normal: Light tails with asymmetry
- Generalized Laplace: Custom skewed Laplace-like distribution
- NIG (Normal Inverse Gaussian): Semi-heavy tails, skew, widely used in finance

References (NIG):
    Barndorff-Nielsen, O.E. (1997). Normal Inverse Gaussian Distributions
    and Stochastic Volatility Modelling. Scandinavian Journal of Statistics.
"""

from .generalized_laplace import GeneralizedLaplaceDistribution
from .normal import NormalDistribution
from .student_t import StudentTDistribution
from .skew_normal import SkewNormalDistribution
from .nig import NIGDistribution, NIGParameters
from .registry import DistributionRegistry

__all__ = [
    "GeneralizedLaplaceDistribution",
    "NormalDistribution",
    "StudentTDistribution",
    "SkewNormalDistribution",
    "NIGDistribution",
    "NIGParameters",
    "DistributionRegistry",
]

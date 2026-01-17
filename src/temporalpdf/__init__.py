"""
temporalpdf: Time-Evolving Probability Distribution Modeling Library

A Python library for distributional regression and probabilistic forecasting
with time-evolving uncertainty. Predict distribution parameters (not just
point estimates) and make risk-aware decisions.

Key Features:
- Multiple distribution types including NIG (Normal Inverse Gaussian)
- Time-evolving parameters (mean drift, volatility growth)
- Sophisticated volatility models (GARCH, mean-reverting, term structure)
- Proper scoring rules (CRPS, Log Score) for evaluation
- Decision utilities (VaR, CVaR, Kelly criterion)
- Comprehensive visualization (3D surfaces, heatmaps)

Basic Example:
    >>> import temporalpdf as tpdf
    >>>
    >>> # Create a NIG distribution (widely used in finance)
    >>> dist = tpdf.NIG()
    >>> params = tpdf.NIGParameters(
    ...     mu=0.001,      # Location
    ...     delta=0.02,    # Scale
    ...     alpha=15.0,    # Tail heaviness
    ...     beta=-2.0,     # Skewness
    ... )
    >>>
    >>> # Risk measures
    >>> print(f"VaR 95%: {tpdf.var(dist, params, alpha=0.05):.2%}")
    >>> print(f"CVaR 95%: {tpdf.cvar(dist, params, alpha=0.05):.2%}")
    >>> print(f"Kelly fraction: {tpdf.kelly_fraction(dist, params):.1%}")
    >>>
    >>> # Evaluate proper scoring rule
    >>> score = tpdf.crps(dist, params, y=actual_return)

Volatility Models Example:
    >>> # Mean-reverting volatility (exponential decay to long-run)
    >>> params = tpdf.NIGParameters(
    ...     mu=0.0, delta=0.04, alpha=15.0, beta=-2.0,  # Elevated vol
    ...     volatility_model=tpdf.mean_reverting(sigma_long=0.02, kappa=0.1)
    ... )
    >>>
    >>> # GARCH(1,1) volatility forecast
    >>> params = tpdf.NIGParameters(
    ...     mu=0.0, delta=0.03, alpha=15.0, beta=-2.0,
    ...     volatility_model=tpdf.garch_forecast(omega=1e-5, alpha=0.1, beta=0.85)
    ... )

References:
    Barndorff-Nielsen, O.E. (1997). Normal Inverse Gaussian Distributions.
    Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.
    Gneiting, T. & Raftery, A.E. (2007). Strictly Proper Scoring Rules.
    Kelly, J.L. (1956). A New Interpretation of Information Rate.
"""

__version__ = "0.1.0"
__author__ = "Greg Butcher"
__email__ = "gregbutcher307@gmail.com"

# Core abstractions
from .core.distribution import TimeEvolvingDistribution, DistributionParameters
from .core.parameters import (
    GeneralizedLaplaceParameters,
    NormalParameters,
    StudentTParameters,
    SkewNormalParameters,
)
from .core.grid import EvaluationGrid
from .core.result import PDFResult, ValidationResult
from .core.volatility import (
    VolatilityModel,
    LinearGrowth,
    ExponentialDecay,
    SquareRootDiffusion,
    GARCHForecast,
    TermStructure,
    constant_volatility,
    linear_growth,
    mean_reverting,
    garch_forecast,
)

# Distribution implementations (with short aliases)
from .distributions.generalized_laplace import GeneralizedLaplaceDistribution
from .distributions.normal import NormalDistribution
from .distributions.student_t import StudentTDistribution
from .distributions.skew_normal import SkewNormalDistribution
from .distributions.nig import NIGDistribution, NIGParameters
from .distributions.registry import DistributionRegistry

# Short aliases for distributions
GeneralizedLaplace = GeneralizedLaplaceDistribution
Normal = NormalDistribution
StudentT = StudentTDistribution
SkewNormal = SkewNormalDistribution
NIG = NIGDistribution

# Scoring rules
from .scoring import CRPS, LogScore, crps, log_score, crps_normal

# Decision utilities
from .decision import (
    VaR,
    CVaR,
    var,
    cvar,
    KellyCriterion,
    kelly_fraction,
    fractional_kelly,
    prob_greater_than,
    prob_less_than,
    prob_between,
)

# Coefficient extraction
from .coefficients.config import ExtractionConfig
from .coefficients.rolling import RollingCoefficientExtractor
from .coefficients.functions import (
    calculate_mean,
    calculate_volatility,
    calculate_skewness,
    calculate_mean_rate,
    calculate_volatility_growth,
)

# Visualization
from .visualization.plotter import PDFPlotter
from .visualization.interactive import InteractivePlotter
from .visualization.styles import (
    PlotStyle,
    DEFAULT_STYLE,
    PUBLICATION_STYLE,
    PRESENTATION_STYLE,
    DARK_STYLE,
)

# Validation
from .validation.validator import Validator, CrossValidator
from .validation.metrics import log_likelihood, mae, mse, r_squared, rmse

# Analysis
from .analysis.decomposition import (
    decompose_stl,
    decompose_stl_with_seasonality,
    decompose_fourier,
    decompose_wavelet,
    decompose_moving_average,
    decompose_exponential_smoothing,
    get_dominant_frequencies,
)


def evaluate(
    distribution: str | TimeEvolvingDistribution,  # type: ignore[type-arg]
    params: DistributionParameters,
    value_range: tuple[float, float] = (-0.2, 0.2),
    time_range: tuple[float, float] = (0.0, 60.0),
    value_points: int = 200,
    time_points: int = 100,
) -> PDFResult:
    """
    Convenience function for quick PDF evaluation.

    Creates a grid, evaluates the distribution, and returns a PDFResult.

    Args:
        distribution: Distribution name (str) or instance
        params: Distribution parameters
        value_range: (min, max) for value grid
        time_range: (min, max) for time grid
        value_points: Number of points in value grid
        time_points: Number of points in time grid

    Returns:
        PDFResult with evaluated PDF matrix

    Example:
        >>> result = tpdf.evaluate(
        ...     "normal",
        ...     tpdf.NormalParameters(mu_0=0, sigma_0=1, delta=0.01, beta=0.02),
        ...     time_range=(0, 30)
        ... )
    """
    if isinstance(distribution, str):
        dist = DistributionRegistry.create(distribution)
    else:
        dist = distribution

    grid = EvaluationGrid.from_ranges(
        value_range=value_range,
        time_range=time_range,
        value_points=value_points,
        time_points=time_points,
    )

    pdf_matrix = dist.pdf_matrix(grid.value_grid, grid.time_grid, params)

    return PDFResult(
        pdf_matrix=pdf_matrix,
        value_grid=grid.value_grid,
        time_grid=grid.time_grid,
        distribution_name=dist.name,
        parameters=params.__dict__,
    )


__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core abstractions
    "TimeEvolvingDistribution",
    "DistributionParameters",
    "EvaluationGrid",
    "PDFResult",
    "ValidationResult",
    # Volatility models
    "VolatilityModel",
    "LinearGrowth",
    "ExponentialDecay",
    "SquareRootDiffusion",
    "GARCHForecast",
    "TermStructure",
    "constant_volatility",
    "linear_growth",
    "mean_reverting",
    "garch_forecast",
    # Parameter classes
    "GeneralizedLaplaceParameters",
    "NormalParameters",
    "StudentTParameters",
    "SkewNormalParameters",
    "NIGParameters",
    # Distributions (full names)
    "GeneralizedLaplaceDistribution",
    "NormalDistribution",
    "StudentTDistribution",
    "SkewNormalDistribution",
    "NIGDistribution",
    # Distributions (short aliases)
    "GeneralizedLaplace",
    "Normal",
    "StudentT",
    "SkewNormal",
    "NIG",
    # Registry
    "DistributionRegistry",
    # Scoring rules
    "CRPS",
    "LogScore",
    "crps",
    "log_score",
    "crps_normal",
    # Decision utilities
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
    # Coefficient extraction
    "ExtractionConfig",
    "RollingCoefficientExtractor",
    "calculate_mean",
    "calculate_volatility",
    "calculate_skewness",
    "calculate_mean_rate",
    "calculate_volatility_growth",
    # Visualization
    "PDFPlotter",
    "InteractivePlotter",
    "PlotStyle",
    "DEFAULT_STYLE",
    "PUBLICATION_STYLE",
    "PRESENTATION_STYLE",
    "DARK_STYLE",
    # Validation
    "Validator",
    "CrossValidator",
    "log_likelihood",
    "mae",
    "mse",
    "r_squared",
    "rmse",
    # Analysis
    "decompose_stl",
    "decompose_stl_with_seasonality",
    "decompose_fourier",
    "decompose_wavelet",
    "decompose_moving_average",
    "decompose_exponential_smoothing",
    "get_dominant_frequencies",
    # Convenience functions
    "evaluate",
]

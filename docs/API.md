# temporalpdf API Reference

Complete API documentation for the temporalpdf library.

## Table of Contents

- [Core Module](#core-module)
  - [TimeEvolvingDistribution](#timeevolvingdistribution)
  - [DistributionParameters](#distributionparameters)
  - [EvaluationGrid](#evaluationgrid)
  - [PDFResult](#pdfresult)
  - [ValidationResult](#validationresult)
- [Distributions](#distributions)
  - [NormalDistribution](#normaldistribution)
  - [StudentTDistribution](#studenttdistribution)
  - [SkewNormalDistribution](#skewnormaldistribution)
  - [GeneralizedLaplaceDistribution](#generalizedlaplacedistribution)
  - [DistributionRegistry](#distributionregistry)
- [Coefficient Extraction](#coefficient-extraction)
  - [ExtractionConfig](#extractionconfig)
  - [RollingCoefficientExtractor](#rollingcoefficientextractor)
- [Visualization](#visualization)
  - [PDFPlotter](#pdfplotter)
  - [PlotStyle](#plotstyle)
- [Validation](#validation)
  - [Validator](#validator)
  - [CrossValidator](#crossvalidator)
  - [Metrics Functions](#metrics-functions)
- [Analysis](#analysis)
  - [Decomposition Functions](#decomposition-functions)
- [Convenience Functions](#convenience-functions)
  - [evaluate()](#evaluate)

---

## Core Module

### TimeEvolvingDistribution

Abstract base class for all time-evolving distributions.

```python
from temporalpdf import TimeEvolvingDistribution

class TimeEvolvingDistribution(ABC, Generic[P]):
    """Base class for distributions where parameters evolve over time."""
```

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `name` | `str` | Human-readable distribution name |
| `parameter_names` | `tuple[str, ...]` | Names of distribution parameters |

#### Methods

**`pdf(x, t, params) -> NDArray`**

Evaluate the PDF at values `x` for time `t`.

```python
pdf_values = dist.pdf(x_array, t=30.0, params=my_params)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `NDArray[float64]` | Array of values to evaluate |
| `t` | `float` | Time point |
| `params` | `DistributionParameters` | Distribution parameters |
| **Returns** | `NDArray[float64]` | Probability density values |

---

**`pdf_matrix(x, time_grid, params) -> NDArray`**

Evaluate PDF over a 2D grid of (time, value).

```python
matrix = dist.pdf_matrix(x_array, time_array, params)
# Shape: (len(time_array), len(x_array))
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `NDArray[float64]` | Value grid |
| `time_grid` | `NDArray[float64]` | Time grid |
| `params` | `DistributionParameters` | Distribution parameters |
| **Returns** | `NDArray[float64]` | 2D array, shape `(len(time_grid), len(x))` |

---

**`expected_value(x, t, params) -> float`**

Calculate E[X] at time t via numerical integration.

```python
ex = dist.expected_value(x_array, t=30.0, params=my_params)
```

---

**`variance(x, t, params) -> float`**

Calculate Var[X] at time t.

---

**`std(x, t, params) -> float`**

Calculate standard deviation at time t.

---

### DistributionParameters

Base class for all parameter dataclasses.

```python
from temporalpdf import DistributionParameters

@dataclass(frozen=True)
class DistributionParameters:
    """Base class for immutable distribution parameters."""
    pass
```

All parameter classes are **frozen dataclasses** (immutable after creation).

---

### EvaluationGrid

Defines the (value, time) domain for PDF evaluation.

```python
from temporalpdf import EvaluationGrid

grid = EvaluationGrid.from_ranges(
    value_range=(-0.3, 0.3),
    time_range=(0, 60),
    value_points=200,
    time_points=100
)
```

#### Factory Method

**`from_ranges(value_range, time_range, value_points, time_points) -> EvaluationGrid`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value_range` | `tuple[float, float]` | required | Min/max values |
| `time_range` | `tuple[float, float]` | required | Min/max time |
| `value_points` | `int` | `200` | Grid resolution for values |
| `time_points` | `int` | `100` | Grid resolution for time |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `value_grid` | `NDArray[float64]` | 1D array of value points |
| `time_grid` | `NDArray[float64]` | 1D array of time points |
| `value_range` | `tuple[float, float]` | Value bounds |
| `time_range` | `tuple[float, float]` | Time bounds |

---

### PDFResult

Container for evaluated PDF with computed properties.

```python
from temporalpdf import PDFResult

# Usually created by evaluate() or distribution.pdf_matrix()
result = tpdf.evaluate(dist, params)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `pdf_matrix` | `NDArray[float64]` | 2D PDF values, shape `(time, value)` |
| `value_grid` | `NDArray[float64]` | Value axis |
| `time_grid` | `NDArray[float64]` | Time axis |
| `distribution_name` | `str` | Name of distribution used |
| `parameters` | `dict` | Parameters used for evaluation |

#### Computed Properties

| Property | Type | Description |
|----------|------|-------------|
| `expected_values` | `NDArray[float64]` | E[X] at each time point |
| `cumulative_expected_value` | `float` | Integral of E[X] over time |
| `peak_density` | `float` | Maximum PDF value |

#### Methods

**`slice_at_time(t) -> tuple[NDArray, NDArray]`**

Get PDF cross-section at specific time.

```python
x_values, pdf_values = result.slice_at_time(t=30.0)
```

---

**`expected_value_at_time(t) -> float`**

Get E[X] at specific time point.

```python
ex = result.expected_value_at_time(t=30.0)
```

---

### ValidationResult

Container for model validation metrics.

```python
from temporalpdf import ValidationResult
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `log_likelihood` | `float` | Average log-likelihood |
| `mae` | `float` | Mean Absolute Error |
| `mse` | `float` | Mean Squared Error |
| `r_squared` | `float` | Coefficient of determination |
| `total_samples` | `int` | Number of samples validated |
| `per_sample_metrics` | `NDArray` | Detailed per-sample metrics |

#### Methods

**`summary() -> str`**

Returns formatted summary string.

```python
print(result.summary())
# Log-Likelihood: -2.34
# MAE: 0.0123
# MSE: 0.000456
# R²: 0.87
```

---

## Distributions

All distributions follow the time-evolving parameter pattern:

- **μ(t) = μ₀ + δ·t** (mean drift)
- **σ(t) = σ₀ · (1 + β·t)** (volatility growth)

### NormalDistribution

Standard Gaussian with time-evolving parameters.

```python
from temporalpdf import Normal, NormalParameters

dist = Normal()
params = NormalParameters(
    mu_0=0.0,      # Initial mean
    sigma_0=0.05,  # Initial std dev (must be > 0)
    delta=0.001,   # Mean drift rate
    beta=0.01      # Volatility growth rate
)
```

**Mathematical Form:**
$$f(x,t) = \frac{1}{\sigma(t)\sqrt{2\pi}} \exp\left(-\frac{(x-\mu(t))^2}{2\sigma(t)^2}\right)$$

**Use Cases:** Symmetric data, central limit theorem applications, baseline comparisons

---

### StudentTDistribution

Heavy-tailed distribution for data with outliers.

```python
from temporalpdf import StudentT, StudentTParameters

dist = StudentT()
params = StudentTParameters(
    mu_0=0.0,
    sigma_0=0.05,
    nu=5.0,        # Degrees of freedom (must be > 0)
    delta=0.001,
    beta=0.01
)
```

**Key Parameter:**
- `nu` (degrees of freedom): Lower values = heavier tails. `nu > 30` ≈ Normal distribution.

**Use Cases:** Financial returns, data with outliers, fat-tailed phenomena

---

### SkewNormalDistribution

Asymmetric extension of Normal distribution.

```python
from temporalpdf import SkewNormal, SkewNormalParameters

dist = SkewNormal()
params = SkewNormalParameters(
    mu_0=0.0,
    sigma_0=0.05,
    alpha=2.0,     # Skewness parameter
    delta=0.001,
    beta=0.01
)
```

**Key Parameter:**
- `alpha`: Skewness. `α = 0` gives Normal. `α > 0` skews right, `α < 0` skews left.

**Use Cases:** Asymmetric distributions, modeling upside/downside bias

---

### GeneralizedLaplaceDistribution

Fully flexible distribution with skew, tail control, and time decay.

```python
from temporalpdf import GeneralizedLaplace, GeneralizedLaplaceParameters

dist = GeneralizedLaplace()
params = GeneralizedLaplaceParameters(
    mu_0=0.0,
    sigma_0=0.05,
    alpha=0.5,         # Skewness
    delta=0.002,
    beta=0.01,
    k=1.0,             # Tail sharpness (0=Gaussian-like, 1=Laplace-like)
    lambda_decay=0.05  # Confidence decay over time
)
```

**Key Parameters:**
- `k`: Tail sharpness. `k = 0` → Gaussian-like tails. `k = 1` → Laplace-like (sharper) tails.
- `lambda_decay`: Exponential decay of PDF magnitude over time.

**Use Cases:** Maximum flexibility, financial modeling, custom tail behavior

---

### DistributionRegistry

Factory for creating distributions by name.

```python
from temporalpdf import DistributionRegistry

# Create by string name
dist = DistributionRegistry.create("normal")
dist = DistributionRegistry.create("student_t")
dist = DistributionRegistry.create("skew_normal")
dist = DistributionRegistry.create("generalized_laplace")

# List available distributions
names = DistributionRegistry.available()
# ['normal', 'student_t', 'skew_normal', 'generalized_laplace']

# Register custom distribution
DistributionRegistry.register("my_dist", MyDistribution)
```

---

## Coefficient Extraction

### ExtractionConfig

Data-agnostic configuration for coefficient extraction.

```python
from temporalpdf import ExtractionConfig

config = ExtractionConfig(
    value_column="close",        # Column with values to analyze
    time_column="date",          # Optional: timestamp column
    group_column="ticker",       # Optional: grouping column
    horizon=60,                  # Rolling window size
    volatility_window=7,         # Inner window for volatility
    pct_change_multiplier=100.0, # Multiplier (100 for %, 1 for fractions)
    dropna=True                  # Drop NaN rows after calculation
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `value_column` | `str` | required | Column containing values |
| `time_column` | `str \| None` | `None` | Timestamp column |
| `group_column` | `str \| None` | `None` | Grouping column |
| `horizon` | `int` | `60` | Rolling window size |
| `volatility_window` | `int` | `7` | Inner volatility window |
| `pct_change_multiplier` | `float` | `100.0` | Percent change multiplier |
| `dropna` | `bool` | `True` | Drop NaN values |

**Validation:**
- `horizon` must be ≥ 2
- `volatility_window` must be ≥ 2 and < `horizon`

---

### RollingCoefficientExtractor

Extract distribution coefficients from time series data.

```python
from temporalpdf import RollingCoefficientExtractor, ExtractionConfig

extractor = RollingCoefficientExtractor()
config = ExtractionConfig(value_column="close", horizon=60)

# Extract coefficients (adds columns to dataframe)
df_with_coeffs = extractor.extract(df, config)
```

#### Methods

**`extract(data, config) -> DataFrame`**

Extract coefficients and add as new columns.

| Added Column | Description |
|--------------|-------------|
| `mean` | Rolling mean of percent changes |
| `volatility` | Rolling standard deviation |
| `skewness` | Rolling skewness |
| `mean_rate` | Rate of change of mean (delta) |
| `volatility_growth` | Rate of volatility growth (beta) |

---

**`to_parameters(row, distribution_type) -> DistributionParameters`**

Convert a DataFrame row to distribution parameters.

```python
# Get latest coefficients as parameters
params = extractor.to_parameters(
    df_with_coeffs.iloc[-1],
    distribution_type="normal"
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `row` | `Series` | DataFrame row with coefficient columns |
| `distribution_type` | `str` | One of: `"normal"`, `"student_t"`, `"skew_normal"`, `"generalized_laplace"` |

---

## Visualization

### PDFPlotter

High-level visualization interface.

```python
from temporalpdf import PDFPlotter, PlotStyle, PUBLICATION_STYLE

plotter = PDFPlotter(style=PUBLICATION_STYLE)
```

#### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `style` | `PlotStyle` | `DEFAULT_STYLE` | Styling configuration |

#### Methods

**`surface_3d(result, **kwargs) -> Figure`**

Create 3D surface plot of PDF evolution.

```python
fig = plotter.surface_3d(
    result,
    title="My Distribution",
    xlabel="Time",
    ylabel="Value",
    zlabel="Density",
    figsize=(12, 8),
    elevation=30,
    azimuth=-60,
    show_cumulative_ex=True
)
```

---

**`heatmap(result, **kwargs) -> Figure`**

Create 2D heatmap visualization.

```python
fig = plotter.heatmap(
    result,
    title="PDF Heatmap",
    figsize=(10, 6),
    colorbar_label="Probability Density"
)
```

---

**`slice_at_time(result, t, **kwargs) -> Figure`**

Plot PDF cross-section at specific time.

```python
fig = plotter.slice_at_time(
    result,
    t=30.0,
    observed_value=0.05,   # Mark observed value
    show_stats=True        # Show E[X] line
)
```

---

**`compare_distributions(results, time_point, **kwargs) -> Figure`**

Overlay multiple distributions at one time point.

```python
fig = plotter.compare_distributions(
    [result1, result2, result3],
    time_point=30.0,
    labels=["Normal", "Student-t", "Skew-Normal"]
)
```

---

**`multi_scenario(results, **kwargs) -> Figure`**

Create multi-panel 3D comparison grid.

```python
fig = plotter.multi_scenario(
    [result1, result2, result3, result4],
    layout=(2, 2),
    figsize=(16, 12)
)
```

---

**`confidence_bands(result, **kwargs) -> Figure`**

Show uncertainty growth over time with confidence intervals.

```python
fig = plotter.confidence_bands(
    result,
    confidence_levels=(0.5, 0.9, 0.95)  # 50%, 90%, 95% CI
)
```

---

**`expected_value_over_time(result, **kwargs) -> Figure`**

Plot E[X] trajectory over time.

```python
fig = plotter.expected_value_over_time(
    result,
    show_cumulative=True
)
```

---

**`save(fig, path, dpi=300, transparent=False)`**

Save figure to file.

```python
plotter.save(fig, "output.png", dpi=300)
```

---

**`show(fig)`**

Display figure interactively.

---

**`close(fig)`**

Close figure to free memory.

---

### PlotStyle

Configuration dataclass for plot styling.

```python
from temporalpdf import PlotStyle, DEFAULT_STYLE, PUBLICATION_STYLE

# Built-in styles
DEFAULT_STYLE       # Standard matplotlib defaults
PUBLICATION_STYLE   # Clean, print-ready
PRESENTATION_STYLE  # Large fonts, bold colors
DARK_STYLE          # Dark background theme

# Custom style
my_style = PlotStyle(
    font_size=12,
    title_size=14,
    cmap="viridis",
    primary_color="#1f77b4",
    grid_alpha=0.3,
    figure_facecolor="white"
)
```

---

## Validation

### Validator

Validate distribution models against observed data.

```python
from temporalpdf import Validator, Normal, NormalParameters
import numpy as np

dist = Normal()
value_grid = np.linspace(-0.3, 0.3, 200)
validator = Validator(dist, value_grid)

result = validator.validate(
    data=df,
    params=my_params,
    observed_column="pct_change",
    time_column="day_index"
)

print(result.summary())
```

#### Methods

**`validate(data, params, observed_column, **kwargs) -> ValidationResult`**

Validate against DataFrame.

| Parameter | Type | Description |
|-----------|------|-------------|
| `data` | `DataFrame` | Data with observed values |
| `params` | `DistributionParameters` | Parameters to validate |
| `observed_column` | `str` | Column with observed values |
| `time_column` | `str \| None` | Column with time values |
| `sample_size` | `int \| None` | Optional sample size limit |

---

**`validate_arrays(observed, time_points, params) -> ValidationResult`**

Validate using raw numpy arrays.

---

**`validate_single(observed, t, params) -> dict`**

Validate single observation (for debugging).

```python
metrics = validator.validate_single(
    observed=0.05,
    t=30.0,
    params=my_params
)
# {'log_likelihood': -2.3, 'mae': 0.01, 'mse': 0.0001, ...}
```

---

### CrossValidator

K-fold cross-validation for robust model assessment.

```python
from temporalpdf import CrossValidator

cv = CrossValidator(dist, value_grid, n_folds=5)
results = cv.cross_validate(
    data=df,
    params=my_params,
    observed_column="pct_change"
)

summary = cv.summary(results)
print(f"MAE: {summary['mae_mean']:.4f} ± {summary['mae_std']:.4f}")
```

---

### Metrics Functions

Standalone metric calculation functions.

```python
from temporalpdf import log_likelihood, mae, mse, r_squared, rmse

ll = log_likelihood(observed_value, pdf_values, value_grid)
error = mae(predicted, observed)
squared_error = mse(predicted, observed)
r2 = r_squared(predicted_array, observed_array)
root_mse = rmse(predicted_array, observed_array)
```

---

## Analysis

### Decomposition Functions

Time series decomposition utilities.

```python
from temporalpdf import (
    decompose_stl,
    decompose_stl_with_seasonality,
    decompose_fourier,
    decompose_wavelet,
    decompose_moving_average,
    decompose_exponential_smoothing,
    get_dominant_frequencies
)
```

**`decompose_stl(series, period) -> tuple[NDArray, NDArray, NDArray]`**

STL (Seasonal-Trend decomposition using LOESS).

```python
trend, seasonal, residual = decompose_stl(series, period=7)
```

---

**`decompose_fourier(series, n_components) -> tuple[NDArray, NDArray]`**

Fourier decomposition into frequency components.

```python
frequencies, magnitudes = decompose_fourier(series, n_components=10)
```

---

**`decompose_wavelet(series, wavelet, level) -> list[NDArray]`**

Wavelet decomposition for multi-resolution analysis.

```python
coefficients = decompose_wavelet(series, wavelet="db4", level=3)
```

---

**`get_dominant_frequencies(series, n_top) -> list[tuple[float, float]]`**

Extract dominant frequency components.

```python
top_freqs = get_dominant_frequencies(series, n_top=5)
# [(freq1, magnitude1), (freq2, magnitude2), ...]
```

---

## Convenience Functions

### evaluate()

Quick PDF evaluation without manual grid setup.

```python
import temporalpdf as tpdf

# Using distribution instance
result = tpdf.evaluate(
    distribution=tpdf.Normal(),
    params=tpdf.NormalParameters(mu_0=0, sigma_0=0.05, delta=0.001, beta=0.01),
    value_range=(-0.3, 0.3),
    time_range=(0, 60),
    value_points=200,
    time_points=100
)

# Using string name
result = tpdf.evaluate(
    "student_t",
    tpdf.StudentTParameters(mu_0=0, sigma_0=0.05, nu=5, delta=0.001, beta=0.01)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distribution` | `str \| TimeEvolvingDistribution` | required | Distribution or name |
| `params` | `DistributionParameters` | required | Distribution parameters |
| `value_range` | `tuple[float, float]` | `(-0.2, 0.2)` | Value bounds |
| `time_range` | `tuple[float, float]` | `(0.0, 60.0)` | Time bounds |
| `value_points` | `int` | `200` | Value grid resolution |
| `time_points` | `int` | `100` | Time grid resolution |

---

## Type Hints

The library is fully typed (PEP 561 compliant). Type stubs are included for IDE support.

```python
from temporalpdf import (
    TimeEvolvingDistribution,
    DistributionParameters,
    PDFResult,
    ValidationResult,
)
from numpy.typing import NDArray
import numpy as np

def my_function(
    dist: TimeEvolvingDistribution[DistributionParameters],
    x: NDArray[np.float64],
) -> PDFResult:
    ...
```

---

## Examples

See the `examples/` directory for complete working examples:

- **`basic_usage.py`**: Core library functionality
- **`custom_distribution.py`**: Creating custom distributions
- **`financial_modeling.py`**: Full financial analysis workflow

See the `showcase/walkthrough.ipynb` notebook for an interactive tutorial.

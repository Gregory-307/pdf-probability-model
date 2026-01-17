# temporalpdf

A Python library for **distributional regression** and **probabilistic forecasting** with time-evolving uncertainty.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Overview

**temporalpdf** predicts **entire probability distributions**, not just point estimates. This enables:

- **Uncertainty quantification**: Know *how uncertain* your predictions are
- **Risk measures**: Compute VaR, CVaR (Expected Shortfall) directly from predictions
- **Position sizing**: Use Kelly criterion for optimal bet sizing
- **Proper evaluation**: Score forecasts with CRPS and log-likelihood

### Why Distributional Regression?

Traditional ML predicts single numbers. But a prediction of "+2% return" with high confidence is fundamentally different from "+2%" with massive uncertainty:

| Metric | Scenario A | Scenario B |
|--------|------------|------------|
| Predicted Return | +2% | +2% |
| True Distribution | N(2%, 1%) | N(2%, 20%) |
| P(loss > 5%) | 0.0001% | 36.3% |
| Optimal Kelly Fraction | 200% | 0.5% |

Same point prediction. Radically different investment decisions.

## Key Features

- **NIG Distribution**: Normal Inverse Gaussian - the gold standard for financial returns (Barndorff-Nielsen 1997)
- **Proper Scoring Rules**: CRPS, Log Score for evaluating distributional forecasts (Gneiting & Raftery 2007)
- **Decision Utilities**: VaR, CVaR, Kelly criterion for risk-aware decisions
- **Time Evolution**: Parameters that drift and grow over prediction horizons
- **Interactive Visualization**: Rotatable 3D plots with Plotly

## Installation

```bash
git clone https://github.com/Gregory-307/temporalpdf.git
cd temporalpdf
pip install -e .
```

## Quick Start

### Basic: Fit and Evaluate

```python
import temporalpdf as tpdf
import numpy as np

# Create a NIG distribution (widely used in finance)
nig = tpdf.NIG()
params = tpdf.NIGParameters(
    mu=0.001,      # Location (expected return)
    delta=0.02,    # Scale (volatility-like)
    alpha=15.0,    # Tail heaviness (higher = lighter tails)
    beta=-2.0,     # Skewness (negative = left-skewed)
)

# Risk measures
print(f"VaR 95%: {tpdf.var(nig, params, alpha=0.05):.2%}")
print(f"CVaR 95%: {tpdf.cvar(nig, params, alpha=0.05):.2%}")
print(f"Kelly fraction: {tpdf.kelly_fraction(nig, params):.1%}")

# Probability queries
print(f"P(return > 0): {tpdf.prob_greater_than(nig, params, 0):.1%}")
print(f"P(loss > 5%): {tpdf.prob_less_than(nig, params, -0.05):.1%}")

# Evaluate forecast quality with CRPS
actual_return = 0.015
score = tpdf.crps(nig, params, y=actual_return)
print(f"CRPS: {score:.4f}")  # Lower is better
```

### Time-Evolving Distributions

```python
import temporalpdf as tpdf

# Parameters that evolve over a 30-day horizon
params = tpdf.NIGParameters(
    mu=0.001,
    delta=0.02,
    alpha=15.0,
    beta=-2.0,
    mu_drift=0.0001,      # Location drifts up
    delta_growth=0.1,     # Scale grows 10% per time unit
)

# Evaluate over time grid
result = tpdf.evaluate(
    tpdf.NIG(),
    params,
    value_range=(-0.2, 0.2),
    time_range=(0, 30),
)

# Visualize
plotter = tpdf.PDFPlotter()
fig = plotter.surface_3d(result, title="30-Day Forecast Distribution")
plotter.save(fig, "forecast.png")
```

### Interactive 3D Plots

```python
import temporalpdf as tpdf

result = tpdf.evaluate(tpdf.NIG(), params, time_range=(0, 30))

# Create rotatable 3D plot
plotter = tpdf.InteractivePlotter()
fig = plotter.surface_3d(result)
plotter.save_html(fig, "interactive_forecast.html")
```

## Distributions

| Distribution | Parameters | Use Case |
|--------------|------------|----------|
| **NIG** (Normal Inverse Gaussian) | mu, delta, alpha, beta | Financial returns, semi-heavy tails, skew |
| **Normal** | mu_0, sigma_0, delta, beta | Baseline, fast computation |
| **Student's t** | + nu | Heavy tails, outlier robustness |
| **Skew-Normal** | + alpha | Light tails with asymmetry |
| **Generalized Laplace** | + k, lambda_decay | Flexible custom distribution |

### Why NIG?

The Normal Inverse Gaussian distribution is preferred for financial returns because:

1. **Captures stylized facts**: Semi-heavy tails, skewness, excess kurtosis
2. **Closed under convolution**: Daily → weekly returns analytically tractable
3. **Well-cited**: Barndorff-Nielsen (1997), standard in quantitative finance
4. **Interpretable parameters**: Each parameter has clear financial meaning

## Scoring Rules

Evaluate distributional forecasts with **proper scoring rules**:

```python
import temporalpdf as tpdf

# CRPS (Continuous Ranked Probability Score)
# Generalizes MAE to probabilistic forecasts
crps_score = tpdf.crps(dist, params, y=actual)

# Log Score (negative log-likelihood)
log_score = tpdf.log_score(dist, params, y=actual)

# Closed-form CRPS for Normal (faster)
crps_norm = tpdf.crps_normal(y=actual, mu=predicted_mean, sigma=predicted_std)
```

## Decision Utilities

Make risk-aware decisions directly from distributional predictions:

```python
import temporalpdf as tpdf

# Value at Risk: "5% chance of losing more than this"
var_95 = tpdf.var(dist, params, alpha=0.05)

# CVaR (Expected Shortfall): "Expected loss in worst 5%"
cvar_95 = tpdf.cvar(dist, params, alpha=0.05)

# Kelly criterion: Optimal position size
kelly = tpdf.kelly_fraction(dist, params)
half_kelly = tpdf.fractional_kelly(dist, params, fraction=0.5)

# Probability queries
p_profit = tpdf.prob_greater_than(dist, params, threshold=0)
p_big_loss = tpdf.prob_less_than(dist, params, threshold=-0.05)
```

## Examples

See `examples/` directory:

- `basic_usage.py`: Core functionality
- `real_data_example.py`: Real Nordic stock data analysis
- `custom_distribution.py`: Extending with custom distributions

## API Reference

### Core

- `TimeEvolvingDistribution`: Base class for distributions
- `NIGParameters`, `NormalParameters`, etc.: Parameter containers
- `evaluate()`: Quick PDF evaluation over grids

### Scoring

- `crps()`, `log_score()`: Proper scoring rules
- `crps_normal()`: Closed-form CRPS for Normal

### Decision

- `var()`, `cvar()`: Risk measures
- `kelly_fraction()`, `fractional_kelly()`: Position sizing
- `prob_greater_than()`, `prob_less_than()`, `prob_between()`: Probability queries

### Visualization

- `PDFPlotter`: Static matplotlib plots
- `InteractivePlotter`: Rotatable Plotly 3D plots

## References

- Barndorff-Nielsen, O.E. (1997). Normal Inverse Gaussian Distributions and Stochastic Volatility Modelling. *Scandinavian Journal of Statistics*.
- Gneiting, T. & Raftery, A.E. (2007). Strictly Proper Scoring Rules, Prediction, and Estimation. *JASA*.
- Kelly, J.L. (1956). A New Interpretation of Information Rate. *Bell System Technical Journal*.
- Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional Value-at-Risk. *Journal of Risk*.

## Development

```bash
pip install -e ".[dev]"
pytest                    # Run tests
mypy src/temporalpdf      # Type check
ruff check src/temporalpdf  # Lint
```

## License

MIT License - see [LICENSE](LICENSE) for details.

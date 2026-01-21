# temporalpdf API Reference

**Version:** 0.1.0
**Author:** Greg Butcher
**License:** MIT

A comprehensive API reference for developers working with temporalpdf - a Python library for distributional regression and probabilistic forecasting with time-evolving uncertainty.

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API by Pipeline Stage](#api-by-pipeline-stage)
   - [Stage 1: Data Preparation & Feature Extraction](#stage-1-data-preparation--feature-extraction)
   - [Stage 2: Distribution Fitting](#stage-2-distribution-fitting)
   - [Stage 3: Model Training](#stage-3-model-training)
   - [Stage 4: Temporal Dynamics](#stage-4-temporal-dynamics)
   - [Stage 5: Volatility Models](#stage-5-volatility-models)
   - [Stage 6: Risk Metrics & Decisions](#stage-6-risk-metrics--decisions)
   - [Stage 7: Barrier Probability](#stage-7-barrier-probability)
   - [Stage 8: Scoring Rules](#stage-8-scoring-rules)
   - [Stage 9: Calibration & Validation](#stage-9-calibration--validation)
   - [Stage 10: Visualization](#stage-10-visualization)
   - [Stage 11: Time Series Decomposition](#stage-11-time-series-decomposition)
   - [Stage 12: Coefficient Extraction](#stage-12-coefficient-extraction)
   - [Stage 13: PDF Evaluation](#stage-13-pdf-evaluation)
5. [Complete Pipeline Flow](#complete-pipeline-flow)
6. [Dependencies](#dependencies)

---

## Installation

```bash
pip install temporalpdf

# For ML features (DistributionalRegressor, BarrierModel)
pip install temporalpdf[ml]  # or: pip install torch
```

---

## Quick Start

```python
import temporalpdf as tpdf
import numpy as np

# Fit a distribution to data
returns = np.random.standard_t(5, size=500) * 0.02
params = tpdf.fit(returns, distribution="student_t")

# Risk metrics
print(f"VaR 95%: {tpdf.var(tpdf.StudentT(), params, alpha=0.05):.2%}")
print(f"CVaR 95%: {tpdf.cvar(tpdf.StudentT(), params, alpha=0.05):.2%}")

# Barrier probability
p = tpdf.barrier_prob_mc(params, horizon=20, barrier=0.05, distribution="student_t")
print(f"P(cumsum >= 5% in 20 days): {p:.1%}")
```

---

## Core Concepts

### Pipeline 1 vs Pipeline 2

**Pipeline 1 (Traditional):**
```
Data → Model → Point Prediction
```

**Pipeline 2 (temporalpdf):**
```
Data → Model → Distribution Parameters (μ, σ, ν) → Full Distribution → Decisions
```

Pipeline 2 predicts **distribution parameters**, giving you:
- Point prediction (mean/median)
- Uncertainty quantification (std, quantiles)
- Risk metrics (VaR, CVaR, Kelly)
- Barrier/threshold probabilities
- Full distributional shape

### Supported Distributions

| Distribution | Parameters | Use Case |
|--------------|------------|----------|
| Normal | μ, σ | Baseline, symmetric data |
| Student-t | μ, σ, ν | Fat tails, financial returns |
| NIG | μ, δ, α, β | Heavy tails + skewness |
| Skew-Normal | μ, σ, α | Moderate skewness |
| Generalized Laplace | μ, σ, α | Sharp peaks |

---

## API by Pipeline Stage

---

## Stage 1: Data Preparation & Feature Extraction

Functions for extracting features that predict distribution parameters.

### Feature Extraction Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `extract_calibration_features` | `(data, window=60, step=1) → ndarray` | Rolling window features → (n_windows, 12) matrix |
| `calibration_features` | `(data) → dict` | Single window → dict of 12 features |
| `get_feature_names` | `() → list[str]` | Returns list of 12 feature names |

### Individual Feature Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `hill_estimator` | `(data, k=10) → float` | Tail index estimate (predicts ν) |
| `jarque_bera_stat` | `(data) → float` | Normality test statistic |
| `realized_moments` | `(data) → dict` | Returns {mean, std, skewness, kurtosis} |
| `volatility_clustering` | `(data) → float` | Autocorrelation of \|returns\| |
| `garch_proxy` | `(data, short_window=5) → float` | Recent/overall volatility ratio |
| `vol_regime_indicator` | `(data, short_window=5) → float` | High/low vol regime indicator |
| `extreme_event_frequency` | `(data, threshold=2.0) → float` | Fraction beyond threshold σ |
| `tail_asymmetry` | `(data, threshold=2.0) → float` | Upper/lower tail ratio |
| `max_drawdown` | `(data) → float` | Maximum cumulative loss |

**Example:**
```python
features = tpdf.extract_calibration_features(returns, window=60)
# Shape: (n_samples - 60 + 1, 12)
```

---

## Stage 2: Distribution Fitting

Functions for fitting distributions to data.

### Fitting Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `fit` | `(data, distribution) → Parameters` | Fit any distribution |
| `fit_nig` | `(data) → NIGParameters` | Fit NIG distribution |
| `fit_student_t` | `(data) → StudentTParameters` | Fit Student-t distribution |
| `fit_normal` | `(data) → NormalParameters` | Fit Normal distribution |

### Selection Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `compare_distributions` | `(data, distributions) → dict` | Compare fits via AIC/BIC |
| `select_best_distribution` | `(data) → tuple` | Auto-select best distribution |
| `discover` | `(data) → DiscoveryResult` | Full automated discovery |

### Parameter Classes

```python
tpdf.NIGParameters(mu, delta, alpha, beta, volatility_model=None)
tpdf.StudentTParameters(mu_0, sigma_0, nu, delta=0, beta=0, volatility_model=None)
tpdf.NormalParameters(mu_0, sigma_0, delta=0, beta=0, volatility_model=None)
tpdf.SkewNormalParameters(mu_0, sigma_0, alpha, delta=0, beta=0)
tpdf.GeneralizedLaplaceParameters(mu_0, sigma_0, alpha, delta=0, beta=0)
```

### Distribution Classes

| Full Name | Short Alias |
|-----------|-------------|
| `NIGDistribution` | `NIG` |
| `StudentTDistribution` | `StudentT` |
| `NormalDistribution` | `Normal` |
| `SkewNormalDistribution` | `SkewNormal` |
| `GeneralizedLaplaceDistribution` | `GeneralizedLaplace` |

**Example:**
```python
# Fit and compare
params = tpdf.fit(returns, distribution="student_t")
comparison = tpdf.compare_distributions(returns, ["normal", "student_t", "nig"])
best_dist, best_params = tpdf.select_best_distribution(returns)
```

---

## Stage 3: Model Training

ML models for predicting distribution parameters from features.

### DistributionalRegressor

Trains on CRPS (distribution accuracy) - predicts distribution parameters.

```python
tpdf.DistributionalRegressor(
    distribution="student_t",  # "normal", "student_t", "nig"
    loss="crps",               # "crps" or "log_score"
    hidden_dims=[64, 32],      # MLP architecture
    learning_rate=1e-3,
    n_epochs=100,
    batch_size=32,
    n_samples=100,             # Samples for CRPS estimation
    device="cpu",              # "cpu" or "cuda"
    verbose=True,
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `.fit` | `(X, y) → self` | Train on features X and target values y |
| `.predict` | `(X) → ndarray` | Raw params array (n, n_params) |
| `.predict_distribution` | `(X) → list[Parameters]` | List of parameter objects |
| `.sample` | `(X, n_samples) → ndarray` | Generate samples (n, n_samples) |

### BarrierModel

Trains on Brier score (barrier probability accuracy) - for direct barrier prediction.

```python
tpdf.BarrierModel(
    n_features=12,
    hidden_dims=[64, 32],
    n_sims=64,                 # Doubled with antithetic variates
    learning_rate=1e-3,
    n_epochs=100,
    batch_size=32,
    device="cpu",
    verbose=True,
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `.fit` | `(X, barriers, horizons, y_hit) → self` | Train on barrier hit labels |
| `.predict` | `(X, barrier, horizon) → ndarray` | Barrier probabilities |
| `.predict_params` | `(X) → ndarray` | Underlying (μ, σ, ν) parameters |

**Example:**
```python
# DistributionalRegressor
model = tpdf.DistributionalRegressor(distribution="student_t", loss="crps")
model.fit(X_train, y_train)
params = model.predict_distribution(X_test)

# BarrierModel (84% better Brier score for barrier prediction)
barrier_model = tpdf.BarrierModel(n_features=12)
barrier_model.fit(X_train, barriers, horizons, y_hit)
probs = barrier_model.predict(X_test, barrier=0.05, horizon=20)
```

---

## Stage 4: Temporal Dynamics

Time-varying parameter models.

### TemporalModel

```python
tpdf.TemporalModel(
    distribution="student_t",
    tracking=tpdf.ParameterTracker("student_t", window=60),
    weighting=tpdf.EMA(span=20),
    dynamics={"sigma_0": tpdf.GARCH(1, 1)},
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `.fit` | `(data) → self` | Fit to historical data |
| `.project` | `(horizon, n_paths) → Projection` | Forward parameter projection |

### ParameterTracker

```python
tpdf.ParameterTracker(
    distribution="student_t",
    window=60,
    step=1,
    min_window=None,
)
```

### Dynamics Models

| Class | Constructor | Description |
|-------|-------------|-------------|
| `Constant` | `()` | No change over time |
| `RandomWalk` | `()` | Brownian motion |
| `MeanReverting` | `(kappa=0.1)` | Exponential decay to mean |
| `AR` | `(order=1)` | Autoregressive |
| `GARCH` | `(p=1, q=1)` | Volatility clustering |

### Weighting Schemes

| Class | Constructor | Description |
|-------|-------------|-------------|
| `SMA` | `(window)` | Simple moving average |
| `EMA` | `(span)` | Exponential moving average |
| `Linear` | `()` | Linear decay weights |
| `PowerDecay` | `(power)` | Power law decay |
| `Gaussian` | `(sigma)` | Gaussian kernel |
| `Custom` | `(weights)` | User-defined weights |

### High-Level Facade

```python
model = tpdf.temporal_model(data, distribution="student_t")
```

---

## Stage 5: Volatility Models

Models for time-evolving volatility in distribution parameters.

### Factory Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `constant_volatility` | `() → VolatilityModel` | Fixed σ over time |
| `linear_growth` | `(beta) → LinearGrowth` | σ(t) = σ₀ + β·t |
| `mean_reverting` | `(sigma_long, kappa) → ExponentialDecay` | σ(t) → σ_long exponentially |
| `garch_forecast` | `(omega, alpha, beta) → GARCHForecast` | GARCH(1,1) projection |

### Volatility Model Classes

- `VolatilityModel` (base class)
- `LinearGrowth`
- `ExponentialDecay`
- `SquareRootDiffusion`
- `GARCHForecast`
- `TermStructure`

**Example:**
```python
params = tpdf.StudentTParameters(
    mu_0=0.001, sigma_0=0.02, nu=5.0,
    volatility_model=tpdf.mean_reverting(sigma_long=0.015, kappa=0.1)
)
```

---

## Stage 6: Risk Metrics & Decisions

Functions for risk quantification and decision-making.

### Core Risk Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `var` | `(dist, params, alpha=0.05) → float` | Value at Risk |
| `cvar` | `(dist, params, alpha=0.05) → float` | Conditional VaR (Expected Shortfall) |
| `kelly_fraction` | `(dist, params) → float` | Optimal bet size (Kelly criterion) |
| `fractional_kelly` | `(dist, params, fraction=0.5) → float` | Conservative Kelly |

### Risk with Confidence Intervals

| Function | Signature | Description |
|----------|-----------|-------------|
| `var_with_ci` | `(dist, params, alpha, ci=0.95) → RiskMetric` | VaR with CI |
| `cvar_with_ci` | `(dist, params, alpha, ci=0.95) → RiskMetric` | CVaR with CI |
| `kelly_with_ci` | `(dist, params, ci=0.95) → RiskMetric` | Kelly with CI |

### Probability Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `prob_greater_than` | `(dist, params, x) → float` | P(X > x) |
| `prob_less_than` | `(dist, params, x) → float` | P(X < x) |
| `prob_between` | `(dist, params, a, b) → float` | P(a < X < b) |

### Classes

- `VaR`, `CVaR`, `KellyCriterion` — Callable classes
- `RiskMetric` — Result with value + CI
- `DecisionSummary` — Aggregated decision metrics

**Example:**
```python
dist = tpdf.StudentT()
var_95 = tpdf.var(dist, params, alpha=0.05)
cvar_95 = tpdf.cvar(dist, params, alpha=0.05)
kelly = tpdf.kelly_fraction(dist, params)
p_loss = tpdf.prob_less_than(dist, params, -0.02)
```

---

## Stage 7: Barrier Probability

Functions for computing P(max cumulative sum >= barrier over horizon).

### Analytical Methods

| Function | Signature | Description |
|----------|-----------|-------------|
| `barrier_prob_normal` | `(mu, sigma, horizon, barrier) → float` | Exact via reflection principle |
| `barrier_prob_student_t` | `(mu, sigma, nu, horizon, barrier) → float` | Fat-tail approximation |
| `barrier_prob_nig` | `(mu, delta, alpha, beta, horizon, barrier) → float` | NIG approximation |
| `barrier_prob_analytical_student_t` | `(mu, sigma, nu, horizon, barrier) → float` | Fast approximation (16x faster) |

### Simulation Methods

| Function | Signature | Description |
|----------|-----------|-------------|
| `barrier_prob_mc` | `(params, horizon, barrier, n_sims=10000, distribution) → float` | Monte Carlo |
| `barrier_prob_qmc` | `(params, horizon, barrier, n_sims=8192, distribution) → float` | Quasi-Monte Carlo (Sobol) |
| `barrier_prob_importance_sampling` | `(params, horizon, barrier, n_sims=10000) → float` | For rare events (Normal only) |

### Temporal Methods

| Function | Signature | Description |
|----------|-----------|-------------|
| `barrier_prob_temporal` | `(data, horizon, barrier, distribution, n_sims, window, dynamics) → float` | With time-varying params |
| `compare_static_vs_temporal` | `(data, horizon, barrier, distribution, n_sims) → dict` | Compare static vs dynamic |

**Example:**
```python
# Analytical (fast)
p = tpdf.barrier_prob_normal(mu=0.001, sigma=0.02, horizon=20, barrier=0.05)

# Monte Carlo (flexible)
p = tpdf.barrier_prob_mc(params, horizon=20, barrier=0.05, n_sims=10000, distribution="student_t")

# With GARCH dynamics
p = tpdf.barrier_prob_temporal(returns, horizon=20, barrier=0.05, distribution="student_t")

# Compare approaches
comp = tpdf.compare_static_vs_temporal(returns, horizon=20, barrier=0.05)
print(f"Static: {comp['static']:.1%}, Temporal: {comp['temporal']:.1%}")
```

---

## Stage 8: Scoring Rules

Proper scoring rules for evaluating probabilistic predictions.

### Scoring Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `crps` | `(dist, params, y) → float` | Continuous Ranked Probability Score |
| `log_score` | `(dist, params, y) → float` | Negative log likelihood |
| `crps_normal` | `(mu, sigma, y) → float` | Closed-form CRPS for Normal |

### Scoring Classes

- `CRPS` — Callable CRPS scorer
- `LogScore` — Callable log score scorer

**Example:**
```python
# Evaluate predictions
score = tpdf.crps(tpdf.StudentT(), params, actual_return)
```

---

## Stage 9: Calibration & Validation

Functions for model calibration and validation.

### ConformalPredictor

Distribution-free calibrated prediction intervals.

```python
tpdf.ConformalPredictor(
    predictor,          # Any object with .predict(X) method
    X_cal,              # Calibration features
    y_cal,              # Calibration targets
    distribution="student_t",
)
```

| Method | Signature | Description |
|--------|-----------|-------------|
| `.predict_interval` | `(X, alpha=0.1) → (lower, upper)` | Calibrated intervals |
| `.coverage` | `(X, y, alpha=0.1) → float` | Empirical coverage |
| `.interval_width` | `(X, alpha=0.1) → ndarray` | Interval widths |

### Validation Classes

```python
tpdf.Validator(dist, params)
tpdf.CrossValidator(dist)
```

### Backtesting

| Function | Signature | Description |
|----------|-----------|-------------|
| `rolling_var_backtest` | `(data, params, alpha) → dict` | VaR backtest |
| `Backtest` | `(model, data)` | Full backtest framework |
| `backtest` | `(model, data)` | High-level facade |

### Metrics

| Function | Description |
|----------|-------------|
| `log_likelihood(...)` | Log likelihood |
| `mae(...)` | Mean absolute error |
| `mse(...)` | Mean squared error |
| `rmse(...)` | Root mean squared error |
| `r_squared(...)` | R-squared |

**Example:**
```python
# Conformal prediction
conformal = tpdf.ConformalPredictor(model, X_cal, y_cal, distribution="student_t")
lower, upper = conformal.predict_interval(X_test, alpha=0.1)
coverage = conformal.coverage(X_test, y_test, alpha=0.1)
print(f"90% interval coverage: {coverage:.1%}")
```

---

## Stage 10: Visualization

Functions for visualizing distributions and results.

### Plotters

```python
tpdf.PDFPlotter(result)           # Static plots
tpdf.InteractivePlotter(result)   # Interactive 3D plots
```

### Plot Styles

| Style | Description |
|-------|-------------|
| `DEFAULT_STYLE` | Standard matplotlib style |
| `PUBLICATION_STYLE` | Clean style for papers |
| `PRESENTATION_STYLE` | Bold style for slides |
| `DARK_STYLE` | Dark background |

---

## Stage 11: Time Series Decomposition

Functions for decomposing time series.

| Function | Signature | Description |
|----------|-----------|-------------|
| `decompose_stl` | `(data) → dict` | STL decomposition |
| `decompose_stl_with_seasonality` | `(data, period) → dict` | STL with custom period |
| `decompose_fourier` | `(data) → dict` | Fourier decomposition |
| `decompose_wavelet` | `(data) → dict` | Wavelet decomposition |
| `decompose_moving_average` | `(data, window) → dict` | Moving average decomposition |
| `decompose_exponential_smoothing` | `(data, alpha) → dict` | Exponential smoothing |
| `get_dominant_frequencies` | `(data) → list` | Find dominant frequencies |

---

## Stage 12: Coefficient Extraction

Functions for extracting time-varying coefficients.

### Classes

```python
tpdf.RollingCoefficientExtractor(config)
tpdf.ExtractionConfig(...)
```

### Functions

| Function | Signature | Description |
|----------|-----------|-------------|
| `calculate_mean` | `(data) → float` | Rolling mean |
| `calculate_volatility` | `(data) → float` | Rolling volatility |
| `calculate_skewness` | `(data) → float` | Rolling skewness |
| `calculate_mean_rate` | `(data) → float` | Mean drift rate |
| `calculate_volatility_growth` | `(data) → float` | Volatility growth rate |

---

## Stage 13: PDF Evaluation

Functions for evaluating probability density functions.

### Quick Evaluation

```python
result = tpdf.evaluate(
    distribution="student_t",      # or distribution instance
    params=params,
    value_range=(-0.2, 0.2),
    time_range=(0.0, 60.0),
    value_points=200,
    time_points=100,
)
# Returns PDFResult
```

### Grid Creation

```python
grid = tpdf.EvaluationGrid.from_ranges(
    value_range=(-0.2, 0.2),
    time_range=(0.0, 60.0),
    value_points=200,
    time_points=100,
)
```

### Result Classes

| Class | Description |
|-------|-------------|
| `PDFResult` | PDF matrix + grids + metadata |
| `ValidationResult` | Validation metrics + details |
| `DiscoveryResult` | Auto-discovery results |
| `BacktestResult` | Backtest results |
| `Projection` | Temporal parameter projection |
| `PredictiveDistribution` | Full predictive distribution |

---

## Complete Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAW DATA                                        │
│                         (returns, prices, etc.)                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: FEATURE EXTRACTION                                                 │
│  extract_calibration_features(data, window=60)                               │
│  → Feature matrix (n_samples, 12)                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│  STAGE 2: DISTRIBUTION FIT    │   │  STAGE 3: MODEL TRAINING      │
│  fit(data, "student_t")       │   │  DistributionalRegressor()    │
│  select_best_distribution()   │   │  BarrierModel()               │
│  → Parameters                 │   │  → Trained model              │
└───────────────────────────────┘   └───────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DISTRIBUTION PARAMETERS (μ, σ, ν, etc.)                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────┬───────────┬───┴───┬───────────┬───────────┐
        ▼           ▼           ▼       ▼           ▼           ▼
┌───────────┐ ┌───────────┐ ┌───────┐ ┌───────┐ ┌───────────┐ ┌───────────┐
│ STAGE 6   │ │ STAGE 7   │ │STAGE 8│ │STAGE 9│ │ STAGE 4   │ │ STAGE 5   │
│ RISK      │ │ BARRIER   │ │SCORING│ │CONFML │ │ TEMPORAL  │ │ VOLATILITY│
│           │ │ PROB      │ │       │ │       │ │           │ │           │
│ var()     │ │ barrier_  │ │ crps()│ │Conform│ │ Temporal  │ │ garch_    │
│ cvar()    │ │ prob_mc() │ │ log_  │ │Predict│ │ Model     │ │ forecast()│
│ kelly_    │ │ barrier_  │ │ score │ │ or    │ │ .project()│ │ mean_     │
│ fraction()│ │ prob_     │ │       │ │       │ │           │ │ reverting │
│           │ │ temporal()│ │       │ │       │ │           │ │           │
└───────────┘ └───────────┘ └───────┘ └───────┘ └───────────┘ └───────────┘
        │           │           │       │           │           │
        └───────────┴───────────┴───┬───┴───────────┴───────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DECISIONS / OUTPUT                                   │
│  - Risk limits (VaR, CVaR)                                                   │
│  - Position sizing (Kelly)                                                   │
│  - Probability forecasts                                                     │
│  - Calibrated intervals                                                      │
│  - Backtest results                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Dependencies

### Required
- `numpy >= 1.20`
- `scipy >= 1.7`
- `pandas >= 1.3`

### Optional
- `torch >= 2.0` — For DistributionalRegressor, BarrierModel
- `matplotlib >= 3.5` — For visualization
- `plotly >= 5.0` — For interactive plots

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01 | Initial release with full Pipeline 2 optimization |

---

## License

MIT License - see LICENSE file for details.

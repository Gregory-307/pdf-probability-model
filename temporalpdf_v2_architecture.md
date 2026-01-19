# temporalpdf v2: Complete Architecture Document

## Overview

temporalpdf v2 is a probabilistic forecasting and decision library for quantitative trading. It provides a complete pipeline from raw market data to actionable trading decisions, with full uncertainty quantification at every stage.

**Core Philosophy**: Every prediction is a distribution, not a point estimate. Decisions (position sizing, risk limits) should account for both return uncertainty AND parameter uncertainty.

**What makes this unique**:
1. Distribution discovery with statistical significance testing
2. Temporal dynamics modeling (how distribution params evolve over time)
3. Weighting schemes (SMA/EMA/WMA) for parameter estimation
4. Decision utilities that work with ANY distribution (NIG, Student-t, etc.)
5. Full integration from discovery → fitting → projection → decision → backtest

---

## The Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER HAS: Raw Data                              │
│                         (returns, prices, features)                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: DISTRIBUTION DISCOVERY                          │
│                                                                              │
│  "Which distribution family best describes my data?"                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: MODEL TRAINING                               │
│                                                                              │
│  PATH A: Unconditional          PATH B: Conditional                          │
│  "Fit to this window"           "Learn f(features) → params"                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 3: TEMPORAL DYNAMICS                              │
│                                                                              │
│  3A: Parameter Tracking    - How have params evolved historically?           │
│  3B: Weighting Scheme      - SMA/EMA/WMA for current estimate               │
│  3C: Dynamics Model        - GARCH/Mean-Revert/Regime for each param        │
│  3D: Forward Projection    - Monte Carlo param paths                         │
│  3E: Predictive Dist       - Integrate over param uncertainty               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 4: DECISION LAYER                                 │
│                                                                              │
│  VaR, CVaR, Kelly, Probability Queries - with uncertainty bands              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 5: BACKTESTING                                    │
│                                                                              │
│  Rolling validation, Kupiec test, Christoffersen test, CRPS tracking         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER GETS: Trading Decision                          │
│                                                                              │
│  "Go long 0.47 Kelly, 5-day VaR is 2.3%, 73% probability of profit"         │
│  With confidence intervals on all estimates                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Distribution Discovery

### Purpose
Determine which distribution family best describes the data, with statistical rigor.

### Process
1. Fit multiple candidate distributions (Normal, Student-t, NIG, Skew-Normal, Generalized Laplace)
2. Score each with proper scoring rules (CRPS, Log Score) on held-out test data
3. Perform k-fold cross-validation for robust comparison
4. Run paired t-tests to determine if differences are statistically significant
5. Report confidence level in the selection

### Input
- Historical returns (1D array or Series)
- List of candidate distributions
- Scoring metrics to use
- Test fraction and CV folds

### Output
- Ranked list of distributions with scores
- Best distribution with confidence level
- p-values for pairwise comparisons

### API

```python
# ============================================================
# STAGE 1: Distribution Discovery
# ============================================================

discovery = tpdf.discover(
    data=returns,
    candidates=["normal", "student_t", "nig", "skew_normal", "gen_laplace"],
    scoring=["crps", "log_score"],
    test_fraction=0.2,
    cv_folds=5,
    significance_level=0.05
)

# View results
print(discovery.summary())
# ┌─────────────┬────────┬───────────┬─────────┬────────────┐
# │ Distribution│ CRPS   │ Log Score │ p-value │ Rank       │
# ├─────────────┼────────┼───────────┼─────────┼────────────┤
# │ nig         │ 0.0234 │ -1.42     │ --      │ 1 (best)   │
# │ student_t   │ 0.0251 │ -1.38     │ 0.003   │ 2          │
# │ skew_normal │ 0.0267 │ -1.31     │ 0.001   │ 3          │
# │ gen_laplace │ 0.0278 │ -1.27     │ <0.001  │ 4          │
# │ normal      │ 0.0312 │ -1.19     │ <0.001  │ 5          │
# └─────────────┴────────┴───────────┴─────────┴────────────┘
# 
# Best: nig
# Confidence: HIGH (significantly better than all others at α=0.05)

# Access results programmatically
best_dist = discovery.best           # "nig"
confidence = discovery.confidence    # "high"
scores = discovery.scores            # DataFrame with all scores
pvalues = discovery.pairwise_pvalues # Matrix of pairwise p-values

# Detailed diagnostics
discovery.plot_score_comparison()    # Bar chart of scores
discovery.plot_qq(distribution="nig") # Q-Q plot for best fit
discovery.plot_tail_comparison()     # Compare tail behavior

# Get fitted params for best distribution
best_params = discovery.best_params  # Params fitted during discovery
```

### Implementation Notes

**Build**:
- Selection logic with train/test split
- Significance testing (paired t-test, Wilcoxon signed-rank)
- Confidence scoring heuristic
- Summary tables and diagnostics

**Import**:
- `scipy.stats` for distribution fitting
- `properscoring` for CRPS calculation

---

## Stage 2: Model Training

### Purpose
Estimate distribution parameters from data. Two paths: unconditional (fit to window) or conditional (learn feature→param mapping).

### Path A: Unconditional Fitting

Fit distribution to a window of historical data without features.

```python
# ============================================================
# STAGE 2A: Unconditional Fitting
# ============================================================

# Simple fit to data
params = tpdf.fit(returns, distribution="nig")
# Returns: NIGParameters(mu=0.0012, delta=0.023, alpha=15.2, beta=-1.8)

# Fit with weighting (see Stage 3B for weight options)
params = tpdf.fit(
    returns,
    distribution="nig",
    weights=tpdf.weights.EMA(halflife=20)
)

# Fit multiple distributions at once
all_params = tpdf.fit_all(
    returns,
    distributions=["normal", "student_t", "nig"]
)
# Returns: {"normal": NormalParams(...), "student_t": StudentTParams(...), "nig": NIGParams(...)}

# Access param values
print(params.mu)      # Location
print(params.delta)   # Scale (volatility)
print(params.alpha)   # Tail heaviness
print(params.beta)    # Skewness

# Params are immutable dataclasses
params_updated = params.with_delta(0.025)  # Create new with modified delta
```

### Path B: Conditional Fitting

Learn a model that maps features to distribution parameters.

```python
# ============================================================
# STAGE 2B: Conditional Fitting
# ============================================================

# Create conditional model
model = tpdf.ConditionalModel(
    distribution="nig",
    backend="xgboostlss",  # or "ngboost", "lightgbmlss"
)

# Fit to features and returns
model.fit(
    X_train,  # Features DataFrame
    y_train,  # Returns array
    eval_set=(X_val, y_val),
    early_stopping_rounds=50
)

# Hyperparameter tuning
model.tune(
    X_train, y_train,
    param_grid={
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 7],
        "n_estimators": [100, 200, 500]
    },
    cv=5,
    scoring="crps"
)

# Predict params for new observations
params_df = model.predict(X_test)
# Returns DataFrame:
#       mu      delta    alpha    beta
# 0   0.0015   0.021    15.1    -1.3
# 1   0.0008   0.025    14.8    -1.5
# 2   0.0021   0.019    15.5    -1.1
# ...

# Predict full distribution
distributions = model.predict_distribution(X_test)

# Sample from predicted distributions
samples = model.sample(X_test, n_samples=1000)

# Predict quantiles directly
quantiles = model.predict_quantiles(X_test, q=[0.05, 0.5, 0.95])

# Explainability - what drives each parameter?
model.plot_shap(parameter="delta")        # What drives volatility?
model.plot_shap(parameter="beta")         # What drives skewness?
model.plot_partial_dependence("delta", feature="vix")

# Feature importance per parameter
importance = model.feature_importance()
# Returns: {"mu": {...}, "delta": {...}, "alpha": {...}, "beta": {...}}
```

### Implementation Notes

**Build**:
- Unified `fit()` interface
- NIG MLE optimizer (scipy's is awkward for this parameterization)
- `ConditionalModel` wrapper class
- SHAP integration for explainability

**Import**:
- `scipy.stats` for Normal/Student-t/Skew-Normal fitting
- `xgboostlss` / `ngboost` / `lightgbmlss` for conditional models
- `shap` for explainability

---

## Stage 3: Temporal Dynamics

### Purpose
Model how distribution parameters evolve over time, project them forward, and produce predictive distributions that account for parameter uncertainty.

### Sub-stages

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 3: TEMPORAL DYNAMICS                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3A: Parameter Tracking                                              │    │
│  │     Roll through history, fit distribution at each point            │    │
│  │     Output: Time series of params (μ_t, σ_t, ν_t, ...)             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3B: Weighting Scheme                                                │    │
│  │     How to weight historical data for "current" param estimate      │    │
│  │     SMA / EMA / WMA / Custom / Regime-weighted                      │    │
│  │     Output: Point estimate of current params                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3C: Dynamics Model                                                  │    │
│  │     Model how each param evolves over time                          │    │
│  │     GARCH / Mean-Reverting / Random Walk / Regime-Switching         │    │
│  │     Output: Fitted dynamics model for each param                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3D: Forward Projection                                              │    │
│  │     Project params forward using fitted dynamics                    │    │
│  │     Monte Carlo simulation of param paths                           │    │
│  │     Output: P(θ | t) - distribution over params at each future t    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ 3E: Predictive Distribution                                         │    │
│  │     Integrate over parameter uncertainty                            │    │
│  │     P(r_t) = ∫ P(r_t | θ) · P(θ | t) dθ                            │    │
│  │     Output: Full predictive distribution accounting for all         │    │
│  │             uncertainty                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3A: Parameter Tracking

Track how distribution parameters have evolved historically by fitting at each point in time.

```python
# ============================================================
# 3A: Parameter Tracking
# ============================================================

# Create a tracker
tracker = tpdf.ParameterTracker(
    distribution="nig",
    window=60,      # Fit to 60-day rolling windows
    step=1,         # Re-fit every day
    min_window=30,  # Minimum window size at start
)

# Fit to historical data
param_history = tracker.fit(returns)

# param_history is a DataFrame:
#            mu       delta    alpha    beta
# 2024-01-01 0.0012   0.021    15.3    -1.2
# 2024-01-02 0.0011   0.022    15.1    -1.3
# 2024-01-03 0.0015   0.024    14.8    -1.1
# 2024-01-04 0.0009   0.023    14.9    -1.2
# ...

# Visualize parameter evolution
tracker.plot()                          # All params over time
tracker.plot(params=["delta", "alpha"]) # Specific params
tracker.plot_correlation()              # Param correlations over time

# Statistics on param evolution
tracker.summary()
# ┌───────┬─────────┬─────────┬─────────┬─────────┬──────────┐
# │ Param │ Mean    │ Std     │ Min     │ Max     │ AutoCorr │
# ├───────┼─────────┼─────────┼─────────┼─────────┼──────────┤
# │ mu    │ 0.0011  │ 0.0008  │ -0.0015 │ 0.0035  │ 0.89     │
# │ delta │ 0.022   │ 0.005   │ 0.012   │ 0.041   │ 0.95     │
# │ alpha │ 15.1    │ 1.2     │ 11.3    │ 18.2    │ 0.82     │
# │ beta  │ -1.2    │ 0.4     │ -2.1    │ -0.3    │ 0.78     │
# └───────┴─────────┴─────────┴─────────┴─────────┴──────────┘

# Alternative tracking methods
tracker_expanding = tpdf.ParameterTracker(
    distribution="nig",
    method="expanding",  # Use all data up to each point
    min_window=60
)

tracker_event = tpdf.ParameterTracker(
    distribution="nig",
    method="event",      # Re-fit only on specific events
    trigger=lambda returns: abs(returns[-1]) > 0.03  # Refit on large moves
)
```

### 3B: Weighting Schemes

Different methods to weight historical data when estimating current parameters.

```python
# ============================================================
# 3B: Weighting Schemes
# ============================================================

# The question: given N days of data, how do we weight them
# when estimating current distribution params?

# 1. Simple Moving Average (SMA)
#    All observations in window weighted equally
#    weight[i] = 1/N for all i in window
weights_sma = tpdf.weights.SMA(window=60)

# 2. Exponential Moving Average (EMA)
#    Recent observations weighted more, exponential decay
#    weight[i] = (1-α) * α^i, where α = exp(-ln(2)/halflife)
weights_ema = tpdf.weights.EMA(halflife=20)

# 3. Linear Weighted Moving Average (WMA)
#    Linear decay: most recent has weight N, oldest has weight 1
#    weight[i] = N - i
weights_linear = tpdf.weights.Linear(window=60)

# 4. Square Root Decay
#    Slower decay than linear, more weight to older data than EMA
#    weight[i] = 1 / sqrt(i + 1)
weights_sqrt = tpdf.weights.SqrtDecay(window=120)

# 5. Power Decay
#    Configurable decay rate
#    weight[i] = 1 / (i + 1)^p
weights_power = tpdf.weights.PowerDecay(power=0.5, window=120)

# 6. Gaussian Decay
#    Bell-curve weighting centered on recent data
#    weight[i] = exp(-0.5 * (i/σ)²)
weights_gaussian = tpdf.weights.Gaussian(sigma=30)

# 7. Custom Function
#    Any function of (index, total_length) -> weight
weights_custom = tpdf.weights.Custom(
    func=lambda i, n: np.exp(-0.5 * (i/20)**2) if i < 60 else 0
)

# 8. Regime-Weighted
#    Weight by similarity to current market regime
weights_regime = tpdf.weights.RegimeWeighted(
    regime_detector=tpdf.regime.HMM(n_states=3),
    similarity="state_probability"  # or "mahalanobis", "euclidean"
)

# 9. Volatility-Weighted
#    Upweight periods with similar volatility to current
weights_vol = tpdf.weights.VolatilityWeighted(
    vol_window=20,
    similarity_bandwidth=0.5
)

# Visualize and compare weights
tpdf.weights.plot_comparison(
    [weights_sma, weights_ema, weights_linear, weights_sqrt],
    window=60,
    labels=["SMA", "EMA(20)", "Linear", "Sqrt"]
)

# Use weights in fitting
params = tpdf.fit(returns, distribution="nig", weights=weights_ema)

# Use weights in tracking
tracker = tpdf.ParameterTracker(
    distribution="nig",
    window=120,
    weights=weights_ema  # Apply EMA within each window
)

# Get weight values for inspection
w = weights_ema.get_weights(n=60)  # Array of 60 weights
print(f"Most recent weight: {w[0]:.4f}")
print(f"Oldest weight: {w[-1]:.4f}")
print(f"Effective sample size: {weights_ema.effective_sample_size(60):.1f}")
```

### 3C: Dynamics Models

Model how each distribution parameter evolves over time.

```python
# ============================================================
# 3C: Dynamics Models
# ============================================================

# Given param_history (from 3A), model how each param changes

# 1. Constant
#    Parameter stays at long-run average, no dynamics
#    θ_{t+1} = θ_long
dynamics_const = tpdf.dynamics.Constant()

# 2. Random Walk
#    Parameter drifts randomly
#    θ_{t+1} = θ_t + drift + σ·ε, where ε ~ N(0,1)
dynamics_rw = tpdf.dynamics.RandomWalk(
    drift=True,      # Estimate drift from data
    # drift=0.0001   # Or specify fixed drift
)

# 3. AR(1) - Autoregressive
#    Mean-reverting with AR(1) dynamics
#    θ_{t+1} = c + φ·θ_t + σ·ε
dynamics_ar1 = tpdf.dynamics.AR(order=1)

# 4. AR(p) - Higher order autoregressive
#    θ_{t+1} = c + φ_1·θ_t + φ_2·θ_{t-1} + ... + σ·ε
dynamics_ar2 = tpdf.dynamics.AR(order=2)

# 5. Mean-Reverting (Ornstein-Uhlenbeck)
#    Continuous-time mean reversion
#    dθ = κ·(θ_long - θ)·dt + σ·dW
dynamics_ou = tpdf.dynamics.MeanReverting(
    estimate_long_run=True,   # Estimate θ_long from data
    # long_run=15.0           # Or specify manually
)

# 6. GARCH(p,q) - For volatility parameters
#    σ²_{t+1} = ω + Σα_i·ε²_{t-i+1} + Σβ_j·σ²_{t-j+1}
dynamics_garch = tpdf.dynamics.GARCH(p=1, q=1)

# 7. GJR-GARCH - Asymmetric GARCH
#    Allows different response to positive/negative shocks
#    σ²_{t+1} = ω + (α + γ·I_{ε<0})·ε²_t + β·σ²_t
dynamics_gjr = tpdf.dynamics.GJRGARCH(p=1, q=1)

# 8. EGARCH - Exponential GARCH
#    Models log-variance, always positive
#    log(σ²_{t+1}) = ω + α·g(z_t) + β·log(σ²_t)
dynamics_egarch = tpdf.dynamics.EGARCH(p=1, q=1)

# 9. HAR - Heterogeneous Autoregressive
#    Common for realized volatility, uses daily/weekly/monthly
#    σ_{t+1} = c + β_d·σ_t + β_w·σ_{t,week} + β_m·σ_{t,month}
dynamics_har = tpdf.dynamics.HAR()

# 10. Regime-Switching
#     Parameter jumps between discrete states
#     θ_t ∈ {θ_1, θ_2, ..., θ_K}, transitions via Markov chain
dynamics_regime = tpdf.dynamics.RegimeSwitching(
    n_regimes=3,
    # Or provide initial transition matrix:
    # transition_matrix=np.array([[0.95, 0.04, 0.01], ...])
)

# 11. Markov-Switching AR
#     AR dynamics with regime-dependent parameters
dynamics_msar = tpdf.dynamics.MarkovSwitchingAR(
    n_regimes=2,
    ar_order=1
)

# 12. State-Space / Kalman Filter
#     Full state-space model
#     State:       θ_{t+1} = A·θ_t + B·u_t + Q·ε_t
#     Observation: y_t = C·θ_t + R·η_t
dynamics_kalman = tpdf.dynamics.StateSpace(
    state_transition="ar1",
    observation_model="identity"
)

# 13. Neural Network
#     Learn dynamics with LSTM/GRU
dynamics_nn = tpdf.dynamics.NeuralNetwork(
    architecture="lstm",
    hidden_size=32,
    n_layers=2
)

# Fit dynamics to param history
dynamics_garch.fit(param_history["delta"])

# Inspect fitted model
print(dynamics_garch.summary())
# GARCH(1,1) Results:
# ω (omega):  1.23e-06
# α (alpha):  0.089
# β (beta):   0.891
# Persistence (α+β): 0.980
# Long-run variance: 6.15e-05
# Long-run vol: 0.78%
# Half-life: 34.3 days
# Log-likelihood: 1523.4
# AIC: -3040.8

# Diagnostic plots
dynamics_garch.plot_fit()           # Fitted vs actual
dynamics_garch.plot_residuals()     # Standardized residuals
dynamics_garch.plot_acf()           # ACF of residuals
```

### 3C (continued): Combining Dynamics for Multiple Parameters

```python
# ============================================================
# 3C: Combined Dynamics for All Parameters
# ============================================================

# Different params may need different dynamics models
param_dynamics = tpdf.ParameterDynamics(
    param_history=param_history,
    models={
        "mu": tpdf.dynamics.RandomWalk(drift=True),
        "delta": tpdf.dynamics.GARCH(1, 1),
        "alpha": tpdf.dynamics.MeanReverting(),
        "beta": tpdf.dynamics.AR(1),
    }
)

# Fit all dynamics models
param_dynamics.fit()

# Summary of all fitted models
print(param_dynamics.summary())
# ┌───────┬──────────────────┬─────────────────────────────────────┐
# │ Param │ Model            │ Fitted Values                       │
# ├───────┼──────────────────┼─────────────────────────────────────┤
# │ mu    │ Random Walk      │ drift=0.0001, σ=0.0003              │
# │ delta │ GARCH(1,1)       │ ω=1.2e-6, α=0.089, β=0.891          │
# │ alpha │ Mean-Reverting   │ θ_long=15.0, κ=0.05, σ=0.8          │
# │ beta  │ AR(1)            │ c=-0.15, φ=0.92, σ=0.12             │
# └───────┴──────────────────┴─────────────────────────────────────┘

# Model selection - automatically choose best dynamics for each param
param_dynamics_auto = tpdf.ParameterDynamics(
    param_history=param_history,
    models="auto",  # Automatically select best model per param
    candidates=[
        tpdf.dynamics.Constant(),
        tpdf.dynamics.RandomWalk(),
        tpdf.dynamics.AR(1),
        tpdf.dynamics.MeanReverting(),
        tpdf.dynamics.GARCH(1, 1),
    ],
    selection_criterion="bic"  # or "aic", "cross_val"
)

param_dynamics_auto.fit()
print(param_dynamics_auto.selected_models)
# {"mu": "RandomWalk", "delta": "GARCH(1,1)", "alpha": "MeanReverting", "beta": "AR(1)"}
```

### 3D: Forward Projection

Project parameters forward using fitted dynamics models.

```python
# ============================================================
# 3D: Forward Projection
# ============================================================

# Get current param estimate (using weighting from 3B)
current_params = tpdf.fit(
    returns,
    distribution="nig",
    weights=tpdf.weights.EMA(halflife=20)
)

# Project forward using fitted dynamics
projection = param_dynamics.project(
    current_params=current_params,
    horizon=30,           # 30 days ahead
    n_paths=1000,         # Monte Carlo paths
    confidence_levels=[0.5, 0.9, 0.95]  # Confidence bands to compute
)

# projection contains:
# - mean: Expected params at each time
# - std: Std dev of params at each time
# - quantiles: Quantiles at each time (for confidence bands)
# - paths: All simulated paths (if store_paths=True)

# Access projected values
projection.mean(t=5)        # Expected params at t=5
# NIGParameters(mu=0.0013, delta=0.024, alpha=15.1, beta=-1.2)

projection.std(t=5)         # Std dev at t=5
# {"mu": 0.0004, "delta": 0.003, "alpha": 0.6, "beta": 0.15}

projection.quantile(0.95, t=5)  # 95th percentile at t=5
projection.quantile(0.05, t=5)  # 5th percentile at t=5

# Get full path of a single parameter
delta_mean = projection.param_path("delta", stat="mean")      # Array of length horizon
delta_upper = projection.param_path("delta", stat="q95")      # 95% upper bound
delta_lower = projection.param_path("delta", stat="q05")      # 5% lower bound

# Visualization
projection.plot()                        # All params with confidence bands
projection.plot(params=["delta"])        # Single param
projection.plot_fan(param="delta")       # Fan chart
projection.plot_paths(param="delta", n_paths=50)  # Show sample paths

# Access raw paths for custom analysis
if projection.paths is not None:
    # paths shape: (n_paths, horizon, n_params)
    delta_paths = projection.paths[:, :, 1]  # All delta paths
    
    # Probability delta exceeds threshold at t=10
    prob_high_vol = np.mean(delta_paths[:, 10] > 0.03)
```

### 3E: Predictive Distribution

Combine return distribution with parameter uncertainty to get full predictive distribution.

```python
# ============================================================
# 3E: Predictive Distribution
# ============================================================

# The predictive distribution at time t accounts for:
# 1. Return uncertainty (given params): P(r | θ)
# 2. Parameter uncertainty: P(θ | t)
# 
# Full predictive: P(r_t) = ∫ P(r | θ) · P(θ | t) dθ

# Create predictive distribution
predictive = tpdf.PredictiveDistribution(
    distribution="nig",
    param_projection=projection,  # From 3D
)

# Get predictive distribution at specific horizon
pred_5d = predictive.at(t=5)

# pred_5d is a mixture distribution that accounts for param uncertainty
# It's represented as a weighted mixture of NIG distributions

# Evaluate PDF/CDF
x = np.linspace(-0.1, 0.1, 1000)
pdf_values = pred_5d.pdf(x)
cdf_values = pred_5d.cdf(x)

# Quantiles (accounting for param uncertainty)
var_95 = pred_5d.ppf(0.05)   # 5th percentile (95% VaR)
median = pred_5d.ppf(0.5)

# Moments
mean_5d = pred_5d.mean()
std_5d = pred_5d.std()
skew_5d = pred_5d.skewness()
kurt_5d = pred_5d.kurtosis()

# Sample from predictive
samples = pred_5d.sample(10000)

# Compare: with vs without param uncertainty
pred_no_uncertainty = tpdf.PredictiveDistribution(
    distribution="nig",
    params=current_params,  # Fixed params, no uncertainty
)

pred_5d_fixed = pred_no_uncertainty.at(t=5)

# The predictive with uncertainty will have fatter tails
print(f"Std (with param uncertainty): {pred_5d.std():.4f}")
print(f"Std (fixed params): {pred_5d_fixed.std():.4f}")

# Visualization
predictive.plot_pdf(t=5)                    # PDF at t=5
predictive.plot_pdf_evolution(t_range=(1, 30))  # How PDF changes over time
predictive.plot_comparison(                 # Compare with/without uncertainty
    other=pred_no_uncertainty,
    t=5,
    labels=["With param uncertainty", "Fixed params"]
)
```

### Complete Stage 3 API

Putting it all together with the `TemporalModel` class:

```python
# ============================================================
# COMPLETE STAGE 3 WORKFLOW
# ============================================================

# Create temporal model - combines all sub-stages
temporal = tpdf.TemporalModel(
    distribution="nig",
    
    # 3A: Parameter tracking configuration
    tracking=tpdf.tracking.Rolling(
        window=60,
        step=1,
        min_window=30
    ),
    
    # 3B: Weighting scheme for current estimate
    weighting=tpdf.weights.EMA(halflife=20),
    
    # 3C: Dynamics model for each parameter
    dynamics={
        "mu": tpdf.dynamics.RandomWalk(drift=True),
        "delta": tpdf.dynamics.GARCH(1, 1),
        "alpha": tpdf.dynamics.MeanReverting(),
        "beta": tpdf.dynamics.AR(1),
    },
    # Or use automatic selection:
    # dynamics="auto",
)

# Fit the entire temporal model to data
temporal.fit(returns)

# What we now have:
# - param_history: How params evolved historically
# - current_params: Current param estimate (weighted)
# - dynamics_models: Fitted dynamics for each param

# Inspect results
print(temporal.current_params)
print(temporal.dynamics_summary())
temporal.plot_param_history()
temporal.plot_dynamics_diagnostics()

# Project forward
projection = temporal.project(
    horizon=30,
    n_paths=1000,
    confidence=0.95
)

projection.plot()

# Get predictive distribution
predictive = temporal.predictive_distribution()

# Evaluate at specific horizon
pred_5d = predictive.at(t=5)
print(f"5-day expected return: {pred_5d.mean():.4f}")
print(f"5-day volatility: {pred_5d.std():.4f}")

# Full decision summary with uncertainty (connects to Stage 4)
decision = temporal.decision(
    t=5,
    alpha=0.05,
    integrate_param_uncertainty=True
)

print(decision)
# ┌─────────────────────┬──────────┬───────────────────┐
# │ Metric              │ Mean     │ 90% CI            │
# ├─────────────────────┼──────────┼───────────────────┤
# │ VaR (95%)           │ -2.31%   │ [-2.68%, -1.94%]  │
# │ CVaR (95%)          │ -3.87%   │ [-4.52%, -3.21%]  │
# │ Kelly fraction      │ 0.47     │ [0.31, 0.63]      │
# │ P(profit)           │ 73.2%    │ [68.1%, 78.4%]    │
# │ Expected return     │ +0.89%   │ [+0.52%, +1.26%]  │
# └─────────────────────┴──────────┴───────────────────┘
```

---

## Stage 4: Decision Layer

### Purpose
Convert predictive distributions into actionable trading decisions: position sizing, risk limits, and probability assessments.

### API

```python
# ============================================================
# STAGE 4: DECISION LAYER
# ============================================================

# Option 1: Use with TemporalModel (recommended - includes param uncertainty)
decision = temporal.decision(t=5, alpha=0.05)

# Option 2: Use directly with a distribution and params
forecast = tpdf.Forecast(
    distribution="nig",
    params=params,
    volatility_model=tpdf.dynamics.GARCH(1, 1).fit(vol_history)
)

# ============================================================
# Risk Measures
# ============================================================

# Value at Risk
var_1d = forecast.var(alpha=0.05, t=1)    # 1-day 95% VaR
var_5d = forecast.var(alpha=0.05, t=5)    # 5-day 95% VaR
var_99 = forecast.var(alpha=0.01, t=1)    # 1-day 99% VaR

# Conditional Value at Risk (Expected Shortfall)
cvar_1d = forecast.cvar(alpha=0.05, t=1)  # Expected loss given we're in worst 5%
cvar_5d = forecast.cvar(alpha=0.05, t=5)

# Multiple confidence levels at once
var_table = forecast.var_table(
    alphas=[0.01, 0.025, 0.05, 0.10],
    horizons=[1, 5, 10, 20]
)
# Returns DataFrame with VaR for each (alpha, horizon) combination

# ============================================================
# Position Sizing
# ============================================================

# Kelly Criterion - optimal fraction for log-wealth maximization
kelly = forecast.kelly(t=5)               # Full Kelly
half_kelly = forecast.kelly(t=5, fraction=0.5)  # Half Kelly

# Kelly with constraints
kelly_constrained = forecast.kelly(
    t=5,
    max_position=0.5,     # Never exceed 50% allocation
    max_leverage=2.0,     # Max 2x leverage
    risk_free_rate=0.05   # 5% annual risk-free rate
)

# Kelly for multiple assets (portfolio)
kelly_portfolio = tpdf.kelly_portfolio(
    forecasts=[forecast1, forecast2, forecast3],
    correlation_matrix=corr,
    t=5
)

# ============================================================
# Probability Queries
# ============================================================

# Probability of profit
p_profit = forecast.prob(">", 0, t=5)

# Probability of loss exceeding threshold
p_loss_5pct = forecast.prob("<", -0.05, t=5)

# Probability of return in range
p_range = forecast.prob("between", -0.02, 0.05, t=5)

# Probability of hitting target
p_target = forecast.prob(">=", 0.10, t=20)  # 10% gain in 20 days

# Multiple probability queries
probs = forecast.prob_table(
    thresholds=[-0.10, -0.05, -0.02, 0, 0.02, 0.05, 0.10],
    t=5
)
# Returns: probability of exceeding each threshold

# ============================================================
# Expected Values
# ============================================================

expected_return = forecast.mean(t=5)
expected_vol = forecast.std(t=5)
expected_skew = forecast.skewness(t=5)
expected_kurt = forecast.kurtosis(t=5)

# ============================================================
# Quantiles
# ============================================================

median = forecast.quantile(0.5, t=5)
q05 = forecast.quantile(0.05, t=5)
q95 = forecast.quantile(0.95, t=5)

quantiles = forecast.quantile([0.05, 0.25, 0.5, 0.75, 0.95], t=5)

# ============================================================
# Complete Decision Summary
# ============================================================

summary = forecast.decision_summary(
    t=5,
    alpha=0.05,
    kelly_fraction=0.5,
    risk_free_rate=0.05
)

print(summary)
# ┌─────────────────────┬──────────┐
# │ Metric              │ Value    │
# ├─────────────────────┼──────────┤
# │ Expected return     │ +0.89%   │
# │ Volatility          │ 1.42%    │
# │ Skewness            │ -0.31    │
# │ Kurtosis            │ 4.82     │
# │ VaR (95%)           │ -2.31%   │
# │ CVaR (95%)          │ -3.87%   │
# │ Full Kelly          │ 0.47     │
# │ Half Kelly          │ 0.24     │
# │ P(profit)           │ 73.2%    │
# │ P(loss > 5%)        │ 4.1%     │
# │ Sharpe (annualized) │ 1.23     │
# └─────────────────────┴──────────┘

# Export for use in trading system
summary.to_dict()
summary.to_json()

# ============================================================
# Decision with Parameter Uncertainty
# ============================================================

# When using TemporalModel, get decisions with confidence intervals
decision_with_ci = temporal.decision(
    t=5,
    alpha=0.05,
    integrate_param_uncertainty=True,
    confidence_level=0.90
)

print(decision_with_ci)
# ┌─────────────────────┬──────────┬───────────────────┐
# │ Metric              │ Mean     │ 90% CI            │
# ├─────────────────────┼──────────┼───────────────────┤
# │ VaR (95%)           │ -2.31%   │ [-2.68%, -1.94%]  │
# │ CVaR (95%)          │ -3.87%   │ [-4.52%, -3.21%]  │
# │ Kelly fraction      │ 0.47     │ [0.31, 0.63]      │
# │ P(profit)           │ 73.2%    │ [68.1%, 78.4%]    │
# │ Expected return     │ +0.89%   │ [+0.52%, +1.26%]  │
# └─────────────────────┴──────────┴───────────────────┘
```

### Implementation Notes

**Build**:
- All VaR/CVaR/Kelly computations
- Integration with arbitrary distributions (not just Normal)
- Kelly for non-Normal distributions (numerical integration)
- Probability queries
- Monte Carlo integration over param uncertainty

**Import**:
- `scipy.integrate` for numerical integration
- `scipy.optimize` for Kelly optimization

---

## Stage 5: Backtesting

### Purpose
Validate the entire pipeline historically: does the model produce well-calibrated forecasts? Are VaR exceedances at expected rates?

### API

```python
# ============================================================
# STAGE 5: BACKTESTING
# ============================================================

# Full pipeline backtest
backtest = tpdf.backtest(
    data=full_history,             # Full return series
    
    # Stage 1: Distribution (can be fixed or re-selected periodically)
    distribution="nig",            # Fixed distribution
    # Or: distribution="auto", reselect_every=252  # Re-select annually
    
    # Stage 2: Fitting
    lookback=252,                  # Fit to trailing 252 days
    
    # Stage 3: Temporal dynamics
    weighting=tpdf.weights.EMA(halflife=20),
    dynamics={
        "delta": tpdf.dynamics.GARCH(1, 1),
        "mu": tpdf.dynamics.RandomWalk(),
        "alpha": tpdf.dynamics.MeanReverting(),
        "beta": tpdf.dynamics.Constant(),
    },
    
    # Stage 4: What to forecast
    forecast_horizon=5,            # 5-day ahead forecasts
    alpha=0.05,                    # 95% VaR
    
    # Backtest configuration
    start_date="2020-01-01",       # When to start testing
    step=1,                        # Re-forecast every day
)

# Run the backtest
backtest.run()

# ============================================================
# Results Summary
# ============================================================

print(backtest.summary())
# ┌─────────────────────────┬──────────┬─────────┐
# │ Metric                  │ Value    │ Status  │
# ├─────────────────────────┼──────────┼─────────┤
# │ Expected exceedance rate│ 5.00%    │         │
# │ Actual exceedance rate  │ 5.32%    │ ✓       │
# │ Kupiec p-value          │ 0.72     │ PASS    │
# │ Christoffersen p-value  │ 0.45     │ PASS    │
# │ Mean CRPS               │ 0.0231   │         │
# │ Mean Log Score          │ -1.42    │         │
# │ Calibration (PIT test)  │ 0.38     │ PASS    │
# │ Overall Status          │          │ PASS    │
# └─────────────────────────┴──────────┴─────────┘

# ============================================================
# Detailed Results
# ============================================================

# Exceedance analysis
print(f"Total forecasts: {backtest.n_forecasts}")
print(f"VaR exceedances: {backtest.n_exceedances}")
print(f"Exceedance rate: {backtest.exceedance_rate:.2%}")

# Time series of results
results_df = backtest.results
# DataFrame with columns:
#   date, actual_return, var_forecast, cvar_forecast, 
#   exceedance, crps, log_score, pit_value, ...

# Statistical tests
tests = backtest.tests()
# ┌─────────────────────┬──────────┬─────────┬─────────┐
# │ Test                │ Statistic│ p-value │ Result  │
# ├─────────────────────┼──────────┼─────────┼─────────┤
# │ Kupiec (UC)         │ 0.13     │ 0.72    │ PASS    │
# │ Christoffersen (Ind)│ 0.57     │ 0.45    │ PASS    │
# │ Christoffersen (CC) │ 0.70     │ 0.70    │ PASS    │
# │ Dynamic Quantile    │ 1.23     │ 0.31    │ PASS    │
# │ Berkowitz           │ 2.31     │ 0.51    │ PASS    │
# │ PIT Uniformity (KS) │ 0.032    │ 0.38    │ PASS    │
# └─────────────────────┴──────────┴─────────┴─────────┘

# ============================================================
# Visualization
# ============================================================

# VaR vs actual returns
backtest.plot()

# Detailed plots
backtest.plot_exceedances()           # Highlight exceedance events
backtest.plot_var_evolution()         # How VaR changed over time
backtest.plot_crps_evolution()        # CRPS over time
backtest.plot_pit_histogram()         # PIT uniformity check
backtest.plot_qq()                    # Q-Q plot of PITs
backtest.plot_coverage_by_period()    # Exceedance rate by year/month

# ============================================================
# Comparison Backtests
# ============================================================

# Compare multiple configurations
backtest_normal = tpdf.backtest(data=returns, distribution="normal", ...)
backtest_t = tpdf.backtest(data=returns, distribution="student_t", ...)
backtest_nig = tpdf.backtest(data=returns, distribution="nig", ...)

backtest_normal.run()
backtest_t.run()
backtest_nig.run()

comparison = tpdf.compare_backtests(
    [backtest_normal, backtest_t, backtest_nig],
    labels=["Normal", "Student-t", "NIG"]
)

print(comparison.summary())
# ┌─────────────┬───────────┬───────────┬───────────┐
# │ Metric      │ Normal    │ Student-t │ NIG       │
# ├─────────────┼───────────┼───────────┼───────────┤
# │ Exc. rate   │ 7.2%      │ 5.8%      │ 5.3%      │
# │ Kupiec p    │ 0.02*     │ 0.31      │ 0.72      │
# │ Mean CRPS   │ 0.0298    │ 0.0254    │ 0.0231    │
# │ Mean LS     │ -1.19     │ -1.35     │ -1.42     │
# │ Status      │ FAIL      │ PASS      │ PASS      │
# └─────────────┴───────────┴───────────┴───────────┘
# * indicates failure at α=0.05

comparison.plot()

# ============================================================
# Conditional Model Backtest
# ============================================================

# For conditional models (with features)
backtest_cond = tpdf.backtest(
    returns=returns,
    features=features,
    model=tpdf.ConditionalModel(distribution="nig", backend="xgboostlss"),
    lookback=252,
    forecast_horizon=5,
    alpha=0.05,
    retrain_every=20,      # Retrain model every 20 days
)

backtest_cond.run()
```

### Implementation Notes

**Build**:
- Rolling backtest framework
- Kupiec test (unconditional coverage)
- Christoffersen test (independence and conditional coverage)
- Dynamic Quantile test
- Berkowitz test
- PIT (Probability Integral Transform) analysis
- Comparison framework

**Import**:
- `properscoring` for CRPS
- `scipy.stats` for statistical tests

---

## Complete API Example

Full end-to-end workflow:

```python
import temporalpdf as tpdf
import pandas as pd

# Load data
returns = pd.read_csv("returns.csv", index_col=0, parse_dates=True)["return"]

# ============================================================
# STAGE 1: Distribution Discovery
# ============================================================
print("Stage 1: Discovering best distribution...")

discovery = tpdf.discover(
    data=returns,
    candidates=["normal", "student_t", "nig", "skew_normal"],
    scoring=["crps", "log_score"],
    cv_folds=5,
    significance_level=0.05
)

print(discovery.summary())
best_dist = discovery.best
print(f"\nBest distribution: {best_dist} (confidence: {discovery.confidence})")

# ============================================================
# STAGE 2: Fitting (Unconditional)
# ============================================================
print("\nStage 2: Fitting distribution...")

params = tpdf.fit(returns, distribution=best_dist)
print(f"Fitted params: {params}")

# ============================================================
# STAGE 3: Temporal Dynamics
# ============================================================
print("\nStage 3: Modeling temporal dynamics...")

temporal = tpdf.TemporalModel(
    distribution=best_dist,
    tracking=tpdf.tracking.Rolling(window=60, step=1),
    weighting=tpdf.weights.EMA(halflife=20),
    dynamics={
        "mu": tpdf.dynamics.RandomWalk(drift=True),
        "delta": tpdf.dynamics.GARCH(1, 1),
        "alpha": tpdf.dynamics.MeanReverting(),
        "beta": tpdf.dynamics.AR(1),
    }
)

temporal.fit(returns)
print(temporal.dynamics_summary())

# Project forward
projection = temporal.project(horizon=30, n_paths=1000)
projection.plot()

# ============================================================
# STAGE 4: Decision
# ============================================================
print("\nStage 4: Computing trading decisions...")

# Decision for 5-day horizon
decision = temporal.decision(
    t=5,
    alpha=0.05,
    integrate_param_uncertainty=True,
    confidence_level=0.90
)

print(decision)

# Quick access to key metrics
print(f"\n5-day 95% VaR: {decision.var:.2%}")
print(f"5-day 95% CVaR: {decision.cvar:.2%}")
print(f"Half-Kelly position: {decision.kelly * 0.5:.2%}")
print(f"Probability of profit: {decision.prob_profit:.1%}")

# ============================================================
# STAGE 5: Backtesting
# ============================================================
print("\nStage 5: Backtesting...")

backtest = tpdf.backtest(
    data=returns,
    distribution=best_dist,
    lookback=252,
    weighting=tpdf.weights.EMA(halflife=20),
    dynamics={
        "mu": tpdf.dynamics.RandomWalk(),
        "delta": tpdf.dynamics.GARCH(1, 1),
        "alpha": tpdf.dynamics.MeanReverting(),
        "beta": tpdf.dynamics.AR(1),
    },
    forecast_horizon=5,
    alpha=0.05,
    start_date="2022-01-01",
)

backtest.run()
print(backtest.summary())
backtest.plot()

# ============================================================
# Final Output
# ============================================================
print("\n" + "="*60)
print("TRADING RECOMMENDATION")
print("="*60)
print(f"Distribution: {best_dist}")
print(f"Current params: {temporal.current_params}")
print(f"5-day forecast horizon")
print(f"")
print(f"  Expected return: {decision.expected_return:+.2%}")
print(f"  VaR (95%):       {decision.var:.2%} [{decision.var_ci[0]:.2%}, {decision.var_ci[1]:.2%}]")
print(f"  CVaR (95%):      {decision.cvar:.2%}")
print(f"  Kelly fraction:  {decision.kelly:.2f} [{decision.kelly_ci[0]:.2f}, {decision.kelly_ci[1]:.2f}]")
print(f"  P(profit):       {decision.prob_profit:.1%}")
print(f"")
print(f"Recommended position: {decision.kelly * 0.5:.1%} (half-Kelly)")
print(f"Backtest status: {backtest.status}")
```

---

## Implementation: Build vs. Import

| Component | Build | Import |
|-----------|-------|--------|
| **Stage 1** | | |
| Distribution fitting | Unified interface, NIG optimizer | `scipy.stats` |
| CRPS / Log Score | Wrapper | `properscoring` |
| Selection logic | ✓ Full implementation | |
| Significance testing | ✓ Full implementation | |
| **Stage 2** | | |
| Unconditional fit | NIG MLE, unified interface | `scipy.stats` |
| Conditional fit | Wrapper class | `xgboostlss`, `ngboost` |
| SHAP integration | Wrapper | `shap` |
| **Stage 3** | | |
| Parameter tracking | ✓ Full implementation | |
| Weighting schemes | ✓ Full implementation | |
| Dynamics models | ✓ Full implementation | `arch` (optional, for GARCH fitting) |
| Forward projection | ✓ Full implementation | |
| Predictive distribution | ✓ Full implementation | |
| **Stage 4** | | |
| VaR / CVaR | ✓ Full implementation | |
| Kelly criterion | ✓ Full implementation | |
| Probability queries | ✓ Full implementation | |
| **Stage 5** | | |
| Rolling backtest | ✓ Full implementation | |
| Kupiec / Christoffersen | ✓ Full implementation | |
| PIT analysis | ✓ Full implementation | |
| Visualization | ✓ Full implementation | `matplotlib`, `plotly` |

---

## File Structure

```
temporalpdf/
├── __init__.py                    # Main API exports
├── version.py
│
├── discovery/
│   ├── __init__.py
│   ├── selection.py               # select_best_distribution
│   ├── scoring.py                 # CRPS/LogScore wrappers
│   ├── significance.py            # Statistical tests
│   └── diagnostics.py             # Q-Q plots, tail comparison
│
├── distributions/
│   ├── __init__.py
│   ├── base.py                    # Protocol/ABC
│   ├── scipy_wrapper.py           # Wrap scipy distributions
│   ├── nig.py                     # NIG with custom MLE
│   └── generalized_laplace.py     # Custom distribution
│
├── fitting/
│   ├── __init__.py
│   ├── unconditional.py           # fit() function
│   └── conditional.py             # ConditionalModel class
│
├── temporal/
│   ├── __init__.py
│   ├── model.py                   # TemporalModel class
│   ├── tracking.py                # ParameterTracker
│   ├── weights.py                 # SMA, EMA, WMA, etc.
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── base.py                # DynamicsModel ABC
│   │   ├── constant.py
│   │   ├── random_walk.py
│   │   ├── ar.py
│   │   ├── mean_reverting.py
│   │   ├── garch.py
│   │   ├── egarch.py
│   │   ├── har.py
│   │   ├── regime_switching.py
│   │   └── state_space.py
│   ├── projection.py              # Forward projection
│   └── predictive.py              # PredictiveDistribution
│
├── decision/
│   ├── __init__.py
│   ├── forecast.py                # Forecast class
│   ├── risk.py                    # VaR, CVaR
│   ├── kelly.py                   # Kelly criterion
│   ├── probability.py             # Probability queries
│   └── summary.py                 # DecisionSummary
│
├── backtest/
│   ├── __init__.py
│   ├── runner.py                  # Backtest class
│   ├── tests.py                   # Kupiec, Christoffersen, etc.
│   ├── comparison.py              # Compare backtests
│   └── visualization.py           # Backtest plots
│
├── visualization/
│   ├── __init__.py
│   ├── distributions.py           # PDF/CDF plots
│   ├── evolution.py               # 3D surfaces, heatmaps
│   ├── diagnostics.py             # Q-Q, ACF, etc.
│   └── interactive.py             # Plotly interactive
│
└── utils/
    ├── __init__.py
    ├── validation.py              # Input validation
    └── numerical.py               # Numerical helpers
```

---

## Dependencies

### Required
- `numpy >= 1.21`
- `pandas >= 1.3`
- `scipy >= 1.7`
- `properscoring >= 0.1`
- `matplotlib >= 3.4`

### Optional
- `arch >= 5.0` (for GARCH model fitting helper)
- `xgboostlss >= 0.4` (for conditional models)
- `ngboost >= 0.3` (alternative conditional backend)
- `shap >= 0.40` (for explainability)
- `plotly >= 5.0` (for interactive plots)
- `numba >= 0.54` (for performance)

---

## Migration from temporalpdf v1

Key changes:
1. Distribution implementations replaced with scipy wrappers
2. `TemporalModel` replaces manual parameter evolution
3. Stage 3 fully expanded with tracking, weighting, dynamics
4. All decisions now support parameter uncertainty
5. Backtest framework expanded with more tests

```python
# v1
from temporalpdf import NIG, fit_nig, var, kelly_fraction

params = fit_nig(returns)
var_95 = var(NIG(), params, alpha=0.05, t=5)
kelly = kelly_fraction(NIG(), params, t=5)

# v2
import temporalpdf as tpdf

temporal = tpdf.TemporalModel(distribution="nig", ...)
temporal.fit(returns)
decision = temporal.decision(t=5, alpha=0.05)
var_95 = decision.var
kelly = decision.kelly
```

---

## Roadmap

### Phase 1: Core Refactor
- [ ] Slim distributions to scipy wrappers
- [ ] Implement TemporalModel class
- [ ] Implement all weighting schemes
- [ ] Implement core dynamics models (Constant, RW, AR, MeanReverting, GARCH)

### Phase 2: Decision Layer
- [ ] Refactor decision utilities to work with TemporalModel
- [ ] Add parameter uncertainty integration
- [ ] Add confidence intervals to all outputs

### Phase 3: Backtesting
- [ ] Expand backtest framework
- [ ] Add Christoffersen, DQ, Berkowitz tests
- [ ] Add comparison framework

### Phase 4: Conditional Models
- [ ] XGBoostLSS wrapper
- [ ] NGBoost wrapper
- [ ] SHAP integration

### Phase 5: Polish
- [ ] Documentation
- [ ] Examples
- [ ] Performance optimization
- [ ] Test coverage

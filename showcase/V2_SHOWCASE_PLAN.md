# temporalpdf V2 Showcase Plan

## Current State

### Existing Showcase Files
| File | Description | API Version |
|------|-------------|-------------|
| `stock_trading_comparison.ipynb` | S&P 500 trading strategy comparison | V1 (basic NIG, VaR, Kelly) |
| `m5_comparison.py` | M5 retail pipeline comparison | V1 (uses scipy.stats, not tpdf) |
| `m5_eda.py` | M5 exploratory data analysis | N/A |

### Problem
The existing showcases don't demonstrate V2's key differentiators:
- ParameterTracker for rolling window estimation
- Weighting schemes (EMA, PowerDecay, etc.)
- Dynamics models (MeanReverting, GARCH, etc.)
- TemporalModel orchestration
- Projection of parameters through time
- PredictiveDistribution with parameter uncertainty
- DecisionSummary with confidence intervals
- Discovery with proper scoring rules
- Backtest with statistical tests (Kupiec, Christoffersen)

---

## V2 Showcase: Three-Part Structure

### Part 1: Distribution Discovery (Discovery Module)
**File**: `v2_01_discovery.py` (or `.ipynb`)

**Narrative**: "Which distribution family best describes my returns?"

**Flow**:
```python
import temporalpdf as tpdf
import numpy as np

# Load data (S&P 500 returns or M5 sales changes)
data = load_returns()

# V2 Discovery: Cross-validated comparison with proper scoring
result = tpdf.discover(
    data,
    candidates=["normal", "student_t", "nig"],
    cv_folds=5,
    scoring="crps"
)

print(result.summary())
# Best distribution: nig (confidence: high)
# CRPS scores: normal=0.0123, student_t=0.0098, nig=0.0089
# p-value vs runner-up: 0.003 (significant)

# Use the discovered distribution
nig = tpdf.NIG()
best_params = result.best_params
```

**Key V2 Features Demonstrated**:
- `tpdf.discover()` function
- `DiscoveryResult` with scores, confidence, pairwise p-values
- CRPS as proper scoring rule
- Statistical significance testing

---

### Part 2: Temporal Parameter Evolution (Temporal Module)
**File**: `v2_02_temporal.py` (or `.ipynb`)

**Narrative**: "How do distribution parameters change over time? What will they be tomorrow?"

**Flow**:
```python
# Build a TemporalModel
model = tpdf.TemporalModel(
    distribution="nig",
    tracking=tpdf.ParameterTracker(
        distribution="nig",
        window=60,
        step=1
    ),
    weighting=tpdf.EMA(halflife=20),  # Recent data matters more
    dynamics={
        "delta": tpdf.MeanReverting(),  # Volatility mean-reverts
        "mu": tpdf.RandomWalk(),        # Drift random walks
    }
)

# Fit to historical data
model.fit(returns)

# See how parameters evolved
param_df = model.tracking.fit(returns)
# Columns: date, mu, delta, alpha, beta

# Project parameters forward
projection = model.project(horizon=30, n_paths=1000)
# projection.mean(t=10)  -> expected parameters at t=10
# projection.quantile(0.95, t=10) -> 95th percentile parameters

# Get predictive distribution that integrates over parameter uncertainty
predictive = model.predictive(t=10, n_samples=5000)
var_95 = predictive.var(alpha=0.05)
cvar_95 = predictive.cvar(alpha=0.05)
```

**Key V2 Features Demonstrated**:
- `TemporalModel` class
- `ParameterTracker` for rolling estimation
- Weighting schemes (`EMA`, `PowerDecay`, etc.)
- Dynamics models (`MeanReverting`, `RandomWalk`, `GARCH`)
- `Projection` class
- `PredictiveDistribution` with integrated uncertainty

**Visualization**:
- 3D surface: time × parameter value × density
- Parameter path confidence bands
- Predicted vs realized parameter evolution

---

### Part 3: Decision Under Uncertainty (Decision + Backtest)
**File**: `v2_03_decision_backtest.py` (or `.ipynb`)

**Narrative**: "How do I make decisions with these distributions, and do they work?"

**Flow**:
```python
# Get decision summary with confidence intervals
decision = model.decision(t=5, alpha=0.05)

print(f"VaR(95%): {decision.var.value:.3f}%")
print(f"  CI: [{decision.var.confidence_interval[0]:.3f}, {decision.var.confidence_interval[1]:.3f}]")
print(f"CVaR(95%): {decision.cvar.value:.3f}%")
print(f"Kelly: {decision.kelly.value:.3f}")
print(f"P(profit): {decision.prob_profit:.1%}")

# Backtest VaR predictions
backtest = tpdf.Backtest(
    model=model,
    data=test_returns,
    horizon=1
)
results = backtest.run(alpha=0.05)

# Statistical validation
print(f"Violations: {results.violations} / {len(test_returns)} ({results.violation_rate:.1%})")
print(f"Kupiec p-value: {results.kupiec_pvalue:.3f}")  # Test if violation rate matches alpha
print(f"Christoffersen p-value: {results.christoffersen_pvalue:.3f}")  # Test for clustering
print(f"VaR is valid: {results.is_valid}")
```

**Key V2 Features Demonstrated**:
- `DecisionSummary` with `RiskMetric` values
- Confidence intervals on all risk metrics
- `Backtest` class
- Kupiec test for unconditional coverage
- Christoffersen test for independence
- Validation framework

---

## Showcase Notebook: Full Integration

**File**: `v2_complete_showcase.ipynb`

**Purpose**: Single notebook that ties all three parts together into one coherent workflow.

**Structure**:
```
1. Introduction
   - What is temporalpdf?
   - Pipeline 1 vs Pipeline 2 (core concept)

2. Discovery Phase
   - Load financial returns data
   - Compare distributions with tpdf.discover()
   - Select best distribution

3. Temporal Modeling
   - Track parameters through time
   - Fit dynamics models
   - Project future parameters

4. Decision Making
   - VaR, CVaR, Kelly with confidence intervals
   - Trading strategy: filter by risk metrics

5. Backtesting
   - Run VaR backtest
   - Validate with Kupiec/Christoffersen
   - Compare to naive approach

6. Results
   - Strategy comparison (point vs distribution)
   - Risk metrics table with CIs
   - Backtest validation
```

---

## Data Options

### Option A: S&P 500 Returns (existing)
- **Pros**: Already have data, simple narrative
- **Cons**: Generic, everyone uses this

### Option B: M5 Retail Demand
- **Pros**: Real dataset, uncertainty competition existed
- **Cons**: Intermittent demand (many zeros) complicates things

### Option C: Cryptocurrency Returns
- **Pros**: Heavy tails showcase NIG well, interesting
- **Cons**: May seem "crypto bro"

**Recommendation**: Option A (S&P 500) for the complete showcase, with Option B as a supplementary example showing different use case.

---

## Key Visualizations

1. **Discovery Panel**
   - Distribution PDFs overlaid on histogram
   - CRPS scores bar chart
   - QQ plots for each candidate

2. **Temporal Evolution**
   - Parameter time series with confidence bands
   - 3D surface: time × return × density
   - Forecast fan chart

3. **Decision Dashboard**
   - VaR/CVaR over time
   - Kelly fraction evolution
   - Probability heatmap (P(loss > x) vs time)

4. **Backtest Validation**
   - VaR breach plot (returns with VaR threshold)
   - Violation clustering analysis
   - Coverage plot over time

---

## Implementation Priority

| Priority | Task | Estimated Effort |
|----------|------|------------------|
| 1 | `v2_complete_showcase.ipynb` | Main deliverable |
| 2 | `v2_01_discovery.py` | Standalone module demo |
| 3 | `v2_02_temporal.py` | Standalone module demo |
| 4 | `v2_03_decision_backtest.py` | Standalone module demo |
| 5 | Update `stock_trading_comparison.ipynb` to V2 | Upgrade existing |
| 6 | Update `m5_comparison.py` to use tpdf V2 | Upgrade existing |

---

## Success Criteria

1. **Complete showcase notebook demonstrates entire V2 API flow**:
   - Discovery → TemporalModel → Projection → Decision → Backtest

2. **Each V2 feature is shown at least once**:
   - [ ] `discover()` with `DiscoveryResult`
   - [ ] `ParameterTracker`
   - [ ] At least 2 weighting schemes (e.g., EMA, PowerDecay)
   - [ ] At least 2 dynamics models (e.g., MeanReverting, GARCH)
   - [ ] `TemporalModel.fit()`, `.project()`, `.predictive()`, `.decision()`
   - [ ] `Projection` with `.mean()`, `.quantile()`
   - [ ] `PredictiveDistribution` with `.var()`, `.cvar()`
   - [ ] `DecisionSummary` with `RiskMetric` confidence intervals
   - [ ] `Backtest` with Kupiec and Christoffersen tests
   - [ ] `*_with_ci` functions (`var_with_ci`, `cvar_with_ci`, `kelly_with_ci`)

3. **Visualizations are publication-quality**:
   - Clean, informative, no excessive decoration
   - Proper axis labels, legends, titles
   - Saved as PNGs for README inclusion

4. **Narrative is clear**:
   - Reader understands WHY distributions matter (not just HOW)
   - Comparison to point prediction shows concrete benefit
   - Statistical validation (backtest) proves the approach works

---

## Next Steps

1. Create `v2_complete_showcase.ipynb` as the main deliverable
2. Ensure it runs end-to-end without errors
3. Generate all visualizations
4. Update `showcase/README.md` to point to new V2 showcase
5. (Optional) Create standalone module demos

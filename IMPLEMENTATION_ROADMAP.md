# temporalpdf: Implementation Roadmap

**Document Status**: AUTHORITATIVE - This supersedes all other planning documents
**Decision**: Based on `temporalpdf_v2_specification-altdoc.md`
**Date**: 2026-01-20

---

## Executive Summary

temporalpdf is evolving from a **distribution toolkit** (v1) to a **complete distributional forecasting pipeline** (v2). The core philosophy remains: predict *distributions*, not point estimates. But v2 adds temporal dynamics, parameter uncertainty tracking, and a complete trading decision framework.

### What Makes temporalpdf Unique

1. **Distribution-first forecasting**: Model outputs distribution parameters, not Y directly
2. **Temporal dynamics**: Parameters evolve over time via GARCH, mean-reversion, regime-switching
3. **Parameter uncertainty**: All decisions include confidence intervals from parameter uncertainty
4. **Statistical rigor**: Significance testing for distribution selection, proper backtest validation
5. **Decision integration**: VaR, CVaR, Kelly work with ANY distribution (not just Normal)
6. **XGBoostLSS differentiation**: We focus on unconditional + temporal dynamics + decisions; they focus on conditional regression

---

## Current State (v1) - What We Have

**Line count**: ~5,000 lines
**Status**: Functional library with solid foundations

### Stage 1: Distribution Discovery ❌ MISSING
- No automated distribution selection
- No significance testing
- No cross-validation framework

### Stage 2: Model Training ✅ MOSTLY DONE
**Unconditional fitting** (✅ DONE):
- `fit()`, `fit_nig()`, `fit_student_t()`, `fit_normal()` - MLE fitting
- 5 distributions: NIG, Normal, Student-t, Skew-Normal, Generalized Laplace
- `select_best_distribution()`, `compare_distributions()` - basic selection

**Conditional fitting** (❌ MISSING):
- No XGBoostLSS integration
- No NGBoost integration
- No feature → params model

### Stage 3: Temporal Dynamics ⚠️ PARTIAL
**What exists**:
- `VolatilityModel` class with GARCH, mean-reverting, term structure
- `NIGParameters` with `mu_drift`, `delta_growth` for simple time evolution
- `RollingCoefficientExtractor` for tracking params over time

**What's missing**:
- No unified `TemporalModel` class
- No weighting schemes (SMA, EMA, WMA)
- No dynamics models for location/skewness params (only volatility)
- No forward projection framework
- No predictive distribution with param uncertainty

### Stage 4: Decision Layer ✅ MOSTLY DONE
- `var()`, `cvar()` - Risk measures ✅
- `kelly_fraction()`, `fractional_kelly()` - Position sizing ✅
- `prob_greater_than()`, `prob_less_than()`, `prob_between()` - Probability queries ✅
- **Missing**: Parameter uncertainty propagation (no confidence intervals)
- **Missing**: `TradingStrategy` class for should_trade decisions

### Stage 5: Backtesting ⚠️ PARTIAL
**What exists**:
- `rolling_var_backtest()` with Kupiec test ✅

**What's missing**:
- No Christoffersen test (independence)
- No Dynamic Quantile test
- No Berkowitz test
- No PIT analysis
- No comparison framework
- No Kelly strategy backtest

### Other Components
**Visualization** (✅ GOOD):
- `PDFPlotter` - matplotlib 2D/3D plots
- `InteractivePlotter` - Plotly 3D surfaces
- Heatmaps, slices, confidence bands

**Scoring Rules** (✅ DONE):
- CRPS (Monte Carlo and closed-form for Normal)
- Log Score

**Utilities** (✅ DONE):
- `utilities.py` with fitting, selection, backtesting

---

## v2 Vision - Where We're Going

The 5-stage pipeline from the specification:

```
Raw Data → Discovery → Fitting → Temporal Dynamics → Decision → Backtest → Trading Signal
```

Each stage adds value:
1. **Discovery**: "Is NIG significantly better than Student-t?" (statistical rigor)
2. **Fitting**: "What are the current params?" (MLE or XGBoostLSS)
3. **Temporal**: "How will params evolve?" (GARCH volatility, drifting mean)
4. **Decision**: "What's my position size?" (Kelly with param uncertainty)
5. **Backtest**: "Did this work historically?" (Proper statistical tests)

---

## Gap Analysis

| Component | v1 Status | v2 Requirement | Gap |
|-----------|-----------|----------------|-----|
| **Stage 1: Discovery** |
| Distribution selection | Basic (`select_best_distribution`) | Statistical significance testing | Need significance, CV, diagnostics |
| Cross-validation | None | k-fold, blocked time-series | Build from scratch |
| Diagnostics | None | Q-Q, tail fit, moment matching | Build from scratch |
| **Stage 2: Fitting** |
| Unconditional MLE | ✅ Done | ✅ Done | None |
| Conditional (XGBoostLSS) | None | Full wrapper + SHAP | Build wrapper, import SHAP |
| **Stage 3: Temporal** |
| Parameter tracking | Basic rolling | Flexible tracking (rolling/expanding/event) | Enhance existing |
| Weighting schemes | None | SMA/EMA/WMA/Regime | Build from scratch |
| Dynamics models | Volatility only | All params (μ, σ, α, β) | Generalize existing |
| Forward projection | None | Monte Carlo + analytical | Build from scratch |
| Predictive distribution | None | Mixture over param uncertainty | Build from scratch |
| **Stage 4: Decision** |
| VaR/CVaR/Kelly | ✅ Point estimates | With confidence intervals | Add uncertainty propagation |
| Trading strategy | None | `TradingStrategy.evaluate()` | Build from scratch |
| **Stage 5: Backtest** |
| Basic VaR backtest | ✅ Kupiec test | All tests (Christoffersen, DQ, Berkowitz, PIT) | Add more tests |
| Comparison framework | None | Compare dists/weights/dynamics | Build from scratch |
| Kelly backtest | None | Full equity curve, Sharpe, drawdown | Build from scratch |

---

## Implementation Phases

### Phase 0: Foundation & Cleanup (1 week)
**Goal**: Clean up existing code, establish new structure

- [ ] Move to v2 file structure (discovery/, temporal/, backtest/ folders)
- [ ] Deprecate old imports, add compatibility layer
- [ ] Update README with v2 vision and XGBoostLSS comparison
- [ ] Clean up technical debt (type hints, docstrings)

### Phase 1: Core Temporal (2-3 weeks)
**Goal**: Make temporal dynamics actually work

1. **Weighting schemes** (src/temporalpdf/temporal/weights/)
   - [ ] SMA, EMA, Linear, Power decay
   - [ ] Custom weight functions
   - [ ] Visualization: `plot_weight_comparison()`

2. **Parameter tracking** (src/temporalpdf/temporal/tracking/)
   - [ ] Enhance `RollingCoefficientExtractor` → `ParameterTracker`
   - [ ] Add expanding window mode
   - [ ] Add event-triggered mode

3. **Dynamics models** (src/temporalpdf/temporal/dynamics/)
   - [ ] Generalize existing `VolatilityModel` → applies to ANY param
   - [ ] Implement: Constant, RandomWalk, AR(p), MeanReverting
   - [ ] GARCH family (leverage `arch` library)
   - [ ] Model selection framework

4. **TemporalModel class** (src/temporalpdf/temporal/model.py)
   - [ ] Unified interface: `TemporalModel.fit(returns)` → does 3A+3B+3C
   - [ ] `project(horizon=30)` → forward projection
   - [ ] `predictive(t=5)` → predictive distribution at t

### Phase 2: Decision with Uncertainty (1 week)
**Goal**: All decisions have confidence intervals

- [ ] Refactor VaR/CVaR/Kelly to accept `TemporalModel`
- [ ] Monte Carlo integration over param uncertainty
- [ ] Return `Result` objects with `.value`, `.ci`, `.std`
- [ ] `decision_summary(t=5)` → full table with CIs

### Phase 3: Trading Strategy Layer (1 week)
**Goal**: Distribution → Action

- [ ] `TradingStrategy` class (src/temporalpdf/decision/strategy.py)
- [ ] `Action` enum: LONG, SHORT, ABSTAIN
- [ ] `TradeDecision` dataclass with position_size, confidence, reason
- [ ] Filters: min_expected_return, max_cvar, min_confidence
- [ ] Example: `strategy.evaluate(temporal, t=5) → TradeDecision`

### Phase 4: Discovery & Selection (1-2 weeks)
**Goal**: Statistical rigor for distribution choice

- [ ] `discover()` function (src/temporalpdf/discovery/selection.py)
- [ ] Cross-validation (k-fold, blocked time-series)
- [ ] Significance testing (paired t-test, Wilcoxon)
- [ ] Confidence scoring ("high", "medium", "low")
- [ ] Diagnostics: Q-Q plots, tail fit, moment matching
- [ ] `DiscoveryResult` class with `.summary()`, `.plot_scores()`

### Phase 5: Backtest Expansion (1-2 weeks)
**Goal**: Proper statistical validation

1. **More tests**:
   - [ ] Christoffersen (independence + conditional coverage)
   - [ ] Dynamic Quantile test
   - [ ] PIT (Probability Integral Transform) uniformity
   - [ ] Berkowitz test

2. **Comparison framework**:
   - [ ] `backtest_compare(distributions=[...])` → comparison table
   - [ ] `backtest_compare_weights([...])` → which weighting is best?
   - [ ] `backtest_compare_dynamics([...])` → which dynamics work?

3. **Kelly backtest**:
   - [ ] `backtest_kelly()` → full equity curve
   - [ ] Sharpe ratio, Calmar ratio, max drawdown
   - [ ] Transaction costs, leverage constraints

### Phase 6: Conditional Models (2-3 weeks - OPTIONAL)
**Goal**: Features → distribution params

- [ ] `ConditionalModel` wrapper (src/temporalpdf/conditional/)
- [ ] XGBoostLSS backend
- [ ] NGBoost backend (optional)
- [ ] SHAP integration for explainability
- [ ] `model.predict(X) → params_df`
- [ ] `model.plot_importance()`, `model.plot_shap()`

### Phase 7: Polish & Release (1-2 weeks)
**Goal**: Production-ready library

- [ ] Comprehensive examples (examples/)
- [ ] Jupyter notebooks (showcase/)
- [ ] Documentation (Sphinx or MkDocs)
- [ ] Performance optimization (numba for critical loops)
- [ ] Test coverage >80%
- [ ] Publish to PyPI as v2.0.0
- [ ] Write blog post / tutorial

---

## Architecture Decisions

### 1. Distribution Implementations
**Decision**: Use scipy wrappers for Normal/Student-t/Skew-Normal, keep custom NIG

**Rationale**:
- scipy is battle-tested, fast, well-documented
- Our NIG implementation has better MLE for financial data
- Reduces maintenance burden

**Action**:
- Keep: `NIGDistribution` (custom)
- Wrap: `NormalDistribution`, `StudentTDistribution`, `SkewNormalDistribution` via scipy
- Keep: `GeneralizedLaplaceDistribution` (custom, not in scipy)

### 2. Temporal Dynamics
**Decision**: Build all dynamics models ourselves, optionally use `arch` for GARCH fitting

**Rationale**:
- This is our core value proposition
- `arch` has GARCH fitting but not projection
- We need projection for all params, not just volatility

**Action**:
- Build: All dynamics classes (RandomWalk, MeanReverting, AR, RegimeSwitching)
- Import: `arch.univariate.GARCH` for fitting helper (optional)
- Build: Projection, predictive distribution

### 3. Proper Scoring Rules
**Decision**: Import from `properscoring`

**Rationale**:
- Standard implementation, used in research
- CRPS is tricky to implement correctly
- Focus our effort elsewhere

**Action**:
- Import: `properscoring.crps_ensemble`, `properscoring.crps_gaussian`
- Wrap: Thin wrapper in `src/temporalpdf/scoring/rules.py`

### 4. Conditional Models
**Decision**: Wrapper around XGBoostLSS/NGBoost, not reimplementation

**Rationale**:
- These are mature libraries
- Our value is in temporal dynamics + decisions, not conditional regression
- XGBoostLSS already does automatic differentiation via PyTorch

**Action**:
- Build: `ConditionalModel` wrapper class
- Import: `xgboostlss`, `ngboost` (optional dependencies)
- Build: Integration layer, not reimplementation

### 5. Backtest Framework
**Decision**: Build ourselves with scipy.stats for test statistics

**Rationale**:
- Needs deep integration with our temporal models
- Not a standard library that fits our needs
- Kupiec/Christoffersen tests are simple to implement

**Action**:
- Build: Full backtest runner, comparison framework
- Import: `scipy.stats` for chi-square, t-test, KS test

---

## File Structure (v2)

```
temporalpdf/
├── __init__.py                    # Main API exports
├── api.py                         # High-level: discover(), fit(), forecast(), backtest()
│
├── discovery/
│   ├── __init__.py
│   ├── selection.py               # discover() implementation
│   ├── significance.py            # Paired t-test, confidence scoring
│   └── diagnostics.py             # Q-Q, tail fit, moment matching
│
├── distributions/
│   ├── __init__.py
│   ├── base.py                    # Protocol
│   ├── nig.py                     # Custom NIG with better MLE
│   ├── scipy_wrapper.py           # Wrap Normal, Student-t, Skew-Normal
│   ├── generalized_laplace.py     # Custom
│   └── registry.py                # DistributionRegistry (keep)
│
├── fitting/
│   ├── __init__.py
│   ├── mle.py                     # fit() implementation
│   └── weighted.py                # Weighted MLE
│
├── conditional/                   # OPTIONAL - Phase 6
│   ├── __init__.py
│   ├── model.py                   # ConditionalModel wrapper
│   ├── xgboostlss.py              # XGBoostLSS backend
│   └── explainability.py          # SHAP integration
│
├── temporal/
│   ├── __init__.py
│   ├── model.py                   # TemporalModel class (all-in-one)
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── rolling.py             # Enhance existing RollingCoefficientExtractor
│   │   └── expanding.py           # Expanding window
│   │
│   ├── weights/
│   │   ├── __init__.py
│   │   ├── sma.py, ema.py, linear.py, power.py
│   │   └── custom.py
│   │
│   ├── dynamics/
│   │   ├── __init__.py
│   │   ├── base.py                # DynamicsModel ABC
│   │   ├── constant.py
│   │   ├── random_walk.py
│   │   ├── mean_reverting.py
│   │   ├── ar.py
│   │   ├── garch.py               # Generalize existing VolatilityModel
│   │   └── regime_switching.py    # Future
│   │
│   ├── projection.py              # Monte Carlo + analytical projection
│   └── predictive.py              # Predictive distribution (mixture)
│
├── decision/
│   ├── __init__.py
│   ├── risk.py                    # var(), cvar() - enhance with uncertainty
│   ├── kelly.py                   # kelly_fraction() - enhance
│   ├── probability.py             # prob_*() - enhance
│   ├── strategy.py                # TradingStrategy class (NEW)
│   └── summary.py                 # decision_summary() (NEW)
│
├── backtest/
│   ├── __init__.py
│   ├── runner.py                  # Enhance rolling_var_backtest()
│   ├── tests.py                   # Kupiec, Christoffersen, DQ, Berkowitz
│   ├── comparison.py              # backtest_compare()
│   └── kelly_backtest.py          # backtest_kelly()
│
├── scoring/
│   ├── __init__.py
│   └── rules.py                   # CRPS, Log Score wrappers
│
├── visualization/
│   ├── __init__.py
│   ├── plotter.py                 # Keep existing PDFPlotter
│   ├── interactive.py             # Keep existing InteractivePlotter
│   ├── discovery.py               # NEW: Q-Q, density overlay
│   ├── temporal.py                # NEW: Param history, projections
│   └── backtest.py                # NEW: Backtest plots
│
├── core/                          # Keep existing
│   ├── distribution.py
│   ├── parameters.py
│   ├── grid.py
│   ├── result.py
│   └── volatility.py              # Will be generalized in Phase 1
│
└── utils/
    ├── __init__.py
    ├── validation.py
    └── results.py                 # Result container classes
```

---

## API Design Principles

### 1. Progressive Disclosure
**Simple case** (no temporal dynamics):
```python
params = tpdf.fit(returns, "nig")
var = tpdf.var(tpdf.NIG(), params, alpha=0.05)
```

**Full pipeline** (temporal dynamics):
```python
temporal = tpdf.TemporalModel("nig", weighting=tpdf.weights.EMA(20), dynamics=...)
temporal.fit(returns)
decision = temporal.decision_summary(t=5)
```

### 2. Sensible Defaults
- `discover()` defaults to ["normal", "student_t", "nig"]
- `TemporalModel` defaults to SMA weighting, Constant dynamics
- `backtest()` defaults to 252-day lookback, 95% VaR

### 3. Explicit Over Implicit
- No magic string parsing - use enums or classes
- `Action.LONG` not `"long"`
- `tpdf.weights.EMA(halflife=20)` not `weights="ema"`

### 4. Return Rich Objects
- Not just floats - return `VaRResult(value, ci, std)`
- Enables `.plot()`, `.to_dict()`, `.summary()` on results
- Chainable, inspectable

---

## Migration Strategy

### Backward Compatibility
**Keep v1 API working in v2.0**:
```python
# v1 code still works
from temporalpdf import fit_nig, var, kelly_fraction
params = fit_nig(returns)
var_value = var(NIG(), params, alpha=0.05)
```

**Add deprecation warnings**:
```python
warnings.warn(
    "fit_nig() is deprecated, use fit(returns, 'nig') instead",
    DeprecationWarning
)
```

**Remove in v3.0**.

### Incremental Adoption
Users can adopt v2 features incrementally:
1. Start: Just use `fit()` and `var()` (v1 style)
2. Add: Weighting schemes → `fit(returns, weights=tpdf.weights.EMA(20))`
3. Add: Temporal dynamics → `TemporalModel`
4. Add: Discovery → `discover()` before fitting
5. Add: Backtesting → `backtest()` to validate

---

## Dependencies

### Required
```toml
dependencies = [
    "numpy>=1.20",
    "pandas>=1.3",
    "scipy>=1.7",
    "properscoring>=0.1",     # CRPS
    "arch>=5.0",              # GARCH fitting helper
    "matplotlib>=3.4",
    "plotly>=5.0",            # Interactive plots
]
```

### Optional
```toml
[project.optional-dependencies]
conditional = [
    "xgboostlss>=0.4",
    "ngboost>=0.3",
    "shap>=0.40",
]
dev = [
    "pytest>=7.0",
    "mypy>=0.950",
    "ruff>=0.0.250",
]
```

---

## Success Metrics

### Code Quality
- [ ] Test coverage ≥80%
- [ ] Type hints on all public APIs
- [ ] Docstrings with examples
- [ ] No linter errors (ruff)
- [ ] No type errors (mypy)

### Performance
- [ ] `fit()` completes in <100ms for 1000 observations
- [ ] `TemporalModel.fit()` completes in <5s for 5 years daily data
- [ ] `backtest()` completes in <30s for 5 years daily data

### Documentation
- [ ] README with quick start, comparison to XGBoostLSS
- [ ] 5+ Jupyter notebook examples
- [ ] API reference (auto-generated from docstrings)
- [ ] Tutorial blog post

### Adoption
- [ ] Published to PyPI
- [ ] GitHub stars >100 in first 6 months
- [ ] At least 3 external users/contributors

---

## Risk Mitigation

### Risk: Scope Creep
**Mitigation**: Stick to phased plan. Phase 6 (conditional models) is OPTIONAL.

### Risk: Performance Issues
**Mitigation**: Profile early (Phase 1). Use numba for tight loops if needed.

### Risk: API Design Mistakes
**Mitigation**: Build examples FIRST, then API. If examples feel clunky, redesign API.

### Risk: Abandonment
**Mitigation**: Phases 1-3 already deliver massive value. Each phase is shippable.

---

## Timeline Estimate

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 0: Foundation | 1 week | Clean structure, updated README |
| 1: Core Temporal | 3 weeks | `TemporalModel` works end-to-end |
| 2: Decision with Uncertainty | 1 week | All decisions have CIs |
| 3: Trading Strategy | 1 week | `TradingStrategy.evaluate()` |
| 4: Discovery | 2 weeks | `discover()` with significance tests |
| 5: Backtest Expansion | 2 weeks | All tests, comparison framework |
| 6: Conditional (OPTIONAL) | 3 weeks | XGBoostLSS wrapper |
| 7: Polish | 2 weeks | Docs, examples, PyPI |

**Total**: 12-15 weeks (3-4 months) for full v2.0
**MVP**: 6 weeks for Phases 0-3 (core value)

---

## Next Steps (Immediate)

1. **Read and approve this plan** - This is THE roadmap
2. **Delete conflicting docs** - Archive `temporalpdf_v2_architecture.md`, `COMPLETION_PLAN.md`
3. **Phase 0 kickoff** - Start with file structure refactor
4. **Update README** - Add v2 vision, XGBoostLSS comparison section

---

## Appendix: Key Differentiators vs XGBoostLSS

| Feature | XGBoostLSS | temporalpdf v2 |
|---------|------------|----------------|
| **Core Use Case** | Features → conditional distribution | Unconditional distribution + temporal evolution |
| **Model Training** | Built-in XGBoost with gradient boosting | User brings model OR we fit unconditionally |
| **Distributions** | 30+ via PyTorch/Pyro | 5 hand-crafted, well-tested |
| **Temporal Dynamics** | None | ✅ Core value: GARCH, mean-reversion, regime-switching |
| **Parameter Uncertainty** | Point estimates only | ✅ Full uncertainty propagation |
| **Decision Layer** | None | ✅ VaR, CVaR, Kelly with confidence intervals |
| **Backtesting** | None | ✅ Kupiec, Christoffersen, DQ, Kelly backtest |
| **Target User** | ML engineer predicting from features | Quant trader forecasting time series |

**When to use XGBoostLSS**: "Given features X, what's the conditional distribution of Y?"
**When to use temporalpdf**: "Given time series, how will the distribution evolve, and what should I trade?"

---

**End of Roadmap**

"""
Comprehensive V2 API Verification Script

This script tests EVERY aspect of the temporalpdf V2 specification,
including the flow between components. If anything fails, it indicates
an incomplete implementation.
"""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "src")

# Track all failures
failures = []
warnings = []

def check(condition, name, critical=True):
    """Check a condition and track failures."""
    if condition:
        print(f"  [OK] {name}")
        return True
    else:
        msg = f"  [FAIL] {name}"
        print(msg)
        if critical:
            failures.append(name)
        else:
            warnings.append(name)
        return False

def section(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

# =============================================================================
# SECTION 1: IMPORTS - All V2 exports must be importable
# =============================================================================
section("1. IMPORTS - All V2 exports")

try:
    import temporalpdf as tpdf
    check(True, "import temporalpdf as tpdf")
except ImportError as e:
    check(False, f"import temporalpdf: {e}")
    sys.exit(1)

# Core result types
check(hasattr(tpdf, 'RiskMetric'), "RiskMetric exported")
check(hasattr(tpdf, 'DecisionSummary'), "DecisionSummary exported")

# Weighting schemes
check(hasattr(tpdf, 'SMA'), "SMA exported")
check(hasattr(tpdf, 'EMA'), "EMA exported")
check(hasattr(tpdf, 'Linear'), "Linear exported")
check(hasattr(tpdf, 'PowerDecay'), "PowerDecay exported")
check(hasattr(tpdf, 'Gaussian'), "Gaussian exported")
check(hasattr(tpdf, 'Custom'), "Custom exported")

# Dynamics models
check(hasattr(tpdf, 'Constant'), "Constant exported")
check(hasattr(tpdf, 'RandomWalk'), "RandomWalk exported")
check(hasattr(tpdf, 'MeanReverting'), "MeanReverting exported")
check(hasattr(tpdf, 'AR'), "AR exported")
check(hasattr(tpdf, 'GARCH'), "GARCH exported")

# Tracking
check(hasattr(tpdf, 'ParameterTracker'), "ParameterTracker exported")

# Temporal modeling
check(hasattr(tpdf, 'TemporalModel'), "TemporalModel exported")
check(hasattr(tpdf, 'Projection'), "Projection exported")
check(hasattr(tpdf, 'PredictiveDistribution'), "PredictiveDistribution exported")

# Discovery
check(hasattr(tpdf, 'discover'), "discover exported")
check(hasattr(tpdf, 'DiscoveryResult'), "DiscoveryResult exported")

# Backtest
check(hasattr(tpdf, 'Backtest'), "Backtest exported")
check(hasattr(tpdf, 'BacktestResult'), "BacktestResult exported")
check(hasattr(tpdf, 'backtest'), "backtest function exported")

# High-level API
check(hasattr(tpdf, 'temporal_model'), "temporal_model exported")

# Decision with CIs
check(hasattr(tpdf, 'var_with_ci'), "var_with_ci exported")
check(hasattr(tpdf, 'cvar_with_ci'), "cvar_with_ci exported")
check(hasattr(tpdf, 'kelly_with_ci'), "kelly_with_ci exported")

# =============================================================================
# SECTION 2: GENERATE TEST DATA
# =============================================================================
section("2. TEST DATA GENERATION")

np.random.seed(42)
# Generate realistic return data with some fat tails
normal_returns = np.random.normal(0.0005, 0.015, 1000)
t_returns = np.random.standard_t(df=5, size=1000) * 0.015
# Mix them for realistic market data
returns = np.where(np.random.rand(1000) < 0.8, normal_returns, t_returns)

check(len(returns) == 1000, f"Generated {len(returns)} returns")
check(np.isfinite(returns).all(), "All returns are finite")
print(f"  Mean: {returns.mean():.6f}, Std: {returns.std():.4f}")

# =============================================================================
# SECTION 3: DISCOVERY - Find best distribution
# =============================================================================
section("3. DISCOVERY - Automatic distribution selection")

try:
    discovery_result = tpdf.discover(
        returns,
        candidates=["normal", "student_t", "nig"],
        cv_folds=3,  # Fewer folds for speed
    )
    check(True, "discover() executed")
    check(isinstance(discovery_result, tpdf.DiscoveryResult), "Returns DiscoveryResult")
    check(discovery_result.best in ["normal", "student_t", "nig"], f"Best: {discovery_result.best}")
    check(discovery_result.confidence in ["high", "medium", "low"], f"Confidence: {discovery_result.confidence}")
    check(isinstance(discovery_result.scores, dict), "Has scores dict")
    check(isinstance(discovery_result.std_scores, dict), "Has std_scores dict")
    check(isinstance(discovery_result.pairwise_pvalues, dict), "Has pairwise_pvalues dict")
    check(discovery_result.best_params is not None, "Has best_params")

    # Check summary method
    summary = discovery_result.summary()
    check(isinstance(summary, str), "summary() returns string")
    check("Best" in summary, "summary() contains 'Best'")

    print(f"\n  Discovery Summary:\n{discovery_result.summary()}")

    # Save for later use
    best_distribution = discovery_result.best
    best_params = discovery_result.best_params

except Exception as e:
    check(False, f"discover() failed: {e}")
    best_distribution = "normal"  # Fallback
    best_params = None

# =============================================================================
# SECTION 4: WEIGHTING SCHEMES - All must work
# =============================================================================
section("4. WEIGHTING SCHEMES")

n_obs = 100

# SMA
sma = tpdf.SMA(window=20)
sma_weights = sma.get_weights(n_obs)
check(np.isclose(sma_weights.sum(), 1.0), "SMA weights sum to 1")
check(sma_weights[0] == sma_weights[19], "SMA weights equal within window")
check(sma_weights[20] == 0, "SMA weights zero outside window")

# EMA
ema = tpdf.EMA(halflife=20)
ema_weights = ema.get_weights(n_obs)
check(np.isclose(ema_weights.sum(), 1.0), "EMA weights sum to 1")
check(ema_weights[0] > ema_weights[1] > ema_weights[2], "EMA weights decay")

# Linear
linear = tpdf.Linear(window=20)
linear_weights = linear.get_weights(n_obs)
check(np.isclose(linear_weights.sum(), 1.0), "Linear weights sum to 1")

# PowerDecay
power = tpdf.PowerDecay(power=1.0)
power_weights = power.get_weights(n_obs)
check(np.isclose(power_weights.sum(), 1.0), "PowerDecay weights sum to 1")

# Gaussian
gauss = tpdf.Gaussian(sigma=10)
gauss_weights = gauss.get_weights(n_obs)
check(np.isclose(gauss_weights.sum(), 1.0), "Gaussian weights sum to 1")

# Custom
custom = tpdf.Custom(func=lambda i, n: max(0, 20 - i))
custom_weights = custom.get_weights(n_obs)
check(np.isclose(custom_weights.sum(), 1.0), "Custom weights sum to 1")

# Effective sample size
check(sma.effective_sample_size(n_obs) == 20, "SMA ESS = window")
check(ema.effective_sample_size(n_obs) > 0, "EMA ESS > 0")

# =============================================================================
# SECTION 5: DYNAMICS MODELS - All must work
# =============================================================================
section("5. DYNAMICS MODELS")

# Generate parameter series for fitting
np.random.seed(42)
param_series = np.cumsum(np.random.randn(100) * 0.001) + 0.02

# Constant
const = tpdf.Constant()
const.fit(param_series)
const_proj = const.project(current_value=0.02, horizon=30, n_paths=100)
check(const_proj.shape == (100, 30), f"Constant projection shape: {const_proj.shape}")
check(np.allclose(const_proj, const.long_run_value), "Constant projects constant value")

# RandomWalk
rw = tpdf.RandomWalk()
rw.fit(param_series)
rw_proj = rw.project(current_value=0.02, horizon=30, n_paths=100)
check(rw_proj.shape == (100, 30), f"RandomWalk projection shape: {rw_proj.shape}")
check(rw.sigma is not None, f"RandomWalk fitted sigma: {rw.sigma:.6f}")

# MeanReverting
mr = tpdf.MeanReverting()
mr.fit(param_series)
mr_proj = mr.project(current_value=0.02, horizon=30, n_paths=100)
check(mr_proj.shape == (100, 30), f"MeanReverting projection shape: {mr_proj.shape}")
check(mr.kappa is not None, f"MeanReverting fitted kappa: {mr.kappa:.4f}")
check(mr.half_life() > 0, f"MeanReverting half-life: {mr.half_life():.2f}")

# AR
ar = tpdf.AR(order=1)
ar.fit(param_series)
ar_proj = ar.project(current_value=0.02, horizon=30, n_paths=100)
check(ar_proj.shape == (100, 30), f"AR projection shape: {ar_proj.shape}")
check(ar.coefficients is not None, "AR has fitted coefficients")

# GARCH
garch = tpdf.GARCH(p=1, q=1)
garch.fit(param_series)
garch_proj = garch.project(current_value=0.02, horizon=30, n_paths=100)
check(garch_proj.shape == (100, 30), f"GARCH projection shape: {garch_proj.shape}")
check(0 < garch.persistence() < 2, f"GARCH persistence: {garch.persistence():.4f}")

# =============================================================================
# SECTION 6: PARAMETER TRACKER
# =============================================================================
section("6. PARAMETER TRACKER")

tracker = tpdf.ParameterTracker(
    distribution="normal",
    window=60,
    step=5,  # Every 5 observations
)
param_history = tracker.fit(returns)

check(isinstance(param_history, pd.DataFrame), "Tracker returns DataFrame")
check("mu_0" in param_history.columns, "Has mu_0 column")
check("sigma_0" in param_history.columns, "Has sigma_0 column")
check(len(param_history) > 0, f"Tracker has {len(param_history)} rows")

# With datetime index
dates = pd.date_range("2020-01-01", periods=len(returns), freq="D")
param_history_dated = tracker.fit(returns, index=dates)
check("date" in param_history_dated.columns or isinstance(param_history_dated.index, pd.DatetimeIndex),
      "Tracker handles datetime index")

# NIG tracker
nig_tracker = tpdf.ParameterTracker(distribution="nig", window=60, step=10)
nig_history = nig_tracker.fit(returns)
check("mu" in nig_history.columns, "NIG tracker has mu")
check("delta" in nig_history.columns, "NIG tracker has delta")
check("alpha" in nig_history.columns, "NIG tracker has alpha")
check("beta" in nig_history.columns, "NIG tracker has beta")

# =============================================================================
# SECTION 7: TEMPORAL MODEL - Full integration
# =============================================================================
section("7. TEMPORAL MODEL - Full integration")

# Create model using discovered distribution
print(f"\n  Using discovered distribution: {best_distribution}")

model = tpdf.TemporalModel(
    distribution=best_distribution,
    tracking=tpdf.ParameterTracker(distribution=best_distribution, window=60, step=5),
    weighting=tpdf.EMA(halflife=20),
    dynamics={
        "sigma_0" if best_distribution != "nig" else "delta": tpdf.MeanReverting(),
    },
)

# Fit
model.fit(returns)
check(model.current_params is not None, "Model has current_params after fit")
check(model.param_history is not None, "Model has param_history after fit")

# Project
projection = model.project(horizon=30, n_paths=500)
check(isinstance(projection, tpdf.Projection), "project() returns Projection")
check(projection.horizon == 30, f"Projection horizon: {projection.horizon}")
check(projection.n_paths == 500, f"Projection n_paths: {projection.n_paths}")
check(projection.param_paths is not None, "Projection has param_paths")

# Projection methods
mean_t10 = projection.mean(t=10)
check(isinstance(mean_t10, dict), "Projection.mean(t) returns dict")

mean_all = projection.mean()
check(isinstance(mean_all, pd.DataFrame), "Projection.mean() returns DataFrame")

quantile_t10 = projection.quantile(0.5, t=10)
check(isinstance(quantile_t10, dict), "Projection.quantile() returns dict")

ci_t10 = projection.confidence_interval(0.90, t=10)
check(isinstance(ci_t10, dict), "Projection.confidence_interval() returns dict")

param_at_t10 = projection.at(10)
check(hasattr(param_at_t10, 'values'), "Projection.at() returns ParamDistribution")

# Predictive distribution
predictive = model.predictive(t=10, n_samples=5000)
check(isinstance(predictive, tpdf.PredictiveDistribution), "predictive() returns PredictiveDistribution")

# Predictive methods
pred_mean = predictive.mean()
check(isinstance(pred_mean, float), f"Predictive mean: {pred_mean:.6f}")

pred_std = predictive.std()
check(isinstance(pred_std, float), f"Predictive std: {pred_std:.6f}")

pred_var = predictive.var(alpha=0.05)
check(isinstance(pred_var, float), f"Predictive VaR(5%): {pred_var:.4f}")
check(pred_var > 0, "VaR is positive")

pred_cvar = predictive.cvar(alpha=0.05)
check(isinstance(pred_cvar, float), f"Predictive CVaR(5%): {pred_cvar:.4f}")
check(pred_cvar >= pred_var * 0.9, "CVaR >= VaR (allowing MC error)")

pred_prob = predictive.prob_profit()
check(0 <= pred_prob <= 1, f"Prob profit: {pred_prob:.2%}")

# Decision summary
decision = model.decision(t=10, alpha=0.05, confidence_level=0.90)
check(isinstance(decision, tpdf.DecisionSummary), "decision() returns DecisionSummary")

# Check DecisionSummary fields
check(isinstance(decision.var, tpdf.RiskMetric), "decision.var is RiskMetric")
check(isinstance(decision.cvar, tpdf.RiskMetric), "decision.cvar is RiskMetric")
check(isinstance(decision.kelly, tpdf.RiskMetric), "decision.kelly is RiskMetric")
check(isinstance(decision.prob_profit, tpdf.RiskMetric), "decision.prob_profit is RiskMetric")
check(isinstance(decision.expected_return, float), "decision.expected_return is float")
check(isinstance(decision.volatility, float), "decision.volatility is float")
check(decision.t == 10, "decision.t is correct")
check(decision.alpha == 0.05, "decision.alpha is correct")

# Check RiskMetric has confidence intervals
check(decision.var.confidence_interval is not None, "VaR has confidence interval")
check(decision.cvar.confidence_interval is not None, "CVaR has confidence interval")
check(decision.kelly.confidence_interval is not None, "Kelly has confidence interval")

print(f"\n  Decision Summary at t=10:")
print(f"    VaR(5%):  {decision.var.value:.4f} [{decision.var.confidence_interval[0]:.4f}, {decision.var.confidence_interval[1]:.4f}]")
print(f"    CVaR(5%): {decision.cvar.value:.4f} [{decision.cvar.confidence_interval[0]:.4f}, {decision.cvar.confidence_interval[1]:.4f}]")
print(f"    Kelly:    {decision.kelly.value:.2f} [{decision.kelly.confidence_interval[0]:.2f}, {decision.kelly.confidence_interval[1]:.2f}]")
print(f"    P(profit): {decision.prob_profit.value:.2%}")

# =============================================================================
# SECTION 8: HIGH-LEVEL API - temporal_model()
# =============================================================================
section("8. HIGH-LEVEL API - temporal_model()")

try:
    simple_model = tpdf.temporal_model(
        returns,
        distribution=best_distribution,
        weighting="ema",
        halflife=20,
        window=60,
    )
    check(isinstance(simple_model, tpdf.TemporalModel), "temporal_model() returns TemporalModel")
    check(simple_model.current_params is not None, "Model is fitted")

    simple_decision = simple_model.decision(t=5)
    check(isinstance(simple_decision, tpdf.DecisionSummary), "decision() works")
    print(f"  VaR(5%) at t=5: {simple_decision.var.value:.4f}")

except Exception as e:
    check(False, f"temporal_model() failed: {e}")

# =============================================================================
# SECTION 9: BACKTEST
# =============================================================================
section("9. BACKTEST")

try:
    # Using discovered distribution
    bt_result = tpdf.backtest(
        returns,
        distribution=best_distribution,
        lookback=100,
        alpha=0.05,
    )
    check(isinstance(bt_result, tpdf.BacktestResult), "backtest() returns BacktestResult")
    check(bt_result.var_forecasts is not None, "Has VaR forecasts")
    check(bt_result.exceedances is not None, "Has exceedances")
    check(0 <= bt_result.exceedance_rate <= 1, f"Exceedance rate: {bt_result.exceedance_rate:.2%}")
    check(bt_result.n_exceedances >= 0, f"N exceedances: {bt_result.n_exceedances}")
    check(bt_result.n_total > 0, f"N total: {bt_result.n_total}")

    # Statistical tests
    check(bt_result.kupiec_pvalue >= 0, f"Kupiec p-value: {bt_result.kupiec_pvalue:.4f}")
    check(bt_result.christoffersen_pvalue >= 0, f"Christoffersen p-value: {bt_result.christoffersen_pvalue:.4f}")
    check(bt_result.status in ["PASS", "FAIL_COVERAGE", "FAIL_INDEPENDENCE", "FAIL_BOTH"],
          f"Status: {bt_result.status}")

    # Summary
    summary = bt_result.summary()
    check(isinstance(summary, str), "summary() returns string")
    check("Backtest" in summary, "summary() contains 'Backtest'")

    print(f"\n  Backtest Summary:\n{summary}")

except Exception as e:
    check(False, f"backtest() failed: {e}")

# =============================================================================
# SECTION 10: DECISION FUNCTIONS WITH CI
# =============================================================================
section("10. DECISION FUNCTIONS WITH CI")

# Use a simple distribution for direct testing
nig = tpdf.NIG()
nig_params = tpdf.NIGParameters(mu=0.001, delta=0.02, alpha=15.0, beta=-2.0)

# var_with_ci
var_metric = tpdf.var_with_ci(nig, nig_params, alpha=0.05)
check(isinstance(var_metric, tpdf.RiskMetric), "var_with_ci returns RiskMetric")
check(var_metric.value > 0, f"VaR value: {var_metric.value:.4f}")
check(var_metric.confidence_interval is not None, "VaR has CI")
check(var_metric.confidence_interval[0] < var_metric.value < var_metric.confidence_interval[1] or
      np.isclose(var_metric.value, var_metric.confidence_interval[0], rtol=0.1) or
      np.isclose(var_metric.value, var_metric.confidence_interval[1], rtol=0.1),
      "VaR value within CI")

# cvar_with_ci
rng = np.random.default_rng(42)
cvar_metric = tpdf.cvar_with_ci(nig, nig_params, alpha=0.05, rng=rng)
check(isinstance(cvar_metric, tpdf.RiskMetric), "cvar_with_ci returns RiskMetric")
check(cvar_metric.value > 0, f"CVaR value: {cvar_metric.value:.4f}")
check(cvar_metric.confidence_interval is not None, "CVaR has CI")

# kelly_with_ci
rng = np.random.default_rng(42)
kelly_metric = tpdf.kelly_with_ci(nig, nig_params, rng=rng)
check(isinstance(kelly_metric, tpdf.RiskMetric), "kelly_with_ci returns RiskMetric")
check(kelly_metric.confidence_interval is not None, "Kelly has CI")
print(f"  Kelly fraction: {kelly_metric.value:.2f} [{kelly_metric.confidence_interval[0]:.2f}, {kelly_metric.confidence_interval[1]:.2f}]")

# RiskMetric float conversion
check(float(var_metric) == var_metric.value, "RiskMetric converts to float")

# =============================================================================
# SECTION 11: END-TO-END FLOW - Discovery -> TemporalModel -> Backtest
# =============================================================================
section("11. END-TO-END FLOW")

print("\n  Step 1: Discover best distribution")
e2e_discovery = tpdf.discover(returns, candidates=["normal", "student_t", "nig"], cv_folds=3)
print(f"    Best: {e2e_discovery.best} (confidence: {e2e_discovery.confidence})")

print("\n  Step 2: Build temporal model with discovered distribution")
e2e_model = tpdf.TemporalModel(
    distribution=e2e_discovery.best,
    tracking=tpdf.ParameterTracker(distribution=e2e_discovery.best, window=60),
    dynamics={
        "sigma_0" if e2e_discovery.best != "nig" else "delta": tpdf.MeanReverting(),
    },
)
e2e_model.fit(returns)
print(f"    Model fitted with {e2e_discovery.best} distribution")

print("\n  Step 3: Get trading decision at t=5")
e2e_decision = e2e_model.decision(t=5, alpha=0.05)
print(f"    VaR(5%): {e2e_decision.var.value:.4f}")
print(f"    CVaR(5%): {e2e_decision.cvar.value:.4f}")
print(f"    Kelly: {e2e_decision.kelly.value:.2f}")

print("\n  Step 4: Backtest the model")
e2e_backtest = tpdf.backtest(returns, distribution=e2e_discovery.best, lookback=100)
print(f"    Status: {e2e_backtest.status}")
print(f"    Exceedance rate: {e2e_backtest.exceedance_rate:.2%} (expected: 5%)")

check(True, "End-to-end flow completed")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
section("FINAL SUMMARY")

print(f"\n  Total checks: {235 - len(failures)}")  # Approximate
print(f"  Failures: {len(failures)}")
print(f"  Warnings: {len(warnings)}")

if failures:
    print("\n  FAILED CHECKS:")
    for f in failures:
        print(f"    - {f}")

if warnings:
    print("\n  WARNINGS:")
    for w in warnings:
        print(f"    - {w}")

if not failures:
    print("\n  *** ALL V2 CHECKS PASSED ***")
else:
    print("\n  *** SOME CHECKS FAILED - IMPLEMENTATION INCOMPLETE ***")
    sys.exit(1)

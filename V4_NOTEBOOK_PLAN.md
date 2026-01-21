# V4 Notebook Plan: temporalpdf Comprehensive Showcase

## Overview

**Goal:** Demonstrate temporalpdf's value proposition against traditional quant approaches with rigorous, reproducible tests.

**Data:** Synthetic financial returns with known ground truth (so we can measure accuracy)

**Structure:** 7 sections, ~2000 lines, touches **78% of API reference**

**Location:** `v4_comprehensive_showcase.ipynb` (root)

---

## API Coverage Analysis

| Stage | Functions Used | Coverage |
|-------|----------------|----------|
| 1. Feature Extraction | `extract_calibration_features`, `hill_estimator`, `jarque_bera_stat`, `realized_moments`, `volatility_clustering` | 10/12 (83%) |
| 2. Distribution Fitting | `fit`, `fit_student_t`, `compare_distributions`, `select_best_distribution`, `StudentTParameters`, `NormalParameters` | 8/12 (67%) |
| 3. Model Training | `DistributionalRegressor`, `BarrierModel` | 2/2 (100%) |
| 4. Temporal Dynamics | `TemporalModel`, `ParameterTracker`, `GARCH`, `MeanReverting` | 6/12 (50%) |
| 5. Volatility Models | `garch_forecast`, `mean_reverting` | 2/6 (33%) |
| 6. Risk Metrics | `var`, `cvar`, `kelly_fraction`, `prob_less_than` | 6/11 (55%) |
| 7. Barrier Probability | `barrier_prob_mc`, `barrier_prob_qmc`, `barrier_prob_temporal`, `barrier_prob_analytical_student_t`, `compare_static_vs_temporal` | 7/9 (78%) |
| 8. Scoring Rules | `crps`, `log_score` | 2/3 (67%) |
| 9. Calibration | `ConformalPredictor` | 1/5 (20%) |
| 10-13. Other | `evaluate`, visualization | 3/15 (20%) |

**Total: ~47/87 functions = 54% direct usage, 78% of core functionality**

---

## Section-by-Section Plan

### Section 1: Setup & Data Generation (~100 lines)

**What it does:**
- Generate 3 synthetic datasets with known ground truth:
  1. **Stationary:** Fixed μ=0.0005, σ=0.02, ν=5
  2. **Regime-switching:** 70% low vol (σ=0.01), 30% high vol (σ=0.03)
  3. **Trending volatility:** σ increases from 0.01 to 0.04 over time

**Why synthetic:**
- We know the TRUE parameters
- We can measure if methods recover them
- Reproducible results

**Functions used:**
- `numpy` for data generation
- Sets up ground truth for later comparison

---

### Section 2: The Baseline Approaches (~300 lines)

**What it does:**
Compare 4 traditional quant approaches:

| Approach | Method | Output |
|----------|--------|--------|
| **Baseline 1:** Historical VaR | 5th percentile of rolling window | Single number |
| **Baseline 2:** Point Prediction | XGBoost → predict return directly | Point estimate |
| **Baseline 3:** Post-hoc Uncertainty | XGBoost + fit Normal to residuals | μ, σ (no fat tails) |
| **Baseline 4:** GARCH(1,1) | arch package | σ forecast only |

**What we measure:**
- VaR accuracy (Kupiec test for correct coverage)
- Calibration (do 5% VaR breaches happen 5% of the time?)
- Interval width (narrower = better, given correct coverage)

**Expected results:**
- Historical VaR: Poor in regime changes (uses stale data)
- Point prediction: No uncertainty at all
- Post-hoc Normal: Underestimates tail risk (assumes thin tails)
- GARCH: Good volatility, but no full distribution

---

### Section 3: Pipeline 2 — Distribution Parameter Prediction (~400 lines)

**What it does:**
Show the core temporalpdf workflow:

```
Features → DistributionalRegressor → (μ, σ, ν) → Full Distribution
```

**Functions used:**
- `extract_calibration_features`
- `DistributionalRegressor(distribution="student_t", loss="crps")`
- `fit`, `compare_distributions`
- `crps`, `log_score`

**Comparisons:**

| Test | Pipeline 1 | Pipeline 2 | Expected Winner |
|------|------------|------------|-----------------|
| Parameter recovery | N/A | Predict μ, σ, ν | P2 recovers true params |
| CRPS score | Post-hoc fit | CRPS-trained | P2 by 10-20% |
| Log score | Post-hoc fit | CRPS-trained | P2 by 5-15% |
| Tail calibration | Assumes Normal | Learns ν | P2 much better |

**Expected results:**
- DistributionalRegressor recovers true ν ≈ 5 (±1)
- CRPS 15-25% better than post-hoc Normal fit
- Predicted σ tracks regime changes

---

### Section 4: Risk Metrics Comparison (~300 lines)

**What it does:**
Compare VaR/CVaR accuracy across methods.

**Functions used:**
- `var`, `cvar`
- `prob_less_than`
- `kelly_fraction`

**Test design:**
1. For each method, compute 95% VaR daily
2. Check actual breach rate (should be 5%)
3. Measure conditional loss when breached (CVaR accuracy)

**Comparison table:**

| Method | Target | Metric |
|--------|--------|--------|
| Historical VaR | 5% breaches | Kupiec p-value |
| Normal assumption | 5% breaches | Kupiec p-value |
| Student-t (temporalpdf) | 5% breaches | Kupiec p-value |
| GARCH + Normal | 5% breaches | Kupiec p-value |

**Expected results:**
- Historical: ~7-8% breaches (too many, stale)
- Normal: ~7-9% breaches (fat tails missed)
- Student-t: ~4.5-5.5% breaches (correct)
- GARCH+Normal: ~6% breaches (vol right, tails wrong)

---

### Section 5: Barrier Probability — The Killer Feature (~400 lines)

**What it does:**
Demonstrate barrier probability prediction — something traditional approaches CAN'T do well.

**Scenario:**
"What's the probability my portfolio cumulative return exceeds +5% (or -5%) over the next 20 days?"

**Functions used:**
- `barrier_prob_mc`
- `barrier_prob_qmc`
- `barrier_prob_temporal`
- `barrier_prob_analytical_student_t`
- `compare_static_vs_temporal`
- `BarrierModel`

**Comparison table:**

| Method | How it works | Limitation |
|--------|--------------|------------|
| **Naive MC** | Fixed params, simulate | Ignores vol dynamics |
| **Normal analytical** | Reflection principle | Thin tails |
| **temporalpdf MC** | Fitted Student-t params | Static params |
| **temporalpdf QMC** | Sobol sequences | 2-4x lower variance |
| **temporalpdf Temporal** | GARCH dynamics | Full dynamics |
| **BarrierModel** | End-to-end trained | Best calibration |

**Tests:**
1. **Accuracy test:** Generate 1000 paths with known params, compute true barrier prob, compare estimates
2. **Variance test:** Run each method 100 times, measure standard deviation
3. **Speed test:** Time each method
4. **Calibration test:** Predict barrier prob, check actual hit rate

**Expected results:**

| Method | Accuracy (MAE) | Variance | Speed |
|--------|----------------|----------|-------|
| Naive Normal | 0.08 | Medium | Fast |
| temporalpdf MC | 0.03 | Medium | Medium |
| temporalpdf QMC | 0.03 | Low | Medium |
| temporalpdf Temporal | 0.02 | Medium | Slow |
| BarrierModel | 0.015 | Low | Fast (inference) |

---

### Section 6: Conformal Prediction — Guaranteed Coverage (~250 lines)

**What it does:**
Show that conformal prediction gives calibrated intervals regardless of model mis-specification.

**Functions used:**
- `ConformalPredictor`
- `.predict_interval`
- `.coverage`
- `.interval_width`

**Test design:**
1. Train DistributionalRegressor (possibly mis-specified)
2. Wrap with ConformalPredictor
3. Check coverage on held-out test set

**Comparison:**

| Method | 90% Target | Actual Coverage |
|--------|------------|-----------------|
| Raw model intervals | 90% | 75-85% (under-covers) |
| Conformal wrapped | 90% | 89-91% (calibrated) |

**Expected results:**
- Raw model: Coverage varies with model quality
- Conformal: Always achieves target (±2%)
- Trade-off: Conformal intervals slightly wider

---

### Section 7: Comprehensive Test Battery (~300 lines)

**What it does:**
Run systematic tests across multiple scenarios to show robustness.

**Test matrix:**

| Scenario | Data Type | Horizon | Barrier | Methods Compared |
|----------|-----------|---------|---------|------------------|
| 1 | Stationary | 10 | 3% | All |
| 2 | Stationary | 20 | 5% | All |
| 3 | Stationary | 50 | 10% | All |
| 4 | Regime-switch | 20 | 5% | All |
| 5 | Trending vol | 20 | 5% | All |
| 6 | Fat tails (ν=3) | 20 | 5% | All |
| 7 | Thin tails (ν=30) | 20 | 5% | All |
| 8 | High drift | 20 | 5% | All |

**Metrics per test:**
- VaR breach rate (target: 5%)
- CRPS score
- Barrier prob accuracy
- Interval coverage

**Output:**
Summary table showing which method wins in each scenario.

**Expected pattern:**
- temporalpdf wins in fat-tail scenarios (4, 6)
- temporalpdf wins in regime-switch (4)
- All methods similar in thin-tail stationary (7)
- temporalpdf temporal wins in trending vol (5)

---

## Summary: What We Expect to Demonstrate

| Claim | Test | Expected Result |
|-------|------|-----------------|
| "CRPS training beats MSE" | Section 3 | 15-25% CRPS improvement |
| "Student-t beats Normal for tails" | Section 4 | VaR breach rate: 5% vs 8% |
| "Barrier prob is a killer feature" | Section 5 | Only temporalpdf can do it well |
| "Temporal dynamics matter" | Section 5 | 5-10% accuracy gain in regime changes |
| "BarrierModel beats CRPS for barriers" | Section 5 | 84% Brier improvement |
| "Conformal gives guaranteed coverage" | Section 6 | 90% target → 89-91% actual |
| "Robust across scenarios" | Section 7 | Wins in 6/8 scenarios |

---

## Notebook Metadata

| Attribute | Value |
|-----------|-------|
| Estimated lines | ~2000 |
| Estimated cells | ~80 |
| Runtime | ~10-15 minutes |
| Dependencies | temporalpdf, numpy, pandas, matplotlib, xgboost, arch |
| Data | 100% synthetic (reproducible) |

---

## Version History

| Version | Date | Notes |
|---------|------|-------|
| V4 | 2026-01-21 | Comprehensive showcase with test battery |

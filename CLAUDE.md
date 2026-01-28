# temporalpdf - Project Context

## CORE CONCEPT - DO NOT FORGET

### Pipeline 1 (Standard - what everyone else does)
```
Data → Model → Y (point prediction)
```
- Model directly predicts target value
- No uncertainty
- No distribution
- Just a number

### Pipeline 2 (temporalpdf - what this repo enables)
```
Data → Model → Distribution Parameters (μ, σ, ν, etc.) → Distribution → Y + uncertainty
```
- Model predicts DISTRIBUTION PARAMETERS, not Y directly
- Distribution gives you:
  - Point prediction (mean/median of distribution)
  - Standard deviation
  - Quantiles
  - Full shape (skewness, kurtosis)
  - Risk metrics (VaR, CVaR)
  - Everything needed for strategy/decisions

### Key Insight
The model (XGBoost, neural net, whatever) is trained as a MULTI-OUTPUT model where outputs are distribution parameters. This is an intermediary step between features and the final Y value.

### Comparison Must Show
1. Pipeline 1 residuals (actual - predicted)
2. Pipeline 2 residuals (actual - distribution_mean)
3. Pipeline 2 ALSO gives uncertainty, intervals, risk metrics

Both have residuals. Both have point predictions. Pipeline 2 just gets its point prediction FROM the distribution, and gets uncertainty for free.

## M5 Showcase

The showcase demonstrates:
1. Load M5 retail data with features (lag sales, price, SNAP, events)
2. Pipeline 1: XGBoost → predict sales directly
3. Pipeline 2: XGBoost (multi-output) → predict distribution parameters → distribution → sales + uncertainty
4. Compare residuals side by side
5. Show Pipeline 2's additional outputs (intervals, risk)

## What temporalpdf Provides

- Distribution classes (NIG, Student-t, Normal, etc.)
- Parameter fitting functions
- Distribution selection (compare_distributions, select_best_distribution)
- Time-evolving distributions (parameters that grow/drift with forecast horizon)
- Scoring rules (CRPS)
- Decision utilities (VaR, CVaR, Kelly)
- Visualization (3D temporal plots)

## DO NOT

- Fit distributions to residuals AFTER making predictions (that's just Pipeline 1 with post-hoc decoration)
- Forget that Pipeline 2 trains a model to output distribution parameters
- Show only Pipeline 1 and call it a comparison
- Make visualizations with text boxes instead of actual plots

## Documentation Progress

### Completed
- **docs/ARCHITECTURE.md** — Comprehensive architecture reference covering all 50 source files, 10 submodules, 4 root files. Contains:
  - Four-layer architecture (Foundation → Operations → Outputs → Integration) with dependency rule rationale
  - Module-by-module breakdown with Description and Why It Exists columns
  - Class hierarchy (Mermaid classDiagram) showing inheritance, protocol satisfaction, parameter↔distribution mapping
  - Interface specifications inline (Distribution protocol, TimeEvolvingDistribution ABC, VolatilityModel, WeightScheme, DynamicsModel)
  - Option tables at every pipeline stage (5 distributions, 5 volatility models, 6 weighting schemes, 5 dynamics models, etc.)
  - Design rationale inline where questions arise (frozen params, V1/V2 coexistence, core/ vs temporal/ for volatility, NIG closure)
  - Data flow diagrams (complete pipeline, module dependencies)
  - Usage patterns (4 patterns with code)
  - All abbreviations expanded on first use

### Remaining Phases
- **Core Concept** — The "reason for being" of the repo (distributional regression, Pipeline 1 vs 2)
- **Math** — Distribution math, scoring rule math, decision utility math
- **Mental Models** — How to think about when to use what
- **Visual Guide** — What the plots mean, how to interpret them

### User Notes
- User does not fully understand their own repository and needs documentation to learn it over 2-3 days
- Work in phases: explore/research first, then plan/write
- Do NOT use the explorer agent on the full repo (too large) — use it on specific smaller modules
- Information belongs where the reader encounters the question, not in separate appendix sections

# temporalpdf Completion Plan

## Background & Motivation

This library was developed in **early 2025** after encountering [XGBoostLSS](https://github.com/StatMixedML/XGBoostLSS). Rather than simply using XGBoostLSS, I wanted to implement distributional regression myself to deeply understand the mechanics and mathematics behind it.

### XGBoostLSS Comparison

| Aspect | XGBoostLSS | temporalpdf |
|--------|------------|-------------|
| **Focus** | General distributional regression | Trading/forecasting decisions |
| **ML Integration** | Built-in XGBoost training | Distribution-agnostic (bring your own model) |
| **Distributions** | 30+ via PyTorch | 5 hand-implemented (NIG, Normal, Student-t, Skew-Normal, Gen. Laplace) |
| **Gradients** | Automatic (PyTorch) | Manual derivation |
| **Decision Layer** | None | Kelly, VaR, CVaR, probability queries |
| **Time Evolution** | None | Drift/growth parameters over horizon |
| **Visualization** | Basic | Interactive 3D Plotly surfaces |
| **Scoring** | None built-in | CRPS, Log Score |

**When to use XGBoostLSS**: Pure distributional regression, many distribution choices, auto-tuning via Optuna.

**When to use temporalpdf**: Distribution → trading decision pipeline, time-evolving forecasts, risk-aware position sizing.

---

## Current State Assessment

### Done Well
- 5 distribution families (NIG, Normal, Student-t, Skew-Normal, Generalized Laplace)
- Decision utilities (Kelly, VaR, CVaR, probability queries)
- Proper scoring rules (CRPS, Log Score)
- Rich visualization (3D surfaces, heatmaps, slices, confidence bands, Plotly interactive)
- Time evolution with drift/growth parameters
- Good test coverage
- M5 showcase examples
- Clean API, type hints, docstrings

### Missing Pieces

| Gap | Priority | Effort |
|-----|----------|--------|
| ML integration (fit distribution params from data) | HIGH | Medium |
| Strategy layer (should_trade decision) | HIGH | Low |
| README "learning exercise" framing | HIGH | Low |
| XGBoostLSS comparison benchmark | MEDIUM | Medium |
| PyPI publish | MEDIUM | Low |
| More showcase examples | LOW | Medium |
| Backtest integration | LOW | High |

---

## Phase 1: Quick Wins (1-2 hours)

### 1.1 Update README with honest framing

Add after the overview section:

```markdown
## Background

This library was developed in early 2025 after encountering
[XGBoostLSS](https://github.com/StatMixedML/XGBoostLSS). Rather than simply using
that excellent library, I wanted to implement distributional regression myself to
deeply understand the mechanics and mathematics behind predicting entire distributions.

**temporalpdf** differentiates by focusing on:
- **Decision utilities**: Built-in Kelly criterion, VaR/CVaR filters, probability queries
- **Time evolution**: Parameters that drift and grow over prediction horizons
- **Trading-oriented API**: Position sizing, risk thresholds, strategy layer
- **Visualization**: Interactive 3D distribution evolution plots

For pure distributional regression with XGBoost and 30+ distribution families,
XGBoostLSS is the mature choice. For distribution → trading decision pipelines
with time-evolving uncertainty, temporalpdf fills that gap.
```

### 1.2 Add Strategy Layer

Create `src/temporalpdf/decision/strategy.py`:

```python
"""Trading decision layer: Distribution → Action."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from .kelly import kelly_fraction, fractional_kelly
from .risk import var, cvar
from .probability import prob_greater_than


class Action(Enum):
    LONG = "long"
    SHORT = "short"
    ABSTAIN = "abstain"


@dataclass
class TradeDecision:
    action: Action
    position_size: float  # As fraction of capital
    confidence: float     # P(profitable)
    expected_return: float
    var_95: float
    cvar_95: float
    reason: str


class TradingStrategy:
    """
    Convert distributional predictions to trading decisions.

    Example:
        >>> strategy = TradingStrategy(
        ...     min_expected_return=0.001,
        ...     max_cvar=0.05,
        ...     min_confidence=0.55,
        ...     kelly_fraction=0.5,
        ... )
        >>> decision = strategy.evaluate(dist, params)
        >>> if decision.action != Action.ABSTAIN:
        ...     execute_trade(decision.action, decision.position_size)
    """

    def __init__(
        self,
        min_expected_return: float = 0.0,
        max_var: float | None = None,
        max_cvar: float | None = 0.10,
        min_confidence: float = 0.5,
        kelly_fraction: float = 0.5,
        risk_free_rate: float = 0.0,
    ):
        self.min_expected_return = min_expected_return
        self.max_var = max_var
        self.max_cvar = max_cvar
        self.min_confidence = min_confidence
        self.kelly_frac = kelly_fraction
        self.risk_free_rate = risk_free_rate

    def evaluate(self, dist, params, t: float = 0.0) -> TradeDecision:
        """Evaluate distribution and return trading decision."""

        # Compute metrics
        exp_ret = dist.mean(t, params)
        var_95 = var(dist, params, alpha=0.05, t=t)
        cvar_95 = cvar(dist, params, alpha=0.05, t=t)
        conf = prob_greater_than(dist, params, 0.0, t=t)

        # Check filters
        if abs(exp_ret) < self.min_expected_return:
            return TradeDecision(
                Action.ABSTAIN, 0.0, conf, exp_ret, var_95, cvar_95,
                f"Expected return {exp_ret:.4f} below threshold"
            )

        if self.max_cvar and cvar_95 > self.max_cvar:
            return TradeDecision(
                Action.ABSTAIN, 0.0, conf, exp_ret, var_95, cvar_95,
                f"CVaR {cvar_95:.4f} exceeds max {self.max_cvar}"
            )

        if self.max_var and var_95 > self.max_var:
            return TradeDecision(
                Action.ABSTAIN, 0.0, conf, exp_ret, var_95, cvar_95,
                f"VaR {var_95:.4f} exceeds max {self.max_var}"
            )

        # Determine direction
        if exp_ret > 0 and conf < self.min_confidence:
            return TradeDecision(
                Action.ABSTAIN, 0.0, conf, exp_ret, var_95, cvar_95,
                f"Confidence {conf:.2%} below threshold {self.min_confidence:.2%}"
            )

        if exp_ret < 0 and (1 - conf) < self.min_confidence:
            return TradeDecision(
                Action.ABSTAIN, 0.0, conf, exp_ret, var_95, cvar_95,
                f"Short confidence {1-conf:.2%} below threshold"
            )

        # Compute position size
        size = fractional_kelly(
            dist, params,
            fraction=self.kelly_frac,
            t=t,
            risk_free_rate=self.risk_free_rate
        )

        action = Action.LONG if exp_ret > 0 else Action.SHORT

        return TradeDecision(
            action=action,
            position_size=abs(size),
            confidence=conf if action == Action.LONG else 1 - conf,
            expected_return=exp_ret,
            var_95=var_95,
            cvar_95=cvar_95,
            reason="All filters passed"
        )
```

---

## Phase 2: ML Integration (2-4 hours)

### 2.1 Create `src/temporalpdf/fitting/` module

```
fitting/
├── __init__.py
├── mle.py          # Maximum likelihood estimation
├── moment.py       # Method of moments
└── xgboost.py      # Optional XGBoost wrapper (requires xgboost)
```

### 2.2 Basic MLE fitting

```python
"""fitting/mle.py - Maximum likelihood parameter estimation."""

from scipy.optimize import minimize
import numpy as np


def fit_mle(dist_class, data, initial_params=None):
    """
    Fit distribution parameters via MLE.

    Args:
        dist_class: Distribution class (e.g., NIG)
        data: Array of observations
        initial_params: Starting parameter values

    Returns:
        Fitted parameter object
    """
    dist = dist_class()

    def neg_log_likelihood(param_vec):
        params = dist.params_from_vector(param_vec)
        ll = np.sum(np.log(dist.pdf(data, t=0, params=params) + 1e-10))
        return -ll

    result = minimize(
        neg_log_likelihood,
        initial_params or dist.default_params_vector(),
        method='L-BFGS-B',
        bounds=dist.param_bounds(),
    )

    return dist.params_from_vector(result.x)
```

---

## Phase 3: Polish & Publish (2-3 hours)

### 3.1 Add XGBoostLSS comparison notebook

`showcase/xgboostlss_comparison.ipynb`:
- Same dataset (M5 or synthetic)
- Both implementations side by side
- Show where temporalpdf adds value (strategy layer, time evolution, viz)

### 3.2 Publish to PyPI

```bash
pip install build twine
python -m build
twine upload dist/*
```

### 3.3 GitHub polish

- Add badges
- Add CHANGELOG.md
- Tag v0.1.0 release
- Add GitHub Actions for tests

---

## Phase 4: Optional Enhancements

| Enhancement | Description |
|-------------|-------------|
| Streamlit demo | Interactive web app for exploration |
| Backtest module | Connect to vectorbt or backtrader |
| More distributions | Variance Gamma, Skewed Student-t |
| Crypto defaults | Pre-tuned params for BTC, ETH |
| Real-time mode | Streaming parameter updates |

---

## Final README Structure

```
# temporalpdf

## Overview
## Background (learning exercise story, XGBoostLSS context)
## Installation
## Quick Start
## Key Features
  - Distributions
  - Decision Utilities (strategy layer)
  - Visualization
  - Scoring Rules
## Comparison with XGBoostLSS
## Examples
## API Reference
## Development
## License
```

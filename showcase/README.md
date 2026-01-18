# temporalpdf Showcase

## Start Here

**`why_distributions_beat_point_predictions.ipynb`** - The main showcase.

```bash
cd showcase
jupyter notebook why_distributions_beat_point_predictions.ipynb
```

## What It Covers

1. **Core API** - `tpdf.fit_nig()` → `tpdf.var()` → decide
2. **Fat Tails** - Normal underestimates extreme events by 3-10x
3. **Same Mean, Different Risk** - Why point predictions fail
4. **Distribution Selection** - `tpdf.select_best_distribution()`
5. **VaR Backtest** - `tpdf.rolling_var_backtest()`
6. **Regime Adaptation** - Distributions adjust to volatility

## Quick Example

```python
import temporalpdf as tpdf

# Fit distribution to data
params = tpdf.fit_nig(returns)

# Get risk metrics
dist = tpdf.NIG()
var_5 = tpdf.var(dist, params, alpha=0.05)
cvar_5 = tpdf.cvar(dist, params, alpha=0.05)
kelly = tpdf.kelly_fraction(dist, params)

# Probability queries
p_loss = tpdf.prob_less_than(dist, params, threshold=-1.0)

# Make decision
if expected_return > 0 and var_5 < max_risk:
    execute_trade(position_size=kelly)
```

## Interactive CLI Tool

```bash
python interactive_threshold_explorer.py --asset sp500
python interactive_threshold_explorer.py --asset btc
python interactive_threshold_explorer.py --asset eurusd
```

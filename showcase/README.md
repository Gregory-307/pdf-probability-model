# temporalpdf Showcase

## Main Notebook

**`stock_trading_comparison.ipynb`** - The core demonstration.

Compares two ML pipelines on the same data:
- **Pipeline 1**: XGBoost → point prediction (single number)
- **Pipeline 2**: XGBoost → distribution parameters → VaR filter

Result: Distribution prediction achieves ~2x better Sharpe ratio by filtering high-risk trades.

```bash
cd showcase
jupyter notebook stock_trading_comparison.ipynb
```

## Other Files

| File | Description |
|------|-------------|
| `expanded_showcase.ipynb` | 3D visualization, multi-asset comparison |
| `why_distributions_beat_point_predictions.ipynb` | Library API walkthrough |
| `interactive_threshold_explorer.py` | CLI tool for exploring distributions |

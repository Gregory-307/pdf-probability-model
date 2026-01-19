# M5 Walmart Dataset - temporalpdf Showcase Setup

## Links
- **HuggingFace**: https://huggingface.co/datasets/autogluon/fev_datasets (config: `m5_1W`)
- **Kaggle Accuracy**: https://www.kaggle.com/competitions/m5-forecasting-accuracy/data
- **Kaggle Uncertainty**: https://www.kaggle.com/competitions/m5-forecasting-uncertainty/data

## Dataset Overview
- 3,049 products across 10 Walmart stores in 3 US states (CA, TX, WI)
- ~1,941 days of daily sales history (2011-2016)
- Intermittent demand pattern: lots of zeros, promotional spikes
- HuggingFace version: weekly aggregated, 30.5k rows

## Load Data
```python
from datasets import load_dataset
m5 = load_dataset("autogluon/fev_datasets", "m5_1W", split="train")
```

## Task

### Pipeline Comparison
1. **Baseline**: XGBoost point prediction (predict next-day/week unit sales)
2. **Our approach**: XGBoost → NIG distribution parameters (mu, delta, alpha, beta) via temporalpdf

### Evaluation Metrics
- **Point**: RMSE, MAE
- **Distributional**: CRPS, pinball loss at quantiles (0.5, 0.75, 0.95, 0.99)
- **Decision quality**: simulated inventory decisions using VaR/CVaR/Kelly from distribution

## Why This Dataset Showcases Distributional Prediction Value

1. **Intermittent demand** (many zeros) breaks point prediction assumptions
2. **Promotional events** cause regime shifts (sudden demand spikes)
3. **Original M5 Uncertainty competition** proved distributional forecasting adds independent value over point forecasting
4. **Published LightGBM baselines** exist for direct comparison

## Key Features to Engineer
- Lag features (sales_t-1, sales_t-7, sales_t-28)
- Rolling statistics (mean, std over 7/14/28 days)
- Calendar (day of week, month, holiday flags, SNAP days)
- Price features (current price, price changes)
- Category/store embeddings

## Success Criteria

Show that NIG-based distributional prediction either:
1. Achieves better CRPS than point prediction + assumed Normal noise
2. Produces better-calibrated prediction intervals (coverage closer to nominal)
3. Leads to better risk-adjusted decisions when feeding into Kelly/VaR optimization

## The Demo Narrative

> "Here's XGBoost predicting 'sell 3 units tomorrow' → MSE = X"
>
> "Here's XGBoost → NIG distribution → 'sell 0-2 units with 80% probability, but 5% chance of 15+ units' → CRPS = Y, and when we use this for inventory decisions via Kelly criterion, we get Z% better risk-adjusted returns"

## Alternative Dataset

**Electricity Price Forecasting** (EPF)
- `autogluon/fev_datasets` configs: `epf_de`, `epf_nordpool`
- Extreme price spikes (100x normal, sometimes negative)
- `epftoolbox` explicitly benchmarks LEAR (point) vs LEAR+GARCH (distributional)
- More industrial credibility, less name recognition than M5

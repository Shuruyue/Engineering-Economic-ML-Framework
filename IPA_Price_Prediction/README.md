# IPA Price Prediction (Quarterly IPA Price Forecasting)

This sub-project is the final version of the IPA (Isopropyl Alcohol) price prediction for the Engineering Economics course, focusing on **reproducibility, deliverability, and maintainability**.

## Final Optimization Highlights (Sealed)

1. Data collection stabilization: cache field validation, output schema standardization, graceful degradation when dependencies are missing.
2. Feature engineering refactoring: switched to a "quarterly-first" feature pipeline to prevent lag/rolling semantic drift from weekly-then-quarterly averaging.
3. Backtesting and ensemble strengthening: both XGBoost and SARIMA use walk-forward backtest for hyperparameter tuning; ensemble uses inverse CV error weighting with automatic weak model suppression.
4. Recursive prediction consistency: future periods recompute exogenous derived features (lag/rolling/interaction) at each step, avoiding stale template values.
5. Deliverable artifacts: HTML, charts, CSV, JSON manifest, multi-year summary, and walk-forward backtest details are all generated in one run.
6. Documentation sealed: execution parameters, output definitions, and maintenance boundaries are clearly defined.

## Project Goals

- Use historical data from `2012-01-01` to `2024-12-31`.
- Forecast `2025` and `2026` Q1–Q4 IPA prices (TWD/KG).
- Incorporate energy prices, exchange rates, stock market proxies, geopolitical events, and seasonal features.

## Model Pipeline Summary

- Data sources:
  - Target variable: IPA price series reconstructed from charts
  - Exogenous variables: Yahoo Finance + event indicators
- Feature engineering:
  - lag / rolling / pct change / time seasonal encoding
- Models:
  - Walk-forward tuned tree model (`XGBoost`, auto-fallback to `GradientBoosting` if unavailable)
  - Walk-forward tuned `SARIMA`
  - Inverse CV `MAPE` squared weighting for ensemble, with automatic weak model suppression
- Forecasting:
  - Recursive quarter-by-quarter prediction
  - Exogenous variable damped drift projection
  - Adaptive confidence intervals derived from volatility + residual distribution

## Directory Structure

```text
IPA_Price_Prediction/
├── Code/
│   ├── ipa_data_collector.py
│   ├── ipa_feature_engineering.py
│   ├── ipa_models.py
│   ├── ipa_price_prediction.py
│   ├── requirements.txt
│   ├── Data/
│   ├── figures/
│   ├── reports/
│   └── tests/
├── IPA_Price_Forecasting_Project_Proposal.md
├── Technical_Optimization_2026.md
├── text.ipynb
└── README.md
```

## Quick Start

```bash
cd IPA_Price_Prediction/Code
pip install -r requirements.txt
python ipa_price_prediction.py --years 2025 2026
```

Common parameters:

```bash
# Forecast a single year
python ipa_price_prediction.py --years 2025

# Force refresh external data (ignore cache)
python ipa_price_prediction.py --years 2025 2026 --refresh-cache

# Custom output directories
python ipa_price_prediction.py --years 2025 2026 --figures-dir figures --reports-dir reports
```

## Output Artifacts

After execution, the following files are generated:

- `Code/figures/`
  - `historical_prices.png`
  - `prediction_2025.png`
  - `prediction_2026.png`
  - `model_comparison.png`
- `Code/reports/`
  - `ipa_forecast_2025.html`
  - `ipa_forecast_2026.html`
  - `ipa_forecast_2025.csv`
  - `ipa_forecast_2026.csv`
  - `ipa_forecast_all_years.csv`
  - `model_metrics.csv`
  - `feature_importance.csv`
  - `sensitivity_analysis.csv`
  - `ensemble_weights.json`
  - `walk_forward_backtest.csv`
  - `run_manifest_2025.json`
  - `run_manifest_2026.json`

## Maintenance Recommendations

This project has entered its final version. Only essential maintenance is recommended going forward:

- Package/API compatibility fixes
- Cache data refresh
- Report year extension

Supplementary technical notes:

- `Technical_Optimization_2026.md`: Three rounds of optimization details, validation results, and research references.

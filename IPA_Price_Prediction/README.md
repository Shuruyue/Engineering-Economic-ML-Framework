# IPA Price Prediction (異丙醇價格預測模型)

本專案為工程經濟課程之 IPA (Isopropyl Alcohol) 價格預測實作，輸出季度價格與情境區間。

## 專案目標

- 使用 2012-01 至 2024-12 歷史資料。
- 預測 2025 與 2026 年 Q1-Q4 IPA 價格 (TWD/KG)。
- 納入多因子: 能源、匯率、股市代理、地緣事件、季節性。

## 方法摘要

- 資料處理:
  - 外部市場資料下載與整併
  - 時間序列前向補值 (避免未來資訊洩漏)
  - 分位數異常值裁切
- 特徵工程:
  - lag / rolling / pct change / time seasonal encoding
- 模型:
  - Walk-forward 調參 XGBoost
  - SARIMA 基準模型
  - 依 MAPE 自動加權 ensemble
- 預測:
  - 遞迴式逐季預測
  - 外生變數阻尼漂移投影 (damped drift)
  - 自適應不確定區間 (歷史波動 + 測試殘差)
- 執行效率:
  - 市場資料本地快取 (`Code/Data/market_data_*.csv`)
  - 一次訓練可同時輸出多年度報告

## 目錄結構

```text
IPA_Price_Prediction/
├── Code/
│   ├── ipa_data_collector.py
│   ├── ipa_feature_engineering.py
│   ├── ipa_models.py
│   ├── ipa_price_prediction.py
│   ├── Data/
│   ├── figures/
│   └── reports/
├── IPA_Price_Forecasting_Project_Proposal.md
└── README.md
```

## 快速開始

```bash
cd Code
pip install yfinance xgboost scikit-learn matplotlib statsmodels
python ipa_price_prediction.py --years 2025 2026
```

常用參數:

```bash
# 只預測單一年份
python ipa_price_prediction.py --years 2025

# 強制重新抓取外部資料 (忽略快取)
python ipa_price_prediction.py --years 2025 2026 --refresh-cache
```

## 輸出檔案

- 圖表: `Code/figures/`
- 報告:
  - `Code/reports/ipa_forecast_2025.html`
  - `Code/reports/ipa_forecast_2026.html`

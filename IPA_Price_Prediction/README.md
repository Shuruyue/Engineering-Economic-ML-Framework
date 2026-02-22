# IPA Price Prediction (異丙醇季度價格預測)

本子專案為工程經濟課程的 IPA (Isopropyl Alcohol) 價格預測最終版，聚焦「可重現、可交付、可維護」。

## 最終優化重點 (封版)

1. 資料蒐集穩定化: 快取欄位驗證、輸出 schema 標準化、缺依賴時可降級執行。  
2. 特徵工程完整化: 目標序列驗證與插值 fallback（`cubic -> linear`）。  
3. 模型評估一致化: 以內部模型 key 計算自適應權重，避免名稱不一致。  
4. 交付工件補齊: HTML、圖表、CSV、JSON manifest、多年度總表一次產出。  
5. 文件封版: 執行參數、輸出定義、維護邊界明確化。  

## 專案目標

- 使用歷史資料區間 `2012-01-01` 到 `2024-12-31`。
- 預測 `2025` 與 `2026` 年 Q1-Q4 IPA 價格 (TWD/KG)。
- 融合能源、匯率、股市代理變數、地緣事件與季節性特徵。

## 模型流程摘要

- 資料來源:
  - 目標值: 由圖表還原之 IPA 價格序列
  - 外生變數: Yahoo Finance + 事件指標
- 特徵工程:
  - lag / rolling / pct change / time seasonal encoding
- 模型:
  - Walk-forward 調參樹模型 (`XGBoost`，若缺套件自動降級 `GradientBoosting`)
  - `SARIMA` 基準模型
  - 依測試 `MAPE` 反向加權 ensemble
- 預測:
  - 遞迴式逐季推進
  - 外生變數阻尼漂移投影
  - 波動 + 殘差導出的自適應區間

## 目錄結構

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
│   └── reports/
├── IPA_Price_Forecasting_Project_Proposal.md
└── README.md
```

## 快速開始

```bash
cd IPA_Price_Prediction/Code
pip install -r requirements.txt
python ipa_price_prediction.py --years 2025 2026
```

常用參數:

```bash
# 只預測單一年份
python ipa_price_prediction.py --years 2025

# 強制重新抓取外部資料 (忽略快取)
python ipa_price_prediction.py --years 2025 2026 --refresh-cache

# 自訂輸出目錄
python ipa_price_prediction.py --years 2025 2026 --figures-dir figures --reports-dir reports
```

## 輸出工件

執行後會產生:

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
  - `run_manifest_2025.json`
  - `run_manifest_2026.json`

## 維護模式建議

本專案已進入最終版，後續建議僅做必要維護:

- 套件/API 相容性修正
- 快取資料刷新
- 報告年份延伸

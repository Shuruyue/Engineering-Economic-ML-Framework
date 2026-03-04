# IPA Price Prediction (異丙醇季度價格預測)

本子專案為工程經濟課程的 IPA (Isopropyl Alcohol) 價格預測最終版，聚焦「可重現、可交付、可維護」。

## 最終優化重點 (封版)

1. 資料蒐集穩定化: 快取欄位驗證、輸出 schema 標準化、缺依賴時可降級執行。  
2. 特徵工程重構: 改為「季度先行」特徵管線，避免先週後季造成 lag/rolling 語意偏移。  
3. 回測與集成強化: XGBoost / SARIMA 均採 walk-forward backtest 調參，ensemble 以 CV 誤差加權且自動剔除弱模型。  
4. 遞迴預測一致化: 未來期會同步重算外生衍生特徵（lag/rolling/interaction），避免沿用過時模板值。  
5. 交付工件補齊: HTML、圖表、CSV、JSON manifest、多年度總表與 walk-forward 回測明細一次產出。  
6. 文件封版: 執行參數、輸出定義、維護邊界明確化。  

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
  - Walk-forward 調參 `SARIMA`
  - 依 CV `MAPE` 反向加權（平方）ensemble，並自動抑制高誤差模型
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
│   ├── reports/
│   └── tests/
├── IPA_Price_Forecasting_Project_Proposal.md
├── Technical_Optimization_2026.md
├── text.ipynb
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
  - `walk_forward_backtest.csv`
  - `run_manifest_2025.json`
  - `run_manifest_2026.json`

## 維護模式建議

本專案已進入最終版，後續建議僅做必要維護:

- 套件/API 相容性修正
- 快取資料刷新
- 報告年份延伸

補充技術說明:

- `Technical_Optimization_2026.md`: 三輪優化內容、驗證結果與論文依據。

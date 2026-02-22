# IPA 價格預測專案優化企畫書

文件版本: v1.1  
更新日期: 2026-02-22  
適用期間: 2025Q1-2026Q4 預測與課堂成果展示

## 1. 專案定位與目標

本專案定位為工程經濟課程之 IPA (Isopropyl Alcohol) 價格預測應用案例，重點是可解釋、可重現、可交付。

主要目標:
- 建立可重現的季度預測流程，輸出 2025 與 2026 年 Q1-Q4 價格區間。
- 納入多因子影響，反映成本、匯率、需求與事件衝擊。
- 提供圖表與 HTML 報告，支援課程展示與報告歸檔。

## 2. 問題定義

預測標的:
- 目標變數: IPA 單價 (TWD/KG)
- 頻率: 季度
- 預測步長: 每次 4 季 (Q1-Q4)

決策用途:
- 採購預算估算
- 成本敏感度分析
- 基準/樂觀/保守情境比較

## 3. 資料策略與來源

### 3.1 已落地資料

1. 目標序列:
- 來源: 既有圖表還原的 IPA 序列 (2012-01 至 2024-12)
- 頻率: 週資料，後續彙整為季資料

2. 外生變數:
- Yahoo Finance: `CL=F`, `BZ=F`, `NG=F`, `USDTWD=X`, `DX-Y.NYB`, `^TWII`, `TSM`
- 事件指標: COVID-19、俄烏戰爭、紅海危機、中美貿易戰

### 3.2 資料治理

- 型別/欄位完整性檢查
- 時間索引去重與排序
- 僅前向填補，避免 look-ahead bias
- 異常值分位裁切 (1%/99%)
- 本地快取以提高穩定性與重現性 (`Code/Data/market_data_*.csv`)

## 4. 特徵工程設計

核心特徵:
- 自回歸: lag1-lag4
- 滾動統計: MA/STD/MAX/MIN (4, 8, 12 視窗)
- 變化率: 1 期、4 期、13 期
- 時間特徵: Year/Quarter/Month/Week + sin/cos 季節編碼
- 外生特徵: 能源、匯率、股市代理、事件指標

原則:
- 以可解釋為優先
- 保持可遞迴推進，確保未來期可生成特徵

## 5. 模型策略

### 5.1 模型組合

1. XGBoost:
- 主力模型，處理非線性與特徵交互
- 以 walk-forward 驗證進行參數選擇

2. SARIMA:
- 作為時序基準模型
- 提供趨勢/季節結構補充

### 5.2 集成方式

- 以反向 MAPE 產生權重:
  - 權重 = `(1 / MAPE_i) / Σ(1 / MAPE_j)`
- 單模型失敗時自動退化為可用模型輸出

### 5.3 未來期推進

- XGBoost 採遞迴逐季預測:
  - 先預測下一季
  - 回填 target 派生特徵 (lag/rolling/change)
  - 持續推進到目標季度
- 外生變數採阻尼漂移投影:
  - 依最近 8 季估計 drift 並逐季衰減
  - 事件型變數維持狀態值
- SARIMA 直接多步預測並切出目標年度

### 5.4 不確定區間

- 區間由歷史波動率 + 測試殘差分布動態估計
- 避免固定比例上下限造成失真

## 6. 評估方法

- 時序切分訓練/測試，不打亂時間順序
- 訓練集內 walk-forward 驗證

主要指標:
- MAE
- RMSE
- MAPE
- Direction Accuracy

建議驗收門檻:
- MAPE <= 8%
- Direction Accuracy >= 65%

## 7. 常見問題與處理

1. 外部資料格式/下載失敗
- 加入 Yahoo 回傳格式健壯處理 (含 MultiIndex close 抽取)
- 快取可降低外部依賴風險

2. 時序資料洩漏
- 全流程禁用反向填補
- 僅使用歷史資料與遞迴生成特徵

3. 重複訓練耗時
- 改為一次訓練、多年度輸出 (`--years 2025 2026`)

4. 結構性事件衝擊
- 事件虛擬變數保留
- 情境區間反映殘差不確定性

## 8. 本次最終優化成果

- `ipa_data_collector.py`
  - 強化 yfinance 相容性
  - 新增快取讀寫與 refresh 控制
  - 保留因果向前填補
- `ipa_price_prediction.py`
  - CLI 參數化 (`--years`, `--refresh-cache`, `--no-cache`)
  - 一次訓練多年度輸出
  - 外生變數阻尼漂移投影
  - 遞迴式季度預測 + 自適應不確定區間
- `ipa_models.py`
  - 清理噪音輸出與不必要 import
- 文件與報告
  - 移除 emoji/圖示字元
  - 更新 README 與報告輸出

## 9. 結案維護建議

此案建議採「結案維護模式」:
- 不再新增模型家族或改動核心口徑
- 僅允許必要維護:
  - API 相容性修正
  - 報告年度更新
  - 快取刷新

## 10. 執行與重現

安裝:

```bash
cd IPA_Price_Prediction/Code
pip install yfinance xgboost scikit-learn matplotlib statsmodels
```

執行:

```bash
python ipa_price_prediction.py --years 2025 2026
```

刷新資料快取:

```bash
python ipa_price_prediction.py --years 2025 2026 --refresh-cache
```

輸出:
- 圖表: `IPA_Price_Prediction/Code/figures/`
- 報告: `IPA_Price_Prediction/Code/reports/ipa_forecast_2025.html`
- 報告: `IPA_Price_Prediction/Code/reports/ipa_forecast_2026.html`

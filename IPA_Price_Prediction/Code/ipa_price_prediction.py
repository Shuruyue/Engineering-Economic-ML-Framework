"""
IPA Price Prediction Model - Main Script
"""

import argparse
import json
import os
import re
from datetime import datetime
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

from ipa_data_collector import IPADataCollector
from ipa_feature_engineering import IPAFeatureEngineer
from ipa_models import ARIMAModel, ModelEvaluator, XGBoostModel


class IPAPricePredictor:
    TIME_FEATURES = {
        'Year', 'Quarter', 'Month', 'Week',
        'Month_sin', 'Month_cos', 'Quarter_sin', 'Quarter_cos'
    }
    EVENT_FEATURE_KEYWORDS = {
        'COVID19', 'Ukraine_War', 'RedSea_Crisis', 'US_China_Trade', 'Geopolitical_Risk'
    }

    def __init__(
        self,
        target_year=2025,
        start_date='2012-01-01',
        end_date='2024-12-31',
        cache_dir='Data',
        figures_dir='figures',
        reports_dir='reports',
        use_cache=True,
        refresh_cache=False,
        random_seed=42
    ):
        self.target_year = target_year
        self.start_date = start_date
        self.end_date = end_date
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = self._resolve_path(cache_dir)
        self.figure_dir = self._resolve_path(figures_dir)
        self.report_dir = self._resolve_path(reports_dir)
        self.use_cache = use_cache
        self.refresh_cache = refresh_cache
        self.random_seed = random_seed

        self.data = None
        self.quarterly_data = None
        self.models = {}
        self.predictions = {}
        self.results = []
        self.feature_cols = []
        self.exogenous_feature_cols = []
        self.X = None
        self.y = None
        self.y_test = None
        self.future_predictions = None
        self.sensitivity_results = None
        self.feature_importance_df = None
        self.ensemble_weights = {}
        self.artifacts = {}
        self.model_display_names = {
            'xgboost': 'XGBoost',
            'sarima': 'SARIMA',
            'ensemble': 'Ensemble'
        }
        self.generated_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        np.random.seed(self.random_seed)

    def _resolve_path(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(self.base_dir, path)

    @staticmethod
    def _safe_mape(y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denominator = np.where(np.abs(y_true) < 1e-6, 1.0, np.abs(y_true))
        return float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100)

    @staticmethod
    def _is_target_derived_feature(feature_name):
        return feature_name.startswith('IPA_Price_TWD_')

    def _clip_outliers(self, df, quantile_low=0.01, quantile_high=0.99, exclude_cols=None):
        exclude_cols = set(exclude_cols or [])
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in exclude_cols:
                continue
            lower = result[col].quantile(quantile_low)
            upper = result[col].quantile(quantile_high)
            result[col] = result[col].clip(lower=lower, upper=upper)
        return result

    def _walk_forward_mape(self, X, y, params, n_splits=4):
        if len(X) < 16:
            return float('inf')
        usable_splits = min(n_splits, max(2, len(X) // 8))
        splitter = TimeSeriesSplit(n_splits=usable_splits)
        fold_scores = []
        for train_idx, val_idx in splitter.split(X):
            if len(train_idx) < 8 or len(val_idx) == 0:
                continue
            model = XGBoostModel(**params)
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[val_idx])
            fold_scores.append(self._safe_mape(y[val_idx], pred))
        return float(np.mean(fold_scores)) if fold_scores else float('inf')

    def _fit_xgboost_with_cv(self, X_train, y_train):
        param_grid = [
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05},
            {'n_estimators': 120, 'max_depth': 4, 'learning_rate': 0.05},
            {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.08},
            {'n_estimators': 180, 'max_depth': 5, 'learning_rate': 0.08},
        ]
        print("Walk-forward tuning for XGBoost...")
        best_params, best_score = None, float('inf')
        for params in param_grid:
            cv_mape = self._walk_forward_mape(X_train, y_train, params=params)
            print(f"  Params {params} -> CV MAPE: {cv_mape:.2f}%")
            if cv_mape < best_score:
                best_score, best_params = cv_mape, params
        if best_params is None:
            best_params = {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1}
            best_score = float('nan')
        print(f"Best XGBoost params: {best_params}")
        model = XGBoostModel(**best_params)
        model.fit(X_train, y_train)
        return model, best_params, best_score

    def _derive_ensemble_weights(self):
        valid_rows = [
            r for r in self.results
            if r.get('ModelKey') in {'xgboost', 'sarima'}
            and r.get('MAPE') is not None
            and np.isfinite(r['MAPE'])
        ]
        if not valid_rows:
            self.ensemble_weights = {}
            return
        inverse_error = {}
        for row in valid_rows:
            inverse_error[row['ModelKey']] = 1.0 / max(float(row['MAPE']), 1e-6)
        total = sum(inverse_error.values())
        self.ensemble_weights = {m: w / total for m, w in inverse_error.items()} if total > 0 else {}
        print("Adaptive ensemble weights:")
        for model_key, weight in self.ensemble_weights.items():
            model_name = self.model_display_names.get(model_key, model_key)
            print(f"  {model_name}: {weight:.2%}")

    def _evaluate_ensemble_on_test(self):
        if self.y_test is None or len(self.y_test) == 0:
            return None
        if 'xgboost_test' not in self.predictions and 'sarima_test' not in self.predictions:
            return None
        self.results = [r for r in self.results if r.get('ModelKey') != 'ensemble']

        w_xgb = self.ensemble_weights.get('xgboost', 0.5)
        w_sarima = self.ensemble_weights.get('sarima', 0.5)
        total = w_xgb + w_sarima
        if total <= 0:
            w_xgb, w_sarima = 0.5, 0.5
        else:
            w_xgb, w_sarima = w_xgb / total, w_sarima / total

        if 'xgboost_test' in self.predictions and 'sarima_test' in self.predictions:
            ensemble_test = (
                np.asarray(self.predictions['xgboost_test'], dtype=float) * w_xgb +
                np.asarray(self.predictions['sarima_test'], dtype=float) * w_sarima
            )
        elif 'xgboost_test' in self.predictions:
            ensemble_test = np.asarray(self.predictions['xgboost_test'], dtype=float)
        else:
            ensemble_test = np.asarray(self.predictions['sarima_test'], dtype=float)

        ensemble_results = ModelEvaluator.evaluate(self.y_test, ensemble_test, self.model_display_names['ensemble'])
        ensemble_results['ModelKey'] = 'ensemble'
        self.results.append(ensemble_results)
        ModelEvaluator.print_results(ensemble_results)
        self.predictions['ensemble_test'] = ensemble_test
        return ensemble_results

    def load_and_prepare_data(self):
        print("\n" + "=" * 60)
        print(f"IPA Price Prediction Model - Forecasting {self.target_year}")
        print("=" * 60)

        # Reset run-specific states before data preparation.
        self.models = {}
        self.predictions = {}
        self.results = []
        self.feature_cols = []
        self.exogenous_feature_cols = []
        self.future_predictions = None
        self.sensitivity_results = None
        self.feature_importance_df = None
        self.ensemble_weights = {}

        feature_engineer = IPAFeatureEngineer()

        print("\n[1/4] Creating IPA price data...")
        ipa_data = feature_engineer.create_ipa_price_data()

        print("\n[2/4] Collecting external data...")
        collector = IPADataCollector(
            start_date=self.start_date,
            end_date=self.end_date,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
            refresh_cache=self.refresh_cache
        )
        external_data = collector.collect_all_data()

        print("\n[3/4] Merging data...")
        self.data = ipa_data.copy() if external_data is None or external_data.empty else ipa_data.join(external_data, how='left')
        self.data = self.data.sort_index().ffill()
        self.data = self._clip_outliers(self.data, exclude_cols=['IPA_Price_TWD'])

        print("\n[4/4] Feature engineering...")
        self.data = feature_engineer.prepare_features(self.data, target_col='IPA_Price_TWD')
        self.quarterly_data = feature_engineer.resample_to_quarterly(self.data).sort_index()

        print("\n[OK] Data preparation complete")
        print(f"  Weekly data: {len(self.data)} records")
        print(f"  Quarterly data: {len(self.quarterly_data)} records")
        return self

    def train_models(self, test_size=0.2):
        print("\n" + "=" * 60)
        print("Model Training")
        print("=" * 60)

        if self.quarterly_data is None or self.quarterly_data.empty:
            raise ValueError("Quarterly data is empty. Run load_and_prepare_data() first.")

        df = self.quarterly_data.copy()
        target_col = 'IPA_Price_TWD'
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]
        if not feature_cols:
            raise ValueError("No numeric feature columns available for training.")

        X = df[feature_cols].to_numpy(dtype=float)
        y = df[target_col].to_numpy(dtype=float)
        split_idx = int(len(X) * (1 - test_size))
        split_idx = min(max(split_idx, 8), len(X) - 4)
        if split_idx <= 0 or split_idx >= len(X):
            raise ValueError("Dataset too small for time-series split.")

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        print(f"\nTraining set: {len(X_train)} records")
        print(f"Test set: {len(X_test)} records")

        print("\n" + "-" * 40)
        print("Training XGBoost model...")
        xgb_model, best_params, cv_mape = self._fit_xgboost_with_cv(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_name = xgb_model.get_display_name()
        self.model_display_names['xgboost'] = xgb_name
        xgb_results = ModelEvaluator.evaluate(y_test, xgb_pred, xgb_name)
        xgb_results['ModelKey'] = 'xgboost'
        xgb_results['CV_MAPE'] = cv_mape
        xgb_results['Params'] = str(best_params)
        self.results.append(xgb_results)
        ModelEvaluator.print_results(xgb_results)
        self.models['xgboost'] = xgb_model
        self.predictions['xgboost_test'] = np.asarray(xgb_pred, dtype=float)
        self.feature_importance_df = xgb_model.get_feature_importance(feature_cols)
        print("\nTop 10 important features:")
        print(self.feature_importance_df.head(10).to_string(index=False))

        print("\n" + "-" * 40)
        print("Training SARIMA model...")
        try:
            arima_model = ARIMAModel(order=(1, 1, 1), seasonal_order=(1, 1, 1, 4))
            arima_model.fit(y_train)
            arima_pred_array = np.asarray(arima_model.predict(len(y_test)), dtype=float)
            arima_results = ModelEvaluator.evaluate(y_test, arima_pred_array, "SARIMA")
            arima_results['ModelKey'] = 'sarima'
            self.results.append(arima_results)
            ModelEvaluator.print_results(arima_results)
            self.models['sarima'] = arima_model
            self.predictions['sarima_test'] = arima_pred_array
        except Exception as e:
            print(f"SARIMA training failed: {e}")

        self.X, self.y, self.y_test, self.feature_cols = X, y, y_test, feature_cols
        self.exogenous_feature_cols = [
            c for c in self.feature_cols
            if (not self._is_target_derived_feature(c)) and (c not in self.TIME_FEATURES)
        ]
        self._derive_ensemble_weights()
        print("\n" + "-" * 40)
        print("Evaluating Ensemble model...")
        self._evaluate_ensemble_on_test()
        print("\n" + "-" * 40)
        print("Sensitivity analysis...")
        self.sensitivity_results = self._perform_sensitivity_analysis()
        return self

    def _perform_sensitivity_analysis(self, perturbation=0.1):
        if 'xgboost' not in self.models:
            return None
        model = self.models['xgboost']
        base_pred = model.predict(self.X)
        base_mean = np.mean(base_pred)
        if abs(base_mean) < 1e-6:
            return None

        sensitivity = {}
        key_features = ['WTI_Price', 'Brent_Price', 'USD_TWD', 'DXY', 'NatGas_Price', 'IPA_Price_TWD_lag1', 'TSM_Price', 'TWII']
        for i, feature in enumerate(self.feature_cols):
            feature_name = str(feature).split(',')[0].replace('(', '').replace("'", "").strip()
            if not any(kf in feature_name for kf in key_features):
                continue
            X_up, X_down = self.X.copy(), self.X.copy()
            X_up[:, i] *= (1 + perturbation)
            X_down[:, i] *= (1 - perturbation)
            impact = (np.mean(model.predict(X_up)) - np.mean(model.predict(X_down))) / (2 * base_mean) * 100
            sensitivity[feature_name] = impact

        sorted_sens = sorted(sensitivity.items(), key=lambda x: abs(x[1]), reverse=True)
        print("\nKey variable sensitivity (+/-10% perturbation):")
        print("-" * 45)
        for feat, impact in sorted_sens[:5]:
            direction = "positive" if impact > 0 else "negative"
            print(f"  {feat:20s}: {direction:8s} {abs(impact):.2f}%")
        return dict(sorted_sens)

    def _compute_forecast_horizon(self, quarters_ahead):
        last_period = self.quarterly_data.index[-1].to_period('Q')
        next_period = last_period + 1
        target_start = pd.Period(f"{self.target_year}Q1", freq='Q')
        step_offset = target_start.ordinal - next_period.ordinal
        if step_offset < 0:
            raise ValueError(f"Target year {self.target_year} is earlier than available data end {last_period}.")
        total_steps = step_offset + quarters_ahead
        periods = [next_period + i for i in range(total_steps)]
        return total_steps, step_offset, periods, periods[step_offset: step_offset + quarters_ahead]

    @staticmethod
    def _safe_pct_change(values, periods):
        values = np.asarray(values, dtype=float)
        if values.size <= periods:
            return 0.0
        base = values[-(periods + 1)]
        return 0.0 if abs(base) < 1e-6 else float((values[-1] - base) / base)

    def _project_exogenous_features(self, future_periods):
        if not self.exogenous_feature_cols:
            return pd.DataFrame(index=pd.PeriodIndex(future_periods, freq='Q'))

        history_df = self.quarterly_data[self.exogenous_feature_cols].copy()
        forecast_index = pd.PeriodIndex(future_periods, freq='Q')
        projection = pd.DataFrame(index=forecast_index, columns=self.exogenous_feature_cols, dtype=float)

        for col in self.exogenous_feature_cols:
            series = history_df[col].dropna().astype(float)
            if series.empty:
                projection[col] = 0.0
                continue
            last_value = float(series.iloc[-1])
            if any(key in col for key in self.EVENT_FEATURE_KEYWORDS):
                projection[col] = last_value
                continue
            lookback = min(8, len(series))
            recent = series.iloc[-lookback:].to_numpy(dtype=float)
            slope = float(np.polyfit(np.arange(lookback), recent, 1)[0]) if lookback >= 3 else 0.0
            scale = max(abs(np.mean(recent)), 1.0)
            slope = float(np.clip(slope, -0.08 * scale, 0.08 * scale))
            value = last_value
            for step, period in enumerate(forecast_index, start=1):
                value += slope * (0.85 ** (step - 1))
                projection.at[period, col] = max(value, 0.0) if last_value >= 0 else value
        return projection.ffill().bfill().fillna(0.0)

    def _build_future_feature_row(self, future_period, target_history, template_row, exogenous_values=None):
        row = template_row.copy()
        history = np.asarray(target_history, dtype=float)
        ts = future_period.to_timestamp(how='end')

        if exogenous_values is not None:
            for col in self.exogenous_feature_cols:
                if col in row.index and col in exogenous_values.index:
                    row[col] = float(exogenous_values[col])

        if 'Year' in row.index:
            row['Year'] = ts.year
        if 'Quarter' in row.index:
            row['Quarter'] = future_period.quarter
        if 'Month' in row.index:
            row['Month'] = ts.month
        if 'Week' in row.index:
            row['Week'] = int(ts.isocalendar().week)
        if 'Month_sin' in row.index:
            row['Month_sin'] = np.sin(2 * np.pi * ts.month / 12)
        if 'Month_cos' in row.index:
            row['Month_cos'] = np.cos(2 * np.pi * ts.month / 12)
        if 'Quarter_sin' in row.index:
            row['Quarter_sin'] = np.sin(2 * np.pi * future_period.quarter / 4)
        if 'Quarter_cos' in row.index:
            row['Quarter_cos'] = np.cos(2 * np.pi * future_period.quarter / 4)

        for col in row.index:
            lag_match = re.fullmatch(r'IPA_Price_TWD_lag(\d+)', col)
            if lag_match:
                lag = int(lag_match.group(1))
                if history.size >= lag:
                    row[col] = history[-lag]
                continue
            ma_match = re.fullmatch(r'IPA_Price_TWD_ma(\d+)', col)
            if ma_match:
                w = int(ma_match.group(1))
                d = history[-w:] if history.size >= w else history
                row[col] = float(np.mean(d)) if d.size else 0.0
                continue
            std_match = re.fullmatch(r'IPA_Price_TWD_std(\d+)', col)
            if std_match:
                w = int(std_match.group(1))
                d = history[-w:] if history.size >= w else history
                row[col] = float(np.std(d, ddof=0)) if d.size else 0.0
                continue
            max_match = re.fullmatch(r'IPA_Price_TWD_max(\d+)', col)
            if max_match:
                w = int(max_match.group(1))
                d = history[-w:] if history.size >= w else history
                row[col] = float(np.max(d)) if d.size else 0.0
                continue
            min_match = re.fullmatch(r'IPA_Price_TWD_min(\d+)', col)
            if min_match:
                w = int(min_match.group(1))
                d = history[-w:] if history.size >= w else history
                row[col] = float(np.min(d)) if d.size else 0.0
                continue
            if col == 'IPA_Price_TWD_pct_change':
                row[col] = self._safe_pct_change(history, periods=1)
            elif col == 'IPA_Price_TWD_monthly_change':
                row[col] = self._safe_pct_change(history, periods=4)
            elif col == 'IPA_Price_TWD_quarterly_change':
                row[col] = self._safe_pct_change(history, periods=13)

        return row.reindex(self.feature_cols).fillna(0.0).astype(float)

    def predict_future(self, quarters_ahead=4):
        print("\n" + "=" * 60)
        print(f"Predicting {self.target_year} Q1-Q4")
        print("=" * 60)

        total_steps, step_offset, all_periods, target_periods = self._compute_forecast_horizon(quarters_ahead)
        predictions = {}
        exog_projection = self._project_exogenous_features(all_periods)

        if 'xgboost' in self.models:
            xgb_path = []
            target_history = self.y.tolist()
            template_row = self.quarterly_data[self.feature_cols].iloc[-1].copy()
            for period in all_periods:
                exogenous_values = exog_projection.loc[period] if not exog_projection.empty else None
                feature_row = self._build_future_feature_row(
                    period,
                    target_history,
                    template_row,
                    exogenous_values=exogenous_values
                )
                next_pred = float(self.models['xgboost'].predict(feature_row.to_numpy(dtype=float).reshape(1, -1))[0])
                xgb_path.append(next_pred)
                target_history.append(next_pred)
                template_row = feature_row.copy()
            predictions['xgboost'] = xgb_path[step_offset:step_offset + quarters_ahead]

        if 'sarima' in self.models:
            try:
                sarima_values = np.asarray(self.models['sarima'].predict(total_steps), dtype=float)
                predictions['sarima'] = sarima_values[step_offset:step_offset + quarters_ahead].tolist()
            except Exception as e:
                print(f"SARIMA prediction failed: {e}")

        if 'xgboost' in predictions and 'sarima' in predictions:
            w_xgb = self.ensemble_weights.get('xgboost', 0.5)
            w_sarima = self.ensemble_weights.get('sarima', 0.5)
            total = w_xgb + w_sarima
            if total <= 0:
                w_xgb, w_sarima = 0.5, 0.5
            else:
                w_xgb, w_sarima = w_xgb / total, w_sarima / total
            predictions['ensemble'] = (
                np.asarray(predictions['xgboost'], dtype=float) * w_xgb +
                np.asarray(predictions['sarima'], dtype=float) * w_sarima
            ).tolist()
        elif 'xgboost' in predictions:
            predictions['ensemble'] = predictions['xgboost']
        elif 'sarima' in predictions:
            predictions['ensemble'] = predictions['sarima']
        else:
            fallback = float(np.mean(self.y[-4:])) if len(self.y) >= 4 else float(np.mean(self.y))
            predictions['ensemble'] = [fallback] * quarters_ahead

        base_pred = np.asarray(predictions['ensemble'], dtype=float)
        recent_history = self.y[-12:] if len(self.y) >= 12 else self.y
        mean_history = np.mean(np.abs(recent_history))
        market_volatility = float(np.std(recent_history) / mean_history) if mean_history > 1e-6 else 0.03

        residual_ratio = 0.03
        if self.y_test is not None and len(self.y_test) > 0:
            if 'xgboost_test' in self.predictions and 'sarima_test' in self.predictions:
                w_xgb = self.ensemble_weights.get('xgboost', 0.5)
                w_sarima = self.ensemble_weights.get('sarima', 0.5)
                total = w_xgb + w_sarima
                if total <= 0:
                    w_xgb, w_sarima = 0.5, 0.5
                else:
                    w_xgb, w_sarima = w_xgb / total, w_sarima / total
                base_test = (
                    np.asarray(self.predictions['xgboost_test']) * w_xgb +
                    np.asarray(self.predictions['sarima_test']) * w_sarima
                )
            elif 'xgboost_test' in self.predictions:
                base_test = np.asarray(self.predictions['xgboost_test'])
            elif 'sarima_test' in self.predictions:
                base_test = np.asarray(self.predictions['sarima_test'])
            else:
                base_test = None

            if base_test is not None:
                residual = np.asarray(self.y_test) - base_test
                denom = np.mean(np.abs(self.y_test))
                denom = denom if denom > 1e-6 else 1.0
                q75_abs_res = float(np.quantile(np.abs(residual), 0.75))
                residual_ratio = max(residual_ratio, q75_abs_res / denom)

        uncertainty_ratio = max(0.03, market_volatility, residual_ratio)
        predictions['optimistic'] = (base_pred * (1 + uncertainty_ratio)).tolist()
        predictions['pessimistic'] = (base_pred * (1 - uncertainty_ratio)).tolist()

        pred_df = pd.DataFrame({
            'Quarter': [str(period) for period in target_periods],
            'Base': predictions['ensemble'],
            'Optimistic': predictions['optimistic'],
            'Pessimistic': predictions['pessimistic']
        })
        if 'xgboost' in predictions:
            pred_df[self.model_display_names.get('xgboost', 'XGBoost')] = predictions['xgboost']
        if 'sarima' in predictions:
            pred_df[self.model_display_names.get('sarima', 'SARIMA')] = predictions['sarima']

        self.future_predictions = pred_df
        print("\n" + "-" * 40)
        print(f"{self.target_year} Price Prediction (TWD/KG)")
        print("-" * 40)
        print(pred_df.to_string(index=False))
        print(f"Adaptive uncertainty band: +/- {uncertainty_ratio * 100:.2f}%")
        return pred_df

    def create_visualizations(self, save_path=None):
        print("\n" + "=" * 60)
        print("Generating Visualization Charts")
        print("=" * 60)

        save_path = self.figure_dir if save_path is None else self._resolve_path(save_path)
        os.makedirs(save_path, exist_ok=True)
        figures = []

        fig1, ax1 = plt.subplots(figsize=(14, 6))
        ax1.plot(self.quarterly_data.index, self.quarterly_data['IPA_Price_TWD'], 'b-', linewidth=2, label='IPA Price')
        ax1.set_title('Isopropyl Alcohol (IPA) Historical Price Trend (2012-2024)', fontsize=14)
        ax1.set_xlabel('Time', fontsize=12)
        ax1.set_ylabel('Price (TWD/KG)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        fig1.savefig(f'{save_path}/historical_prices.png', dpi=120, bbox_inches='tight')
        figures.append(fig1)
        print("  [OK] Historical price trend chart")

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        pred_df = self.future_predictions
        x = range(len(pred_df))
        ax2.fill_between(x, pred_df['Pessimistic'], pred_df['Optimistic'], alpha=0.3, color='blue', label='Prediction Range')
        ax2.plot(x, pred_df['Base'], 'bo-', linewidth=2, markersize=10, label='Base Prediction')
        ax2.plot(x, pred_df['Optimistic'], 'g--', linewidth=1.5, label='Optimistic Scenario')
        ax2.plot(x, pred_df['Pessimistic'], 'r--', linewidth=1.5, label='Pessimistic Scenario')
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(pred_df['Quarter'])
        ax2.set_title(f'{self.target_year} IPA Price Prediction', fontsize=14)
        ax2.set_xlabel('Quarter', fontsize=12)
        ax2.set_ylabel('Price (TWD/KG)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        for i, row in pred_df.iterrows():
            ax2.annotate(f"{row['Base']:.1f}", (i, row['Base']), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
        fig2.savefig(f'{save_path}/prediction_{self.target_year}.png', dpi=120, bbox_inches='tight')
        figures.append(fig2)
        print(f"  [OK] {self.target_year} prediction chart")

        if self.results:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            model_names = [r['Model'] for r in self.results]
            mapes = [r['MAPE'] for r in self.results]
            palette = ['steelblue', 'seagreen', 'coral', 'slategray']
            bars = ax3.bar(model_names, mapes, color=palette[:len(model_names)])
            ax3.set_title('Model MAPE Comparison', fontsize=14)
            ax3.set_ylabel('MAPE (%)', fontsize=12)
            for bar, mape in zip(bars, mapes):
                ax3.annotate(f'{mape:.2f}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()), textcoords="offset points", xytext=(0, 5), ha='center')
            fig3.savefig(f'{save_path}/model_comparison.png', dpi=120, bbox_inches='tight')
            figures.append(fig3)
            print("  [OK] Model comparison chart")

        plt.close('all')
        print(f"\n[OK] All charts saved to: {save_path}/")
        return figures

    def generate_report(self, output_path=None):
        print("\n" + "=" * 60)
        print("Generating Analysis Report")
        print("=" * 60)

        output_path = self.report_dir if output_path is None else self._resolve_path(output_path)
        os.makedirs(output_path, exist_ok=True)
        rows_html = ''.join([
            f"<tr><td>{row['Quarter']}</td><td><strong>{row['Base']:.2f}</strong></td>"
            f"<td style='color:green'>{row['Optimistic']:.2f}</td>"
            f"<td style='color:red'>{row['Pessimistic']:.2f}</td></tr>"
            for _, row in self.future_predictions.iterrows()
        ])
        metrics_html_parts = []
        for r in self.results:
            cv_part = ""
            if r.get('CV_MAPE') is not None and np.isfinite(r.get('CV_MAPE')):
                cv_part = f" | CV MAPE {float(r['CV_MAPE']):.2f}%"
            metrics_html_parts.append(
                f"<div><strong>{r['Model']}</strong>: MAPE {r['MAPE']:.2f}%{cv_part}</div>"
            )
        metrics_html = ''.join(metrics_html_parts)
        weights_html = ''.join([
            f"<li>{self.model_display_names.get(k, k)}: {v:.2%}</li>"
            for k, v in self.ensemble_weights.items()
        ]) or "<li>N/A</li>"
        model_method = (
            f"{self.model_display_names.get('xgboost', 'XGBoost')} + {self.model_display_names.get('sarima', 'SARIMA')} adaptive ensemble"
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>IPA Price Prediction Report - {self.target_year}</title>
  <style>
    body {{ font-family: 'Segoe UI', 'Microsoft JhengHei', sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
    th {{ background: #3498db; color: #fff; }}
    .summary {{ background: #f3f8ff; padding: 16px; border-radius: 8px; }}
    .metrics {{ background: #f7f7f7; padding: 12px; border-radius: 8px; }}
    img {{ max-width: 100%; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>Isopropyl Alcohol (IPA) Price Prediction Report</h1>
  <p><strong>Forecast Year:</strong> {self.target_year} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
  <div class="summary">
    <p><strong>Base Mean:</strong> {self.future_predictions['Base'].mean():.2f} TWD/KG</p>
    <p><strong>Optimistic Mean:</strong> {self.future_predictions['Optimistic'].mean():.2f} TWD/KG</p>
    <p><strong>Pessimistic Mean:</strong> {self.future_predictions['Pessimistic'].mean():.2f} TWD/KG</p>
  </div>
  <h2>Quarterly Prediction</h2>
  <table>
    <tr><th>Quarter</th><th>Base</th><th>Optimistic</th><th>Pessimistic</th></tr>
    {rows_html}
  </table>
  <h2>Figures</h2>
  <p><img src="../figures/historical_prices.png" alt="Historical"></p>
  <p><img src="../figures/prediction_{self.target_year}.png" alt="Prediction"></p>
  <p><img src="../figures/model_comparison.png" alt="Model Comparison"></p>
  <h2>Model Evaluation</h2>
  <div class="metrics">{metrics_html}</div>
  <h2>Ensemble Weights</h2>
  <ul>{weights_html}</ul>
  <h2>Exported Files</h2>
  <ul>
    <li><a href="./ipa_forecast_{self.target_year}.csv">Quarterly forecast CSV</a></li>
    <li><a href="./model_metrics.csv">Model metrics CSV</a></li>
    <li><a href="./feature_importance.csv">Feature importance CSV</a></li>
    <li><a href="./sensitivity_analysis.csv">Sensitivity analysis CSV</a></li>
    <li><a href="./run_manifest_{self.target_year}.json">Run manifest JSON</a></li>
  </ul>
  <h2>Notes</h2>
  <ul>
    <li>Data range: January 2012 to December 2024</li>
    <li>Method: Walk-forward tuned {model_method}</li>
    <li>Factors: energy prices, exchange rates, geopolitical events, seasonality</li>
    <li>Scenario range: based on market volatility and residual distribution</li>
  </ul>
  <p>Disclaimer: prediction is for reference only.</p>
</body>
</html>
"""

        report_file = f'{output_path}/ipa_forecast_{self.target_year}.html'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[OK] HTML report saved: {report_file}")
        return report_file

    @staticmethod
    def _json_compatible(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        return value

    def save_artifacts(self, output_path=None):
        output_path = self.report_dir if output_path is None else self._resolve_path(output_path)
        os.makedirs(output_path, exist_ok=True)

        artifact_paths = {}

        if self.future_predictions is not None and not self.future_predictions.empty:
            pred_csv = os.path.join(output_path, f'ipa_forecast_{self.target_year}.csv')
            self.future_predictions.to_csv(pred_csv, index=False)
            artifact_paths['forecast_csv'] = pred_csv

        if self.results:
            metrics_df = pd.DataFrame(self.results)
            if 'ModelKey' in metrics_df.columns:
                metrics_df = metrics_df.drop(columns=['ModelKey'])
            metrics_csv = os.path.join(output_path, 'model_metrics.csv')
            metrics_df.to_csv(metrics_csv, index=False)
            artifact_paths['model_metrics_csv'] = metrics_csv

        if self.feature_importance_df is not None and not self.feature_importance_df.empty:
            fi_csv = os.path.join(output_path, 'feature_importance.csv')
            self.feature_importance_df.to_csv(fi_csv, index=False)
            artifact_paths['feature_importance_csv'] = fi_csv

        if self.sensitivity_results:
            sens_df = pd.DataFrame(
                [{'feature': k, 'impact_pct': v} for k, v in self.sensitivity_results.items()]
            )
            sens_csv = os.path.join(output_path, 'sensitivity_analysis.csv')
            sens_df.to_csv(sens_csv, index=False)
            artifact_paths['sensitivity_csv'] = sens_csv

        weights_json = os.path.join(output_path, 'ensemble_weights.json')
        weight_payload = {
            self.model_display_names.get(k, k): float(v)
            for k, v in self.ensemble_weights.items()
        }
        with open(weights_json, 'w', encoding='utf-8') as f:
            json.dump(weight_payload, f, ensure_ascii=False, indent=2)
        artifact_paths['ensemble_weights_json'] = weights_json

        manifest_json = os.path.join(output_path, f'run_manifest_{self.target_year}.json')
        manifest_payload = {
            'generated_at': self.generated_at,
            'target_year': int(self.target_year),
            'start_date': self.start_date,
            'end_date': self.end_date,
            'cache_dir': self.cache_dir,
            'use_cache': bool(self.use_cache),
            'refresh_cache': bool(self.refresh_cache),
            'random_seed': int(self.random_seed),
            'model_display_names': self.model_display_names,
            'ensemble_weights': {
                k: float(v) for k, v in self.ensemble_weights.items()
            },
        }
        with open(manifest_json, 'w', encoding='utf-8') as f:
            json.dump(manifest_payload, f, ensure_ascii=False, indent=2, default=self._json_compatible)
        artifact_paths['manifest_json'] = manifest_json

        self.artifacts[self.target_year] = artifact_paths
        print(f"[OK] Artifacts exported: {output_path}")
        return artifact_paths

    def run_full_pipeline(self, test_size=0.2):
        self.load_and_prepare_data()
        self.train_models(test_size=test_size)
        self.predict_future()
        self.create_visualizations(save_path=self.figure_dir)
        self.generate_report(output_path=self.report_dir)
        self.save_artifacts(output_path=self.report_dir)
        print("\n" + "=" * 60)
        print("[OK] Prediction pipeline complete")
        print("=" * 60)
        return self

    def run_forecasts_for_years(self, years, test_size=0.2):
        years = sorted(set(int(y) for y in years))
        if not years:
            raise ValueError("At least one forecast year is required.")
        self.target_year = years[0]
        self.load_and_prepare_data()
        self.train_models(test_size=test_size)

        outputs = {}
        summary_frames = []
        for year in years:
            self.target_year = year
            pred_df = self.predict_future()
            self.create_visualizations(save_path=self.figure_dir)
            report_file = self.generate_report(output_path=self.report_dir)
            artifact_paths = self.save_artifacts(output_path=self.report_dir)
            summary_frames.append(pred_df.assign(Forecast_Year=year))
            outputs[year] = {
                'prediction': pred_df.copy(),
                'report_file': report_file,
                'artifacts': artifact_paths
            }

        if summary_frames:
            summary_df = pd.concat(summary_frames, axis=0, ignore_index=True)
            summary_file = os.path.join(self.report_dir, 'ipa_forecast_all_years.csv')
            summary_df.to_csv(summary_file, index=False)
            print(f"[OK] Multi-year summary CSV saved: {summary_file}")

        print("\n" + "=" * 60)
        print(f"[OK] Multi-year forecasting complete: {', '.join(str(y) for y in years)}")
        print("=" * 60)
        return outputs


def parse_args():
    parser = argparse.ArgumentParser(description="IPA quarterly price forecasting pipeline.")
    parser.add_argument('--years', nargs='+', type=int, default=[2025, 2026], help='Forecast years, e.g. --years 2025 2026')
    parser.add_argument('--start-date', type=str, default='2012-01-01', help='Historical start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-12-31', help='Historical end date (YYYY-MM-DD)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test ratio for final time split (default: 0.2)')
    parser.add_argument('--cache-dir', type=str, default='Data', help='Cache directory for merged market data')
    parser.add_argument('--figures-dir', type=str, default='figures', help='Directory for output figures')
    parser.add_argument('--reports-dir', type=str, default='reports', help='Directory for output reports and CSV/JSON artifacts')
    parser.add_argument('--no-cache', action='store_true', help='Disable local cache')
    parser.add_argument('--refresh-cache', action='store_true', help='Force refresh market data from source')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predictor = IPAPricePredictor(
        target_year=min(args.years),
        start_date=args.start_date,
        end_date=args.end_date,
        cache_dir=args.cache_dir,
        figures_dir=args.figures_dir,
        reports_dir=args.reports_dir,
        use_cache=not args.no_cache,
        refresh_cache=args.refresh_cache,
        random_seed=args.seed
    )
    predictor.run_forecasts_for_years(args.years, test_size=args.test_size)

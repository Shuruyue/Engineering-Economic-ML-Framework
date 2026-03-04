"""
IPA Price Prediction Model - Feature Engineering Module
Feature Engineering for Isopropyl Alcohol Price Prediction

This module is responsible for:
- Extracting IPA price data from charts
- Creating lag features
- Creating rolling statistics features
- Seasonal features
- Data normalization
"""

from __future__ import annotations

import json
import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)

# Default path for external IPA price JSON data
_DEFAULT_IPA_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'Data', 'ipa_prices.json',
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LAG_PERIODS: List[int] = [1, 2, 3, 4]
DEFAULT_WEEKLY_ROLLING_WINDOWS: List[int] = [4, 8, 12]
DEFAULT_QUARTERLY_ROLLING_WINDOWS: List[int] = [2, 4, 8]
DEFAULT_EXOGENOUS_WINDOWS: List[int] = [4, 8]
DEFAULT_QUARTERLY_EXOGENOUS_WINDOWS: List[int] = [2, 4]

EXOGENOUS_COLS: List[str] = [
    'WTI_Price', 'Brent_Price', 'NatGas_Price',
    'USD_TWD', 'DXY', 'TWII', 'TSM_Price',
]

INTERACTION_PAIRS: List[Tuple[str, str]] = [
    ('WTI_Price', 'USD_TWD'),
    ('Brent_Price', 'USD_TWD'),
    ('NatGas_Price', 'USD_TWD'),
]


class IPAFeatureEngineer:
    """Isopropyl Alcohol Price Prediction - Feature Engineer."""

    # Re-export for backward compatibility
    EXOGENOUS_COLS = EXOGENOUS_COLS
    INTERACTION_PAIRS = INTERACTION_PAIRS

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_target_series(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("IPA target series is empty.")
        if target_col not in df.columns:
            raise ValueError(f"Missing target column: {target_col}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Target index must be DatetimeIndex.")
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        if df[target_col].isna().all():
            raise ValueError("Target series contains only NaN values.")
        return df

    # ------------------------------------------------------------------
    # Price data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_price_points_from_json(json_path: str) -> List[Tuple[str, float]]:
        """Load IPA price points from external JSON file.

        Parameters
        ----------
        json_path : str
            Absolute path to the JSON file containing price data.

        Returns
        -------
        list[tuple[str, float]]
            List of (date_str, price) tuples.
        """
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        entries = data.get('prices', [])
        if not entries:
            raise ValueError(f"No price entries found in {json_path}")
        return [(entry['date'], entry['price']) for entry in entries]

    @staticmethod
    def _get_hardcoded_price_points() -> List[Tuple[str, float]]:
        """Hardcoded IPA price points used as fallback when JSON file is unavailable.

        Data Source: User-provided price trend chart (2012/01 - 2024/12)
        Unit: TWD (NTD/KG)
        """
        return [
            # 2012
            ('2012-01', 48), ('2012-02', 47), ('2012-03', 47), ('2012-04', 47),
            ('2012-05', 46), ('2012-06', 45), ('2012-07', 45), ('2012-08', 46),
            ('2012-09', 47), ('2012-10', 48), ('2012-11', 48), ('2012-12', 47),
            # 2013
            ('2013-01', 47), ('2013-02', 48), ('2013-03', 49), ('2013-04', 49),
            ('2013-05', 49), ('2013-06', 48), ('2013-07', 47), ('2013-08', 47),
            ('2013-09', 47), ('2013-10', 47), ('2013-11', 47), ('2013-12', 48),
            # 2014
            ('2014-01', 48), ('2014-02', 50), ('2014-03', 52), ('2014-04', 52),
            ('2014-05', 51), ('2014-06', 50), ('2014-07', 48), ('2014-08', 45),
            ('2014-09', 42), ('2014-10', 38), ('2014-11', 35), ('2014-12', 32),
            # 2015
            ('2015-01', 30), ('2015-02', 28), ('2015-03', 27), ('2015-04', 28),
            ('2015-05', 29), ('2015-06', 30), ('2015-07', 30), ('2015-08', 28),
            ('2015-09', 27), ('2015-10', 27), ('2015-11', 27), ('2015-12', 27),
            # 2016
            ('2016-01', 27), ('2016-02', 27), ('2016-03', 28), ('2016-04', 30),
            ('2016-05', 32), ('2016-06', 33), ('2016-07', 34), ('2016-08', 35),
            ('2016-09', 35), ('2016-10', 36), ('2016-11', 37), ('2016-12', 38),
            # 2017
            ('2017-01', 39), ('2017-02', 40), ('2017-03', 40), ('2017-04', 39),
            ('2017-05', 38), ('2017-06', 37), ('2017-07', 37), ('2017-08', 38),
            ('2017-09', 38), ('2017-10', 39), ('2017-11', 40), ('2017-12', 41),
            # 2018
            ('2018-01', 42), ('2018-02', 42), ('2018-03', 41), ('2018-04', 40),
            ('2018-05', 40), ('2018-06', 39), ('2018-07', 39), ('2018-08', 39),
            ('2018-09', 39), ('2018-10', 40), ('2018-11', 40), ('2018-12', 39),
            # 2019
            ('2019-01', 38), ('2019-02', 37), ('2019-03', 36), ('2019-04', 36),
            ('2019-05', 36), ('2019-06', 35), ('2019-07', 35), ('2019-08', 35),
            ('2019-09', 35), ('2019-10', 35), ('2019-11', 36), ('2019-12', 37),
            # 2020
            ('2020-01', 38), ('2020-02', 42), ('2020-03', 48), ('2020-04', 50),
            ('2020-05', 48), ('2020-06', 45), ('2020-07', 43), ('2020-08', 42),
            ('2020-09', 42), ('2020-10', 43), ('2020-11', 44), ('2020-12', 45),
            # 2021
            ('2021-01', 48), ('2021-02', 52), ('2021-03', 55), ('2021-04', 56),
            ('2021-05', 55), ('2021-06', 52), ('2021-07', 50), ('2021-08', 48),
            ('2021-09', 47), ('2021-10', 47), ('2021-11', 47), ('2021-12', 46),
            # 2022
            ('2022-01', 46), ('2022-02', 47), ('2022-03', 50), ('2022-04', 52),
            ('2022-05', 50), ('2022-06', 48), ('2022-07', 46), ('2022-08', 44),
            ('2022-09', 43), ('2022-10', 42), ('2022-11', 42), ('2022-12', 42),
            # 2023
            ('2023-01', 43), ('2023-02', 44), ('2023-03', 44), ('2023-04', 43),
            ('2023-05', 42), ('2023-06', 41), ('2023-07', 40), ('2023-08', 40),
            ('2023-09', 40), ('2023-10', 41), ('2023-11', 42), ('2023-12', 43),
            # 2024
            ('2024-01', 44), ('2024-02', 46), ('2024-03', 48), ('2024-04', 50),
            ('2024-05', 52), ('2024-06', 53), ('2024-07', 52), ('2024-08', 50),
            ('2024-09', 48), ('2024-10', 45), ('2024-11', 43), ('2024-12', 41),
        ]

    def create_ipa_price_data(
        self,
        interpolation: str = 'cubic',
        json_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Create IPA price data from external JSON or hardcoded fallback.

        Parameters
        ----------
        interpolation : str
            Interpolation method for weekly resampling ('cubic' or 'linear').
        json_path : str, optional
            Path to JSON file. Defaults to ``Data/ipa_prices.json`` next to this module.

        Returns
        -------
        pd.DataFrame
            Weekly IPA price data with DatetimeIndex.
        """
        json_path = json_path or _DEFAULT_IPA_JSON_PATH

        try:
            price_points = self._load_price_points_from_json(json_path)
            print(f"[OK] Loaded IPA prices from: {json_path} ({len(price_points)} entries)")
        except Exception as e:
            logger.warning("Failed to load JSON (%s), using hardcoded fallback", e)
            print(f"[WARN] Failed to load JSON ({e}), using hardcoded fallback")
            price_points = self._get_hardcoded_price_points()

        dates = [pd.to_datetime(p[0]) for p in price_points]
        prices = [p[1] for p in price_points]

        ipa_df = pd.DataFrame({'IPA_Price_TWD': prices}, index=dates)
        ipa_df.index.name = 'Date'
        ipa_df = self._validate_target_series(ipa_df, 'IPA_Price_TWD')

        # Interpolate to weekly data
        ipa_weekly = ipa_df.resample('W-SUN').mean()
        try:
            ipa_weekly = ipa_weekly.interpolate(method=interpolation)
        except Exception:
            ipa_weekly = ipa_weekly.interpolate(method='linear')
        ipa_weekly = ipa_weekly.ffill().bfill()
        ipa_weekly = self._validate_target_series(ipa_weekly, 'IPA_Price_TWD')

        return ipa_weekly

    # ------------------------------------------------------------------
    # Feature creation
    # ------------------------------------------------------------------

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Create lag features for *target_col*."""
        lags = lags if lags is not None else DEFAULT_LAG_PERIODS
        result = df.copy()
        for lag in lags:
            result[f'{target_col}_lag{lag}'] = result[target_col].shift(lag)
        return result

    def create_rolling_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Create rolling statistics features (mean, std, max, min)."""
        windows = windows if windows is not None else DEFAULT_WEEKLY_ROLLING_WINDOWS
        result = df.copy()
        for window in windows:
            result[f'{target_col}_ma{window}'] = result[target_col].rolling(window=window).mean()
            result[f'{target_col}_std{window}'] = result[target_col].rolling(window=window).std()
            result[f'{target_col}_max{window}'] = result[target_col].rolling(window=window).max()
            result[f'{target_col}_min{window}'] = result[target_col].rolling(window=window).min()
        return result

    def create_change_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        monthly_period: int = 4,
        quarterly_period: int = 13,
    ) -> pd.DataFrame:
        """Create rate-of-change features."""
        result = df.copy()
        result[f'{target_col}_pct_change'] = result[target_col].pct_change()
        result[f'{target_col}_monthly_change'] = result[target_col].pct_change(periods=monthly_period)
        result[f'{target_col}_quarterly_change'] = result[target_col].pct_change(periods=quarterly_period)
        return result

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features with cyclical encoding."""
        result = df.copy()

        result['Year'] = result.index.year
        result['Month'] = result.index.month
        result['Quarter'] = result.index.quarter
        result['Week'] = result.index.isocalendar().week.astype(int)

        # Seasonal encoding (sine/cosine)
        result['Month_sin'] = np.sin(2 * np.pi * result['Month'] / 12)
        result['Month_cos'] = np.cos(2 * np.pi * result['Month'] / 12)
        result['Quarter_sin'] = np.sin(2 * np.pi * result['Quarter'] / 4)
        result['Quarter_cos'] = np.cos(2 * np.pi * result['Quarter'] / 4)

        return result

    def create_exogenous_features(
        self,
        df: pd.DataFrame,
        windows: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Create lag and rolling features for exogenous variables, plus interaction features."""
        windows = windows or DEFAULT_EXOGENOUS_WINDOWS
        result = df.copy()

        for col in EXOGENOUS_COLS:
            if col not in result.columns:
                continue
            result[f'{col}_lag1'] = result[col].shift(1)
            for w in windows:
                result[f'{col}_ma{w}'] = result[col].rolling(window=w, min_periods=1).mean()

        for col_a, col_b in INTERACTION_PAIRS:
            if col_a in result.columns and col_b in result.columns:
                result[f'{col_a}_x_{col_b}'] = result[col_a] * result[col_b]

        return result

    # ------------------------------------------------------------------
    # Resample
    # ------------------------------------------------------------------

    def resample_to_quarterly(
        self,
        df: pd.DataFrame,
        target_col: str = 'IPA_Price_TWD',
    ) -> pd.DataFrame:
        """Convert weekly data to quarterly averages."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        quarterly = df[numeric_cols].resample('QE').mean()
        quarterly['Quarter_Label'] = quarterly.index.to_period('Q').astype(str)
        return quarterly

    # ------------------------------------------------------------------
    # Full pipelines
    # ------------------------------------------------------------------

    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'IPA_Price_TWD',
    ) -> pd.DataFrame:
        """Complete weekly feature engineering pipeline."""
        print("Starting feature engineering...")
        df = self._validate_target_series(df, target_col)

        print("  - Creating lag features...")
        df = self.create_lag_features(df, target_col)
        print("  - Creating rolling statistics features...")
        df = self.create_rolling_features(df, target_col)
        print("  - Creating rate of change features...")
        df = self.create_change_features(df, target_col)
        print("  - Creating time features...")
        df = self.create_time_features(df)
        print("  - Creating exogenous derived features...")
        df = self.create_exogenous_features(df)

        df = df.dropna()
        print(f"[OK] Feature engineering complete! Total {len(df.columns)} features")
        return df

    def prepare_quarterly_features(
        self,
        quarterly_df: pd.DataFrame,
        target_col: str = 'IPA_Price_TWD',
    ) -> pd.DataFrame:
        """Quarterly-first feature engineering pipeline.

        Avoids averaging already-derived weekly lag/rolling features into quarter
        buckets, which can distort time semantics.
        """
        print("Starting quarterly feature engineering...")
        df = self._validate_target_series(quarterly_df.copy(), target_col)

        df = self.create_lag_features(df, target_col, lags=DEFAULT_LAG_PERIODS)
        df = self.create_rolling_features(df, target_col, windows=DEFAULT_QUARTERLY_ROLLING_WINDOWS)
        df = self.create_change_features(df, target_col, monthly_period=2, quarterly_period=4)
        df = self.create_time_features(df)
        df = self.create_exogenous_features(df, windows=DEFAULT_QUARTERLY_EXOGENOUS_WINDOWS)
        df = df.dropna()

        print(f"[OK] Quarterly feature engineering complete! Total {len(df.columns)} features")
        return df

    # ------------------------------------------------------------------
    # Scaling
    # ------------------------------------------------------------------

    def scale_features(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        fit: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Standardize features and optionally scale the target."""
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            if y is not None:
                y_scaled = self.target_scaler.fit_transform(y.values.reshape(-1, 1))
                return X_scaled, y_scaled.flatten()
        else:
            X_scaled = self.scaler.transform(X)
            if y is not None:
                y_scaled = self.target_scaler.transform(y.values.reshape(-1, 1))
                return X_scaled, y_scaled.flatten()

        return X_scaled, y

    def inverse_scale_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """Inverse transform target variable."""
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


# Main test program
if __name__ == "__main__":
    fe = IPAFeatureEngineer()

    print("Creating IPA price data from chart...")
    ipa_data = fe.create_ipa_price_data()
    print(f"Weekly data count: {len(ipa_data)}")
    print(ipa_data.head(10))
    print(ipa_data.tail(10))

    featured_data = fe.prepare_features(ipa_data)
    print(f"\nData count after feature engineering: {len(featured_data)}")
    print("Column list:")
    print(featured_data.columns.tolist())

    quarterly = fe.resample_to_quarterly(featured_data)
    print(f"\nQuarterly data count: {len(quarterly)}")
    print(quarterly.tail(10))

"""
Unit tests for IPA Price Prediction main module (IPAPricePredictor utilities)
"""

import os
import re
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ipa_price_prediction import IPAPricePredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def predictor():
    """Create a predictor without running data loading."""
    p = IPAPricePredictor(target_year=2025, random_seed=42)
    return p


# ---------------------------------------------------------------------------
# Static / utility methods
# ---------------------------------------------------------------------------

class TestSafeMAPE:
    def test_normal_case(self):
        y_true = np.array([40.0, 42.0, 44.0])
        y_pred = np.array([41.0, 43.0, 43.0])
        result = IPAPricePredictor._safe_mape(y_true, y_pred)
        assert isinstance(result, float)
        assert result > 0

    def test_zero_denominator_protection(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 1.0])
        # Should not raise - uses 1.0 when denominator is near zero
        result = IPAPricePredictor._safe_mape(y_true, y_pred)
        assert np.isfinite(result)

    def test_perfect_prediction(self):
        y = np.array([40.0, 42.0])
        result = IPAPricePredictor._safe_mape(y, y)
        assert result == pytest.approx(0.0)


class TestIsTargetDerivedFeature:
    def test_lag_feature(self):
        assert IPAPricePredictor._is_target_derived_feature('IPA_Price_TWD_lag1')

    def test_ma_feature(self):
        assert IPAPricePredictor._is_target_derived_feature('IPA_Price_TWD_ma4')

    def test_non_target_feature(self):
        assert not IPAPricePredictor._is_target_derived_feature('WTI_Price')
        assert not IPAPricePredictor._is_target_derived_feature('Year')


class TestSafePctChange:
    def test_normal(self):
        values = np.array([40.0, 42.0, 44.0])
        result = IPAPricePredictor._safe_pct_change(values, periods=1)
        # (44 - 42) / 42
        assert result == pytest.approx((44.0 - 42.0) / 42.0)

    def test_too_short(self):
        values = np.array([40.0])
        result = IPAPricePredictor._safe_pct_change(values, periods=1)
        assert result == 0.0

    def test_zero_base(self):
        values = np.array([0.0, 1.0])
        result = IPAPricePredictor._safe_pct_change(values, periods=1)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Clip outliers
# ---------------------------------------------------------------------------

class TestClipOutliers:
    def test_basic_clipping(self, predictor):
        df = pd.DataFrame({
            'a': [1, 2, 3, 100, 5, 6, 7, 8, 9, 10],
            'b': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        })
        result = predictor._clip_outliers(df, quantile_low=0.1, quantile_high=0.9)
        assert result['a'].max() <= df['a'].quantile(0.9)

    def test_exclude_cols(self, predictor):
        df = pd.DataFrame({
            'IPA_Price_TWD': [1, 2, 3, 100, 5],
            'other': [1, 2, 3, 100, 5],
        })
        result = predictor._clip_outliers(df, exclude_cols=['IPA_Price_TWD'])
        # Excluded column should remain unchanged
        assert result['IPA_Price_TWD'].max() == 100
        # Other column should be clipped
        assert result['other'].max() <= df['other'].quantile(0.99)


# ---------------------------------------------------------------------------
# Ensemble weight derivation
# ---------------------------------------------------------------------------

class TestDeriveEnsembleWeights:
    def test_basic_weights(self, predictor):
        predictor.results = [
            {'ModelKey': 'xgboost', 'MAPE': 2.0},
            {'ModelKey': 'sarima', 'MAPE': 8.0},
        ]
        predictor._derive_ensemble_weights()
        assert 'xgboost' in predictor.ensemble_weights
        assert 'sarima' in predictor.ensemble_weights
        # XGBoost should have higher weight (lower MAPE)
        assert predictor.ensemble_weights['xgboost'] > predictor.ensemble_weights['sarima']
        # Weights should sum to 1
        total = sum(predictor.ensemble_weights.values())
        assert total == pytest.approx(1.0)

    def test_no_valid_results(self, predictor):
        predictor.results = [
            {'ModelKey': 'unknown', 'MAPE': 5.0},
        ]
        predictor._derive_ensemble_weights()
        assert predictor.ensemble_weights == {}

    def test_nan_mape_excluded(self, predictor):
        predictor.results = [
            {'ModelKey': 'xgboost', 'MAPE': 2.0},
            {'ModelKey': 'sarima', 'MAPE': float('nan')},
        ]
        predictor._derive_ensemble_weights()
        assert 'sarima' not in predictor.ensemble_weights
        assert predictor.ensemble_weights['xgboost'] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Build future feature row
# ---------------------------------------------------------------------------

class TestBuildFutureFeatureRow:
    def test_lag_features_populated(self, predictor):
        feature_cols = ['IPA_Price_TWD_lag1', 'IPA_Price_TWD_lag2', 'Year', 'Quarter']
        predictor.feature_cols = feature_cols
        predictor.exogenous_feature_cols = []

        template = pd.Series(
            [0.0, 0.0, 2024, 4],
            index=feature_cols,
            dtype=float,
        )
        target_history = [40.0, 42.0, 44.0, 46.0]
        future_period = pd.Period('2025Q1', freq='Q')

        row = predictor._build_future_feature_row(future_period, target_history, template)

        assert row['IPA_Price_TWD_lag1'] == 46.0  # last value
        assert row['IPA_Price_TWD_lag2'] == 44.0  # second last
        assert row['Year'] == 2025
        assert row['Quarter'] == 1


# ---------------------------------------------------------------------------
# JSON serialization utility
# ---------------------------------------------------------------------------

class TestJsonCompatible:
    def test_numpy_int(self):
        assert IPAPricePredictor._json_compatible(np.int64(42)) == 42

    def test_numpy_float(self):
        result = IPAPricePredictor._json_compatible(np.float64(3.14))
        assert isinstance(result, float)

    def test_numpy_array(self):
        result = IPAPricePredictor._json_compatible(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_timestamp(self):
        ts = pd.Timestamp('2025-01-01')
        result = IPAPricePredictor._json_compatible(ts)
        assert isinstance(result, str)

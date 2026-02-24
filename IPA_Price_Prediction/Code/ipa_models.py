"""
IPA Price Prediction Model - Machine Learning Models Module
ML Models for Isopropyl Alcohol Price Prediction

Models included:
- ARIMA/SARIMA time series model
- LSTM neural network
- XGBoost gradient boosting
- Ensemble model
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import optional packages
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.callbacks import EarlyStopping
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False


class ModelEvaluator:
    """
    Model Evaluator
    """
    
    @staticmethod
    def evaluate(y_true, y_pred, model_name="Model"):
        """Calculate evaluation metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
        
        # Direction accuracy
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_acc = np.mean(true_direction == pred_direction) * 100
        else:
            direction_acc = 0
            
        results = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Direction_Accuracy': direction_acc
        }
        
        return results
    
    @staticmethod
    def print_results(results):
        """Print evaluation results"""
        print(f"\n{'='*50}")
        print(f"Model: {results['Model']}")
        print(f"{'='*50}")
        print(f"MAE:  {results['MAE']:.4f}")
        print(f"RMSE: {results['RMSE']:.4f}")
        print(f"MAPE: {results['MAPE']:.2f}%")
        print(f"Direction Accuracy: {results['Direction_Accuracy']:.2f}%")


class ARIMAModel:
    """
    ARIMA Time Series Model
    """
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 4)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        
    def fit(self, y, exog=None):
        """Train model"""
        if not ARIMA_AVAILABLE:
            raise ImportError("statsmodels not installed")
            
        print(f"Training SARIMA{self.order}x{self.seasonal_order}...")
        
        self.model = SARIMAX(
            y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        self.fitted_model = self.model.fit(disp=False)
        print("[OK] SARIMA training complete")
        
        return self
    
    def predict(self, steps, exog=None):
        """Predict"""
        if self.fitted_model is None:
            raise ValueError("Model not trained yet")
            
        forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        return forecast
    
    def get_fitted_values(self):
        """Get fitted values"""
        return self.fitted_model.fittedvalues


class LSTMModel:
    """
    LSTM Neural Network Model
    """
    
    def __init__(self, lookback=8, units=64, dropout=0.2, epochs=100):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.epochs = epochs
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def _prepare_sequences(self, X, y=None):
        """Prepare LSTM sequence data"""
        X_seq = []
        y_seq = []
        
        for i in range(self.lookback, len(X)):
            X_seq.append(X[i-self.lookback:i])
            if y is not None:
                y_seq.append(y[i])
                
        X_seq = np.array(X_seq)
        
        if y is not None:
            y_seq = np.array(y_seq)
            return X_seq, y_seq
        return X_seq
    
    def build_model(self, input_shape):
        """Build LSTM model architecture"""
        if not LSTM_AVAILABLE:
            raise ImportError("TensorFlow not installed")
            
        model = Sequential([
            Bidirectional(LSTM(self.units, return_sequences=True), 
                         input_shape=input_shape),
            Dropout(self.dropout),
            LSTM(self.units // 2),
            Dropout(self.dropout),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def fit(self, X, y, validation_split=0.2):
        """Train model"""
        print("Training LSTM neural network...")
        
        # Prepare sequence data
        X_seq, y_seq = self._prepare_sequences(X, y)
        
        # Build model
        self.model = self.build_model((X_seq.shape[1], X_seq.shape[2]))
        
        # Early stopping
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train
        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=16,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=0
        )
        
        print(f"[OK] LSTM training complete (epochs: {len(history.history['loss'])})")
        
        return self, history
    
    def predict(self, X):
        """Predict"""
        X_seq = self._prepare_sequences(X)
        predictions = self.model.predict(X_seq, verbose=0)
        return predictions.flatten()


class XGBoostModel:
    """
    XGBoost Gradient Boosting Model
    """
    
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, subsample=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.model = None
        self.backend_name = None
        self.display_name = None
        
    def fit(self, X, y):
        """Train model"""
        if not XGB_AVAILABLE:
            # Use sklearn's GradientBoosting as alternative
            print("XGBoost unavailable, using GradientBoostingRegressor backend...")
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42
            )
            self.backend_name = 'gradient_boosting'
            self.display_name = 'GradientBoosting'
        else:
            print("Training XGBoost...")
            self.model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=42,
                n_jobs=1
            )
            self.backend_name = 'xgboost'
            self.display_name = 'XGBoost'
            
        self.model.fit(X, y)
        print(f"[OK] {self.display_name} training complete")
        
        return self
    
    def predict(self, X):
        """Predict"""
        return self.model.predict(X)
    
    def get_feature_importance(self, feature_names):
        """Get feature importance"""
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def get_display_name(self):
        if self.display_name:
            return self.display_name
        return 'XGBoost'


# Main test program
if __name__ == "__main__":
    print("Model Module Test")
    print("="*50)
    
    # Generate test data
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 + 40
    
    # Split data
    train_size = int(n_samples * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Test XGBoost
    print("\nTesting XGBoost model...")
    xgb_model = XGBoostModel(n_estimators=50)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    
    results = ModelEvaluator.evaluate(y_test, xgb_pred, "XGBoost")
    ModelEvaluator.print_results(results)
    
    print("\n[OK] Model module test complete!")

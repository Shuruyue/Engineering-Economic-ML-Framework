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

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class IPAFeatureEngineer:
    """
    Isopropyl Alcohol Price Prediction - Feature Engineer
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
    def create_ipa_price_data(self):
        """
        Create IPA price data based on user-provided chart
        
        Data Source: User-provided price trend chart (2012/01 - 2025/12/27)
        Unit: TWD (NTD/KG)
        
        Sampling Method: One point per week, extracted from chart visualization
        """
        
        # Key price points observed from chart (monthly average estimates)
        # Format: (year-month, price)
        price_points = [
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
            
            # 2015 (Significant decline)
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
            
            # 2020 (COVID impact)
            ('2020-01', 38), ('2020-02', 42), ('2020-03', 48), ('2020-04', 50),
            ('2020-05', 48), ('2020-06', 45), ('2020-07', 43), ('2020-08', 42),
            ('2020-09', 42), ('2020-10', 43), ('2020-11', 44), ('2020-12', 45),
            
            # 2021 (Price peak)
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
        
        # Convert to DataFrame
        dates = [pd.to_datetime(p[0]) for p in price_points]
        prices = [p[1] for p in price_points]
        
        ipa_df = pd.DataFrame({
            'IPA_Price_TWD': prices
        }, index=dates)
        ipa_df.index.name = 'Date'
        
        # Interpolate to weekly data
        ipa_weekly = ipa_df.resample('W').interpolate(method='cubic')
        
        return ipa_weekly
    
    def create_lag_features(self, df, target_col, lags=[1, 2, 3, 4]):
        """
        Create lag features
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        target_col : str
            Target column name
        lags : list
            List of lag periods
        """
        result = df.copy()
        
        for lag in lags:
            result[f'{target_col}_lag{lag}'] = result[target_col].shift(lag)
            
        return result
    
    def create_rolling_features(self, df, target_col, windows=[4, 8, 12]):
        """
        Create rolling statistics features
        
        Parameters:
        -----------
        df : DataFrame
            Input data
        target_col : str
            Target column name
        windows : list
            Rolling window sizes (in weeks)
        """
        result = df.copy()
        
        for window in windows:
            # Rolling mean
            result[f'{target_col}_ma{window}'] = result[target_col].rolling(window=window).mean()
            # Rolling standard deviation
            result[f'{target_col}_std{window}'] = result[target_col].rolling(window=window).std()
            # Rolling max and min
            result[f'{target_col}_max{window}'] = result[target_col].rolling(window=window).max()
            result[f'{target_col}_min{window}'] = result[target_col].rolling(window=window).min()
            
        return result
    
    def create_change_features(self, df, target_col):
        """
        Create rate of change features
        """
        result = df.copy()
        
        # Weekly change rate
        result[f'{target_col}_pct_change'] = result[target_col].pct_change()
        
        # Monthly change rate (4 weeks)
        result[f'{target_col}_monthly_change'] = result[target_col].pct_change(periods=4)
        
        # Quarterly change rate (13 weeks)
        result[f'{target_col}_quarterly_change'] = result[target_col].pct_change(periods=13)
        
        return result
    
    def create_time_features(self, df):
        """
        Create time-based features
        """
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
    
    def resample_to_quarterly(self, df, target_col='IPA_Price_TWD'):
        """
        Convert weekly data to quarterly data
        """
        # Average numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        quarterly = df[numeric_cols].resample('QE').mean()
        
        # Add quarter labels
        quarterly['Quarter_Label'] = quarterly.index.to_period('Q').astype(str)
        
        return quarterly
    
    def prepare_features(self, df, target_col='IPA_Price_TWD'):
        """
        Complete feature engineering pipeline
        """
        print("Starting feature engineering...")
        
        # 1. Lag features
        print("  - Creating lag features...")
        df = self.create_lag_features(df, target_col)
        
        # 2. Rolling statistics
        print("  - Creating rolling statistics features...")
        df = self.create_rolling_features(df, target_col)
        
        # 3. Rate of change
        print("  - Creating rate of change features...")
        df = self.create_change_features(df, target_col)
        
        # 4. Time features
        print("  - Creating time features...")
        df = self.create_time_features(df)
        
        # Remove missing values
        df = df.dropna()
        
        print(f"[OK] Feature engineering complete! Total {len(df.columns)} features")
        
        return df
    
    def scale_features(self, X, y=None, fit=True):
        """
        Standardize features
        """
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
    
    def inverse_scale_target(self, y_scaled):
        """
        Inverse transform target variable
        """
        return self.target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


# Main test program
if __name__ == "__main__":
    # Create feature engineer
    fe = IPAFeatureEngineer()
    
    # Create IPA price data
    print("Creating IPA price data from chart...")
    ipa_data = fe.create_ipa_price_data()
    print(f"Weekly data count: {len(ipa_data)}")
    print(ipa_data.head(10))
    print(ipa_data.tail(10))
    
    # Perform feature engineering
    featured_data = fe.prepare_features(ipa_data)
    print(f"\nData count after feature engineering: {len(featured_data)}")
    print("Column list:")
    print(featured_data.columns.tolist())
    
    # Convert to quarterly
    quarterly = fe.resample_to_quarterly(featured_data)
    print(f"\nQuarterly data count: {len(quarterly)}")
    print(quarterly.tail(10))

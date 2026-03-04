"""
IPA Price Prediction Model - Data Collection Module
Data Collector for Isopropyl Alcohol Price Prediction

This module is responsible for collecting data from multiple sources:
- Yahoo Finance (yfinance): Crude oil, natural gas, exchange rates
- Event indicators: Geopolitical and macro shock proxies
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Data collection packages
try:
    import yfinance as yf
except ImportError:
    yf = None

class IPADataCollector:
    """
    Isopropyl Alcohol Price Prediction - Data Collector
    """

    MARKET_COLUMNS = [
        'WTI_Price', 'Brent_Price', 'NatGas_Price',
        'USD_TWD', 'DXY', 'TWII', 'TSM_Price'
    ]
    EVENT_COLUMNS = [
        'COVID19', 'Ukraine_War', 'RedSea_Crisis',
        'US_China_Trade', 'Geopolitical_Risk'
    ]
    REQUIRED_CACHE_COLUMNS = MARKET_COLUMNS + EVENT_COLUMNS
    
    def __init__(
        self,
        start_date='2012-01-01',
        end_date='2024-12-31',
        cache_dir='Data',
        use_cache=True,
        refresh_cache=False
    ):
        """
        Initialize data collector
        
        Parameters:
        -----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        cache_dir : str
            Local cache directory for merged market data
        use_cache : bool
            Whether to load/store cache
        refresh_cache : bool
            Whether to force refresh from remote source
        """
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.refresh_cache = refresh_cache
        self.data = {}

        safe_start = self.start_date.replace('-', '')
        safe_end = self.end_date.replace('-', '')
        self.cache_file = os.path.join(
            self.cache_dir,
            f"market_data_{safe_start}_{safe_end}.csv"
        )

    def _validate_cached_data(self, df):
        if df is None or df.empty:
            return False, "cached data is empty"
        if not isinstance(df.index, pd.DatetimeIndex):
            return False, "cache index is not DatetimeIndex"

        missing_columns = [col for col in self.REQUIRED_CACHE_COLUMNS if col not in df.columns]
        if missing_columns:
            return False, f"missing columns: {missing_columns}"

        return True, "ok"

    def _normalize_output_schema(self, df):
        """
        Ensure merged output keeps stable schema for downstream steps.
        """
        result = df.copy()
        for col in self.REQUIRED_CACHE_COLUMNS:
            if col not in result.columns:
                result[col] = 0.0 if col in self.EVENT_COLUMNS else np.nan

        result = result[self.REQUIRED_CACHE_COLUMNS]
        result = result.apply(pd.to_numeric, errors='coerce')
        result[self.EVENT_COLUMNS] = result[self.EVENT_COLUMNS].fillna(0).astype(int)
        result = result.sort_index().ffill()
        return result

    def _download_close_price(self, ticker, column_name):
        """
        Download close price series and normalize output schema.
        Handles both normal and MultiIndex yfinance responses.
        """
        if yf is None:
            return None

        raw = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False
        )
        if raw is None or raw.empty:
            raise ValueError(f"No data returned for {ticker}")

        if isinstance(raw.columns, pd.MultiIndex):
            close_cols = [c for c in raw.columns if str(c[0]).lower() == 'close']
            if not close_cols:
                series = raw.iloc[:, 0]
            else:
                selected = raw[close_cols[0]]
                if isinstance(selected, pd.DataFrame):
                    series = selected.iloc[:, 0]
                else:
                    series = selected
        else:
            close_col = 'Close' if 'Close' in raw.columns else raw.columns[0]
            series = raw[close_col]

        result = series.to_frame(name=column_name)
        result.index = pd.to_datetime(result.index)
        result = result.loc[~result.index.duplicated(keep='first')].sort_index()
        return result
        
    def fetch_crude_oil_prices(self):
        """
        Fetch crude oil prices (WTI and Brent)
        
        WTI: CL=F (NYMEX WTI Crude Oil Futures)
        Brent: BZ=F (Brent Crude Oil Futures)
        """
        print("Fetching crude oil prices...")

        if yf is None:
            print("  [WARN] yfinance unavailable, crude oil prices skipped")
            return None
        
        try:
            # WTI crude oil
            wti = self._download_close_price('CL=F', 'WTI_Price')
            
            # Brent crude oil
            brent = self._download_close_price('BZ=F', 'Brent_Price')

            candidates = [df for df in [wti, brent] if df is not None and not df.empty]
            if not candidates:
                raise ValueError("No crude oil data returned")

            # Merge
            oil_prices = pd.concat(candidates, axis=1)
            self.data['crude_oil'] = oil_prices
            print(f"  [OK] Crude oil prices: {len(oil_prices)} records")
            return oil_prices
            
        except Exception as e:
            print(f"  [ERROR] Failed to fetch crude oil prices: {e}")
            return None
    
    def fetch_natural_gas_prices(self):
        """
        Fetch natural gas prices
        
        NG=F: Henry Hub Natural Gas Futures
        """
        print("Fetching natural gas prices...")

        if yf is None:
            print("  [WARN] yfinance unavailable, natural gas prices skipped")
            return None
        
        try:
            ng = self._download_close_price('NG=F', 'NatGas_Price')
            if ng is None or ng.empty:
                raise ValueError("No natural gas data returned")
            self.data['natural_gas'] = ng
            print(f"  [OK] Natural gas prices: {len(ng)} records")
            return ng
            
        except Exception as e:
            print(f"  [ERROR] Failed to fetch natural gas prices: {e}")
            return None
    
    def fetch_exchange_rates(self):
        """
        Fetch exchange rate data
        
        USDTWD=X: USD/TWD
        DX-Y.NYB: US Dollar Index (DXY)
        """
        print("Fetching exchange rate data...")

        if yf is None:
            print("  [WARN] yfinance unavailable, exchange rate data skipped")
            return None
        
        try:
            # USD/TWD
            usdtwd = self._download_close_price('USDTWD=X', 'USD_TWD')
            
            # DXY US Dollar Index
            dxy = self._download_close_price('DX-Y.NYB', 'DXY')
            
            candidates = [df for df in [usdtwd, dxy] if df is not None and not df.empty]
            if not candidates:
                raise ValueError("No exchange rate data returned")

            # Merge
            fx_rates = pd.concat(candidates, axis=1)
            self.data['exchange_rates'] = fx_rates
            print(f"  [OK] Exchange rate data: {len(fx_rates)} records")
            return fx_rates
            
        except Exception as e:
            print(f"  [ERROR] Failed to fetch exchange rate data: {e}")
            return None
    
    def fetch_stock_indices(self):
        """
        Fetch related stock indices
        
        ^TWII: Taiwan Weighted Index
        TSM: TSMC ADR (Semiconductor proxy indicator)
        """
        print("Fetching stock indices...")

        if yf is None:
            print("  [WARN] yfinance unavailable, stock indices skipped")
            return None
        
        try:
            # Taiwan Weighted Index
            twii = self._download_close_price('^TWII', 'TWII')
            
            # TSMC ADR (as semiconductor industry proxy)
            tsm = self._download_close_price('TSM', 'TSM_Price')
            
            candidates = [df for df in [twii, tsm] if df is not None and not df.empty]
            if not candidates:
                raise ValueError("No stock index data returned")

            indices = pd.concat(candidates, axis=1)
            self.data['stock_indices'] = indices
            print(f"  [OK] Stock indices: {len(indices)} records")
            return indices
            
        except Exception as e:
            print(f"  [ERROR] Failed to fetch stock indices: {e}")
            return None
    
    def create_event_indicators(self):
        """
        Create geopolitical and major event indicators
        
        Event encoding:
        - COVID-19 pandemic (2020/01 - 2023/05)
        - Russia-Ukraine War (2022/02 - ongoing)
        - Red Sea Crisis (2023/11 - ongoing)
        - US-China Trade War (2018/03 - ongoing)
        """
        print("Creating event indicators...")
        
        # Create date range
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        events = pd.DataFrame(index=date_range)
        
        # COVID-19 pandemic
        events['COVID19'] = 0
        events.loc['2020-01-01':'2023-05-31', 'COVID19'] = 1
        
        # Russia-Ukraine War
        events['Ukraine_War'] = 0
        events.loc['2022-02-24':, 'Ukraine_War'] = 1
        
        # Red Sea Crisis
        events['RedSea_Crisis'] = 0
        events.loc['2023-11-01':, 'RedSea_Crisis'] = 1
        
        # US-China Trade War
        events['US_China_Trade'] = 0
        events.loc['2018-03-01':, 'US_China_Trade'] = 1
        
        # Global financial risk indicator (simplified)
        events['Geopolitical_Risk'] = events['COVID19'] + events['Ukraine_War'] + events['RedSea_Crisis']
        
        self.data['events'] = events
        print(f"  [OK] Event indicators: {len(events)} records")
        return events
    
    def collect_all_data(self):
        """
        Collect all data and merge into a single DataFrame
        """
        if self.use_cache and not self.refresh_cache and os.path.exists(self.cache_file):
            try:
                cached = pd.read_csv(self.cache_file, parse_dates=['Date'], index_col='Date')
                cached = cached.sort_index()
                valid, reason = self._validate_cached_data(cached)
                if not valid:
                    raise ValueError(f"invalid cache ({reason})")
                print("\n" + "=" * 50)
                print("Loading cached market data...")
                print("=" * 50 + "\n")
                print(f"[OK] Loaded cache: {self.cache_file}")
                print(f"  Total records: {len(cached)}")
                print(f"  Number of columns: {len(cached.columns)}")
                print(f"  Time range: {cached.index.min()} ~ {cached.index.max()}")
                return self._normalize_output_schema(cached)
            except Exception as e:
                print(f"[WARN] Failed to load cache, fallback to remote collection: {e}")

        print("\n" + "="*50)
        print("Starting to collect all data...")
        print("="*50 + "\n")
        
        # Collect various data types
        self.fetch_crude_oil_prices()
        self.fetch_natural_gas_prices()
        self.fetch_exchange_rates()
        self.fetch_stock_indices()
        self.create_event_indicators()
        
        # Merge all data
        print("\nMerging data...")
        
        dfs_to_merge = []
        for name, df in self.data.items():
            if df is not None and not df.empty:
                # Ensure index is DatetimeIndex
                if not isinstance(df.index, pd.DatetimeIndex):
                    continue
                # Remove duplicate indices
                df = df.loc[~df.index.duplicated(keep='first')]
                dfs_to_merge.append(df)
        
        if not dfs_to_merge:
            print("  [ERROR] No available data")
            return pd.DataFrame()
        
        # Use concat to merge
        all_data = pd.concat(dfs_to_merge, axis=1)
        
        # Remove duplicate columns
        all_data = all_data.loc[:, ~all_data.columns.duplicated()]
        
        # Keep causal filling only to avoid future information leakage
        all_data = self._normalize_output_schema(all_data)
        
        print("\n[OK] Data collection complete!")
        print(f"  Total records: {len(all_data)}")
        print(f"  Number of columns: {len(all_data.columns)}")
        print(f"  Time range: {all_data.index.min()} ~ {all_data.index.max()}")

        if self.use_cache:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                all_data.to_csv(self.cache_file, index_label='Date')
                print(f"[OK] Cache saved: {self.cache_file}")
            except Exception as e:
                print(f"[WARN] Failed to write cache: {e}")
        
        return all_data
    
    def resample_to_quarterly(self, df):
        """
        Convert daily frequency data to quarterly data
        
        Parameters:
        -----------
        df : DataFrame
            Daily frequency data
            
        Returns:
        --------
        DataFrame
            Quarterly data (Q1, Q2, Q3, Q4)
        """
        print("\nConverting to quarterly data...")
        
        # Quarterly average
        quarterly = df.resample('QE').mean()
        
        # Add quarter labels
        quarterly['Year'] = quarterly.index.year
        quarterly['Quarter'] = quarterly.index.quarter
        quarterly['Quarter_Label'] = quarterly.apply(
            lambda x: f"{int(x['Year'])}Q{int(x['Quarter'])}", axis=1
        )
        
        print(f"[OK] Quarterly data: {len(quarterly)} records")
        return quarterly
    
    def save_data(self, df, filename):
        """
        Save data to CSV
        """
        df.to_csv(filename)
        print(f"[OK] Data saved to: {filename}")


def load_ipa_prices_from_csv(filepath):
    """
    Load IPA price data from CSV
    
    Expected format:
    - Date: Date column
    - Price: Price column (TWD/KG)
    """
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    return df


# Main test program
if __name__ == "__main__":
    # Initialize collector
    collector = IPADataCollector(
        start_date='2012-01-01',
        end_date='2024-12-31'
    )
    
    # Collect all data
    data = collector.collect_all_data()
    
    # Display data summary
    print("\n" + "="*50)
    print("Data Summary")
    print("="*50)
    print(data.describe())
    
    # Convert to quarterly data
    quarterly_data = collector.resample_to_quarterly(data)
    print("\nQuarterly Data:")
    print(quarterly_data.tail(10))

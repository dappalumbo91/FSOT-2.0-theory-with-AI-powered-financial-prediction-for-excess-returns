"""
FSOT 2.5 Comprehensive Data Framework
====================================

Ultra-comprehensive data collection system with maximum coverage
across all asset classes, frequencies, and data sources.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json
import os
import pickle
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Optional, Tuple
import threading
import time
import schedule
from concurrent.futures import ThreadPoolExecutor
import logging
import ta  # Technical analysis library

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    filename='fsot_comprehensive_data.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ComprehensiveDataFramework:
    """
    Ultra-comprehensive data collection and processing system
    """

    def __init__(self, model_dir='comprehensive_models', data_dir='comprehensive_data'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.performance_history = {}
        self.data_sources = {}
        self.is_learning = False

        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/historical", exist_ok=True)
        os.makedirs(f"{data_dir}/streaming", exist_ok=True)
        os.makedirs(f"{data_dir}/economic", exist_ok=True)
        os.makedirs(f"{data_dir}/news", exist_ok=True)
        os.makedirs(f"{data_dir}/options", exist_ok=True)

        # Initialize ultra-comprehensive data sources
        self._initialize_comprehensive_data_sources()

        # Load existing models if available
        self._load_existing_models()

        # Start comprehensive learning thread
        self.executor = ThreadPoolExecutor(max_workers=16)  # Increased workers
        self.learning_thread = threading.Thread(target=self._comprehensive_learning_loop, daemon=True)
        self.learning_thread.start()

    def _initialize_comprehensive_data_sources(self):
        """Initialize maximum coverage data sources"""
        self.data_sources = {
            'stocks_major': {
                'provider': 'yfinance',
                'symbols': [
                    # Tech Giants
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX',
                    'CRM', 'ADBE', 'ORCL', 'INTC', 'AMD', 'IBM', 'CSCO', 'QCOM',
                    # Financial
                    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SPGI',
                    # Healthcare
                    'JNJ', 'PFE', 'UNH', 'MRK', 'ABT', 'TMO', 'DHR', 'BMY', 'LLY',
                    # Consumer
                    'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS', 'NKE', 'SBUX', 'COST',
                    # Energy
                    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY',
                    # Industrial
                    'BA', 'CAT', 'GE', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX',
                    # Materials
                    'LIN', 'APD', 'ECL', 'SHW', 'FCX', 'NEM', 'GOLD', 'AA',
                    # Communication
                    'VZ', 'T', 'TMUS', 'CMCSA', 'CHTR', 'EA', 'ATVI', 'TTWO',
                    # Utilities
                    'NEE', 'DUK', 'SO', 'D', 'EXC', 'AEP', 'SRE', 'PEG',
                    # Real Estate
                    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'O', 'WELL', 'AVB'
                ],
                'frequency': '1d',
                'lookback_days': 365*5,  # 5 years for comprehensive training
                'intraday': True  # Enable intraday data
            },
            'stocks_emerging': {
                'provider': 'yfinance',
                'symbols': [
                    'SQ', 'SHOP', 'UBER', 'LYFT', 'ZM', 'DOCU', 'CRWD', 'DDOG',
                    'NET', 'FSLY', 'AFRM', 'COIN', 'HOOD', 'RIVN', 'LCID', 'SOFI',
                    'PLTR', 'SNOW', 'MDB', 'HUBS', 'OKTA', 'ZS', 'PANW', 'FTNT'
                ],
                'frequency': '1d',
                'lookback_days': 365*3,
                'intraday': True
            },
            'crypto_major': {
                'provider': 'coingecko',
                'symbols': [
                    'BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'DOGE', 'AVAX',
                    'MATIC', 'LINK', 'ALGO', 'VET', 'ICP', 'FIL', 'TRX', 'ETC',
                    'XLM', 'THETA', 'FTM', 'HBAR', 'EGLD', 'NEAR', 'FLOW', 'MANA',
                    'SAND', 'AXS', 'CHZ', 'ENJ', 'BAT', 'COMP', 'MKR', 'SUSHI'
                ],
                'frequency': 'hourly',
                'lookback_days': 365*2,
                'intraday': True
            },
            'crypto_defi': {
                'provider': 'coingecko',
                'symbols': [
                    'UNI', 'AAVE', 'CRV', 'YFI', 'BAL', 'REN', 'KNC', 'ZRX',
                    'LRC', 'REP', 'GNT', 'STORJ', 'ANT', 'BAT', 'OMG', 'SNT'
                ],
                'frequency': 'hourly',
                'lookback_days': 365*2,
                'intraday': True
            },
            'forex_major': {
                'provider': 'yfinance',
                'symbols': [
                    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X',
                    'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X',
                    'AUDJPY=X', 'CADJPY=X', 'CHFJPY=X', 'NZDJPY=X', 'EURAUD=X',
                    'GBPAUD=X', 'AUDCAD=X', 'AUDCHF=X', 'CADCHF=X', 'NZDCHF=X'
                ],
                'frequency': '1h',  # Hourly forex data
                'lookback_days': 365*3,
                'intraday': True
            },
            'forex_emerging': {
                'provider': 'yfinance',
                'symbols': [
                    'USDMXN=X', 'USDZAR=X', 'USDBRL=X', 'USDTRY=X', 'USDRUB=X',
                    'USDINR=X', 'USDKRW=X', 'USDTHB=X', 'USDSGD=X', 'USDHKD=X'
                ],
                'frequency': '1h',
                'lookback_days': 365*2,
                'intraday': True
            },
            'commodities_energy': {
                'provider': 'yfinance',
                'symbols': [
                    'CL=F', 'BZ=F', 'NG=F', 'HO=F', 'RB=F',  # Oil, Brent, Natural Gas, Heating Oil, Gasoline
                    'B0=F', 'C0=F', 'O0=F', 'S0=F'  # Brent, Corn, Oats, Soybeans
                ],
                'frequency': '1h',
                'lookback_days': 365*3,
                'intraday': True
            },
            'commodities_metals': {
                'provider': 'yfinance',
                'symbols': [
                    'GC=F', 'SI=F', 'HG=F', 'PL=F', 'PA=F',  # Gold, Silver, Copper, Platinum, Palladium
                    'ALI=F', 'ZNC=F', 'ZSC=F', 'ZCC=F'  # Aluminum, Zinc, Steel, Cocoa
                ],
                'frequency': '1h',
                'lookback_days': 365*3,
                'intraday': True
            },
            'commodities_agriculture': {
                'provider': 'yfinance',
                'symbols': [
                    'ZC=F', 'ZW=F', 'ZS=F', 'ZM=F', 'ZL=F', 'ZF=F',  # Corn, Wheat, Soybeans, Soybean Meal, Soybean Oil, Feeder Cattle
                    'HE=F', 'LE=F', 'GF=F', 'CC=F', 'KC=F', 'CT=F'  # Lean Hogs, Live Cattle, Feeder Cattle, Cocoa, Coffee, Cotton
                ],
                'frequency': '1d',
                'lookback_days': 365*3,
                'intraday': True
            },
            'indices_global': {
                'provider': 'yfinance',
                'symbols': [
                    '^GSPC', '^IXIC', '^DJI', '^VIX', '^TNX', '^RUT', '^NYA',  # US
                    '^FTSE', '^GDAXI', '^FCHI', '^STOXX50E', '^N225', '^HSI',  # Europe/Asia
                    '^BSESN', '^NSEI', '^AXJO', '^GSPTSE', '^BVSP', '^MXX',  # Other global
                    '^VIX3M', '^VIX6M', '^VIX9M', '^VIX1Y'  # VIX variants
                ],
                'frequency': '1h',
                'lookback_days': 365*5,
                'intraday': True
            },
            'bonds': {
                'provider': 'yfinance',
                'symbols': [
                    '^TNX', '^IRX', '^FVX', '^TYX', '^TLT', '^IEF', '^SHY',  # US Treasuries
                    'BND', 'AGG', 'LQD', 'HYG', 'JNK', 'EMB'  # Bond ETFs
                ],
                'frequency': '1h',
                'lookback_days': 365*3,
                'intraday': True
            },
            'economic_indicators': {
                'provider': 'fred',
                'symbols': [
                    'GDP', 'UNRATE', 'FEDFUNDS', 'CPIAUCSL', 'CPILFESL', 'PCE',
                    'INDPRO', 'HOUST', 'DGS10', 'DGS2', 'T10Y2Y', 'BAMLH0A0HYM2',
                    'DEXUSEU', 'DEXJPUS', 'DEXCHUS', 'DEXCAUS', 'ICSA', 'DCOILWTICO'
                ],
                'frequency': 'monthly',
                'lookback_days': 365*10,  # 10 years for economic data
                'intraday': False
            }
        }

    def fetch_comprehensive_data(self, symbol: str, data_type: str = 'stocks_major') -> pd.DataFrame:
        """
        Fetch ultra-comprehensive data with maximum coverage
        """
        try:
            config = self.data_sources.get(data_type, {})

            if data_type in ['stocks_major', 'stocks_emerging', 'forex_major', 'forex_emerging',
                           'commodities_energy', 'commodities_metals', 'commodities_agriculture',
                           'indices_global', 'bonds']:
                return self._fetch_yfinance_comprehensive(symbol, config)

            elif data_type.startswith('crypto'):
                return self._fetch_crypto_comprehensive(symbol, config)

            elif data_type == 'economic_indicators':
                return self._fetch_economic_data(symbol, config)

        except Exception as e:
            logging.error(f"Error fetching comprehensive data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_yfinance_comprehensive(self, symbol: str, config: Dict) -> pd.DataFrame:
        """Fetch comprehensive yfinance data with intraday support"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=config['lookback_days'])

            # Determine interval based on frequency and lookback
            if config.get('intraday', False):
                if config['frequency'] == '1h' and config['lookback_days'] <= 730:  # 2 years
                    interval = '1h'
                elif config['frequency'] == '1d':
                    interval = '1d'
                else:
                    interval = '1d'
            else:
                interval = '1d'

            # Fetch data
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                prepost=True  # Include pre/post market data
            )

            if data.empty:
                return pd.DataFrame()

            # Flatten MultiIndex columns if present
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # Add comprehensive technical indicators
            data = self._add_ultra_comprehensive_indicators(data)

            # Add market regime indicators
            data = self._add_market_regime_indicators(data)

            return data

        except Exception as e:
            logging.error(f"Error fetching yfinance comprehensive data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_crypto_comprehensive(self, symbol: str, config: Dict) -> pd.DataFrame:
        """Fetch comprehensive cryptocurrency data"""
        try:
            coin_id = self._get_coin_id(symbol)
            days = config['lookback_days']

            # Determine granularity
            if config.get('intraday', False) and config['frequency'] == 'hourly':
                granularity = 'hourly'
            else:
                granularity = 'daily'

            if granularity == 'hourly':
                # CoinGecko hourly data (last 90 days max)
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': min(days, 90),  # Hourly data limited to 90 days
                    'interval': 'hourly'
                }
            else:
                url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                params = {
                    'vs_currency': 'usd',
                    'days': days,
                    'interval': 'daily'
                }

            response = requests.get(url, params=params)
            response.raise_status_code()

            data = response.json()

            # Convert to DataFrame
            if granularity == 'hourly':
                prices = data.get('prices', [])
                if not prices:
                    return pd.DataFrame()
                df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            else:
                prices = data.get('prices', [])
                if not prices:
                    return pd.DataFrame()
                df = pd.DataFrame(prices, columns=['timestamp', 'close'])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Add additional price data
            if 'total_volumes' in data:
                df['volume'] = [x[1] for x in data['total_volumes']]

            if 'market_caps' in data:
                df['market_cap'] = [x[1] for x in data['market_caps']]

            # Create OHLC from close (approximation for crypto)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df[['open', 'close']].max(axis=1) * 1.05  # Higher volatility
            df['low'] = df[['open', 'close']].min(axis=1) * 0.95   # Higher volatility

            # Add comprehensive technical indicators
            df = self._add_ultra_comprehensive_indicators(df)

            # Add crypto-specific indicators
            df = self._add_crypto_specific_indicators(df)

            return df

        except Exception as e:
            logging.error(f"Error fetching comprehensive crypto data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_economic_data(self, symbol: str, config: Dict) -> pd.DataFrame:
        """Fetch economic indicator data from FRED"""
        try:
            # This would require FRED API key - for now return empty
            # In production, integrate with FRED API
            logging.info(f"Economic data fetch for {symbol} - requires FRED API integration")
            return pd.DataFrame()
        except Exception as e:
            logging.error(f"Error fetching economic data for {symbol}: {e}")
            return pd.DataFrame()

    def _get_coin_id(self, symbol: str) -> str:
        """Map symbol to CoinGecko coin ID"""
        coin_mapping = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'BNB': 'binancecoin',
            'ADA': 'cardano', 'SOL': 'solana', 'DOT': 'polkadot',
            'DOGE': 'dogecoin', 'AVAX': 'avalanche-2', 'MATIC': 'matic-network',
            'LINK': 'chainlink', 'ALGO': 'algorand', 'VET': 'vechain',
            'ICP': 'internet-computer', 'FIL': 'filecoin', 'TRX': 'tron',
            'ETC': 'ethereum-classic', 'XLM': 'stellar', 'THETA': 'theta-token',
            'FTM': 'fantom', 'HBAR': 'hedera-hashgraph', 'EGLD': 'elrond-erd-2',
            'NEAR': 'near', 'FLOW': 'flow', 'MANA': 'decentraland',
            'SAND': 'the-sandbox', 'AXS': 'axie-infinity', 'CHZ': 'chiliz',
            'ENJ': 'enjincoin', 'BAT': 'basic-attention-token', 'COMP': 'compound-governance-token',
            'MKR': 'maker', 'SUSHI': 'sushi', 'UNI': 'uniswap', 'AAVE': 'aave',
            'CRV': 'curve-dao-token', 'YFI': 'yearn-finance', 'BAL': 'balancer',
            'REN': 'ren', 'KNC': 'kyber-network-crystal', 'ZRX': '0x',
            'LRC': 'loopring', 'REP': 'augur', 'GNT': 'golem', 'STORJ': 'storj',
            'ANT': 'aragon', 'OMG': 'omisego', 'SNT': 'status'
        }
        return coin_mapping.get(symbol, symbol.lower())

    def _add_ultra_comprehensive_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add maximum technical indicators"""
        try:
            # Ensure we have required columns
            if 'Close' not in df.columns:
                df['Close'] = df['close'] if 'close' in df.columns else df.iloc[:, 0]

            if 'High' not in df.columns:
                df['High'] = df['high'] if 'high' in df.columns else df['Close'] * 1.02

            if 'Low' not in df.columns:
                df['Low'] = df['low'] if 'low' in df.columns else df['Close'] * 0.98

            if 'Volume' not in df.columns:
                df['Volume'] = df['volume'] if 'volume' in df.columns else 1

            # Basic price indicators
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['realized_volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

            # Multiple timeframe moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
                df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)

            # MACD with multiple signals
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()
            df['MACD_crossover'] = np.where(df['MACD'] > df['MACD_signal'], 1, -1)

            # RSI with multiple periods
            for period in [6, 14, 21]:
                df[f'RSI_{period}'] = ta.momentum.rsi(df['Close'], window=period)

            # Stochastic Oscillators
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
            df['stoch_k_slow'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])

            # Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], window=14)

            # Bollinger Bands with multiple deviations
            for std in [1, 2, 3]:
                bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=std)
                df[f'BB_upper_{std}'] = bb.bollinger_hband()
                df[f'BB_middle_{std}'] = bb.bollinger_mband()
                df[f'BB_lower_{std}'] = bb.bollinger_lband()
                df[f'BB_width_{std}'] = bb.bollinger_wband()
                df[f'BB_percent_{std}'] = bb.bollinger_pband()

            # Keltner Channels
            keltner = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'])
            df['KC_upper'] = keltner.keltner_channel_hband()
            df['KC_middle'] = keltner.keltner_channel_mband()
            df['KC_lower'] = keltner.keltner_channel_lband()

            # Donchian Channels
            df['donchian_upper'] = df['High'].rolling(window=20).max()
            df['donchian_lower'] = df['Low'].rolling(window=20).min()
            df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2

            # Ichimoku Cloud
            ichimoku = ta.trend.IchimokuIndicator(df['High'], df['Low'])
            df['tenkan_sen'] = ichimoku.ichimoku_conversion_line()
            df['kijun_sen'] = ichimoku.ichimoku_base_line()
            df['senkou_span_a'] = ichimoku.ichimoku_a()
            df['senkou_span_b'] = ichimoku.ichimoku_b()
            df['chikou_span'] = ichimoku.ichimoku_chikou()

            # Commodity Channel Index
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)

            # Average True Range
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)

            # Volume indicators
            if df['Volume'].sum() > 0:
                df['volume_sma'] = ta.trend.sma_indicator(df['Volume'], window=20)
                df['volume_ratio'] = df['Volume'] / df['volume_sma']
                df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
                df['volume_price_trend'] = ta.volume.volume_price_trend(df['Close'], df['Volume'])
                df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])

            # Momentum indicators
            for period in [10, 20, 30]:
                df[f'ROC_{period}'] = ta.momentum.roc(df['Close'], window=period)
                df[f'MOM_{period}'] = ta.momentum.roc(df['Close'], window=period)

            # Rate of Change Ratio
            df['ROC_ratio'] = df['ROC_10'] / df['ROC_20']

            # Support/Resistance levels
            df['support_20'] = df['Low'].rolling(window=20).min()
            df['resistance_20'] = df['High'].rolling(window=20).max()
            df['support_50'] = df['Low'].rolling(window=50).min()
            df['resistance_50'] = df['High'].rolling(window=50).max()

            # Pivot Points
            df['pivot_point'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['pivot_r1'] = 2 * df['pivot_point'] - df['Low']
            df['pivot_s1'] = 2 * df['pivot_point'] - df['High']

            # Fibonacci retracements (simplified)
            high_20 = df['High'].rolling(window=20).max()
            low_20 = df['Low'].rolling(window=20).min()
            df['fib_236'] = high_20 - (high_20 - low_20) * 0.236
            df['fib_382'] = high_20 - (high_20 - low_20) * 0.382
            df['fib_618'] = high_20 - (high_20 - low_20) * 0.618

            # Remove NaN values
            df = df.dropna()

            # Replace infinity values with NaN and then drop them
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.dropna()

            return df

        except Exception as e:
            logging.error(f"Error adding ultra comprehensive indicators: {e}")
            return df

    def _add_market_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime detection indicators"""
        try:
            # Trend strength
            df['trend_strength'] = abs(df['EMA_20'] - df['EMA_50']) / df['EMA_50']

            # Volatility regime
            df['volatility_regime'] = np.where(df['ATR'] > df['ATR'].rolling(window=50).mean(), 1, -1)

            # Volume regime
            if 'volume_ratio' in df.columns:
                df['volume_regime'] = np.where(df['volume_ratio'] > 1.2, 1, np.where(df['volume_ratio'] < 0.8, -1, 0))

            # Momentum regime
            df['momentum_regime'] = np.where(df['ROC_20'] > 5, 1, np.where(df['ROC_20'] < -5, -1, 0))

            # Mean reversion signals
            df['mean_reversion'] = (df['Close'] - df['SMA_20']) / df['SMA_20']

            return df

        except Exception as e:
            logging.error(f"Error adding market regime indicators: {e}")
            return df

    def _add_crypto_specific_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cryptocurrency-specific indicators"""
        try:
            # Market cap indicators
            if 'market_cap' in df.columns:
                df['market_cap_sma'] = ta.trend.sma_indicator(df['market_cap'], window=20)
                df['market_cap_ratio'] = df['market_cap'] / df['market_cap_sma']

            # Higher volatility adjustments
            df['crypto_volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(365)  # Annualized

            # Crypto-specific momentum
            df['crypto_momentum'] = df['Close'] / df['Close'].shift(30) - 1

            return df

        except Exception as e:
            logging.error(f"Error adding crypto-specific indicators: {e}")
            return df

    def train_comprehensive_model(self, symbol: str, data_type: str = 'stocks_major') -> Dict:
        """
        Train comprehensive model with maximum data coverage
        """
        try:
            print(f"🔬 Training comprehensive model for {symbol}...")

            # Fetch comprehensive data
            df = self.fetch_comprehensive_data(symbol, data_type)

            if df.empty or len(df) < 300:  # Need more data for comprehensive training
                print(f"❌ Insufficient comprehensive data for {symbol}")
                return {'status': 'failed', 'reason': 'insufficient_data'}

            # Prepare ultra-comprehensive features and targets
            features = self._prepare_ultra_comprehensive_features(df)

            # Create multiple prediction horizons
            targets = self._create_multi_horizon_targets(df)

            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1))
            for target_name in targets.keys():
                valid_idx &= ~targets[target_name].isna()

            features = features[valid_idx]
            for target_name in targets.keys():
                targets[target_name] = targets[target_name][valid_idx]

            if len(features) < 200:
                return {'status': 'failed', 'reason': 'insufficient_valid_data'}

            # Advanced feature selection
            if symbol not in self.feature_selectors:
                self.feature_selectors[symbol] = SelectKBest(score_func=f_regression, k=min(50, features.shape[1]))

            features_selected = self.feature_selectors[symbol].fit_transform(features, targets['target_1d'])
            selected_feature_names = features.columns[self.feature_selectors[symbol].get_support()].tolist()

            # Enhanced time series split
            tscv = TimeSeriesSplit(n_splits=5)  # More splits for better validation
            splits = list(tscv.split(features_selected))

            # Use the last split for final training
            train_idx, test_idx = splits[-1]
            X_train = features_selected[train_idx]
            X_test = features_selected[test_idx]
            y_train = targets['target_1d'].iloc[train_idx]
            y_test = targets['target_1d'].iloc[test_idx]

            # Advanced scaling
            if symbol not in self.scalers:
                self.scalers[symbol] = RobustScaler()

            X_train_scaled = self.scalers[symbol].fit_transform(X_train)
            X_test_scaled = self.scalers[symbol].transform(X_test)

            # Train ultra-comprehensive ensemble model
            if symbol not in self.models:
                self.models[symbol] = self._create_ultra_comprehensive_ensemble_model()
                print(f"🆕 Created comprehensive model for {symbol}")
            else:
                print(f"🔄 Updating comprehensive model for {symbol}")

            # Train the model
            self.models[symbol].fit(X_train_scaled, y_train)

            # Comprehensive evaluation
            y_pred = self.models[symbol].predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Directional accuracy
            actual_direction = np.sign(y_test)
            predicted_direction = np.sign(y_pred)
            directional_accuracy = np.mean(actual_direction == predicted_direction)

            # Additional metrics
            mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.inf))) * 100
            rmse = np.sqrt(mse)

            # Store comprehensive performance history
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []

            self.performance_history[symbol].append({
                'timestamp': datetime.now(),
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'mape': mape,
                'n_samples': len(y_test),
                'selected_features': selected_feature_names,
                'data_points': len(df),
                'feature_count': len(selected_feature_names)
            })

            # Save comprehensive model
            self._save_comprehensive_model(symbol)

            print(f"✅ MSE: {mse:.6f}")
            print(f"✅ R²: {r2:.4f}")
            print(f"✅ Directional Accuracy: {directional_accuracy:.1%}")
            print(f"✅ MAPE: {mape:.2f}%")
            print(f"✅ Features Used: {len(selected_feature_names)}")

            return {
                'status': 'success',
                'symbol': symbol,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'mape': mape,
                'n_samples': len(y_test),
                'selected_features': selected_feature_names,
                'data_points': len(df),
                'feature_count': len(selected_feature_names)
            }

        except Exception as e:
            logging.error(f"Error training comprehensive model for {symbol}: {e}")
            return {'status': 'failed', 'reason': str(e)}

    def _prepare_ultra_comprehensive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare maximum feature set for comprehensive training"""
        feature_cols = []

        # Price and volume features
        price_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'returns', 'log_returns']
        feature_cols.extend([col for col in price_features if col in df.columns])

        # Trend features
        trend_features = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200',
            'MACD', 'MACD_signal', 'MACD_hist', 'MACD_crossover'
        ]
        feature_cols.extend([col for col in trend_features if col in df.columns])

        # Momentum features
        momentum_features = [
            'RSI_6', 'RSI_14', 'RSI_21', 'stoch_k', 'stoch_d', 'stoch_k_slow',
            'williams_r', 'ROC_10', 'ROC_20', 'ROC_30', 'MOM_10', 'MOM_20', 'MOM_30',
            'ROC_ratio'
        ]
        feature_cols.extend([col for col in momentum_features if col in df.columns])

        # Volatility features
        volatility_features = [
            'BB_upper_1', 'BB_middle_1', 'BB_lower_1', 'BB_width_1', 'BB_percent_1',
            'BB_upper_2', 'BB_middle_2', 'BB_lower_2', 'BB_width_2', 'BB_percent_2',
            'BB_upper_3', 'BB_middle_3', 'BB_lower_3', 'BB_width_3', 'BB_percent_3',
            'KC_upper', 'KC_middle', 'KC_lower', 'donchian_upper', 'donchian_lower', 'donchian_middle',
            'ATR', 'realized_volatility'
        ]
        feature_cols.extend([col for col in volatility_features if col in df.columns])

        # Ichimoku features
        ichimoku_features = [
            'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span'
        ]
        feature_cols.extend([col for col in ichimoku_features if col in df.columns])

        # Volume features
        volume_features = [
            'volume_sma', 'volume_ratio', 'OBV', 'volume_price_trend', 'VWAP'
        ]
        feature_cols.extend([col for col in volume_features if col in df.columns])

        # Support/Resistance and Pivot features
        sr_features = [
            'support_20', 'resistance_20', 'support_50', 'resistance_50',
            'pivot_point', 'pivot_r1', 'pivot_s1', 'fib_236', 'fib_382', 'fib_618'
        ]
        feature_cols.extend([col for col in sr_features if col in df.columns])

        # Market regime features
        regime_features = [
            'trend_strength', 'volatility_regime', 'volume_regime', 'momentum_regime', 'mean_reversion'
        ]
        feature_cols.extend([col for col in regime_features if col in df.columns])

        # Crypto-specific features
        crypto_features = [
            'market_cap', 'market_cap_sma', 'market_cap_ratio', 'crypto_volatility', 'crypto_momentum'
        ]
        feature_cols.extend([col for col in crypto_features if col in df.columns])

        # Lagged features (extended)
        base_cols = ['Close', 'returns', 'RSI_14', 'MACD', 'stoch_k', 'ATR']
        for col in base_cols:
            if col in df.columns:
                for lag in [1, 2, 3, 5, 10, 15, 20]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    feature_cols.append(f'{col}_lag_{lag}')

        # Rolling statistics (extended)
        for col in ['returns', 'Close', 'Volume']:
            if col in df.columns:
                for window in [5, 10, 20, 30, 50]:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_rolling_skew_{window}'] = df[col].rolling(window=window).skew()
                    df[f'{col}_rolling_kurt_{window}'] = df[col].rolling(window=window).kurt()
                    feature_cols.extend([
                        f'{col}_rolling_mean_{window}',
                        f'{col}_rolling_std_{window}',
                        f'{col}_rolling_skew_{window}',
                        f'{col}_rolling_kurt_{window}'
                    ])

        # Rate of change features
        for col in ['Close', 'Volume']:
            if col in df.columns:
                for period in [1, 3, 5, 10, 20]:
                    df[f'{col}_roc_{period}'] = df[col].pct_change(period)
                    feature_cols.append(f'{col}_roc_{period}')

        if not feature_cols:
            feature_cols = ['Close']

        features_df = df[feature_cols].copy()

        # Replace infinity values with NaN and drop them
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()

        return features_df

    def _create_multi_horizon_targets(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Create multiple prediction horizons"""
        targets = {}

        # 1-day ahead return
        targets['target_1d'] = df['returns'].shift(-1)

        # 3-day ahead return
        targets['target_3d'] = df['Close'].pct_change(3).shift(-3)

        # 5-day ahead return
        targets['target_5d'] = df['Close'].pct_change(5).shift(-5)

        # 10-day ahead return
        targets['target_10d'] = df['Close'].pct_change(10).shift(-10)

        # Directional targets (for classification)
        for horizon in ['1d', '3d', '5d', '10d']:
            targets[f'target_{horizon}_direction'] = np.sign(targets[f'target_{horizon}'])

        return targets

    def _create_ultra_comprehensive_ensemble_model(self):
        """Create maximum coverage ensemble model"""
        from sklearn.ensemble import VotingRegressor, StackingRegressor, BaggingRegressor

        # Base models with different algorithms
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        et = ExtraTreesRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

        gb = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )

        # Bagging regressor for additional diversity
        bag = BaggingRegressor(
            estimator=RandomForestRegressor(n_estimators=100, random_state=42),
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        # Meta model
        ridge = Ridge(alpha=0.01, random_state=42)

        # Stacking ensemble with more base models
        estimators = [
            ('rf', rf),
            ('et', et),
            ('gb', gb),
            ('bag', bag)
        ]

        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=ridge,
            cv=5,  # More cross-validation folds
            n_jobs=-1
        )

        return ensemble

    def _comprehensive_learning_loop(self):
        """Ultra-comprehensive continuous learning loop"""
        print("🚀 Starting ultra-comprehensive continuous learning loop...")

        # Schedule comprehensive updates
        schedule.every().day.at("00:00").do(self._ultra_comprehensive_model_update)
        schedule.every(4).hours.do(self._incremental_comprehensive_update)

        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logging.error(f"Error in comprehensive learning loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _ultra_comprehensive_model_update(self):
        """Ultra-comprehensive daily model update"""
        print("📅 Running ultra-comprehensive model update...")

        # Update top symbols from each category (expanded)
        symbols_to_update = []
        for data_type, config in self.data_sources.items():
            symbols_to_update.extend(config['symbols'][:12])  # Top 12 from each category

        # Use thread pool for parallel updates
        futures = []
        for symbol in symbols_to_update:
            data_type = self._get_data_type_for_symbol(symbol)
            future = self.executor.submit(self.train_comprehensive_model, symbol, data_type)
            futures.append(future)

        # Wait for completion with results
        successful_updates = 0
        total_symbols = len(futures)

        for future in futures:
            try:
                result = future.result(timeout=900)  # 15 minute timeout
                if result['status'] == 'success':
                    print(f"✅ Comprehensive update for {result['symbol']}: R²={result['r2']:.4f}, Features={result['feature_count']}")
                    successful_updates += 1
                else:
                    print(f"❌ Failed to update {result.get('symbol', 'unknown')}: {result.get('reason', 'unknown')}")
            except Exception as e:
                print(f"⚠️  Error in comprehensive model update: {e}")

        print(f"✅ Ultra-comprehensive update complete: {successful_updates}/{total_symbols} successful")

    def _incremental_comprehensive_update(self):
        """Incremental comprehensive model update"""
        if self.is_learning:
            return

        self.is_learning = True
        try:
            print("🔄 Running incremental comprehensive update...")

            # Quick update for top performers (expanded)
            top_symbols = self._get_top_performing_symbols(15)

            for symbol in top_symbols:
                try:
                    data_type = self._get_data_type_for_symbol(symbol)
                    result = self.train_comprehensive_model(symbol, data_type)
                    if result['status'] == 'success':
                        print(f"✅ Incremental update for {symbol}: R²={result['r2']:.4f}")
                except Exception as e:
                    logging.error(f"Error in incremental update for {symbol}: {e}")

        finally:
            self.is_learning = False

    def _get_data_type_for_symbol(self, symbol: str) -> str:
        """Determine comprehensive data type for a symbol"""
        for data_type, config in self.data_sources.items():
            if symbol in config['symbols']:
                return data_type
        return 'stocks_major'  # Default

    def _get_top_performing_symbols(self, n: int = 15) -> List[str]:
        """Get top performing symbols based on comprehensive metrics"""
        if not self.performance_history:
            # Return expanded default top symbols
            return [
                'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'META', 'AMZN',
                'BTC', 'ETH', 'EURUSD=X', 'GC=F', '^GSPC', '^IXIC'
            ]

        symbol_scores = {}
        for symbol, history in self.performance_history.items():
            if history:
                latest = history[-1]
                # Enhanced scoring based on multiple metrics
                r2_score = max(0, latest['r2']) * 0.4  # Only positive R² contributes
                dir_acc_score = (latest['directional_accuracy'] - 0.5) * 0.4  # Above 50% directional accuracy
                mape_score = max(0, (1 - latest['mape']/100)) * 0.2  # Lower MAPE is better
                score = r2_score + dir_acc_score + mape_score
                symbol_scores[symbol] = score

        # Sort by score and return top n
        sorted_symbols = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, score in sorted_symbols[:n]]

    def _save_comprehensive_model(self, symbol: str):
        """Save comprehensive model components"""
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_comprehensive_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{symbol}_comprehensive_scaler.pkl")
            selector_path = os.path.join(self.model_dir, f"{symbol}_comprehensive_selector.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump(self.models[symbol], f)

            if symbol in self.scalers:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[symbol], f)

            if symbol in self.feature_selectors:
                with open(selector_path, 'wb') as f:
                    pickle.dump(self.feature_selectors[symbol], f)

        except Exception as e:
            logging.error(f"Error saving comprehensive model for {symbol}: {e}")

    def _load_existing_models(self):
        """Load existing trained comprehensive models"""
        try:
            for filename in os.listdir(self.model_dir):
                if filename.endswith('_comprehensive_model.pkl'):
                    symbol = filename.replace('_comprehensive_model.pkl', '')
                    model_path = os.path.join(self.model_dir, filename)
                    scaler_path = os.path.join(self.model_dir, f"{symbol}_comprehensive_scaler.pkl")
                    selector_path = os.path.join(self.model_dir, f"{symbol}_comprehensive_selector.pkl")

                    with open(model_path, 'rb') as f:
                        self.models[symbol] = pickle.load(f)

                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)

                    if os.path.exists(selector_path):
                        with open(selector_path, 'rb') as f:
                            self.feature_selectors[symbol] = pickle.load(f)

                    print(f"✅ Loaded existing comprehensive model for {symbol}")

        except Exception as e:
            print(f"⚠️  Error loading existing comprehensive models: {e}")

    def get_comprehensive_performance_report(self) -> Dict:
        """Generate ultra-comprehensive performance report"""
        report = {
            'overall_stats': {},
            'symbol_performance': {},
            'improvement_trends': {},
            'best_performers': [],
            'worst_performers': [],
            'feature_importance': {},
            'market_condition_analysis': {},
            'data_coverage_stats': {}
        }

        # Overall statistics
        all_r2_scores = []
        all_directional_accuracies = []
        all_mape_scores = []
        all_rmse_scores = []

        for symbol, history in self.performance_history.items():
            if history:
                latest = history[-1]
                all_r2_scores.append(latest['r2'])
                all_directional_accuracies.append(latest['directional_accuracy'])
                all_mape_scores.append(latest['mape'])
                all_rmse_scores.append(latest['rmse'])

                report['symbol_performance'][symbol] = {
                    'latest_r2': latest['r2'],
                    'latest_directional_accuracy': latest['directional_accuracy'],
                    'latest_mape': latest['mape'],
                    'latest_rmse': latest['rmse'],
                    'improvement_trend': self._calculate_improvement_trend(history),
                    'n_updates': len(history),
                    'selected_features': latest.get('selected_features', []),
                    'data_points': latest.get('data_points', 0),
                    'feature_count': latest.get('feature_count', 0)
                }

        if all_r2_scores:
            report['overall_stats'] = {
                'avg_r2': np.mean(all_r2_scores),
                'avg_directional_accuracy': np.mean(all_directional_accuracies),
                'avg_mape': np.mean(all_mape_scores),
                'avg_rmse': np.mean(all_rmse_scores),
                'best_r2': max(all_r2_scores),
                'best_directional_accuracy': max(all_directional_accuracies),
                'worst_r2': min(all_r2_scores),
                'total_models': len(self.models),
                'total_symbols_tracked': len(self.performance_history),
                'total_data_points': sum(h[-1]['data_points'] for h in self.performance_history.values() if h),
                'avg_features_per_model': np.mean([h[-1]['feature_count'] for h in self.performance_history.values() if h])
            }

        # Sort symbols by performance
        sorted_symbols = sorted(
            report['symbol_performance'].items(),
            key=lambda x: x[1]['latest_r2'],
            reverse=True
        )

        report['best_performers'] = sorted_symbols[:10]
        report['worst_performers'] = sorted_symbols[-10:]

        # Data coverage statistics
        total_symbols_available = sum(len(config['symbols']) for config in self.data_sources.values())
        report['data_coverage_stats'] = {
            'total_symbols_available': total_symbols_available,
            'symbols_with_models': len(self.models),
            'coverage_percentage': len(self.models) / total_symbols_available * 100,
            'data_types_covered': len(self.data_sources)
        }

        return report


def main():
    """
    Main function for ultra-comprehensive data framework
    """
    print("🚀 FSOT 2.5 Ultra-Comprehensive Data Framework")
    print("=" * 70)

    # Initialize ultra-comprehensive framework
    ucdf = ComprehensiveDataFramework()

    print(f"📊 Initialized with {len(ucdf.models)} existing comprehensive models")
    total_symbols = sum(len(config['symbols']) for config in ucdf.data_sources.values())
    print(f"🎯 Tracking {total_symbols} symbols across {len(ucdf.data_sources)} data types")

    # Initial comprehensive training for diverse symbols
    print("\n🔬 Performing initial ultra-comprehensive model training...")

    # Diverse symbol selection across all categories
    key_symbols = [
        # Major stocks
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'META', 'AMZN',
        # Crypto
        'BTC', 'ETH', 'BNB', 'ADA', 'SOL',
        # Forex
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X',
        # Commodities
        'GC=F', 'CL=F', 'SI=F',
        # Indices
        '^GSPC', '^IXIC', '^VIX'
    ]

    for symbol in key_symbols:
        print(f"📈 Comprehensive training {symbol}...")
        data_type = ucdf._get_data_type_for_symbol(symbol)
        result = ucdf.train_comprehensive_model(symbol, data_type)
        if result['status'] == 'success':
            print(f"✅ R²: {result['r2']:.4f}, Features: {result['feature_count']}, Data Points: {result['data_points']}")
        else:
            print(f"❌ Failed: {result.get('reason', 'unknown')}")

    # Generate comprehensive performance report
    print("\n📊 Generating ultra-comprehensive performance report...")
    report = ucdf.get_comprehensive_performance_report()

    if report['overall_stats']:
        stats = report['overall_stats']
        print("\n🎯 Ultra-Comprehensive Overall Performance:")
        print(f"  Average R²: {stats['avg_r2']:.4f}")
        print(f"  Average Directional Accuracy: {stats['avg_directional_accuracy']:.1%}")
        print(f"  Average MAPE: {stats['avg_mape']:.2f}%")
        print(f"  Average RMSE: {stats['avg_rmse']:.6f}")
        print(f"  Best R²: {stats['best_r2']:.4f}")
        print(f"  Total Models: {stats['total_models']}")
        print(f"  Total Symbols Tracked: {stats['total_symbols_tracked']}")
        print(f"  Total Data Points: {stats['total_data_points']:,}")
        print(f"  Average Features per Model: {stats['avg_features_per_model']:.1f}")

    # Data coverage stats
    coverage = report['data_coverage_stats']
    print("\n📊 Data Coverage:")
    print(f"  Total Symbols Available: {coverage['total_symbols_available']}")
    print(f"  Symbols with Models: {coverage['symbols_with_models']}")
    print(f"  Coverage Percentage: {coverage['coverage_percentage']:.1f}%")
    print(f"  Data Types Covered: {coverage['data_types_covered']}")

    # Show top performers
    if report['best_performers']:
        print("\n🏆 Ultra-Comprehensive Top Performers:")
        for i, (symbol, perf) in enumerate(report['best_performers'][:5], 1):
            print(f"  {i}. {symbol}: R²={perf['latest_r2']:.4f}, DirAcc={perf['latest_directional_accuracy']:.1%}, Features={perf['feature_count']}")

    print("\n✅ Ultra-comprehensive data framework initialized!")
    print("🔄 Models will continue to learn and improve automatically")
    print("📊 Check fsot_comprehensive_data.log for detailed logs")

    # Keep the system running
    try:
        while True:
            time.sleep(60)  # Check every minute for scheduled tasks
    except KeyboardInterrupt:
        print("\n🛑 Shutting down ultra-comprehensive data framework...")
        ucdf.executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
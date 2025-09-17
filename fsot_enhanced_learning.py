"""
FSOT 2.5 Enhanced Continuous Learning Framework
===============================================

Advanced system with improved features, better models, and sophisticated
financial prediction techniques for superior accuracy.
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
    filename='fsot_enhanced_learning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class EnhancedContinuousLearningFramework:
    """
    Enhanced continuous learning system with advanced ML techniques
    """

    def __init__(self, model_dir='enhanced_models', data_dir='enhanced_training_data'):
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

        # Initialize enhanced data sources
        self._initialize_enhanced_data_sources()

        # Load existing models if available
        self._load_existing_models()

        # Start continuous learning thread
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.learning_thread = threading.Thread(target=self._enhanced_learning_loop, daemon=True)
        self.learning_thread.start()

    def _initialize_enhanced_data_sources(self):
        """Initialize comprehensive data sources with more symbols"""
        self.data_sources = {
            'stocks_major': {
                'provider': 'yfinance',
                'symbols': [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
                    'NFLX', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'JNJ',
                    'PFE', 'UNH', 'MRK', 'ABT', 'SPY', 'QQQ', 'IWM', 'DIA',
                    'XOM', 'CVX', 'KO', 'PEP', 'WMT', 'HD', 'MCD', 'DIS',
                    'VZ', 'T', 'IBM', 'ORCL', 'CRM', 'ADBE', 'INTC', 'AMD'
                ],
                'frequency': '1d',
                'lookback_days': 365*3  # 3 years for better training
            },
            'crypto': {
                'provider': 'coingecko',
                'symbols': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'DOGE', 'AVAX', 'MATIC', 'LINK'],
                'frequency': 'daily',
                'lookback_days': 365*2
            },
            'forex': {
                'provider': 'yfinance',
                'symbols': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X', 'NZDUSD=X'],
                'frequency': '1d',
                'lookback_days': 365*2
            },
            'commodities': {
                'provider': 'yfinance',
                'symbols': ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'HG=F', 'ZC=F'],  # Gold, Silver, Oil, Natural Gas, Copper, Corn
                'frequency': '1d',
                'lookback_days': 365*2
            },
            'indices': {
                'provider': 'yfinance',
                'symbols': ['^GSPC', '^IXIC', '^DJI', '^VIX', '^TNX', '^RUT', '^NYA'],
                'frequency': '1d',
                'lookback_days': 365*3
            }
        }

    def _load_existing_models(self):
        """Load existing trained models"""
        try:
            for filename in os.listdir(self.model_dir):
                if filename.endswith('_model.pkl'):
                    symbol = filename.replace('_model.pkl', '')
                    model_path = os.path.join(self.model_dir, filename)
                    scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
                    selector_path = os.path.join(self.model_dir, f"{symbol}_selector.pkl")

                    with open(model_path, 'rb') as f:
                        self.models[symbol] = pickle.load(f)

                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)

                    if os.path.exists(selector_path):
                        with open(selector_path, 'rb') as f:
                            self.feature_selectors[symbol] = pickle.load(f)

                    print(f"✅ Loaded existing enhanced model for {symbol}")

        except Exception as e:
            print(f"⚠️  Error loading existing models: {e}")

    def fetch_enhanced_data(self, symbol: str, data_type: str = 'stocks_major') -> pd.DataFrame:
        """
        Fetch enhanced data with more comprehensive features
        """
        try:
            if data_type in ['stocks_major', 'forex', 'commodities', 'indices']:
                # Get extended historical data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.data_sources[data_type]['lookback_days'])

                data = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if data.empty:
                    return pd.DataFrame()

                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                # Add comprehensive technical indicators
                data = self._add_enhanced_technical_indicators(data)

                return data

            elif data_type == 'crypto':
                return self._fetch_enhanced_crypto_data(symbol)

        except Exception as e:
            logging.error(f"Error fetching enhanced data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_enhanced_crypto_data(self, symbol: str) -> pd.DataFrame:
        """Fetch enhanced cryptocurrency data"""
        try:
            coin_id = symbol.lower()
            if symbol == 'BTC':
                coin_id = 'bitcoin'
            elif symbol == 'ETH':
                coin_id = 'ethereum'

            days = self.data_sources['crypto']['lookback_days']

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
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Add other price data if available
            if 'total_volumes' in data:
                df['volume'] = [x[1] for x in data['total_volumes']]

            # Create OHLC from close (approximation for crypto)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df[['open', 'close']].max(axis=1) * 1.05  # Higher volatility
            df['low'] = df[['open', 'close']].min(axis=1) * 0.95   # Higher volatility

            # Add technical indicators
            df = self._add_enhanced_technical_indicators(df)

            return df

        except Exception as e:
            logging.error(f"Error fetching enhanced crypto data for {symbol}: {e}")
            return pd.DataFrame()

    def _add_enhanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators using ta library"""
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

            # Trend indicators
            df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
            df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            df['EMA_10'] = ta.trend.ema_indicator(df['Close'], window=10)
            df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
            df['EMA_50'] = ta.trend.ema_indicator(df['Close'], window=50)

            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_hist'] = macd.macd_diff()

            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)

            # Stochastic Oscillator
            stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_middle'] = bb.bollinger_mband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = bb.bollinger_wband()

            # Volume indicators
            if df['Volume'].sum() > 0:
                df['volume_sma'] = ta.trend.sma_indicator(df['Volume'], window=20)
                df['volume_ratio'] = df['Volume'] / df['volume_sma']
                df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

            # Volatility indicators
            df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
            df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)

            # Momentum indicators
            df['ROC_10'] = ta.momentum.roc(df['Close'], window=10)
            df['MOM_10'] = ta.momentum.roc(df['Close'], window=10)  # Rate of change

            # Support/Resistance levels (simplified)
            df['support_20'] = df['Low'].rolling(window=20).min()
            df['resistance_20'] = df['High'].rolling(window=20).max()

            # Ichimoku Cloud (simplified)
            df['tenkan_sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
            df['kijun_sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2

            # Williams %R
            df['williams_r'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], window=14)

            # Commodity Channel Index
            df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)

            # Drop NaN values
            df = df.dropna()

            return df

        except Exception as e:
            logging.error(f"Error adding enhanced technical indicators: {e}")
            return df

    def train_enhanced_model(self, symbol: str, data_type: str = 'stocks_major') -> Dict:
        """
        Train enhanced model with advanced techniques
        """
        try:
            print(f"🔬 Training enhanced model for {symbol}...")

            # Fetch enhanced data
            df = self.fetch_enhanced_data(symbol, data_type)

            if df.empty or len(df) < 200:  # Need more data for enhanced training
                print(f"❌ Insufficient enhanced data for {symbol}")
                return {'status': 'failed', 'reason': 'insufficient_data'}

            # Prepare enhanced features and target
            features = self._prepare_enhanced_features(df)

            # Use multiple prediction horizons
            targets = self._create_multi_horizon_targets(df)

            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1))
            for target_name in targets.keys():
                valid_idx &= ~targets[target_name].isna()

            features = features[valid_idx]
            for target_name in targets.keys():
                targets[target_name] = targets[target_name][valid_idx]

            if len(features) < 100:
                return {'status': 'failed', 'reason': 'insufficient_valid_data'}

            # Feature selection
            if symbol not in self.feature_selectors:
                self.feature_selectors[symbol] = SelectKBest(score_func=f_regression, k=min(30, features.shape[1]))

            features_selected = self.feature_selectors[symbol].fit_transform(features, targets['target_1d'])
            selected_feature_names = features.columns[self.feature_selectors[symbol].get_support()].tolist()

            # Split data with time series split
            tscv = TimeSeriesSplit(n_splits=3)
            splits = list(tscv.split(features_selected))

            # Use the last split for final training
            train_idx, test_idx = splits[-1]
            X_train = features_selected[train_idx]
            X_test = features_selected[test_idx]
            y_train = targets['target_1d'].iloc[train_idx]
            y_test = targets['target_1d'].iloc[test_idx]

            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = RobustScaler()  # More robust to outliers

            X_train_scaled = self.scalers[symbol].fit_transform(X_train)
            X_test_scaled = self.scalers[symbol].transform(X_test)

            # Train enhanced ensemble model
            if symbol not in self.models:
                self.models[symbol] = self._create_enhanced_ensemble_model()
                print(f"🆕 Created enhanced model for {symbol}")
            else:
                print(f"🔄 Updating enhanced model for {symbol}")

            # Train the model
            self.models[symbol].fit(X_train_scaled, y_train)

            # Evaluate performance
            y_pred = self.models[symbol].predict(X_test_scaled)

            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Calculate directional accuracy
            actual_direction = np.sign(y_test)
            predicted_direction = np.sign(y_pred)
            directional_accuracy = np.mean(actual_direction == predicted_direction)

            # Calculate additional metrics
            mape = np.mean(np.abs((y_test - y_pred) / y_test.replace(0, np.inf))) * 100

            # Store performance history
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []

            self.performance_history[symbol].append({
                'timestamp': datetime.now(),
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'mape': mape,
                'n_samples': len(y_test),
                'selected_features': selected_feature_names
            })

            # Save enhanced model
            self._save_enhanced_model(symbol)

            print(".6f")
            print(".4f")
            print(".1%")
            print(".2f")
            return {
                'status': 'success',
                'symbol': symbol,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'mape': mape,
                'n_samples': len(y_test),
                'selected_features': selected_feature_names
            }

        except Exception as e:
            logging.error(f"Error training enhanced model for {symbol}: {e}")
            return {'status': 'failed', 'reason': str(e)}

    def _prepare_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare comprehensive feature set for enhanced training"""
        feature_cols = []

        # Price and volume features
        price_features = ['Close', 'High', 'Low', 'Open', 'Volume', 'returns', 'log_returns']
        feature_cols.extend([col for col in price_features if col in df.columns])

        # Trend features
        trend_features = [
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50',
            'MACD', 'MACD_signal', 'MACD_hist'
        ]
        feature_cols.extend([col for col in trend_features if col in df.columns])

        # Momentum features
        momentum_features = [
            'RSI', 'stoch_k', 'stoch_d', 'ROC_10', 'MOM_10', 'williams_r', 'CCI'
        ]
        feature_cols.extend([col for col in momentum_features if col in df.columns])

        # Volatility features
        volatility_features = [
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width', 'ATR', 'volatility_20'
        ]
        feature_cols.extend([col for col in volatility_features if col in df.columns])

        # Volume features
        volume_features = ['volume_sma', 'volume_ratio', 'OBV']
        feature_cols.extend([col for col in volume_features if col in df.columns])

        # Support/Resistance and Ichimoku
        sr_features = ['support_20', 'resistance_20', 'tenkan_sen', 'kijun_sen']
        feature_cols.extend([col for col in sr_features if col in df.columns])

        # Lagged features (past values)
        base_cols = ['Close', 'returns', 'RSI', 'MACD']
        for col in base_cols:
            if col in df.columns:
                for lag in [1, 2, 3, 5, 10]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                    feature_cols.append(f'{col}_lag_{lag}')

        # Rolling statistics
        for col in ['returns', 'Close']:
            if col in df.columns:
                for window in [5, 10, 20]:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                    df[f'{col}_rolling_skew_{window}'] = df[col].rolling(window=window).skew()
                    feature_cols.extend([
                        f'{col}_rolling_mean_{window}',
                        f'{col}_rolling_std_{window}',
                        f'{col}_rolling_skew_{window}'
                    ])

        if not feature_cols:
            feature_cols = ['Close']

        return df[feature_cols].copy()

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

    def _create_enhanced_ensemble_model(self):
        """Create enhanced ensemble model with multiple algorithms"""
        from sklearn.ensemble import VotingRegressor, StackingRegressor

        # Base models
        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )

        et = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

        # Meta model
        ridge = Ridge(alpha=0.1, random_state=42)

        # Stacking ensemble
        estimators = [
            ('rf', rf),
            ('et', et),
            ('gb', gb)
        ]

        ensemble = StackingRegressor(
            estimators=estimators,
            final_estimator=ridge,
            cv=3,
            n_jobs=-1
        )

        return ensemble

    def _save_enhanced_model(self, symbol: str):
        """Save enhanced model components"""
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
            selector_path = os.path.join(self.model_dir, f"{symbol}_selector.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump(self.models[symbol], f)

            if symbol in self.scalers:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[symbol], f)

            if symbol in self.feature_selectors:
                with open(selector_path, 'wb') as f:
                    pickle.dump(self.feature_selectors[symbol], f)

        except Exception as e:
            logging.error(f"Error saving enhanced model for {symbol}: {e}")

    def _enhanced_learning_loop(self):
        """Enhanced continuous learning loop"""
        print("🚀 Starting enhanced continuous learning loop...")

        # Schedule enhanced updates
        schedule.every().day.at("01:00").do(self._comprehensive_model_update)
        schedule.every(6).hours.do(self._incremental_model_update)

        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logging.error(f"Error in enhanced learning loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _comprehensive_model_update(self):
        """Comprehensive daily model update"""
        print("📅 Running comprehensive enhanced model update...")

        # Update top symbols from each category
        symbols_to_update = []
        for data_type, config in self.data_sources.items():
            symbols_to_update.extend(config['symbols'][:8])  # Top 8 from each category

        # Use thread pool for parallel updates
        futures = []
        for symbol in symbols_to_update:
            data_type = self._get_data_type_for_symbol(symbol)
            future = self.executor.submit(self.train_enhanced_model, symbol, data_type)
            futures.append(future)

        # Wait for completion with results
        successful_updates = 0
        for future in futures:
            try:
                result = future.result(timeout=600)  # 10 minute timeout
                if result['status'] == 'success':
                    print(f"✅ Enhanced update for {result['symbol']}: R²={result['r2']:.4f}")
                    successful_updates += 1
                else:
                    print(f"❌ Failed to update {result.get('symbol', 'unknown')}: {result.get('reason', 'unknown')}")
            except Exception as e:
                print(f"⚠️  Error in enhanced model update: {e}")

        print(f"✅ Comprehensive update complete: {successful_updates}/{len(futures)} successful")

    def _incremental_model_update(self):
        """Incremental model update with latest data"""
        if self.is_learning:
            return

        self.is_learning = True
        try:
            print("🔄 Running incremental enhanced model update...")

            # Quick update for top performers
            top_symbols = self._get_top_performing_symbols(10)

            for symbol in top_symbols:
                try:
                    data_type = self._get_data_type_for_symbol(symbol)
                    result = self.train_enhanced_model(symbol, data_type)
                    if result['status'] == 'success':
                        print(f"✅ Incremental update for {symbol}: R²={result['r2']:.4f}")
                except Exception as e:
                    logging.error(f"Error in incremental update for {symbol}: {e}")

        finally:
            self.is_learning = False

    def _get_data_type_for_symbol(self, symbol: str) -> str:
        """Determine data type for a symbol"""
        for data_type, config in self.data_sources.items():
            if symbol in config['symbols']:
                return data_type
        return 'stocks_major'  # Default

    def _get_top_performing_symbols(self, n: int = 10) -> List[str]:
        """Get top performing symbols based on recent performance"""
        if not self.performance_history:
            # Return default top symbols
            return ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'META', 'AMZN']

        symbol_scores = {}
        for symbol, history in self.performance_history.items():
            if history:
                latest = history[-1]
                # Score based on R² and directional accuracy
                score = latest['r2'] * 0.7 + (latest['directional_accuracy'] - 0.5) * 0.3
                symbol_scores[symbol] = score

        # Sort by score and return top n
        sorted_symbols = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
        return [symbol for symbol, score in sorted_symbols[:n]]

    def get_enhanced_performance_report(self) -> Dict:
        """Generate comprehensive enhanced performance report"""
        report = {
            'overall_stats': {},
            'symbol_performance': {},
            'improvement_trends': {},
            'best_performers': [],
            'worst_performers': [],
            'feature_importance': {},
            'market_condition_analysis': {}
        }

        # Overall statistics
        all_r2_scores = []
        all_directional_accuracies = []
        all_mape_scores = []

        for symbol, history in self.performance_history.items():
            if history:
                latest = history[-1]
                all_r2_scores.append(latest['r2'])
                all_directional_accuracies.append(latest['directional_accuracy'])
                all_mape_scores.append(latest['mape'])

                report['symbol_performance'][symbol] = {
                    'latest_r2': latest['r2'],
                    'latest_directional_accuracy': latest['directional_accuracy'],
                    'latest_mape': latest['mape'],
                    'improvement_trend': self._calculate_improvement_trend(history),
                    'n_updates': len(history),
                    'selected_features': latest.get('selected_features', [])
                }

        if all_r2_scores:
            report['overall_stats'] = {
                'avg_r2': np.mean(all_r2_scores),
                'avg_directional_accuracy': np.mean(all_directional_accuracies),
                'avg_mape': np.mean(all_mape_scores),
                'best_r2': max(all_r2_scores),
                'best_directional_accuracy': max(all_directional_accuracies),
                'total_models': len(self.models),
                'total_symbols_tracked': len(self.performance_history)
            }

        # Sort symbols by performance
        sorted_symbols = sorted(
            report['symbol_performance'].items(),
            key=lambda x: x[1]['latest_r2'],
            reverse=True
        )

        report['best_performers'] = sorted_symbols[:5]
        report['worst_performers'] = sorted_symbols[-5:]

        return report

    def _calculate_improvement_trend(self, history: List[Dict]) -> float:
        """Calculate improvement trend in performance"""
        if len(history) < 2:
            return 0.0

        # Calculate trend in R² scores
        r2_scores = [h['r2'] for h in history[-10:]]  # Last 10 updates

        if len(r2_scores) < 2:
            return 0.0

        # Simple linear trend
        x = np.arange(len(r2_scores))
        slope = np.polyfit(x, r2_scores, 1)[0]

        return slope


def main():
    """
    Main function for enhanced continuous learning system
    """
    print("🚀 FSOT 2.5 Enhanced Continuous Learning Framework")
    print("=" * 60)

    # Initialize enhanced continuous learning system
    eclf = EnhancedContinuousLearningFramework()

    print(f"📊 Initialized with {len(eclf.models)} existing enhanced models")
    print(f"🎯 Tracking {sum(len(config['symbols']) for config in eclf.data_sources.values())} symbols")

    # Initial enhanced training for key symbols
    print("\n🔬 Performing initial enhanced model training...")

    key_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'META', 'AMZN',
        'BTC', 'ETH', 'EURUSD=X', 'GC=F'
    ]

    for symbol in key_symbols:
        print(f"📈 Enhanced training {symbol}...")
        data_type = eclf._get_data_type_for_symbol(symbol)
        result = eclf.train_enhanced_model(symbol, data_type)
        if result['status'] == 'success':
            print(f"✅ R²: {result['r2']:.4f}")
        else:
            print(f"❌ Failed: {result.get('reason', 'unknown')}")

    # Generate enhanced performance report
    print("\n📊 Generating enhanced performance report...")
    report = eclf.get_enhanced_performance_report()

    if report['overall_stats']:
        stats = report['overall_stats']
        print("\n🎯 Enhanced Overall Performance:")
        print(f"  Average R²: {stats['avg_r2']:.4f}")
        print(f"  Average Directional Accuracy: {stats['avg_directional_accuracy']:.1%}")
        print(f"  Average MAPE: {stats['avg_mape']:.2f}%")
        print(f"  Total Models: {stats['total_models']}")
        print(f"  Symbols Tracked: {stats['total_symbols_tracked']}")

    # Show top performers
    if report['best_performers']:
        print("\n🏆 Enhanced Top Performers:")
        for i, (symbol, perf) in enumerate(report['best_performers'][:3], 1):
            print(f"  {i}. {symbol}: R²={perf['latest_r2']:.4f}")
    print("\n✅ Enhanced continuous learning system initialized!")
    print("🔄 Models will continue to learn and improve automatically")
    print("📊 Check fsot_enhanced_learning.log for detailed logs")

    # Keep the system running
    try:
        while True:
            time.sleep(60)  # Check every minute for scheduled tasks
    except KeyboardInterrupt:
        print("\n🛑 Shutting down enhanced continuous learning system...")
        eclf.executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
"""
FSOT 2.5 Continuous Learning Framework
=====================================

Advanced system for continuous model training, real-time data integration,
and adaptive learning capabilities to improve accuracy over time.
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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from typing import Dict, List, Optional, Tuple
import threading
import time
import schedule
from concurrent.futures import ThreadPoolExecutor
import logging

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    filename='fsot_continuous_learning.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ContinuousLearningFramework:
    """
    Advanced continuous learning system for FSOT 2.5
    """

    def __init__(self, model_dir='models', data_dir='training_data'):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.models = {}
        self.scalers = {}
        self.performance_history = {}
        self.data_sources = {}
        self.is_learning = False

        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/historical", exist_ok=True)
        os.makedirs(f"{data_dir}/streaming", exist_ok=True)

        # Initialize data sources
        self._initialize_data_sources()

        # Load existing models if available
        self._load_existing_models()

        # Start continuous learning thread
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.learning_thread.start()

    def _initialize_data_sources(self):
        """Initialize comprehensive data sources"""
        self.data_sources = {
            'stocks': {
                'provider': 'yfinance',
                'symbols': [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
                    'NFLX', 'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'JNJ',
                    'PFE', 'UNH', 'MRK', 'ABT', 'SPY', 'QQQ', 'IWM', 'DIA'
                ],
                'frequency': '1d',
                'lookback_days': 365*2
            },
            'crypto': {
                'provider': 'coingecko',
                'symbols': ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'DOT', 'DOGE', 'AVAX'],
                'frequency': 'daily',
                'lookback_days': 365
            },
            'forex': {
                'provider': 'yfinance',
                'symbols': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X'],
                'frequency': '1d',
                'lookback_days': 365
            },
            'commodities': {
                'provider': 'yfinance',
                'symbols': ['GC=F', 'SI=F', 'CL=F', 'NG=F'],  # Gold, Silver, Oil, Natural Gas
                'frequency': '1d',
                'lookback_days': 365
            },
            'indices': {
                'provider': 'yfinance',
                'symbols': ['^GSPC', '^IXIC', '^DJI', '^VIX', '^TNX'],
                'frequency': '1d',
                'lookback_days': 365*2
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

                    with open(model_path, 'rb') as f:
                        self.models[symbol] = pickle.load(f)

                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            self.scalers[symbol] = pickle.load(f)

                    print(f"✅ Loaded existing model for {symbol}")

        except Exception as e:
            print(f"⚠️  Error loading existing models: {e}")

    def fetch_latest_data(self, symbol: str, data_type: str = 'stocks') -> pd.DataFrame:
        """
        Fetch the most up-to-date data for a symbol
        """
        try:
            if data_type == 'stocks':
                # Get last 2 years of data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.data_sources['stocks']['lookback_days'])

                data = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if data.empty:
                    return pd.DataFrame()

                # Flatten MultiIndex columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                # Add technical indicators
                data = self._add_technical_indicators(data)

                return data

            elif data_type == 'crypto':
                # Use CoinGecko API for crypto
                return self._fetch_crypto_data(symbol)

            elif data_type == 'forex':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.data_sources['forex']['lookback_days'])

                data = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                return data

            elif data_type == 'commodities':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.data_sources['commodities']['lookback_days'])

                data = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.droplevel(1)

                return data

        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_crypto_data(self, symbol: str) -> pd.DataFrame:
        """Fetch cryptocurrency data from CoinGecko"""
        try:
            # Map common symbols to CoinGecko IDs
            symbol_map = {
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'BNB': 'binancecoin',
                'ADA': 'cardano',
                'SOL': 'solana',
                'DOT': 'polkadot',
                'DOGE': 'dogecoin',
                'AVAX': 'avalanche-2'
            }

            coin_id = symbol_map.get(symbol, symbol.lower())
            days = self.data_sources['crypto']['lookback_days']

            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }

            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # Convert to DataFrame
            prices = data['prices']
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Add other price data if available
            if 'total_volumes' in data:
                df['volume'] = [x[1] for x in data['total_volumes']]

            # Create OHLC from close (approximation)
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df['high'] = df[['open', 'close']].max(axis=1) * 1.02  # Approximate
            df['low'] = df[['open', 'close']].min(axis=1) * 0.98   # Approximate

            return df

        except Exception as e:
            logging.error(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # Basic price indicators
            df['returns'] = df['Close'].pct_change()
            df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))

            # Moving averages
            for period in [5, 10, 20, 50, 200]:
                df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
                df[f'EMA_{period}'] = df['Close'].ewm(span=period).mean()

            # Volatility indicators
            df['volatility_20'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
            df['volatility_60'] = df['returns'].rolling(window=60).std() * np.sqrt(252)

            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_hist'] = df['MACD'] - df['MACD_signal']

            # Bollinger Bands
            df['BB_middle'] = df['Close'].rolling(window=20).mean()
            df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']

            # Volume indicators
            if 'Volume' in df.columns:
                df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
                df['volume_ratio'] = df['Volume'] / df['volume_sma_20']

            # Momentum indicators
            for period in [1, 3, 5, 10]:
                df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1

            # Drop NaN values
            df = df.dropna()

            return df

        except Exception as e:
            logging.error(f"Error adding technical indicators: {e}")
            return df

    def train_online_model(self, symbol: str, data_type: str = 'stocks') -> Dict:
        """
        Train or update model for a symbol using online learning
        """
        try:
            print(f"🔬 Training model for {symbol}...")

            # Fetch latest data
            df = self.fetch_latest_data(symbol, data_type)

            if df.empty or len(df) < 100:
                print(f"❌ Insufficient data for {symbol}")
                return {'status': 'failed', 'reason': 'insufficient_data'}

            # Prepare features and target
            features = self._prepare_features(df)
            target = df['returns'].shift(-1).fillna(0)  # Predict next day's return

            # Remove NaN values
            valid_idx = ~(features.isna().any(axis=1) | target.isna())
            features = features[valid_idx]
            target = target[valid_idx]

            if len(features) < 50:
                return {'status': 'failed', 'reason': 'insufficient_valid_data'}

            # Split data (use most recent 30% for testing)
            split_idx = int(len(features) * 0.7)
            X_train = features.iloc[:split_idx]
            X_test = features.iloc[split_idx:]
            y_train = target.iloc[:split_idx]
            y_test = target.iloc[split_idx:]

            # Scale features
            if symbol not in self.scalers:
                self.scalers[symbol] = StandardScaler()

            X_train_scaled = self.scalers[symbol].fit_transform(X_train)
            X_test_scaled = self.scalers[symbol].transform(X_test)

            # Train or update model
            if symbol not in self.models:
                # New model - use ensemble approach
                self.models[symbol] = self._create_ensemble_model()
                print(f"🆕 Created new model for {symbol}")
            else:
                print(f"🔄 Updating existing model for {symbol}")

            # Online learning - partial fit if supported
            if hasattr(self.models[symbol], 'partial_fit'):
                # For models that support online learning
                self.models[symbol].partial_fit(X_train_scaled, y_train)
            else:
                # Retrain with all available data
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

            # Store performance history
            if symbol not in self.performance_history:
                self.performance_history[symbol] = []

            self.performance_history[symbol].append({
                'timestamp': datetime.now(),
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'n_samples': len(y_test)
            })

            # Save model
            self._save_model(symbol)

            print(f"  MSE: {mse:.6f}")
            print(f"  R²: {r2:.4f}")
            print(f"  Directional Accuracy: {directional_accuracy:.1%}")

            return {
                'status': 'success',
                'symbol': symbol,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'n_samples': len(y_test)
            }

        except Exception as e:
            logging.error(f"Error training model for {symbol}: {e}")
            return {'status': 'failed', 'reason': str(e)}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for training"""
        feature_cols = []

        # Price-based features
        price_features = ['Close', 'Volume', 'returns', 'log_returns']
        feature_cols.extend([col for col in price_features if col in df.columns])

        # Technical indicators
        tech_features = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_200',
            'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
            'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
            'volatility_20', 'volatility_60'
        ]
        feature_cols.extend([col for col in tech_features if col in df.columns])

        # Momentum features
        momentum_features = ['momentum_1', 'momentum_3', 'momentum_5', 'momentum_10']
        feature_cols.extend([col for col in momentum_features if col in df.columns])

        # Volume features
        volume_features = ['volume_sma_20', 'volume_ratio']
        feature_cols.extend([col for col in volume_features if col in df.columns])

        if not feature_cols:
            # Fallback to basic features
            feature_cols = ['Close']
            if 'Volume' in df.columns:
                feature_cols.append('Volume')

        return df[feature_cols].copy()

    def _create_ensemble_model(self):
        """Create ensemble model for better performance"""
        from sklearn.ensemble import VotingRegressor

        # Create multiple models
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )

        sgd = SGDRegressor(
            max_iter=1000,
            tol=1e-3,
            random_state=42
        )

        # Ensemble model
        ensemble = VotingRegressor([
            ('rf', rf),
            ('gb', gb),
            ('sgd', sgd)
        ])

        return ensemble

    def _save_model(self, symbol: str):
        """Save trained model and scaler"""
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_model.pkl")
            scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")

            with open(model_path, 'wb') as f:
                pickle.dump(self.models[symbol], f)

            if symbol in self.scalers:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers[symbol], f)

        except Exception as e:
            logging.error(f"Error saving model for {symbol}: {e}")

    def predict_with_model(self, symbol: str, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using trained model
        """
        try:
            if symbol not in self.models:
                return np.zeros(len(features))

            # Scale features
            if symbol in self.scalers:
                features_scaled = self.scalers[symbol].transform(features)
            else:
                features_scaled = features.values

            # Make prediction
            predictions = self.models[symbol].predict(features_scaled)

            return predictions

        except Exception as e:
            logging.error(f"Error making prediction for {symbol}: {e}")
            return np.zeros(len(features))

    def _continuous_learning_loop(self):
        """Main continuous learning loop"""
        print("🚀 Starting continuous learning loop...")

        # Schedule daily model updates
        schedule.every().day.at("02:00").do(self._daily_model_update)
        schedule.every(4).hours.do(self._update_all_models)

        while True:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logging.error(f"Error in learning loop: {e}")
                time.sleep(300)  # Wait 5 minutes on error

    def _daily_model_update(self):
        """Daily comprehensive model update"""
        print("📅 Running daily model update...")

        # Update all major symbols
        symbols_to_update = []
        for data_type, config in self.data_sources.items():
            symbols_to_update.extend(config['symbols'][:5])  # Top 5 from each category

        # Use thread pool for parallel updates
        futures = []
        for symbol in symbols_to_update:
            future = self.executor.submit(self.train_online_model, symbol)
            futures.append(future)

        # Wait for completion
        for future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                if result['status'] == 'success':
                    print(f"✅ Updated model for {result['symbol']}")
                else:
                    print(f"❌ Failed to update {result.get('symbol', 'unknown')}: {result.get('reason', 'unknown')}")
            except Exception as e:
                print(f"⚠️  Error in model update: {e}")

        print("✅ Daily model update complete")

    def _update_all_models(self):
        """Update all models with latest data"""
        if self.is_learning:
            return  # Skip if already running

        self.is_learning = True
        try:
            print("🔄 Updating all models with latest data...")

            # Quick update for all symbols
            for data_type, config in self.data_sources.items():
                for symbol in config['symbols'][:3]:  # Top 3 from each category
                    try:
                        result = self.train_online_model(symbol, data_type)
                        if result['status'] == 'success':
                            print(f"✅ Quick update for {symbol}")
                    except Exception as e:
                        logging.error(f"Error updating {symbol}: {e}")

        finally:
            self.is_learning = False

    def get_model_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            'overall_stats': {},
            'symbol_performance': {},
            'improvement_trends': {},
            'best_performers': [],
            'worst_performers': []
        }

        # Overall statistics
        all_r2_scores = []
        all_directional_accuracies = []

        for symbol, history in self.performance_history.items():
            if history:
                latest = history[-1]
                all_r2_scores.append(latest['r2'])
                all_directional_accuracies.append(latest['directional_accuracy'])

                report['symbol_performance'][symbol] = {
                    'latest_r2': latest['r2'],
                    'latest_directional_accuracy': latest['directional_accuracy'],
                    'improvement_trend': self._calculate_improvement_trend(history),
                    'n_updates': len(history)
                }

        if all_r2_scores:
            report['overall_stats'] = {
                'avg_r2': np.mean(all_r2_scores),
                'avg_directional_accuracy': np.mean(all_directional_accuracies),
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

    def export_models_for_production(self, export_dir: str = 'production_models'):
        """Export models for production use"""
        os.makedirs(export_dir, exist_ok=True)

        exported_models = {}

        for symbol, model in self.models.items():
            try:
                model_path = os.path.join(export_dir, f"{symbol}_model.pkl")
                scaler_path = os.path.join(export_dir, f"{symbol}_scaler.pkl")

                # Save model
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

                # Save scaler if available
                if symbol in self.scalers:
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(self.scalers[symbol], f)

                # Save performance history
                perf_path = os.path.join(export_dir, f"{symbol}_performance.json")
                if symbol in self.performance_history:
                    with open(perf_path, 'w') as f:
                        # Convert datetime objects to strings
                        perf_data = []
                        for entry in self.performance_history[symbol]:
                            perf_entry = entry.copy()
                            perf_entry['timestamp'] = entry['timestamp'].isoformat()
                            perf_data.append(perf_entry)

                        json.dump(perf_data, f, indent=2)

                exported_models[symbol] = {
                    'model_path': model_path,
                    'scaler_path': scaler_path if symbol in self.scalers else None,
                    'performance_path': perf_path,
                    'latest_performance': self.performance_history.get(symbol, [{}])[-1]
                }

                print(f"✅ Exported {symbol} model to production")

            except Exception as e:
                logging.error(f"Error exporting {symbol}: {e}")

        # Save export summary
        summary_path = os.path.join(export_dir, 'export_summary.json')
        with open(summary_path, 'w') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_models': len(exported_models),
                'models': list(exported_models.keys())
            }, f, indent=2)

        print(f"🎉 Exported {len(exported_models)} models to {export_dir}")
        return exported_models


def main():
    """
    Main function for continuous learning system
    """
    print("🚀 FSOT 2.5 Continuous Learning Framework")
    print("=" * 50)

    # Initialize continuous learning system
    clf = ContinuousLearningFramework()

    print(f"📊 Initialized with {len(clf.models)} existing models")
    print(f"🎯 Tracking {sum(len(config['symbols']) for config in clf.data_sources.values())} symbols")

    # Initial training/update for key symbols
    print("\n🔬 Performing initial model training...")

    key_symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY', 'BTC', 'ETH']

    for symbol in key_symbols:
        print(f"📈 Training {symbol}...")
        result = clf.train_online_model(symbol)
        if result['status'] == 'success':
            print(f"✅ R²: {result['r2']:.4f}")
        else:
            print(f"❌ Failed: {result.get('reason', 'unknown')}")

    # Generate performance report
    print("\n📊 Generating performance report...")
    report = clf.get_model_performance_report()

    if report['overall_stats']:
        stats = report['overall_stats']
        print("\n🎯 Overall Performance:")
        print(f"  Average R²: {stats['avg_r2']:.4f}")
        print(f"  Average Directional Accuracy: {stats['avg_directional_accuracy']:.1%}")
        print(f"  Total Models: {stats['total_models']}")
        print(f"  Symbols Tracked: {stats['total_symbols_tracked']}")

    # Show top performers
    if report['best_performers']:
        print("\n🏆 Top Performers:")
        for i, (symbol, perf) in enumerate(report['best_performers'][:3], 1):
            print(f"  {i}. {symbol}: R²={perf['latest_r2']:.4f}")
    # Export models for production
    print("\n💾 Exporting models for production...")
    clf.export_models_for_production()

    print("\n✅ Continuous learning system initialized!")
    print("🔄 Models will continue to learn and improve automatically")
    print("📊 Check fsot_continuous_learning.log for detailed logs")

    # Keep the system running
    try:
        while True:
            time.sleep(60)  # Check every minute for scheduled tasks
    except KeyboardInterrupt:
        print("\n🛑 Shutting down continuous learning system...")
        clf.executor.shutdown(wait=True)


if __name__ == "__main__":
    main()
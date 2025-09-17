"""
FSOT 2.5 Backtesting and Benchmarking Framework
Comprehensive testing against historical data and competitor comparison
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import our FSOT system
from fsot_finance_predictor import (
    load_stock_data, predict_excess_returns, generate_trading_strategy,
    evaluate_strategy, add_enhanced_technical_indicators, apply_trailing_stops,
    detect_market_regime
)

class BacktestingFramework:
    """
    Comprehensive backtesting framework for FSOT 2.5 validation
    """

    def __init__(self, initial_capital=100000, transaction_costs=0.001):
        self.initial_capital = initial_capital
        self.transaction_costs = transaction_costs
        self.results = {}

    def run_backtest(self, symbol, start_date='2020-01-01', end_date='2024-01-01',
                    strategy='fsot', benchmark_strategies=None):
        """
        Run comprehensive backtest for a given symbol
        """
        print(f"🔬 Running backtest for {symbol} ({start_date} to {end_date})")
        print(f"Strategy: {strategy}")

        # Load historical data
        df = load_stock_data([symbol], start_date, end_date)
        if df.empty:
            print(f"❌ No data available for {symbol}")
            return None

        # Ensure we have datetime index (data should already be properly formatted)
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"⚠️  Unexpected index type: {type(df.index)}, attempting conversion")
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                print(f"⚠️  Failed to convert index to datetime: {e}")
                # Create a proper datetime index
                df = df.reset_index(drop=True)
                df.index = pd.date_range(start=start_date, periods=len(df), freq='D')

        # Split data for walk-forward analysis (use 80% for training, 20% for testing)
        split_idx = int(len(df) * 0.8)
        train_data = df.iloc[:split_idx].copy()
        test_data = df.iloc[split_idx:].copy()

        # Format dates for display
        try:
            train_start = train_data.index[0].strftime('%Y-%m-%d') if hasattr(train_data.index[0], 'strftime') else str(train_data.index[0])
            train_end = train_data.index[-1].strftime('%Y-%m-%d') if hasattr(train_data.index[-1], 'strftime') else str(train_data.index[-1])
            test_start = test_data.index[0].strftime('%Y-%m-%d') if hasattr(test_data.index[0], 'strftime') else str(test_data.index[0])
            test_end = test_data.index[-1].strftime('%Y-%m-%d') if hasattr(test_data.index[-1], 'strftime') else str(test_data.index[-1])
        except:
            train_start = str(train_data.index[0])
            train_end = str(train_data.index[-1])
            test_start = str(test_data.index[0])
            test_end = str(test_data.index[-1])

        print(f"📊 Training data: {len(train_data)} days ({train_start} to {train_end})")
        print(f"📊 Testing data: {len(test_data)} days ({test_start} to {test_end})")

        # Run FSOT strategy
        fsot_results = self._run_fsot_strategy(test_data.copy())

        # Run benchmark strategies
        benchmark_results = {}
        if benchmark_strategies:
            for bench_strategy in benchmark_strategies:
                benchmark_results[bench_strategy] = self._run_benchmark_strategy(
                    test_data.copy(), bench_strategy
                )

        # Calculate comprehensive metrics
        results = {
            'symbol': symbol,
            'fsot_results': fsot_results,
            'benchmark_results': benchmark_results,
            'market_data': test_data,
            'train_period': (train_start, train_end),
            'test_period': (test_start, test_end)
        }

        self.results[symbol] = results
        return results

    def _run_fsot_strategy(self, df):
        """Run FSOT 2.5 strategy with enhanced features and market regime adaptation"""
        try:
            print(f"  📊 Data shape: {df.shape}, columns: {df.columns.tolist()[:5]}...")

            # Step 1: Detect market regime for adaptive parameters
            df = detect_market_regime(df)
            print(f"  ✅ Detected market regimes, data shape: {df.shape}")

            # Step 2: Generate predictions first
            predictions = predict_excess_returns(df, use_ml=True)
            print(f"  ✅ Generated {len(predictions)} predictions")

            # Step 3: Generate trading strategy with regime-adaptive confidence scoring
            df = self._generate_adaptive_strategy(df)
            print(f"  ✅ Generated adaptive trading strategy, data shape: {df.shape}")

            # Step 4: Apply regime-adaptive trailing stops
            df = self._apply_adaptive_trailing_stops(df)
            print(f"  ✅ Applied adaptive trailing stops, data shape: {df.shape}")

            # Use trailing stop returns for final calculation
            if 'trailing_strategy_return' in df.columns:
                df['strategy_return'] = df['trailing_strategy_return']
                df['cum_strategy'] = (1 + df['strategy_return']).cumprod()

            # Check if required columns exist
            required_cols = ['strategy_return', 'cum_strategy', 'cum_market']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"  ❌ Missing columns: {missing_cols}")
                return None

            # Evaluate performance
            metrics = evaluate_strategy(df)
            print(f"  ✅ Evaluated strategy: {type(metrics)}")

            if isinstance(metrics, dict) and 'error' in metrics:
                print(f"  ❌ Strategy evaluation error: {metrics['error']}")
                return None

            # Calculate portfolio value over time
            portfolio_value = self._calculate_portfolio_value(df)

            return {
                'metrics': metrics,
                'predictions': predictions,
                'portfolio_value': portfolio_value,
                'trades': df
            }
        except Exception as e:
            print(f"⚠️  FSOT strategy error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_adaptive_strategy(self, df):
        """Generate trading strategy with regime-adaptive parameters"""
        if 'predicted_excess_return' not in df.columns:
            print(f"⚠️  Prediction column 'predicted_excess_return' not found, using simple strategy")
            df['position'] = 0
            df['confidence'] = 0.5
            df['strategy_return'] = 0
            df['cum_strategy'] = 1.0
            df['cum_market'] = (1 + df['excess_return']).cumprod()
            return df

        # Use regime-specific confidence thresholds
        df['confidence_threshold'] = df['regime_confidence_threshold']

        # Calculate prediction confidence based on multiple factors
        df['pred_magnitude'] = df['predicted_excess_return'].abs()
        df['pred_volatility'] = df['predicted_excess_return'].rolling(20).std().fillna(df['predicted_excess_return'].std())

        # Confidence score combines magnitude, consistency, and recent accuracy
        df['confidence'] = (
            0.4 * (df['pred_magnitude'] / df['pred_magnitude'].max()) +  # Magnitude weight
            0.3 * (1 / (1 + df['pred_volatility'])) +  # Consistency weight
            0.3 * df['predicted_excess_return'].rolling(10).corr(df['excess_return']).fillna(0.5).clip(0, 1)  # Recent accuracy
        )

        # Enhanced position sizing with confidence weighting and regime adaptation
        df['raw_position'] = np.where(df['predicted_excess_return'] > 0.002, 2,  # Strong bullish
                                      np.where(df['predicted_excess_return'] > 0, 1,  # Moderate bullish
                                               np.where(df['predicted_excess_return'] > -0.002, 0,  # Neutral
                                                        np.where(df['predicted_excess_return'] > -0.005, -1, -2))))  # Bearish positions

        # Apply confidence filter with regime-specific thresholds
        df['position'] = np.where(df['confidence'] >= df['confidence_threshold'],
                                  df['raw_position'] * df['regime_position_size'],  # Scale by regime position size
                                  0)  # No position if low confidence

        # Cap position sizes to prevent excessive risk
        df['position'] = df['position'].clip(-2, 2)

        # Calculate strategy returns with position sizing
        df['strategy_return'] = df['position'] * df['excess_return'] * 0.5

        # Calculate cumulative returns
        df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
        df['cum_market'] = (1 + df['excess_return']).cumprod()

        # Add performance tracking columns
        df['trade_active'] = (df['position'] != 0).astype(int)
        df['confidence_filtered'] = (df['confidence'] >= df['confidence_threshold']).astype(int)

        # Calculate rolling win rate for adaptive adjustments
        df['rolling_win'] = (df['strategy_return'] > 0).rolling(20).mean().fillna(0.5)

        print(f"🎯 Adaptive strategy generated:")
        print(f"   • Average confidence: {df['confidence'].mean():.3f}")
        print(f"   • Trades taken: {df['trade_active'].sum()} / {len(df)} ({df['trade_active'].mean()*100:.1f}%)")
        print(f"   • Regime distribution: {df['regime'].value_counts().to_dict()}")

        return df

    def _apply_adaptive_trailing_stops(self, df):
        """Apply trailing stops with regime-adaptive parameters"""
        if 'strategy_return' not in df.columns:
            return df

        df = df.copy()

        # Use regime-specific trailing stop percentages
        df['trailing_pct'] = df['regime_trailing_pct']

        # Initialize trailing stop tracking
        df['peak_value'] = 1.0
        df['trailing_stop_level'] = 1.0
        df['trailing_strategy_return'] = 0.0
        df['stop_triggered'] = False

        current_peak = 1.0
        position_active = False

        for i in range(len(df)):
            current_return = df.iloc[i]['strategy_return']

            # Update cumulative return for trailing stop calculation
            if i == 0:
                cumulative_return = 1 + current_return
            else:
                cumulative_return = df.iloc[i-1]['cum_strategy'] * (1 + current_return)

            # Update peak value when we have a position
            if df.iloc[i]['position'] != 0:
                if not position_active:
                    # New position opened
                    position_active = True
                    current_peak = cumulative_return
                else:
                    # Update peak if we're in profit
                    current_peak = max(current_peak, cumulative_return)

                # Calculate trailing stop level
                trailing_stop = current_peak * (1 - df.iloc[i]['trailing_pct'])

                # Check if stop is triggered
                if cumulative_return <= trailing_stop:
                    # Stop triggered - close position
                    df.iloc[i, df.columns.get_loc('trailing_strategy_return')] = trailing_stop - df.iloc[i-1]['cum_strategy'] if i > 0 else 0
                    df.iloc[i, df.columns.get_loc('stop_triggered')] = True
                    position_active = False
                    current_peak = cumulative_return
                else:
                    # Normal return
                    df.iloc[i, df.columns.get_loc('trailing_strategy_return')] = current_return
            else:
                # No position
                df.iloc[i, df.columns.get_loc('trailing_strategy_return')] = 0
                position_active = False
                current_peak = cumulative_return

            # Update tracking columns
            df.iloc[i, df.columns.get_loc('peak_value')] = current_peak
            df.iloc[i, df.columns.get_loc('trailing_stop_level')] = current_peak * (1 - df.iloc[i]['trailing_pct'])

        # Recalculate cumulative returns with trailing stops
        df['cum_trailing'] = (1 + df['trailing_strategy_return']).cumprod()

        # Count stop triggers
        stops_triggered = df['stop_triggered'].sum()
        if stops_triggered > 0:
            print(f"🛑 Trailing stops triggered {stops_triggered} times ({stops_triggered/len(df)*100:.1f}%)")

        return df

    def _run_benchmark_strategy(self, df, strategy_name):
        """Run benchmark strategies for comparison"""
        try:
            if strategy_name == 'buy_and_hold':
                return self._buy_and_hold_strategy(df)
            elif strategy_name == 'moving_average':
                return self._moving_average_strategy(df)
            elif strategy_name == 'rsi_strategy':
                return self._rsi_strategy(df)
            elif strategy_name == 'macd_strategy':
                return self._macd_strategy(df)
            else:
                print(f"⚠️  Unknown strategy: {strategy_name}")
                return None
        except Exception as e:
            print(f"⚠️  Benchmark strategy {strategy_name} error: {e}")
            return None

    def _buy_and_hold_strategy(self, df):
        """Simple buy and hold strategy"""
        df = df.copy()
        df['position'] = 1  # Always long
        df['strategy_return'] = df['excess_return']
        df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
        df['cum_market'] = (1 + df['excess_return']).cumprod()

        # Calculate metrics manually for benchmark strategies
        metrics = self._calculate_benchmark_metrics(df)
        portfolio_value = self._calculate_portfolio_value(df)

        return {
            'metrics': metrics,
            'portfolio_value': portfolio_value,
            'trades': df
        }

    def _moving_average_strategy(self, df):
        """Moving average crossover strategy"""
        df = df.copy()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()

        # Generate signals
        df['signal'] = 0
        df.loc[df['sma_20'] > df['sma_50'], 'signal'] = 1
        df.loc[df['sma_20'] < df['sma_50'], 'signal'] = -1

        # Apply transaction costs
        df['position'] = df['signal'].shift(1).fillna(0)
        df['strategy_return'] = df['position'] * df['excess_return'] - \
                               abs(df['position'].diff().fillna(0)) * self.transaction_costs

        df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
        df['cum_market'] = (1 + df['excess_return']).cumprod()

        metrics = self._calculate_benchmark_metrics(df)
        portfolio_value = self._calculate_portfolio_value(df)

        return {
            'metrics': metrics,
            'portfolio_value': portfolio_value,
            'trades': df
        }

    def _rsi_strategy(self, df):
        """RSI-based mean reversion strategy"""
        df = df.copy()
        # Calculate RSI if not already present
        if 'rsi' not in df.columns:
            df = add_enhanced_technical_indicators(df)

        # Generate signals
        df['signal'] = 0
        df.loc[df['rsi'] < 30, 'signal'] = 1  # Oversold - buy
        df.loc[df['rsi'] > 70, 'signal'] = -1  # Overbought - sell

        df['position'] = df['signal'].shift(1).fillna(0)
        df['strategy_return'] = df['position'] * df['excess_return'] - \
                               abs(df['position'].diff().fillna(0)) * self.transaction_costs

        df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
        df['cum_market'] = (1 + df['excess_return']).cumprod()

        metrics = self._calculate_benchmark_metrics(df)
        portfolio_value = self._calculate_portfolio_value(df)

        return {
            'metrics': metrics,
            'portfolio_value': portfolio_value,
            'trades': df
        }

    def _macd_strategy(self, df):
        """MACD strategy"""
        df = df.copy()
        # Calculate MACD if not already present
        if 'macd' not in df.columns:
            df = add_enhanced_technical_indicators(df)

        # Generate signals
        df['signal'] = 0
        df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
        df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1

        df['position'] = df['signal'].shift(1).fillna(0)
        df['strategy_return'] = df['position'] * df['excess_return'] - \
                               abs(df['position'].diff().fillna(0)) * self.transaction_costs

        df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
        df['cum_market'] = (1 + df['excess_return']).cumprod()

        metrics = self._calculate_benchmark_metrics(df)
        portfolio_value = self._calculate_portfolio_value(df)

        return {
            'metrics': metrics,
            'portfolio_value': portfolio_value,
            'trades': df
        }

    def _calculate_benchmark_metrics(self, df):
        """Calculate basic metrics for benchmark strategies"""
        strat_rets = df['strategy_return'].dropna()

        if len(strat_rets) == 0:
            return {'error': 'No strategy returns available'}

        # Basic metrics
        cum_ret = df['cum_strategy'].iloc[-1] - 1
        ann_ret = (1 + cum_ret) ** (252 / len(strat_rets)) - 1
        vol = strat_rets.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0

        # Maximum Drawdown
        cum_returns = (1 + strat_rets).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Win rate
        wins = (strat_rets > 0).sum()
        win_rate = wins / len(strat_rets) if len(strat_rets) > 0 else 0

        return {
            'cumulative_excess_return': cum_ret,
            'annualized_return': ann_ret,
            'annualized_sharpe': sharpe,
            'annualized_vol': vol,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'var_95': np.percentile(strat_rets, 5),
            'var_99': np.percentile(strat_rets, 1),
            'calmar_ratio': ann_ret / abs(max_drawdown) if max_drawdown < 0 else 0,
            'sortino_ratio': ann_ret / (strat_rets[strat_rets < 0].std() * np.sqrt(252)) if len(strat_rets[strat_rets < 0]) > 0 else 0,
            'information_ratio': 0,  # Not applicable for benchmarks
            'alpha': 0,  # Not applicable for benchmarks
            'beta': 1,  # Market beta for buy and hold
            'avg_rolling_sharpe': sharpe,
            'correlation_preds_actual': 0  # Not applicable for benchmarks
        }

    def _calculate_portfolio_value(self, df):
        """Calculate portfolio value over time"""
        portfolio_value = [self.initial_capital]

        for i in range(1, len(df)):
            prev_value = portfolio_value[-1]
            daily_return = df['strategy_return'].iloc[i]
            new_value = prev_value * (1 + daily_return)
            portfolio_value.append(new_value)

        return portfolio_value

    def calculate_accuracy_metrics(self, df):
        """Calculate comprehensive accuracy metrics"""
        if 'predicted_excess_return' not in df.columns or 'excess_return' not in df.columns:
            return None

        valid_data = df.dropna(subset=['predicted_excess_return', 'excess_return'])

        if len(valid_data) < 10:
            return None

        actual = valid_data['excess_return']
        predicted = valid_data['predicted_excess_return']

        # Basic error metrics
        mae = np.mean(np.abs(actual - predicted))
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100

        # Directional accuracy
        actual_direction = np.sign(actual)
        predicted_direction = np.sign(predicted)
        directional_accuracy = np.mean(actual_direction == predicted_direction)

        # Correlation
        correlation = actual.corr(predicted)

        # R-squared
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Information coefficient (IC)
        ic = correlation

        # Profitability metrics
        profitable_predictions = (predicted_direction == actual_direction) & (actual_direction == 1)
        unprofitable_predictions = (predicted_direction == actual_direction) & (actual_direction == -1)

        hit_rate = profitable_predictions.sum() / (profitable_predictions.sum() + unprofitable_predictions.sum()) if (profitable_predictions.sum() + unprofitable_predictions.sum()) > 0 else 0

        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            'r_squared': r_squared,
            'information_coefficient': ic,
            'hit_rate': hit_rate,
            'n_predictions': len(valid_data)
        }

    def run_statistical_tests(self, results):
        """Run statistical significance tests"""
        if not results or 'fsot_results' not in results:
            return None

        fsot_returns = results['fsot_results']['trades']['strategy_return'].dropna()
        market_returns = results['fsot_results']['trades']['excess_return'].dropna()

        # Sharpe ratio test
        fsot_sharpe = results['fsot_results']['metrics']['annualized_sharpe']
        market_sharpe = results['benchmark_results'].get('buy_and_hold', {}).get('metrics', {}).get('annualized_sharpe', 0)

        # T-test for returns
        t_stat, p_value = stats.ttest_ind(fsot_returns, market_returns, equal_var=False)

        # Maximum drawdown comparison
        fsot_max_dd = results['fsot_results']['metrics']['max_drawdown']
        market_max_dd = results['benchmark_results'].get('buy_and_hold', {}).get('metrics', {}).get('max_drawdown', 0)

        return {
            'sharpe_ratio_test': {
                'fsot_sharpe': fsot_sharpe,
                'market_sharpe': market_sharpe,
                'sharpe_difference': fsot_sharpe - market_sharpe
            },
            'returns_t_test': {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            },
            'drawdown_comparison': {
                'fsot_max_dd': fsot_max_dd,
                'market_max_dd': market_max_dd,
                'dd_improvement': market_max_dd - fsot_max_dd
            }
        }

    def generate_comprehensive_report(self, symbol):
        """Generate comprehensive performance report"""
        if symbol not in self.results:
            print(f"❌ No results available for {symbol}")
            return None

        results = self.results[symbol]

        print(f"\n" + "="*80)
        print(f"📊 COMPREHENSIVE BACKTEST REPORT - {symbol}")
        print("="*80)

        # FSOT Performance
        fsot_metrics = results['fsot_results']['metrics']
        print(f"\n🎯 FSOT 2.5 PERFORMANCE:")
        print(f"  Cumulative Return: {fsot_metrics['cumulative_excess_return']*100:.2f}%")
        print(f"  Annualized Return: {fsot_metrics['annualized_return']*100:.2f}%")
        print(f"  Sharpe Ratio: {fsot_metrics['annualized_sharpe']:.2f}")
        print(f"  Max Drawdown: {fsot_metrics['max_drawdown']*100:.2f}%")
        print(f"  Win Rate: {fsot_metrics['win_rate']*100:.1f}%")

        # Accuracy Metrics
        accuracy = self.calculate_accuracy_metrics(results['fsot_results']['trades'])
        if accuracy:
            print(f"\n🎯 PREDICTION ACCURACY:")
            print(f"  Directional Accuracy: {accuracy['directional_accuracy']*100:.1f}%")
            print(f"  Correlation: {accuracy['correlation']:.3f}")
            print(f"  RMSE: {accuracy['rmse']:.6f}")
            print(f"  MAE: {accuracy['mae']:.6f}")

        # Benchmark Comparisons
        print(f"\n🏆 BENCHMARK COMPARISONS:")
        for bench_name, bench_result in results['benchmark_results'].items():
            if bench_result:
                bench_metrics = bench_result['metrics']
                fsot_return = fsot_metrics['cumulative_excess_return']
                bench_return = bench_metrics['cumulative_excess_return']

                print(f"  {bench_name.upper()}:")
                print(f"    Return: {bench_return*100:.2f}% (FSOT: {fsot_return*100:.2f}%)")
                print(f"    Sharpe: {bench_metrics['annualized_sharpe']:.2f} (FSOT: {fsot_metrics['annualized_sharpe']:.2f})")
                print(f"    Outperformance: {(fsot_return - bench_return)*100:.2f}%")

        # Statistical Tests
        stat_tests = self.run_statistical_tests(results)
        if stat_tests:
            print(f"\n📈 STATISTICAL SIGNIFICANCE:")
            print(f"  Returns T-Test P-Value: {stat_tests['returns_t_test']['p_value']:.4f}")
            print(f"  Statistically Significant: {'Yes' if stat_tests['returns_t_test']['significant'] else 'No'}")

        print(f"\n" + "="*80)

        return {
            'fsot_metrics': fsot_metrics,
            'accuracy_metrics': accuracy,
            'benchmark_comparisons': results['benchmark_results'],
            'statistical_tests': stat_tests
        }

    def create_performance_visualization(self, symbol, save_path=None):
        """Create comprehensive performance visualization"""
        if symbol not in self.results:
            return None

        results = self.results[symbol]
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle(f'FSOT 2.5 Backtest Analysis - {symbol}', fontsize=16, fontweight='bold')

        # 1. Cumulative Returns Comparison
        ax = axes[0, 0]
        fsot_cum = results['fsot_results']['trades']['cum_strategy']
        market_cum = results['fsot_results']['trades']['cum_market']

        ax.plot(fsot_cum.index, fsot_cum.values, label='FSOT 2.5', linewidth=2, color='blue')
        ax.plot(market_cum.index, market_cum.values, label='Buy & Hold', linewidth=2, color='red', alpha=0.7)

        # Add benchmark strategies
        colors = ['green', 'orange', 'purple', 'brown']
        for i, (bench_name, bench_result) in enumerate(results['benchmark_results'].items()):
            if bench_result and i < len(colors):
                bench_cum = bench_result['trades']['cum_strategy']
                ax.plot(bench_cum.index, bench_cum.values,
                       label=bench_name.replace('_', ' ').title(),
                       linewidth=1.5, color=colors[i], alpha=0.8)

        ax.set_title('Cumulative Returns Comparison')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Prediction Accuracy Scatter
        ax = axes[0, 1]
        valid_data = results['fsot_results']['trades'].dropna(subset=['predicted_excess_return', 'excess_return'])
        if len(valid_data) > 0:
            ax.scatter(valid_data['excess_return'], valid_data['predicted_excess_return'],
                      alpha=0.6, color='purple', s=30)

            # Add trend line
            z = np.polyfit(valid_data['excess_return'], valid_data['predicted_excess_return'], 1)
            p = np.poly1d(z)
            ax.plot(valid_data['excess_return'], p(valid_data['excess_return']),
                   color='red', linewidth=2)

            ax.set_xlabel('Actual Returns')
            ax.set_ylabel('Predicted Returns')
            ax.set_title('Prediction Accuracy')
            ax.grid(True, alpha=0.3)

        # 3. Rolling Sharpe Ratio
        ax = axes[1, 0]
        if len(results['fsot_results']['trades']) >= 30:
            rolling_sharpe = results['fsot_results']['trades']['strategy_return'].rolling(30).apply(
                lambda x: x.mean() / x.std() * np.sqrt(252), raw=False)
            ax.plot(rolling_sharpe.index, rolling_sharpe.values, color='blue', linewidth=2)
            ax.axhline(y=results['fsot_results']['metrics']['annualized_sharpe'],
                      color='red', linestyle='--', label='Average Sharpe')
            ax.set_title('Rolling Sharpe Ratio (30-day)')
            ax.set_ylabel('Sharpe Ratio')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Drawdown Analysis
        ax = axes[1, 1]
        fsot_returns = results['fsot_results']['trades']['strategy_return'].fillna(0)
        cum_returns = (1 + fsot_returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max

        ax.fill_between(results['fsot_results']['trades'].index, drawdowns, 0,
                       color='red', alpha=0.3)
        ax.plot(results['fsot_results']['trades'].index, drawdowns,
               color='darkred', linewidth=1)
        ax.set_title('Strategy Drawdown')
        ax.set_ylabel('Drawdown %')
        ax.grid(True, alpha=0.3)

        # 5. Monthly Returns Heatmap
        ax = axes[2, 0]
        monthly_returns = results['fsot_results']['trades']['strategy_return'].groupby(
            pd.Grouper(freq='M')).apply(lambda x: (1 + x).prod() - 1)

        if len(monthly_returns) > 0:
            monthly_df = monthly_returns.reset_index()
            monthly_df['Year'] = monthly_df['Date'].dt.year
            monthly_df['Month'] = monthly_df['Date'].dt.month

            pivot_table = monthly_df.pivot(index='Year', columns='Month', values='strategy_return')
            sns.heatmap(pivot_table, annot=True, fmt='.1%', cmap='RdYlGn', center=0, ax=ax)
            ax.set_title('Monthly Returns Heatmap')

        # 6. Performance Summary
        ax = axes[2, 1]
        ax.axis('off')

        metrics = results['fsot_results']['metrics']
        summary_text = f"""Performance Summary:

Total Return: {metrics['cumulative_excess_return']*100:.1f}%
Annual Return: {metrics['annualized_return']*100:.1f}%
Sharpe Ratio: {metrics['annualized_sharpe']:.2f}
Max Drawdown: {metrics['max_drawdown']*100:.1f}%
Win Rate: {metrics['win_rate']*100:.1f}%

Risk Metrics:
VaR 95%: {metrics['var_95']*100:.1f}%
VaR 99%: {metrics['var_99']*100:.1f}%
Calmar Ratio: {metrics['calmar_ratio']:.2f}
Sortino Ratio: {metrics['sortino_ratio']:.2f}"""

        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Backtest visualization saved to {save_path}")

        plt.show()

def run_comprehensive_backtest():
    """Run comprehensive backtest across multiple symbols and time periods"""
    print("🚀 Starting Comprehensive FSOT 2.5 Backtest")
    print("="*60)

    # Initialize backtesting framework
    backtester = BacktestingFramework(initial_capital=100000, transaction_costs=0.001)

    # Test symbols across different market caps and sectors
    test_symbols = [
        'AAPL',    # Large cap tech
        'MSFT',    # Large cap tech
        'GOOGL',   # Large cap tech
        'JPM',     # Financial
        'JNJ',     # Healthcare
        'PG',      # Consumer goods
        'SPY'      # S&P 500 ETF
    ]

    benchmark_strategies = ['buy_and_hold', 'moving_average', 'rsi_strategy', 'macd_strategy']

    all_results = {}

    for symbol in test_symbols:
        try:
            print(f"\n🔬 Testing {symbol}...")

            # Run backtest
            results = backtester.run_backtest(
                symbol=symbol,
                start_date='2020-01-01',
                end_date='2024-01-01',
                benchmark_strategies=benchmark_strategies
            )

            if results:
                # Generate report
                report = backtester.generate_comprehensive_report(symbol)

                # Create visualization
                backtester.create_performance_visualization(symbol, f'fsot_backtest_{symbol}.png')

                all_results[symbol] = {
                    'results': results,
                    'report': report
                }

        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            continue

    # Generate aggregate analysis
    print(f"\n" + "="*80)
    print("📊 AGGREGATE BACKTEST ANALYSIS")
    print("="*80)

    if all_results:
        # Calculate aggregate metrics
        fsot_returns = []
        benchmark_returns = {bench: [] for bench in benchmark_strategies}

        for symbol, data in all_results.items():
            if data['report']:
                fsot_returns.append(data['report']['fsot_metrics']['cumulative_excess_return'])

                for bench in benchmark_strategies:
                    if bench in data['report']['benchmark_comparisons']:
                        bench_return = data['report']['benchmark_comparisons'][bench]['metrics']['cumulative_excess_return']
                        benchmark_returns[bench].append(bench_return)

        if fsot_returns:
            avg_fsot_return = np.mean(fsot_returns)
            print(f"\n🎯 AVERAGE FSOT PERFORMANCE:")
            print(f"  Average Return: {avg_fsot_return*100:.2f}%")
            print(f"  Return Std Dev: {np.std(fsot_returns)*100:.2f}%")

            print(f"\n🏆 BENCHMARK COMPARISONS:")
            for bench, returns in benchmark_returns.items():
                if returns:
                    avg_bench_return = np.mean(returns)
                    outperformance = avg_fsot_return - avg_bench_return
                    print(f"  {bench.replace('_', ' ').title()}: {avg_bench_return*100:.2f}% "
                          f"(Outperformance: {outperformance*100:.2f}%)")

    print(f"\n✅ Backtest completed for {len(all_results)} symbols")
    print("="*80)

    return all_results

if __name__ == "__main__":
    # Run comprehensive backtest
    results = run_comprehensive_backtest()

    print("\n🎉 FSOT 2.5 Backtesting Framework Complete!")
    print("📊 Results saved as PNG files for each symbol")
    print("🔬 Use the BacktestingFramework class for custom analysis")
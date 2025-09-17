"""
FSOT 2.5 Competitor Analysis Module
Compare FSOT performance against popular trading strategies and algorithms
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class CompetitorAnalysis:
    """
    Comprehensive competitor analysis for FSOT 2.5
    """

    def __init__(self):
        self.competitors = {}
        self.benchmark_data = {}

    def add_competitor(self, name, strategy_func, description=""):
        """Add a competitor strategy for comparison"""
        self.competitors[name] = {
            'function': strategy_func,
            'description': description,
            'results': {}
        }

    def initialize_standard_competitors(self):
        """Initialize standard competitor strategies"""

        # 1. Buy and Hold
        def buy_and_hold(df):
            df['position'] = 1
            df['strategy_return'] = df['excess_return']
            df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
            return df

        self.add_competitor('buy_and_hold', buy_and_hold,
                           "Simple buy and hold strategy - market benchmark")

        # 2. Moving Average Crossover
        def moving_average_crossover(df, short_window=20, long_window=50):
            df['sma_short'] = df['close'].rolling(short_window).mean()
            df['sma_long'] = df['close'].rolling(long_window).mean()

            df['signal'] = 0
            df.loc[df['sma_short'] > df['sma_long'], 'signal'] = 1
            df.loc[df['sma_short'] < df['sma_long'], 'signal'] = -1

            df['position'] = df['signal'].shift(1).fillna(0)
            df['strategy_return'] = df['position'] * df['excess_return']
            df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
            return df

        self.add_competitor('ma_crossover_20_50', moving_average_crossover,
                           "20/50 day moving average crossover strategy")

        # 3. RSI Strategy
        def rsi_strategy(df, oversold=30, overbought=70):
            # Calculate RSI if not present
            if 'rsi' not in df.columns:
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                df['rsi'] = 100 - (100 / (1 + rs))

            df['signal'] = 0
            df.loc[df['rsi'] < oversold, 'signal'] = 1
            df.loc[df['rsi'] > overbought, 'signal'] = -1

            df['position'] = df['signal'].shift(1).fillna(0)
            df['strategy_return'] = df['position'] * df['excess_return']
            df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
            return df

        self.add_competitor('rsi_30_70', rsi_strategy,
                           "RSI strategy (buy < 30, sell > 70)")

        # 4. MACD Strategy
        def macd_strategy(df):
            # Calculate MACD if not present
            if 'macd' not in df.columns:
                exp1 = df['close'].ewm(span=12, adjust=False).mean()
                exp2 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = exp1 - exp2
                df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

            df['signal'] = 0
            df.loc[df['macd'] > df['macd_signal'], 'signal'] = 1
            df.loc[df['macd'] < df['macd_signal'], 'signal'] = -1

            df['position'] = df['signal'].shift(1).fillna(0)
            df['strategy_return'] = df['position'] * df['excess_return']
            df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
            return df

        self.add_competitor('macd', macd_strategy,
                           "MACD crossover strategy")

        # 5. Bollinger Bands Strategy
        def bollinger_bands_strategy(df, window=20, std_dev=2):
            df['sma'] = df['close'].rolling(window).mean()
            df['std'] = df['close'].rolling(window).std()
            df['upper_band'] = df['sma'] + (df['std'] * std_dev)
            df['lower_band'] = df['sma'] - (df['std'] * std_dev)

            df['signal'] = 0
            df.loc[df['close'] < df['lower_band'], 'signal'] = 1  # Buy when price below lower band
            df.loc[df['close'] > df['upper_band'], 'signal'] = -1  # Sell when price above upper band

            df['position'] = df['signal'].shift(1).fillna(0)
            df['strategy_return'] = df['position'] * df['excess_return']
            df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
            return df

        self.add_competitor('bollinger_bands', bollinger_bands_strategy,
                           "Bollinger Bands mean reversion strategy")

        # 6. Momentum Strategy
        def momentum_strategy(df, lookback=20):
            df['momentum'] = df['close'] / df['close'].shift(lookback) - 1

            df['signal'] = 0
            df.loc[df['momentum'] > 0, 'signal'] = 1
            df.loc[df['momentum'] < 0, 'signal'] = -1

            df['position'] = df['signal'].shift(1).fillna(0)
            df['strategy_return'] = df['position'] * df['excess_return']
            df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
            return df

        self.add_competitor('momentum_20', momentum_strategy,
                           "20-day momentum strategy")

        # 7. Pairs Trading Strategy (simplified)
        def pairs_trading_strategy(df, symbol2_data=None):
            # This is a simplified version - in practice would need two correlated assets
            if symbol2_data is None:
                return buy_and_hold(df)  # Fallback to buy and hold

            # Calculate spread
            spread = df['close'] - symbol2_data['close']

            # Simple mean reversion on spread
            spread_mean = spread.rolling(20).mean()
            spread_std = spread.rolling(20).std()

            df['signal'] = 0
            df.loc[spread < (spread_mean - spread_std), 'signal'] = 1  # Buy when spread is low
            df.loc[spread > (spread_mean + spread_std), 'signal'] = -1  # Sell when spread is high

            df['position'] = df['signal'].shift(1).fillna(0)
            df['strategy_return'] = df['position'] * df['excess_return']
            df['cum_strategy'] = (1 + df['strategy_return']).cumprod()
            return df

        self.add_competitor('pairs_trading', pairs_trading_strategy,
                           "Pairs trading strategy (simplified)")

    def run_competitor_analysis(self, symbol, df, fsot_results=None):
        """
        Run comprehensive competitor analysis
        """
        print(f"🏆 Running competitor analysis for {symbol}")

        competitor_results = {}

        for name, competitor in self.competitors.items():
            try:
                print(f"  Running {name}...")
                competitor_df = df.copy()
                result_df = competitor['function'](competitor_df)

                # Calculate metrics
                metrics = self._calculate_strategy_metrics(result_df)

                competitor_results[name] = {
                    'data': result_df,
                    'metrics': metrics,
                    'description': competitor['description']
                }

            except Exception as e:
                print(f"  ❌ Error running {name}: {e}")
                continue

        # Compare with FSOT if provided
        if fsot_results:
            competitor_results['fsot_2_5'] = fsot_results

        return competitor_results

    def _calculate_strategy_metrics(self, df):
        """Calculate comprehensive strategy metrics"""
        strat_rets = df['strategy_return'].dropna()

        if len(strat_rets) == 0:
            return {'error': 'No strategy returns available'}

        # Basic metrics
        cum_ret = df['cum_strategy'].iloc[-1] - 1
        ann_ret = (1 + cum_ret) ** (252 / len(strat_rets)) - 1
        vol = strat_rets.std() * np.sqrt(252)
        sharpe = ann_ret / vol if vol > 0 else 0

        # Risk metrics
        var_95 = np.percentile(strat_rets, 5)
        var_99 = np.percentile(strat_rets, 1)

        cum_returns = (1 + strat_rets).cumprod()
        running_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Trading metrics
        wins = (strat_rets > 0).sum()
        losses = (strat_rets < 0).sum()
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        avg_win = strat_rets[strat_rets > 0].mean() if wins > 0 else 0
        avg_loss = abs(strat_rets[strat_rets < 0].mean()) if losses > 0 else 0
        profit_factor = (avg_win * wins) / (avg_loss * losses) if losses > 0 and avg_loss > 0 else float('inf')

        # Advanced ratios
        calmar = ann_ret / abs(max_drawdown) if max_drawdown < 0 else 0
        downside_rets = strat_rets[strat_rets < 0]
        sortino = ann_ret / (downside_rets.std() * np.sqrt(252)) if len(downside_rets) > 0 else 0

        return {
            'cumulative_return': cum_ret,
            'annualized_return': ann_ret,
            'annualized_volatility': vol,
            'sharpe_ratio': sharpe,
            'var_95': var_95,
            'var_99': var_99,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'calmar_ratio': calmar,
            'sortino_ratio': sortino,
            'total_trades': len(strat_rets),
            'avg_trade_return': strat_rets.mean()
        }

    def create_competitor_comparison_chart(self, competitor_results, save_path=None):
        """Create comprehensive competitor comparison visualization"""
        if not competitor_results:
            return

        # Prepare data for plotting
        strategies = []
        returns = []
        sharpes = []
        max_drawdowns = []
        win_rates = []

        for name, result in competitor_results.items():
            if 'metrics' in result:
                metrics = result['metrics']
                if 'error' not in metrics:
                    strategies.append(name.replace('_', ' ').title())
                    returns.append(metrics['cumulative_return'] * 100)
                    sharpes.append(metrics['sharpe_ratio'])
                    max_drawdowns.append(abs(metrics['max_drawdown']) * 100)
                    win_rates.append(metrics['win_rate'] * 100)

        if not strategies:
            print("❌ No valid competitor data to plot")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FSOT 2.5 vs Competitor Strategies', fontsize=16, fontweight='bold')

        # 1. Returns Comparison
        bars = axes[0, 0].bar(range(len(strategies)), returns, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Total Returns Comparison')
        axes[0, 0].set_ylabel('Total Return (%)')
        axes[0, 0].set_xticks(range(len(strategies)))
        axes[0, 0].set_xticklabels(strategies, rotation=45, ha='right')

        # Highlight FSOT if present
        if 'Fsot 2 5' in strategies:
            fsot_idx = strategies.index('Fsot 2 5')
            bars[fsot_idx].set_color('darkblue')

        # Add value labels
        for bar, value in zip(bars, returns):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 2. Sharpe Ratio Comparison
        bars = axes[0, 1].bar(range(len(strategies)), sharpes, color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('Sharpe Ratio Comparison')
        axes[0, 1].set_ylabel('Sharpe Ratio')
        axes[0, 1].set_xticks(range(len(strategies)))
        axes[0, 1].set_xticklabels(strategies, rotation=45, ha='right')

        if 'Fsot 2 5' in strategies:
            fsot_idx = strategies.index('Fsot 2 5')
            bars[fsot_idx].set_color('darkgreen')

        # 3. Maximum Drawdown Comparison
        bars = axes[1, 0].bar(range(len(strategies)), max_drawdowns, color='salmon', alpha=0.8)
        axes[1, 0].set_title('Maximum Drawdown Comparison')
        axes[1, 0].set_ylabel('Max Drawdown (%)')
        axes[1, 0].set_xticks(range(len(strategies)))
        axes[1, 0].set_xticklabels(strategies, rotation=45, ha='right')

        if 'Fsot 2 5' in strategies:
            fsot_idx = strategies.index('Fsot 2 5')
            bars[fsot_idx].set_color('darkred')

        # 4. Win Rate Comparison
        bars = axes[1, 1].bar(range(len(strategies)), win_rates, color='gold', alpha=0.8)
        axes[1, 1].set_title('Win Rate Comparison')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].set_xticks(range(len(strategies)))
        axes[1, 1].set_xticklabels(strategies, rotation=45, ha='right')

        if 'Fsot 2 5' in strategies:
            fsot_idx = strategies.index('Fsot 2 5')
            bars[fsot_idx].set_color('orange')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Competitor comparison saved to {save_path}")

        plt.show()

    def create_cumulative_returns_chart(self, competitor_results, save_path=None):
        """Create cumulative returns comparison chart"""
        if not competitor_results:
            return

        plt.figure(figsize=(16, 10))

        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

        for i, (name, result) in enumerate(competitor_results.items()):
            if 'data' in result and 'cum_strategy' in result['data'].columns:
                color = colors[i % len(colors)]
                label = name.replace('_', ' ').title()

                if name.lower() == 'fsot_2_5':
                    plt.plot(result['data'].index, result['data']['cum_strategy'],
                            linewidth=3, color='darkblue', label='FSOT 2.5', zorder=10)
                else:
                    plt.plot(result['data'].index, result['data']['cum_strategy'],
                            linewidth=2, color=color, alpha=0.8, label=label)

        plt.title('Cumulative Returns: FSOT 2.5 vs Competitors', fontsize=16, fontweight='bold')
        plt.ylabel('Cumulative Return (Multiple)')
        plt.xlabel('Date')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Cumulative returns chart saved to {save_path}")

        plt.show()

    def generate_competitor_report(self, competitor_results):
        """Generate comprehensive competitor analysis report"""
        if not competitor_results:
            return

        print(f"\n" + "="*100)
        print("🏆 COMPREHENSIVE COMPETITOR ANALYSIS REPORT")
        print("="*100)

        # Sort by Sharpe ratio for ranking
        sorted_competitors = sorted(
            [(name, result) for name, result in competitor_results.items()
             if 'metrics' in result and 'error' not in result['metrics']],
            key=lambda x: x[1]['metrics']['sharpe_ratio'],
            reverse=True
        )

        print(f"\n📊 STRATEGY RANKINGS (by Sharpe Ratio):")
        print("-" * 80)

        for rank, (name, result) in enumerate(sorted_competitors, 1):
            metrics = result['metrics']
            description = result.get('description', '')

            print(f"\n{rank}. {name.upper().replace('_', ' ')}")
            if description:
                print(f"   {description}")
            print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"   Total Return: {metrics['cumulative_return']*100:.1f}%")
            print(f"   Annual Return: {metrics['annualized_return']*100:.1f}%")
            print(f"   Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
            print(f"   Win Rate: {metrics['win_rate']*100:.1f}%")
            print(f"   Profit Factor: {metrics['profit_factor']:.2f}")

        # Find best performer
        if sorted_competitors:
            best_name, best_result = sorted_competitors[0]
            best_metrics = best_result['metrics']

            print(f"\n🏆 BEST PERFORMING STRATEGY: {best_name.upper().replace('_', ' ')}")
            print(f"   Sharpe Ratio: {best_metrics['sharpe_ratio']:.2f}")
            print(f"   Total Return: {best_metrics['cumulative_return']*100:.1f}%")

            # Check if FSOT is in top 3
            fsot_rank = None
            for rank, (name, _) in enumerate(sorted_competitors, 1):
                if name.lower() == 'fsot_2_5':
                    fsot_rank = rank
                    break

            if fsot_rank:
                print(f"   FSOT 2.5 Ranking: #{fsot_rank} out of {len(sorted_competitors)}")
                if fsot_rank == 1:
                    print("   🎉 FSOT 2.5 is the top performer!")
                elif fsot_rank <= 3:
                    print("   ✅ FSOT 2.5 is in the top 3!")
                else:
                    print("   📈 FSOT 2.5 shows competitive performance")

        print(f"\n" + "="*100)

    def run_statistical_significance_tests(self, competitor_results):
        """Run statistical tests to determine if FSOT outperforms competitors"""
        if 'fsot_2_5' not in competitor_results:
            return None

        fsot_returns = competitor_results['fsot_2_5']['data']['strategy_return'].dropna()

        results = {}

        for name, competitor in competitor_results.items():
            if name == 'fsot_2_5' or 'data' not in competitor:
                continue

            try:
                comp_returns = competitor['data']['strategy_return'].dropna()

                # Align the series
                common_index = fsot_returns.index.intersection(comp_returns.index)
                if len(common_index) < 30:  # Need minimum sample size
                    continue

                fsot_aligned = fsot_returns.loc[common_index]
                comp_aligned = comp_returns.loc[common_index]

                # T-test for difference in means
                t_stat, p_value = stats.ttest_ind(fsot_aligned, comp_aligned, equal_var=False)

                # Sharpe ratio comparison
                fsot_sharpe = competitor_results['fsot_2_5']['metrics']['sharpe_ratio']
                comp_sharpe = competitor['metrics']['sharpe_ratio']

                results[name] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'fsot_sharpe': fsot_sharpe,
                    'competitor_sharpe': comp_sharpe,
                    'sharpe_difference': fsot_sharpe - comp_sharpe,
                    'sample_size': len(common_index)
                }

            except Exception as e:
                print(f"⚠️  Error in statistical test for {name}: {e}")
                continue

        return results

def run_competitor_comparison_demo():
    """Run comprehensive competitor comparison demo"""
    print("🏆 FSOT 2.5 Competitor Analysis Demo")
    print("="*50)

    # Import FSOT system
    from fsot_finance_predictor import load_stock_data, predict_excess_returns, generate_trading_strategy

    # Initialize competitor analysis
    competitor_analysis = CompetitorAnalysis()
    competitor_analysis.initialize_standard_competitors()

    # Test symbol
    symbol = 'AAPL'
    print(f"Testing {symbol} against competitor strategies...")

    # Load data
    df = load_stock_data([symbol], '2020-01-01', '2024-01-01')
    if df.empty:
        print(f"❌ No data available for {symbol}")
        return

    # Run FSOT strategy
    print("Running FSOT 2.5 strategy...")
    fsot_predictions = predict_excess_returns(df, use_ml=True)
    fsot_df = generate_trading_strategy(df)

    fsot_results = {
        'data': fsot_df,
        'metrics': {},  # Will be calculated by competitor analysis
        'description': 'FSOT 2.5 Hybrid Model'
    }

    # Run competitor analysis
    competitor_results = competitor_analysis.run_competitor_analysis(
        symbol, df, fsot_results
    )

    # Calculate metrics for all strategies
    for name, result in competitor_results.items():
        if 'data' in result:
            metrics = competitor_analysis._calculate_strategy_metrics(result['data'])
            competitor_results[name]['metrics'] = metrics

    # Generate reports and visualizations
    competitor_analysis.generate_competitor_report(competitor_results)

    # Create comparison charts
    competitor_analysis.create_competitor_comparison_chart(
        competitor_results, 'fsot_competitor_comparison.png'
    )

    competitor_analysis.create_cumulative_returns_chart(
        competitor_results, 'fsot_cumulative_returns_comparison.png'
    )

    # Statistical significance tests
    stat_results = competitor_analysis.run_statistical_significance_tests(competitor_results)

    if stat_results:
        print(f"\n📈 STATISTICAL SIGNIFICANCE TESTS:")
        print("-" * 50)
        for name, stats in stat_results.items():
            print(f"{name.upper().replace('_', ' ')}:")
            print(f"  P-value: {stats['p_value']:.4f}")
            print(f"  Significant difference: {'Yes' if stats['significant'] else 'No'}")
            print(f"  Sharpe difference: {stats['sharpe_difference']:.2f}")
            print()

    print("✅ Competitor analysis complete!")
    print("📊 Charts saved as PNG files")

if __name__ == "__main__":
    run_competitor_comparison_demo()
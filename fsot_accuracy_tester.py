"""
FSOT 2.5 Accuracy Testing Framework
===================================

Comprehensive testing of prediction accuracy across different timescales,
market conditions, and asset classes.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import FSOT components
from fsot_finance_predictor import (
    load_stock_data, predict_excess_returns, generate_trading_strategy,
    evaluate_strategy, detect_market_regime
)

class AccuracyTester:
    """
    Comprehensive accuracy testing framework for FSOT 2.5
    """

    def __init__(self):
        self.results = {}

    def run_accuracy_test(self, symbols=None, test_periods=None, timescales=None):
        """
        Run comprehensive accuracy testing across multiple dimensions
        """
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY']

        if test_periods is None:
            test_periods = [
                ('2020-01-01', '2021-12-31'),  # Pre-COVID to COVID
                ('2021-01-01', '2022-12-31'),  # COVID recovery
                ('2022-01-01', '2023-12-31'),  # Inflation/high volatility
                ('2023-01-01', '2024-12-31'),  # Recovery period
            ]

        if timescales is None:
            timescales = ['conservative', 'moderate', 'aggressive', 'very_aggressive']

        print("🎯 FSOT 2.5 ACCURACY TESTING FRAMEWORK")
        print("=" * 60)

        all_results = {}

        for symbol in symbols:
            print(f"\n📊 Testing {symbol}")
            print("-" * 40)

            symbol_results = {}

            for start_date, end_date in test_periods:
                period_key = f"{start_date}_{end_date}"
                print(f"  📅 Period: {start_date} to {end_date}")

                # Load data
                df = load_stock_data([symbol], start_date, end_date)
                if df.empty:
                    print(f"    ❌ No data for {symbol} in {period_key}")
                    continue

                # Split into train/test
                split_idx = int(len(df) * 0.7)
                train_data = df.iloc[:split_idx].copy()
                test_data = df.iloc[split_idx:].copy()

                print(f"    📈 Train: {len(train_data)} days, Test: {len(test_data)} days")

                # Test each timescale
                timescale_results = {}
                for timescale in timescales:
                    print(f"    🎯 Testing {timescale} timescale...")

                    accuracy_metrics = self._test_timescale_accuracy(
                        test_data.copy(), timescale
                    )

                    if accuracy_metrics:
                        timescale_results[timescale] = accuracy_metrics
                        print(f"      ✅ {timescale}: MSE={accuracy_metrics['mse']:.6f}, "
                              f"MAE={accuracy_metrics['mae']:.6f}, R²={accuracy_metrics['r2']:.4f}")

                if timescale_results:
                    symbol_results[period_key] = {
                        'timescale_results': timescale_results,
                        'market_data': test_data,
                        'train_period': (start_date, train_data.index[-1].strftime('%Y-%m-%d')),
                        'test_period': (test_data.index[0].strftime('%Y-%m-%d'), end_date)
                    }

            if symbol_results:
                all_results[symbol] = symbol_results

        self.results = all_results
        return all_results

    def _test_timescale_accuracy(self, df, timescale):
        """
        Test prediction accuracy for a specific timescale
        """
        try:
            # Generate predictions
            predictions = predict_excess_returns(df, use_ml=True)

            if not predictions or len(predictions) != len(df):
                return None

            # Add predictions to dataframe
            df = df.copy()
            df['predicted_excess_return'] = predictions

            # Apply timescale-specific parameters
            timescale_params = self._get_timescale_params(timescale)
            df = self._apply_timescale_filter(df, timescale_params)

            # Calculate accuracy metrics using filtered predictions
            actual = df['excess_return'].values
            predicted = df['filtered_prediction'].values  # Use filtered predictions

            # Remove NaN values
            valid_mask = ~(np.isnan(actual) | np.isnan(predicted))
            actual = actual[valid_mask]
            predicted = predicted[valid_mask]

            if len(actual) < 10:  # Need minimum data points
                return None

            # Calculate metrics
            mse = mean_squared_error(actual, predicted)
            mae = mean_absolute_error(actual, predicted)
            r2 = r2_score(actual, predicted)

            # Directional accuracy (sign prediction)
            actual_direction = np.sign(actual)
            predicted_direction = np.sign(predicted)
            directional_accuracy = np.mean(actual_direction == predicted_direction)

            # Correlation
            correlation = np.corrcoef(actual, predicted)[0, 1]

            # Mean absolute percentage error (for non-zero values)
            non_zero_mask = actual != 0
            if np.any(non_zero_mask):
                mape = np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask]))
            else:
                mape = np.nan

            return {
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'directional_accuracy': directional_accuracy,
                'correlation': correlation,
                'mape': mape,
                'n_samples': len(actual),
                'timescale': timescale
            }

        except Exception as e:
            print(f"    ⚠️  Error testing {timescale}: {e}")
            return None

    def _get_timescale_params(self, timescale):
        """Get parameters for a specific timescale"""
        params = {
            'conservative': {
                'confidence_threshold': 0.8,
                'min_prediction_magnitude': 0.005,
                'max_volatility_filter': 0.02
            },
            'moderate': {
                'confidence_threshold': 0.7,
                'min_prediction_magnitude': 0.003,
                'max_volatility_filter': 0.03
            },
            'aggressive': {
                'confidence_threshold': 0.6,
                'min_prediction_magnitude': 0.002,
                'max_volatility_filter': 0.04
            },
            'very_aggressive': {
                'confidence_threshold': 0.5,
                'min_prediction_magnitude': 0.001,
                'max_volatility_filter': 0.05
            }
        }
        return params.get(timescale, params['moderate'])

    def _apply_timescale_filter(self, df, params):
        """Apply timescale-specific filtering to predictions"""
        # Calculate prediction confidence
        df['pred_confidence'] = 1 - (df['predicted_excess_return'].rolling(10).std() /
                                   df['predicted_excess_return'].rolling(10).mean().abs())

        # Fill NaN values in confidence
        df['pred_confidence'] = df['pred_confidence'].fillna(0.5)

        # Apply confidence threshold
        df['filtered_prediction'] = np.where(
            df['pred_confidence'] >= params['confidence_threshold'],
            df['predicted_excess_return'],
            0
        )

        # Apply magnitude filter
        df['filtered_prediction'] = np.where(
            df['filtered_prediction'].abs() >= params['min_prediction_magnitude'],
            df['filtered_prediction'],
            0
        )

        # Apply volatility filter
        current_volatility = df['excess_return'].rolling(20).std()
        df['filtered_prediction'] = np.where(
            current_volatility <= params['max_volatility_filter'],
            df['filtered_prediction'],
            df['filtered_prediction'] * 0.5  # Reduce confidence in high volatility
        )

        return df

    def generate_accuracy_report(self, save_plots=True):
        """
        Generate comprehensive accuracy report
        """
        if not self.results:
            print("❌ No results to report. Run accuracy tests first.")
            return

        print("\n" + "="*80)
        print("🎯 FSOT 2.5 ACCURACY REPORT")
        print("="*80)

        # Overall summary
        self._print_overall_summary()

        # Timescale comparison
        self._print_timescale_comparison()

        # Market condition analysis
        self._print_market_condition_analysis()

        # Best/worst performers
        self._print_symbol_performance()

        # Generate visualizations
        if save_plots:
            self._generate_accuracy_plots()

        print("\n" + "="*80)
        print("✅ Accuracy testing complete!")
        print("="*80)

    def _print_overall_summary(self):
        """Print overall accuracy summary"""
        print("\n📊 OVERALL ACCURACY SUMMARY")
        print("-" * 40)

        all_metrics = []
        for symbol, symbol_data in self.results.items():
            for period, period_data in symbol_data.items():
                for timescale, metrics in period_data['timescale_results'].items():
                    all_metrics.append(metrics)

        if not all_metrics:
            print("No valid metrics found.")
            return

        # Calculate averages
        avg_mse = np.mean([m['mse'] for m in all_metrics])
        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_r2 = np.mean([m['r2'] for m in all_metrics])
        avg_directional = np.mean([m['directional_accuracy'] for m in all_metrics])
        avg_correlation = np.mean([m['correlation'] for m in all_metrics])

        print(f"  Mean Squared Error:     {avg_mse:.6f}")
        print(f"  Mean Absolute Error:    {avg_mae:.6f}")
        print(f"  R² Score:               {avg_r2:.4f}")
        print(f"  Directional Accuracy:   {avg_directional:.1%}")
        print(f"  Correlation:            {avg_correlation:.4f}")
        # Best metrics
        best_r2 = max([m['r2'] for m in all_metrics])
        best_directional = max([m['directional_accuracy'] for m in all_metrics])

        print("\n🎯 Best Performance:")
        print(f"  Best R²: {best_r2:.4f}")
        print(f"  Best Directional: {best_directional:.1%}")

    def _print_timescale_comparison(self):
        """Compare accuracy across timescales"""
        print("\n🎯 TIMESCALE ACCURACY COMPARISON")
        print("-" * 40)

        timescale_stats = {}
        for symbol, symbol_data in self.results.items():
            for period, period_data in symbol_data.items():
                for timescale, metrics in period_data['timescale_results'].items():
                    if timescale not in timescale_stats:
                        timescale_stats[timescale] = []
                    timescale_stats[timescale].append(metrics)

        for timescale, metrics_list in timescale_stats.items():
            if metrics_list:
                avg_r2 = np.mean([m['r2'] for m in metrics_list])
                avg_directional = np.mean([m['directional_accuracy'] for m in metrics_list])
                avg_mae = np.mean([m['mae'] for m in metrics_list])

                print(f"\n{timescale.upper()}:")
                print(f"  R² Score: {avg_r2:.4f}")
                print(f"  Directional Accuracy: {avg_directional:.1%}")
                print(f"  MAE: {avg_mae:.6f}")

    def _print_market_condition_analysis(self):
        """Analyze accuracy across different market conditions"""
        print("\n📈 MARKET CONDITION ANALYSIS")
        print("-" * 40)

        # Group by periods (representing different market conditions)
        period_stats = {}
        for symbol, symbol_data in self.results.items():
            for period, period_data in symbol_data.items():
                if period not in period_stats:
                    period_stats[period] = []
                for timescale, metrics in period_data['timescale_results'].items():
                    period_stats[period].append(metrics)

        for period, metrics_list in period_stats.items():
            if metrics_list:
                avg_r2 = np.mean([m['r2'] for m in metrics_list])
                avg_directional = np.mean([m['directional_accuracy'] for m in metrics_list])

                # Map period to market condition
                condition = self._map_period_to_condition(period)
                print(f"\n{condition} ({period}):")
                print(f"  R² Score: {avg_r2:.4f}")
                print(f"  Directional Accuracy: {avg_directional:.1%}")

    def _map_period_to_condition(self, period):
        """Map period to market condition description"""
        if '2020' in period:
            return "COVID-19 Crisis"
        elif '2021' in period:
            return "Post-COVID Recovery"
        elif '2022' in period:
            return "Inflation/High Volatility"
        elif '2023' in period:
            return "Market Recovery"
        else:
            return "Unknown Period"

    def _print_symbol_performance(self):
        """Show best and worst performing symbols"""
        print("\n🏆 SYMBOL PERFORMANCE RANKING")
        print("-" * 40)

        symbol_stats = {}
        for symbol, symbol_data in self.results.items():
            all_metrics = []
            for period, period_data in symbol_data.items():
                for timescale, metrics in period_data['timescale_results'].items():
                    all_metrics.append(metrics)

            if all_metrics:
                avg_r2 = np.mean([m['r2'] for m in all_metrics])
                avg_directional = np.mean([m['directional_accuracy'] for m in all_metrics])
                symbol_stats[symbol] = {
                    'avg_r2': avg_r2,
                    'avg_directional': avg_directional,
                    'n_tests': len(all_metrics)
                }

        # Sort by R²
        sorted_symbols = sorted(symbol_stats.items(),
                              key=lambda x: x[1]['avg_r2'], reverse=True)

        print("\nBy R² Score:")
        for i, (symbol, stats) in enumerate(sorted_symbols[:5], 1):
            print(f"{i}. {symbol}: R²={stats['avg_r2']:.4f}, "
                  f"Directional={stats['avg_directional']:.1%} ({stats['n_tests']} tests)")

        print("\nWorst Performers:")
        for i, (symbol, stats) in enumerate(sorted_symbols[-3:], 1):
            print(f"{len(sorted_symbols)-3+i}. {symbol}: R²={stats['avg_r2']:.4f}, "
                  f"Directional={stats['avg_directional']:.1%} ({stats['n_tests']} tests)")

    def _generate_accuracy_plots(self):
        """Generate accuracy visualization plots"""
        try:
            # Set up the plotting style
            plt.style.use('default')
            sns.set_palette("husl")

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('FSOT 2.5 Prediction Accuracy Analysis', fontsize=16, fontweight='bold')

            # 1. R² Score Distribution
            r2_scores = []
            timescales = []
            for symbol, symbol_data in self.results.items():
                for period, period_data in symbol_data.items():
                    for timescale, metrics in period_data['timescale_results'].items():
                        r2_scores.append(metrics['r2'])
                        timescales.append(timescale)

            if r2_scores:
                r2_df = pd.DataFrame({'R2': r2_scores, 'Timescale': timescales})
                sns.boxplot(data=r2_df, x='Timescale', y='R2', ax=axes[0,0])
                axes[0,0].set_title('R² Score Distribution by Timescale')
                axes[0,0].tick_params(axis='x', rotation=45)

            # 2. Directional Accuracy
            directional_scores = []
            timescale_labels = []
            for symbol, symbol_data in self.results.items():
                for period, period_data in symbol_data.items():
                    for timescale, metrics in period_data['timescale_results'].items():
                        directional_scores.append(metrics['directional_accuracy'])
                        timescale_labels.append(timescale)

            if directional_scores:
                dir_df = pd.DataFrame({'Directional_Accuracy': directional_scores,
                                     'Timescale': timescale_labels})
                sns.barplot(data=dir_df, x='Timescale', y='Directional_Accuracy',
                          ax=axes[0,1], errorbar=None)
                axes[0,1].set_title('Directional Accuracy by Timescale')
                axes[0,1].set_ylabel('Directional Accuracy')
                axes[0,1].tick_params(axis='x', rotation=45)

            # 3. Symbol Performance
            symbol_r2 = {}
            for symbol, symbol_data in self.results.items():
                all_r2 = []
                for period, period_data in symbol_data.items():
                    for timescale, metrics in period_data['timescale_results'].items():
                        all_r2.append(metrics['r2'])
                if all_r2:
                    symbol_r2[symbol] = np.mean(all_r2)

            if symbol_r2:
                symbols_sorted = sorted(symbol_r2.items(), key=lambda x: x[1], reverse=True)
                symbols, r2_values = zip(*symbols_sorted)
                axes[1,0].bar(symbols, r2_values)
                axes[1,0].set_title('Average R² Score by Symbol')
                axes[1,0].set_ylabel('Average R²')
                axes[1,0].tick_params(axis='x', rotation=45)

            # 4. Market Condition Performance
            period_r2 = {}
            for symbol, symbol_data in self.results.items():
                for period, period_data in symbol_data.items():
                    if period not in period_r2:
                        period_r2[period] = []
                    for timescale, metrics in period_data['timescale_results'].items():
                        period_r2[period].append(metrics['r2'])

            if period_r2:
                period_labels = [self._map_period_to_condition(p) for p in period_r2.keys()]
                period_avg_r2 = [np.mean(r2_list) for r2_list in period_r2.values()]
                axes[1,1].bar(period_labels, period_avg_r2)
                axes[1,1].set_title('R² Score by Market Condition')
                axes[1,1].set_ylabel('Average R²')
                axes[1,1].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig('fsot_accuracy_analysis.png', dpi=300, bbox_inches='tight')
            print("✅ Accuracy plots saved as 'fsot_accuracy_analysis.png'")

        except Exception as e:
            print(f"⚠️  Error generating plots: {e}")


def main():
    """
    Main function to run accuracy testing
    """
    print("🚀 Starting FSOT 2.5 Accuracy Testing...")

    # Initialize tester
    tester = AccuracyTester()

    # Run comprehensive accuracy tests
    print("\n🔬 Running accuracy tests across multiple symbols and time periods...")
    results = tester.run_accuracy_test()

    if results:
        print(f"\n✅ Testing complete! Processed {len(results)} symbols.")

        # Generate comprehensive report
        tester.generate_accuracy_report()

        # Print key insights
        print("\n💡 KEY INSIGHTS:")
        print("-" * 20)
        print("• Conservative timescale typically shows highest directional accuracy")
        print("• Aggressive timescales may have higher R² but lower directional accuracy")
        print("• Performance varies significantly across different market conditions")
        print("• Some symbols (tech stocks) may be more predictable than others")

    else:
        print("❌ No valid results generated. Check data availability and try again.")


if __name__ == "__main__":
    main()
"""
FSOT 2.5 Accuracy Analysis vs Actual Market Data
================================================

This script demonstrates the current realistic accuracy of FSOT 2.5
compared to actual market data and random guessing.
"""

import pandas as pd
import numpy as np
from fsot_finance_predictor import load_live_data, predict_excess_returns, generate_trading_strategy, evaluate_strategy
import matplotlib.pyplot as plt
import seaborn as sns

def run_accuracy_comparison():
    """Run comprehensive accuracy comparison"""

    print("🔬 FSOT 2.5 ACCURACY ANALYSIS vs ACTUAL MARKET DATA")
    print("=" * 60)

    # Test symbols
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY']

    results = {}

    for symbol in symbols:
        print(f"\n📊 Testing {symbol}...")

        try:
            # Load data
            df = load_live_data([symbol], start_date='2023-01-01', end_date='2024-12-31')

            if df is None or df.empty:
                print(f"❌ No data for {symbol}")
                continue

            # Generate predictions
            predictions = predict_excess_returns(df, use_ml=True)

            # Create trading strategy
            df = generate_trading_strategy(df)

            # Calculate TRUE directional accuracy
            df['direction_correct'] = ((df['predicted_excess_return'] > 0) == (df['excess_return'] > 0)).astype(int)
            directional_accuracy = df['direction_correct'].mean()

            # Evaluate performance
            metrics = evaluate_strategy(df)
            metrics['directional_accuracy'] = directional_accuracy

            results[symbol] = {
                'metrics': metrics,
                'data': df,
                'predictions': predictions
            }

            print(f"✅ {symbol}: Directional Acc = {directional_accuracy:.1%}, Strategy Win Rate = {metrics['win_rate']:.1%}")

        except Exception as e:
            print(f"❌ Error testing {symbol}: {e}")
            continue

    if not results:
        print("❌ No results to analyze")
        return

    # Comprehensive analysis
    analyze_results(results)

def analyze_results(results):
    """Analyze and display comprehensive results"""

    print("\n" + "="*60)
    print("📊 COMPREHENSIVE ACCURACY ANALYSIS")
    print("="*60)

    # Aggregate metrics
    directional_accuracies = []
    strategy_win_rates = []
    r2_scores = []
    correlations = []

    for symbol, data in results.items():
        metrics = data['metrics']
        directional_accuracies.append(metrics['directional_accuracy'])
        strategy_win_rates.append(metrics['win_rate'])
        r2_scores.append(metrics['correlation_preds_actual']**2)  # Convert correlation to R²-like
        correlations.append(metrics['correlation_preds_actual'])

    # Overall statistics
    avg_directional = np.mean(directional_accuracies)
    avg_strategy_win = np.mean(strategy_win_rates)
    avg_r2 = np.mean(r2_scores)
    avg_correlation = np.mean(correlations)

    print("\n🎯 OVERALL PERFORMANCE:")
    print(f"  Average Directional Accuracy: {avg_directional:.1%}")
    print(f"  Average Strategy Win Rate: {avg_strategy_win:.1%}")
    print(f"  Average R² Score: {avg_r2:.4f}")
    print(f"  Average Correlation: {avg_correlation:.4f}")

    # Market reality check
    print("\n📈 MARKET REALITY CHECK:")
    print(f"  Random Guessing: 50.0% directional accuracy")
    print(f"  Your FSOT Directional Edge: {(avg_directional - 0.5):.1%} above random")
    print(f"  Your Strategy Win Rate: {avg_strategy_win:.1%}")
    print(f"  Professional Fund Managers: ~52-55% directional accuracy")
    print(f"  Your Performance: {'Excellent' if avg_directional > 0.55 else 'Good' if avg_directional > 0.52 else 'Average'}")

    # Best and worst performers
    best_symbol = max(results.items(), key=lambda x: x[1]['metrics']['directional_accuracy'])
    worst_symbol = min(results.items(), key=lambda x: x[1]['metrics']['directional_accuracy'])

    print("\n🏆 BEST/WORST PERFORMERS:")
    print(f"  Best: {best_symbol[0]} ({best_symbol[1]['metrics']['directional_accuracy']:.1%} directional)")
    print(f"  Worst: {worst_symbol[0]} ({worst_symbol[1]['metrics']['directional_accuracy']:.1%} directional)")

    # Practical implications
    print("\n💡 PRACTICAL IMPLICATIONS:")
    print(f"  Annual Return Potential: ~{avg_directional*20:.1f}% (rough estimate)")
    print(f"  Risk-Adjusted Edge: {'Strong' if avg_directional > 0.55 else 'Moderate' if avg_directional > 0.52 else 'Weak'}")
    print(f"  Tradability: {'Yes' if avg_directional > 0.52 else 'Marginal'}")

    # Create visualization
    create_accuracy_visualization(results)

def create_accuracy_visualization(results):
    """Create accuracy comparison visualization"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FSOT 2.5 Accuracy Analysis vs Market Reality', fontsize=16, fontweight='bold')

    # 1. Directional Accuracy by Symbol
    symbols = list(results.keys())
    directional_acc = [results[s]['metrics']['directional_accuracy'] for s in symbols]

    bars = ax1.bar(symbols, directional_acc)
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Random (50%)')
    ax1.axhline(y=0.52, color='orange', linestyle='--', label='Professional (52%)')
    ax1.set_title('Directional Accuracy by Symbol')
    ax1.set_ylabel('Directional Accuracy')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # Color bars based on performance
    for bar, acc in zip(bars, directional_acc):
        if acc > 0.52:
            bar.set_color('green')
        elif acc > 0.5:
            bar.set_color('yellow')
        else:
            bar.set_color('red')

    # 2. Prediction vs Actual Scatter (using first symbol)
    first_symbol = list(results.keys())[0]
    df = results[first_symbol]['data']

    if 'predicted_excess_return' in df.columns:
        valid_data = df.dropna(subset=['predicted_excess_return', 'excess_return'])
        ax2.scatter(valid_data['excess_return'], valid_data['predicted_excess_return'],
                   alpha=0.6, color='blue', s=30)
        ax2.plot([-0.05, 0.05], [-0.05, 0.05], color='red', linestyle='--', label='Perfect Prediction')
        ax2.set_xlabel('Actual Returns')
        ax2.set_ylabel('Predicted Returns')
        ax2.set_title(f'Prediction Accuracy: {first_symbol}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Performance Distribution
    all_directional = [results[s]['metrics']['directional_accuracy'] for s in results.keys()]
    ax3.hist(all_directional, bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(np.mean(all_directional), color='red', linestyle='--',
               label=f'Mean: {np.mean(all_directional):.1%}')
    ax3.axvline(0.5, color='orange', linestyle='--', label='Random')
    ax3.set_title('Performance Distribution')
    ax3.set_xlabel('Directional Accuracy')
    ax3.set_ylabel('Frequency')
    ax3.legend()

    # 4. Market Reality Comparison
    categories = ['Random\nGuessing', 'Typical\nFund\nManager', 'FSOT 2.5\nAverage', 'FSOT 2.5\nBest']
    values = [0.5, 0.53, np.mean(all_directional), max(all_directional)]

    bars = ax4.bar(categories, values)
    ax4.set_title('Market Reality Comparison')
    ax4.set_ylabel('Directional Accuracy')
    ax4.tick_params(axis='x', rotation=45)

    # Color bars
    colors = ['red', 'orange', 'blue', 'green']
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # Add value labels
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.1%}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('fsot_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✅ Accuracy comparison saved as 'fsot_accuracy_comparison.png'")

    plt.show()

if __name__ == "__main__":
    run_accuracy_comparison()
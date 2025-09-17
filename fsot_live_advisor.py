"""
FSOT 2.5 Live Market Advisor
===========================

Real-time market analysis and investment recommendations using FSOT 2.5
Provides live data monitoring and suggests optimal investment opportunities
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import FSOT components
from fsot_finance_predictor import (
    predict_excess_returns, generate_trading_strategy,
    evaluate_strategy, add_enhanced_technical_indicators,
    detect_market_regime, apply_trailing_stops
)

class LiveMarketAdvisor:
    """
    Live market advisor using FSOT 2.5 for real-time analysis and recommendations
    """

    def __init__(self, watchlist=None, update_interval_minutes=15, investment_timescale='moderate'):
        self.watchlist = watchlist or [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX',
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C',
            'JNJ', 'PFE', 'UNH', 'MRK', 'ABT',
            'SPY', 'QQQ', 'IWM', 'DIA',
            # High-volatility additions for speculative analysis
            'SOXS', 'TQQQ', 'UVXY', 'VXX', 'SPXL', 'TECL'
        ]
        self.update_interval = update_interval_minutes
        self.last_update = None
        self.market_data = {}
        self.analysis_results = {}
        self.recommendations = {}

        # Investment timescale settings
        self.investment_timescale = investment_timescale
        self.timescale_settings = self._get_timescale_settings()

    def _get_timescale_settings(self):
        """
        Define trading parameters based on investment timescale
        """
        settings = {
            'conservative': {
                'name': 'Conservative',
                'description': 'Long-term, low-risk approach',
                'holding_period_days': 90,
                'confidence_threshold': 0.8,
                'position_size_max': 0.5,
                'trailing_stop_pct': 0.05,
                'profit_target_pct': 0.15,
                'max_trades_per_day': 1,
                'risk_per_trade_pct': 0.02,
                'description_long': 'Focuses on capital preservation with steady, long-term growth. Uses high confidence thresholds and tight risk controls.'
            },
            'moderate': {
                'name': 'Moderate',
                'description': 'Balanced approach for medium-term trading',
                'holding_period_days': 30,
                'confidence_threshold': 0.7,
                'position_size_max': 1.0,
                'trailing_stop_pct': 0.08,
                'profit_target_pct': 0.25,
                'max_trades_per_day': 2,
                'risk_per_trade_pct': 0.05,
                'description_long': 'Balances growth potential with risk management. Suitable for most investors seeking moderate returns.'
            },
            'aggressive': {
                'name': 'Aggressive',
                'description': 'Short-term, high-risk approach',
                'holding_period_days': 7,
                'confidence_threshold': 0.6,
                'position_size_max': 1.5,
                'trailing_stop_pct': 0.12,
                'profit_target_pct': 0.40,
                'max_trades_per_day': 3,
                'risk_per_trade_pct': 0.08,
                'description_long': 'Maximizes profit potential through frequent trading and higher risk tolerance. Requires active monitoring.'
            },
            'very_aggressive': {
                'name': 'Very Aggressive',
                'description': 'Day trading, maximum risk approach',
                'holding_period_days': 1,
                'confidence_threshold': 0.5,
                'position_size_max': 2.0,
                'trailing_stop_pct': 0.15,
                'profit_target_pct': 0.60,
                'max_trades_per_day': 5,
                'risk_per_trade_pct': 0.12,
                'description_long': 'Maximum profit potential with highest risk. Designed for experienced traders with high risk tolerance.'
            }
        }

        if self.investment_timescale not in settings:
            print(f"⚠️  Unknown timescale '{self.investment_timescale}', using 'moderate'")
            self.investment_timescale = 'moderate'

        return settings[self.investment_timescale]

    def get_live_data(self, symbols, lookback_days=90):
        """
        Fetch recent market data for analysis
        """
        print(f"📊 Fetching live data for {len(symbols)} symbols...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        data = {}
        for symbol in symbols:
            try:
                print(f"  📈 Loading {symbol}...")
                df = yf.download(symbol, start=start_date, end=end_date, progress=False)

                if not df.empty:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(df.columns, pd.MultiIndex):
                        # Flatten MultiIndex columns
                        df.columns = df.columns.get_level_values(0)

                    # Standardize column names to lowercase
                    df.columns = df.columns.str.lower()

                    # Ensure we have required columns
                    if 'close' not in df.columns:
                        print(f"    ⚠️  No close data for {symbol}")
                        continue

                    # Add required columns if missing
                    for col in ['open', 'high', 'low']:
                        if col not in df.columns:
                            df[col] = df['close']

                    if 'volume' not in df.columns:
                        df['volume'] = 1000000  # Default volume

                    # Add excess returns
                    df['excess_return'] = df['close'].pct_change().fillna(0)

                    # Add technical indicators
                    df = add_enhanced_technical_indicators(df)

                    data[symbol] = df
                    print(f"    ✅ Loaded {len(df)} days of data for {symbol}")
                else:
                    print(f"    ❌ No data available for {symbol}")

            except Exception as e:
                print(f"    ❌ Error loading {symbol}: {e}")
                continue

        print(f"✅ Successfully loaded data for {len(data)}/{len(symbols)} symbols")
        return data

    def analyze_symbol(self, symbol, df):
        """
        Perform comprehensive FSOT analysis on a symbol
        """
        try:
            print(f"🔬 Analyzing {symbol}...")

            # Step 1: Detect market regime
            df_with_regime = detect_market_regime(df.copy())

            # Step 2: Generate predictions
            df_with_predictions = df_with_regime.copy()
            predictions = predict_excess_returns(df_with_predictions, use_ml=True)

            # Step 3: Generate trading strategy with regime adaptation
            strategy_df = generate_trading_strategy(df_with_predictions)

            # Step 4: Apply trailing stops
            strategy_df = apply_trailing_stops(strategy_df)

            # Step 5: Evaluate performance
            metrics = evaluate_strategy(strategy_df)

            # Step 6: Generate current recommendation
            current_recommendation = self._generate_recommendation(symbol, strategy_df, metrics)

            return {
                'symbol': symbol,
                'data': strategy_df,
                'metrics': metrics,
                'regime': strategy_df['regime'].iloc[-1] if 'regime' in strategy_df.columns else 'unknown',
                'recommendation': current_recommendation,
                'last_price': df['close'].iloc[-1],
                'price_change_1d': df['close'].pct_change().iloc[-1],
                'price_change_5d': df['close'].pct_change(5).iloc[-1],
                'price_change_30d': df['close'].pct_change(30).iloc[-1],
                'volatility_30d': df['close'].pct_change().rolling(30).std().iloc[-1],
                'volume_trend': df['volume'].pct_change(5).mean() if 'volume' in df.columns else 0
            }

        except Exception as e:
            print(f"❌ Error analyzing {symbol}: {e}")
            return None

    def _generate_recommendation(self, symbol, df, metrics):
        """
        Generate investment recommendation based on FSOT analysis and timescale settings
        """
        try:
            # Get current position and confidence
            current_position = df['position'].iloc[-1] if 'position' in df.columns else 0
            current_confidence = df['confidence'].iloc[-1] if 'confidence' in df.columns else 0.5
            current_regime = df['regime'].iloc[-1] if 'regime' in df.columns else 'sideways'

            # Get recent performance metrics
            recent_win_rate = df['strategy_return'].tail(20).gt(0).mean() if len(df) >= 20 else 0.5
            recent_sharpe = df['strategy_return'].tail(60).mean() / df['strategy_return'].tail(60).std() * np.sqrt(252) if len(df) >= 60 else 0

            # Use timescale-specific confidence threshold
            timescale_confidence_threshold = self.timescale_settings['confidence_threshold']

            # Adjust confidence based on timescale
            adjusted_confidence = current_confidence
            if self.investment_timescale == 'conservative':
                # Conservative: require higher confidence
                adjusted_confidence = min(current_confidence, current_confidence * 1.2)
            elif self.investment_timescale == 'aggressive':
                # Aggressive: accept lower confidence for more opportunities
                adjusted_confidence = max(current_confidence, current_confidence * 0.8)
            elif self.investment_timescale == 'very_aggressive':
                # Very aggressive: significantly lower threshold
                adjusted_confidence = max(current_confidence, current_confidence * 0.6)

            # Scoring system with timescale adjustments
            score = 0

            # Position score (0-30 points) - adjusted by timescale
            position_multiplier = 1.0
            if self.investment_timescale == 'conservative':
                position_multiplier = 0.7  # Reduce position scores for conservative
            elif self.investment_timescale in ['aggressive', 'very_aggressive']:
                position_multiplier = 1.3  # Increase position scores for aggressive

            if current_position >= 2:
                score += 30 * position_multiplier  # Strong buy
            elif current_position >= 1:
                score += 20 * position_multiplier  # Moderate buy
            elif current_position >= 0:
                score += 10 * position_multiplier  # Hold/weak buy
            elif current_position >= -1:
                score += 5 * position_multiplier   # Weak sell
            else:
                score += 0   # Strong sell

            # Confidence score (0-25 points) - adjusted by timescale
            confidence_score = adjusted_confidence * 25
            if self.investment_timescale == 'conservative':
                confidence_score *= 1.2  # Weight confidence more heavily
            elif self.investment_timescale == 'very_aggressive':
                confidence_score *= 0.8  # Weight confidence less heavily
            score += confidence_score

            # Recent performance score (0-20 points)
            score += recent_win_rate * 20

            # Sharpe ratio score (0-15 points) - adjusted by timescale
            sharpe_multiplier = 1.0
            if self.investment_timescale == 'conservative':
                sharpe_multiplier = 1.5  # Conservative values Sharpe more
            elif self.investment_timescale == 'very_aggressive':
                sharpe_multiplier = 0.5  # Aggressive cares less about Sharpe

            if recent_sharpe > 2:
                score += 15 * sharpe_multiplier
            elif recent_sharpe > 1:
                score += 10 * sharpe_multiplier
            elif recent_sharpe > 0:
                score += 5 * sharpe_multiplier

            # Market regime adjustment (0-10 points) - timescale specific
            regime_score = 0
            if current_regime == 'bull':
                if current_position > 0:
                    regime_score = 10
                    if self.investment_timescale in ['aggressive', 'very_aggressive']:
                        regime_score *= 1.5  # Aggressive loves bull markets
            elif current_regime == 'bear':
                if current_position < 0:
                    regime_score = 10
                    if self.investment_timescale in ['aggressive', 'very_aggressive']:
                        regime_score *= 1.5  # Aggressive loves bear markets too
            elif current_regime == 'volatile':
                regime_score = 5  # Mixed signal in volatile markets
                if self.investment_timescale == 'very_aggressive':
                    regime_score *= 2  # Very aggressive loves volatility

            score += regime_score

            # Timescale-specific score adjustments
            if self.investment_timescale == 'conservative':
                score *= 0.9  # Slightly reduce scores to be more conservative
            elif self.investment_timescale == 'aggressive':
                score *= 1.1  # Slightly increase scores for more opportunities
            elif self.investment_timescale == 'very_aggressive':
                score *= 1.2  # Significantly increase scores for maximum opportunities

            # Determine recommendation with timescale-specific thresholds
            if self.investment_timescale == 'conservative':
                thresholds = {'STRONG_BUY': 85, 'BUY': 70, 'HOLD': 50, 'WEAK_SELL': 30}
            elif self.investment_timescale == 'moderate':
                thresholds = {'STRONG_BUY': 80, 'BUY': 60, 'HOLD': 40, 'WEAK_SELL': 20}
            elif self.investment_timescale == 'aggressive':
                thresholds = {'STRONG_BUY': 75, 'BUY': 55, 'HOLD': 35, 'WEAK_SELL': 15}
            else:  # very_aggressive
                thresholds = {'STRONG_BUY': 70, 'BUY': 50, 'HOLD': 30, 'WEAK_SELL': 10}

            if score >= thresholds['STRONG_BUY']:
                recommendation = 'STRONG_BUY'
                confidence_level = 'High'
            elif score >= thresholds['BUY']:
                recommendation = 'BUY'
                confidence_level = 'Medium-High'
            elif score >= thresholds['HOLD']:
                recommendation = 'HOLD'
                confidence_level = 'Medium'
            elif score >= thresholds['WEAK_SELL']:
                recommendation = 'WEAK_SELL'
                confidence_level = 'Low-Medium'
            else:
                recommendation = 'SELL'
                confidence_level = 'Low'

            return {
                'action': recommendation,
                'confidence_level': confidence_level,
                'score': score,
                'position': current_position,
                'regime': current_regime,
                'recent_win_rate': recent_win_rate,
                'recent_sharpe': recent_sharpe,
                'timescale': self.investment_timescale,
                'adjusted_confidence': adjusted_confidence,
                'rationale': self._generate_rationale(recommendation, current_position, adjusted_confidence, current_regime, score)
            }

        except Exception as e:
            print(f"❌ Error generating recommendation for {symbol}: {e}")
            return {
                'action': 'HOLD',
                'confidence_level': 'Unknown',
                'score': 50,
                'position': 0,
                'regime': 'unknown',
                'recent_win_rate': 0.5,
                'recent_sharpe': 0,
                'timescale': self.investment_timescale,
                'adjusted_confidence': 0.5,
                'rationale': 'Analysis error - maintain current position'
            }

    def _generate_rationale(self, recommendation, position, confidence, regime, score):
        """
        Generate human-readable rationale for the recommendation with timescale context
        """
        timescale_info = f" ({self.timescale_settings['name']} timescale)"

        rationales = {
            'STRONG_BUY': [
                f"Strong bullish signal with {position:.1f} position size{timescale_info}",
                f"High confidence ({confidence:.1f}) in {regime} market regime",
                f"Recommendation score: {score:.1f}/100 - exceeds threshold for aggressive buying",
                "Recent performance shows consistent gains",
                "FSOT analysis indicates favorable risk-reward profile",
                f"Timescale setting allows for {self.timescale_settings['max_trades_per_day']} trades/day with {self.timescale_settings['risk_per_trade_pct']*100:.1f}% risk per trade"
            ],
            'BUY': [
                f"Positive signal with {position:.1f} position size{timescale_info}",
                f"Moderate confidence ({confidence:.1f}) in {regime} market",
                f"Recommendation score: {score:.1f}/100 - suitable for accumulation",
                "Analysis suggests potential upside opportunity",
                "Consider accumulating position gradually",
                f"Current {regime} regime aligns with {self.timescale_settings['name'].lower()} trading style"
            ],
            'HOLD': [
                f"Neutral signal with {position:.1f} position size{timescale_info}",
                f"Moderate confidence ({confidence:.1f}) in {regime} market",
                f"Recommendation score: {score:.1f}/100 - suggests maintaining position",
                "Market conditions suggest maintaining current position",
                "Monitor for clearer signals before adjusting",
                f"{self.timescale_settings['name']} approach recommends patience in uncertain conditions"
            ],
            'WEAK_SELL': [
                f"Weak bearish signal with {position:.1f} position size{timescale_info}",
                f"Low confidence ({confidence:.1f}) in {regime} market",
                f"Recommendation score: {score:.1f}/100 - consider gradual reduction",
                "Consider reducing position size gradually",
                "Monitor for potential trend reversal",
                f"Timescale setting suggests {self.timescale_settings['holding_period_days']}-day holding period"
            ],
            'SELL': [
                f"Bearish signal with {position:.1f} position size{timescale_info}",
                f"Low confidence ({confidence:.1f}) in {regime} market",
                f"Recommendation score: {score:.1f}/100 - below acceptable threshold",
                "Analysis suggests reducing or exiting position",
                "Risk management indicates defensive posture",
                f"{self.timescale_settings['name']} approach prioritizes capital preservation"
            ]
        }

        return rationales.get(recommendation, ["Analysis in progress"])

    def generate_performance_feedback(self, symbol, df, metrics):
        """
        Generate detailed feedback on trade performance and advice for improvement
        """
        try:
            feedback = {
                'symbol': symbol,
                'timescale': self.investment_timescale,
                'overall_performance': 'neutral',
                'issues_identified': [],
                'recommendations': [],
                'risk_assessment': 'moderate',
                'market_alignment': 'good'
            }

            # Analyze recent performance
            if len(df) >= 20:
                recent_returns = df['strategy_return'].tail(20)
                win_rate = (recent_returns > 0).mean()
                avg_return = recent_returns.mean()
                volatility = recent_returns.std()

                # Performance assessment
                if win_rate > 0.6 and avg_return > 0.001:
                    feedback['overall_performance'] = 'excellent'
                elif win_rate > 0.5 and avg_return > 0:
                    feedback['overall_performance'] = 'good'
                elif win_rate < 0.4 or avg_return < -0.001:
                    feedback['overall_performance'] = 'poor'
                else:
                    feedback['overall_performance'] = 'neutral'

                # Identify issues
                if win_rate < 0.4:
                    feedback['issues_identified'].append("Low win rate - strategy may need adjustment")
                if volatility > 0.02:
                    feedback['issues_identified'].append("High volatility - consider reducing position sizes")
                if avg_return < 0:
                    feedback['issues_identified'].append("Negative average returns - review entry/exit timing")

                # Risk assessment
                if volatility > 0.03 or win_rate < 0.3:
                    feedback['risk_assessment'] = 'high'
                elif volatility < 0.01 and win_rate > 0.6:
                    feedback['risk_assessment'] = 'low'

            # Market regime alignment
            current_regime = df['regime'].iloc[-1] if 'regime' in df.columns else 'unknown'
            if current_regime == 'volatile' and self.investment_timescale in ['conservative', 'moderate']:
                feedback['market_alignment'] = 'caution'
                feedback['issues_identified'].append("Volatile market may not suit current timescale")
            elif current_regime == 'sideways' and self.investment_timescale == 'very_aggressive':
                feedback['market_alignment'] = 'caution'
                feedback['issues_identified'].append("Sideways market limits aggressive trading opportunities")

            # Generate recommendations
            if feedback['overall_performance'] == 'poor':
                feedback['recommendations'].append("Consider switching to a more conservative timescale")
                feedback['recommendations'].append("Review confidence thresholds - may be too low")
                feedback['recommendations'].append("Focus on higher-quality setups with better risk-reward ratios")

            if feedback['risk_assessment'] == 'high':
                feedback['recommendations'].append("Reduce position sizes to manage volatility")
                feedback['recommendations'].append("Implement stricter stop-loss rules")
                feedback['recommendations'].append("Consider diversifying across more assets")

            if feedback['market_alignment'] == 'caution':
                feedback['recommendations'].append(f"Current {current_regime} market may not suit {self.investment_timescale} approach")
                feedback['recommendations'].append("Monitor market regime changes closely")
                feedback['recommendations'].append("Adjust confidence thresholds based on market conditions")

            if not feedback['issues_identified']:
                feedback['recommendations'].append("Strategy performing well - continue current approach")
                feedback['recommendations'].append("Monitor for market regime changes")
                feedback['recommendations'].append("Consider scaling up successful positions")

            return feedback

        except Exception as e:
            print(f"❌ Error generating performance feedback for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timescale': self.investment_timescale,
                'overall_performance': 'unknown',
                'issues_identified': ['Analysis error'],
                'recommendations': ['Review strategy parameters'],
                'risk_assessment': 'unknown',
                'market_alignment': 'unknown'
            }

    def update_market_analysis(self):
        """
        Update market data and analysis for all watchlist symbols
        """
        print(f"\n🚀 Starting live market analysis update...")
        print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Fetch fresh data
        self.market_data = self.get_live_data(self.watchlist)

        # Analyze each symbol
        self.analysis_results = {}
        self.recommendations = {}

        for symbol, df in self.market_data.items():
            result = self.analyze_symbol(symbol, df)
            if result:
                self.analysis_results[symbol] = result
                self.recommendations[symbol] = result['recommendation']

        self.last_update = datetime.now()
        print(f"✅ Analysis complete at {self.last_update.strftime('%H:%M:%S')}")

    def get_top_recommendations(self, top_n=10, action_filter=None):
        """
        Get top investment recommendations
        """
        if not self.recommendations:
            print("❌ No recommendations available. Run update_market_analysis() first.")
            return []

        # Filter recommendations
        filtered_recs = self.recommendations
        if action_filter:
            filtered_recs = {k: v for k, v in filtered_recs.items() if v['action'] == action_filter}

        # Sort by score
        sorted_recs = sorted(filtered_recs.items(),
                           key=lambda x: x[1]['score'],
                           reverse=True)

        return sorted_recs[:top_n]

    def get_market_overview(self):
        """
        Generate market overview with key statistics
        """
        if not self.analysis_results:
            return "No analysis data available"

        overview = {
            'total_symbols': len(self.analysis_results),
            'last_update': self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'Never',
            'recommendation_breakdown': {},
            'regime_breakdown': {},
            'average_metrics': {},
            'top_performers': [],
            'market_sentiment': 'neutral'
        }

        # Count recommendations
        rec_counts = {}
        regime_counts = {}
        scores = []
        win_rates = []
        returns_30d = []

        for symbol, result in self.analysis_results.items():
            # Recommendation counts
            action = result['recommendation']['action']
            rec_counts[action] = rec_counts.get(action, 0) + 1

            # Regime counts
            regime = result['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

            # Collect metrics
            scores.append(result['recommendation']['score'])
            win_rates.append(result['recommendation']['recent_win_rate'])
            returns_30d.append(result.get('price_change_30d', 0))

        overview['recommendation_breakdown'] = rec_counts
        overview['regime_breakdown'] = regime_counts
        overview['average_metrics'] = {
            'avg_score': np.mean(scores) if scores else 0,
            'avg_win_rate': np.mean(win_rates) if win_rates else 0,
            'avg_30d_return': np.mean(returns_30d) if returns_30d else 0
        }

        # Determine market sentiment
        buy_signals = rec_counts.get('STRONG_BUY', 0) + rec_counts.get('BUY', 0)
        sell_signals = rec_counts.get('SELL', 0) + rec_counts.get('WEAK_SELL', 0)

        if buy_signals > sell_signals * 1.5:
            overview['market_sentiment'] = 'bullish'
        elif sell_signals > buy_signals * 1.5:
            overview['market_sentiment'] = 'bearish'
        else:
            overview['market_sentiment'] = 'neutral'

        return overview

    def generate_investment_report(self):
        """
        Generate comprehensive investment report
        """
        if not self.analysis_results:
            return "No analysis data available. Please run update_market_analysis() first."

        overview = self.get_market_overview()
        top_buys = self.get_top_recommendations(5, 'STRONG_BUY')
        top_holds = self.get_top_recommendations(5, 'BUY')

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          FSOT 2.5 LIVE MARKET ADVISOR                      ║
║                          {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 MARKET OVERVIEW
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Symbols Analyzed: {overview['total_symbols']}
• Market Sentiment: {overview['market_sentiment'].upper()}
• Last Update: {overview['last_update']}

📈 RECOMMENDATION BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        for action, count in overview['recommendation_breakdown'].items():
            percentage = (count / overview['total_symbols']) * 100
            report += f"• {action}: {count} symbols ({percentage:.1f}%)\n"

        report += f"""
🏷️  MARKET REGIME BREAKDOWN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        for regime, count in overview['regime_breakdown'].items():
            percentage = (count / overview['total_symbols']) * 100
            report += f"• {regime.title()}: {count} symbols ({percentage:.1f}%)\n"

        report += f"""
📊 AVERAGE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Average Recommendation Score: {overview['average_metrics']['avg_score']:.1f}/100
• Average Win Rate: {overview['average_metrics']['avg_win_rate']:.1%}
• Average 30-Day Return: {overview['average_metrics']['avg_30d_return']:.1%}

🎯 INVESTMENT TIMESCALE: {self.timescale_settings['name'].upper()}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Strategy: {self.timescale_settings['description']}
• Holding Period: {self.timescale_settings['holding_period_days']} days
• Confidence Threshold: {self.timescale_settings['confidence_threshold']:.1f}
• Max Position Size: {self.timescale_settings['position_size_max']:.1f}x
• Risk per Trade: {self.timescale_settings['risk_per_trade_pct']:.1%}
• Max Trades/Day: {self.timescale_settings['max_trades_per_day']}
• Description: {self.timescale_settings['description_long']}

🎯 TOP STRONG BUY RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        if top_buys:
            for i, (symbol, rec) in enumerate(top_buys, 1):
                result = self.analysis_results[symbol]
                report += f"{i}. {symbol:6} | Score: {rec['score']:5.1f} | ${result['last_price']:8.2f} | {result['price_change_30d']:+.1%} 30d\n"
                report += f"   💡 {rec['rationale'][0]}\n\n"
        else:
            report += "No strong buy recommendations at this time.\n"

        report += f"""
💰 TOP BUY RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

        if top_holds:
            for i, (symbol, rec) in enumerate(top_holds, 1):
                result = self.analysis_results[symbol]
                report += f"{i}. {symbol:6} | Score: {rec['score']:5.1f} | ${result['last_price']:8.2f} | {result['price_change_30d']:+.1%} 30d\n"
                report += f"   💡 {rec['rationale'][0]}\n\n"
        else:
            report += "No buy recommendations at this time.\n"

        report += f"""
⚠️  IMPORTANT DISCLAIMER
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This analysis is for informational purposes only and should not be considered
as financial advice. Always conduct your own research and consult with a
qualified financial advisor before making investment decisions.

The FSOT 2.5 system uses advanced mathematical models and machine learning
to analyze market patterns, but past performance does not guarantee future
results. Market conditions can change rapidly.

═══════════════ END OF REPORT ════════════════════════════════════════════════
"""

        return report

def run_live_advisor_demo():
    """
    Demo function for the live market advisor with timescale comparison
    """
    print("🚀 Starting FSOT 2.5 Live Market Advisor Demo")
    print("=" * 60)

    timescales = ['conservative', 'moderate', 'aggressive', 'very_aggressive']

    for timescale in timescales:
        print(f"\n📊 ANALYZING WITH {timescale.upper()} TIMESCALE")
        print("=" * 50)

        # Initialize advisor with specific timescale
        demo_watchlist = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'SPY']
        advisor = LiveMarketAdvisor(watchlist=demo_watchlist, investment_timescale=timescale)

        # Update market analysis
        advisor.update_market_analysis()

        # Generate and display report
        report = advisor.generate_investment_report()
        print(report)

        # Show top recommendations
        print(f"\n🎯 TOP 3 RECOMMENDATIONS ({timescale.upper()}):")
        top_recs = advisor.get_top_recommendations(3)
        for i, (symbol, rec) in enumerate(top_recs, 1):
            result = advisor.analysis_results[symbol]
            print(f"{i}. {symbol}: {rec['action']} (Score: {rec['score']:.1f})")
            print(f"   Current Price: ${result['last_price']:.2f}")
            print(f"   30-Day Change: {result['price_change_30d']:+.1%}")
            print(f"   Rationale: {rec['rationale'][0]}")
            print()

        # Show performance feedback for top recommendation
        if top_recs:
            top_symbol = top_recs[0][0]
            feedback = advisor.generate_performance_feedback(
                top_symbol,
                advisor.analysis_results[top_symbol]['data'],
                advisor.analysis_results[top_symbol]['metrics']
            )

            print(f"📈 PERFORMANCE FEEDBACK FOR {top_symbol} ({timescale.upper()}):")
            print(f"Overall Performance: {feedback['overall_performance'].upper()}")
            print(f"Risk Assessment: {feedback['risk_assessment'].upper()}")
            print(f"Market Alignment: {feedback['market_alignment'].upper()}")

            if feedback['issues_identified']:
                print("Issues Identified:")
                for issue in feedback['issues_identified']:
                    print(f"  • {issue}")

            print("Recommendations:")
            for rec in feedback['recommendations']:
                print(f"  • {rec}")
            print()

def compare_timescales_demo():
    """
    Demo comparing different investment timescales
    """
    print("🔄 FSOT 2.5 Timescale Comparison Demo")
    print("=" * 50)

    demo_watchlist = ['AAPL', 'MSFT', 'NVDA', 'TSLA']
    timescales = ['conservative', 'moderate', 'aggressive']

    results = {}

    for timescale in timescales:
        print(f"\n📊 Testing {timescale.upper()} timescale...")
        advisor = LiveMarketAdvisor(watchlist=demo_watchlist, investment_timescale=timescale)
        advisor.update_market_analysis()

        # Count recommendations by type
        rec_counts = {}
        for symbol, result in advisor.analysis_results.items():
            action = result['recommendation']['action']
            rec_counts[action] = rec_counts.get(action, 0) + 1

        results[timescale] = {
            'recommendations': rec_counts,
            'avg_score': sum(r['recommendation']['score'] for r in advisor.analysis_results.values()) / len(advisor.analysis_results),
            'settings': advisor.timescale_settings
        }

        print(f"  {timescale.upper()}: {rec_counts}")
        print(f"  Average Score: {results[timescale]['avg_score']:.1f}")
        print(f"  Settings: {results[timescale]['settings']['description']}")

    print("\n💡 TIMESCALE COMPARISON INSIGHTS:")
    print("=" * 40)
    print("• Conservative: Fewer trades, higher quality signals")
    print("• Moderate: Balanced approach, most versatile")
    print("• Aggressive: More opportunities, higher risk")
    print("• Choose based on your risk tolerance and available time")

if __name__ == "__main__":
    # Run main demo
    run_live_advisor_demo()

    # Run comparison demo
    print("\n" + "="*60)
    compare_timescales_demo()
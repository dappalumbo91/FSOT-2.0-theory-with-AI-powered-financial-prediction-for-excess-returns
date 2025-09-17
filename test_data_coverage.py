"""
FSOT 2.5 Data Coverage Test
===========================

Quick test to verify comprehensive data collection is working
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import requests
import json
import os
import warnings

warnings.filterwarnings('ignore')

def test_data_coverage():
    """Test comprehensive data coverage across all asset classes"""

    print("🧪 FSOT 2.5 Data Coverage Test")
    print("=" * 40)

    # Test symbols from different categories
    test_symbols = {
        'stocks': ['AAPL', 'MSFT', 'GOOGL'],
        'crypto': ['BTC', 'ETH'],
        'forex': ['EURUSD=X', 'GBPUSD=X'],
        'commodities': ['GC=F', 'CL=F'],
        'indices': ['^GSPC', '^IXIC']
    }

    total_symbols = sum(len(symbols) for symbols in test_symbols.values())
    successful_fetches = 0

    print(f"🎯 Testing {total_symbols} symbols across {len(test_symbols)} categories")

    for category, symbols in test_symbols.items():
        print(f"\n📊 Testing {category.upper()}...")
        for symbol in symbols:
            try:
                print(f"  📈 Fetching {symbol}...")

                if category in ['stocks', 'forex', 'commodities', 'indices']:
                    # Test yfinance data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365)  # 1 year for quick test

                    data = yf.download(symbol, start=start_date, end=end_date, progress=False)

                    if not data.empty:
                        print(f"    ✅ {symbol}: {len(data)} data points, {len(data.columns)} columns")
                        successful_fetches += 1
                    else:
                        print(f"    ❌ {symbol}: No data received")

                elif category == 'crypto':
                    # Test crypto data
                    coin_id = symbol.lower()
                    if symbol == 'BTC':
                        coin_id = 'bitcoin'
                    elif symbol == 'ETH':
                        coin_id = 'ethereum'

                    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
                    params = {
                        'vs_currency': 'usd',
                        'days': 365,
                        'interval': 'daily'
                    }

                    response = requests.get(url, params=params, timeout=10)
                    response.raise_for_status()

                    data = response.json()
                    if 'prices' in data and data['prices']:
                        print(f"    ✅ {symbol}: {len(data['prices'])} data points")
                        successful_fetches += 1
                    else:
                        print(f"    ❌ {symbol}: No crypto data received")

            except Exception as e:
                print(f"    ❌ {symbol}: Error - {str(e)}")

    print("\n📊 Test Results:")
    print(f"  Total Symbols Tested: {total_symbols}")
    print(f"  Successful Fetches: {successful_fetches}")
    print(f"  Success Rate: {successful_fetches/total_symbols*100:.1f}%")

    if successful_fetches > 0:
        print("✅ Data collection is working! You can now run the comprehensive framework.")
        print("💡 Run: python fsot_comprehensive_data.py")
    else:
        print("❌ Data collection issues detected. Check your internet connection and API access.")


if __name__ == "__main__":
    test_data_coverage()
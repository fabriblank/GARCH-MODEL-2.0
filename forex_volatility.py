#!/usr/bin/env python3
"""
GARCH-MODEL-2.0: Forex Volatility Filter
Predicts if trading day will be good based on GARCH + VIX
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("📱 GARCH-MODEL-2.0: Forex Daily Volatility Filter")
print("=" * 70)

def download_forex_data():
    """Download forex pairs and VIX data"""
    print("\n📊 Downloading data...")
    
    # Forex major pairs
    forex_pairs = {
        'EUR/USD': 'EURUSD=X',
        'GBP/USD': 'GBPUSD=X', 
        'USD/JPY': 'JPY=X',
        'USD/CHF': 'CHF=X',
        'AUD/USD': 'AUDUSD=X',
        'USD/CAD': 'CAD=X',
        'NZD/USD': 'NZDUSD=X'
    }
    
    # Get last 90 days data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    all_data = {}
    
    for pair_name, ticker in forex_pairs.items():
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if not data.empty:
                all_data[pair_name] = data['Close']
                print(f"✅ {pair_name}: {len(data)} days")
            else:
                print(f"❌ {pair_name}: No data")
        except:
            print(f"⚠️  {pair_name}: Download failed")
    
    # Download VIX
    vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
    
    return all_data, vix_data['Close'] if not vix_data.empty else None

def calculate_returns(price_series):
    """Calculate daily percentage returns"""
    returns = price_series.pct_change().dropna() * 100
    return returns

def simple_garch_forecast(returns):
    """
    Simplified GARCH(1,1) forecast
    Returns expected volatility for next day
    """
    if len(returns) < 30:
        return returns.std()
    
    # GARCH(1,1) simplified formula: σ² = ω + α*r² + β*σ²
    omega = 0.05  # Long-run variance
    alpha = 0.1   # Reaction to shocks
    beta = 0.85   # Persistence
    
    # Initialize
    variance = returns.var()
    last_return = returns.iloc[-1]
    
    # One-step forecast
    forecast_variance = omega + alpha * (last_return**2) + beta * variance
    
    return np.sqrt(forecast_variance)

def analyze_pair(pair_name, prices, vix_series):
    """Analyze one forex pair"""
    print(f"\n{'='*50}")
    print(f"📈 {pair_name}")
    print(f"{'='*50}")
    
    returns = calculate_returns(prices)
    
    if len(returns) < 20:
        print("⚠️  Not enough data for analysis")
        return None
    
    # 1. Current stats
    current_vol = returns.std()
    avg_return = returns.mean()
    high_vol_days = len(returns[abs(returns) > 1.0]) / len(returns) * 100
    
    print(f"📊 Current volatility: {current_vol:.3f}%")
    print(f"📈 Average daily move: {avg_return:.3f}%")
    print(f"🔥 High vol days (>1%): {high_vol_days:.1f}%")
    
    # 2. GARCH forecast
    garch_vol = simple_garch_forecast(returns)
    print(f"🔮 GARCH forecast volatility: {garch_vol:.3f}%")
    
    # 3. VIX correlation if available
    if vix_series is not None and len(vix_series) > 20:
        # Align dates
        common_dates = returns.index.intersection(vix_series.index)
        if len(common_dates) > 10:
            corr = returns.loc[common_dates].abs().corr(vix_series.loc[common_dates])
            print(f"📊 VIX correlation: {corr:.3f}")
    
    # 4. Trading recommendation
    if garch_vol > 0.7:  # Threshold for "good volatility day"
        recommendation = "✅ GOOD TRADING DAY"
        reason = "Expected high volatility"
    elif garch_vol > 0.4:
        recommendation = "⚠️  MODERATE DAY"
        reason = "Medium expected volatility"
    else:
        recommendation = "❌ LOW VOLATILITY DAY"
        reason = "Expected calm market"
    
    print(f"\n🎯 RECOMMENDATION: {recommendation}")
    print(f"📝 Reason: {reason}")
    
    return {
        'pair': pair_name,
        'current_vol': current_vol,
        'forecast_vol': garch_vol,
        'recommendation': recommendation
    }

def main():
    """Main function"""
    print("\n🚀 Starting analysis...")
    
    # Download data
    forex_data, vix_data = download_forex_data()
    
    if not forex_data:
        print("❌ Failed to download forex data")
        return
    
    # Analyze each pair
    results = []
    
    for pair_name, prices in forex_data.items():
        result = analyze_pair(pair_name, prices, vix_data)
        if result:
            results.append(result)
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 SUMMARY: Best pairs for tomorrow")
    print("=" * 70)
    
    good_pairs = [r for r in results if "GOOD" in r['recommendation']]
    
    if good_pairs:
        print("\n🎯 RECOMMENDED PAIRS (High volatility expected):")
        for r in good_pairs:
            print(f"  • {r['pair']}: Forecast vol = {r['forecast_vol']:.3f}%")
    else:
        print("\n⚠️  No pairs show high volatility expectations")
        print("   Consider waiting for better market conditions")
    
    print("\n" + "=" * 70)
    print("📝 NOTES:")
    print("- Based on GARCH(1,1) volatility forecasting")
    print("- VIX correlation considered where available")
    print("- GOOD = Expected daily range > 0.7%")
    print("- Updates daily, run before trading session")
    print("=" * 70)

if __name__ == "__main__":
    main()

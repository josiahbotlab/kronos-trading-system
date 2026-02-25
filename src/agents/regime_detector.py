"""
Market Regime Detector

Detects market conditions to recommend the best trading strategy:
- TREND_UP: Price > 50 SMA > 200 SMA, ADX > 25 → Buy & Hold
- TREND_DOWN: Price < 50 SMA < 200 SMA, ADX > 25 → Mean Reversion
- RANGING: ADX < 20, price near SMAs → Breakout
- VOLATILE: ATR > 2x average → Breakout with wider stops

Usage:
    from src.agents.regime_detector import detect_regime, get_regime_recommendation

    regime = detect_regime('SPY')
    print(f"SPY is in {regime} mode")

    # Or run directly:
    python src/agents/regime_detector.py
    python src/agents/regime_detector.py --symbols SPY,TSLA,AMD
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Literal, Dict, Any, Tuple

import numpy as np
import pandas as pd
from termcolor import cprint
from dotenv import load_dotenv

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Regime detection parameters
SMA_FAST = 50
SMA_SLOW = 200
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25    # Above this = trending
ADX_RANGE_THRESHOLD = 20    # Below this = ranging
ATR_PERIOD = 14
ATR_VOLATILITY_MULT = 2.0   # ATR > 2x average = high volatility

# Regime types
RegimeType = Literal['TREND_UP', 'TREND_DOWN', 'RANGING', 'VOLATILE']

# Strategy recommendations
STRATEGY_MAP = {
    'TREND_UP': {
        'strategy': 'Buy & Hold',
        'description': 'Strong uptrend - hold positions, avoid counter-trend trades',
        'color': 'green'
    },
    'TREND_DOWN': {
        'strategy': 'Mean Reversion',
        'description': 'Downtrend with bounces - buy oversold dips for quick exits',
        'color': 'red'
    },
    'RANGING': {
        'strategy': 'Breakout',
        'description': 'Range-bound market - trade breakouts from consolidation',
        'color': 'yellow'
    },
    'VOLATILE': {
        'strategy': 'Breakout (Wide Stops)',
        'description': 'High volatility - use breakouts with 2x normal stop distance',
        'color': 'magenta'
    }
}


def get_alpaca_api():
    """Get Alpaca API instance."""
    if not ALPACA_AVAILABLE:
        return None
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return None

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')


def fetch_daily_data(symbol: str, days: int = 250) -> pd.DataFrame:
    """Fetch daily OHLCV data from Alpaca."""
    api = get_alpaca_api()
    if not api:
        raise RuntimeError("Alpaca API not available")

    # Use data from 15+ days ago to avoid SIP restrictions
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days + 50)  # Extra buffer for SMA calculation

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    bars = api.get_bars(
        symbol,
        tradeapi.TimeFrame.Day,
        start=start_str,
        end=end_str,
        limit=days + 50,
        feed='iex'
    ).df

    if bars.empty:
        raise ValueError(f"No data returned for {symbol}")

    bars = bars.reset_index()
    bars.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'trade_count', 'vwap']

    return bars[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return prices.rolling(window=period).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ATR_PERIOD) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = ADX_PERIOD) -> pd.Series:
    """
    Calculate Average Directional Index (ADX).
    ADX measures trend strength regardless of direction.
    """
    # Calculate +DM and -DM
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Calculate ATR
    atr = calculate_atr(high, low, close, period)

    # Calculate smoothed +DI and -DI
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    # Calculate DX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # Calculate ADX (smoothed DX)
    adx = dx.rolling(window=period).mean()

    return adx


def analyze_regime(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze market regime from price data.

    Returns dict with:
    - regime: TREND_UP, TREND_DOWN, RANGING, or VOLATILE
    - indicators: SMA values, ADX, ATR, etc.
    - confidence: how strong the regime signal is
    """
    # Calculate indicators
    close = data['Close']
    high = data['High']
    low = data['Low']

    sma_50 = calculate_sma(close, SMA_FAST)
    sma_200 = calculate_sma(close, SMA_SLOW)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)

    # Get latest values
    current_price = close.iloc[-1]
    current_sma_50 = sma_50.iloc[-1]
    current_sma_200 = sma_200.iloc[-1]
    current_adx = adx.iloc[-1]
    current_atr = atr.iloc[-1]
    avg_atr = atr.iloc[-20:].mean()  # 20-day average ATR

    # Calculate regime scores
    indicators = {
        'price': current_price,
        'sma_50': current_sma_50,
        'sma_200': current_sma_200,
        'adx': current_adx,
        'atr': current_atr,
        'avg_atr': avg_atr,
        'atr_ratio': current_atr / avg_atr if avg_atr > 0 else 1.0,
        'price_vs_sma50': ((current_price - current_sma_50) / current_sma_50 * 100) if current_sma_50 > 0 else 0,
        'sma50_vs_sma200': ((current_sma_50 - current_sma_200) / current_sma_200 * 100) if current_sma_200 > 0 else 0
    }

    # Determine regime
    regime = None
    confidence = 0
    reasons = []

    # Check for HIGH VOLATILITY first (overrides other regimes)
    if indicators['atr_ratio'] > ATR_VOLATILITY_MULT:
        regime = 'VOLATILE'
        confidence = min(100, int((indicators['atr_ratio'] - 1) * 50))
        reasons.append(f"ATR {indicators['atr_ratio']:.1f}x above average")

    # Check for TRENDING UP
    elif (current_price > current_sma_50 and
          current_sma_50 > current_sma_200 and
          current_adx > ADX_TREND_THRESHOLD):
        regime = 'TREND_UP'
        confidence = min(100, int(current_adx * 2))
        reasons.append(f"Price > SMA50 > SMA200")
        reasons.append(f"ADX = {current_adx:.1f} (strong trend)")

    # Check for TRENDING DOWN
    elif (current_price < current_sma_50 and
          current_sma_50 < current_sma_200 and
          current_adx > ADX_TREND_THRESHOLD):
        regime = 'TREND_DOWN'
        confidence = min(100, int(current_adx * 2))
        reasons.append(f"Price < SMA50 < SMA200")
        reasons.append(f"ADX = {current_adx:.1f} (strong trend)")

    # Check for RANGING
    elif current_adx < ADX_RANGE_THRESHOLD:
        regime = 'RANGING'
        confidence = min(100, int((ADX_RANGE_THRESHOLD - current_adx) * 5))
        reasons.append(f"ADX = {current_adx:.1f} (weak trend)")
        reasons.append("Price oscillating around moving averages")

    # Default to moderate trend or transition
    else:
        # Weak trend or transition period
        if current_price > current_sma_50:
            regime = 'TREND_UP'
            confidence = 30
            reasons.append("Weak uptrend (price > SMA50)")
        else:
            regime = 'TREND_DOWN'
            confidence = 30
            reasons.append("Weak downtrend (price < SMA50)")
        reasons.append(f"ADX = {current_adx:.1f} (moderate)")

    return {
        'regime': regime,
        'confidence': confidence,
        'reasons': reasons,
        'indicators': indicators,
        'recommendation': STRATEGY_MAP[regime]
    }


def detect_regime(symbol: str) -> RegimeType:
    """
    Detect the current market regime for a symbol.

    Args:
        symbol: Stock/ETF symbol

    Returns:
        RegimeType: 'TREND_UP', 'TREND_DOWN', 'RANGING', or 'VOLATILE'
    """
    data = fetch_daily_data(symbol)
    analysis = analyze_regime(data)
    return analysis['regime']


def detect_regime_verbose(symbol: str) -> Dict[str, Any]:
    """
    Detect regime with full analysis details.

    Returns dict with regime, confidence, reasons, indicators, and recommendation.
    """
    data = fetch_daily_data(symbol)
    return analyze_regime(data)


def get_regime_recommendation(regime: RegimeType) -> Dict[str, str]:
    """Get strategy recommendation for a regime."""
    return STRATEGY_MAP.get(regime, STRATEGY_MAP['RANGING'])


def print_regime_analysis(symbol: str, analysis: Dict[str, Any]):
    """Pretty print regime analysis."""
    regime = analysis['regime']
    confidence = analysis['confidence']
    rec = analysis['recommendation']
    ind = analysis['indicators']

    color = rec['color']

    cprint(f"\n{'='*60}", color)
    cprint(f"  {symbol} - {regime}", color, attrs=['bold'])
    cprint(f"{'='*60}", color)

    cprint(f"\n  Confidence: {confidence}%", "white")
    for reason in analysis['reasons']:
        cprint(f"  - {reason}", "white")

    cprint(f"\n  INDICATORS:", "cyan")
    cprint(f"    Price:      ${ind['price']:.2f}", "white")
    cprint(f"    SMA(50):    ${ind['sma_50']:.2f} ({ind['price_vs_sma50']:+.1f}% from price)", "white")
    cprint(f"    SMA(200):   ${ind['sma_200']:.2f}", "white")
    cprint(f"    ADX:        {ind['adx']:.1f}", "white")
    cprint(f"    ATR:        ${ind['atr']:.2f} ({ind['atr_ratio']:.1f}x avg)", "white")

    cprint(f"\n  RECOMMENDATION:", "yellow")
    cprint(f"    Strategy:   {rec['strategy']}", "yellow", attrs=['bold'])
    cprint(f"    {rec['description']}", "white")

    cprint(f"\n{'='*60}\n", color)


def main():
    """Main entry point - analyze multiple symbols."""
    parser = argparse.ArgumentParser(description="Market Regime Detector")
    parser.add_argument(
        "--symbols",
        type=str,
        default="SPY,TSLA,AMD",
        help="Comma-separated list of symbols to analyze (default: SPY,TSLA,AMD)"
    )
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    cprint("\n" + "=" * 60, "cyan")
    cprint("  MARKET REGIME DETECTOR", "cyan", attrs=['bold'])
    cprint("=" * 60, "cyan")
    cprint(f"  Analyzing: {', '.join(symbols)}", "white")
    cprint(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", "white")

    results = []

    for symbol in symbols:
        try:
            analysis = detect_regime_verbose(symbol)
            print_regime_analysis(symbol, analysis)
            results.append({
                'symbol': symbol,
                'regime': analysis['regime'],
                'confidence': analysis['confidence'],
                'strategy': analysis['recommendation']['strategy']
            })
        except Exception as e:
            cprint(f"\nError analyzing {symbol}: {e}", "red")

    # Summary table
    if results:
        cprint("\n" + "=" * 60, "cyan")
        cprint("  SUMMARY", "cyan", attrs=['bold'])
        cprint("=" * 60, "cyan")
        cprint(f"\n  {'Symbol':<8} {'Regime':<12} {'Confidence':<12} {'Strategy'}", "white", attrs=['bold'])
        cprint(f"  {'-'*8} {'-'*12} {'-'*12} {'-'*20}", "white")

        for r in results:
            regime_color = STRATEGY_MAP[r['regime']]['color']
            cprint(f"  {r['symbol']:<8} {r['regime']:<12} {r['confidence']:>8}%    {r['strategy']}", regime_color)

        cprint(f"\n{'='*60}\n", "cyan")


if __name__ == "__main__":
    main()

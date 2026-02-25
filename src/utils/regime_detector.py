"""
Market Regime Detector (SPY-based)

Detects overall market regime using SPY hourly bars from Alpaca.
Used by stock bots to gate entries based on regime performance.

Returns: trending_up, trending_down, ranging, volatile, unknown

Usage:
    from src.utils.regime_detector import detect_market_regime
    regime = detect_market_regime()  # uses default Alpaca API
    regime = detect_market_regime(api)  # pass existing API instance
"""

import os
import time
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 5-minute cache to avoid redundant API calls during multi-symbol scan
_cache = {"regime": None, "ts": 0}
_CACHE_TTL = 300  # seconds


def _get_api(api=None):
    """Get or create Alpaca API instance."""
    if api is not None:
        return api
    try:
        from src.utils.order_utils import get_api
        return get_api()
    except Exception:
        pass
    try:
        import alpaca_trade_api as tradeapi
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / '.env')
        key = os.getenv("ALPACA_API_KEY")
        secret = os.getenv("ALPACA_SECRET_KEY")
        paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        return tradeapi.REST(key, secret, base_url, api_version='v2')
    except Exception:
        return None


def _fetch_spy_bars(api, count=60):
    """Fetch SPY hourly bars from Alpaca."""
    import alpaca_trade_api as tradeapi

    cal_days = max((count // 7) * 2 + 4, 14)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=cal_days)

    bars = api.get_bars(
        'SPY',
        tradeapi.TimeFrame.Hour,
        start=start.isoformat(),
        end=end.isoformat(),
        feed='iex'
    ).df

    if bars.empty:
        return None

    return bars.tail(count)


def _calc_sma(values, period):
    """Simple moving average of last N values."""
    if len(values) < period:
        return np.mean(values) if len(values) > 0 else 0
    return float(np.mean(values[-period:]))


def _calc_atr(highs, lows, closes, period=14):
    """Average True Range."""
    if len(highs) < period + 1:
        return 0
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1])
        )
        trs.append(tr)
    if len(trs) < period:
        return np.mean(trs) if trs else 0
    return float(np.mean(trs[-period:]))


def detect_market_regime(api=None) -> str:
    """
    Detect current market regime from SPY hourly bars.

    Uses SMA(20)/SMA(50) crossover + ATR(14)/ATR(50) volatility ratio.

    Returns: trending_up, trending_down, ranging, volatile, unknown
    """
    # Check cache
    now = time.time()
    if _cache["regime"] and (now - _cache["ts"]) < _CACHE_TTL:
        return _cache["regime"]

    try:
        api_instance = _get_api(api)
        if api_instance is None:
            return "unknown"

        bars = _fetch_spy_bars(api_instance, count=60)
        if bars is None or len(bars) < 50:
            return "unknown"

        closes = [float(v) for v in bars['close'].values]
        highs = [float(v) for v in bars['high'].values]
        lows = [float(v) for v in bars['low'].values]

        sma20 = _calc_sma(closes, 20)
        sma50 = _calc_sma(closes, 50)
        atr14 = _calc_atr(highs, lows, closes, 14)
        atr50 = _calc_atr(highs, lows, closes, 50)

        price = closes[-1]
        vol_ratio = atr14 / atr50 if atr50 > 0 else 1.0

        # Determine regime
        if vol_ratio > 1.8:
            regime = "volatile"
        elif price > sma20 > sma50:
            regime = "trending_up"
        elif price < sma20 < sma50:
            regime = "trending_down"
        else:
            regime = "ranging"

        _cache["regime"] = regime
        _cache["ts"] = now
        return regime

    except Exception:
        return "unknown"


if __name__ == "__main__":
    regime = detect_market_regime()
    print(f"Current market regime: {regime}")

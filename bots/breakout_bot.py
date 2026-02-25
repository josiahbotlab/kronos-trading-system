"""
Breakout Bot — MoonDev Methodology

Buy when price breaks above 24-hour high (resistance breakout).
Uses wider stop loss to ride breakouts (from RBI optimization).

Entry: Price > 24h rolling high + 0.1% buffer
Exit:  TP (4%) or SL (12%) via bracket order

Parameters (from RBI backtest optimization):
    TP: 4%   |  SL: 12%  (wide SL to survive volatility)

Note: MoonDev found that breakout strategies need wide SL to work
on crypto; same principle applies to volatile stocks like TSLA.

Usage:
    python bots/breakout_bot.py --dry-run --symbols TSLA,AMD
    python bots/breakout_bot.py --symbols TSLA --size 500
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bots.base_bot import BaseBot


# ─────────────────────────────────────────────────────────────
# STRATEGY PARAMETERS
# ─────────────────────────────────────────────────────────────

LOOKBACK_HOURS = 24      # Rolling high window
BUFFER_PCT = 0.001       # 0.1% buffer above high

TP_PCT = 4.0             # Take profit (RBI optimized from 6%)
SL_PCT = 12.0            # Wide stop loss (RBI: breakouts need room)


class BreakoutBot(BaseBot):
    """
    Breakout bot — buys when price breaks above 24h resistance.

    Uses wide SL (12%) because breakouts need room to breathe.
    Backtested winner on TSLA (+15% on 1-year data).
    """

    BOT_NAME = "breakout_bot"
    STRATEGY = "BREAKOUT"
    DEFAULT_SYMBOLS = ['AMD', 'NVDA', 'GOOGL', 'META', 'AAPL', 'MSFT', 'AMZN', 'TSLA', 'QQQ']
    TP_PCT = TP_PCT
    SL_PCT = SL_PCT

    def check_signal(self, symbol, price, prices, candles):
        """
        Check for bullish breakout entry (LONG only).

        Condition:
        - Price > 24h rolling high + 0.1% buffer → LONG breakout

        Returns:
            (should_enter, metadata)
        """
        if len(candles) < LOOKBACK_HOURS:
            self._log(f"{symbol}: Need {LOOKBACK_HOURS} candles, have {len(candles)}", "yellow")
            return False, {}

        recent = candles[-LOOKBACK_HOURS:]
        high_24h = max(c['high'] for c in recent)
        low_24h = min(c['low'] for c in recent)

        buffer = high_24h * BUFFER_PCT

        should_enter = price > high_24h + buffer

        signal_str = "SIGNAL (LONG)" if should_enter else "NO SIGNAL"
        self._log(
            f"{symbol} @ ${price:.2f} | "
            f"24h High=${high_24h:.2f} Low=${low_24h:.2f} | "
            f"→ {signal_str}",
            "green" if should_enter else "white"
        )

        metadata = {
            'high_24h': high_24h,
            'low_24h': low_24h,
            'direction': 'LONG',
            'signal_strength': '1/1' if should_enter else '0/1',
            'reasoning': (
                f"Bullish breakout: Price ${price:.2f} vs 24h high "
                f"${high_24h:.2f} ({TP_PCT}% TP / {SL_PCT}% SL)"
            ) if should_enter else None,
        }

        return should_enter, metadata


if __name__ == '__main__':
    BreakoutBot.main()

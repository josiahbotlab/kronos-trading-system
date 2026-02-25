"""
MACD Bullish Crossover Bot — MoonDev Methodology

Buy on MACD bullish crossover (MACD line crosses above signal line).
Backtested: +58% on AMD over 1 year.

Entry: MACD(12,26,9) line crosses above signal line
Exit:  TP (5%) or SL (3%) via bracket order

Parameters (updated from RBI ideas.txt):
    TP: 5%  |  SL: 3%

Usage:
    python bots/macd_bot.py --dry-run --symbols AMD,NVDA
    python bots/macd_bot.py --symbols AMD --size 500
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bots.base_bot import BaseBot, calc_macd


# ─────────────────────────────────────────────────────────────
# STRATEGY PARAMETERS
# ─────────────────────────────────────────────────────────────

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

TP_PCT = 5.0             # Take profit
SL_PCT = 3.0             # Stop loss


class MACDBot(BaseBot):
    """
    MACD bullish crossover bot — buys when MACD crosses above signal.

    Looks for momentum shift from bearish to bullish.
    Works well on trending stocks like AMD.
    """

    BOT_NAME = "macd_bot"
    STRATEGY = "MACD"
    DEFAULT_SYMBOLS = ['AMD', 'NVDA', 'TSLA', 'AAPL']
    TP_PCT = TP_PCT
    SL_PCT = SL_PCT

    def check_signal(self, symbol, price, prices, candles):
        """
        Check for MACD bullish crossover.

        Condition:
        - Previous bar: MACD <= Signal
        - Current bar: MACD > Signal
        → Bullish crossover detected

        Returns:
            (should_enter, metadata)
        """
        min_bars = MACD_SLOW + MACD_SIGNAL + 2
        if len(prices) < min_bars:
            self._log(f"{symbol}: Need {min_bars} bars, have {len(prices)}", "yellow")
            return False, {}

        # Current MACD values
        macd_curr, signal_curr, hist_curr = calc_macd(prices, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        # Previous MACD values (one bar back)
        macd_prev, signal_prev, hist_prev = calc_macd(prices[:-1], MACD_FAST, MACD_SLOW, MACD_SIGNAL)

        if macd_curr is None or macd_prev is None:
            self._log(f"{symbol}: MACD calculation failed", "yellow")
            return False, {}

        # Bullish crossover: MACD was <= signal, now MACD > signal
        bullish_cross = (macd_prev <= signal_prev) and (macd_curr > signal_curr)

        signal_str = "SIGNAL (bullish cross)" if bullish_cross else "NO SIGNAL"
        self._log(
            f"{symbol} @ ${price:.2f} | "
            f"MACD={macd_curr:.4f} Signal={signal_curr:.4f} Hist={hist_curr:.4f} | "
            f"Prev MACD={macd_prev:.4f} Signal={signal_prev:.4f} | "
            f"→ {signal_str}",
            "green" if bullish_cross else "white"
        )

        metadata = {
            'macd': macd_curr,
            'signal': signal_curr,
            'histogram': hist_curr,
            'macd_prev': macd_prev,
            'signal_prev': signal_prev,
            'signal_strength': '1/1' if bullish_cross else '0/1',
            'direction': 'LONG',
            'reasoning': (
                f"MACD bullish crossover: MACD={macd_curr:.4f} > Signal={signal_curr:.4f} "
                f"({TP_PCT}% TP / {SL_PCT}% SL)"
            ) if bullish_cross else None,
        }

        return bullish_cross, metadata


if __name__ == '__main__':
    MACDBot.main()

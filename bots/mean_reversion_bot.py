"""
Mean Reversion Bot — MoonDev Methodology

Buy when price is oversold (RSI < 30) AND below SMA(20).
Exit when price reverts to the mean or stop loss hits.

Entry: RSI(14) < 30 AND price > 2% below SMA(20)
Exit:  TP (4%) or SL (2%) via bracket order

Parameters (from backtesting + RBI):
    TP: 4%  |  SL: 2%

Usage:
    python bots/mean_reversion_bot.py --dry-run --symbols PLTR,PYPL
    python bots/mean_reversion_bot.py --symbols AMD,META --size 500
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bots.base_bot import BaseBot, calc_rsi, calc_sma


# ─────────────────────────────────────────────────────────────
# STRATEGY PARAMETERS
# ─────────────────────────────────────────────────────────────

SMA_PERIOD = 20
RSI_PERIOD = 14
RSI_OVERSOLD = 30        # RSI threshold for oversold
SMA_DEVIATION = 0.02     # Price must be > 2% below SMA

TP_PCT = 4.0             # Take profit
SL_PCT = 2.0             # Stop loss


class MeanReversionBot(BaseBot):
    """
    Mean reversion bot — buys oversold conditions and targets reversion to SMA.

    Best on range-bound and trending-down symbols where prices snap back.
    """

    BOT_NAME = "mean_reversion_bot"
    STRATEGY = "MEAN_REV"
    DEFAULT_SYMBOLS = ['PLTR', 'PYPL', 'AMD', 'META', 'INTC']
    TP_PCT = TP_PCT
    SL_PCT = SL_PCT

    def check_signal(self, symbol, price, prices, candles):
        """
        Check for mean reversion entry.

        Conditions:
        1. RSI(14) < 30 — oversold
        2. Price > 2% below SMA(20) — extended from mean

        Returns:
            (should_enter, metadata)
        """
        min_bars = max(RSI_PERIOD + 1, SMA_PERIOD)
        if len(prices) < min_bars:
            self._log(f"{symbol}: Insufficient data ({len(prices)}/{min_bars})", "yellow")
            return False, {}

        rsi = calc_rsi(prices, RSI_PERIOD)
        sma = calc_sma(prices, SMA_PERIOD)
        sma_dev = (sma - price) / sma if sma > 0 else 0

        rsi_oversold = rsi < RSI_OVERSOLD
        below_sma = sma_dev >= SMA_DEVIATION

        should_enter = rsi_oversold and below_sma

        rsi_mark = "Y" if rsi_oversold else "N"
        sma_mark = "Y" if below_sma else "N"
        signal_str = "SIGNAL" if should_enter else "NO SIGNAL"

        self._log(
            f"{symbol} @ ${price:.2f} | "
            f"RSI={rsi:.1f}[{rsi_mark}] "
            f"SMA20=${sma:.2f} dev={sma_dev*100:.1f}%[{sma_mark}] "
            f"→ {signal_str}",
            "green" if should_enter else "white"
        )

        conditions_met = sum([rsi_oversold, below_sma])
        metadata = {
            'rsi': rsi,
            'sma_20': sma,
            'sma_deviation_pct': sma_dev * 100,
            'signal_strength': f"{conditions_met}/2",
            'direction': 'LONG',
            'reasoning': (
                f"Oversold: RSI={rsi:.0f}<{RSI_OVERSOLD}, "
                f"Price {sma_dev*100:.1f}% below SMA20 "
                f"({TP_PCT}% TP / {SL_PCT}% SL)"
            ),
        }

        return should_enter, metadata


if __name__ == '__main__':
    MeanReversionBot.main()

"""
Bollinger Band Bounce Bot — MoonDev Methodology

Buy when price touches lower Bollinger Band with RSI confirmation.
Double confirmation: oversold + at statistical extreme.

Entry: Price <= Lower BB(20, 2) AND RSI(14) < 30
Exit:  TP (4%) or SL (2%) via bracket order

Parameters (from RBI ideas.txt):
    TP: 4%  |  SL: 2%

Usage:
    python bots/bb_bounce_bot.py --dry-run --symbols META,AMD
    python bots/bb_bounce_bot.py --symbols META --size 500
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bots.base_bot import BaseBot, calc_rsi, calc_bollinger_bands


# ─────────────────────────────────────────────────────────────
# STRATEGY PARAMETERS
# ─────────────────────────────────────────────────────────────

BB_PERIOD = 20
BB_STD = 2.0
RSI_PERIOD = 14
RSI_OVERSOLD = 30        # RSI must be < 30 for confirmation

TP_PCT = 4.0             # Take profit
SL_PCT = 2.0             # Stop loss


class BBBounceBot(BaseBot):
    """
    BB Bounce bot — buys at lower Bollinger Band with RSI oversold confirmation.

    Requires both statistical extreme (BB) and momentum confirmation (RSI).
    Higher-quality signals but fires less often.
    """

    BOT_NAME = "bb_bounce_bot"
    STRATEGY = "BB_BOUNCE"
    DEFAULT_SYMBOLS = ['META', 'AMD', 'NVDA', 'AAPL', 'PLTR']
    TP_PCT = TP_PCT
    SL_PCT = SL_PCT

    def check_signal(self, symbol, price, prices, candles):
        """
        Check for BB lower bounce with RSI confirmation.

        Conditions:
        1. Price <= Lower BB(20, 2) — at statistical extreme
        2. RSI(14) < 30 — oversold confirmation

        Both must pass.

        Returns:
            (should_enter, metadata)
        """
        min_bars = max(BB_PERIOD, RSI_PERIOD + 1)
        if len(prices) < min_bars:
            self._log(f"{symbol}: Need {min_bars} bars, have {len(prices)}", "yellow")
            return False, {}

        upper, middle, lower = calc_bollinger_bands(prices, BB_PERIOD, BB_STD)
        if lower is None:
            self._log(f"{symbol}: BB calculation failed", "yellow")
            return False, {}

        rsi = calc_rsi(prices, RSI_PERIOD)

        at_lower_band = price <= lower
        rsi_oversold = rsi < RSI_OVERSOLD

        should_enter = at_lower_band and rsi_oversold

        bb_mark = "Y" if at_lower_band else "N"
        rsi_mark = "Y" if rsi_oversold else "N"
        conditions_met = sum([at_lower_band, rsi_oversold])
        signal_str = "SIGNAL" if should_enter else "NO SIGNAL"

        self._log(
            f"{symbol} @ ${price:.2f} | "
            f"BB: Upper=${upper:.2f} Mid=${middle:.2f} Lower=${lower:.2f} [{bb_mark}] "
            f"RSI={rsi:.1f}[{rsi_mark}] "
            f"→ {conditions_met}/2 {signal_str}",
            "green" if should_enter else "white"
        )

        metadata = {
            'bb_upper': upper,
            'bb_middle': middle,
            'bb_lower': lower,
            'rsi': rsi,
            'signal_strength': f"{conditions_met}/2",
            'direction': 'LONG',
            'reasoning': (
                f"BB Bounce: Price ${price:.2f} <= Lower BB ${lower:.2f}, "
                f"RSI={rsi:.0f}<{RSI_OVERSOLD} "
                f"({TP_PCT}% TP / {SL_PCT}% SL)"
            ) if should_enter else None,
        }

        return should_enter, metadata


if __name__ == '__main__':
    BBBounceBot.main()

"""
Momentum Pullback Bot — MoonDev Methodology

Trades WITH the trend using pullback entries.
Best performer in live trading: 12 trades, 60% win rate.

Entry Conditions (3-of-4 must pass, uptrend required):
1. Price > SMA(20) — confirms uptrend
2. RSI(14) between 40-80 — momentum zone
3. Price near EMA(9) within 3% — pullback entry
4. Volume >= average (disabled by default on IEX feed)

Parameters (from backtesting):
    TP: 6%  |  SL: 3%  (2:1 R/R)

Usage:
    python bots/momentum_bot.py --dry-run --symbols AMD,NVDA
    python bots/momentum_bot.py --symbols TSLA,AAPL,MSFT --size 500
    python bots/momentum_bot.py --duration 60  # run for 1 hour
"""

import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from bots.base_bot import BaseBot, calc_rsi, calc_sma, calc_ema
from termcolor import cprint
import numpy as np


# ─────────────────────────────────────────────────────────────
# STRATEGY PARAMETERS
# ─────────────────────────────────────────────────────────────

# Indicators
SMA_PERIOD = 20
EMA_PERIOD = 9
RSI_PERIOD = 14
RSI_MIN = 40            # Not oversold
RSI_MAX = 80            # Not extremely overbought
EMA_TOLERANCE = 0.03    # 3% tolerance for "near EMA"
VOLUME_MULT = 0.0       # 0 = disabled (IEX feed unreliable for volume)

# Risk
TP_PCT = 6.0            # Take profit
SL_PCT = 3.0            # Stop loss (2:1 R/R)


class MomentumBot(BaseBot):
    """
    Momentum pullback bot — trades pullbacks to 9 EMA in confirmed uptrends.

    MoonDev philosophy: "Trade what the market gives you."
    In bull markets, buy the dip to the moving average.
    """

    BOT_NAME = "momentum_bot"
    STRATEGY = "MOMENTUM"
    DEFAULT_SYMBOLS = ['NVDA', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'AMD']
    TP_PCT = TP_PCT
    SL_PCT = SL_PCT

    def check_signal(self, symbol, price, prices, candles):
        """
        Check for momentum pullback entry signal.

        Conditions:
        1. Price > SMA(20) — uptrend confirmed
        2. RSI(14) in 40-80 range — momentum zone
        3. Price within 3% of EMA(9) — pullback entry point
        4. Volume >= average (disabled when VOLUME_MULT=0)

        Need 3/4 conditions AND #1 (uptrend) must always pass.

        Returns:
            (should_enter, metadata_dict)
        """
        min_bars = max(SMA_PERIOD, EMA_PERIOD, RSI_PERIOD + 1)
        if len(prices) < min_bars:
            self._log(
                f"{symbol}: Insufficient data ({len(prices)}/{min_bars} bars)",
                "yellow"
            )
            return False, {}

        # Calculate indicators
        sma_20 = calc_sma(prices, SMA_PERIOD)
        ema_9 = calc_ema(prices, EMA_PERIOD)
        rsi = calc_rsi(prices, RSI_PERIOD)

        # Volume ratio
        avg_volume = np.mean([c['volume'] for c in candles[-20:]]) if len(candles) >= 20 else np.mean([c['volume'] for c in candles])
        current_volume = candles[-1]['volume'] if candles else 0
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # Evaluate conditions
        above_sma = price > sma_20
        rsi_ok = RSI_MIN <= rsi <= RSI_MAX

        ema_distance = abs(price - ema_9) / price if price > 0 else 1.0
        at_ema = ema_distance <= EMA_TOLERANCE
        bouncing_off_ema = price > ema_9 and ema_distance <= 0.01

        volume_ok = volume_ratio >= VOLUME_MULT if VOLUME_MULT > 0 else True

        # Count passing conditions
        conditions = [above_sma, rsi_ok, (at_ema or bouncing_off_ema), volume_ok]
        passing = sum(conditions)

        # Signal: all 4 pass, OR 3/4 pass and uptrend is confirmed
        should_enter = passing >= 4 or (passing >= 3 and above_sma)

        # Verbose logging — always show what the bot is checking
        sma_mark = "Y" if above_sma else "N"
        rsi_mark = "Y" if rsi_ok else "N"
        ema_mark = "Y" if (at_ema or bouncing_off_ema) else "N"
        vol_mark = "Y" if volume_ok else "N"

        signal_str = "SIGNAL" if should_enter else "NO SIGNAL"
        near_miss = " (3/4 near-miss)" if passing == 3 and should_enter else ""

        self._log(
            f"{symbol} @ ${price:.2f} | "
            f"SMA20=${sma_20:.2f}[{sma_mark}] "
            f"RSI={rsi:.1f}[{rsi_mark}] "
            f"EMA9=${ema_9:.2f} dist={ema_distance*100:.1f}%[{ema_mark}] "
            f"Vol={volume_ratio:.1f}x[{vol_mark}] "
            f"→ {passing}/4 {signal_str}{near_miss}",
            "green" if should_enter else "white"
        )

        metadata = {
            'sma_20': sma_20,
            'ema_9': ema_9,
            'rsi': rsi,
            'ema_distance_pct': ema_distance * 100,
            'volume_ratio': volume_ratio,
            'conditions_met': passing,
            'signal_strength': f"{passing}/4",
            'direction': 'LONG',
            'reasoning': (
                f"Uptrend pullback: Price>${sma_20:.0f}SMA, "
                f"RSI={rsi:.0f}, EMA9 dist={ema_distance*100:.1f}%, "
                f"{passing}/4 conditions ({TP_PCT}% TP / {SL_PCT}% SL)"
            ),
        }

        return should_enter, metadata


if __name__ == '__main__':
    MomentumBot.main()

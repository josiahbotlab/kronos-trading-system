#!/usr/bin/env python3
"""
Cascade P99 Strategy
====================
Moon Dev's top performer: 425% return, 13% max DD, 2.98 Sharpe.

Only triggers on the top 1% (P99) of liquidation events.
These extreme cascades represent the highest-conviction momentum signals.

Logic:
- Track rolling distribution of liquidation USD
- Only trade when current candle's liquidation > 99th percentile
- Enter in cascade direction with aggressive sizing
- Tight trailing stop (these moves are fast)

The edge: P99 events are so extreme that momentum is almost guaranteed
to continue for at least a few more bars.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class CascadeP99(BaseStrategy):
    name = "cascade_p99"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # P99 detection
            "percentile": 99,                # only trade top 1% events
            "lookback_bars": 500,            # bars to calculate percentile over
            "min_history": 100,              # need at least this many bars before trading
            "liq_ratio_threshold": 0.6,      # 60%+ one-sided to determine direction

            # Entry
            "entry_strength": 1.0,           # full size on P99 events

            # Exit
            "trailing_stop_pct": 1.5,        # tight trailing stop
            "take_profit_pct": 8.0,          # let winners run
            "max_hold_bars": 12,             # these are fast moves
            "exit_on_liq_dry": True,         # exit when liquidation activity drops

            # Filters
            "cooldown_bars": 5,              # wait after trade before re-entry

            "max_history": 600,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        self._cooldown = 0
        self._entry_liq_usd = 0.0

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # --- IN POSITION ---
        if self._in_trade:
            self._bars_held += 1

            # Track extremes
            if self._trade_direction == 1:
                self._peak = max(self._peak, candle.high)
                stop = self._peak * (1 - self.get_param("trailing_stop_pct") / 100)
                if candle.low <= stop:
                    return self._exit("trailing_stop")
            else:
                self._trough = min(self._trough, candle.low)
                stop = self._trough * (1 + self.get_param("trailing_stop_pct") / 100)
                if candle.high >= stop:
                    return self._exit("trailing_stop")

            # Max hold
            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            # Liquidation dried up (momentum fading)
            if self.get_param("exit_on_liq_dry"):
                if candle.liquidation_usd < self._entry_liq_usd * 0.1:
                    if self._bars_held >= 2:  # give at least 2 bars
                        return self._exit("liq_dried")

            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            return Signal(direction=None)

        # Need enough history to calculate percentile
        min_hist = self.get_param("min_history")
        if len(self._candle_history) < min_hist:
            return Signal(direction=None)

        # Get liquidation history
        liq_values = self.liq_usd(self.get_param("lookback_bars"))
        # Filter to non-zero only for percentile calculation
        nonzero = liq_values[liq_values > 0]
        if len(nonzero) < 20:
            return Signal(direction=None)  # not enough liq data

        # Calculate P99 threshold
        pct = self.get_param("percentile")
        threshold = np.percentile(nonzero, pct)

        # Is current candle a P99 event?
        if candle.liquidation_usd < threshold:
            return Signal(direction=None)

        # Determine direction
        total = candle.liquidation_usd
        if total == 0:
            return Signal(direction=None)

        ratio_thresh = self.get_param("liq_ratio_threshold")
        short_ratio = candle.short_liq_usd / total
        long_ratio = candle.long_liq_usd / total

        if short_ratio >= ratio_thresh:
            direction = 1   # shorts getting rekt = bullish
        elif long_ratio >= ratio_thresh:
            direction = -1  # longs getting rekt = bearish
        else:
            return Signal(direction=None)  # mixed, skip

        # ENTER
        self._in_trade = True
        self._trade_direction = direction
        self._bars_held = 0
        self._peak = candle.high
        self._trough = candle.low
        self._entry_liq_usd = candle.liquidation_usd

        return Signal(
            direction=direction,
            strength=self.get_param("entry_strength"),
            tag=f"p99_{'bull' if direction == 1 else 'bear'}",
            metadata={"liq_usd": candle.liquidation_usd, "threshold": threshold},
        )

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._trade_direction = 0
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False
        self._trade_direction = 0

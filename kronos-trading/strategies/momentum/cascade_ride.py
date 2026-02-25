#!/usr/bin/env python3
"""
Cascade Ride Strategy
=====================
Moon Dev's bread-and-butter momentum play.
When liquidation cascades happen, price keeps moving in that direction.

Logic:
- Detect liquidation cascade (total USD > threshold in current candle)
- Determine direction: short liqs (BUY side) = price going UP, long liqs (SELL side) = price going DOWN
- Enter in the direction of the cascade
- Exit on reversal signal or trailing stop

Key insight: When shorts get liquidated, forced BUY orders push price higher,
causing MORE shorts to get liquidated. Ride the wave.

Moon Dev stats: 19/20 momentum strategies were profitable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class CascadeRide(BaseStrategy):
    name = "cascade_ride"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Cascade detection
            "liq_threshold_usd": 50000,     # min USD liquidated to trigger
            "liq_count_min": 3,             # min number of liquidation events
            "liq_ratio_threshold": 0.65,    # 65%+ of liqs on one side = directional cascade

            # Entry
            "use_confirmation": True,        # wait for price to confirm direction
            "confirmation_bars": 1,          # bars to wait for confirmation

            # Exit
            "trailing_stop_pct": 2.0,        # trailing stop loss %
            "take_profit_pct": 5.0,          # take profit %
            "max_hold_bars": 24,             # max bars to hold (prevent stale positions)
            "exit_on_reverse_cascade": True,  # exit if cascade flips direction

            # Filters
            "min_volume_sma_ratio": 1.2,     # volume must be 1.2x average
            "cooldown_bars": 3,              # bars to wait after exit before re-entry

            "max_history": 200,
        }

    def on_init(self):
        self._bars_in_position = 0
        self._peak_price = 0.0
        self._trough_price = float('inf')
        self._entry_direction = 0
        self._cooldown = 0
        self._pending_direction = 0
        self._confirm_countdown = 0

    def _detect_cascade(self, candle: CandleData) -> int:
        """
        Detect a liquidation cascade and return direction.
        Returns: 1 (bullish cascade), -1 (bearish cascade), 0 (no cascade)
        """
        threshold = self.get_param("liq_threshold_usd")
        min_count = self.get_param("liq_count_min")
        ratio_thresh = self.get_param("liq_ratio_threshold")

        # Not enough liquidation activity
        if candle.liquidation_usd < threshold or candle.liq_count < min_count:
            return 0

        total = candle.liquidation_usd
        if total == 0:
            return 0

        short_ratio = candle.short_liq_usd / total  # shorts getting rekt = bullish
        long_ratio = candle.long_liq_usd / total     # longs getting rekt = bearish

        if short_ratio >= ratio_thresh:
            return 1   # bullish - shorts are getting liquidated, price going up
        elif long_ratio >= ratio_thresh:
            return -1  # bearish - longs are getting liquidated, price going down

        return 0

    def _check_volume_filter(self) -> bool:
        """Check if current volume is above average."""
        ratio = self.get_param("min_volume_sma_ratio")
        vols = self.volumes(20)
        if len(vols) < 20:
            return True  # not enough data, skip filter
        avg_vol = np.mean(vols[:-1])
        return avg_vol > 0 and vols[-1] / avg_vol >= ratio

    def on_candle(self, candle: CandleData) -> Signal:
        # Cooldown tracking
        if self._cooldown > 0:
            self._cooldown -= 1

        # Track bars in position
        if self._entry_direction != 0:
            self._bars_in_position += 1

        # --- POSITION MANAGEMENT (if in a trade) ---
        if self._entry_direction != 0:
            # Update trailing stop tracking
            if self._entry_direction == 1:
                self._peak_price = max(self._peak_price, candle.high)
                trailing_stop = self._peak_price * (1 - self.get_param("trailing_stop_pct") / 100)
                tp_price = self._peak_price * (1 + self.get_param("take_profit_pct") / 100)

                # Trailing stop hit
                if candle.low <= trailing_stop:
                    return self._exit("trailing_stop")

            else:  # short
                self._trough_price = min(self._trough_price, candle.low)
                trailing_stop = self._trough_price * (1 + self.get_param("trailing_stop_pct") / 100)
                tp_price = self._trough_price * (1 - self.get_param("take_profit_pct") / 100)

                # Trailing stop hit
                if candle.high >= trailing_stop:
                    return self._exit("trailing_stop")

            # Max hold time
            if self._bars_in_position >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            # Reverse cascade exit
            if self.get_param("exit_on_reverse_cascade"):
                cascade_dir = self._detect_cascade(candle)
                if cascade_dir != 0 and cascade_dir != self._entry_direction:
                    return self._exit("reverse_cascade")

            return Signal(direction=None)  # hold

        # --- ENTRY LOGIC (no position) ---
        if self._cooldown > 0:
            return Signal(direction=None)

        # Handle confirmation waiting
        if self._confirm_countdown > 0:
            self._confirm_countdown -= 1
            if self._confirm_countdown == 0:
                # Check if price confirmed the direction
                if self._pending_direction == 1 and candle.close > candle.open:
                    return self._enter(1, "cascade_confirmed")
                elif self._pending_direction == -1 and candle.close < candle.open:
                    return self._enter(-1, "cascade_confirmed")
                else:
                    self._pending_direction = 0  # confirmation failed
            return Signal(direction=None)

        # Detect cascade
        cascade_dir = self._detect_cascade(candle)
        if cascade_dir == 0:
            return Signal(direction=None)

        # Volume filter
        if not self._check_volume_filter():
            return Signal(direction=None)

        # Enter immediately or wait for confirmation
        if self.get_param("use_confirmation"):
            self._pending_direction = cascade_dir
            self._confirm_countdown = self.get_param("confirmation_bars")
            return Signal(direction=None)
        else:
            return self._enter(cascade_dir, "cascade_immediate")

    def _enter(self, direction: int, tag: str) -> Signal:
        """Set up position tracking and return entry signal."""
        self._entry_direction = direction
        self._bars_in_position = 0
        self._peak_price = 0.0
        self._trough_price = float('inf')
        return Signal(direction=direction, strength=1.0, tag=tag)

    def _exit(self, reason: str) -> Signal:
        """Clean up and return exit signal."""
        self._entry_direction = 0
        self._bars_in_position = 0
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._entry_direction = 0

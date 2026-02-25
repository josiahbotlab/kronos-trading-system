#!/usr/bin/env python3
"""
24-Hour Consolidation Breakout v1.0
=====================================
Source: Moon Dev ChatGPT Bot Build (v5qTYnMiUw4)
Confidence: 85%

Detects when price compresses into a tight range (consolidation),
then trades the breakout when price escapes with momentum.

Transcript insights:
- Consolidation = 24h range / low < 2% threshold
- "If range is less than two percent, return the average price as consolidation"
- Entry when price breaks above/below the consolidation range
- 6% take profit, 3% stop loss
- Tested on 1h candles aggregated over 24h
- Volume confirmation on breakout improves reliability

Web research (consolidation breakout):
- Close-based breakouts, NOT wick-based (fewer false signals)
- Volume expansion > 1.2x 20-bar average confirms breakout
- False breakouts are the primary failure mode without volume filter
- Stop loss: 1 ATR inside consolidation range from breakout edge
- Time stop: if price re-enters range within 3-5 bars, exit (false breakout)
- For crypto: 1.5% threshold may be better than 2% (BTC is volatile)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class ConsolidationBreakout(BaseStrategy):
    name = "consolidation_breakout"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Consolidation detection
            "consol_window": 24,           # bars to check (24h on 1h candles)
            "range_pct_threshold": 3.0,    # tighter = more explosive breakouts
            "min_consol_bars": 5,          # moderate narrowness check

            # Breakout confirmation
            "use_vol_confirm": True,       # filter false breakouts
            "vol_lookback": 20,
            "vol_mult": 1.2,              # volume > 1.2x average

            # Re-entry protection
            "use_reenter_stop": False,     # disabled — too aggressive
            "reenter_bars": 3,

            "min_history": 30,

            # Entry
            "entry_strength": 0.75,

            # Exit (tighter for consolidation breakouts)
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "trailing_stop_pct": 1.5,
            "trail_after_bars": 3,
            "max_hold_bars": 16,
            "cooldown_bars": 3,

            "max_history": 300,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._entry_price = 0.0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        self._cooldown = 0
        self._range_high = 0.0
        self._range_low = 0.0

    def _detect_consolidation(self) -> tuple[bool, float, float]:
        """
        Check if PREVIOUS N bars form a consolidation (excludes current candle).
        Returns (is_consolidating, range_high, range_low).
        """
        window = self.get_param("consol_window")
        if len(self._candle_history) < window + 1:
            return False, 0, 0

        # Use previous window bars (exclude current candle for breakout detection)
        prev_candles = self._candle_history[-(window + 1):-1]
        highs = np.array([c.high for c in prev_candles])
        lows = np.array([c.low for c in prev_candles])
        closes = np.array([c.close for c in prev_candles])

        range_high = float(np.max(highs))
        range_low = float(np.min(lows))
        mid_price = (range_high + range_low) / 2

        if mid_price < 1e-8:
            return False, 0, 0

        range_pct = (range_high - range_low) / mid_price * 100
        is_narrow = range_pct < self.get_param("range_pct_threshold")

        # Check sustained narrowness
        if is_narrow and self.get_param("min_consol_bars") > 0:
            min_bars = self.get_param("min_consol_bars")
            bodies = np.abs(closes - np.array([c.open for c in prev_candles]))
            avg_body = np.mean(bodies)
            if avg_body > 0:
                narrow_count = int(np.sum(bodies < avg_body * 1.0))
            else:
                narrow_count = len(bodies)
            is_narrow = narrow_count >= min_bars

        return is_narrow, range_high, range_low

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # --- IN POSITION ---
        if self._in_trade:
            self._bars_held += 1

            stop_pct = self.get_param("stop_loss_pct") / 100
            tp_pct = self.get_param("take_profit_pct") / 100

            if self._trade_direction == 1:
                # Stop loss
                if candle.low <= self._entry_price * (1 - stop_pct):
                    return self._exit("stop_loss")
                # Take profit
                if candle.high >= self._entry_price * (1 + tp_pct):
                    return self._exit("take_profit")
                # Trailing stop after N bars
                if self._bars_held >= self.get_param("trail_after_bars"):
                    self._peak = max(self._peak, candle.high)
                    trail = self._peak * (1 - self.get_param("trailing_stop_pct") / 100)
                    if candle.low <= trail:
                        return self._exit("trailing_stop")
                # False breakout: price re-enters consolidation range
                if self.get_param("use_reenter_stop") and self._bars_held <= self.get_param("reenter_bars"):
                    if candle.close < self._range_high:
                        return self._exit("false_breakout")
            else:
                if candle.high >= self._entry_price * (1 + stop_pct):
                    return self._exit("stop_loss")
                if candle.low <= self._entry_price * (1 - tp_pct):
                    return self._exit("take_profit")
                if self._bars_held >= self.get_param("trail_after_bars"):
                    self._trough = min(self._trough, candle.low)
                    trail = self._trough * (1 + self.get_param("trailing_stop_pct") / 100)
                    if candle.high >= trail:
                        return self._exit("trailing_stop")
                if self.get_param("use_reenter_stop") and self._bars_held <= self.get_param("reenter_bars"):
                    if candle.close > self._range_low:
                        return self._exit("false_breakout")

            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            return Signal(direction=None)

        # Check for consolidation
        is_consol, range_high, range_low = self._detect_consolidation()
        if not is_consol:
            return Signal(direction=None)

        # Check for breakout: CLOSE-based (not wick)
        direction = None
        if candle.close > range_high:
            direction = 1   # bullish breakout
        elif candle.close < range_low:
            direction = -1  # bearish breakout

        if direction is None:
            return Signal(direction=None)

        # Volume confirmation
        if self.get_param("use_vol_confirm"):
            vols = self.volumes(self.get_param("vol_lookback"))
            if len(vols) >= self.get_param("vol_lookback"):
                avg_vol = np.mean(vols[:-1])
                if avg_vol > 0 and candle.volume < avg_vol * self.get_param("vol_mult"):
                    return Signal(direction=None)

        # Enter trade
        self._in_trade = True
        self._trade_direction = direction
        self._entry_price = candle.close
        self._bars_held = 0
        self._peak = candle.high
        self._trough = candle.low
        self._range_high = range_high
        self._range_low = range_low

        range_pct = (range_high - range_low) / candle.close * 100

        return Signal(
            direction=direction,
            strength=self.get_param("entry_strength"),
            tag=f"consol_break_{'up' if direction == 1 else 'down'}",
            metadata={"range_pct": range_pct, "range_high": range_high, "range_low": range_low},
        )

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._trade_direction = 0
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False
        self._trade_direction = 0


PARAM_RANGES = {
    "consol_window": [16, 24, 32],
    "range_pct_threshold": [1.5, 2.0, 3.0],
    "min_consol_bars": [6, 10, 14],
    "vol_mult": [1.0, 1.2, 1.5],
    "stop_loss_pct": [2.0, 3.0, 4.0],
    "take_profit_pct": [4.0, 6.0, 8.0],
    "trailing_stop_pct": [1.5, 2.0, 3.0],
    "max_hold_bars": [16, 24, 32],
}

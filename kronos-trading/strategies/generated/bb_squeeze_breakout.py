#!/usr/bin/env python3
"""
Bollinger Band Squeeze Breakout v1.0
=====================================
Source: Harvard Algo Trading CS50 Lecture + TTM Squeeze (John Carter)
Confidence: 85%

TTM Squeeze concept: when Bollinger Bands contract INSIDE Keltner Channels,
volatility is compressed (squeeze). When BB expands back outside KC,
a directional breakout is imminent.

Transcript insights:
- BB squeeze identified as one of the "safest play" strategies
- Consistent performance across backtests
- Combined with ADX for trend strength confirmation

Web research (John Carter / TTM Squeeze):
- BB(20, 2.0) standard, KC(20, 1.5 * ATR) — the 1.5 multiplier is critical
- Squeeze ON: BB lower > KC lower AND BB upper < KC upper
- Squeeze FIRES: first bar where BB expands outside KC
- Momentum histogram = LinReg of (close - avg(Donchian_mid, SMA))
- Direction from momentum: positive & rising = long, negative & falling = short
- ADX < 20 during squeeze confirms genuine consolidation
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class BbSqueezeBreakout(BaseStrategy):
    name = "bb_squeeze_breakout"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Bollinger Bands
            "bb_period": 20,
            "bb_std": 2.0,

            # Keltner Channel
            "kc_period": 20,
            "kc_atr_mult": 1.5,       # Carter's original 1.5x ATR

            # Momentum (LinReg of delta)
            "momentum_period": 20,

            # ADX filter (optional)
            "use_adx_filter": False,    # disabled by default for more signals
            "adx_period": 14,
            "adx_squeeze_max": 25,     # ADX must be below this during squeeze

            "min_history": 60,

            # Entry
            "entry_strength": 0.8,

            # Exit
            "trailing_stop_pct": 2.5,
            "take_profit_pct": 6.0,
            "max_hold_bars": 24,
            "cooldown_bars": 3,

            # Momentum exit: exit when momentum reverses for N bars
            "momentum_reversal_bars": 3,

            "max_history": 300,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        self._cooldown = 0
        self._was_squeeze = False     # previous bar was in squeeze
        self._prev_momentum = None
        self._momentum_reversal_count = 0

    def _compute_ema_array(self, values: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _compute_keltner(self) -> tuple[float, float, float] | None:
        """Compute Keltner Channel: (upper, middle, lower)."""
        period = self.get_param("kc_period")
        mult = self.get_param("kc_atr_mult")

        if len(self._candle_history) < period + 1:
            return None

        closes = self.closes(period * 2)
        kc_mid = self._compute_ema_array(closes, period)[-1]

        atr_val = self.atr(period)
        if atr_val is None:
            return None

        return (kc_mid + mult * atr_val, kc_mid, kc_mid - mult * atr_val)

    def _compute_momentum(self) -> float | None:
        """Carter's momentum: linear regression of (close - avg(donchian_mid, sma))."""
        period = self.get_param("momentum_period")
        if len(self._candle_history) < period + 10:
            return None

        closes = self.closes(period)
        highs = self.highs(period)
        lows = self.lows(period)

        donchian_mid = (np.max(highs) + np.min(lows)) / 2
        sma_val = float(np.mean(closes))

        delta = closes[-1] - (donchian_mid + sma_val) / 2

        # Simple linear regression value of delta series
        # Use last N deltas
        n = min(period, len(closes))
        delta_series = []
        for i in range(n):
            h = self.highs(period - i)
            l = self.lows(period - i)
            c = self.closes(period - i)
            if len(h) < 1 or len(l) < 1 or len(c) < 1:
                break
            d_mid = (np.max(h) + np.min(l)) / 2
            s_val = float(np.mean(c))
            delta_series.append(c[-1] - (d_mid + s_val) / 2)

        if len(delta_series) < 5:
            return delta

        # LinReg value (endpoint of regression line)
        y = np.array(delta_series[::-1])  # oldest to newest
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0] * (len(y) - 1) + coeffs[1])

    def _is_squeeze(self) -> bool | None:
        """Check if BB is inside KC (squeeze is ON)."""
        bb = self.bollinger_bands(self.get_param("bb_period"), self.get_param("bb_std"))
        kc = self._compute_keltner()

        if bb is None or kc is None:
            return None

        bb_upper, bb_mid, bb_lower = bb
        kc_upper, kc_mid, kc_lower = kc

        # Squeeze ON: BB fits entirely inside KC
        return bb_lower > kc_lower and bb_upper < kc_upper

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        momentum = self._compute_momentum()

        # --- IN POSITION ---
        if self._in_trade:
            self._bars_held += 1

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

            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            # Momentum reversal exit
            if momentum is not None and self._prev_momentum is not None:
                if self._trade_direction == 1 and momentum < self._prev_momentum:
                    self._momentum_reversal_count += 1
                elif self._trade_direction == -1 and momentum > self._prev_momentum:
                    self._momentum_reversal_count += 1
                else:
                    self._momentum_reversal_count = 0

                if self._momentum_reversal_count >= self.get_param("momentum_reversal_bars"):
                    self._prev_momentum = momentum
                    return self._exit("momentum_reversal")

            self._prev_momentum = momentum
            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            squeeze = self._is_squeeze()
            if squeeze is not None:
                self._was_squeeze = squeeze
            self._prev_momentum = momentum
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            self._prev_momentum = momentum
            return Signal(direction=None)

        # Check squeeze state
        squeeze = self._is_squeeze()
        if squeeze is None:
            self._prev_momentum = momentum
            return Signal(direction=None)

        # Detect squeeze FIRE: was in squeeze, now not
        squeeze_fired = self._was_squeeze and not squeeze
        self._was_squeeze = squeeze

        if not squeeze_fired:
            self._prev_momentum = momentum
            return Signal(direction=None)

        # Squeeze fired! Determine direction from momentum
        if momentum is None:
            self._prev_momentum = momentum
            return Signal(direction=None)

        direction = None
        if momentum > 0:
            direction = 1   # bullish breakout
        elif momentum < 0:
            direction = -1  # bearish breakout

        if direction is None:
            self._prev_momentum = momentum
            return Signal(direction=None)

        # Enter trade
        self._in_trade = True
        self._trade_direction = direction
        self._bars_held = 0
        self._peak = candle.high
        self._trough = candle.low
        self._momentum_reversal_count = 0
        self._prev_momentum = momentum

        return Signal(
            direction=direction,
            strength=self.get_param("entry_strength"),
            tag=f"squeeze_{'bull' if direction == 1 else 'bear'}",
            metadata={"momentum": momentum},
        )

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._trade_direction = 0
        self._cooldown = self.get_param("cooldown_bars")
        self._momentum_reversal_count = 0
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False
        self._trade_direction = 0


PARAM_RANGES = {
    "bb_period": [15, 20, 25],
    "bb_std": [1.5, 2.0, 2.5],
    "kc_atr_mult": [1.0, 1.5, 2.0],
    "momentum_period": [15, 20, 25],
    "trailing_stop_pct": [1.5, 2.0, 3.0],
    "take_profit_pct": [4.0, 5.0, 7.0],
    "max_hold_bars": [12, 16, 24],
    "momentum_reversal_bars": [1, 2, 3],
}

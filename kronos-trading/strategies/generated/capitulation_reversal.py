#!/usr/bin/env python3
"""
Capitulation Reversal v2.0
=============================
Source: Clawdbot For Trading is UNSTOPPABLE (71,057% ROI)
Confidence: 90%

Improved based on transcript insights:
- "Capitulation filter was too restrictive, killing too many trades"
  → Loosened from 2.5σ to 2.0σ volume spike threshold
- "Pure capitulation reversal: 536% return, 68% WR, -34% DD" (on ETH)
- "With trend filter: 74% return, 55% WR"
  → Added confirmation candle: wait for reversal candle after spike
- Liquidation data as PRIMARY signal (volume as fallback)
- Red-then-green pattern: spike candle + next candle reversal = entry

Web research: 2.5x average volume spike, RSI < 20 for extreme oversold,
liquidation data superior to volume as capitulation proxy.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class CapitulationReversal(BaseStrategy):
    name = "capitulation_reversal"
    version = "2.0"

    def default_params(self) -> dict:
        return {
            # Volume spike detection (loosened from v1)
            "vol_lookback": 50,
            "vol_spike_std": 2.0,         # lowered from 2.5 per transcript

            # Liquidation as primary signal
            "use_liq_primary": True,
            "liq_spike_multiplier": 2.5,  # liq must be 2.5x rolling average

            # Candle body filter
            "min_body_atr": 0.6,          # loosened from 0.8
            "atr_period": 14,

            # Confirmation candle (red-then-green pattern)
            "use_confirmation": False,    # disabled: was killing win rate

            # RSI extreme filter
            "use_rsi_extreme": False,     # disabled: was too restrictive
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,

            "min_history": 60,

            # Entry
            "entry_strength": 0.7,

            # Exit
            "stop_loss_pct": 2.5,
            "take_profit_pct": 5.0,       # wider TP for reversal
            "trailing_stop_pct": 1.5,
            "trail_after_bars": 3,
            "max_hold_bars": 18,
            "cooldown_bars": 5,

            "max_history": 200,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._entry_price = 0.0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        self._cooldown = 0
        self._pending_signal = None  # for confirmation candle pattern

    def _detect_capitulation(self, candle: CandleData) -> int:
        """Detect capitulation event. Returns candle direction (-1=bearish spike, +1=bullish spike)."""
        # --- Liquidation-based detection (primary) ---
        if self.get_param("use_liq_primary") and candle.liquidation_usd > 0:
            liq_values = self.liq_usd(self.get_param("vol_lookback"))
            nonzero = liq_values[liq_values > 0]
            if len(nonzero) >= 10:
                avg_liq = float(np.mean(nonzero))
                if candle.liquidation_usd >= avg_liq * self.get_param("liq_spike_multiplier"):
                    # Determine direction from liquidation type
                    if candle.long_liq_usd > candle.short_liq_usd:
                        return -1  # longs liquidated = bearish spike
                    else:
                        return 1   # shorts liquidated = bullish spike

        # --- Volume-based detection (fallback) ---
        vols = self.volumes(self.get_param("vol_lookback"))
        if len(vols) < self.get_param("vol_lookback"):
            return 0
        mean_vol = np.mean(vols[:-1])
        std_vol = np.std(vols[:-1], ddof=1)
        if std_vol < 1e-8:
            return 0
        if vols[-1] <= mean_vol + self.get_param("vol_spike_std") * std_vol:
            return 0

        # Volume spike detected — determine direction from candle body
        body = candle.close - candle.open
        atr_val = self.atr(self.get_param("atr_period"))
        if atr_val is None or atr_val < 1e-8:
            return 0
        if abs(body) < self.get_param("min_body_atr") * atr_val:
            return 0
        return 1 if body > 0 else -1

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # --- IN POSITION ---
        if self._in_trade:
            self._bars_held += 1

            stop_pct = self.get_param("stop_loss_pct") / 100
            tp_pct = self.get_param("take_profit_pct") / 100

            if self._trade_direction == 1:
                if candle.low <= self._entry_price * (1 - stop_pct):
                    return self._exit("stop_loss")
                if candle.high >= self._entry_price * (1 + tp_pct):
                    return self._exit("take_profit")
                if self._bars_held >= self.get_param("trail_after_bars"):
                    self._peak = max(self._peak, candle.high)
                    trail = self._peak * (1 - self.get_param("trailing_stop_pct") / 100)
                    if candle.low <= trail:
                        return self._exit("trailing_stop")
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

            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            self._pending_signal = None
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            return Signal(direction=None)

        # --- Confirmation candle pattern ---
        if self.get_param("use_confirmation") and self._pending_signal is not None:
            spike_dir = self._pending_signal
            self._pending_signal = None

            # Check if current candle confirms reversal
            # After bearish spike (-1), we need a green candle → go long
            # After bullish spike (+1), we need a red candle → go short
            body = candle.close - candle.open
            if spike_dir == -1 and body > 0:  # red spike, green confirmation
                return self._enter(1, candle, "cap_confirm_long")
            elif spike_dir == 1 and body < 0:  # green spike, red confirmation
                return self._enter(-1, candle, "cap_confirm_short")
            # No confirmation — signal expires
            return Signal(direction=None)

        # Detect capitulation
        spike_dir = self._detect_capitulation(candle)
        if spike_dir == 0:
            return Signal(direction=None)

        # Optional RSI extreme filter
        if self.get_param("use_rsi_extreme"):
            rsi_val = self.rsi(self.get_param("rsi_period"))
            if rsi_val is not None:
                if spike_dir == -1 and rsi_val > self.get_param("rsi_oversold"):
                    return Signal(direction=None)  # not oversold enough
                if spike_dir == 1 and rsi_val < self.get_param("rsi_overbought"):
                    return Signal(direction=None)  # not overbought enough

        # Use confirmation candle pattern if enabled
        if self.get_param("use_confirmation"):
            self._pending_signal = spike_dir
            return Signal(direction=None)

        # Direct entry (no confirmation)
        direction = -spike_dir  # fade the spike
        return self._enter(direction, candle, f"cap_rev_{'long' if direction == 1 else 'short'}")

    def _enter(self, direction: int, candle: CandleData, tag: str) -> Signal:
        self._in_trade = True
        self._trade_direction = direction
        self._entry_price = candle.close
        self._bars_held = 0
        self._peak = candle.high
        self._trough = candle.low

        return Signal(
            direction=direction,
            strength=self.get_param("entry_strength"),
            tag=tag,
            metadata={"volume": candle.volume, "liq_usd": candle.liquidation_usd},
        )

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._trade_direction = 0
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False
        self._trade_direction = 0
        self._pending_signal = None


PARAM_RANGES = {
    "vol_spike_std": [1.5, 2.0, 2.5],
    "liq_spike_multiplier": [2.0, 2.5, 3.0],
    "min_body_atr": [0.4, 0.6, 0.8],
    "rsi_oversold": [20, 25, 30],
    "rsi_overbought": [70, 75, 80],
    "stop_loss_pct": [2.0, 2.5, 3.5],
    "take_profit_pct": [4.0, 5.0, 7.0],
    "trailing_stop_pct": [1.0, 1.5, 2.0],
    "max_hold_bars": [12, 18, 25],
}

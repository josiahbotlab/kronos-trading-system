#!/usr/bin/env python3
"""
Intraday Parabolic Short (Mean Reversion) v1.0
=================================================
Source: Moon Dev No BS Guide (K6FBI8mZ5Us)
Confidence: 95%

Short-only strategy targeting parabolic price spikes — exponentially
accelerating moves that deviate far above the rolling mean. These
moves are unsustainable and revert violently.

Transcript insights:
- Short when price > rolling mean + N standard deviations
- 20-period rolling mean, std dev thresholds 1.0 to 3.0
- Lookback windows: 10, 15, 20, 25, 30
- "Statistical arbitrage strategy that targets intraday parabolic price spikes"
- Fixed percentage stop-loss and take-profit

Web research (parabolic shorting):
- N = 2.5 sweet spot for crypto (high baseline volatility)
- Volume climax is strongest confirmation: blow-off top has highest volume
- NEVER short at the highs — wait for first lower high (reversal confirm)
- Mean reversion target: 20-EMA or VWAP
- Stop loss: day's high or most recent swing high
- Use longer lookback (50-100) for std dev to avoid the spike inflating it
- "Parabolic Burst" framework: 3+ consecutive gain bars, range expansion,
  volume expansion on 2 consecutive bars
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class ParabolicShort(BaseStrategy):
    name = "parabolic_short"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Parabolic detection (optimized from robustness best params)
            "lookback": 20,                # shorter = more responsive
            "zscore_threshold": 2.5,       # z-score above this = parabolic

            # Reversal confirmation (don't short at highs)
            "use_lower_high_confirm": True,
            "consec_up_bars": 2,           # 2 consecutive up bars (was 3, too strict)

            # Volume climax (optional)
            "use_vol_confirm": False,
            "vol_lookback": 20,
            "vol_spike_mult": 1.5,

            "min_history": 40,

            # Entry
            "entry_strength": 0.7,

            # Exit (optimized from robustness best params)
            "mean_reversion_ema": 30,      # wider target (was 20)
            "stop_loss_pct": 2.0,          # tighter stop (was 3.0)
            "take_profit_pct": 6.0,        # wider target (was 4.0)
            "trailing_stop_pct": 1.5,
            "trail_after_bars": 3,
            "max_hold_bars": 24,           # longer hold (was 16)
            "cooldown_bars": 4,

            "max_history": 300,
        }

    def on_init(self):
        self._in_trade = False
        self._entry_price = 0.0
        self._bars_held = 0
        self._trough = float('inf')
        self._cooldown = 0
        self._prev_high = 0.0
        self._spike_detected = False     # parabolic spike detected, waiting for confirm

    def _detect_parabolic(self, candle: CandleData) -> tuple[bool, float]:
        """
        Detect parabolic spike above rolling mean.
        Returns (is_parabolic, z_score).
        """
        lookback = self.get_param("lookback")
        if len(self._candle_history) < lookback:
            return False, 0.0

        closes = self.closes(lookback)
        mean = float(np.mean(closes))
        std = float(np.std(closes, ddof=1))

        if std < 1e-8:
            return False, 0.0

        zscore = (candle.close - mean) / std

        if zscore < self.get_param("zscore_threshold"):
            return False, zscore

        # Check consecutive up bars (parabolic acceleration)
        consec = self.get_param("consec_up_bars")
        if consec > 0 and len(self._candle_history) >= consec + 1:
            recent = self._candle_history[-(consec + 1):]
            all_up = all(recent[i].close > recent[i - 1].close for i in range(1, len(recent)))
            if not all_up:
                return False, zscore

        return True, zscore

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # --- IN POSITION (SHORT) ---
        if self._in_trade:
            self._bars_held += 1

            stop_pct = self.get_param("stop_loss_pct") / 100
            tp_pct = self.get_param("take_profit_pct") / 100

            # Stop loss (price goes up more)
            if candle.high >= self._entry_price * (1 + stop_pct):
                self._prev_high = candle.high
                return self._exit("stop_loss")

            # Take profit (price drops toward mean)
            if candle.low <= self._entry_price * (1 - tp_pct):
                self._prev_high = candle.high
                return self._exit("take_profit")

            # Mean reversion exit: price crosses below 20-EMA
            ema_val = self.ema(self.get_param("mean_reversion_ema"))
            if ema_val is not None and candle.close < ema_val:
                self._prev_high = candle.high
                return self._exit("mean_reversion")

            # Trailing stop after N bars
            if self._bars_held >= self.get_param("trail_after_bars"):
                self._trough = min(self._trough, candle.low)
                trail = self._trough * (1 + self.get_param("trailing_stop_pct") / 100)
                if candle.high >= trail:
                    self._prev_high = candle.high
                    return self._exit("trailing_stop")

            if self._bars_held >= self.get_param("max_hold_bars"):
                self._prev_high = candle.high
                return self._exit("max_hold")

            self._prev_high = candle.high
            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            self._prev_high = candle.high
            self._spike_detected = False
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            self._prev_high = candle.high
            return Signal(direction=None)

        is_parabolic, zscore = self._detect_parabolic(candle)

        # State machine: detect spike, then wait for lower high
        if is_parabolic and not self._spike_detected:
            self._spike_detected = True
            self._prev_high = candle.high
            return Signal(direction=None)  # wait for confirmation

        if self._spike_detected:
            # Check for lower high (first sign of reversal)
            if self.get_param("use_lower_high_confirm"):
                if candle.high >= self._prev_high:
                    # Still making higher highs — not ready
                    self._prev_high = candle.high
                    # But if we've waited too long, reset
                    if not is_parabolic:
                        _, current_z = self._detect_parabolic(candle)
                        if current_z < self.get_param("zscore_threshold") * 0.7:
                            self._spike_detected = False
                    return Signal(direction=None)

                # Lower high confirmed! Short entry.
            else:
                # No confirmation needed, short immediately on next bar
                pass

            # Volume confirmation (optional)
            if self.get_param("use_vol_confirm"):
                vols = self.volumes(self.get_param("vol_lookback"))
                if len(vols) >= self.get_param("vol_lookback"):
                    avg_vol = np.mean(vols[:-1])
                    if avg_vol > 0 and candle.volume < avg_vol * self.get_param("vol_spike_mult"):
                        self._prev_high = candle.high
                        return Signal(direction=None)

            # SHORT entry
            self._spike_detected = False
            self._in_trade = True
            self._entry_price = candle.close
            self._bars_held = 0
            self._trough = candle.low
            self._prev_high = candle.high

            return Signal(
                direction=-1,
                strength=self.get_param("entry_strength"),
                tag="parabolic_short",
                metadata={"zscore": zscore},
            )

        self._prev_high = candle.high
        return Signal(direction=None)

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._spike_detected = False
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False


PARAM_RANGES = {
    "lookback": [20, 30, 50],
    "zscore_threshold": [2.0, 2.5, 3.0],
    "consec_up_bars": [2, 3, 4],
    "mean_reversion_ema": [10, 20, 30],
    "stop_loss_pct": [2.0, 3.0, 4.0],
    "take_profit_pct": [3.0, 4.0, 6.0],
    "trailing_stop_pct": [1.0, 1.5, 2.0],
    "max_hold_bars": [10, 16, 24],
}

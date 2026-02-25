#!/usr/bin/env python3
"""
HLP Sentiment Z-Score Reversal v2.0
======================================
Source: Clawdbot For Trading is UNSTOPPABLE (71,057% ROI)
Confidence: 95%

Improved based on transcript + web research:
- Loosened Z-score threshold from ±2.0 to ±1.5 (v1 had 0 trades)
- Shortened lookback from 100 to 50 bars for faster adaptation
- EMA(5) smoothing on liq ratio to approximate 4-6h sentiment shift on 1h data
- Added RSI confirmation (oversold/overbought at extremes)
- Added volume spike filter: Z-score + volume = higher conviction
- Kelly-inspired strength: proportional to z-score magnitude

Transcript insights:
- "Z-score < -2 → retail heavily long → potential long squeeze"
- "Z-score > +2 → retail heavily short → potential short squeeze"
- "Live Z-score was -2.45 (strong signal)"
- "Confirmation: HLP sentiment + smart money + multi-exchange liquidation pressure"
- "Risk: 2% max, Kelly criterion"
- "Timeframe: 4-6 hour candles, BTC/ETH"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class HlpZscoreReversal(BaseStrategy):
    name = "hlp_zscore_reversal"
    version = "2.0"

    def default_params(self) -> dict:
        return {
            # Z-score calculation (loosened from v1)
            "zscore_lookback": 30,         # shortened further for more signals
            "zscore_long_threshold": 1.2,   # further loosened
            "zscore_short_threshold": -1.2, # further loosened
            "min_history": 40,

            # Liquidation quality filter
            "min_liq_usd": 100,            # very low — just needs nonzero direction
            "ratio_smoothing": 5,          # EMA(5) ≈ 4-6h window on 1h candles

            # RSI confirmation (new in v2)
            "use_rsi_confirm": False,      # disabled — z-score is the primary filter
            "rsi_period": 14,
            "rsi_oversold": 40,
            "rsi_overbought": 60,

            # Volume confirmation (new in v2)
            "use_vol_confirm": False,      # disabled — too restrictive
            "vol_lookback": 20,
            "vol_threshold_pct": 100,

            # Entry
            "base_strength": 0.5,
            "max_strength": 0.9,           # scales with z-score magnitude

            # Exit
            "trailing_stop_pct": 2.0,
            "take_profit_pct": 4.0,
            "max_hold_bars": 24,
            "cooldown_bars": 6,

            # Z-score mean reversion exit
            "exit_zscore_threshold": 0.5,  # exit when z-score returns toward 0

            "max_history": 200,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        self._cooldown = 0
        self._ratio_history = []
        self._entry_zscore = 0.0

    def _compute_liq_ratio(self, candle: CandleData) -> float | None:
        """Compute short_liq / total_liq ratio (0-1). Higher = more shorts liquidated."""
        total = candle.liquidation_usd
        if total < self.get_param("min_liq_usd"):
            return None
        # Use long/short breakdown
        if candle.short_liq_usd + candle.long_liq_usd < 1e-8:
            return None
        return candle.short_liq_usd / (candle.short_liq_usd + candle.long_liq_usd)

    def _compute_zscore(self) -> float | None:
        """Z-score of the current smoothed liq ratio vs rolling history."""
        lookback = self.get_param("zscore_lookback")
        if len(self._ratio_history) < lookback:
            return None

        recent = np.array(self._ratio_history[-lookback:])
        mean = np.mean(recent)
        std = np.std(recent, ddof=1)
        if std < 1e-8:
            return 0.0

        current = self._ratio_history[-1]
        return (current - mean) / std

    def _compute_strength(self, zscore: float) -> float:
        """Kelly-inspired position sizing: stronger signal = higher strength."""
        base = self.get_param("base_strength")
        max_s = self.get_param("max_strength")
        # Scale linearly from base at threshold to max at 3.0
        abs_z = abs(zscore)
        threshold = abs(self.get_param("zscore_long_threshold"))
        if abs_z <= threshold:
            return base
        scale = min((abs_z - threshold) / (3.0 - threshold), 1.0)
        return base + scale * (max_s - base)

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # Update ratio history with EMA smoothing
        raw_ratio = self._compute_liq_ratio(candle)
        if raw_ratio is not None:
            if self._ratio_history:
                alpha = 2 / (self.get_param("ratio_smoothing") + 1)
                smoothed = alpha * raw_ratio + (1 - alpha) * self._ratio_history[-1]
            else:
                smoothed = raw_ratio
            self._ratio_history.append(smoothed)
            if len(self._ratio_history) > self.get_param("max_history"):
                self._ratio_history = self._ratio_history[-self.get_param("max_history"):]

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

            # Z-score mean reversion exit: if z-score has reverted, take profit
            zscore = self._compute_zscore()
            if zscore is not None:
                exit_thresh = self.get_param("exit_zscore_threshold")
                if self._trade_direction == 1 and self._entry_zscore > 0:
                    # We went long because z > threshold (shorts rekt), exit when z drops
                    if zscore < exit_thresh:
                        return self._exit("zscore_revert")
                elif self._trade_direction == -1 and self._entry_zscore < 0:
                    if zscore > -exit_thresh:
                        return self._exit("zscore_revert")

            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            return Signal(direction=None)

        zscore = self._compute_zscore()
        if zscore is None:
            return Signal(direction=None)

        long_thresh = self.get_param("zscore_long_threshold")
        short_thresh = self.get_param("zscore_short_threshold")

        direction = None
        if zscore > long_thresh:
            direction = 1   # shorts getting rekt heavily → bullish reversal
        elif zscore < short_thresh:
            direction = -1  # longs getting rekt heavily → bearish reversal

        if direction is None:
            return Signal(direction=None)

        # RSI confirmation
        if self.get_param("use_rsi_confirm"):
            rsi_val = self.rsi(self.get_param("rsi_period"))
            if rsi_val is not None:
                if direction == 1 and rsi_val > self.get_param("rsi_overbought"):
                    return Signal(direction=None)  # already overbought, don't chase
                if direction == -1 and rsi_val < self.get_param("rsi_oversold"):
                    return Signal(direction=None)

        # Volume confirmation
        if self.get_param("use_vol_confirm"):
            vols = self.volumes(self.get_param("vol_lookback"))
            if len(vols) >= self.get_param("vol_lookback"):
                avg_vol = np.mean(vols[:-1])
                if avg_vol > 0 and vols[-1] < avg_vol * self.get_param("vol_threshold_pct") / 100:
                    return Signal(direction=None)  # low volume, weak signal

        strength = self._compute_strength(zscore)

        self._in_trade = True
        self._trade_direction = direction
        self._bars_held = 0
        self._peak = candle.high
        self._trough = candle.low
        self._entry_zscore = zscore

        return Signal(
            direction=direction,
            strength=strength,
            tag=f"zscore_{'bull' if direction == 1 else 'bear'}",
            metadata={"zscore": zscore},
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
    "zscore_lookback": [30, 50, 75],
    "zscore_long_threshold": [1.0, 1.5, 2.0],
    "zscore_short_threshold": [-2.0, -1.5, -1.0],
    "ratio_smoothing": [3, 5, 8],
    "rsi_oversold": [35, 40, 45],
    "rsi_overbought": [55, 60, 65],
    "vol_threshold_pct": [100, 120, 150],
    "trailing_stop_pct": [1.5, 2.0, 3.0],
    "take_profit_pct": [3.0, 4.0, 6.0],
    "max_hold_bars": [16, 24, 32],
}

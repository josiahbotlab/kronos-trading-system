#!/usr/bin/env python3
"""
OBV Capitulation Divergence v2.0
===================================
Source: Clawdbot For Trading is UNSTOPPABLE (71,057% ROI)
Confidence: 70%

Improved with proper pivot-based divergence detection from web research:
- 5-bar pivot high/low detection for swing identification
- Regular divergence: price lower low + OBV higher low (bullish)
- Hidden divergence: price higher low + OBV lower low (trend continuation)
- Price tolerance filter to avoid noise (0.5-2%)
- RSI confirmation at extremes
- Liquidation boost for high-conviction entries

Transcript insight: "red volume spikes = longs liquidated, green = shorts liquidated"
Web research: 4H optimal for crypto divergence, 5-bar pivots, combine with RSI
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class ObvDivergence(BaseStrategy):
    name = "obv_divergence"
    version = "2.0"

    def default_params(self) -> dict:
        return {
            # OBV divergence detection
            "pivot_bars": 3,              # bars on each side to confirm pivot (loosened from 5)
            "min_pivots": 2,              # need at least 2 pivots to compare
            "lookback_bars": 40,          # window to find pivots (wider for more pivots)
            "price_tolerance_pct": 0.5,   # min % move between pivots (loosened)
            "min_history": 50,

            # Divergence types
            "use_hidden_div": True,       # also trade hidden (continuation) divergence

            # Liquidation boost
            "use_liq_boost": True,
            "liq_percentile": 70,

            # RSI confirmation
            "use_rsi_confirm": True,
            "rsi_period": 14,
            "rsi_oversold": 40,           # loosened from 35
            "rsi_overbought": 60,         # loosened from 65

            # Entry
            "entry_strength": 0.6,
            "boosted_strength": 0.85,

            # Exit
            "stop_loss_pct": 2.5,
            "take_profit_pct": 4.5,
            "max_hold_bars": 24,
            "cooldown_bars": 6,

            "max_history": 200,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._entry_price = 0.0
        self._bars_held = 0
        self._cooldown = 0
        self._obv_history = []

    def _update_obv(self, candle: CandleData):
        """Maintain running OBV using candle direction."""
        if not self._obv_history:
            self._obv_history.append(candle.volume if candle.close >= candle.open else -candle.volume)
        else:
            prev_close = self._candle_history[-2].close if len(self._candle_history) >= 2 else candle.open
            if candle.close > prev_close:
                self._obv_history.append(self._obv_history[-1] + candle.volume)
            elif candle.close < prev_close:
                self._obv_history.append(self._obv_history[-1] - candle.volume)
            else:
                self._obv_history.append(self._obv_history[-1])

        max_hist = self.get_param("max_history")
        if len(self._obv_history) > max_hist:
            self._obv_history = self._obv_history[-max_hist:]

    def _find_pivot_lows(self, values: np.ndarray, n_bars: int) -> list[int]:
        """Find pivot low indices — local minima with n_bars on each side."""
        pivots = []
        for i in range(n_bars, len(values) - n_bars):
            is_pivot = True
            for j in range(1, n_bars + 1):
                if values[i] > values[i - j] or values[i] > values[i + j]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append(i)
        return pivots

    def _find_pivot_highs(self, values: np.ndarray, n_bars: int) -> list[int]:
        """Find pivot high indices — local maxima with n_bars on each side."""
        pivots = []
        for i in range(n_bars, len(values) - n_bars):
            is_pivot = True
            for j in range(1, n_bars + 1):
                if values[i] < values[i - j] or values[i] < values[i + j]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append(i)
        return pivots

    def _detect_divergence(self) -> int:
        """Detect OBV-price divergence using pivot analysis.
        Returns: 1 for bullish, -1 for bearish, 0 for none."""
        lookback = self.get_param("lookback_bars")
        pivot_bars = self.get_param("pivot_bars")
        min_pivots = self.get_param("min_pivots")
        tol = self.get_param("price_tolerance_pct") / 100

        if len(self._obv_history) < lookback or len(self._candle_history) < lookback:
            return 0

        closes = self.closes(lookback)
        obvs = np.array(self._obv_history[-lookback:])

        # --- Bullish divergence: price pivot lows ---
        price_lows = self._find_pivot_lows(closes, pivot_bars)
        if len(price_lows) >= min_pivots:
            # Compare last two pivot lows
            p1, p2 = price_lows[-2], price_lows[-1]
            price_move = (closes[p2] - closes[p1]) / closes[p1]

            # Regular bullish: price lower low, OBV higher low
            if price_move < -tol and obvs[p2] > obvs[p1]:
                # Pivot must be recent (within last pivot_bars+4 of window end)
                if p2 >= lookback - pivot_bars - 4:
                    return 1

            # Hidden bullish: price higher low, OBV lower low (trend continuation)
            if self.get_param("use_hidden_div"):
                if price_move > tol and obvs[p2] < obvs[p1]:
                    if p2 >= lookback - pivot_bars - 4:
                        return 1

        # --- Bearish divergence: price pivot highs ---
        price_highs = self._find_pivot_highs(closes, pivot_bars)
        if len(price_highs) >= min_pivots:
            p1, p2 = price_highs[-2], price_highs[-1]
            price_move = (closes[p2] - closes[p1]) / closes[p1]

            # Regular bearish: price higher high, OBV lower high
            if price_move > tol and obvs[p2] < obvs[p1]:
                if p2 >= lookback - pivot_bars - 4:
                    return -1

            # Hidden bearish: price lower high, OBV higher high
            if self.get_param("use_hidden_div"):
                if price_move < -tol and obvs[p2] > obvs[p1]:
                    if p2 >= lookback - pivot_bars - 4:
                        return -1

        return 0

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        self._update_obv(candle)

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
            else:
                if candle.high >= self._entry_price * (1 + stop_pct):
                    return self._exit("stop_loss")
                if candle.low <= self._entry_price * (1 - tp_pct):
                    return self._exit("take_profit")

            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            return Signal(direction=None)

        # Detect divergence
        div = self._detect_divergence()
        if div == 0:
            return Signal(direction=None)

        # RSI confirmation
        if self.get_param("use_rsi_confirm"):
            rsi_val = self.rsi(self.get_param("rsi_period"))
            if rsi_val is not None:
                if div == 1 and rsi_val > self.get_param("rsi_oversold"):
                    return Signal(direction=None)
                if div == -1 and rsi_val < self.get_param("rsi_overbought"):
                    return Signal(direction=None)

        # Liquidation boost
        strength = self.get_param("entry_strength")
        liq_confirmed = False
        if self.get_param("use_liq_boost") and candle.liquidation_usd > 0:
            liq_values = self.liq_usd(100)
            nonzero = liq_values[liq_values > 0]
            if len(nonzero) >= 10:
                threshold = np.percentile(nonzero, self.get_param("liq_percentile"))
                if candle.liquidation_usd >= threshold:
                    strength = self.get_param("boosted_strength")
                    liq_confirmed = True

        self._in_trade = True
        self._trade_direction = div
        self._entry_price = candle.close
        self._bars_held = 0

        return Signal(
            direction=div,
            strength=strength,
            tag=f"obv_div_{'bull' if div == 1 else 'bear'}",
            metadata={"liq_confirmed": liq_confirmed},
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
    "pivot_bars": [3, 5, 7],
    "lookback_bars": [20, 30, 40],
    "price_tolerance_pct": [0.5, 1.0, 1.5],
    "rsi_period": [10, 14, 20],
    "rsi_oversold": [30, 35, 40],
    "rsi_overbought": [60, 65, 70],
    "liq_percentile": [60, 70, 80],
    "stop_loss_pct": [2.0, 2.5, 3.0],
    "take_profit_pct": [3.5, 4.5, 6.0],
    "max_hold_bars": [16, 24, 32],
}

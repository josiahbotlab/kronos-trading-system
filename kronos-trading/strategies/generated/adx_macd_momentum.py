#!/usr/bin/env python3
"""
ADX Rising + MACD Momentum v2.0
==================================
Source: Clawedbot Internal Quant Zoom Call (Deleting in 72 hours)
Confidence: 75%

Improved with proper Wilder's ADX + directional indicator (DI) system:
- ADX thresholds: 0-20 weak, 20-40 developing, 40-60 strong trend
- Use +DI/-DI crossover for direction (not just MACD)
- MACD histogram crossover as confirmation
- "100% timeframe robustness across all 7 datasets" per transcript
- "Every parameter combo profitable" per transcript

Transcript insights:
- ADX + MACD momentum was extremely robust across timeframes
- Heavy OOS decay but still positive
- "BTC 6-hour reversal is a unicorn edge" (momentum everywhere else)

Web research: Wilder's smoothing (RMA not EMA), ADX > 25 entry,
MACD 12/26/9 standard, +DI/-DI crossover for trend direction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class AdxMacdMomentum(BaseStrategy):
    name = "adx_macd_momentum"
    version = "2.0"

    def default_params(self) -> dict:
        return {
            # ADX with DI crossover
            "adx_period": 14,
            "adx_threshold": 25,
            "adx_rising_bars": 3,         # back to 3 for quality

            # MACD (standard params)
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,

            # DI crossover for direction
            "use_di_direction": False,    # disabled: MACD crossover only (less noise)

            "min_history": 60,

            # Entry
            "entry_strength": 0.75,

            # Exit
            "trailing_stop_pct": 2.0,
            "take_profit_pct": 5.0,
            "max_hold_bars": 24,
            "cooldown_bars": 4,

            # Trend continuation: optional SMA filter
            "use_sma_filter": True,
            "sma_period": 50,

            "max_history": 300,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        self._cooldown = 0
        self._adx_history = []
        self._prev_macd_hist = None

    def _compute_ema_array(self, values: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema = np.zeros_like(values)
        ema[0] = values[0]
        for i in range(1, len(values)):
            ema[i] = alpha * values[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _compute_adx_full(self) -> tuple[float, float, float] | None:
        """Compute ADX, +DI, -DI using Wilder's method.
        Returns (adx, plus_di, minus_di) or None."""
        period = self.get_param("adx_period")
        need = period * 3
        if len(self._candle_history) < need:
            return None

        candles = self._candle_history[-need:]
        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_high = candles[i - 1].high
            prev_low = candles[i - 1].low
            prev_close = candles[i - 1].close

            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)

            up = high - prev_high
            down = prev_low - low
            plus_dm.append(up if up > down and up > 0 else 0)
            minus_dm.append(down if down > up and down > 0 else 0)

        # Wilder's smoothing (RMA) - equivalent to EMA with alpha = 1/period
        alpha = 1 / period
        def wilder_smooth(arr):
            result = np.zeros(len(arr))
            result[0] = arr[0]
            for i in range(1, len(arr)):
                result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
            return result

        tr_arr = np.array(tr_list)
        atr = wilder_smooth(tr_arr)
        smooth_pdm = wilder_smooth(np.array(plus_dm))
        smooth_ndm = wilder_smooth(np.array(minus_dm))

        plus_di = 100 * smooth_pdm / np.maximum(atr, 1e-8)
        minus_di = 100 * smooth_ndm / np.maximum(atr, 1e-8)

        di_sum = plus_di + minus_di
        di_diff = np.abs(plus_di - minus_di)
        dx = 100 * di_diff / np.maximum(di_sum, 1e-8)

        adx = wilder_smooth(dx)
        return float(adx[-1]), float(plus_di[-1]), float(minus_di[-1])

    def _compute_macd(self) -> tuple[float, float, float] | None:
        fast = self.get_param("macd_fast")
        slow = self.get_param("macd_slow")
        sig = self.get_param("macd_signal")
        need = slow + sig + 10

        if len(self._candle_history) < need:
            return None

        closes = self.closes(need)
        fast_ema = self._compute_ema_array(closes, fast)
        slow_ema = self._compute_ema_array(closes, slow)
        macd_line = fast_ema - slow_ema
        signal_line = self._compute_ema_array(macd_line, sig)
        histogram = macd_line - signal_line

        return float(macd_line[-1]), float(signal_line[-1]), float(histogram[-1])

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # Track ADX history
        adx_result = self._compute_adx_full()
        adx_val = None
        plus_di = None
        minus_di = None
        if adx_result is not None:
            adx_val, plus_di, minus_di = adx_result
            self._adx_history.append(adx_val)
            if len(self._adx_history) > 20:
                self._adx_history = self._adx_history[-20:]

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

            # Exit if ADX drops significantly (trend weakening)
            if adx_val is not None and adx_val < self.get_param("adx_threshold") * 0.7:
                return self._exit("adx_weakened")

            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            return Signal(direction=None)

        # ADX check: above threshold
        if adx_val is None or adx_val < self.get_param("adx_threshold"):
            return Signal(direction=None)

        # ADX rising check
        rising_bars = self.get_param("adx_rising_bars")
        if len(self._adx_history) < rising_bars + 1:
            return Signal(direction=None)
        adx_rising = all(
            self._adx_history[-(i)] > self._adx_history[-(i + 1)]
            for i in range(1, rising_bars + 1)
        )
        if not adx_rising:
            return Signal(direction=None)

        # MACD check
        macd = self._compute_macd()
        if macd is None:
            return Signal(direction=None)
        macd_line, signal_line, histogram = macd

        if self._prev_macd_hist is None:
            self._prev_macd_hist = histogram
            return Signal(direction=None)

        prev_hist = self._prev_macd_hist
        self._prev_macd_hist = histogram

        # Determine direction
        direction = None

        if self.get_param("use_di_direction") and plus_di is not None:
            # DI crossover: +DI > -DI = bullish, +DI < -DI = bearish
            # Confirmed by MACD histogram direction
            if plus_di > minus_di and histogram > 0:
                direction = 1
            elif minus_di > plus_di and histogram < 0:
                direction = -1
        else:
            # Fallback: MACD histogram crossover
            if prev_hist <= 0 and histogram > 0:
                direction = 1
            elif prev_hist >= 0 and histogram < 0:
                direction = -1

        if direction is None:
            return Signal(direction=None)

        # SMA trend filter
        if self.get_param("use_sma_filter"):
            sma_val = self.sma(self.get_param("sma_period"))
            if sma_val is not None:
                if direction == 1 and candle.close < sma_val:
                    return Signal(direction=None)
                if direction == -1 and candle.close > sma_val:
                    return Signal(direction=None)

        self._in_trade = True
        self._trade_direction = direction
        self._bars_held = 0
        self._peak = candle.high
        self._trough = candle.low

        return Signal(
            direction=direction,
            strength=self.get_param("entry_strength"),
            tag=f"adx_macd_{'bull' if direction == 1 else 'bear'}",
            metadata={"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di, "macd_hist": histogram},
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
    "adx_period": [10, 14, 20],
    "adx_threshold": [20, 25, 30],
    "adx_rising_bars": [1, 2, 3],
    "macd_fast": [8, 12, 16],
    "macd_slow": [20, 26, 30],
    "trailing_stop_pct": [1.5, 2.0, 3.0],
    "take_profit_pct": [4.0, 5.0, 7.0],
    "max_hold_bars": [16, 24, 32],
}

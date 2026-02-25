#!/usr/bin/env python3
"""
Kalman Filter Bollinger Band Breakout v1.0
============================================
Source: Moon Dev Free Trading Bot Course (c154a5CDr2w)
Confidence: 90%

Replaces the SMA center line in Bollinger Bands with a 1D Kalman filter
for dynamic, adaptive price smoothing. The Kalman filter reacts faster
than SMA to regime changes while producing smoother signals.

Transcript insights:
- Tests std deviations between 2.0 and 4.0
- Specific test at 2.7 std, optimized down to 1.5
- "Kalman filter for dynamic price smoothing as the center line"
- Bidirectional: long breakouts above upper band, short below lower
- Tested on 1m and 1d timeframes

Web research (Kalman filter for trading):
- 1D scalar Kalman: predict step (x_minus = x, P_minus = P + Q)
  update step (K = P_minus/(P_minus+R), x = x_minus + K*(z-x_minus))
- Q (process noise): 1e-5 (smooth) to 1e-3 (responsive)
- R (measurement noise): 0.01 to 0.1
- Q/R ratio controls smoothness: 0.01 = like 200-MA, 0.1 = like 20-EMA
- Converges after ~20-50 bars
- Std dev computed from residuals (price - kalman estimate)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class KalmanBbBreakout(BaseStrategy):
    name = "kalman_bb_breakout"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Kalman filter tuning
            "kalman_q": 1e-3,             # more responsive for 1h data
            "kalman_r": 0.05,             # measurement noise (higher = smoother)

            # Band parameters
            "band_std_mult": 3.0,         # wider bands = fewer false breakouts
            "residual_lookback": 20,      # bars for std dev of residuals

            # Volume confirmation on breakout
            "use_vol_confirm": False,
            "vol_lookback": 20,
            "vol_mult": 1.2,             # volume > 1.2x average

            "min_history": 50,

            # Entry
            "entry_strength": 0.75,

            # Exit
            "trailing_stop_pct": 2.5,
            "take_profit_pct": 6.0,
            "max_hold_bars": 24,
            "cooldown_bars": 3,

            # Mean reversion exit: price returns to Kalman line
            "exit_on_kalman_cross": False,  # disabled — was exiting too early

            "max_history": 300,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        self._cooldown = 0

        # Kalman filter state
        self._kalman_x = None    # state estimate (price)
        self._kalman_p = 1.0     # estimate covariance
        self._residuals = []     # price - kalman for std dev calc
        self._prev_close = None  # for breakout detection

    def _kalman_update(self, price: float) -> float:
        """Run one step of 1D Kalman filter. Returns smoothed estimate."""
        Q = self.get_param("kalman_q")
        R = self.get_param("kalman_r")

        if self._kalman_x is None:
            self._kalman_x = price
            return price

        # Predict
        x_minus = self._kalman_x
        p_minus = self._kalman_p + Q

        # Update
        K = p_minus / (p_minus + R)
        self._kalman_x = x_minus + K * (price - x_minus)
        self._kalman_p = (1 - K) * p_minus

        return self._kalman_x

    def _get_kalman_bands(self) -> tuple[float, float, float] | None:
        """Get Kalman-based Bollinger Bands: (upper, kalman_mid, lower)."""
        lookback = self.get_param("residual_lookback")
        if len(self._residuals) < lookback:
            return None

        mid = self._kalman_x
        recent_residuals = np.array(self._residuals[-lookback:])
        std = float(np.std(recent_residuals, ddof=1))

        if std < 1e-10:
            return None

        mult = self.get_param("band_std_mult")
        return (mid + mult * std, mid, mid - mult * std)

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # Update Kalman filter
        kalman_val = self._kalman_update(candle.close)

        # Track residuals for band calculation
        residual = candle.close - kalman_val
        self._residuals.append(residual)
        max_res = self.get_param("max_history")
        if len(self._residuals) > max_res:
            self._residuals = self._residuals[-max_res:]

        # --- IN POSITION ---
        if self._in_trade:
            self._bars_held += 1

            if self._trade_direction == 1:
                self._peak = max(self._peak, candle.high)
                stop = self._peak * (1 - self.get_param("trailing_stop_pct") / 100)
                if candle.low <= stop:
                    self._prev_close = candle.close
                    return self._exit("trailing_stop")

                # Mean reversion exit: price crosses back below Kalman
                if self.get_param("exit_on_kalman_cross") and candle.close < kalman_val:
                    self._prev_close = candle.close
                    return self._exit("kalman_cross")
            else:
                self._trough = min(self._trough, candle.low)
                stop = self._trough * (1 + self.get_param("trailing_stop_pct") / 100)
                if candle.high >= stop:
                    self._prev_close = candle.close
                    return self._exit("trailing_stop")

                # Mean reversion exit: price crosses back above Kalman
                if self.get_param("exit_on_kalman_cross") and candle.close > kalman_val:
                    self._prev_close = candle.close
                    return self._exit("kalman_cross")

            if self._bars_held >= self.get_param("max_hold_bars"):
                self._prev_close = candle.close
                return self._exit("max_hold")

            self._prev_close = candle.close
            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            self._prev_close = candle.close
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            self._prev_close = candle.close
            return Signal(direction=None)

        bands = self._get_kalman_bands()
        if bands is None:
            self._prev_close = candle.close
            return Signal(direction=None)

        upper, mid, lower = bands

        # Need previous close for crossover detection
        if self._prev_close is None:
            self._prev_close = candle.close
            return Signal(direction=None)

        direction = None

        # Breakout: close crosses above upper band
        if self._prev_close <= upper and candle.close > upper:
            direction = 1
        # Breakdown: close crosses below lower band
        elif self._prev_close >= lower and candle.close < lower:
            direction = -1

        self._prev_close = candle.close

        if direction is None:
            return Signal(direction=None)

        # Volume confirmation (optional)
        if self.get_param("use_vol_confirm"):
            vols = self.volumes(self.get_param("vol_lookback"))
            if len(vols) >= self.get_param("vol_lookback"):
                avg_vol = np.mean(vols[:-1])
                if avg_vol > 0 and candle.volume < avg_vol * self.get_param("vol_mult"):
                    return Signal(direction=None)

        self._in_trade = True
        self._trade_direction = direction
        self._bars_held = 0
        self._peak = candle.high
        self._trough = candle.low

        return Signal(
            direction=direction,
            strength=self.get_param("entry_strength"),
            tag=f"kalman_{'bull' if direction == 1 else 'bear'}",
            metadata={"kalman": kalman_val, "upper": upper, "lower": lower, "residual_std": (upper - mid) / self.get_param("band_std_mult")},
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
    "kalman_q": [1e-5, 1e-4, 1e-3],
    "kalman_r": [0.01, 0.05, 0.1],
    "band_std_mult": [1.5, 2.0, 2.5, 3.0],
    "residual_lookback": [15, 20, 30],
    "trailing_stop_pct": [1.5, 2.0, 3.0],
    "take_profit_pct": [4.0, 5.0, 7.0],
    "max_hold_bars": [12, 20, 28],
}

#!/usr/bin/env python3
"""
Consecutive Down Day Reversal v1.0
====================================
Source: Moon Dev RBI Backtest Session (RkKB725yyn4)
Confidence: 90%

Larry Connors-style mean reversion: buy after N consecutive lower closes
within an established uptrend. Tested with ADX, MFI, VWAP, Kalman filters.

Transcript insights:
- "4 consecutive bars downward, then enter a trade"
- "long only" — only buy dips, never short
- Tested 5 variations: base, ADX, MFI, VWAP, Kaufman filter
- Multi-dataset testing across timeframes (1m, 5m, 15m, 1h, 1d, gap data)
- Optimize take profit and stop loss parameters

Web research (Larry Connors' High/Low Method):
- Trend filter: close > 200-SMA (only trade pullbacks in uptrends)
- Pullback filter: close < 5-SMA (confirms short-term weakness)
- RSI-2 < 10 for extreme oversold (strongest single filter)
- Exit: close > 5-SMA (mean reversion complete)
- Win rates typically 65-80% with proper filters
- Critical: "consecutive down closes" = close < previous close, NOT red candle
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class ConsecutiveDownReversal(BaseStrategy):
    name = "consecutive_down_reversal"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Core signal
            "consec_down_bars": 4,         # consecutive lower closes to trigger

            # Trend filter (100-SMA — relaxed from 200 for 1h BTC)
            "use_trend_filter": True,
            "trend_sma_period": 100,

            # Pullback filter (5-SMA)
            "use_pullback_filter": True,
            "pullback_sma_period": 5,

            # RSI-2 confirmation
            "use_rsi_confirm": True,
            "rsi_period": 2,
            "rsi_threshold": 25,           # moderate (more signals than 10)

            # Volume confirmation (optional)
            "use_vol_confirm": False,
            "vol_lookback": 20,
            "vol_spike_mult": 1.5,         # volume > 1.5x avg

            "min_history": 110,            # need 100 bars for SMA

            # Entry
            "entry_strength": 0.7,

            # Exit: close > 10-SMA (wider target, less premature exits)
            "exit_sma_period": 10,
            "max_hold_bars": 16,           # allow more time for reversion
            "stop_loss_pct": 4.0,          # wider stop for mean reversion
            "cooldown_bars": 2,

            "max_history": 500,
        }

    def on_init(self):
        self._in_trade = False
        self._entry_price = 0.0
        self._bars_held = 0
        self._cooldown = 0

    def _count_consecutive_down(self) -> int:
        """Count consecutive lower closes (close < prev close)."""
        count = 0
        candles = self._candle_history
        for i in range(len(candles) - 1, 0, -1):
            if candles[i].close < candles[i - 1].close:
                count += 1
            else:
                break
        return count

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # --- IN POSITION ---
        if self._in_trade:
            self._bars_held += 1

            # Exit 1: Close above exit SMA (mean reversion complete)
            exit_sma = self.sma(self.get_param("exit_sma_period"))
            if exit_sma is not None and candle.close > exit_sma:
                return self._exit("sma_cross")

            # Exit 2: Stop loss
            stop = self._entry_price * (1 - self.get_param("stop_loss_pct") / 100)
            if candle.low <= stop:
                return self._exit("stop_loss")

            # Exit 3: Max hold
            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            return Signal(direction=None)

        # --- NO POSITION ---
        if self._cooldown > 0:
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            return Signal(direction=None)

        # Check consecutive down closes
        consec = self._count_consecutive_down()
        if consec < self.get_param("consec_down_bars"):
            return Signal(direction=None)

        # Trend filter: close > 200-SMA (only buy in uptrends)
        if self.get_param("use_trend_filter"):
            trend_sma = self.sma(self.get_param("trend_sma_period"))
            if trend_sma is not None and candle.close < trend_sma:
                return Signal(direction=None)

        # Pullback filter: close < 5-SMA (confirms short-term weakness)
        if self.get_param("use_pullback_filter"):
            pullback_sma = self.sma(self.get_param("pullback_sma_period"))
            if pullback_sma is not None and candle.close > pullback_sma:
                return Signal(direction=None)

        # RSI-2 confirmation
        if self.get_param("use_rsi_confirm"):
            rsi_val = self.rsi(self.get_param("rsi_period"))
            if rsi_val is not None and rsi_val > self.get_param("rsi_threshold"):
                return Signal(direction=None)

        # Volume confirmation (optional)
        if self.get_param("use_vol_confirm"):
            vols = self.volumes(self.get_param("vol_lookback"))
            if len(vols) >= self.get_param("vol_lookback"):
                avg_vol = np.mean(vols[:-1])
                if avg_vol > 0 and candle.volume < avg_vol * self.get_param("vol_spike_mult"):
                    return Signal(direction=None)

        # LONG entry: buy the dip
        self._in_trade = True
        self._entry_price = candle.close
        self._bars_held = 0

        return Signal(
            direction=1,
            strength=self.get_param("entry_strength"),
            tag=f"consec_down_{consec}",
            metadata={"consecutive_down": consec},
        )

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False


PARAM_RANGES = {
    "consec_down_bars": [3, 4, 5],
    "trend_sma_period": [100, 200, 300],
    "rsi_period": [2, 3, 5],
    "rsi_threshold": [10, 15, 25],
    "exit_sma_period": [5, 10, 15],
    "stop_loss_pct": [2.0, 3.0, 4.0],
    "max_hold_bars": [8, 12, 16],
}

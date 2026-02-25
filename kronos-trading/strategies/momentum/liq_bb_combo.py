#!/usr/bin/env python3
"""
Liquidation Bollinger Combo
============================
Combines liquidation cascade detection with Bollinger Band analysis.

Two modes:
1. MOMENTUM: Cascade + BB breakout = ride the move
   - Liquidation cascade detected AND price breaks above/below BB
   - Strong confirmation of directional move

2. REVERSAL: Cascade + BB extreme = fade the move
   - Cascade exhaustion AND price at BB extreme
   - Mean reversion after overextension

Moon Dev insight: Combining liquidation data with technical indicators
produces higher-conviction signals than either alone.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class LiquidationBollingerCombo(BaseStrategy):
    name = "liq_bb_combo"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Mode
            "mode": "momentum",              # "momentum" or "reversal"

            # Liquidation thresholds
            "liq_percentile": 80,            # P80 for momentum, P90+ for reversal
            "lookback_bars": 200,
            "min_history": 50,
            "liq_direction_threshold": 0.6,

            # Bollinger Bands
            "bb_period": 20,
            "bb_std": 2.0,
            "bb_squeeze_threshold": 0.5,     # BB width relative to price (for squeeze detection)

            # RSI filter
            "use_rsi": True,
            "rsi_period": 14,

            # Momentum mode exits
            "momentum_trailing_stop_pct": 2.5,
            "momentum_take_profit_pct": 6.0,
            "momentum_max_hold": 20,

            # Reversal mode exits
            "reversal_stop_loss_pct": 1.5,
            "reversal_take_profit_pct": 3.0,
            "reversal_max_hold": 12,

            # General
            "entry_strength": 0.8,
            "cooldown_bars": 5,

            "max_history": 300,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_dir = 0
        self._entry_price = 0.0
        self._peak = 0.0
        self._trough = float('inf')
        self._bars_held = 0
        self._cooldown = 0
        self._trade_mode = ""

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # --- IN TRADE ---
        if self._in_trade:
            self._bars_held += 1
            return self._manage_position(candle)

        # --- NO POSITION ---
        if self._cooldown > 0 or len(self._candle_history) < self.get_param("min_history"):
            return Signal(direction=None)

        mode = self.get_param("mode")
        if mode == "momentum":
            return self._momentum_logic(candle)
        else:
            return self._reversal_logic(candle)

    def _momentum_logic(self, candle: CandleData) -> Signal:
        """
        Momentum: Cascade + BB breakout.
        Liquidation cascade + price breaking BB = strong trend confirmation.
        """
        # Check for cascade
        cascade_dir = self._detect_cascade(candle)
        if cascade_dir == 0:
            return Signal(direction=None)

        # Check Bollinger Band breakout
        bb = self.bollinger_bands(self.get_param("bb_period"), self.get_param("bb_std"))
        if not bb:
            return Signal(direction=None)

        upper, middle, lower = bb

        # Bullish: cascade up + price above upper BB (or breaking above middle)
        if cascade_dir == 1 and candle.close > middle:
            # RSI filter: not already extremely overbought
            if self.get_param("use_rsi"):
                rsi_val = self.rsi(self.get_param("rsi_period"))
                if rsi_val and rsi_val > 85:
                    return Signal(direction=None)

            return self._enter(1, "momentum_bull_bb")

        # Bearish: cascade down + price below lower BB (or breaking below middle)
        if cascade_dir == -1 and candle.close < middle:
            if self.get_param("use_rsi"):
                rsi_val = self.rsi(self.get_param("rsi_period"))
                if rsi_val and rsi_val < 15:
                    return Signal(direction=None)

            return self._enter(-1, "momentum_bear_bb")

        return Signal(direction=None)

    def _reversal_logic(self, candle: CandleData) -> Signal:
        """
        Reversal: Cascade exhaustion + BB extreme.
        After big cascade, price at BB extreme = overextended, fade it.
        """
        # Need recent cascade (look back a few bars)
        recent_liqs = self.liq_usd(5)
        if len(recent_liqs) < 3:
            return Signal(direction=None)

        # Was there a cascade in last 3-5 bars?
        max_recent_liq = np.max(recent_liqs[:-1]) if len(recent_liqs) > 1 else 0

        liq_values = self.liq_usd(self.get_param("lookback_bars"))
        nonzero = liq_values[liq_values > 0]
        if len(nonzero) < 20:
            return Signal(direction=None)

        threshold = np.percentile(nonzero, self.get_param("liq_percentile"))
        if max_recent_liq < threshold:
            return Signal(direction=None)

        # Current bar should show decay
        if candle.liquidation_usd > max_recent_liq * 0.3:
            return Signal(direction=None)  # still cascading

        # Check BB extreme
        bb = self.bollinger_bands(self.get_param("bb_period"), self.get_param("bb_std"))
        if not bb:
            return Signal(direction=None)

        upper, middle, lower = bb

        # Price above upper BB after bullish cascade -> fade short
        if candle.close > upper:
            if self.get_param("use_rsi"):
                rsi_val = self.rsi(self.get_param("rsi_period"))
                if rsi_val and rsi_val < 65:
                    return Signal(direction=None)
            return self._enter(-1, "reversal_fade_upper_bb")

        # Price below lower BB after bearish cascade -> fade long
        if candle.close < lower:
            if self.get_param("use_rsi"):
                rsi_val = self.rsi(self.get_param("rsi_period"))
                if rsi_val and rsi_val > 35:
                    return Signal(direction=None)
            return self._enter(1, "reversal_fade_lower_bb")

        return Signal(direction=None)

    def _detect_cascade(self, candle: CandleData) -> int:
        """Detect liquidation cascade."""
        liq_values = self.liq_usd(self.get_param("lookback_bars"))
        nonzero = liq_values[liq_values > 0]
        if len(nonzero) < 20:
            return 0

        threshold = np.percentile(nonzero, self.get_param("liq_percentile"))
        if candle.liquidation_usd < threshold:
            return 0

        total = candle.liquidation_usd
        if total == 0:
            return 0

        dir_thresh = self.get_param("liq_direction_threshold")
        if candle.short_liq_usd / total >= dir_thresh:
            return 1
        elif candle.long_liq_usd / total >= dir_thresh:
            return -1
        return 0

    def _enter(self, direction: int, tag: str) -> Signal:
        self._in_trade = True
        self._trade_dir = direction
        self._trade_mode = self.get_param("mode")
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        return Signal(
            direction=direction,
            strength=self.get_param("entry_strength"),
            tag=tag,
        )

    def _manage_position(self, candle: CandleData) -> Signal:
        """Position management based on trade mode."""
        if self._trade_mode == "momentum":
            return self._manage_momentum(candle)
        else:
            return self._manage_reversal(candle)

    def _manage_momentum(self, candle: CandleData) -> Signal:
        """Trailing stop for momentum trades."""
        trailing_pct = self.get_param("momentum_trailing_stop_pct") / 100

        if self._trade_dir == 1:
            self._peak = max(self._peak, candle.high) if self._peak > 0 else candle.high
            stop = self._peak * (1 - trailing_pct)
            if candle.low <= stop:
                return self._exit("trailing_stop")
        else:
            self._trough = min(self._trough, candle.low)
            stop = self._trough * (1 + trailing_pct)
            if candle.high >= stop:
                return self._exit("trailing_stop")

        if self._bars_held >= self.get_param("momentum_max_hold"):
            return self._exit("max_hold")

        return Signal(direction=None)

    def _manage_reversal(self, candle: CandleData) -> Signal:
        """Fixed stop/target for reversal trades."""
        sl = self.get_param("reversal_stop_loss_pct") / 100
        tp = self.get_param("reversal_take_profit_pct") / 100

        if not self._entry_price:
            self._entry_price = candle.open

        if self._trade_dir == 1:
            if candle.low <= self._entry_price * (1 - sl):
                return self._exit("stop_loss")
            if candle.high >= self._entry_price * (1 + tp):
                return self._exit("take_profit")
        else:
            if candle.high >= self._entry_price * (1 + sl):
                return self._exit("stop_loss")
            if candle.low <= self._entry_price * (1 - tp):
                return self._exit("take_profit")

        if self._bars_held >= self.get_param("reversal_max_hold"):
            return self._exit("max_hold")

        return Signal(direction=None)

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._trade_dir = 0
        self._entry_price = 0.0
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False
        self._trade_dir = 0

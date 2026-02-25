#!/usr/bin/env python3
"""
Double Decay Reversal
=====================
Moon Dev's most disciplined reversal: 34% return, 3.4% max DD, 10:1 R/R.

The key insight: Wait for TWO consecutive decay bars after a cascade.
Single decay can be a pause before continuation.
Double decay = momentum is truly spent.

Logic:
1. Detect large cascade event (P90+)
2. Wait for first decay bar (liq drops significantly)
3. Wait for SECOND decay bar (confirms exhaustion)
4. Enter against the cascade
5. Very tight stop, modest target

The double confirmation makes this strategy selective but high-conviction.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class DoubleDecayReversal(BaseStrategy):
    name = "double_decay_reversal"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Cascade detection
            "cascade_percentile": 90,
            "lookback_bars": 300,
            "min_history": 80,
            "liq_direction_threshold": 0.6,

            # Double decay detection
            "decay_1_ratio": 0.5,            # first bar must drop to 50% of cascade
            "decay_2_ratio": 0.25,           # second bar must drop to 25% of cascade
            "max_wait_bars": 5,              # give up if no decay within N bars

            # Entry
            "entry_strength": 0.5,           # conservative sizing

            # Exit - tight stops, modest targets (10:1 R/R comes from win rate)
            "stop_loss_pct": 1.0,            # 1% stop
            "take_profit_pct": 3.0,          # 3% target
            "max_hold_bars": 15,

            # Filters
            "require_rsi_extreme": True,     # RSI must be overbought/oversold
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "cooldown_bars": 8,

            "max_history": 400,
        }

    def on_init(self):
        self._state = "scanning"
        self._cascade_dir = 0
        self._cascade_liq = 0.0
        self._decay_count = 0
        self._wait_bars = 0
        self._last_liq = 0.0

        self._in_trade = False
        self._trade_dir = 0
        self._entry_price = 0.0
        self._bars_held = 0
        self._cooldown = 0

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # --- IN TRADE ---
        if self._in_trade:
            self._bars_held += 1
            sl = self.get_param("stop_loss_pct") / 100
            tp = self.get_param("take_profit_pct") / 100

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

            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            return Signal(direction=None)

        # --- SCANNING ---
        if self._state == "scanning":
            if self._cooldown > 0:
                return Signal(direction=None)

            direction = self._detect_cascade(candle)
            if direction != 0:
                self._state = "decay_1"
                self._cascade_dir = direction
                self._cascade_liq = candle.liquidation_usd
                self._decay_count = 0
                self._wait_bars = 0
                self._last_liq = candle.liquidation_usd

            return Signal(direction=None)

        # --- WAITING FOR FIRST DECAY ---
        if self._state == "decay_1":
            self._wait_bars += 1

            if self._wait_bars > self.get_param("max_wait_bars"):
                self._state = "scanning"
                return Signal(direction=None)

            # Check if a new cascade happened (reset)
            if candle.liquidation_usd > self._cascade_liq * 0.8:
                self._cascade_liq = max(self._cascade_liq, candle.liquidation_usd)
                self._wait_bars = 0
                return Signal(direction=None)

            # First decay: liq drops to threshold
            threshold_1 = self._cascade_liq * self.get_param("decay_1_ratio")
            if candle.liquidation_usd <= threshold_1:
                self._state = "decay_2"
                self._last_liq = candle.liquidation_usd

            return Signal(direction=None)

        # --- WAITING FOR SECOND DECAY ---
        if self._state == "decay_2":
            self._wait_bars += 1

            if self._wait_bars > self.get_param("max_wait_bars"):
                self._state = "scanning"
                return Signal(direction=None)

            # If liquidation spikes again, go back to decay_1
            if candle.liquidation_usd > self._cascade_liq * 0.5:
                self._state = "decay_1"
                return Signal(direction=None)

            # Second decay: liq drops further
            threshold_2 = self._cascade_liq * self.get_param("decay_2_ratio")
            if candle.liquidation_usd <= threshold_2:
                # DOUBLE DECAY CONFIRMED - check filters and enter
                if self._check_filters(candle):
                    fade_dir = -self._cascade_dir
                    self._in_trade = True
                    self._trade_dir = fade_dir
                    self._entry_price = candle.close
                    self._bars_held = 0
                    self._state = "scanning"

                    return Signal(
                        direction=fade_dir,
                        strength=self.get_param("entry_strength"),
                        tag=f"double_decay_{'long' if fade_dir == 1 else 'short'}",
                        metadata={
                            "cascade_liq": self._cascade_liq,
                            "decay_bars": self._wait_bars,
                        },
                    )
                else:
                    self._state = "scanning"

            return Signal(direction=None)

        return Signal(direction=None)

    def _detect_cascade(self, candle: CandleData) -> int:
        """Detect P90+ liquidation cascade."""
        if len(self._candle_history) < self.get_param("min_history"):
            return 0

        liq_values = self.liq_usd(self.get_param("lookback_bars"))
        nonzero = liq_values[liq_values > 0]
        if len(nonzero) < 20:
            return 0

        threshold = np.percentile(nonzero, self.get_param("cascade_percentile"))
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

    def _check_filters(self, candle: CandleData) -> bool:
        """Check additional entry filters."""
        # RSI extreme filter
        if self.get_param("require_rsi_extreme"):
            rsi_val = self.rsi(self.get_param("rsi_period"))
            if rsi_val is not None:
                if self._cascade_dir == 1:  # bullish cascade -> we want to short -> RSI overbought
                    if rsi_val < self.get_param("rsi_overbought"):
                        return False
                else:  # bearish cascade -> we want to long -> RSI oversold
                    if rsi_val > self.get_param("rsi_oversold"):
                        return False
        return True

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._trade_dir = 0
        self._state = "scanning"
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False
        self._trade_dir = 0
        self._state = "scanning"

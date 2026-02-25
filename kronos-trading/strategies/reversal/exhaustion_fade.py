#!/usr/bin/env python3
"""
Exhaustion Fade Strategy
========================
Reversal strategy that fades after a liquidation cascade exhausts itself.

Logic:
- Wait for a massive cascade (P95+ liquidation event)
- Wait for "exhaustion" signals: volume drops, liquidation rate slows
- Enter AGAINST the cascade direction (mean reversion)
- Tight stop (if wrong, the cascade continues and we're out fast)

Key insight: After extreme liquidation events, the forced buying/selling
is done. The overextended price snaps back.

Moon Dev notes: Only 4/20 reversal strategies profitable, but the ones
that work have incredible risk/reward.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class ExhaustionFade(BaseStrategy):
    name = "exhaustion_fade"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Cascade detection (need big event first)
            "cascade_percentile": 95,        # P95+ event triggers watch mode
            "lookback_bars": 300,
            "min_history": 100,

            # Exhaustion detection
            "decay_bars": 3,                 # wait N bars for decay after cascade
            "decay_ratio": 0.3,              # liq must drop to 30% of cascade level
            "price_extension_atr": 2.0,      # price must be 2 ATR from mean

            # Entry
            "use_bb_confirmation": True,      # require price outside Bollinger Bands
            "bb_period": 20,
            "bb_std": 2.0,

            # Exit
            "stop_loss_pct": 1.5,            # tight stop (wrong = cascade continues)
            "take_profit_pct": 3.0,          # modest target (mean reversion)
            "max_hold_bars": 20,

            # Risk
            "entry_strength": 0.5,           # half size (reversal = risky)
            "cooldown_bars": 10,

            "max_history": 400,
        }

    def on_init(self):
        self._state = "scanning"  # scanning -> watching -> ready -> in_trade
        self._cascade_direction = 0  # 1=bullish cascade, -1=bearish cascade
        self._cascade_liq_usd = 0.0
        self._watch_countdown = 0
        self._in_trade = False
        self._trade_direction = 0
        self._entry_price = 0.0
        self._bars_held = 0
        self._cooldown = 0

    def on_candle(self, candle: CandleData) -> Signal:
        if self._cooldown > 0:
            self._cooldown -= 1

        # --- IN TRADE ---
        if self._in_trade:
            self._bars_held += 1

            stop_pct = self.get_param("stop_loss_pct") / 100
            tp_pct = self.get_param("take_profit_pct") / 100

            if self._trade_direction == 1:  # we went long (fading bearish cascade)
                if candle.low <= self._entry_price * (1 - stop_pct):
                    return self._exit("stop_loss")
                if candle.high >= self._entry_price * (1 + tp_pct):
                    return self._exit("take_profit")
            else:  # we went short (fading bullish cascade)
                if candle.high >= self._entry_price * (1 + stop_pct):
                    return self._exit("stop_loss")
                if candle.low <= self._entry_price * (1 - tp_pct):
                    return self._exit("take_profit")

            if self._bars_held >= self.get_param("max_hold_bars"):
                return self._exit("max_hold")

            return Signal(direction=None)

        # --- SCANNING: Look for P95+ cascade ---
        if self._state == "scanning":
            if self._cooldown > 0:
                return Signal(direction=None)

            cascade_dir = self._detect_extreme_cascade(candle)
            if cascade_dir != 0:
                self._state = "watching"
                self._cascade_direction = cascade_dir
                self._cascade_liq_usd = candle.liquidation_usd
                self._watch_countdown = self.get_param("decay_bars")

            return Signal(direction=None)

        # --- WATCHING: Wait for exhaustion ---
        if self._state == "watching":
            self._watch_countdown -= 1

            if self._watch_countdown <= 0:
                # Check for exhaustion
                if self._check_exhaustion(candle):
                    # Enter AGAINST the cascade
                    fade_direction = -self._cascade_direction
                    self._in_trade = True
                    self._trade_direction = fade_direction
                    self._entry_price = candle.close
                    self._bars_held = 0
                    self._state = "scanning"

                    return Signal(
                        direction=fade_direction,
                        strength=self.get_param("entry_strength"),
                        tag=f"exhaustion_fade_{'long' if fade_direction == 1 else 'short'}",
                    )
                else:
                    # No exhaustion detected, cascade still going
                    self._state = "scanning"

            # If a NEW cascade happens in same direction while watching, reset
            if candle.liquidation_usd > self._cascade_liq_usd * 0.5:
                self._watch_countdown = self.get_param("decay_bars")

            return Signal(direction=None)

        return Signal(direction=None)

    def _detect_extreme_cascade(self, candle: CandleData) -> int:
        """Detect P95+ liquidation event. Returns cascade direction."""
        min_hist = self.get_param("min_history")
        if len(self._candle_history) < min_hist:
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

        short_ratio = candle.short_liq_usd / total
        long_ratio = candle.long_liq_usd / total

        if short_ratio >= 0.6:
            return 1   # bullish cascade (shorts rekt)
        elif long_ratio >= 0.6:
            return -1  # bearish cascade (longs rekt)
        return 0

    def _check_exhaustion(self, candle: CandleData) -> bool:
        """Check if cascade has exhausted itself."""
        decay_ratio = self.get_param("decay_ratio")

        # Liquidation must have dropped significantly
        if candle.liquidation_usd > self._cascade_liq_usd * decay_ratio:
            return False  # still cascading

        # Optional: check Bollinger Band extension
        if self.get_param("use_bb_confirmation"):
            bb = self.bollinger_bands(
                self.get_param("bb_period"),
                self.get_param("bb_std"),
            )
            if bb:
                upper, middle, lower = bb
                if self._cascade_direction == 1:
                    # Bullish cascade pushed price up -> should be near upper BB
                    if candle.close < upper:
                        return False  # not extended enough
                elif self._cascade_direction == -1:
                    # Bearish cascade pushed price down -> should be near lower BB
                    if candle.close > lower:
                        return False  # not extended enough

        return True

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._trade_direction = 0
        self._state = "scanning"
        self._cooldown = self.get_param("cooldown_bars")
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False
        self._trade_direction = 0
        self._state = "scanning"

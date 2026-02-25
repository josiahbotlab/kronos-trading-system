#!/usr/bin/env python3
"""
Liquidation Sniping (Lick Momentum) v1.0
==========================================
Source: Clawedbot Internal Quant Zoom Call (Deleting in 72 hours)
Confidence: 75%

Distinct from cascade_p99 by adding:
1. Reversal protection filter (avoids entering when momentum might reverse)
2. Double-confirmation: 2 consecutive momentum bars AFTER P99 event
3. Shorter hold period optimized for momentum capture
4. Directional bias from long/short liquidation ratio

Transcript insights:
- "5-minute is the ONLY timeframe that works for momentum" (we use 1h, adapted)
- "P99 threshold beats P80 every time"
- "Cascade P99 H8: 425% return, 13% DD, 2.98 Sharpe, 62% WR, ~1000 trades"
- "Reversal protection filter keeps trade count low, avoids bad entries"
- "PF 0.94, avg loss > avg win at 50% WR" → needs tighter risk management
- "Two consecutive decay bars after cascade before entry gives insane precision"
  (from double_decay strategy insight, applied here as consecutive momentum bars)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class LiquidationSniping(BaseStrategy):
    name = "liquidation_sniping"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Liquidation threshold
            "liq_percentile": 95,         # P95 on 1h (P99 on 5m equivalent)
            "liq_lookback": 200,          # bars for percentile calculation
            "min_liq_usd": 500,           # lowered for 1h granularity

            # Double confirmation: N consecutive momentum bars after P99 event
            "confirm_bars": 1,            # 1 bar on 1h (2 bars on 5m equiv)
            "confirm_body_atr": 0.2,      # lowered for 1h candles

            # Reversal protection: check if momentum is weakening
            "use_reversal_protection": True,
            "rsi_period": 14,
            "rsi_reversal_long": 65,      # don't go long if RSI already > 65
            "rsi_reversal_short": 35,     # don't go short if RSI already < 35

            # Directional bias from liq type
            "use_liq_direction": True,    # use long/short liq ratio for direction

            "atr_period": 14,
            "min_history": 50,

            # Entry
            "entry_strength": 0.75,

            # Exit (tighter than cascade_p99 for momentum capture)
            "stop_loss_pct": 1.5,         # tighter stop
            "take_profit_pct": 3.0,       # modest target
            "trailing_stop_pct": 1.0,     # tight trail for momentum
            "trail_after_bars": 2,
            "max_hold_bars": 12,          # short hold for momentum
            "cooldown_bars": 4,

            "max_history": 300,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._entry_price = 0.0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        self._cooldown = 0
        self._pending_direction = 0      # direction from P99 event
        self._confirm_count = 0          # bars of confirmation seen

    def _is_p99_event(self, candle: CandleData) -> bool:
        """Check if current liquidation is P99 level."""
        if candle.liquidation_usd < self.get_param("min_liq_usd"):
            return False

        liq_values = self.liq_usd(self.get_param("liq_lookback"))
        nonzero = liq_values[liq_values > 0]
        if len(nonzero) < 30:
            return False

        threshold = np.percentile(nonzero, self.get_param("liq_percentile"))
        return candle.liquidation_usd >= threshold

    def _get_momentum_direction(self, candle: CandleData) -> int:
        """Get direction from liquidation type or candle body."""
        if self.get_param("use_liq_direction"):
            total = candle.long_liq_usd + candle.short_liq_usd
            if total > 0:
                # More shorts liquidated → price going up → momentum is long
                if candle.short_liq_usd > candle.long_liq_usd:
                    return 1
                elif candle.long_liq_usd > candle.short_liq_usd:
                    return -1

        # Fallback: use candle body direction
        body = candle.close - candle.open
        return 1 if body > 0 else -1 if body < 0 else 0

    def _is_momentum_bar(self, candle: CandleData, direction: int) -> bool:
        """Check if this candle continues momentum in the given direction."""
        body = candle.close - candle.open
        atr_val = self.atr(self.get_param("atr_period"))
        if atr_val is None or atr_val < 1e-8:
            return False

        min_body = self.get_param("confirm_body_atr") * atr_val
        if abs(body) < min_body:
            return False

        if direction == 1 and body > 0:
            return True
        if direction == -1 and body < 0:
            return True
        return False

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
            self._pending_direction = 0
            self._confirm_count = 0
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            return Signal(direction=None)

        # --- Double confirmation state machine ---
        # State 1: waiting for P99 event
        if self._pending_direction == 0:
            if self._is_p99_event(candle):
                direction = self._get_momentum_direction(candle)
                if direction != 0:
                    self._pending_direction = direction
                    self._confirm_count = 0
            return Signal(direction=None)

        # State 2: waiting for N consecutive momentum bars
        if self._is_momentum_bar(candle, self._pending_direction):
            self._confirm_count += 1
        else:
            # Momentum broken — reset
            self._pending_direction = 0
            self._confirm_count = 0
            return Signal(direction=None)

        if self._confirm_count < self.get_param("confirm_bars"):
            return Signal(direction=None)

        # Confirmed! Now apply reversal protection
        direction = self._pending_direction
        self._pending_direction = 0
        self._confirm_count = 0

        if self.get_param("use_reversal_protection"):
            rsi_val = self.rsi(self.get_param("rsi_period"))
            if rsi_val is not None:
                if direction == 1 and rsi_val > self.get_param("rsi_reversal_long"):
                    return Signal(direction=None)  # momentum exhausting
                if direction == -1 and rsi_val < self.get_param("rsi_reversal_short"):
                    return Signal(direction=None)

        # MOMENTUM entry: ride the liquidation cascade
        self._in_trade = True
        self._trade_direction = direction
        self._entry_price = candle.close
        self._bars_held = 0
        self._peak = candle.high
        self._trough = candle.low

        return Signal(
            direction=direction,
            strength=self.get_param("entry_strength"),
            tag=f"liq_snipe_{'long' if direction == 1 else 'short'}",
            metadata={"liq_usd": candle.liquidation_usd, "confirm_bars": self.get_param("confirm_bars")},
        )

    def _exit(self, reason: str) -> Signal:
        self._in_trade = False
        self._trade_direction = 0
        self._cooldown = self.get_param("cooldown_bars")
        self._pending_direction = 0
        self._confirm_count = 0
        return Signal(direction=0, tag=f"exit_{reason}")

    def on_trade(self, pnl: float, pnl_pct: float):
        self._in_trade = False
        self._trade_direction = 0
        self._pending_direction = 0
        self._confirm_count = 0


PARAM_RANGES = {
    "liq_percentile": [95, 97, 99],
    "confirm_bars": [1, 2, 3],
    "confirm_body_atr": [0.2, 0.3, 0.5],
    "rsi_reversal_long": [60, 65, 70],
    "rsi_reversal_short": [30, 35, 40],
    "stop_loss_pct": [1.0, 1.5, 2.0],
    "take_profit_pct": [2.0, 3.0, 4.0],
    "trailing_stop_pct": [0.8, 1.0, 1.5],
    "max_hold_bars": [8, 12, 18],
}

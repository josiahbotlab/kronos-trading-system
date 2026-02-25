#!/usr/bin/env python3
"""
Hyperliquid Liquidation Grid (Hyperlick) v1.0
================================================
Source: Most Claude Bots are Slop (Hyperlick deep dive)
Confidence: 65%

Adapted from Hyperlick grid bot concept for backtester framework.

Original concept: "Find biggest whale ($100K+) within 10% of current price,
place 3 orders at 0.2% buffer beyond whale + 0.5% spacing, both sides,
10x leverage, 10% SL/TP"

Backtester adaptation:
- Detect large liquidation clusters (whale-level events)
- When large liquidation event fires near current price → reversal entry
- Use the cluster magnitude to scale entry strength
- Grid concept → single entry with wider TP to simulate grid profit capture
- Both long and short signals based on liquidation direction

Key from transcript:
- Minimum whale size $100K (we use percentile-based threshold)
- Buffer zone 0.2% beyond the liquidation level
- Grid spacing 0.5% between positions
- 10% stop loss and take profit on each position
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
import numpy as np


class HyperliquidLiqGrid(BaseStrategy):
    name = "hyperliquid_liq_grid"
    version = "1.0"

    def default_params(self) -> dict:
        return {
            # Liquidation detection
            "liq_percentile": 90,         # lowered from 95 for more signals
            "liq_lookback": 100,          # bars to compute percentile
            "min_liq_usd": 1000,          # lowered from 5000

            # Directional filter: which side got liquidated
            "liq_ratio_threshold": 0.55,  # lowered from 0.65 for more signals

            # Price proximity: liquidation must be "near" current price
            # (in backtest, liquidation events happen at current candle's price range)
            "use_proximity_filter": True,
            "max_wick_atr": 2.0,          # candle wick < 2x ATR (not a flash crash)
            "atr_period": 14,

            # RSI confirmation for reversal
            "use_rsi_confirm": True,
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,

            "min_history": 50,

            # Entry
            "entry_strength": 0.7,

            # Exit (wider stops per original grid concept)
            "stop_loss_pct": 3.0,         # wider to simulate grid resilience
            "take_profit_pct": 5.0,       # capture more of reversal
            "trailing_stop_pct": 2.0,
            "trail_after_bars": 4,
            "max_hold_bars": 30,
            "cooldown_bars": 8,

            "max_history": 200,
        }

    def on_init(self):
        self._in_trade = False
        self._trade_direction = 0
        self._entry_price = 0.0
        self._bars_held = 0
        self._peak = 0.0
        self._trough = float('inf')
        self._cooldown = 0

    def _is_whale_liquidation(self, candle: CandleData) -> bool:
        """Check if current liquidation is whale-level (top percentile)."""
        if candle.liquidation_usd < self.get_param("min_liq_usd"):
            return False

        liq_values = self.liq_usd(self.get_param("liq_lookback"))
        nonzero = liq_values[liq_values > 0]
        if len(nonzero) < 20:
            return False

        threshold = np.percentile(nonzero, self.get_param("liq_percentile"))
        return candle.liquidation_usd >= threshold

    def _get_liq_direction(self, candle: CandleData) -> int:
        """Determine which side is getting liquidated.
        Returns: 1 if mostly shorts liq'd (bullish), -1 if mostly longs liq'd (bearish), 0 if balanced."""
        total = candle.long_liq_usd + candle.short_liq_usd
        if total < 1e-8:
            return 0

        long_ratio = candle.long_liq_usd / total
        threshold = self.get_param("liq_ratio_threshold")

        if long_ratio > threshold:
            return -1  # mostly longs liquidated → bearish pressure
        elif long_ratio < (1 - threshold):
            return 1   # mostly shorts liquidated → bullish pressure
        return 0

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
            return Signal(direction=None)

        if len(self._candle_history) < self.get_param("min_history"):
            return Signal(direction=None)

        # Check for whale-level liquidation
        if not self._is_whale_liquidation(candle):
            return Signal(direction=None)

        # Determine liquidation direction
        liq_dir = self._get_liq_direction(candle)
        if liq_dir == 0:
            return Signal(direction=None)

        # Proximity filter: reject flash crashes (huge wicks)
        if self.get_param("use_proximity_filter"):
            atr_val = self.atr(self.get_param("atr_period"))
            if atr_val is not None:
                wick = candle.high - candle.low
                if wick > self.get_param("max_wick_atr") * atr_val:
                    return Signal(direction=None)

        # REVERSAL entry: fade the liquidation cascade
        # Longs liquidated (-1) → price dropped → go long (reversal)
        # Shorts liquidated (+1) → price pumped → go short (reversal)
        direction = -liq_dir

        # RSI confirmation
        if self.get_param("use_rsi_confirm"):
            rsi_val = self.rsi(self.get_param("rsi_period"))
            if rsi_val is not None:
                if direction == 1 and rsi_val > self.get_param("rsi_overbought"):
                    return Signal(direction=None)
                if direction == -1 and rsi_val < self.get_param("rsi_oversold"):
                    return Signal(direction=None)

        self._in_trade = True
        self._trade_direction = direction
        self._entry_price = candle.close
        self._bars_held = 0
        self._peak = candle.high
        self._trough = candle.low

        return Signal(
            direction=direction,
            strength=self.get_param("entry_strength"),
            tag=f"liq_grid_{'long' if direction == 1 else 'short'}",
            metadata={
                "liq_usd": candle.liquidation_usd,
                "long_liq": candle.long_liq_usd,
                "short_liq": candle.short_liq_usd,
            },
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
    "liq_percentile": [90, 95, 98],
    "liq_ratio_threshold": [0.6, 0.65, 0.7],
    "max_wick_atr": [1.5, 2.0, 3.0],
    "rsi_oversold": [25, 30, 35],
    "rsi_overbought": [65, 70, 75],
    "stop_loss_pct": [2.5, 3.0, 4.0],
    "take_profit_pct": [4.0, 5.0, 7.0],
    "trailing_stop_pct": [1.5, 2.0, 3.0],
    "max_hold_bars": [20, 30, 40],
}

#!/usr/bin/env python3
"""
Kronos Base Strategy
====================
All strategies inherit from this template.
Provides a clean interface for the backtester.

Strategy lifecycle:
1. __init__() - set parameters
2. on_candle() - called for each new candle, return signals
3. Backtester handles position management, P&L tracking

Signals:
    1  = go long
   -1  = go short
    0  = close position / no action
    None = hold current position (no change)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class Signal:
    """Trading signal from a strategy."""
    direction: Optional[int]  # 1=long, -1=short, 0=close, None=hold
    strength: float = 1.0     # 0.0-1.0, for position sizing
    tag: str = ""             # label for this signal (e.g., "cascade_p99")
    metadata: dict = field(default_factory=dict)  # extra data for logging


@dataclass
class CandleData:
    """Single candle + context passed to strategies."""
    timestamp_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    # Optional enrichment data (set by backtester if available)
    liquidation_usd: float = 0.0       # total liq USD in this candle's window
    short_liq_usd: float = 0.0         # short liquidation USD
    long_liq_usd: float = 0.0          # long liquidation USD
    liq_count: int = 0                  # number of liquidation events
    open_interest: Optional[float] = None
    long_short_ratio: Optional[float] = None


class BaseStrategy(ABC):
    """
    Base class for all Kronos strategies.

    Subclass and implement:
        - name: strategy name
        - on_candle(): return Signal for each candle

    Optionally implement:
        - on_init(): called before backtesting starts
        - on_trade(): called when a trade completes
        - params(): return dict of tunable parameters
    """

    name: str = "base"
    version: str = "1.0"

    def __init__(self, **params):
        """Initialize with parameters. Override defaults with kwargs."""
        self._params = self.default_params()
        self._params.update(params)

        # History buffers (strategies can use these)
        self._candle_history: list[CandleData] = []
        self._max_history: int = self._params.get("max_history", 500)

        # Indicator cache
        self._indicators: dict[str, Any] = {}

    @abstractmethod
    def default_params(self) -> dict:
        """Return default parameter values. Override in subclass."""
        return {}

    @abstractmethod
    def on_candle(self, candle: CandleData) -> Signal:
        """
        Process a new candle and return a trading signal.
        Called by the backtester for each candle in sequence.

        Args:
            candle: Current candle data with optional enrichment

        Returns:
            Signal with direction (1=long, -1=short, 0=close, None=hold)
        """
        return Signal(direction=None)

    def on_init(self):
        """Called before backtesting starts. Override for setup."""
        pass

    def on_trade(self, pnl: float, pnl_pct: float):
        """Called when a trade completes. Override for tracking."""
        pass

    def get_param(self, key: str) -> Any:
        """Get a parameter value."""
        return self._params.get(key)

    def set_param(self, key: str, value: Any):
        """Set a parameter value."""
        self._params[key] = value

    @property
    def params(self) -> dict:
        """Current parameter values."""
        return self._params.copy()

    def _update_history(self, candle: CandleData):
        """Maintain rolling candle history."""
        self._candle_history.append(candle)
        if len(self._candle_history) > self._max_history:
            self._candle_history = self._candle_history[-self._max_history:]

    # --- Common indicator helpers ---

    def closes(self, n: Optional[int] = None) -> np.ndarray:
        """Get array of recent close prices."""
        data = self._candle_history[-n:] if n else self._candle_history
        return np.array([c.close for c in data])

    def highs(self, n: Optional[int] = None) -> np.ndarray:
        data = self._candle_history[-n:] if n else self._candle_history
        return np.array([c.high for c in data])

    def lows(self, n: Optional[int] = None) -> np.ndarray:
        data = self._candle_history[-n:] if n else self._candle_history
        return np.array([c.low for c in data])

    def volumes(self, n: Optional[int] = None) -> np.ndarray:
        data = self._candle_history[-n:] if n else self._candle_history
        return np.array([c.volume for c in data])

    def liq_usd(self, n: Optional[int] = None) -> np.ndarray:
        data = self._candle_history[-n:] if n else self._candle_history
        return np.array([c.liquidation_usd for c in data])

    def sma(self, period: int) -> Optional[float]:
        """Simple moving average of close prices."""
        closes = self.closes(period)
        return float(np.mean(closes)) if len(closes) >= period else None

    def ema(self, period: int) -> Optional[float]:
        """Exponential moving average of close prices."""
        closes = self.closes(period * 2)  # need extra history
        if len(closes) < period:
            return None
        alpha = 2 / (period + 1)
        ema_val = closes[0]
        for price in closes[1:]:
            ema_val = alpha * price + (1 - alpha) * ema_val
        return float(ema_val)

    def rsi(self, period: int = 14) -> Optional[float]:
        """Relative Strength Index."""
        closes = self.closes(period + 1)
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100 - (100 / (1 + rs)))

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Optional[tuple[float, float, float]]:
        """Bollinger Bands: (upper, middle, lower)."""
        closes = self.closes(period)
        if len(closes) < period:
            return None
        middle = float(np.mean(closes))
        std = float(np.std(closes, ddof=1))
        return (middle + std_dev * std, middle, middle - std_dev * std)

    def atr(self, period: int = 14) -> Optional[float]:
        """Average True Range."""
        if len(self._candle_history) < period + 1:
            return None
        trs = []
        candles = self._candle_history[-(period + 1):]
        for i in range(1, len(candles)):
            high = candles[i].high
            low = candles[i].low
            prev_close = candles[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        return float(np.mean(trs[-period:]))

    def percentile(self, values: np.ndarray, pct: float) -> float:
        """Calculate percentile of an array."""
        return float(np.percentile(values, pct)) if len(values) > 0 else 0.0

    def __repr__(self):
        return f"{self.name}(params={self._params})"

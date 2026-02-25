"""
RBI Pipeline - Backtest Strategy Templates

Each strategy is a backtesting.py Strategy subclass with tp_pct and sl_pct
as class-level attributes so the RBI agent can optimize them via bt.run(tp_pct=X, sl_pct=Y).

STRATEGY_REGISTRY maps idea names (from ideas.txt) to their Strategy class.
"""

import numpy as np
from backtesting import Strategy
from backtesting.lib import crossover


# ─────────────────────────────────────────────────────────────
# Indicator helpers (operate on numpy arrays for backtesting.py)
# ─────────────────────────────────────────────────────────────

def SMA(array, period):
    """Simple Moving Average."""
    result = np.full_like(array, np.nan, dtype=float)
    for i in range(period - 1, len(array)):
        result[i] = np.mean(array[i - period + 1:i + 1])
    return result


def EMA(array, period):
    """Exponential Moving Average."""
    result = np.full_like(array, np.nan, dtype=float)
    multiplier = 2 / (period + 1)
    result[period - 1] = np.mean(array[:period])
    for i in range(period, len(array)):
        result[i] = array[i] * multiplier + result[i - 1] * (1 - multiplier)
    return result


def RSI(array, period=14):
    """Relative Strength Index."""
    result = np.full_like(array, np.nan, dtype=float)
    deltas = np.diff(array)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    if avg_loss == 0:
        result[period] = 100.0
    else:
        result[period] = 100 - (100 / (1 + avg_gain / avg_loss))

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100 - (100 / (1 + avg_gain / avg_loss))

    return result


def BollingerBands(array, period=20, std_dev=2.0):
    """Returns (middle, upper, lower) Bollinger Bands."""
    middle = SMA(array, period)
    rolling_std = np.full_like(array, np.nan, dtype=float)
    for i in range(period - 1, len(array)):
        rolling_std[i] = np.std(array[i - period + 1:i + 1], ddof=1)
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    return middle, upper, lower


def MACD(array, fast=12, slow=26, signal=9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = EMA(array, fast)
    ema_slow = EMA(array, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def BBWidth(array, period=20, std_dev=2.0):
    """Bollinger Band Width = (upper - lower) / middle."""
    middle, upper, lower = BollingerBands(array, period, std_dev)
    with np.errstate(divide='ignore', invalid='ignore'):
        width = np.where(middle > 0, (upper - lower) / middle, np.nan)
    return width


# ─────────────────────────────────────────────────────────────
# Strategy: Momentum Pullback
# ─────────────────────────────────────────────────────────────

class MomentumPullback(Strategy):
    """
    Trend-following: buy pullbacks to EMA in uptrends.
    Entry: Price > SMA AND RSI in range AND price near EMA.
    """
    sma_period = 50
    ema_period = 9
    rsi_period = 14
    rsi_entry = 40       # Buy when RSI drops below this (pullback)
    rsi_max = 80
    ema_tolerance = 0.03  # 3% distance from EMA
    tp_pct = 6.0
    sl_pct = 3.0

    def init(self):
        self.sma = self.I(SMA, self.data.Close, self.sma_period)
        self.ema = self.I(EMA, self.data.Close, self.ema_period)
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)

    def next(self):
        if self.position:
            return

        price = self.data.Close[-1]
        if np.isnan(self.sma[-1]) or np.isnan(self.ema[-1]) or np.isnan(self.rsi[-1]):
            return

        above_sma = price > self.sma[-1]
        rsi_ok = self.rsi[-1] <= self.rsi_entry or (self.rsi_entry <= self.rsi[-1] <= self.rsi_max)
        ema_dist = abs(price - self.ema[-1]) / price
        near_ema = ema_dist <= self.ema_tolerance

        # Pullback in uptrend: above SMA, RSI not overbought, near EMA
        if above_sma and rsi_ok and near_ema:
            self.buy(
                sl=price * (1 - self.sl_pct / 100),
                tp=price * (1 + self.tp_pct / 100),
            )


# ─────────────────────────────────────────────────────────────
# Strategy: Mean Reversion (Lower Bollinger Band touch)
# ─────────────────────────────────────────────────────────────

class MeanReversionBB(Strategy):
    """
    Mean reversion: buy when price touches or pierces lower Bollinger Band.
    Simple version — no RSI filter (that's bb_bounce).
    """
    bb_period = 20
    bb_std = 2.0
    tp_pct = 4.0
    sl_pct = 3.0

    def init(self):
        close = self.data.Close
        self.bb_mid, self.bb_upper, self.bb_lower = (
            self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[0], close),
            self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[1], close),
            self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[2], close),
        )

    def next(self):
        if self.position:
            return

        price = self.data.Close[-1]
        if np.isnan(self.bb_lower[-1]):
            return

        if price <= self.bb_lower[-1]:
            self.buy(
                sl=price * (1 - self.sl_pct / 100),
                tp=price * (1 + self.tp_pct / 100),
            )


# ─────────────────────────────────────────────────────────────
# Strategy: Breakout (MoonDev style - N-day high breakout)
# ─────────────────────────────────────────────────────────────

class BreakoutDaily(Strategy):
    """
    Daily breakout: buy when price closes above the N-day high.
    MoonDev finding: small TP + wide SL can outperform conventional wisdom.
    """
    lookback = 20
    tp_pct = 3.0
    sl_pct = 18.0

    def init(self):
        pass

    def next(self):
        if self.position:
            return

        if len(self.data.Close) < self.lookback + 1:
            return

        # N-day high (excluding today)
        high_n = max(self.data.High[-self.lookback - 1:-1])
        price = self.data.Close[-1]

        if price > high_n:
            self.buy(
                sl=price * (1 - self.sl_pct / 100),
                tp=price * (1 + self.tp_pct / 100),
            )


# ─────────────────────────────────────────────────────────────
# Strategy: Gap & Go
# ─────────────────────────────────────────────────────────────

class GapAndGoVolume(Strategy):
    """
    Trade opening gaps with volume confirmation.
    Entry: Gap up > gap_pct on volume > vol_mult * 20-day average.
    """
    gap_pct = 2.0
    vol_mult = 1.5
    tp_pct = 4.0
    sl_pct = 2.0

    def init(self):
        self.sma_vol = self.I(SMA, self.data.Volume, 20)

    def next(self):
        if self.position:
            return

        if len(self.data.Close) < 2:
            return

        prev_close = self.data.Close[-2]
        current_open = self.data.Open[-1]
        current_vol = self.data.Volume[-1]
        avg_vol = self.sma_vol[-1]

        if np.isnan(avg_vol) or avg_vol <= 0:
            return

        gap = (current_open - prev_close) / prev_close * 100
        vol_ratio = current_vol / avg_vol

        if gap >= self.gap_pct and vol_ratio >= self.vol_mult:
            entry = self.data.Close[-1]
            self.buy(
                sl=entry * (1 - self.sl_pct / 100),
                tp=entry * (1 + self.tp_pct / 100),
            )


# ─────────────────────────────────────────────────────────────
# Strategy: MACD Bullish Crossover
# ─────────────────────────────────────────────────────────────

class MACDBullish(Strategy):
    """MACD bullish crossover entry."""
    fast = 12
    slow = 26
    signal_period = 9
    tp_pct = 6.0
    sl_pct = 3.0

    def init(self):
        close = self.data.Close
        self.macd_line, self.signal_line, self.histogram = (
            self.I(lambda c: MACD(c, self.fast, self.slow, self.signal_period)[0], close),
            self.I(lambda c: MACD(c, self.fast, self.slow, self.signal_period)[1], close),
            self.I(lambda c: MACD(c, self.fast, self.slow, self.signal_period)[2], close),
        )

    def next(self):
        if self.position:
            return

        if np.isnan(self.macd_line[-1]) or np.isnan(self.signal_line[-1]):
            return

        if crossover(self.macd_line, self.signal_line):
            price = self.data.Close[-1]
            self.buy(
                sl=price * (1 - self.sl_pct / 100),
                tp=price * (1 + self.tp_pct / 100),
            )


# ─────────────────────────────────────────────────────────────
# Strategy: BB Lower Bounce (with RSI confirmation)
# ─────────────────────────────────────────────────────────────

class BBLowerBounce(Strategy):
    """Bollinger Band lower band bounce with RSI < 30 confirmation."""
    bb_period = 20
    bb_std = 2.0
    rsi_period = 14
    rsi_threshold = 30
    tp_pct = 4.0
    sl_pct = 3.0

    def init(self):
        close = self.data.Close
        self.bb_mid, self.bb_upper, self.bb_lower = (
            self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[0], close),
            self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[1], close),
            self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[2], close),
        )
        self.rsi = self.I(RSI, close, self.rsi_period)

    def next(self):
        if self.position:
            return

        price = self.data.Close[-1]
        if np.isnan(self.bb_lower[-1]) or np.isnan(self.rsi[-1]):
            return

        if price <= self.bb_lower[-1] and self.rsi[-1] < self.rsi_threshold:
            self.buy(
                sl=price * (1 - self.sl_pct / 100),
                tp=price * (1 + self.tp_pct / 100),
            )


# ─────────────────────────────────────────────────────────────
# Strategy: Golden Cross (SMA50/SMA200 crossover)
# ─────────────────────────────────────────────────────────────

class GoldenCross(Strategy):
    """
    Classic trend-following: buy when SMA50 crosses above SMA200.
    Exit: SMA50 crosses below SMA200 (death cross) OR TP/SL hit.
    """
    sma_fast_period = 50
    sma_slow_period = 200
    tp_pct = 10.0
    sl_pct = 7.0

    def init(self):
        self.sma50 = self.I(SMA, self.data.Close, self.sma_fast_period)
        self.sma200 = self.I(SMA, self.data.Close, self.sma_slow_period)
        self._entry_price = 0.0

    def next(self):
        price = self.data.Close[-1]

        if self.position:
            # Exit on death cross (SMA50 crosses below SMA200)
            if crossover(self.sma200, self.sma50):
                self.position.close()
            # Manual TP/SL
            elif price >= self._entry_price * (1 + self.tp_pct / 100):
                self.position.close()
            elif price <= self._entry_price * (1 - self.sl_pct / 100):
                self.position.close()
            return

        if np.isnan(self.sma50[-1]) or np.isnan(self.sma200[-1]):
            return

        if crossover(self.sma50, self.sma200):
            self._entry_price = price
            self.buy()


# ─────────────────────────────────────────────────────────────
# Strategy: Trailing Stop Trend (ride trends with trailing stop)
# ─────────────────────────────────────────────────────────────

class TrailingStopTrend(Strategy):
    """
    Trend-riding: enter on strength, trail stop from highest price.
    Entry: Price > SMA50 AND RSI > 50 AND new 20-day high.
    Exit: Price drops sl_pct% from highest since entry, or hits tp_pct%.
    """
    sma_period = 50
    rsi_period = 14
    rsi_min = 50
    lookback = 20
    tp_pct = 20.0
    sl_pct = 10.0  # trailing stop distance from high

    def init(self):
        self.sma = self.I(SMA, self.data.Close, self.sma_period)
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        self._highest = 0.0
        self._entry_price = 0.0

    def next(self):
        price = self.data.Close[-1]

        if self.position:
            # Track highest price since entry
            if price > self._highest:
                self._highest = price
            # Trailing stop
            trail_stop = self._highest * (1 - self.sl_pct / 100)
            # Fixed TP
            tp_target = self._entry_price * (1 + self.tp_pct / 100)
            if price <= trail_stop or price >= tp_target:
                self.position.close()
            return

        if np.isnan(self.sma[-1]) or np.isnan(self.rsi[-1]):
            return
        if len(self.data.Close) < self.lookback + 1:
            return

        # N-day high (excluding today)
        high_n = max(self.data.High[-self.lookback - 1:-1])

        if price > self.sma[-1] and self.rsi[-1] > self.rsi_min and price > high_n:
            self._highest = price
            self._entry_price = price
            self.buy()


# ─────────────────────────────────────────────────────────────
# Strategy: Range Mean Reversion (buy oversold, sell at mean)
# ─────────────────────────────────────────────────────────────

class RangeMeanReversion(Strategy):
    """
    Buy oversold in sideways markets, exit at mean.
    Entry: RSI < 30 AND price within 5% of 20-day low.
    Exit: RSI > 70 OR price reaches SMA(20) OR TP/SL.
    """
    rsi_period = 14
    rsi_entry = 30
    rsi_exit = 70
    sma_period = 20
    low_lookback = 20
    low_pct = 5.0  # within 5% of 20-day low
    tp_pct = 6.0
    sl_pct = 4.0

    def init(self):
        self.rsi = self.I(RSI, self.data.Close, self.rsi_period)
        self.sma = self.I(SMA, self.data.Close, self.sma_period)
        self._entry_price = 0.0

    def next(self):
        price = self.data.Close[-1]

        if self.position:
            # Exit on RSI overbought
            if not np.isnan(self.rsi[-1]) and self.rsi[-1] > self.rsi_exit:
                self.position.close()
                return
            # Exit at mean (SMA)
            if not np.isnan(self.sma[-1]) and price >= self.sma[-1]:
                self.position.close()
                return
            # Manual TP/SL
            if price >= self._entry_price * (1 + self.tp_pct / 100):
                self.position.close()
            elif price <= self._entry_price * (1 - self.sl_pct / 100):
                self.position.close()
            return

        if np.isnan(self.rsi[-1]) or np.isnan(self.sma[-1]):
            return
        if len(self.data.Close) < self.low_lookback + 1:
            return

        # 20-day low (excluding today)
        low_20 = min(self.data.Low[-self.low_lookback - 1:-1])
        dist_from_low = (price - low_20) / low_20 * 100

        if self.rsi[-1] < self.rsi_entry and dist_from_low <= self.low_pct:
            self._entry_price = price
            self.buy()


# ─────────────────────────────────────────────────────────────
# Strategy: Bollinger Squeeze (volatility expansion breakout)
# ─────────────────────────────────────────────────────────────

class BollingerSqueeze(Strategy):
    """
    Trade volatility expansion after low-volatility squeeze.
    Entry: BB width at 20-bar low (squeeze) AND price breaks above upper band.
    Exit: Price touches middle band OR TP/SL.
    """
    bb_period = 20
    bb_std = 2.0
    squeeze_lookback = 20
    tp_pct = 7.0
    sl_pct = 4.0

    def init(self):
        close = self.data.Close
        self.bb_mid = self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[0], close)
        self.bb_upper = self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[1], close)
        self.bb_width = self.I(BBWidth, close, self.bb_period, self.bb_std)
        self._entry_price = 0.0

    def next(self):
        price = self.data.Close[-1]

        if self.position:
            # Exit if price touches middle band
            if not np.isnan(self.bb_mid[-1]) and price <= self.bb_mid[-1]:
                self.position.close()
                return
            # Manual TP/SL
            if price >= self._entry_price * (1 + self.tp_pct / 100):
                self.position.close()
            elif price <= self._entry_price * (1 - self.sl_pct / 100):
                self.position.close()
            return

        if np.isnan(self.bb_upper[-1]) or np.isnan(self.bb_width[-1]):
            return
        if len(self.data.Close) < self.bb_period + self.squeeze_lookback:
            return

        # Check if BB width is at the N-bar minimum (squeeze)
        curr_w = self.bb_width[-1]
        is_squeeze = True
        for i in range(2, self.squeeze_lookback + 1):
            w = self.bb_width[-i]
            if not np.isnan(w) and w < curr_w:
                is_squeeze = False
                break

        # Breakout above upper band during squeeze
        if is_squeeze and price > self.bb_upper[-1]:
            self._entry_price = price
            self.buy()


# ─────────────────────────────────────────────────────────────
# Strategy: Short Overbought (short overextended moves)
# ─────────────────────────────────────────────────────────────

class ShortOverbought(Strategy):
    """
    Short overextended moves (PAPER TRADING ONLY).
    Entry: RSI > 80 AND price > upper BB AND price > SMA50 * 1.1.
    Exit: RSI < 50 OR price touches middle BB OR TP/SL.
    """
    rsi_period = 14
    rsi_entry = 80
    rsi_exit = 50
    bb_period = 20
    bb_std = 2.0
    sma_period = 50
    extend_pct = 10.0  # price must be > SMA50 * 1.1
    tp_pct = 5.0
    sl_pct = 5.0

    def init(self):
        close = self.data.Close
        self.rsi = self.I(RSI, close, self.rsi_period)
        self.bb_mid = self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[0], close)
        self.bb_upper = self.I(lambda c: BollingerBands(c, self.bb_period, self.bb_std)[1], close)
        self.sma = self.I(SMA, close, self.sma_period)
        self._entry_price = 0.0

    def next(self):
        price = self.data.Close[-1]

        if self.position:
            # Exit if RSI drops below threshold
            if not np.isnan(self.rsi[-1]) and self.rsi[-1] < self.rsi_exit:
                self.position.close()
                return
            # Exit if price touches middle BB
            if not np.isnan(self.bb_mid[-1]) and price <= self.bb_mid[-1]:
                self.position.close()
                return
            # Manual TP/SL (inverted for short: profit when price drops)
            if price <= self._entry_price * (1 - self.tp_pct / 100):
                self.position.close()
            elif price >= self._entry_price * (1 + self.sl_pct / 100):
                self.position.close()
            return

        if np.isnan(self.rsi[-1]) or np.isnan(self.bb_upper[-1]) or np.isnan(self.sma[-1]):
            return

        overextended = price > self.sma[-1] * (1 + self.extend_pct / 100)

        if self.rsi[-1] > self.rsi_entry and price > self.bb_upper[-1] and overextended:
            self._entry_price = price
            self.sell()


# ─────────────────────────────────────────────────────────────
# REGISTRY: Maps idea names from ideas.txt → Strategy classes
# ─────────────────────────────────────────────────────────────

STRATEGY_REGISTRY = {
    'momentum_pullback': MomentumPullback,
    'mean_reversion': MeanReversionBB,
    'breakout_moondev': BreakoutDaily,
    'gap_and_go': GapAndGoVolume,
    'macd_bullish': MACDBullish,
    'bb_bounce': BBLowerBounce,
    'golden_cross': GoldenCross,
    'trailing_stop_trend': TrailingStopTrend,
    'range_mean_reversion': RangeMeanReversion,
    'bollinger_squeeze': BollingerSqueeze,
    'short_overbought': ShortOverbought,
}

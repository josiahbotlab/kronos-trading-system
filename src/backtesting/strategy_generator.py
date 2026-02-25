"""
Strategy Generator and Batch Tester

Systematically generates trading strategy combinations by mixing:
- Different technical indicators
- Different entry conditions
- Different exit conditions
- Different timeframes

Then batch tests them on AMD to find promising strategies.

Usage:
    python -m src.backtesting.strategy_generator --generate --test --top 10
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from itertools import product
import json
import os

# Use ta library instead of talib
import ta
from ta.trend import SMAIndicator, EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel

from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

load_dotenv()

# =============================================================================
# INDICATOR DEFINITIONS
# =============================================================================

@dataclass
class Indicator:
    """Definition of a technical indicator."""
    name: str
    func: Callable
    params: Dict
    description: str


class IndicatorLibrary:
    """Library of common technical indicators using ta library."""

    @staticmethod
    def sma(close: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return SMAIndicator(close, window=period).sma_indicator()

    @staticmethod
    def ema(close: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return EMAIndicator(close, window=period).ema_indicator()

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        return RSIIndicator(close, window=period).rsi()

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        macd_obj = MACD(close, window_fast=fast, window_slow=slow, window_sign=signal)
        return macd_obj.macd(), macd_obj.macd_signal(), macd_obj.macd_diff()

    @staticmethod
    def bollinger_bands(close: pd.Series, period: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands (upper, middle, lower)."""
        bb = BollingerBands(close, window=period, window_dev=std)
        return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        return AverageTrueRange(high, low, close, window=period).average_true_range()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average Directional Index."""
        return ADXIndicator(high, low, close, window=period).adx()

    @staticmethod
    def volume_spike(volume: pd.Series, period: int = 20, multiplier: float = 2.0) -> pd.Series:
        """Detect volume spikes above average."""
        avg_volume = volume.rolling(period).mean()
        return volume > (multiplier * avg_volume)

    @staticmethod
    def support_resistance(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Rolling support and resistance levels."""
        resistance = high.rolling(period).max()
        support = low.rolling(period).min()
        return resistance, support

    @staticmethod
    def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 20, atr_mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Keltner Channels."""
        kc = KeltnerChannel(high, low, close, window=period, window_atr=period)
        return kc.keltner_channel_hband(), kc.keltner_channel_mband(), kc.keltner_channel_lband()

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   fastk: int = 14, slowk: int = 3, slowd: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        stoch = StochasticOscillator(high, low, close, window=fastk, smooth_window=slowk)
        return stoch.stoch(), stoch.stoch_signal()


# =============================================================================
# ENTRY CONDITION TEMPLATES
# =============================================================================

@dataclass
class EntryCondition:
    """Definition of an entry condition."""
    name: str
    description: str
    direction: str  # 'long', 'short', 'both'
    check_func: Callable
    required_indicators: List[str]


class EntryConditions:
    """Library of entry condition templates."""

    @staticmethod
    def price_above_sma(df: pd.DataFrame, sma_col: str) -> pd.Series:
        """Price crosses above SMA."""
        above = df['close'] > df[sma_col]
        cross = above & ~above.shift(1).fillna(False)
        return cross

    @staticmethod
    def price_below_sma(df: pd.DataFrame, sma_col: str) -> pd.Series:
        """Price crosses below SMA."""
        below = df['close'] < df[sma_col]
        cross = below & ~below.shift(1).fillna(False)
        return cross

    @staticmethod
    def rsi_oversold(df: pd.DataFrame, threshold: int = 30) -> pd.Series:
        """RSI crosses above oversold threshold."""
        was_oversold = df['rsi'].shift(1) < threshold
        now_above = df['rsi'] >= threshold
        return was_oversold & now_above

    @staticmethod
    def rsi_overbought(df: pd.DataFrame, threshold: int = 70) -> pd.Series:
        """RSI crosses below overbought threshold."""
        was_overbought = df['rsi'].shift(1) > threshold
        now_below = df['rsi'] <= threshold
        return was_overbought & now_below

    @staticmethod
    def macd_bullish_cross(df: pd.DataFrame) -> pd.Series:
        """MACD line crosses above signal line."""
        above = df['macd'] > df['macd_signal']
        cross = above & ~above.shift(1).fillna(False)
        return cross

    @staticmethod
    def macd_bearish_cross(df: pd.DataFrame) -> pd.Series:
        """MACD line crosses below signal line."""
        below = df['macd'] < df['macd_signal']
        cross = below & ~below.shift(1).fillna(False)
        return cross

    @staticmethod
    def bb_lower_touch(df: pd.DataFrame) -> pd.Series:
        """Price touches lower Bollinger Band."""
        return df['close'] <= df['bb_lower']

    @staticmethod
    def bb_upper_touch(df: pd.DataFrame) -> pd.Series:
        """Price touches upper Bollinger Band."""
        return df['close'] >= df['bb_upper']

    @staticmethod
    def bb_breakout_up(df: pd.DataFrame) -> pd.Series:
        """Price breaks above upper Bollinger Band."""
        above = df['close'] > df['bb_upper']
        cross = above & ~above.shift(1).fillna(False)
        return cross

    @staticmethod
    def bb_breakout_down(df: pd.DataFrame) -> pd.Series:
        """Price breaks below lower Bollinger Band."""
        below = df['close'] < df['bb_lower']
        cross = below & ~below.shift(1).fillna(False)
        return cross

    @staticmethod
    def gap_up(df: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
        """Price gaps up by threshold percentage."""
        prev_close = df['close'].shift(1)
        gap_pct = (df['open'] - prev_close) / prev_close
        return gap_pct >= threshold

    @staticmethod
    def gap_down(df: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
        """Price gaps down by threshold percentage."""
        prev_close = df['close'].shift(1)
        gap_pct = (prev_close - df['open']) / prev_close
        return gap_pct >= threshold

    @staticmethod
    def breakout_above_resistance(df: pd.DataFrame) -> pd.Series:
        """Price breaks above resistance level."""
        above = df['close'] > df['resistance']
        cross = above & ~above.shift(1).fillna(False)
        return cross

    @staticmethod
    def breakout_below_support(df: pd.DataFrame) -> pd.Series:
        """Price breaks below support level."""
        below = df['close'] < df['support']
        cross = below & ~below.shift(1).fillna(False)
        return cross

    @staticmethod
    def volume_spike_entry(df: pd.DataFrame) -> pd.Series:
        """Entry on volume spike with price move."""
        return df['volume_spike'] & (df['close'] > df['open'])

    @staticmethod
    def ema_crossover(df: pd.DataFrame, fast_col: str, slow_col: str) -> pd.Series:
        """Fast EMA crosses above slow EMA."""
        above = df[fast_col] > df[slow_col]
        cross = above & ~above.shift(1).fillna(False)
        return cross

    @staticmethod
    def bb_squeeze_release(df: pd.DataFrame) -> pd.Series:
        """Bollinger Band squeeze releases (BB inside KC then outside)."""
        # BB inside Keltner = squeeze
        squeeze = (df['bb_upper'] < df['kc_upper']) & (df['bb_lower'] > df['kc_lower'])
        # Squeeze release = was squeeze, now not
        release = squeeze.shift(1).fillna(False) & ~squeeze
        return release

    @staticmethod
    def adx_trending(df: pd.DataFrame, threshold: int = 25) -> pd.Series:
        """ADX indicates strong trend."""
        return df['adx'] > threshold


# =============================================================================
# EXIT CONDITION TEMPLATES
# =============================================================================

@dataclass
class ExitCondition:
    """Definition of an exit condition."""
    name: str
    description: str
    exit_type: str  # 'target', 'stop', 'trailing', 'time', 'signal'
    params: Dict


class ExitConditions:
    """Library of exit condition templates."""

    TAKE_PROFIT_LEVELS = [0.02, 0.03, 0.05, 0.08, 0.10]  # 2%, 3%, 5%, 8%, 10%
    STOP_LOSS_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.07]     # 1%, 2%, 3%, 5%, 7%
    TRAILING_STOPS = [0.02, 0.03, 0.05]                   # 2%, 3%, 5%
    TIME_EXITS = [5, 10, 20, 50]                          # bars

    @staticmethod
    def fixed_tp_sl(entry_price: float, tp_pct: float, sl_pct: float,
                    is_long: bool = True) -> Tuple[float, float]:
        """Calculate fixed take profit and stop loss levels."""
        if is_long:
            tp = entry_price * (1 + tp_pct)
            sl = entry_price * (1 - sl_pct)
        else:
            tp = entry_price * (1 - tp_pct)
            sl = entry_price * (1 + sl_pct)
        return tp, sl

    @staticmethod
    def atr_based_stops(entry_price: float, atr: float, tp_mult: float = 2.0,
                        sl_mult: float = 1.0, is_long: bool = True) -> Tuple[float, float]:
        """Calculate ATR-based take profit and stop loss."""
        if is_long:
            tp = entry_price + (tp_mult * atr)
            sl = entry_price - (sl_mult * atr)
        else:
            tp = entry_price - (tp_mult * atr)
            sl = entry_price + (sl_mult * atr)
        return tp, sl


# =============================================================================
# STRATEGY DEFINITION
# =============================================================================

@dataclass
class StrategyDefinition:
    """Complete definition of a trading strategy."""
    name: str
    entry_conditions: List[str]
    entry_logic: str  # 'AND' or 'OR'
    direction: str    # 'long', 'short', 'both'
    take_profit: float
    stop_loss: float
    exit_type: str    # 'fixed', 'atr', 'trailing', 'time'
    extra_params: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'entry_conditions': self.entry_conditions,
            'entry_logic': self.entry_logic,
            'direction': self.direction,
            'take_profit': self.take_profit,
            'stop_loss': self.stop_loss,
            'exit_type': self.exit_type,
            'extra_params': self.extra_params
        }


# =============================================================================
# STRATEGY GENERATOR
# =============================================================================

class StrategyGenerator:
    """Generate strategy combinations systematically."""

    # Entry condition registry
    ENTRY_CONDITIONS = {
        'price_above_sma_20': {'func': 'price_above_sma', 'params': {'sma_col': 'sma_20'}, 'direction': 'long'},
        'price_above_sma_50': {'func': 'price_above_sma', 'params': {'sma_col': 'sma_50'}, 'direction': 'long'},
        'price_below_sma_20': {'func': 'price_below_sma', 'params': {'sma_col': 'sma_20'}, 'direction': 'short'},
        'price_below_sma_50': {'func': 'price_below_sma', 'params': {'sma_col': 'sma_50'}, 'direction': 'short'},
        'rsi_oversold_30': {'func': 'rsi_oversold', 'params': {'threshold': 30}, 'direction': 'long'},
        'rsi_oversold_20': {'func': 'rsi_oversold', 'params': {'threshold': 20}, 'direction': 'long'},
        'rsi_overbought_70': {'func': 'rsi_overbought', 'params': {'threshold': 70}, 'direction': 'short'},
        'rsi_overbought_80': {'func': 'rsi_overbought', 'params': {'threshold': 80}, 'direction': 'short'},
        'macd_bullish': {'func': 'macd_bullish_cross', 'params': {}, 'direction': 'long'},
        'macd_bearish': {'func': 'macd_bearish_cross', 'params': {}, 'direction': 'short'},
        'bb_lower_bounce': {'func': 'bb_lower_touch', 'params': {}, 'direction': 'long'},
        'bb_upper_bounce': {'func': 'bb_upper_touch', 'params': {}, 'direction': 'short'},
        'bb_breakout_up': {'func': 'bb_breakout_up', 'params': {}, 'direction': 'long'},
        'bb_breakout_down': {'func': 'bb_breakout_down', 'params': {}, 'direction': 'short'},
        'gap_up_2pct': {'func': 'gap_up', 'params': {'threshold': 0.02}, 'direction': 'long'},
        'gap_up_3pct': {'func': 'gap_up', 'params': {'threshold': 0.03}, 'direction': 'long'},
        'gap_down_2pct': {'func': 'gap_down', 'params': {'threshold': 0.02}, 'direction': 'short'},
        'resistance_breakout': {'func': 'breakout_above_resistance', 'params': {}, 'direction': 'long'},
        'support_breakdown': {'func': 'breakout_below_support', 'params': {}, 'direction': 'short'},
        'volume_spike': {'func': 'volume_spike_entry', 'params': {}, 'direction': 'long'},
        'ema_cross_9_21': {'func': 'ema_crossover', 'params': {'fast_col': 'ema_9', 'slow_col': 'ema_21'}, 'direction': 'long'},
        'ema_cross_10_20': {'func': 'ema_crossover', 'params': {'fast_col': 'ema_10', 'slow_col': 'ema_20'}, 'direction': 'long'},
        'bb_squeeze': {'func': 'bb_squeeze_release', 'params': {}, 'direction': 'both'},
        'adx_trending': {'func': 'adx_trending', 'params': {'threshold': 25}, 'direction': 'both'},
    }

    # Take profit and stop loss combinations
    TP_SL_COMBOS = [
        (0.03, 0.02),  # 3% TP, 2% SL - Conservative
        (0.05, 0.03),  # 5% TP, 3% SL - Balanced
        (0.08, 0.05),  # 8% TP, 5% SL - Aggressive
        (0.10, 0.07),  # 10% TP, 7% SL - Very Aggressive
        (0.05, 0.02),  # 5% TP, 2% SL - High RR
        (0.08, 0.03),  # 8% TP, 3% SL - High RR
    ]

    def __init__(self):
        self.generated_strategies: List[StrategyDefinition] = []

    def generate_single_condition_strategies(self) -> List[StrategyDefinition]:
        """Generate strategies with single entry conditions."""
        strategies = []

        for cond_name, cond_info in self.ENTRY_CONDITIONS.items():
            for tp, sl in self.TP_SL_COMBOS:
                # Skip if TP/SL doesn't make sense for the condition
                if cond_info['direction'] == 'short':
                    continue  # Focus on long strategies for AMD

                strat = StrategyDefinition(
                    name=f"{cond_name}_tp{int(tp*100)}_sl{int(sl*100)}",
                    entry_conditions=[cond_name],
                    entry_logic='AND',
                    direction=cond_info['direction'],
                    take_profit=tp,
                    stop_loss=sl,
                    exit_type='fixed'
                )
                strategies.append(strat)

        return strategies

    def generate_combo_strategies(self) -> List[StrategyDefinition]:
        """Generate strategies with combined entry conditions (AND logic)."""
        strategies = []

        # Meaningful combinations
        combos = [
            # Trend + Momentum
            (['price_above_sma_20', 'rsi_oversold_30'], 'SMA20_RSI30'),
            (['price_above_sma_50', 'macd_bullish'], 'SMA50_MACD'),
            (['ema_cross_9_21', 'rsi_oversold_30'], 'EMA_RSI'),
            (['ema_cross_9_21', 'volume_spike'], 'EMA_Volume'),

            # Breakout + Confirmation
            (['bb_breakout_up', 'volume_spike'], 'BB_Breakout_Vol'),
            (['resistance_breakout', 'volume_spike'], 'Resistance_Vol'),
            (['gap_up_2pct', 'volume_spike'], 'Gap_Vol'),
            (['bb_squeeze', 'adx_trending'], 'Squeeze_ADX'),

            # Mean Reversion + Confirmation
            (['bb_lower_bounce', 'rsi_oversold_30'], 'BB_Lower_RSI'),
            (['rsi_oversold_20', 'volume_spike'], 'RSI_Extreme_Vol'),

            # Triple Confirmation
            (['price_above_sma_20', 'macd_bullish', 'volume_spike'], 'Triple_Trend'),
            (['bb_squeeze', 'adx_trending', 'volume_spike'], 'Triple_Breakout'),
        ]

        for conditions, combo_name in combos:
            for tp, sl in self.TP_SL_COMBOS:
                strat = StrategyDefinition(
                    name=f"{combo_name}_tp{int(tp*100)}_sl{int(sl*100)}",
                    entry_conditions=conditions,
                    entry_logic='AND',
                    direction='long',
                    take_profit=tp,
                    stop_loss=sl,
                    exit_type='fixed'
                )
                strategies.append(strat)

        return strategies

    def generate_all(self, max_strategies: int = 100) -> List[StrategyDefinition]:
        """Generate all strategy combinations up to max limit."""
        strategies = []

        # Single condition strategies
        single = self.generate_single_condition_strategies()
        strategies.extend(single)

        # Combo strategies
        combos = self.generate_combo_strategies()
        strategies.extend(combos)

        # Limit to max
        self.generated_strategies = strategies[:max_strategies]

        print(f"Generated {len(self.generated_strategies)} strategies")
        return self.generated_strategies

    def to_dataframe(self) -> pd.DataFrame:
        """Convert generated strategies to DataFrame."""
        return pd.DataFrame([s.to_dict() for s in self.generated_strategies])

    def save_to_csv(self, filepath: str):
        """Save generated strategies to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"Saved {len(df)} strategies to {filepath}")


# =============================================================================
# STRATEGY BACKTESTER
# =============================================================================

class StrategyBacktester:
    """Backtest generated strategies on historical data."""

    def __init__(self, symbol: str = 'AMD', start_date: str = '2024-01-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.api_key = os.getenv('ALPACA_API_KEY')
        self.api_secret = os.getenv('ALPACA_SECRET_KEY')
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        self.api = tradeapi.REST(self.api_key, self.api_secret, self.base_url)
        self.data: Optional[pd.DataFrame] = None

    def fetch_data(self) -> pd.DataFrame:
        """Fetch historical data from Alpaca."""
        print(f"Fetching data for {self.symbol} from {self.start_date}...")

        # Use IEX feed for paper trading
        bars = self.api.get_bars(
            self.symbol,
            '1Day',
            start=self.start_date,
            end=datetime.now().strftime('%Y-%m-%d'),
            feed='iex'
        ).df

        df = bars.reset_index()

        # Normalize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if 'open' in lower:
                col_map[col] = 'open'
            elif 'high' in lower:
                col_map[col] = 'high'
            elif 'low' in lower:
                col_map[col] = 'low'
            elif 'close' in lower:
                col_map[col] = 'close'
            elif 'volume' in lower:
                col_map[col] = 'volume'
            elif 'timestamp' in lower or 'time' in lower:
                col_map[col] = 'timestamp'

        df = df.rename(columns=col_map)

        # Select only required columns
        available_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in available_cols if c in df.columns]]
        df = df.sort_values('timestamp').reset_index(drop=True)

        self.data = df
        print(f"Fetched {len(df)} bars")
        return df

    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate all required indicators."""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")

        df = self.data.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # Moving Averages
        df['sma_5'] = IndicatorLibrary.sma(close, 5)
        df['sma_10'] = IndicatorLibrary.sma(close, 10)
        df['sma_20'] = IndicatorLibrary.sma(close, 20)
        df['sma_50'] = IndicatorLibrary.sma(close, 50)
        df['sma_200'] = IndicatorLibrary.sma(close, 200)

        df['ema_5'] = IndicatorLibrary.ema(close, 5)
        df['ema_9'] = IndicatorLibrary.ema(close, 9)
        df['ema_10'] = IndicatorLibrary.ema(close, 10)
        df['ema_20'] = IndicatorLibrary.ema(close, 20)
        df['ema_21'] = IndicatorLibrary.ema(close, 21)

        # RSI
        df['rsi'] = IndicatorLibrary.rsi(close, 14)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = IndicatorLibrary.macd(close)

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = IndicatorLibrary.bollinger_bands(close, 20, 2.0)

        # Keltner Channels (for squeeze detection)
        df['kc_upper'], df['kc_middle'], df['kc_lower'] = IndicatorLibrary.keltner_channels(high, low, close, 20, 1.5)

        # ATR and ADX
        df['atr'] = IndicatorLibrary.atr(high, low, close, 14)
        df['adx'] = IndicatorLibrary.adx(high, low, close, 14)

        # Support/Resistance
        df['resistance'], df['support'] = IndicatorLibrary.support_resistance(high, low, 20)

        # Volume Spike
        df['volume_spike'] = IndicatorLibrary.volume_spike(volume, 20, 2.0)

        self.data = df
        return df

    def check_entry_condition(self, cond_name: str, df: pd.DataFrame, idx: int) -> bool:
        """Check if an entry condition is met at given index."""
        if idx < 50:  # Skip early bars (not enough data for indicators)
            return False

        cond_info = StrategyGenerator.ENTRY_CONDITIONS.get(cond_name)
        if not cond_info:
            return False

        func_name = cond_info['func']
        params = cond_info['params']

        # Get subset of data up to current index
        df_subset = df.iloc[:idx+1].copy()

        try:
            # Call the appropriate function
            if func_name == 'price_above_sma':
                signals = EntryConditions.price_above_sma(df_subset, params['sma_col'])
            elif func_name == 'price_below_sma':
                signals = EntryConditions.price_below_sma(df_subset, params['sma_col'])
            elif func_name == 'rsi_oversold':
                signals = EntryConditions.rsi_oversold(df_subset, params['threshold'])
            elif func_name == 'rsi_overbought':
                signals = EntryConditions.rsi_overbought(df_subset, params['threshold'])
            elif func_name == 'macd_bullish_cross':
                signals = EntryConditions.macd_bullish_cross(df_subset)
            elif func_name == 'macd_bearish_cross':
                signals = EntryConditions.macd_bearish_cross(df_subset)
            elif func_name == 'bb_lower_touch':
                signals = EntryConditions.bb_lower_touch(df_subset)
            elif func_name == 'bb_upper_touch':
                signals = EntryConditions.bb_upper_touch(df_subset)
            elif func_name == 'bb_breakout_up':
                signals = EntryConditions.bb_breakout_up(df_subset)
            elif func_name == 'bb_breakout_down':
                signals = EntryConditions.bb_breakout_down(df_subset)
            elif func_name == 'gap_up':
                signals = EntryConditions.gap_up(df_subset, params['threshold'])
            elif func_name == 'gap_down':
                signals = EntryConditions.gap_down(df_subset, params['threshold'])
            elif func_name == 'breakout_above_resistance':
                signals = EntryConditions.breakout_above_resistance(df_subset)
            elif func_name == 'breakout_below_support':
                signals = EntryConditions.breakout_below_support(df_subset)
            elif func_name == 'volume_spike_entry':
                signals = EntryConditions.volume_spike_entry(df_subset)
            elif func_name == 'ema_crossover':
                signals = EntryConditions.ema_crossover(df_subset, params['fast_col'], params['slow_col'])
            elif func_name == 'bb_squeeze_release':
                signals = EntryConditions.bb_squeeze_release(df_subset)
            elif func_name == 'adx_trending':
                signals = EntryConditions.adx_trending(df_subset, params['threshold'])
            else:
                return False

            return bool(signals.iloc[-1]) if len(signals) > 0 else False

        except Exception:
            return False

    def backtest_strategy(self, strategy: StrategyDefinition) -> Dict:
        """Backtest a single strategy."""
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")

        df = self.data.copy()
        trades = []
        in_position = False
        entry_price = 0.0
        entry_idx = 0

        for idx in range(50, len(df)):
            current_close = df.iloc[idx]['close']

            if in_position:
                # Check exit conditions
                tp_price = entry_price * (1 + strategy.take_profit)
                sl_price = entry_price * (1 - strategy.stop_loss)

                # Check for TP hit (using high of bar)
                if df.iloc[idx]['high'] >= tp_price:
                    pnl_pct = strategy.take_profit
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_price': entry_price,
                        'exit_price': tp_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'TP'
                    })
                    in_position = False

                # Check for SL hit (using low of bar)
                elif df.iloc[idx]['low'] <= sl_price:
                    pnl_pct = -strategy.stop_loss
                    trades.append({
                        'entry_idx': entry_idx,
                        'exit_idx': idx,
                        'entry_price': entry_price,
                        'exit_price': sl_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'SL'
                    })
                    in_position = False

            else:
                # Check entry conditions
                if strategy.entry_logic == 'AND':
                    all_met = all(
                        self.check_entry_condition(cond, df, idx)
                        for cond in strategy.entry_conditions
                    )
                    if all_met:
                        in_position = True
                        entry_price = current_close
                        entry_idx = idx

                elif strategy.entry_logic == 'OR':
                    any_met = any(
                        self.check_entry_condition(cond, df, idx)
                        for cond in strategy.entry_conditions
                    )
                    if any_met:
                        in_position = True
                        entry_price = current_close
                        entry_idx = idx

        # Calculate metrics
        if len(trades) == 0:
            return {
                'strategy': strategy.name,
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'trades': []
            }

        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]

        total_return = sum(t['pnl_pct'] for t in trades)
        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0.0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0.0

        gross_profit = sum(t['pnl_pct'] for t in wins)
        gross_loss = abs(sum(t['pnl_pct'] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Sharpe ratio approximation
        returns = [t['pnl_pct'] for t in trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0

        # Max drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_dd = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        return {
            'strategy': strategy.name,
            'total_trades': len(trades),
            'win_rate': round(win_rate, 2),
            'total_return': round(total_return * 100, 2),  # Convert to percentage
            'avg_win': round(avg_win * 100, 2),
            'avg_loss': round(avg_loss * 100, 2),
            'profit_factor': round(profit_factor, 2),
            'sharpe_ratio': round(sharpe, 2),
            'max_drawdown': round(max_dd * 100, 2),
            'trades': trades
        }

    def batch_test(self, strategies: List[StrategyDefinition],
                   min_trades: int = 5, must_beat_buy_hold: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Test all strategies and filter results."""
        print(f"\nBatch testing {len(strategies)} strategies on {self.symbol}...")

        # Calculate buy and hold return
        if self.data is not None:
            buy_hold_return = (self.data['close'].iloc[-1] - self.data['close'].iloc[50]) / self.data['close'].iloc[50] * 100
            print(f"Buy & Hold return: {buy_hold_return:.2f}%")
        else:
            buy_hold_return = 0.0

        results = []
        for i, strategy in enumerate(strategies):
            if i % 10 == 0:
                print(f"  Testing {i+1}/{len(strategies)}...")

            result = self.backtest_strategy(strategy)
            results.append(result)

        # Create full results DataFrame
        all_results = pd.DataFrame([{k: v for k, v in r.items() if k != 'trades'} for r in results])

        # Filter promising strategies
        promising = all_results[
            (all_results['total_trades'] >= min_trades) &
            (all_results['total_return'] > 0)
        ]

        if must_beat_buy_hold:
            promising = promising[promising['total_return'] > buy_hold_return]

        # Sort by Sharpe ratio
        promising = promising.sort_values('sharpe_ratio', ascending=False)

        print(f"\nResults: {len(all_results)} tested, {len(promising)} promising")

        return all_results, promising


# =============================================================================
# MAIN CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Strategy Generator and Backtester')
    parser.add_argument('--generate', action='store_true', help='Generate strategy combinations')
    parser.add_argument('--test', action='store_true', help='Test strategies on AMD')
    parser.add_argument('--symbol', type=str, default='AMD', help='Symbol to test on')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date for backtest')
    parser.add_argument('--max', type=int, default=60, help='Max strategies to generate')
    parser.add_argument('--top', type=int, default=10, help='Show top N strategies')
    args = parser.parse_args()

    # Output directory
    output_dir = 'csvs'
    os.makedirs(output_dir, exist_ok=True)

    if args.generate or (not args.generate and not args.test):
        print("=" * 60)
        print("STRATEGY GENERATOR")
        print("=" * 60)

        generator = StrategyGenerator()
        strategies = generator.generate_all(max_strategies=args.max)

        # Save generated strategies
        gen_file = os.path.join(output_dir, 'generated_strategies.csv')
        generator.save_to_csv(gen_file)

    if args.test:
        print("\n" + "=" * 60)
        print("STRATEGY BACKTESTER")
        print("=" * 60)

        # Load or generate strategies
        gen_file = os.path.join(output_dir, 'generated_strategies.csv')
        if os.path.exists(gen_file):
            gen_df = pd.read_csv(gen_file)
            strategies = []
            for _, row in gen_df.iterrows():
                strat = StrategyDefinition(
                    name=row['name'],
                    entry_conditions=eval(row['entry_conditions']) if isinstance(row['entry_conditions'], str) else row['entry_conditions'],
                    entry_logic=row['entry_logic'],
                    direction=row['direction'],
                    take_profit=row['take_profit'],
                    stop_loss=row['stop_loss'],
                    exit_type=row['exit_type']
                )
                strategies.append(strat)
        else:
            generator = StrategyGenerator()
            strategies = generator.generate_all(max_strategies=args.max)

        # Run backtests
        backtester = StrategyBacktester(symbol=args.symbol, start_date=args.start)
        backtester.fetch_data()
        backtester.calculate_indicators()

        all_results, promising = backtester.batch_test(strategies)

        # Save results
        all_file = os.path.join(output_dir, 'all_strategy_results.csv')
        promising_file = os.path.join(output_dir, 'promising_strategies.csv')

        all_results.to_csv(all_file, index=False)
        promising.to_csv(promising_file, index=False)

        print(f"\nSaved all results to: {all_file}")
        print(f"Saved promising strategies to: {promising_file}")

        # Show top strategies
        print(f"\n{'='*60}")
        print(f"TOP {args.top} STRATEGIES")
        print("=" * 60)

        top_n = promising.head(args.top)
        for i, (_, row) in enumerate(top_n.iterrows(), 1):
            print(f"\n{i}. {row['strategy']}")
            print(f"   Trades: {row['total_trades']}")
            print(f"   Win Rate: {row['win_rate']:.1f}%")
            print(f"   Total Return: {row['total_return']:.1f}%")
            print(f"   Sharpe Ratio: {row['sharpe_ratio']:.2f}")
            print(f"   Profit Factor: {row['profit_factor']:.2f}")
            print(f"   Max Drawdown: {row['max_drawdown']:.1f}%")


if __name__ == '__main__':
    main()

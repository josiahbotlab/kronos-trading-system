"""
BB Squeeze Strategy Bot

Based on Moon Dev's pattern:
- Bollinger Bands squeeze inside Keltner Channels = volatility compression
- When BB expands back outside KC + ADX > 20 = breakout confirmed
- Enter in direction of the breakout
- Stop loss: 1.5x ATR
- Take profit: 2x ATR

The "squeeze" occurs when Bollinger Bands contract inside Keltner Channels,
indicating low volatility. When BB expands back outside KC, it signals a
potential breakout move.

Usage:
    python src/agents/bb_squeeze_bot.py --backtest         # Backtest mode
    python src/agents/bb_squeeze_bot.py --live             # Live trading
    python src/agents/bb_squeeze_bot.py --mock             # Mock mode
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from termcolor import cprint
from dotenv import load_dotenv

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    cprint("Warning: alpaca-trade-api not installed", "yellow")

# ta library for indicators
try:
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
    from ta.trend import ADXIndicator, EMAIndicator
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    cprint("Warning: ta library not available", "yellow")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# BB Squeeze Parameters
BB_PERIOD = 20          # Bollinger Bands period
BB_STD = 2.0            # Bollinger Bands standard deviation
KC_PERIOD = 20          # Keltner Channel period
KC_ATR_MULT = 1.5       # Keltner Channel ATR multiplier
ADX_PERIOD = 14         # ADX period
ADX_THRESHOLD = 10      # ADX threshold for confirmed breakout (lowered from 20 for more signals)
ATR_PERIOD = 14         # ATR period for SL/TP calculation

# Risk Management
SL_ATR_MULT = 1.5       # Stop loss = 1.5x ATR
TP_ATR_MULT = 2.0       # Take profit = 2x ATR
ORDER_USD_SIZE = 500    # Position size in USD


class BBSqueezeIndicators:
    """Calculate all indicators needed for BB Squeeze strategy."""

    @staticmethod
    def calculate_bollinger_bands(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        if TA_AVAILABLE:
            bb = BollingerBands(close, window=BB_PERIOD, window_dev=BB_STD)
            return bb.bollinger_hband(), bb.bollinger_mavg(), bb.bollinger_lband()
        else:
            # Manual calculation
            middle = close.rolling(window=BB_PERIOD).mean()
            std = close.rolling(window=BB_PERIOD).std()
            upper = middle + (BB_STD * std)
            lower = middle - (BB_STD * std)
            return upper, middle, lower

    @staticmethod
    def calculate_keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Keltner Channels with custom ATR multiplier.

        Always use manual calculation to ensure we use our 1.5x ATR multiplier
        (ta library uses 2x by default which is too wide for squeeze detection).
        """
        # EMA for middle band
        middle = close.ewm(span=KC_PERIOD, adjust=False).mean()

        # ATR calculation using ta library if available
        if TA_AVAILABLE:
            atr_indicator = AverageTrueRange(high, low, close, window=KC_PERIOD)
            atr = atr_indicator.average_true_range()
        else:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=KC_PERIOD).mean()

        upper = middle + (KC_ATR_MULT * atr)
        lower = middle - (KC_ATR_MULT * atr)
        return upper, middle, lower

    @staticmethod
    def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ADX (Average Directional Index)."""
        if TA_AVAILABLE:
            adx = ADXIndicator(high, low, close, window=ADX_PERIOD)
            return adx.adx()
        else:
            # Simplified manual ADX
            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm < 0] = 0

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            atr = tr.rolling(window=ADX_PERIOD).mean()
            plus_di = 100 * (plus_dm.rolling(window=ADX_PERIOD).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=ADX_PERIOD).mean() / atr)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=ADX_PERIOD).mean()
            return adx

    @staticmethod
    def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Calculate ATR (Average True Range)."""
        if TA_AVAILABLE:
            atr = AverageTrueRange(high, low, close, window=ATR_PERIOD)
            return atr.average_true_range()
        else:
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(window=ATR_PERIOD).mean()

    @staticmethod
    def detect_squeeze(bb_upper: pd.Series, bb_lower: pd.Series,
                      kc_upper: pd.Series, kc_lower: pd.Series) -> pd.Series:
        """
        Detect BB Squeeze condition.

        Squeeze ON: BB is inside KC (both upper and lower)
        Squeeze OFF: BB is outside KC
        """
        squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        return squeeze_on

    @staticmethod
    def detect_squeeze_release(squeeze: pd.Series) -> pd.Series:
        """
        Detect when squeeze releases (goes from ON to OFF).

        This is the key signal - volatility expanding after compression.
        """
        squeeze_release = squeeze.shift(1) & ~squeeze
        return squeeze_release

    @staticmethod
    def get_momentum_direction(close: pd.Series, period: int = 12) -> pd.Series:
        """
        Get momentum direction using linear regression slope.

        Positive = bullish momentum
        Negative = bearish momentum
        """
        momentum = close.diff(period)
        return momentum


class BBSqueezeBacktester:
    """Backtester for BB Squeeze strategy."""

    def __init__(self):
        self.api = None
        if ALPACA_AVAILABLE and ALPACA_API_KEY:
            base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
            self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    def fetch_historical_data(self, symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        """Fetch historical daily data for backtesting."""
        if not self.api:
            cprint(f"No API connection for {symbol}", "red")
            return None

        try:
            end = datetime.now()
            start = end - timedelta(days=days + 50)  # Extra days for indicator warmup

            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Day,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                feed='iex'
            ).df

            if bars.empty:
                cprint(f"No data returned for {symbol}", "red")
                return None

            # Reset index to get timestamp as column
            bars = bars.reset_index()
            # Don't reassign columns - they're already correctly named by Alpaca
            # Columns: timestamp (index), close, high, low, trade_count, open, volume, vwap

            return bars

        except Exception as e:
            cprint(f"Error fetching data for {symbol}: {e}", "red")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators needed for the strategy."""
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = BBSqueezeIndicators.calculate_bollinger_bands(df['close'])

        # Keltner Channels
        df['kc_upper'], df['kc_middle'], df['kc_lower'] = BBSqueezeIndicators.calculate_keltner_channels(
            df['high'], df['low'], df['close']
        )

        # ADX
        df['adx'] = BBSqueezeIndicators.calculate_adx(df['high'], df['low'], df['close'])

        # ATR
        df['atr'] = BBSqueezeIndicators.calculate_atr(df['high'], df['low'], df['close'])

        # Squeeze detection
        df['squeeze_on'] = BBSqueezeIndicators.detect_squeeze(
            df['bb_upper'], df['bb_lower'], df['kc_upper'], df['kc_lower']
        )
        df['squeeze_release'] = BBSqueezeIndicators.detect_squeeze_release(df['squeeze_on'])

        # Momentum direction
        df['momentum'] = BBSqueezeIndicators.get_momentum_direction(df['close'])

        return df

    def run_backtest(self, symbol: str, days: int = 365) -> Dict:
        """Run backtest on a symbol."""
        cprint(f"\n{'='*60}", "cyan")
        cprint(f"  BB SQUEEZE BACKTEST: {symbol}", "cyan")
        cprint(f"{'='*60}", "cyan")

        # Fetch data
        df = self.fetch_historical_data(symbol, days)
        if df is None or len(df) < 50:
            return {'error': 'Insufficient data'}

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Skip warmup period
        df = df.iloc[50:].reset_index(drop=True)

        # Track trades
        trades = []
        position = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]

            # Skip if indicators not valid
            if pd.isna(row['adx']) or pd.isna(row['atr']):
                continue

            # Check for exit if in position
            if position is not None:
                current_price = row['close']

                # Check stop loss
                if position['direction'] == 'LONG':
                    if current_price <= position['stop_loss']:
                        pnl = (current_price - position['entry']) / position['entry'] * 100
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': row['timestamp'],
                            'direction': 'LONG',
                            'entry': position['entry'],
                            'exit': current_price,
                            'pnl_pct': pnl,
                            'exit_reason': 'STOP_LOSS'
                        })
                        position = None
                        continue

                    # Check take profit
                    if current_price >= position['take_profit']:
                        pnl = (current_price - position['entry']) / position['entry'] * 100
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': row['timestamp'],
                            'direction': 'LONG',
                            'entry': position['entry'],
                            'exit': current_price,
                            'pnl_pct': pnl,
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        position = None
                        continue

                else:  # SHORT
                    if current_price >= position['stop_loss']:
                        pnl = (position['entry'] - current_price) / position['entry'] * 100
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': row['timestamp'],
                            'direction': 'SHORT',
                            'entry': position['entry'],
                            'exit': current_price,
                            'pnl_pct': pnl,
                            'exit_reason': 'STOP_LOSS'
                        })
                        position = None
                        continue

                    if current_price <= position['take_profit']:
                        pnl = (position['entry'] - current_price) / position['entry'] * 100
                        trades.append({
                            'entry_date': position['entry_date'],
                            'exit_date': row['timestamp'],
                            'direction': 'SHORT',
                            'entry': position['entry'],
                            'exit': current_price,
                            'pnl_pct': pnl,
                            'exit_reason': 'TAKE_PROFIT'
                        })
                        position = None
                        continue

            # Check for entry signal (only if not in position)
            if position is None:
                # Squeeze release + ADX confirmation
                if row['squeeze_release'] and row['adx'] >= ADX_THRESHOLD:
                    entry_price = row['close']
                    atr = row['atr']

                    # Determine direction based on momentum
                    if row['momentum'] > 0:
                        # LONG entry
                        stop_loss = entry_price - (SL_ATR_MULT * atr)
                        take_profit = entry_price + (TP_ATR_MULT * atr)
                        position = {
                            'direction': 'LONG',
                            'entry': entry_price,
                            'entry_date': row['timestamp'],
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'atr': atr
                        }
                    else:
                        # SHORT entry
                        stop_loss = entry_price + (SL_ATR_MULT * atr)
                        take_profit = entry_price - (TP_ATR_MULT * atr)
                        position = {
                            'direction': 'SHORT',
                            'entry': entry_price,
                            'entry_date': row['timestamp'],
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'atr': atr
                        }

        # Close any open position at end
        if position is not None:
            last_price = df.iloc[-1]['close']
            if position['direction'] == 'LONG':
                pnl = (last_price - position['entry']) / position['entry'] * 100
            else:
                pnl = (position['entry'] - last_price) / position['entry'] * 100

            trades.append({
                'entry_date': position['entry_date'],
                'exit_date': df.iloc[-1]['timestamp'],
                'direction': position['direction'],
                'entry': position['entry'],
                'exit': last_price,
                'pnl_pct': pnl,
                'exit_reason': 'END_OF_TEST'
            })

        # Calculate statistics
        results = self._calculate_stats(symbol, trades, df)

        # Print results
        self._print_results(results)

        return results

    def _calculate_stats(self, symbol: str, trades: List[Dict], df: pd.DataFrame) -> Dict:
        """Calculate backtest statistics."""
        if not trades:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'win_rate': 0,
                'total_return': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'long_trades': 0,
                'short_trades': 0,
                'buy_hold_return': 0,
                'trades': []
            }

        # Basic stats
        total_trades = len(trades)
        wins = [t for t in trades if t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] <= 0]

        win_rate = len(wins) / total_trades * 100 if total_trades > 0 else 0
        total_return = sum(t['pnl_pct'] for t in trades)

        avg_win = np.mean([t['pnl_pct'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_pct'] for t in losses]) if losses else 0

        # Profit factor
        gross_profit = sum(t['pnl_pct'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl_pct'] for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Sharpe ratio (simplified)
        returns = [t['pnl_pct'] for t in trades]
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # Max drawdown
        cumulative = np.cumsum([t['pnl_pct'] for t in trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

        # Direction breakdown
        long_trades = len([t for t in trades if t['direction'] == 'LONG'])
        short_trades = len([t for t in trades if t['direction'] == 'SHORT'])

        # Buy and hold comparison
        start_price = df.iloc[0]['close']
        end_price = df.iloc[-1]['close']
        buy_hold_return = (end_price - start_price) / start_price * 100

        # Squeeze statistics
        squeeze_days = df['squeeze_on'].sum()
        total_days = len(df)
        squeeze_pct = squeeze_days / total_days * 100 if total_days > 0 else 0

        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'long_trades': long_trades,
            'short_trades': short_trades,
            'buy_hold_return': buy_hold_return,
            'squeeze_pct': squeeze_pct,
            'trades': trades
        }

    def _print_results(self, results: Dict):
        """Print backtest results."""
        symbol = results['symbol']

        cprint(f"\n  Results for {symbol}:", "yellow")
        cprint(f"  {'─'*50}", "white")

        cprint(f"  Total Trades:     {results['total_trades']}", "white")
        cprint(f"  Win Rate:         {results['win_rate']:.1f}%", "green" if results['win_rate'] > 50 else "red")
        cprint(f"  Total Return:     {results['total_return']:+.2f}%", "green" if results['total_return'] > 0 else "red")
        cprint(f"  Avg Win:          {results['avg_win']:+.2f}%", "green")
        cprint(f"  Avg Loss:         {results['avg_loss']:+.2f}%", "red")
        cprint(f"  Profit Factor:    {results['profit_factor']:.2f}", "green" if results['profit_factor'] > 1 else "red")
        cprint(f"  Sharpe Ratio:     {results['sharpe_ratio']:.2f}", "cyan")
        cprint(f"  Max Drawdown:     {results['max_drawdown']:.2f}%", "red")
        cprint(f"  Long/Short:       {results['long_trades']}/{results['short_trades']}", "white")
        cprint(f"  Squeeze Time:     {results.get('squeeze_pct', 0):.1f}% of days", "white")

        cprint(f"\n  Buy & Hold:       {results['buy_hold_return']:+.2f}%", "cyan")

        # Compare to buy & hold
        outperform = results['total_return'] - results['buy_hold_return']
        color = "green" if outperform > 0 else "red"
        cprint(f"  vs Buy & Hold:    {outperform:+.2f}%", color)

        # Show recent trades
        if results['trades']:
            cprint(f"\n  Recent Trades:", "yellow")
            for trade in results['trades'][-5:]:
                color = "green" if trade['pnl_pct'] > 0 else "red"
                cprint(f"    {trade['direction']:5} | Entry: ${trade['entry']:.2f} → Exit: ${trade['exit']:.2f} | {trade['pnl_pct']:+.2f}% ({trade['exit_reason']})", color)


class BBSqueezeBot:
    """Live trading bot for BB Squeeze strategy."""

    def __init__(self, mock_mode: bool = False, dry_run: bool = False):
        self.mock_mode = mock_mode
        self.dry_run = dry_run
        self.api = None
        self.positions = {}

        if not mock_mode:
            if not ALPACA_AVAILABLE:
                raise RuntimeError("alpaca-trade-api not installed")
            base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
            self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

        self._print_banner()

    def _print_banner(self):
        """Print startup banner."""
        cprint("\n" + "=" * 60, "magenta")
        cprint("  BB SQUEEZE BOT", "magenta", attrs=['bold'])
        cprint("  Moon Dev's Volatility Compression Strategy", "magenta")
        cprint("=" * 60, "magenta")

        cprint("\n  STRATEGY RULES:", "yellow")
        cprint("  ─────────────────────────────────────", "white")
        cprint(f"  1. Detect squeeze: BB inside KC", "white")
        cprint(f"  2. Wait for release: BB expands outside KC", "white")
        cprint(f"  3. Confirm with ADX > {ADX_THRESHOLD}", "white")
        cprint(f"  4. Enter in momentum direction", "white")
        cprint(f"  5. SL: {SL_ATR_MULT}x ATR | TP: {TP_ATR_MULT}x ATR", "white")
        cprint("=" * 60 + "\n", "magenta")

    def fetch_data(self, symbol: str, days: int = 50) -> Optional[pd.DataFrame]:
        """Fetch recent data for live trading."""
        if self.mock_mode:
            # Generate mock data
            return self._generate_mock_data(symbol, days)

        try:
            end = datetime.now()
            start = end - timedelta(days=days)

            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Day,
                start=start.strftime('%Y-%m-%d'),
                end=end.strftime('%Y-%m-%d'),
                feed='iex'
            ).df

            if bars.empty:
                return None

            bars = bars.reset_index()
            # Don't reassign columns - they're already correctly named by Alpaca
            return bars

        except Exception as e:
            cprint(f"Error fetching data for {symbol}: {e}", "red")
            return None

    def _generate_mock_data(self, symbol: str, days: int) -> pd.DataFrame:
        """Generate mock OHLCV data."""
        import random

        base_prices = {'TSLA': 410, 'AMD': 125, 'NVDA': 140}
        base = base_prices.get(symbol, 100)

        data = []
        price = base

        for i in range(days):
            volatility = price * 0.02
            open_price = price
            high = open_price + random.uniform(0, volatility)
            low = open_price - random.uniform(0, volatility)
            close = open_price + random.uniform(-volatility/2, volatility/2)

            data.append({
                'timestamp': datetime.now() - timedelta(days=days-i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': random.uniform(1000000, 5000000)
            })
            price = close

        return pd.DataFrame(data)

    def check_signal(self, symbol: str) -> Optional[Dict]:
        """Check for BB Squeeze signal."""
        df = self.fetch_data(symbol)
        if df is None or len(df) < 30:
            return None

        # Calculate indicators
        backtester = BBSqueezeBacktester()
        df = backtester.calculate_indicators(df)

        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Check for signal
        signal = None

        if latest['squeeze_release'] and latest['adx'] >= ADX_THRESHOLD:
            direction = 'LONG' if latest['momentum'] > 0 else 'SHORT'
            atr = latest['atr']

            if direction == 'LONG':
                stop_loss = latest['close'] - (SL_ATR_MULT * atr)
                take_profit = latest['close'] + (TP_ATR_MULT * atr)
            else:
                stop_loss = latest['close'] + (SL_ATR_MULT * atr)
                take_profit = latest['close'] - (TP_ATR_MULT * atr)

            signal = {
                'symbol': symbol,
                'direction': direction,
                'entry': latest['close'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr,
                'adx': latest['adx'],
                'squeeze_on': prev['squeeze_on']
            }

        return signal

    def run(self, symbols: List[str], interval: int = 60):
        """Run the bot."""
        cprint(f"Starting BB Squeeze Bot on {symbols}...", "green")
        cprint(f"Check interval: {interval}s", "cyan")

        import schedule

        def check_all():
            cprint(f"\n{'─'*50}", "cyan")
            cprint(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "cyan")
            cprint(f"{'─'*50}", "cyan")

            for symbol in symbols:
                signal = self.check_signal(symbol)

                if signal:
                    cprint(f"\n  🔥 SQUEEZE RELEASE: {symbol}", "yellow")
                    cprint(f"    Direction: {signal['direction']}", "magenta")
                    cprint(f"    Entry: ${signal['entry']:.2f}", "white")
                    cprint(f"    SL: ${signal['stop_loss']:.2f} | TP: ${signal['take_profit']:.2f}", "white")
                    cprint(f"    ADX: {signal['adx']:.1f} | ATR: ${signal['atr']:.2f}", "white")
                else:
                    df = self.fetch_data(symbol)
                    if df is not None:
                        backtester = BBSqueezeBacktester()
                        df = backtester.calculate_indicators(df)
                        latest = df.iloc[-1]

                        squeeze_status = "SQUEEZE ON" if latest['squeeze_on'] else "No squeeze"
                        cprint(f"  {symbol}: ${latest['close']:.2f} | {squeeze_status} | ADX: {latest['adx']:.1f}", "white")

        check_all()
        schedule.every(interval).seconds.do(check_all)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            cprint("\nBot stopped.", "yellow")


def run_comparison_backtest():
    """Run backtest on TSLA, AMD, NVDA and compare results."""
    symbols = ['TSLA', 'AMD', 'NVDA']
    backtester = BBSqueezeBacktester()

    all_results = []

    for symbol in symbols:
        results = backtester.run_backtest(symbol, days=365)
        if 'error' not in results:
            all_results.append(results)

    # Print comparison summary
    cprint("\n" + "=" * 80, "cyan")
    cprint("  BB SQUEEZE STRATEGY - COMPARISON SUMMARY", "cyan", attrs=['bold'])
    cprint("=" * 80, "cyan")

    cprint(f"\n  {'Symbol':<8} {'Trades':<8} {'Win%':<8} {'Return':<10} {'PF':<8} {'Sharpe':<8} {'MaxDD':<8} {'B&H':<10} {'vs B&H':<10}", "yellow")
    cprint(f"  {'─'*78}", "white")

    for r in all_results:
        vs_bh = r['total_return'] - r['buy_hold_return']
        color = "green" if vs_bh > 0 else "red"

        cprint(f"  {r['symbol']:<8} {r['total_trades']:<8} {r['win_rate']:<8.1f} {r['total_return']:<+10.2f} {r['profit_factor']:<8.2f} {r['sharpe_ratio']:<8.2f} {r['max_drawdown']:<8.2f} {r['buy_hold_return']:<+10.2f} {vs_bh:<+10.2f}", color)

    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame([{
            'strategy': 'bb_squeeze',
            'symbol': r['symbol'],
            'total_trades': r['total_trades'],
            'win_rate': r['win_rate'],
            'total_return': r['total_return'],
            'avg_win': r['avg_win'],
            'avg_loss': r['avg_loss'],
            'profit_factor': r['profit_factor'],
            'sharpe_ratio': r['sharpe_ratio'],
            'max_drawdown': r['max_drawdown'],
            'buy_hold_return': r['buy_hold_return'],
            'vs_buy_hold': r['total_return'] - r['buy_hold_return']
        } for r in all_results])

        output_path = '/Users/josiahgarcia/trading-bot/csvs/bb_squeeze_results.csv'
        results_df.to_csv(output_path, index=False)
        cprint(f"\n  Results saved to: {output_path}", "green")

    # Compare to other strategies
    cprint("\n" + "=" * 80, "cyan")
    cprint("  COMPARISON TO OTHER STRATEGIES (from promising_strategies.csv)", "cyan")
    cprint("=" * 80, "cyan")

    try:
        promising = pd.read_csv('/Users/josiahgarcia/trading-bot/csvs/promising_strategies.csv')

        cprint(f"\n  Top Strategies by Sharpe Ratio:", "yellow")
        cprint(f"  {'Strategy':<35} {'Trades':<8} {'Win%':<8} {'Return':<10} {'Sharpe':<8}", "yellow")
        cprint(f"  {'─'*70}", "white")

        # Add BB Squeeze to comparison
        for r in all_results:
            cprint(f"  {'bb_squeeze_' + r['symbol']:<35} {r['total_trades']:<8} {r['win_rate']:<8.1f} {r['total_return']:<+10.2f} {r['sharpe_ratio']:<8.2f}", "magenta")

        cprint(f"  {'─'*70}", "white")

        # Show top from promising strategies
        top_promising = promising.nlargest(5, 'sharpe_ratio')
        for _, row in top_promising.iterrows():
            cprint(f"  {row['strategy']:<35} {row['total_trades']:<8} {row['win_rate']:<8.1f} {row['total_return']:<+10.2f} {row['sharpe_ratio']:<8.2f}", "white")

    except Exception as e:
        cprint(f"  Could not load comparison data: {e}", "yellow")

    return all_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="BB Squeeze Strategy Bot")
    parser.add_argument("--backtest", action="store_true", help="Run backtest mode")
    parser.add_argument("--live", action="store_true", help="Run live trading mode")
    parser.add_argument("--mock", action="store_true", help="Use mock data")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no real trades)")
    parser.add_argument("--symbols", nargs="+", default=['TSLA', 'AMD', 'NVDA'], help="Symbols to trade")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()

    if args.backtest:
        run_comparison_backtest()
    elif args.live or args.mock:
        bot = BBSqueezeBot(mock_mode=args.mock, dry_run=args.dry_run)
        bot.run(symbols=args.symbols, interval=args.interval)
    else:
        # Default to backtest
        run_comparison_backtest()


if __name__ == "__main__":
    main()

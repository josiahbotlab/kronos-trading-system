"""
Breakout Strategy Backtester

Backtests the 24-hour high/low breakout strategy using backtesting.py.
- Entry: Price breaks above 24h high (LONG) or below 24h low (SHORT)
- Stop Loss: 1.5x ATR below entry
- Take Profit: 2:1 risk/reward ratio

Usage:
    python src/backtesting/backtest_breakout.py
    python src/backtesting/backtest_breakout.py --symbol AAPL
    python src/backtesting/backtest_breakout.py --days 365
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from termcolor import cprint

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Backtesting library
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    cprint("Error: backtesting.py not installed. Run: pip install backtesting", "red")

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    cprint("Error: alpaca-trade-api not installed", "red")

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Strategy parameters
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
RISK_REWARD_RATIO = 2.0
LOOKBACK_PERIOD = 24  # Hours for high/low calculation

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
RESULTS_HTML_PATH = CSV_DIR / 'backtest_results.html'
RESULTS_CSV_PATH = CSV_DIR / 'backtest_trades.csv'


def fetch_historical_data(symbol='SPY', days=365):
    """
    Fetch historical hourly data from Alpaca.

    Args:
        symbol: Stock symbol
        days: Number of days of history

    Returns:
        DataFrame with OHLCV data
    """
    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-trade-api not installed")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    cprint(f"Fetching {symbol} data from {start_date.date()} to {end_date.date()}...", "cyan")

    # Format dates as YYYY-MM-DD for Alpaca API
    # Use data from 15+ days ago to avoid SIP data restrictions on free tier
    end_date = end_date - timedelta(days=15)
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    cprint(f"Adjusted date range: {start_str} to {end_str}", "white")

    # Fetch hourly bars using IEX feed (free tier)
    bars = api.get_bars(
        symbol,
        tradeapi.TimeFrame.Hour,
        start=start_str,
        end=end_str,
        limit=10000,
        feed='iex'
    ).df

    if bars.empty:
        raise ValueError(f"No data returned for {symbol}")

    # Reset index and rename columns for backtesting.py
    bars = bars.reset_index()
    bars = bars.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    # Set Date as index
    bars['Date'] = pd.to_datetime(bars['Date'])
    bars = bars.set_index('Date')

    # Remove timezone info for backtesting.py compatibility
    bars.index = bars.index.tz_localize(None)

    cprint(f"Fetched {len(bars)} hourly bars", "green")

    return bars[['Open', 'High', 'Low', 'Close', 'Volume']]


def calculate_atr(high, low, close, period=ATR_PERIOD):
    """Calculate Average True Range."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


class BreakoutStrategy(Strategy):
    """
    24-Hour Breakout Strategy

    - LONG: Price breaks above 24-hour high
    - Stop Loss: 1.5x ATR below entry
    - Take Profit: 2:1 risk/reward ratio
    """

    # Strategy parameters (can be optimized)
    atr_period = ATR_PERIOD
    atr_multiplier = ATR_MULTIPLIER
    risk_reward = RISK_REWARD_RATIO
    lookback = LOOKBACK_PERIOD

    def init(self):
        """Initialize indicators."""
        # Calculate rolling high/low over lookback period
        self.rolling_high = self.I(
            lambda x: pd.Series(x).rolling(self.lookback).max(),
            self.data.High,
            name='Rolling High'
        )
        self.rolling_low = self.I(
            lambda x: pd.Series(x).rolling(self.lookback).min(),
            self.data.Low,
            name='Rolling Low'
        )

        # Calculate ATR
        self.atr = self.I(
            calculate_atr,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name='ATR'
        )

        # Track if we're in a position
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None

    def next(self):
        """Execute strategy logic for each bar."""
        price = self.data.Close[-1]
        atr = self.atr[-1]
        high_24h = self.rolling_high[-2] if len(self.rolling_high) > 1 else self.rolling_high[-1]
        low_24h = self.rolling_low[-2] if len(self.rolling_low) > 1 else self.rolling_low[-1]

        # Skip if ATR is not valid
        if pd.isna(atr) or atr <= 0:
            return

        # If we have a position, check for exit
        if self.position:
            # Check stop loss and take profit
            if self.position.is_long:
                if price <= self.stop_loss:
                    self.position.close()
                    return
                elif price >= self.take_profit:
                    self.position.close()
                    return
            else:  # Short position
                if price >= self.stop_loss:
                    self.position.close()
                    return
                elif price <= self.take_profit:
                    self.position.close()
                    return

        # No position - look for entry
        if not self.position:
            # Buffer for breakout confirmation (0.1%)
            buffer = high_24h * 0.001

            # LONG entry: Price breaks above 24h high
            if price > high_24h + buffer:
                stop_distance = atr * self.atr_multiplier
                self.stop_loss = price - stop_distance
                self.take_profit = price + (stop_distance * self.risk_reward)
                self.entry_price = price

                # Calculate position size (use 95% of equity to avoid margin issues)
                self.buy(sl=self.stop_loss, tp=self.take_profit)

            # SHORT entry: Price breaks below 24h low (commented out - long only for now)
            # elif price < low_24h - buffer:
            #     stop_distance = atr * self.atr_multiplier
            #     self.stop_loss = price + stop_distance
            #     self.take_profit = price - (stop_distance * self.risk_reward)
            #     self.entry_price = price
            #     self.sell(sl=self.stop_loss, tp=self.take_profit)


def run_backtest(symbol='SPY', days=365, cash=100000, commission=0.001):
    """
    Run the backtest and return results.

    Args:
        symbol: Stock symbol to backtest
        days: Number of days of historical data
        cash: Starting cash
        commission: Commission per trade (0.1% default)

    Returns:
        Backtest stats dictionary
    """
    if not BACKTESTING_AVAILABLE:
        raise RuntimeError("backtesting.py library not installed")

    # Fetch data
    data = fetch_historical_data(symbol, days)

    # Run backtest
    cprint(f"\nRunning backtest on {symbol}...", "cyan")

    bt = Backtest(
        data,
        BreakoutStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=True
    )

    stats = bt.run()

    # Save results
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Save HTML chart
    bt.plot(filename=str(RESULTS_HTML_PATH), open_browser=False)
    cprint(f"\nChart saved to: {RESULTS_HTML_PATH}", "green")

    # Save trades to CSV
    if hasattr(stats, '_trades') and stats._trades is not None and len(stats._trades) > 0:
        trades_df = stats._trades
        trades_df.to_csv(RESULTS_CSV_PATH)
        cprint(f"Trades saved to: {RESULTS_CSV_PATH}", "green")

    return stats


def print_results(stats, symbol):
    """Print formatted backtest results."""
    cprint("\n" + "=" * 60, "cyan")
    cprint(f"  BACKTEST RESULTS - {symbol} Breakout Strategy", "cyan")
    cprint("=" * 60, "cyan")

    # Calculate additional metrics
    total_return = stats['Return [%]']
    buy_hold_return = stats['Buy & Hold Return [%]']
    win_rate = stats['Win Rate [%]']
    profit_factor = stats.get('Profit Factor', 0)
    max_drawdown = stats['Max. Drawdown [%]']
    num_trades = stats['# Trades']
    avg_trade = stats.get('Avg. Trade [%]', 0)

    # Determine colors based on performance
    return_color = "green" if total_return > 0 else "red"
    vs_bh_color = "green" if total_return > buy_hold_return else "yellow"

    cprint(f"\n{'PERFORMANCE METRICS':-^60}", "white")
    cprint(f"  Total Return:        {total_return:>10.2f}%", return_color)
    cprint(f"  Buy & Hold Return:   {buy_hold_return:>10.2f}%", "white")
    cprint(f"  Strategy vs B&H:     {total_return - buy_hold_return:>+10.2f}%", vs_bh_color)

    cprint(f"\n{'TRADE STATISTICS':-^60}", "white")
    cprint(f"  Number of Trades:    {num_trades:>10}", "white")
    cprint(f"  Win Rate:            {win_rate:>10.1f}%", "green" if win_rate > 50 else "yellow")
    cprint(f"  Profit Factor:       {profit_factor:>10.2f}", "green" if profit_factor > 1 else "red")
    cprint(f"  Avg Trade Return:    {avg_trade:>10.2f}%", "green" if avg_trade > 0 else "red")

    cprint(f"\n{'RISK METRICS':-^60}", "white")
    cprint(f"  Max Drawdown:        {max_drawdown:>10.2f}%", "red" if max_drawdown < -20 else "yellow")
    cprint(f"  Sharpe Ratio:        {stats.get('Sharpe Ratio', 0):>10.2f}", "white")
    cprint(f"  Sortino Ratio:       {stats.get('Sortino Ratio', 0):>10.2f}", "white")
    cprint(f"  Calmar Ratio:        {stats.get('Calmar Ratio', 0):>10.2f}", "white")

    cprint(f"\n{'EQUITY':-^60}", "white")
    cprint(f"  Start:               ${stats.get('_equity_curve', {}).get('Equity', [100000])[0] if hasattr(stats, '_equity_curve') else 100000:>10,.2f}", "white")
    cprint(f"  Final:               ${stats['Equity Final [$]']:>10,.2f}", return_color)
    cprint(f"  Peak:                ${stats['Equity Peak [$]']:>10,.2f}", "green")

    cprint("\n" + "=" * 60, "cyan")

    # Strategy parameters used
    cprint(f"\n{'STRATEGY PARAMETERS':-^60}", "white")
    cprint(f"  Lookback Period:     {LOOKBACK_PERIOD} hours", "white")
    cprint(f"  ATR Period:          {ATR_PERIOD}", "white")
    cprint(f"  ATR Multiplier:      {ATR_MULTIPLIER}x", "white")
    cprint(f"  Risk/Reward:         {RISK_REWARD_RATIO}:1", "white")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest Breakout Strategy")
    parser.add_argument("--symbol", type=str, default="SPY", help="Symbol to backtest (default: SPY)")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    parser.add_argument("--cash", type=int, default=100000, help="Starting cash (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate (default: 0.001)")
    args = parser.parse_args()

    cprint("\n" + "=" * 60, "cyan")
    cprint("  BREAKOUT STRATEGY BACKTESTER", "cyan")
    cprint("=" * 60, "cyan")
    cprint(f"  Symbol: {args.symbol}", "white")
    cprint(f"  Period: {args.days} days", "white")
    cprint(f"  Starting Capital: ${args.cash:,}", "white")
    cprint(f"  Commission: {args.commission * 100:.2f}%", "white")

    try:
        stats = run_backtest(
            symbol=args.symbol,
            days=args.days,
            cash=args.cash,
            commission=args.commission
        )

        print_results(stats, args.symbol)

        # Return key metrics for programmatic use
        return {
            'total_return_pct': stats['Return [%]'],
            'win_rate_pct': stats['Win Rate [%]'],
            'profit_factor': stats.get('Profit Factor', 0),
            'max_drawdown_pct': stats['Max. Drawdown [%]'],
            'num_trades': stats['# Trades'],
            'sharpe_ratio': stats.get('Sharpe Ratio', 0),
            'final_equity': stats['Equity Final [$]']
        }

    except Exception as e:
        cprint(f"\nError running backtest: {e}", "red")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

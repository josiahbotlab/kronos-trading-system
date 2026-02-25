"""
Bollinger Band Breakout Strategy Backtester

Moon Dev's volatility breakout strategy:
- Bollinger Bands: 21 period, 2.7 standard deviations
- Entry: Price CLOSES ABOVE upper band (buying strength)
- Confirmation: Heikin Ashi candle must be GREEN
- Exit: Price returns to middle band (21 SMA) or stop loss
- Stop Loss: 2% below entry

Usage:
    python src/backtesting/backtest_bollinger_breakout.py
    python src/backtesting/backtest_bollinger_breakout.py --symbol TSLA
    python src/backtesting/backtest_bollinger_breakout.py --symbol SPY --days 365
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

# Strategy parameters (Moon Dev's specs)
BB_PERIOD = 21
BB_STD_DEV = 2.7
STOP_LOSS_PCT = 2.0

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
RESULTS_HTML_PATH = CSV_DIR / 'backtest_bollinger_breakout.html'
RESULTS_CSV_PATH = CSV_DIR / 'backtest_bollinger_breakout_trades.csv'


def fetch_historical_data(symbol='TSLA', days=365):
    """Fetch historical hourly data from Alpaca."""
    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-trade-api not installed")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    # Use data from 15+ days ago to avoid SIP restrictions
    end_date = datetime.now() - timedelta(days=15)
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    cprint(f"Fetching {symbol} data from {start_str} to {end_str}...", "cyan")

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

    bars = bars.reset_index()
    bars = bars.rename(columns={
        'timestamp': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })

    bars['Date'] = pd.to_datetime(bars['Date'])
    bars = bars.set_index('Date')
    bars.index = bars.index.tz_localize(None)

    cprint(f"Fetched {len(bars)} hourly bars", "green")

    return bars[['Open', 'High', 'Low', 'Close', 'Volume']]


def calculate_bollinger_bands(close, period=BB_PERIOD, std_dev=BB_STD_DEV):
    """Calculate Bollinger Bands."""
    middle = pd.Series(close).rolling(window=period).mean()
    std = pd.Series(close).rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    return middle, upper, lower


def calculate_heikin_ashi(open_prices, high_prices, low_prices, close_prices):
    """
    Calculate Heikin Ashi candles.
    Returns HA close and whether HA is green (1) or red (0).
    """
    ha_close = (open_prices + high_prices + low_prices + close_prices) / 4

    ha_open = pd.Series(index=open_prices.index, dtype=float)
    ha_open.iloc[0] = (open_prices.iloc[0] + close_prices.iloc[0]) / 2

    for i in range(1, len(ha_open)):
        ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

    # Green = 1, Red = 0
    ha_green = (ha_close > ha_open).astype(int)

    return ha_close, ha_open, ha_green


class BollingerBreakoutStrategy(Strategy):
    """
    Bollinger Band Breakout Strategy

    Moon Dev's rules:
    - Entry: Price closes above upper band (2.7 std) + Heikin Ashi is green
    - Exit: Price returns to middle band OR stop loss (2%)
    """

    bb_period = BB_PERIOD
    bb_std_dev = BB_STD_DEV
    stop_loss_pct = STOP_LOSS_PCT

    def init(self):
        """Initialize indicators."""
        close = pd.Series(self.data.Close)
        open_p = pd.Series(self.data.Open)
        high = pd.Series(self.data.High)
        low = pd.Series(self.data.Low)

        # Bollinger Bands
        self.bb_middle, self.bb_upper, self.bb_lower = self.I(
            calculate_bollinger_bands,
            close,
            self.bb_period,
            self.bb_std_dev,
            name='BB'
        )

        # Heikin Ashi
        ha_close, ha_open, ha_green = calculate_heikin_ashi(open_p, high, low, close)
        self.ha_green = self.I(lambda: ha_green, name='HA_Green')

        self.entry_price = None

    def next(self):
        """Execute strategy logic."""
        price = self.data.Close[-1]
        upper = self.bb_upper[-1]
        middle = self.bb_middle[-1]
        ha_is_green = self.ha_green[-1] == 1

        # Skip if indicators not ready
        if pd.isna(upper) or pd.isna(middle):
            return

        # If in position, check exit
        if self.position:
            stop_price = self.entry_price * (1 - self.stop_loss_pct / 100)

            # Stop loss
            if price <= stop_price:
                self.position.close()
                self.entry_price = None
                return

            # Target: return to middle band
            if price <= middle:
                self.position.close()
                self.entry_price = None
                return

        # No position - check entry
        if not self.position:
            # Entry: price above upper band + HA green
            if price > upper and ha_is_green:
                self.entry_price = price
                stop_loss = price * (1 - self.stop_loss_pct / 100)
                self.buy(sl=stop_loss)


def run_backtest(symbol='TSLA', days=365, cash=100000, commission=0.001):
    """Run the backtest."""
    if not BACKTESTING_AVAILABLE:
        raise RuntimeError("backtesting.py library not installed")

    data = fetch_historical_data(symbol, days)

    cprint(f"\nRunning Bollinger Breakout backtest on {symbol}...", "magenta")

    bt = Backtest(
        data,
        BollingerBreakoutStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True,
        trade_on_close=True
    )

    stats = bt.run()

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    bt.plot(filename=str(RESULTS_HTML_PATH), open_browser=False)
    cprint(f"\nChart saved to: {RESULTS_HTML_PATH}", "green")

    if hasattr(stats, '_trades') and stats._trades is not None and len(stats._trades) > 0:
        stats._trades.to_csv(RESULTS_CSV_PATH)
        cprint(f"Trades saved to: {RESULTS_CSV_PATH}", "green")

    return stats


def print_results(stats, symbol):
    """Print formatted results."""
    cprint("\n" + "=" * 60, "magenta")
    cprint(f"  BOLLINGER BREAKOUT BACKTEST - {symbol}", "magenta")
    cprint("=" * 60, "magenta")

    total_return = stats['Return [%]']
    buy_hold_return = stats['Buy & Hold Return [%]']
    win_rate = stats['Win Rate [%]']
    profit_factor = stats.get('Profit Factor', 0)
    max_drawdown = stats['Max. Drawdown [%]']
    num_trades = stats['# Trades']
    avg_trade = stats.get('Avg. Trade [%]', 0)

    return_color = "green" if total_return > 0 else "red"
    vs_bh_color = "green" if total_return > buy_hold_return else "yellow"

    cprint(f"\n{'STRATEGY PARAMETERS (Moon Dev Specs)':-^60}", "white")
    cprint(f"  Bollinger Period:    {BB_PERIOD}", "white")
    cprint(f"  Bollinger Std Dev:   {BB_STD_DEV}", "white")
    cprint(f"  Stop Loss:           {STOP_LOSS_PCT}%", "white")
    cprint(f"  Entry:               Close > Upper Band + HA Green", "white")
    cprint(f"  Exit:                Close <= Middle Band", "white")

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
    cprint(f"  Start:               ${100000:>10,.2f}", "white")
    cprint(f"  Final:               ${stats['Equity Final [$]']:>10,.2f}", return_color)
    cprint(f"  Peak:                ${stats['Equity Peak [$]']:>10,.2f}", "green")

    cprint("\n" + "=" * 60, "magenta")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest Bollinger Breakout Strategy")
    parser.add_argument("--symbol", type=str, default="TSLA", help="Symbol to backtest")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--cash", type=int, default=100000, help="Starting cash")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    args = parser.parse_args()

    cprint("\n" + "=" * 60, "magenta")
    cprint("  BOLLINGER BREAKOUT BACKTESTER", "magenta")
    cprint("  Moon Dev's Volatility Breakout Strategy", "magenta")
    cprint("=" * 60, "magenta")
    cprint(f"  Symbol: {args.symbol}", "white")
    cprint(f"  Period: {args.days} days", "white")
    cprint(f"  Capital: ${args.cash:,}", "white")

    try:
        stats = run_backtest(
            symbol=args.symbol,
            days=args.days,
            cash=args.cash,
            commission=args.commission
        )

        print_results(stats, args.symbol)

        return {
            'symbol': args.symbol,
            'total_return_pct': stats['Return [%]'],
            'buy_hold_return_pct': stats['Buy & Hold Return [%]'],
            'win_rate_pct': stats['Win Rate [%]'],
            'profit_factor': stats.get('Profit Factor', 0),
            'max_drawdown_pct': stats['Max. Drawdown [%]'],
            'num_trades': stats['# Trades'],
            'sharpe_ratio': stats.get('Sharpe Ratio', 0),
            'final_equity': stats['Equity Final [$]']
        }

    except Exception as e:
        cprint(f"\nError: {e}", "red")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

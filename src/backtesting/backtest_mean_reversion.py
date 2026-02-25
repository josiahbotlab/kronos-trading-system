"""
Mean Reversion Strategy Backtester

Backtests the RSI/SMA mean reversion strategy using backtesting.py.
- Entry: RSI < 30 AND price > 2% below 20-period SMA
- Exit: RSI > 50 OR price >= SMA OR stop loss (3%)

Usage:
    python src/backtesting/backtest_mean_reversion.py
    python src/backtesting/backtest_mean_reversion.py --symbol SPY
    python src/backtesting/backtest_mean_reversion.py --days 365
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

# Strategy parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_EXIT = 50
SMA_PERIOD = 20
SMA_DEVIATION = 2.0  # 2% below SMA
STOP_LOSS_PCT = 3.0  # 3% stop loss

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
RESULTS_HTML_PATH = CSV_DIR / 'backtest_mean_reversion.html'
RESULTS_CSV_PATH = CSV_DIR / 'backtest_mean_reversion_trades.csv'


def fetch_historical_data(symbol='SPY', days=365):
    """
    Fetch historical hourly data from Alpaca.
    """
    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-trade-api not installed")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    # Use data from 15+ days ago to avoid SIP data restrictions
    end_date = datetime.now() - timedelta(days=15)
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    cprint(f"Fetching {symbol} data from {start_str} to {end_str}...", "cyan")

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


def calculate_rsi(prices, period=RSI_PERIOD):
    """Calculate RSI indicator."""
    delta = pd.Series(prices).diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_sma(prices, period=SMA_PERIOD):
    """Calculate Simple Moving Average."""
    return pd.Series(prices).rolling(window=period).mean()


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy

    - Entry: RSI < 30 AND price > 2% below 20-period SMA
    - Exit: RSI > 50 OR price >= SMA OR stop loss (3%)
    """

    # Strategy parameters (can be optimized)
    rsi_period = RSI_PERIOD
    rsi_oversold = RSI_OVERSOLD
    rsi_exit = RSI_EXIT
    sma_period = SMA_PERIOD
    sma_deviation = SMA_DEVIATION
    stop_loss_pct = STOP_LOSS_PCT

    def init(self):
        """Initialize indicators."""
        # Calculate RSI
        self.rsi = self.I(
            calculate_rsi,
            self.data.Close,
            self.rsi_period,
            name='RSI'
        )

        # Calculate SMA
        self.sma = self.I(
            calculate_sma,
            self.data.Close,
            self.sma_period,
            name='SMA'
        )

        # Track entry price for stop loss
        self.entry_price = None

    def next(self):
        """Execute strategy logic for each bar."""
        price = self.data.Close[-1]
        rsi = self.rsi[-1]
        sma = self.sma[-1]

        # Skip if indicators not ready
        if pd.isna(rsi) or pd.isna(sma):
            return

        # Calculate deviation from SMA
        sma_dev = ((sma - price) / sma * 100) if sma > 0 else 0

        # If we have a position, check for exit
        if self.position:
            stop_price = self.entry_price * (1 - self.stop_loss_pct / 100)

            # Exit conditions
            if price <= stop_price:
                self.position.close()
                self.entry_price = None
                return
            elif rsi > self.rsi_exit:
                self.position.close()
                self.entry_price = None
                return
            elif price >= sma:
                self.position.close()
                self.entry_price = None
                return

        # No position - look for entry
        if not self.position:
            # Entry conditions: RSI oversold AND price below SMA
            rsi_oversold = rsi < self.rsi_oversold
            below_sma = sma_dev >= self.sma_deviation

            if rsi_oversold and below_sma:
                self.entry_price = price
                stop_loss = price * (1 - self.stop_loss_pct / 100)
                self.buy(sl=stop_loss)


def run_backtest(symbol='SPY', days=365, cash=100000, commission=0.001):
    """
    Run the backtest and return results.
    """
    if not BACKTESTING_AVAILABLE:
        raise RuntimeError("backtesting.py library not installed")

    # Fetch data
    data = fetch_historical_data(symbol, days)

    # Run backtest
    cprint(f"\nRunning mean reversion backtest on {symbol}...", "cyan")

    bt = Backtest(
        data,
        MeanReversionStrategy,
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
    cprint(f"  MEAN REVERSION BACKTEST - {symbol}", "cyan")
    cprint("=" * 60, "cyan")

    # Calculate metrics
    total_return = stats['Return [%]']
    buy_hold_return = stats['Buy & Hold Return [%]']
    win_rate = stats['Win Rate [%]']
    profit_factor = stats.get('Profit Factor', 0)
    max_drawdown = stats['Max. Drawdown [%]']
    num_trades = stats['# Trades']
    avg_trade = stats.get('Avg. Trade [%]', 0)

    # Determine colors
    return_color = "green" if total_return > 0 else "red"
    vs_bh_color = "green" if total_return > buy_hold_return else "yellow"

    cprint(f"\n{'STRATEGY PARAMETERS':-^60}", "white")
    cprint(f"  RSI Period:          {RSI_PERIOD}", "white")
    cprint(f"  RSI Oversold:        < {RSI_OVERSOLD}", "white")
    cprint(f"  RSI Exit:            > {RSI_EXIT}", "white")
    cprint(f"  SMA Period:          {SMA_PERIOD}", "white")
    cprint(f"  SMA Deviation:       > {SMA_DEVIATION}% below", "white")
    cprint(f"  Stop Loss:           {STOP_LOSS_PCT}%", "white")

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

    cprint("\n" + "=" * 60, "cyan")

    # Compare with breakout strategy
    cprint(f"\n{'STRATEGY COMPARISON NOTE':-^60}", "yellow")
    cprint("  Mean reversion typically works better on range-bound", "white")
    cprint("  markets like SPY, while breakouts work better on", "white")
    cprint("  trending/volatile stocks like TSLA.", "white")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest Mean Reversion Strategy")
    parser.add_argument("--symbol", type=str, default="SPY", help="Symbol to backtest (default: SPY)")
    parser.add_argument("--days", type=int, default=365, help="Days of history (default: 365)")
    parser.add_argument("--cash", type=int, default=100000, help="Starting cash (default: 100000)")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate (default: 0.001)")
    args = parser.parse_args()

    cprint("\n" + "=" * 60, "cyan")
    cprint("  MEAN REVERSION STRATEGY BACKTESTER", "cyan")
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

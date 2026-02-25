"""
Gap and Go Strategy Backtester

Morning momentum strategy for gap ups:
- Entry: Stock gaps UP 2%+ from previous close at open
- Volume: Must be 2x average (simulated in backtest)
- Take Profit: 3% above entry (open price)
- Stop Loss: 2% below entry
- Time Exit: Close at end of day if no TP/SL (simulates 11 AM exit)

Note: This backtest uses daily data to simulate intraday behavior.
We use daily high/low to determine if TP/SL would be hit during the day.

Usage:
    python src/backtesting/backtest_gap_and_go.py
    python src/backtesting/backtest_gap_and_go.py --symbol TSLA
    python src/backtesting/backtest_gap_and_go.py --symbol NVDA --days 365
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
MIN_GAP_PCT = 1.5           # Minimum gap up percentage (lowered for more signals)
VOLUME_MULTIPLIER = 1.5     # Volume must be 1.5x average (lowered for IEX data)
TAKE_PROFIT_PCT = 3.0       # 3% take profit
STOP_LOSS_PCT = 2.0         # 2% stop loss
VOLUME_AVG_PERIOD = 20      # 20-day average volume

# Paths
CSV_DIR = Path(__file__).parent.parent.parent / 'csvs'
RESULTS_CSV_PATH = CSV_DIR / 'backtest_gap_and_go_trades.csv'


def fetch_daily_data(symbol='TSLA', days=365):
    """Fetch daily OHLCV data from Alpaca."""
    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-trade-api not installed")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    # Use data from 15+ days ago to avoid SIP restrictions
    end_date = datetime.now() - timedelta(days=15)
    start_date = end_date - timedelta(days=days + VOLUME_AVG_PERIOD + 10)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    cprint(f"Fetching {symbol} daily data from {start_str} to {end_str}...", "cyan")

    bars = api.get_bars(
        symbol,
        tradeapi.TimeFrame.Day,
        start=start_str,
        end=end_str,
        limit=days + VOLUME_AVG_PERIOD + 10,
        feed='iex'
    ).df

    if bars.empty:
        raise ValueError(f"No data returned for {symbol}")

    bars = bars.reset_index()

    # Rename columns properly (API returns lowercase)
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

    cprint(f"Fetched {len(bars)} daily bars", "green")

    return bars[['Open', 'High', 'Low', 'Close', 'Volume']]


def run_gap_and_go_backtest(data, initial_cash=100000, commission=0.001):
    """
    Run Gap and Go backtest on daily data.

    For each day:
    1. Check if gap >= 2% (open vs prev close)
    2. Check if volume >= 2x average
    3. If entry conditions met, buy at open
    4. Check if intraday high hits TP (3% above open)
    5. Check if intraday low hits SL (2% below open)
    6. If neither, exit at close (time-based exit)
    """
    trades = []
    cash = initial_cash
    position = None

    # Calculate average volume
    data['AvgVolume'] = data['Volume'].rolling(window=VOLUME_AVG_PERIOD).mean()
    data['VolumeRatio'] = data['Volume'] / data['AvgVolume']

    # Calculate gap percentage
    data['PrevClose'] = data['Close'].shift(1)
    data['GapPct'] = ((data['Open'] - data['PrevClose']) / data['PrevClose']) * 100

    # Skip first VOLUME_AVG_PERIOD days for avg volume calculation
    data = data.iloc[VOLUME_AVG_PERIOD:].copy()

    for i, (date, row) in enumerate(data.iterrows()):
        # Skip if we have an open position (shouldn't happen in gap strategy)
        if position is not None:
            continue

        # Check gap condition
        gap_pct = row['GapPct']
        if pd.isna(gap_pct) or gap_pct < MIN_GAP_PCT:
            continue

        # Check volume condition
        vol_ratio = row['VolumeRatio']
        if pd.isna(vol_ratio) or vol_ratio < VOLUME_MULTIPLIER:
            continue

        # Entry conditions met - buy at open
        entry_price = row['Open']
        shares = (cash * 0.95) / entry_price  # Use 95% of cash
        shares = round(shares, 4)

        if shares <= 0:
            continue

        position_value = shares * entry_price
        commission_cost = position_value * commission

        # Calculate TP and SL prices
        tp_price = entry_price * (1 + TAKE_PROFIT_PCT / 100)
        sl_price = entry_price * (1 - STOP_LOSS_PCT / 100)

        # Determine exit - check if intraday high/low would trigger TP/SL
        intraday_high = row['High']
        intraday_low = row['Low']
        close_price = row['Close']

        exit_price = None
        exit_reason = None

        # Check which would be hit first (assume SL checked before TP if both in range)
        # In reality, we'd need minute data to know the order
        # Simplified logic: if low <= SL, assume SL hit first; else if high >= TP, TP hit

        if intraday_low <= sl_price:
            exit_price = sl_price
            exit_reason = "Stop Loss"
        elif intraday_high >= tp_price:
            exit_price = tp_price
            exit_reason = "Take Profit"
        else:
            # Neither hit - exit at close (time-based exit simulation)
            exit_price = close_price
            exit_reason = "Time Exit (EOD)"

        # Calculate P&L
        pnl = (exit_price - entry_price) * shares
        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        exit_commission = (shares * exit_price) * commission
        net_pnl = pnl - commission_cost - exit_commission

        cash += net_pnl

        trades.append({
            'date': date,
            'symbol': 'N/A',  # Will be set by caller
            'gap_pct': gap_pct,
            'volume_ratio': vol_ratio,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'shares': shares,
            'pnl': net_pnl,
            'pnl_pct': pnl_pct,
            'equity': cash
        })

    return trades, cash


def analyze_results(trades, symbol, initial_cash=100000):
    """Analyze and print backtest results."""
    if not trades:
        cprint(f"\nNo trades found for {symbol}", "yellow")
        return None

    df = pd.DataFrame(trades)
    df['symbol'] = symbol

    # Calculate metrics
    total_trades = len(df)
    winners = len(df[df['pnl'] > 0])
    losers = len(df[df['pnl'] <= 0])
    win_rate = (winners / total_trades) * 100 if total_trades > 0 else 0

    total_pnl = df['pnl'].sum()
    total_return = ((df['equity'].iloc[-1] - initial_cash) / initial_cash) * 100

    avg_win = df[df['pnl'] > 0]['pnl'].mean() if winners > 0 else 0
    avg_loss = df[df['pnl'] <= 0]['pnl'].mean() if losers > 0 else 0
    profit_factor = abs(avg_win * winners / (avg_loss * losers)) if losers > 0 and avg_loss != 0 else 0

    avg_gap = df['gap_pct'].mean()
    avg_volume_ratio = df['volume_ratio'].mean()

    # Exit reason breakdown
    tp_exits = len(df[df['exit_reason'] == 'Take Profit'])
    sl_exits = len(df[df['exit_reason'] == 'Stop Loss'])
    time_exits = len(df[df['exit_reason'] == 'Time Exit (EOD)'])

    # Print results
    cprint("\n" + "=" * 60, "yellow")
    cprint(f"  GAP AND GO BACKTEST - {symbol}", "yellow")
    cprint("=" * 60, "yellow")

    cprint(f"\n{'STRATEGY PARAMETERS':-^60}", "white")
    cprint(f"  Min Gap:             {MIN_GAP_PCT}%", "white")
    cprint(f"  Volume Required:     {VOLUME_MULTIPLIER}x average", "white")
    cprint(f"  Take Profit:         {TAKE_PROFIT_PCT}%", "white")
    cprint(f"  Stop Loss:           {STOP_LOSS_PCT}%", "white")

    return_color = "green" if total_return > 0 else "red"
    cprint(f"\n{'PERFORMANCE METRICS':-^60}", "white")
    cprint(f"  Total Return:        {total_return:>10.2f}%", return_color)
    cprint(f"  Final Equity:        ${df['equity'].iloc[-1]:>10,.2f}", return_color)
    cprint(f"  Total P&L:           ${total_pnl:>10,.2f}", return_color)

    cprint(f"\n{'TRADE STATISTICS':-^60}", "white")
    cprint(f"  Total Trades:        {total_trades:>10}", "white")
    cprint(f"  Winners:             {winners:>10}", "green")
    cprint(f"  Losers:              {losers:>10}", "red")
    cprint(f"  Win Rate:            {win_rate:>10.1f}%", "green" if win_rate > 50 else "yellow")
    cprint(f"  Profit Factor:       {profit_factor:>10.2f}", "green" if profit_factor > 1 else "red")

    cprint(f"\n{'EXIT BREAKDOWN':-^60}", "white")
    cprint(f"  Take Profit:         {tp_exits:>10} ({tp_exits/total_trades*100:.1f}%)", "green")
    cprint(f"  Stop Loss:           {sl_exits:>10} ({sl_exits/total_trades*100:.1f}%)", "red")
    cprint(f"  Time Exit:           {time_exits:>10} ({time_exits/total_trades*100:.1f}%)", "yellow")

    cprint(f"\n{'GAP CHARACTERISTICS':-^60}", "white")
    cprint(f"  Avg Gap Size:        {avg_gap:>10.2f}%", "white")
    cprint(f"  Avg Volume Ratio:    {avg_volume_ratio:>10.1f}x", "white")

    # Per-exit-reason P&L
    cprint(f"\n{'P&L BY EXIT TYPE':-^60}", "white")
    for exit_type in ['Take Profit', 'Stop Loss', 'Time Exit (EOD)']:
        subset = df[df['exit_reason'] == exit_type]
        if len(subset) > 0:
            avg_pnl = subset['pnl_pct'].mean()
            color = "green" if avg_pnl > 0 else "red"
            cprint(f"  {exit_type:<20} Avg: {avg_pnl:>+6.2f}%", color)

    cprint("\n" + "=" * 60, "yellow")

    return {
        'symbol': symbol,
        'total_return_pct': total_return,
        'win_rate_pct': win_rate,
        'profit_factor': profit_factor,
        'num_trades': total_trades,
        'tp_exits': tp_exits,
        'sl_exits': sl_exits,
        'time_exits': time_exits,
        'avg_gap_pct': avg_gap,
        'final_equity': df['equity'].iloc[-1]
    }


def run_backtest(symbol='TSLA', days=365, cash=100000, commission=0.001):
    """Run the full backtest."""
    data = fetch_daily_data(symbol, days)

    cprint(f"\nRunning Gap and Go backtest on {symbol}...", "yellow")

    trades, final_cash = run_gap_and_go_backtest(data, cash, commission)

    # Save trades
    if trades:
        CSV_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(trades)
        df['symbol'] = symbol
        df.to_csv(RESULTS_CSV_PATH, index=False)
        cprint(f"Trades saved to: {RESULTS_CSV_PATH}", "green")

    return analyze_results(trades, symbol, cash)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Backtest Gap and Go Strategy")
    parser.add_argument("--symbol", type=str, default="TSLA", help="Symbol to backtest")
    parser.add_argument("--days", type=int, default=365, help="Days of history")
    parser.add_argument("--cash", type=int, default=100000, help="Starting cash")
    parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    args = parser.parse_args()

    cprint("\n" + "=" * 60, "yellow")
    cprint("  GAP AND GO BACKTESTER", "yellow")
    cprint("  Morning Momentum Strategy", "yellow")
    cprint("=" * 60, "yellow")
    cprint(f"  Symbol: {args.symbol}", "white")
    cprint(f"  Period: {args.days} days", "white")
    cprint(f"  Capital: ${args.cash:,}", "white")

    try:
        result = run_backtest(
            symbol=args.symbol,
            days=args.days,
            cash=args.cash,
            commission=args.commission
        )
        return result

    except Exception as e:
        cprint(f"\nError: {e}", "red")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()

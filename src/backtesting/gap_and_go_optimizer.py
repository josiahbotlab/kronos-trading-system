"""
Gap and Go Strategy Optimizer

Optimizes Gap and Go parameters using grid search with backtesting.py

Parameter Ranges:
- gap_threshold: 1.0% to 5.0%
- volume_multiplier: 1.0x to 3.0x
- profit_target: 2% to 8%
- stop_loss: 3% to 15%

Constraint: profit_target >= 0.4 * stop_loss (reasonable risk/reward)

Usage:
    python src/backtesting/gap_and_go_optimizer.py              # AMD (default)
    python src/backtesting/gap_and_go_optimizer.py --symbol TSLA
    python src/backtesting/gap_and_go_optimizer.py --days 365
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from termcolor import cprint
from dotenv import load_dotenv

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
load_dotenv()

# Backtesting imports
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    cprint("Warning: alpaca-trade-api not installed", "yellow")

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Paths
CSV_DIR = PROJECT_ROOT / 'csvs'
CACHE_DIR = CSV_DIR / 'cache'
RESULTS_PATH = CSV_DIR / 'gap_and_go_optimization_results.csv'
BEST_PARAMS_PATH = CSV_DIR / 'gap_and_go_best_params.json'

# Parameter ranges for optimization
PARAM_RANGES = {
    'gap_threshold': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    'volume_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0],
    'profit_target': [2, 3, 4, 5, 6, 7, 8],
    'stop_loss': [3, 5, 7, 10, 12, 15],
}

# Risk/reward constraint: profit_target >= 0.4 * stop_loss
MIN_REWARD_RISK_RATIO = 0.4


def fetch_daily_data(symbol='AMD', days=180):
    """Fetch daily OHLCV data from Alpaca with caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f'{symbol}_daily_{days}d.csv'

    # Check cache (valid for 1 day)
    if cache_file.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age.total_seconds() < 86400:  # 24 hours
            cprint(f"Loading {symbol} daily data from cache...", "cyan")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            cprint(f"Loaded {len(df)} bars from cache", "green")
            return df

    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-trade-api not installed")

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    # Fetch data from 15 days ago to avoid SIP restrictions
    end_date = datetime.now() - timedelta(days=15)
    start_date = end_date - timedelta(days=days + 30)  # Extra for volume average

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    cprint(f"Fetching {symbol} daily data from {start_str} to {end_str}...", "cyan")

    bars = api.get_bars(
        symbol,
        tradeapi.TimeFrame.Day,
        start=start_str,
        end=end_str,
        limit=days + 50,
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

    df = bars[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Save to cache
    df.to_csv(cache_file)
    cprint(f"Fetched and cached {len(df)} daily bars", "green")

    return df


def fetch_hourly_data(symbol='AMD', days=180):
    """Fetch hourly OHLCV data from Alpaca with caching."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f'{symbol}_hourly_{days}d.csv'

    # Check cache
    if cache_file.exists():
        cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if cache_age.total_seconds() < 86400:
            cprint(f"Loading {symbol} hourly data from cache...", "cyan")
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            cprint(f"Loaded {len(df)} bars from cache", "green")
            return df

    if not ALPACA_AVAILABLE:
        raise RuntimeError("alpaca-trade-api not installed")

    base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

    end_date = datetime.now() - timedelta(days=15)
    start_date = end_date - timedelta(days=days)

    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')

    cprint(f"Fetching {symbol} hourly data from {start_str} to {end_str}...", "cyan")

    bars = api.get_bars(
        symbol,
        tradeapi.TimeFrame.Hour,
        start=start_str,
        end=end_str,
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

    df = bars[['Open', 'High', 'Low', 'Close', 'Volume']]

    df.to_csv(cache_file)
    cprint(f"Fetched and cached {len(df)} hourly bars", "green")

    return df


class GapAndGoStrategy(Strategy):
    """
    Gap and Go Strategy for backtesting.py

    Entry:
    - Gap up >= gap_threshold% from previous close -> LONG
    - Gap down <= -gap_threshold% from previous close -> SHORT
    - Volume must be >= volume_multiplier * 20-day avg volume

    Exit:
    - Take profit at profit_target%
    - Stop loss at stop_loss%
    """

    # Parameters (will be optimized)
    gap_threshold = 2.0      # Gap percentage threshold
    volume_multiplier = 1.5  # Volume must be this multiple of average
    profit_target = 3.0      # Take profit percentage
    stop_loss = 2.0          # Stop loss percentage

    def init(self):
        """Initialize indicators."""
        # Calculate previous close (shift by 1)
        self.prev_close = self.I(lambda x: pd.Series(x).shift(1).values, self.data.Close)

        # Calculate gap percentage
        def calc_gap(open_prices, prev_close):
            gap = (pd.Series(open_prices) - pd.Series(prev_close)) / pd.Series(prev_close) * 100
            return gap.values

        self.gap_pct = self.I(calc_gap, self.data.Open, self.prev_close)

        # Calculate 20-day average volume
        self.avg_volume = self.I(
            lambda x: pd.Series(x).rolling(window=20).mean().values,
            self.data.Volume
        )

        # Volume ratio
        def calc_vol_ratio(volume, avg_vol):
            ratio = pd.Series(volume) / pd.Series(avg_vol)
            return ratio.fillna(1).values

        self.vol_ratio = self.I(calc_vol_ratio, self.data.Volume, self.avg_volume)

    def next(self):
        """Execute strategy logic."""
        # Skip if we already have a position
        if self.position:
            return

        # Get current values
        gap = self.gap_pct[-1]
        vol_ratio = self.vol_ratio[-1]
        current_price = self.data.Close[-1]

        # Skip if data is invalid
        if np.isnan(gap) or np.isnan(vol_ratio):
            return

        # Check volume condition
        if vol_ratio < self.volume_multiplier:
            return

        # Calculate SL and TP prices
        if gap >= self.gap_threshold:
            # Gap UP -> Go LONG
            sl_price = current_price * (1 - self.stop_loss / 100)
            tp_price = current_price * (1 + self.profit_target / 100)
            self.buy(sl=sl_price, tp=tp_price)

        elif gap <= -self.gap_threshold:
            # Gap DOWN -> Go SHORT
            sl_price = current_price * (1 + self.stop_loss / 100)
            tp_price = current_price * (1 - self.profit_target / 100)
            self.sell(sl=sl_price, tp=tp_price)


def generate_parameter_combinations():
    """Generate all valid parameter combinations."""
    combinations = []

    for gap, vol, tp, sl in product(
        PARAM_RANGES['gap_threshold'],
        PARAM_RANGES['volume_multiplier'],
        PARAM_RANGES['profit_target'],
        PARAM_RANGES['stop_loss']
    ):
        # Apply constraint: profit_target >= 0.4 * stop_loss
        if tp >= MIN_REWARD_RISK_RATIO * sl:
            combinations.append({
                'gap_threshold': gap,
                'volume_multiplier': vol,
                'profit_target': tp,
                'stop_loss': sl
            })

    return combinations


def run_single_backtest(data, params, cash=100000):
    """Run a single backtest with given parameters."""
    try:
        bt = Backtest(
            data,
            GapAndGoStrategy,
            cash=cash,
            commission=0.001,
            exclusive_orders=True,
            trade_on_close=True
        )

        stats = bt.run(
            gap_threshold=params['gap_threshold'],
            volume_multiplier=params['volume_multiplier'],
            profit_target=params['profit_target'],
            stop_loss=params['stop_loss']
        )

        return {
            'gap_threshold': params['gap_threshold'],
            'volume_multiplier': params['volume_multiplier'],
            'profit_target': params['profit_target'],
            'stop_loss': params['stop_loss'],
            'return_pct': stats['Return [%]'],
            'equity_final': stats['Equity Final [$]'],
            'win_rate': stats['Win Rate [%]'],
            'num_trades': stats['# Trades'],
            'max_drawdown': stats['Max. Drawdown [%]'],
            'sharpe_ratio': stats.get('Sharpe Ratio', 0),
            'profit_factor': stats.get('Profit Factor', 0),
            'avg_trade_pct': stats.get('Avg. Trade [%]', 0),
            'buy_hold_return': stats['Buy & Hold Return [%]'],
        }
    except Exception as e:
        return None


def run_optimization(symbol='AMD', days=180, cash=100000):
    """Run full grid optimization."""
    cprint("\n" + "=" * 80, "cyan")
    cprint("  GAP AND GO OPTIMIZER", "cyan", attrs=['bold'])
    cprint(f"  Symbol: {symbol} | Days: {days} | Cash: ${cash:,}", "cyan")
    cprint("=" * 80, "cyan")

    # Fetch data
    data = fetch_daily_data(symbol, days)

    if len(data) < 30:
        raise ValueError(f"Insufficient data: {len(data)} bars (need at least 30)")

    cprint(f"\nData range: {data.index[0].date()} to {data.index[-1].date()}", "white")
    cprint(f"Total bars: {len(data)}", "white")

    # Generate parameter combinations
    combinations = generate_parameter_combinations()
    total_combos = len(combinations)

    cprint(f"\nTesting {total_combos} parameter combinations...", "yellow")
    cprint(f"Constraint: profit_target >= {MIN_REWARD_RISK_RATIO} * stop_loss", "white")

    # Run all backtests
    results = []
    start_time = datetime.now()

    for i, params in enumerate(combinations):
        if (i + 1) % 50 == 0 or i == 0:
            progress = (i + 1) / total_combos * 100
            elapsed = (datetime.now() - start_time).total_seconds()
            eta = (elapsed / (i + 1)) * (total_combos - i - 1) if i > 0 else 0
            cprint(f"  [{progress:5.1f}%] Testing combo {i+1}/{total_combos} (ETA: {eta:.0f}s)", "white")

        result = run_single_backtest(data, params, cash)
        if result:
            results.append(result)

    elapsed_total = (datetime.now() - start_time).total_seconds()
    cprint(f"\nCompleted {len(results)} backtests in {elapsed_total:.1f}s", "green")

    return results, data


def analyze_results(results, symbol):
    """Analyze and display optimization results."""
    if not results:
        cprint("No valid results to analyze", "red")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Sort by equity final (descending)
    df = df.sort_values('equity_final', ascending=False)

    # Get top 5
    top5 = df.head(5)

    # Get best result
    best = df.iloc[0].to_dict()

    # Print results
    cprint("\n" + "=" * 80, "green")
    cprint("  OPTIMIZATION RESULTS", "green", attrs=['bold'])
    cprint("=" * 80, "green")

    cprint(f"\n  BEST PARAMETERS FOR {symbol}:", "yellow", attrs=['bold'])
    cprint(f"  ─────────────────────────────────────────────", "white")
    cprint(f"  Gap Threshold:      {best['gap_threshold']:.1f}%", "white")
    cprint(f"  Volume Multiplier:  {best['volume_multiplier']:.1f}x", "white")
    cprint(f"  Profit Target:      {best['profit_target']:.1f}%", "white")
    cprint(f"  Stop Loss:          {best['stop_loss']:.1f}%", "white")

    cprint(f"\n  PERFORMANCE METRICS:", "cyan", attrs=['bold'])
    cprint(f"  ─────────────────────────────────────────────", "white")

    ret_color = "green" if best['return_pct'] > 0 else "red"
    cprint(f"  Return:             {best['return_pct']:>+10.2f}%", ret_color)
    cprint(f"  Equity Final:       ${best['equity_final']:>10,.2f}", ret_color)
    cprint(f"  Buy & Hold Return:  {best['buy_hold_return']:>+10.2f}%", "white")
    cprint(f"  Win Rate:           {best['win_rate']:>10.1f}%", "green" if best['win_rate'] > 50 else "yellow")
    cprint(f"  Number of Trades:   {best['num_trades']:>10.0f}", "white")
    cprint(f"  Max Drawdown:       {best['max_drawdown']:>10.2f}%", "red" if best['max_drawdown'] < -20 else "yellow")
    cprint(f"  Sharpe Ratio:       {best['sharpe_ratio']:>10.2f}", "green" if best['sharpe_ratio'] > 1 else "white")
    cprint(f"  Profit Factor:      {best['profit_factor']:>10.2f}", "green" if best['profit_factor'] > 1.5 else "white")
    cprint(f"  Avg Trade:          {best['avg_trade_pct']:>+10.2f}%", "white")

    # Risk/Reward ratio
    rr_ratio = best['profit_target'] / best['stop_loss']
    cprint(f"  Risk/Reward Ratio:  {rr_ratio:>10.2f}", "green" if rr_ratio > 0.5 else "yellow")

    # Top 5 table
    cprint(f"\n  TOP 5 PARAMETER COMBINATIONS:", "magenta", attrs=['bold'])
    cprint(f"  ─────────────────────────────────────────────────────────────────────────────", "white")
    cprint(f"  {'Rank':<5} {'Gap%':<6} {'Vol':<5} {'TP%':<5} {'SL%':<5} {'Return':<10} {'WinRate':<8} {'Trades':<7} {'Sharpe':<7}", "white")
    cprint(f"  ─────────────────────────────────────────────────────────────────────────────", "white")

    for i, row in top5.iterrows():
        rank = top5.index.get_loc(i) + 1
        color = "green" if row['return_pct'] > 0 else "red"
        cprint(
            f"  {rank:<5} {row['gap_threshold']:<6.1f} {row['volume_multiplier']:<5.1f} "
            f"{row['profit_target']:<5.0f} {row['stop_loss']:<5.0f} "
            f"{row['return_pct']:>+8.2f}% {row['win_rate']:>6.1f}% "
            f"{row['num_trades']:>6.0f} {row['sharpe_ratio']:>6.2f}",
            color
        )

    cprint(f"  ─────────────────────────────────────────────────────────────────────────────", "white")

    # Statistics summary
    profitable = len(df[df['return_pct'] > 0])
    avg_return = df['return_pct'].mean()

    cprint(f"\n  OPTIMIZATION STATISTICS:", "white")
    cprint(f"  Total combinations tested: {len(df)}", "white")
    cprint(f"  Profitable combinations:   {profitable} ({profitable/len(df)*100:.1f}%)", "green" if profitable > len(df)/2 else "yellow")
    cprint(f"  Average return:            {avg_return:+.2f}%", "green" if avg_return > 0 else "red")

    cprint("=" * 80 + "\n", "green")

    return best, df


def save_results(results, best, symbol):
    """Save results to CSV and JSON."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Save all results to CSV
    df = pd.DataFrame(results)
    df = df.sort_values('equity_final', ascending=False)
    df.to_csv(RESULTS_PATH, index=False)
    cprint(f"Results saved to: {RESULTS_PATH}", "green")

    # Save best params to JSON
    best_data = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'parameters': {
            'gap_threshold': best['gap_threshold'],
            'volume_multiplier': best['volume_multiplier'],
            'profit_target': best['profit_target'],
            'stop_loss': best['stop_loss'],
        },
        'performance': {
            'return_pct': best['return_pct'],
            'equity_final': best['equity_final'],
            'win_rate': best['win_rate'],
            'num_trades': best['num_trades'],
            'max_drawdown': best['max_drawdown'],
            'sharpe_ratio': best['sharpe_ratio'],
            'profit_factor': best['profit_factor'],
        }
    }

    with open(BEST_PARAMS_PATH, 'w') as f:
        json.dump(best_data, f, indent=2)
    cprint(f"Best params saved to: {BEST_PARAMS_PATH}", "green")

    # Print copyable dict for smart_bot.py
    cprint("\n  COPY THIS TO smart_bot.py / gap_and_go_bot.py:", "yellow")
    cprint("  " + "─" * 50, "white")
    cprint(f"""
# Optimized Gap and Go Parameters for {symbol}
GAP_MIN_PCT = {best['gap_threshold']}
GAP_VOLUME_MULT = {best['volume_multiplier']}
GAP_TP_PCT = {best['profit_target']}
GAP_SL_PCT = {best['stop_loss']}
""", "cyan")


def print_heatmap(results):
    """Print a simple text-based heatmap of results."""
    df = pd.DataFrame(results)

    if df.empty:
        return

    cprint("\n  GAP THRESHOLD vs PROFIT TARGET HEATMAP (avg return %)", "magenta", attrs=['bold'])
    cprint("  " + "─" * 60, "white")

    # Pivot for gap_threshold vs profit_target
    pivot = df.pivot_table(
        values='return_pct',
        index='gap_threshold',
        columns='profit_target',
        aggfunc='mean'
    )

    # Print header
    header = "  Gap\\TP |"
    for col in pivot.columns:
        header += f" {col:>6.0f}%"
    cprint(header, "white")
    cprint("  " + "─" * 60, "white")

    # Print rows
    for idx in pivot.index:
        row = f"  {idx:>5.1f}% |"
        for col in pivot.columns:
            val = pivot.loc[idx, col]
            if pd.isna(val):
                row += "    N/A"
            else:
                if val > 5:
                    color = "green"
                elif val > 0:
                    color = "yellow"
                else:
                    color = "red"
                row += f" {val:>+6.1f}"
        cprint(row, "white")

    cprint("  " + "─" * 60, "white")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Gap and Go Optimizer")
    parser.add_argument('--symbol', type=str, default='AMD', help='Symbol to optimize')
    parser.add_argument('--days', type=int, default=180, help='Days of history')
    parser.add_argument('--cash', type=int, default=100000, help='Starting cash')
    args = parser.parse_args()

    try:
        # Run optimization
        results, data = run_optimization(
            symbol=args.symbol,
            days=args.days,
            cash=args.cash
        )

        # Analyze results
        best, df = analyze_results(results, args.symbol)

        if best:
            # Print heatmap
            print_heatmap(results)

            # Save results
            save_results(results, best, args.symbol)

    except Exception as e:
        cprint(f"\nError: {e}", "red")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

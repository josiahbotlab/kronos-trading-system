"""
Batch Backtester

Runs all trading strategies on multiple symbols and generates a comparison report.

Strategies tested:
1. 24h Breakout - Momentum breakout strategy
2. Mean Reversion - RSI oversold + SMA deviation
3. Bollinger Breakout - Upper band breakout with Heikin Ashi
4. Gap and Go - Morning gap momentum

Output:
- Comparison table showing performance of each strategy per symbol
- Auto-generated SYMBOL_STRATEGY_OVERRIDE dict for smart_bot.py
- CSV export of all results

Usage:
    python src/backtesting/batch_backtest.py
    python src/backtesting/batch_backtest.py --symbols TSLA AAPL MSFT
    python src/backtesting/batch_backtest.py --days 180
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

from termcolor import cprint

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import backtesters
from src.backtesting.backtest_breakout import run_backtest as run_breakout
from src.backtesting.backtest_mean_reversion import run_backtest as run_mean_reversion
from src.backtesting.backtest_bollinger_breakout import run_backtest as run_bollinger
from src.backtesting.backtest_gap_and_go import run_backtest as run_gap_and_go

# Paths
CSV_DIR = PROJECT_ROOT / 'csvs'
RESULTS_PATH = CSV_DIR / 'batch_backtest_results.csv'

# Default symbols (top traded stocks)
DEFAULT_SYMBOLS = [
    'TSLA', 'AMD', 'NVDA', 'AAPL', 'SPY',
    'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX'
]

# Extended list for comprehensive testing
TOP_20_SYMBOLS = [
    'TSLA', 'AMD', 'NVDA', 'AAPL', 'SPY',
    'MSFT', 'GOOGL', 'META', 'AMZN', 'NFLX',
    'QQQ', 'BA', 'DIS', 'INTC', 'JPM',
    'V', 'WMT', 'JNJ', 'PG', 'KO'
]

# Strategy configurations
STRATEGIES = {
    'BREAKOUT': {
        'name': '24h Breakout',
        'runner': run_breakout,
        'color': 'magenta',
        'smart_bot_key': 'BREAKOUT'
    },
    'MEAN_REV': {
        'name': 'Mean Reversion',
        'runner': run_mean_reversion,
        'color': 'red',
        'smart_bot_key': 'MEAN_REV'
    },
    'BOLLINGER': {
        'name': 'Bollinger Breakout',
        'runner': run_bollinger,
        'color': 'blue',
        'smart_bot_key': 'BOLLINGER'
    },
    'GAP_GO': {
        'name': 'Gap and Go',
        'runner': run_gap_and_go,
        'color': 'yellow',
        'smart_bot_key': 'GAP_AND_GO'
    }
}


def run_single_backtest(symbol, strategy_key, days=365, cash=100000):
    """Run a single backtest and return results."""
    strategy = STRATEGIES[strategy_key]

    try:
        result = strategy['runner'](symbol=symbol, days=days, cash=cash)

        if result is None:
            return None

        return {
            'symbol': symbol,
            'strategy': strategy_key,
            'strategy_name': strategy['name'],
            'return_pct': result.get('total_return_pct', 0),
            'win_rate': result.get('win_rate_pct', 0),
            'profit_factor': result.get('profit_factor', 0),
            'num_trades': result.get('num_trades', 0),
            'max_drawdown': result.get('max_drawdown_pct', 0),
            'final_equity': result.get('final_equity', cash),
        }
    except Exception as e:
        cprint(f"  Error running {strategy['name']} on {symbol}: {e}", "red")
        return None


def run_batch_backtest(symbols, days=365, cash=100000):
    """Run all strategies on all symbols."""
    all_results = []
    symbol_best = {}  # Track best strategy per symbol

    total_tests = len(symbols) * len(STRATEGIES)
    completed = 0

    cprint("\n" + "=" * 80, "cyan")
    cprint("  BATCH BACKTESTER", "cyan", attrs=['bold'])
    cprint(f"  Symbols: {len(symbols)} | Strategies: {len(STRATEGIES)} | Days: {days}", "cyan")
    cprint("=" * 80, "cyan")

    for symbol in symbols:
        cprint(f"\n{'─' * 80}", "white")
        cprint(f"  Testing {symbol}...", "white", attrs=['bold'])
        cprint(f"{'─' * 80}", "white")

        symbol_results = []

        for strategy_key, strategy_info in STRATEGIES.items():
            completed += 1
            progress = (completed / total_tests) * 100
            cprint(f"\n  [{progress:5.1f}%] {strategy_info['name']}...", strategy_info['color'])

            result = run_single_backtest(symbol, strategy_key, days, cash)

            if result:
                symbol_results.append(result)
                all_results.append(result)

                # Print quick summary
                ret = result['return_pct']
                wr = result['win_rate']
                trades = result['num_trades']
                color = "green" if ret > 0 else "red"
                cprint(f"    Return: {ret:+.2f}% | Win Rate: {wr:.1f}% | Trades: {trades}", color)

            # Small delay to avoid API rate limits
            time.sleep(0.5)

        # Find best strategy for this symbol
        if symbol_results:
            # Sort by return, then by win rate
            best = max(symbol_results, key=lambda x: (x['return_pct'], x['win_rate']))
            symbol_best[symbol] = best

            cprint(f"\n  Best for {symbol}: {best['strategy_name']} ({best['return_pct']:+.2f}%)", "green")

    return all_results, symbol_best


def print_comparison_table(all_results, symbols):
    """Print a formatted comparison table."""
    cprint("\n" + "=" * 100, "cyan")
    cprint("  STRATEGY COMPARISON TABLE", "cyan", attrs=['bold'])
    cprint("=" * 100, "cyan")

    # Header
    header = f"{'Symbol':<8}"
    for strat_key in STRATEGIES.keys():
        header += f" {strat_key:>12}"
    header += f" {'BEST':>14}"

    cprint(f"\n{header}", "white", attrs=['bold'])
    cprint("-" * 100, "white")

    # Build results by symbol
    results_by_symbol = {}
    for r in all_results:
        sym = r['symbol']
        if sym not in results_by_symbol:
            results_by_symbol[sym] = {}
        results_by_symbol[sym][r['strategy']] = r

    # Print each symbol row
    for symbol in symbols:
        if symbol not in results_by_symbol:
            continue

        row = f"{symbol:<8}"
        best_return = -999
        best_strat = None

        for strat_key in STRATEGIES.keys():
            if strat_key in results_by_symbol[symbol]:
                ret = results_by_symbol[symbol][strat_key]['return_pct']
                row += f" {ret:>+11.2f}%"
                if ret > best_return:
                    best_return = ret
                    best_strat = strat_key
            else:
                row += f" {'N/A':>12}"

        # Add best column
        if best_strat:
            row += f"  {best_strat:<12}"
            color = "green" if best_return > 0 else "red"
        else:
            row += f"  {'N/A':<12}"
            color = "white"

        cprint(row, color)

    cprint("-" * 100, "white")


def print_strategy_summary(all_results):
    """Print summary statistics by strategy."""
    cprint("\n" + "=" * 80, "yellow")
    cprint("  STRATEGY SUMMARY", "yellow", attrs=['bold'])
    cprint("=" * 80, "yellow")

    for strat_key, strat_info in STRATEGIES.items():
        strat_results = [r for r in all_results if r['strategy'] == strat_key]

        if not strat_results:
            continue

        returns = [r['return_pct'] for r in strat_results]
        win_rates = [r['win_rate'] for r in strat_results]

        avg_return = sum(returns) / len(returns)
        avg_win_rate = sum(win_rates) / len(win_rates)
        winners = len([r for r in returns if r > 0])

        color = strat_info['color']
        cprint(f"\n  {strat_info['name']}", color, attrs=['bold'])
        cprint(f"    Avg Return:   {avg_return:>+8.2f}%", "green" if avg_return > 0 else "red")
        cprint(f"    Avg Win Rate: {avg_win_rate:>8.1f}%", "white")
        cprint(f"    Profitable:   {winners}/{len(strat_results)} symbols", "white")


def generate_override_dict(symbol_best):
    """Generate SYMBOL_STRATEGY_OVERRIDE dict for smart_bot.py."""
    cprint("\n" + "=" * 80, "green")
    cprint("  AUTO-GENERATED SYMBOL_STRATEGY_OVERRIDE", "green", attrs=['bold'])
    cprint("  Copy this into smart_bot.py", "green")
    cprint("=" * 80, "green")

    # Determine strategy mapping
    lines = ["\nSYMBOL_STRATEGY_OVERRIDE = {"]

    for symbol, result in sorted(symbol_best.items()):
        ret = result['return_pct']
        strat = result['strategy']
        strat_name = result['strategy_name']

        # Map to smart_bot.py strategy keys
        if ret < 0:
            # If best strategy still loses money, recommend HOLD
            smart_key = 'HOLD'
            comment = f"best was {strat_name} ({ret:+.2f}%) - buy & hold likely better"
        else:
            smart_key = STRATEGIES[strat]['smart_bot_key']
            comment = f"{ret:+.2f}% backtested"

        lines.append(f"    '{symbol}': '{smart_key}',  # {comment}")

    lines.append("}")

    code = "\n".join(lines)
    cprint(code, "white")

    return code


def save_results(all_results, symbol_best, output_path):
    """Save results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(output_path, 'w', newline='') as f:
        fieldnames = [
            'symbol', 'strategy', 'strategy_name', 'return_pct',
            'win_rate', 'profit_factor', 'num_trades', 'max_drawdown'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in all_results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    cprint(f"\nResults saved to: {output_path}", "green")

    # Also save the override dict to a text file
    override_path = output_path.parent / 'symbol_strategy_override.txt'
    override_code = generate_override_dict(symbol_best)

    with open(override_path, 'w') as f:
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Symbols tested: {len(symbol_best)}\n\n")
        f.write(override_code)

    cprint(f"Override dict saved to: {override_path}", "green")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch Backtester")
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=DEFAULT_SYMBOLS,
        help='Symbols to test (default: top 10 traded)'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='Days of history (default: 365)'
    )
    parser.add_argument(
        '--cash',
        type=int,
        default=100000,
        help='Starting cash (default: 100000)'
    )
    parser.add_argument(
        '--top20',
        action='store_true',
        help='Use top 20 symbols instead of default 10'
    )
    args = parser.parse_args()

    symbols = TOP_20_SYMBOLS if args.top20 else args.symbols

    start_time = datetime.now()
    cprint(f"\nStarting batch backtest at {start_time.strftime('%H:%M:%S')}", "cyan")

    # Run all backtests
    all_results, symbol_best = run_batch_backtest(
        symbols=symbols,
        days=args.days,
        cash=args.cash
    )

    # Print comparison table
    print_comparison_table(all_results, symbols)

    # Print strategy summary
    print_strategy_summary(all_results)

    # Generate override dict
    generate_override_dict(symbol_best)

    # Save results
    save_results(all_results, symbol_best, RESULTS_PATH)

    # Print timing
    elapsed = (datetime.now() - start_time).total_seconds()
    cprint(f"\nTotal time: {elapsed/60:.1f} minutes", "cyan")
    cprint("=" * 80 + "\n", "cyan")


if __name__ == "__main__":
    main()

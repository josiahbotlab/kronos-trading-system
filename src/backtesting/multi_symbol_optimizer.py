"""
Multi-Symbol Gap and Go Optimizer

Runs full parameter optimization on multiple symbols to find
which have profitable Gap and Go edges.

Quality Filters for "Tradeable" symbols:
- Return > 0%
- Number of trades >= 5
- Beats Buy & Hold by at least 10%
- Sharpe Ratio > 0

Output:
- csvs/all_symbols_best_params.csv - Best params for ALL symbols
- csvs/tradeable_symbols.json - Only symbols passing filters
- csvs/symbol_optimization_summary.csv - Full summary

Usage:
    python src/backtesting/multi_symbol_optimizer.py
    python src/backtesting/multi_symbol_optimizer.py --symbols AAPL MSFT GOOGL
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
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

# Import from gap_and_go_optimizer
from src.backtesting.gap_and_go_optimizer import (
    fetch_daily_data,
    GapAndGoStrategy,
    run_single_backtest,
)
from backtesting import Backtest

# Paths
CSV_DIR = PROJECT_ROOT / 'csvs'
ALL_PARAMS_PATH = CSV_DIR / 'all_symbols_best_params.csv'
TRADEABLE_PATH = CSV_DIR / 'tradeable_symbols.json'
SUMMARY_PATH = CSV_DIR / 'symbol_optimization_summary.csv'

# Symbols to optimize
DEFAULT_SYMBOLS = [
    'NVDA', 'TSLA', 'META', 'GOOGL', 'MSFT',
    'AAPL', 'QQQ', 'SPY', 'AMZN'
]

# Parameter ranges (slightly reduced for faster optimization)
PARAM_RANGES = {
    'gap_threshold': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    'volume_multiplier': [1.0, 1.5, 2.0],
    'profit_target': [3, 4, 5, 6, 7, 8],
    'stop_loss': [5, 7, 10, 12, 15],
}

# Quality filters
MIN_RETURN_PCT = 0
MIN_TRADES = 5
MIN_OUTPERFORMANCE = 10  # Must beat B&H by this %
MIN_SHARPE = 0


def generate_combinations():
    """Generate all valid parameter combinations."""
    combinations = []

    for gap, vol, tp, sl in product(
        PARAM_RANGES['gap_threshold'],
        PARAM_RANGES['volume_multiplier'],
        PARAM_RANGES['profit_target'],
        PARAM_RANGES['stop_loss']
    ):
        # Constraint: TP must be reasonable relative to SL
        if tp >= 0.4 * sl:
            combinations.append({
                'gap_threshold': gap,
                'volume_multiplier': vol,
                'profit_target': tp,
                'stop_loss': sl
            })

    return combinations


def optimize_symbol(symbol, days=180, cash=100000):
    """
    Run full optimization for a single symbol.
    Returns best parameters and all results.
    """
    cprint(f"\n{'═' * 70}", "cyan")
    cprint(f"  OPTIMIZING: {symbol}", "cyan", attrs=['bold'])
    cprint(f"{'═' * 70}", "cyan")

    try:
        # Fetch data
        data = fetch_daily_data(symbol, days)

        if len(data) < 30:
            cprint(f"  Insufficient data: {len(data)} bars", "red")
            return None, []

        cprint(f"  Data: {data.index[0].date()} to {data.index[-1].date()} ({len(data)} bars)", "white")

    except Exception as e:
        cprint(f"  Error fetching data: {e}", "red")
        return None, []

    # Generate combinations
    combinations = generate_combinations()
    total = len(combinations)
    cprint(f"  Testing {total} parameter combinations...", "yellow")

    # Run all backtests
    results = []
    start_time = time.time()

    for i, params in enumerate(combinations):
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            eta = (elapsed / (i + 1)) * (total - i - 1)
            cprint(f"    [{(i+1)/total*100:5.1f}%] {i+1}/{total} (ETA: {eta:.0f}s)", "white", end='\r')

        result = run_single_backtest(data, params, cash)
        if result:
            results.append(result)

    print()  # New line after progress

    if not results:
        cprint(f"  No valid results for {symbol}", "red")
        return None, []

    elapsed = time.time() - start_time
    cprint(f"  Completed {len(results)} backtests in {elapsed:.1f}s", "green")

    # Find best result (by return, then sharpe)
    df = pd.DataFrame(results)
    df = df.sort_values(['return_pct', 'sharpe_ratio'], ascending=[False, False])
    best = df.iloc[0].to_dict()

    # Print best result
    color = "green" if best['return_pct'] > 0 else "red"
    cprint(f"\n  BEST PARAMETERS for {symbol}:", "yellow")
    cprint(f"    Gap: {best['gap_threshold']:.1f}% | Vol: {best['volume_multiplier']:.1f}x | TP: {best['profit_target']:.0f}% | SL: {best['stop_loss']:.0f}%", "white")
    cprint(f"    Return: {best['return_pct']:+.2f}% | Win Rate: {best['win_rate']:.1f}% | Trades: {best['num_trades']:.0f}", color)
    cprint(f"    Sharpe: {best['sharpe_ratio']:.2f} | Max DD: {best['max_drawdown']:.2f}%", "white")
    cprint(f"    Buy & Hold: {best['buy_hold_return']:+.2f}% | Outperformance: {best['return_pct'] - best['buy_hold_return']:+.2f}%", "white")

    return best, results


def check_quality_filters(result):
    """
    Check if a result passes quality filters.
    Returns (passed, reasons) tuple.
    """
    reasons = []
    passed = True

    if result['return_pct'] <= MIN_RETURN_PCT:
        reasons.append(f"Return {result['return_pct']:.2f}% <= {MIN_RETURN_PCT}%")
        passed = False

    if result['num_trades'] < MIN_TRADES:
        reasons.append(f"Trades {result['num_trades']:.0f} < {MIN_TRADES}")
        passed = False

    outperformance = result['return_pct'] - result['buy_hold_return']
    if outperformance < MIN_OUTPERFORMANCE:
        reasons.append(f"Outperformance {outperformance:+.2f}% < {MIN_OUTPERFORMANCE}%")
        passed = False

    if result['sharpe_ratio'] <= MIN_SHARPE:
        reasons.append(f"Sharpe {result['sharpe_ratio']:.2f} <= {MIN_SHARPE}")
        passed = False

    return passed, reasons


def run_multi_optimization(symbols, days=180, cash=100000):
    """Run optimization on all symbols."""
    cprint("\n" + "=" * 80, "magenta")
    cprint("  MULTI-SYMBOL GAP AND GO OPTIMIZER", "magenta", attrs=['bold'])
    cprint(f"  Symbols: {len(symbols)} | Days: {days} | Starting Cash: ${cash:,}", "magenta")
    cprint("=" * 80, "magenta")

    cprint(f"\n  Parameter Ranges:", "white")
    cprint(f"    Gap Threshold:     {PARAM_RANGES['gap_threshold']}", "white")
    cprint(f"    Volume Multiplier: {PARAM_RANGES['volume_multiplier']}", "white")
    cprint(f"    Profit Target:     {PARAM_RANGES['profit_target']}", "white")
    cprint(f"    Stop Loss:         {PARAM_RANGES['stop_loss']}", "white")

    cprint(f"\n  Quality Filters:", "yellow")
    cprint(f"    Min Return:        > {MIN_RETURN_PCT}%", "white")
    cprint(f"    Min Trades:        >= {MIN_TRADES}", "white")
    cprint(f"    Min Outperformance: >= {MIN_OUTPERFORMANCE}%", "white")
    cprint(f"    Min Sharpe:        > {MIN_SHARPE}", "white")

    # Results storage
    all_best = []
    tradeable = {}
    non_tradeable = {}

    total_start = time.time()

    for i, symbol in enumerate(symbols):
        cprint(f"\n  Progress: {i+1}/{len(symbols)} symbols", "cyan")

        best, results = optimize_symbol(symbol, days, cash)

        if best:
            best['symbol'] = symbol
            all_best.append(best)

            # Check quality filters
            passed, reasons = check_quality_filters(best)

            if passed:
                tradeable[symbol] = {
                    'parameters': {
                        'gap_threshold': best['gap_threshold'],
                        'volume_multiplier': best['volume_multiplier'],
                        'profit_target': best['profit_target'],
                        'stop_loss': best['stop_loss'],
                    },
                    'performance': {
                        'return_pct': best['return_pct'],
                        'win_rate': best['win_rate'],
                        'num_trades': best['num_trades'],
                        'max_drawdown': best['max_drawdown'],
                        'sharpe_ratio': best['sharpe_ratio'],
                        'buy_hold_return': best['buy_hold_return'],
                        'outperformance': best['return_pct'] - best['buy_hold_return'],
                    }
                }
                cprint(f"  ✓ {symbol} PASSED quality filters", "green")
            else:
                non_tradeable[symbol] = {
                    'best_return': best['return_pct'],
                    'reasons': reasons
                }
                cprint(f"  ✗ {symbol} FAILED: {'; '.join(reasons)}", "red")
        else:
            non_tradeable[symbol] = {
                'best_return': 0,
                'reasons': ['No valid backtest results']
            }

    total_elapsed = time.time() - total_start

    return all_best, tradeable, non_tradeable, total_elapsed


def print_final_summary(all_best, tradeable, non_tradeable):
    """Print comprehensive final summary."""
    cprint("\n" + "=" * 90, "green")
    cprint("  FINAL OPTIMIZATION SUMMARY", "green", attrs=['bold'])
    cprint("=" * 90, "green")

    # Tradeable symbols
    cprint(f"\n  TRADEABLE SYMBOLS ({len(tradeable)}):", "green", attrs=['bold'])
    cprint("  " + "─" * 85, "white")

    if tradeable:
        cprint(f"  {'Symbol':<8} {'Gap%':<6} {'Vol':<5} {'TP%':<5} {'SL%':<5} {'Return':<10} {'WinRate':<8} {'Trades':<7} {'Sharpe':<7} {'vs B&H':<10}", "white")
        cprint("  " + "─" * 85, "white")

        for symbol, data in sorted(tradeable.items(), key=lambda x: x[1]['performance']['return_pct'], reverse=True):
            p = data['parameters']
            perf = data['performance']
            cprint(
                f"  {symbol:<8} {p['gap_threshold']:<6.1f} {p['volume_multiplier']:<5.1f} "
                f"{p['profit_target']:<5.0f} {p['stop_loss']:<5.0f} "
                f"{perf['return_pct']:>+8.2f}% {perf['win_rate']:>6.1f}% "
                f"{perf['num_trades']:>6.0f} {perf['sharpe_ratio']:>6.2f} "
                f"{perf['outperformance']:>+8.2f}%",
                "green"
            )
    else:
        cprint("  No symbols passed quality filters", "yellow")

    cprint("  " + "─" * 85, "white")

    # Non-tradeable symbols
    cprint(f"\n  NON-TRADEABLE SYMBOLS ({len(non_tradeable)}):", "red", attrs=['bold'])
    cprint("  " + "─" * 85, "white")

    for symbol, data in non_tradeable.items():
        reasons_str = "; ".join(data['reasons'])
        cprint(f"  {symbol:<8} Best Return: {data['best_return']:>+8.2f}% | Failed: {reasons_str}", "red")

    cprint("  " + "─" * 85, "white")

    # Overall summary
    cprint(f"\n  SUMMARY:", "cyan", attrs=['bold'])
    cprint(f"  ─────────────────────────────────────────────", "white")
    cprint(f"  Total symbols tested:    {len(tradeable) + len(non_tradeable)}", "white")
    cprint(f"  Tradeable:               {len(tradeable)} ({len(tradeable)/(len(tradeable)+len(non_tradeable))*100:.0f}%)", "green")
    cprint(f"  Non-tradeable:           {len(non_tradeable)} ({len(non_tradeable)/(len(tradeable)+len(non_tradeable))*100:.0f}%)", "red")

    if tradeable:
        avg_return = sum(t['performance']['return_pct'] for t in tradeable.values()) / len(tradeable)
        avg_sharpe = sum(t['performance']['sharpe_ratio'] for t in tradeable.values()) / len(tradeable)
        cprint(f"  Avg return (tradeable):  {avg_return:+.2f}%", "green")
        cprint(f"  Avg Sharpe (tradeable):  {avg_sharpe:.2f}", "green")

    cprint("=" * 90 + "\n", "green")


def save_results(all_best, tradeable, non_tradeable):
    """Save all results to files."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Save all best params
    if all_best:
        df = pd.DataFrame(all_best)
        cols = ['symbol', 'gap_threshold', 'volume_multiplier', 'profit_target', 'stop_loss',
                'return_pct', 'win_rate', 'num_trades', 'max_drawdown', 'sharpe_ratio',
                'buy_hold_return', 'profit_factor']
        df = df[[c for c in cols if c in df.columns]]
        df = df.sort_values('return_pct', ascending=False)
        df.to_csv(ALL_PARAMS_PATH, index=False)
        cprint(f"  Saved: {ALL_PARAMS_PATH}", "green")

    # Save tradeable symbols
    tradeable_data = {
        'generated': datetime.now().isoformat(),
        'quality_filters': {
            'min_return_pct': MIN_RETURN_PCT,
            'min_trades': MIN_TRADES,
            'min_outperformance': MIN_OUTPERFORMANCE,
            'min_sharpe': MIN_SHARPE,
        },
        'symbols': tradeable
    }

    with open(TRADEABLE_PATH, 'w') as f:
        json.dump(tradeable_data, f, indent=2)
    cprint(f"  Saved: {TRADEABLE_PATH}", "green")

    # Save summary
    summary_data = []
    for result in all_best:
        passed, reasons = check_quality_filters(result)
        summary_data.append({
            'symbol': result['symbol'],
            'tradeable': passed,
            'gap_threshold': result['gap_threshold'],
            'volume_multiplier': result['volume_multiplier'],
            'profit_target': result['profit_target'],
            'stop_loss': result['stop_loss'],
            'return_pct': result['return_pct'],
            'win_rate': result['win_rate'],
            'num_trades': result['num_trades'],
            'max_drawdown': result['max_drawdown'],
            'sharpe_ratio': result['sharpe_ratio'],
            'buy_hold_return': result['buy_hold_return'],
            'outperformance': result['return_pct'] - result['buy_hold_return'],
            'failed_reasons': '; '.join(reasons) if not passed else '',
        })

    df = pd.DataFrame(summary_data)
    df = df.sort_values('return_pct', ascending=False)
    df.to_csv(SUMMARY_PATH, index=False)
    cprint(f"  Saved: {SUMMARY_PATH}", "green")

    # Print copyable config
    if tradeable:
        cprint("\n  COPY THIS TO smart_bot.py:", "yellow", attrs=['bold'])
        cprint("  " + "─" * 50, "white")

        lines = ["\n# Optimized Gap and Go Parameters by Symbol"]
        lines.append("GAP_GO_SYMBOL_PARAMS = {")

        for symbol, data in sorted(tradeable.items()):
            p = data['parameters']
            perf = data['performance']
            lines.append(f"    '{symbol}': {{'gap': {p['gap_threshold']}, 'vol': {p['volume_multiplier']}, 'tp': {p['profit_target']}, 'sl': {p['stop_loss']}}},  # {perf['return_pct']:+.1f}%")

        lines.append("}")

        code = "\n".join(lines)
        cprint(code, "cyan")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Symbol Gap and Go Optimizer")
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS, help='Symbols to optimize')
    parser.add_argument('--days', type=int, default=180, help='Days of history')
    parser.add_argument('--cash', type=int, default=100000, help='Starting cash')
    args = parser.parse_args()

    start_time = datetime.now()
    cprint(f"\nStarting multi-symbol optimization at {start_time.strftime('%H:%M:%S')}", "cyan")

    # Run optimization
    all_best, tradeable, non_tradeable, elapsed = run_multi_optimization(
        symbols=args.symbols,
        days=args.days,
        cash=args.cash
    )

    # Print summary
    print_final_summary(all_best, tradeable, non_tradeable)

    # Save results
    save_results(all_best, tradeable, non_tradeable)

    cprint(f"\nTotal optimization time: {elapsed/60:.1f} minutes", "cyan")
    cprint("=" * 90 + "\n", "cyan")


if __name__ == "__main__":
    main()

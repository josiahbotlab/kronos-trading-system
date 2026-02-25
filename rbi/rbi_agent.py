#!/usr/bin/env python3
"""
RBI Agent v2 — Research, Backtest, Implement Pipeline

Based on MoonDev's methodology for systematic strategy validation.

Pipeline:
  1. Parse strategy ideas from ideas.txt (YAML format)
  2. Fetch 2 years of historical data via yfinance
  3. Grid-search TP/SL parameters across multiple symbols
  4. Validate against strict criteria on 60%+ of symbols
  5. Output winners to winners.csv with optimal parameters

Validation Criteria (per symbol):
  Strict (default):
    - Minimum 10 trades
    - Win rate >= 50%
    - Profit factor >= 1.2
    - Must beat buy & hold return

  Lenient (--lenient):
    - Minimum 10 trades
    - Win rate >= 50%
    - Profit factor >= 1.2
    - Return > 0% (flagged if doesn't beat B&H)

Overall:
  - Must pass on 60%+ of test symbols

Three-tier output:
  STRONG PASS = all criteria met including beat B&H
  PASS        = profitable + good metrics, but didn't beat B&H
  FAIL        = unprofitable or bad metrics

Usage:
    python3 rbi_agent.py                          # Test new/testing ideas
    python3 rbi_agent.py --idea momentum          # Test specific idea by name
    python3 rbi_agent.py --all                    # Re-test everything
    python3 rbi_agent.py --all --lenient          # Lenient B&H validation
    python3 rbi_agent.py --symbols TSLA,AMD,NVDA  # Override test symbols
    python3 rbi_agent.py -v                       # Verbose output
"""

import argparse
import csv
import logging
import re
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest

# Suppress noisy warnings from yfinance/backtesting
warnings.filterwarnings('ignore', category=FutureWarning)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
RBI_DIR = Path(__file__).parent
PROJECT_ROOT = RBI_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

IDEAS_FILE = RBI_DIR / "ideas.txt"
WINNERS_FILE = RBI_DIR / "winners.csv"
RESULTS_DIR = RBI_DIR / "backtest_results"
OPT_DIR = RBI_DIR / "optimization_results"
LOGS_DIR = RBI_DIR / "logs"

for d in [RESULTS_DIR, OPT_DIR, LOGS_DIR]:
    d.mkdir(exist_ok=True)

# Import strategy classes
from rbi.backtest_templates.strategies import STRATEGY_REGISTRY

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = ['TSLA', 'AMD', 'NVDA', 'AAPL', 'MSFT', 'META', 'QQQ', 'GOOGL', 'AMZN']
BACKTEST_YEARS = 2
BACKTEST_CASH = 10_000
BACKTEST_COMMISSION = 0.001  # 0.1% per trade

# Validation thresholds
MIN_TRADES = 10
MIN_WIN_RATE = 50.0          # percent
MIN_PROFIT_FACTOR = 1.2
MIN_SYMBOL_PASS_RATE = 0.60  # 60% of symbols must pass

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
_data_cache: dict[str, pd.DataFrame] = {}


def setup_logging(verbose: bool) -> logging.Logger:
    log = logging.getLogger("rbi")
    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    log_file = LOGS_DIR / f"rbi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(ch)

    log.debug(f"Log file: {log_file}")
    return log


# ─────────────────────────────────────────────────────────────
# Ideas parser (YAML-like format)
# ─────────────────────────────────────────────────────────────

def parse_ideas(filepath: Path) -> list[dict]:
    """
    Parse ideas.txt with --- delimited blocks.
    Each block has key: value lines. Lists like [4, 6, 8] are parsed as floats.
    """
    text = filepath.read_text()
    ideas = []

    chunks = re.split(r'^---\s*$', text, flags=re.MULTILINE)

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        idea = {}
        for line in chunk.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            m = re.match(r'^(\w[\w_]*):\s*(.+)$', line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()

                # Parse list values like [4, 6, 8, 10]
                if val.startswith('[') and val.endswith(']'):
                    try:
                        items = val[1:-1].split(',')
                        val = [float(x.strip()) for x in items if x.strip()]
                    except ValueError:
                        pass

                idea[key] = val

        if 'name' in idea:
            ideas.append(idea)

    return ideas


def update_idea_status(filepath: Path, idea_name: str, new_status: str):
    """Update the status field for a specific idea in ideas.txt."""
    lines = filepath.read_text().splitlines()
    in_target = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('name:'):
            name_val = stripped.split(':', 1)[1].strip()
            in_target = (name_val == idea_name)

        if in_target and stripped.startswith('status:'):
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}status: {new_status}"
            break

    filepath.write_text('\n'.join(lines) + '\n')


# ─────────────────────────────────────────────────────────────
# Data fetching (with per-run cache)
# ─────────────────────────────────────────────────────────────

def fetch_data(symbol: str, years: int = BACKTEST_YEARS, log: logging.Logger = None) -> pd.DataFrame | None:
    """Fetch daily OHLCV data from yfinance. Cached per run."""
    cache_key = f"{symbol}_{years}"
    if cache_key in _data_cache:
        return _data_cache[cache_key]

    end = datetime.now()
    start = end - timedelta(days=years * 365)

    if log:
        log.debug(f"  Fetching {symbol}: {start.date()} → {end.date()}")

    try:
        df = yf.download(
            symbol,
            start=start.strftime('%Y-%m-%d'),
            end=end.strftime('%Y-%m-%d'),
            interval='1d',
            progress=False,
            multi_level_index=False,
        )

        if df.empty:
            if log:
                log.warning(f"  No data for {symbol}")
            return None

        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(c in df.columns for c in required):
            if log:
                log.warning(f"  Missing columns for {symbol}: {list(df.columns)}")
            return None

        df = df[required].dropna()

        if len(df) < 100:
            if log:
                log.warning(f"  Only {len(df)} bars for {symbol}, need 100+")
            return None

        if log:
            log.debug(f"  Got {len(df)} bars ({df.index[0].date()} → {df.index[-1].date()})")

        _data_cache[cache_key] = df
        return df

    except Exception as e:
        if log:
            log.error(f"  Error fetching {symbol}: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def safe_float(value, default=0.0) -> float:
    """Safely convert a stat value that might be NaN."""
    try:
        v = float(value)
        return default if np.isnan(v) else v
    except (TypeError, ValueError):
        return default


def buy_and_hold_return(data: pd.DataFrame) -> float:
    """Calculate buy & hold return % for a dataset."""
    return (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100


# ─────────────────────────────────────────────────────────────
# Grid search optimization
# ─────────────────────────────────────────────────────────────

def find_optimal_params(
    strategy_class,
    symbol_data: dict[str, pd.DataFrame],
    tp_range: list[float],
    sl_range: list[float],
    log: logging.Logger,
    lenient: bool = False,
) -> tuple[tuple[float, float] | None, list[dict]]:
    """
    Grid search: for each (tp, sl) combo, run across ALL symbols.
    Pick the combo that maximizes the number of passing symbols.
    Break ties by average return.

    In lenient mode, a symbol "passes" if metrics are good (no B&H requirement).

    Returns (best_combo, full_grid_results).
    """
    grid_results = []
    best_combo = None
    best_pass_count = -1
    best_avg_return = -float('inf')

    total_combos = len(tp_range) * len(sl_range)
    log.info(f"  Grid search: {len(tp_range)} TP x {len(sl_range)} SL = {total_combos} combos x {len(symbol_data)} symbols")

    for tp in tp_range:
        for sl in sl_range:
            combo_returns = []
            combo_pass_count = 0
            combo_total_trades = 0

            for symbol, data in symbol_data.items():
                bh = buy_and_hold_return(data)

                try:
                    bt = Backtest(
                        data, strategy_class,
                        cash=BACKTEST_CASH,
                        commission=BACKTEST_COMMISSION,
                        exclusive_orders=True,
                        finalize_trades=True,
                    )
                    stats = bt.run(tp_pct=tp, sl_pct=sl)

                    trades = int(stats['# Trades'])
                    ret = safe_float(stats['Return [%]'])
                    wr = safe_float(stats['Win Rate [%]'])
                    pf = safe_float(stats['Profit Factor'])

                    combo_total_trades += trades
                    combo_returns.append(ret)

                    # Check if this symbol passes with these params
                    metrics_ok = (trades >= MIN_TRADES and wr >= MIN_WIN_RATE
                                  and pf >= MIN_PROFIT_FACTOR and ret > 0)

                    if lenient:
                        if metrics_ok:
                            combo_pass_count += 1
                    else:
                        if metrics_ok and ret > bh:
                            combo_pass_count += 1

                except Exception:
                    continue

            if not combo_returns:
                continue

            avg_ret = float(np.mean(combo_returns))

            grid_results.append({
                'tp_pct': tp,
                'sl_pct': sl,
                'symbols_passed': combo_pass_count,
                'symbols_tested': len(symbol_data),
                'pass_rate': round(combo_pass_count / len(symbol_data) * 100, 1),
                'avg_return': round(avg_ret, 2),
                'total_trades': combo_total_trades,
            })

            # Better = more symbols pass; tie-break on avg return
            if (combo_pass_count > best_pass_count
                    or (combo_pass_count == best_pass_count and avg_ret > best_avg_return)):
                best_pass_count = combo_pass_count
                best_avg_return = avg_ret
                best_combo = (tp, sl)

    return best_combo, grid_results


# ─────────────────────────────────────────────────────────────
# Validation pass
# ─────────────────────────────────────────────────────────────

def validate_with_params(
    strategy_class,
    symbol_data: dict[str, pd.DataFrame],
    tp: float,
    sl: float,
    log: logging.Logger,
    lenient: bool = False,
) -> tuple[str, list[dict]]:
    """
    Run final validation pass with chosen TP/SL on all symbols.

    Returns (verdict, per_symbol_results).
    verdict: "STRONG_PASS" | "PASS" | "FAIL"

    Three-tier per-symbol status:
      STRONG = all criteria met including beat B&H
      PASS   = metrics pass but didn't beat B&H
      FAIL   = metrics don't pass
    """
    results = []
    symbols_strong = 0
    symbols_metrics = 0

    log.info(f"\n  {'Symbol':<7} {'Status':<8} {'Return':>8} {'B&H':>8} {'Trades':>7} {'WinRate':>8} {'PF':>6} {'MaxDD':>7}")
    log.info(f"  {'─'*64}")

    for symbol, data in symbol_data.items():
        bh = buy_and_hold_return(data)

        try:
            bt = Backtest(
                data, strategy_class,
                cash=BACKTEST_CASH,
                commission=BACKTEST_COMMISSION,
                exclusive_orders=True,
                finalize_trades=True,
            )
            stats = bt.run(tp_pct=tp, sl_pct=sl)

            trades = int(stats['# Trades'])
            ret = safe_float(stats['Return [%]'])
            wr = safe_float(stats['Win Rate [%]'])
            pf = safe_float(stats['Profit Factor'])
            dd = abs(safe_float(stats['Max. Drawdown [%]']))
            avg_trade = safe_float(stats['Avg. Trade [%]'])

            # Separate metrics check from B&H check
            metrics_ok = (trades >= MIN_TRADES and wr >= MIN_WIN_RATE
                         and pf >= MIN_PROFIT_FACTOR and ret > 0)
            beats_bh = ret > bh

            if metrics_ok and beats_bh:
                symbols_strong += 1
                symbols_metrics += 1
                status = "STRONG"
            elif metrics_ok:
                symbols_metrics += 1
                status = "PASS"
            else:
                status = "FAIL"

            log.info(
                f"  {symbol:<7} {status:<8} {ret:>+7.1f}% {bh:>+7.1f}% {trades:>7} {wr:>7.1f}% {pf:>5.2f} {dd:>6.1f}%"
            )

            if status == "FAIL":
                checks = {
                    'min_trades': trades >= MIN_TRADES,
                    'win_rate': wr >= MIN_WIN_RATE,
                    'profit_factor': pf >= MIN_PROFIT_FACTOR,
                    'positive_return': ret > 0,
                }
                fails = [k for k, v in checks.items() if not v]
                log.debug(f"           Failed: {', '.join(fails)}")
            elif status == "PASS":
                log.debug(f"           Note: metrics pass but B&H={bh:+.1f}% > return={ret:+.1f}%")

            # Save HTML chart for passing symbols (STRONG or PASS)
            if status in ("STRONG", "PASS"):
                try:
                    chart_name = f"{symbol}_{strategy_class.__name__}"
                    bt.plot(filename=str(RESULTS_DIR / f"{chart_name}.html"), open_browser=False)
                except Exception:
                    pass

            results.append({
                'symbol': symbol,
                'return_pct': round(ret, 2),
                'buy_hold_pct': round(bh, 2),
                'num_trades': trades,
                'win_rate': round(wr, 1),
                'profit_factor': round(pf, 2),
                'max_drawdown': round(dd, 1),
                'avg_trade_pct': round(avg_trade, 2),
                'metrics_passed': metrics_ok,
                'beat_buy_hold': beats_bh,
                'status': status,
            })

        except Exception as e:
            log.error(f"  {symbol:<7} ERROR  {e}")
            results.append({
                'symbol': symbol, 'return_pct': 0, 'buy_hold_pct': round(bh, 2),
                'num_trades': 0, 'win_rate': 0, 'profit_factor': 0,
                'max_drawdown': 0, 'avg_trade_pct': 0,
                'metrics_passed': False, 'beat_buy_hold': False, 'status': 'FAIL',
            })

    total = len(symbol_data)
    strong_rate = symbols_strong / total if total > 0 else 0
    metrics_rate = symbols_metrics / total if total > 0 else 0

    log.info(f"  {'─'*64}")
    log.info(
        f"  STRONG: {symbols_strong}/{total} ({strong_rate*100:.0f}%) "
        f"| Metrics PASS: {symbols_metrics}/{total} ({metrics_rate*100:.0f}%) "
        f"— need {MIN_SYMBOL_PASS_RATE*100:.0f}%"
    )

    # Determine overall verdict
    if strong_rate >= MIN_SYMBOL_PASS_RATE:
        verdict = "STRONG_PASS"
    elif lenient and metrics_rate >= MIN_SYMBOL_PASS_RATE:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    return verdict, results


# ─────────────────────────────────────────────────────────────
# Regime analysis
# ─────────────────────────────────────────────────────────────

def regime_analysis(results: list[dict], log: logging.Logger):
    """Print regime analysis summary based on buy & hold returns across tested symbols."""
    bh_returns = [r['buy_hold_pct'] for r in results if r['buy_hold_pct'] != 0]
    if not bh_returns:
        return

    avg_bh = float(np.mean(bh_returns))
    max_bh = max(bh_returns)
    min_bh = min(bh_returns)
    bh_beat_count = sum(1 for r in results if r['beat_buy_hold'])

    # Classify regime
    if avg_bh > 40:
        regime = "STRONG BULL"
        advice = (
            "B&H returns very high — short-term TP strategies will naturally "
            "underperform. Focus on metrics quality (WR, PF) over absolute return."
        )
    elif avg_bh > 20:
        regime = "BULL"
        advice = (
            "Moderate bull market — strategies with <10% TP may not beat B&H. "
            "Consider regime-aware deployment or wider TP targets."
        )
    elif avg_bh > -5:
        regime = "NEUTRAL"
        advice = (
            "Sideways market — ideal environment for mean-reversion and "
            "range-bound strategies. B&H comparison is meaningful."
        )
    elif avg_bh > -20:
        regime = "BEAR"
        advice = (
            "Bearish conditions — active strategies should outperform B&H easily. "
            "If they don't, check signal quality."
        )
    else:
        regime = "STRONG BEAR"
        advice = "Severe drawdown period — risk management paramount. Any positive return is good."

    log.info(f"\n  REGIME ANALYSIS")
    log.info(f"  {'─'*50}")
    log.info(f"  Avg Buy & Hold:  {avg_bh:+.1f}% (range: {min_bh:+.1f}% to {max_bh:+.1f}%)")
    log.info(f"  Market Regime:   {regime}")
    log.info(f"  Beat B&H:        {bh_beat_count}/{len(results)} symbols")
    log.info(f"  Note: {advice}")


# ─────────────────────────────────────────────────────────────
# Output writers
# ─────────────────────────────────────────────────────────────

def write_winner(idea_name: str, tp: float, sl: float, results: list[dict], verdict: str, regime: str = ''):
    """Append a passing strategy to winners.csv."""
    all_returns = [r['return_pct'] for r in results]
    avg_return = round(float(np.mean(all_returns)), 2) if all_returns else 0

    symbols_metrics = sum(1 for r in results if r['metrics_passed'])
    symbols_bh = sum(1 for r in results if r['beat_buy_hold'])
    total_symbols = len(results)

    # beat_buy_hold column: "3/9" format
    bh_str = f"{symbols_bh}/{total_symbols}"

    # recommendation based on verdict
    if verdict == "STRONG_PASS":
        rec = "DEPLOY"
    elif verdict == "PASS":
        rec = "DEPLOY_CAUTIOUS"
    else:
        rec = "DO_NOT_DEPLOY"

    file_exists = WINNERS_FILE.exists() and WINNERS_FILE.stat().st_size > 0
    with open(WINNERS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                'date', 'strategy_name', 'optimal_tp', 'optimal_sl',
                'avg_return', 'symbols_passed', 'symbols_tested',
                'beat_buy_hold', 'recommendation', 'regime',
            ])
        writer.writerow([
            datetime.now().strftime('%Y-%m-%d %H:%M'),
            idea_name, tp, sl, avg_return,
            symbols_metrics, total_symbols,
            bh_str, rec, regime,
        ])


def save_optimization_grid(idea_name: str, grid_results: list[dict]):
    """Save the full TP/SL grid search results for heat map analysis."""
    if not grid_results:
        return

    df = pd.DataFrame(grid_results)
    df.to_csv(OPT_DIR / f"{idea_name}_grid.csv", index=False)

    # Also create a pivot table (heat map format): rows=SL, cols=TP, values=pass_rate
    pivot = df.pivot_table(index='sl_pct', columns='tp_pct', values='pass_rate', aggfunc='first')
    pivot.to_csv(OPT_DIR / f"{idea_name}_heatmap.csv")


def save_detail_results(idea_name: str, results: list[dict]):
    """Save per-symbol validation results."""
    if not results:
        return
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / f"{idea_name}_details.csv", index=False)


# ─────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────

def process_idea(idea: dict, symbols: list[str], log: logging.Logger, lenient: bool = False) -> str:
    """
    Full RBI pipeline for a single strategy idea:
      1. Look up strategy class
      2. Fetch data for all symbols
      3. Grid-search optimal TP/SL
      4. Validate with best params
      5. Output results

    Returns verdict: "STRONG_PASS" | "PASS" | "FAIL"
    """
    name = idea['name']
    tp_range = idea.get('tp_range', [3.0, 4.0, 5.0, 6.0])
    sl_range = idea.get('sl_range', [2.0, 3.0, 4.0])

    log.info(f"\n{'='*65}")
    log.info(f"  STRATEGY: {name}")
    log.info(f"{'='*65}")
    log.info(f"  Description: {idea.get('description', 'N/A')}")
    log.info(f"  Entry: {idea.get('entry', 'N/A')}")
    log.info(f"  TP range: {tp_range}")
    log.info(f"  SL range: {sl_range}")
    side = idea.get('side', 'long')
    regime = idea.get('regime', '')
    if side != 'long':
        log.info(f"  Side: {side.upper()}")
    if regime:
        log.info(f"  Regime: {regime}")
    if lenient:
        log.info(f"  Mode: LENIENT (B&H check is advisory only)")

    # Step 1: Look up strategy class
    strategy_class = STRATEGY_REGISTRY.get(name)
    if not strategy_class:
        log.error(f"  No strategy class found for '{name}'")
        log.error(f"  Available: {list(STRATEGY_REGISTRY.keys())}")
        return "FAIL"

    log.info(f"  Class: {strategy_class.__name__}")

    # Step 2: Fetch data for all symbols
    log.info(f"\n  Fetching {BACKTEST_YEARS}-year data for {len(symbols)} symbols...")
    symbol_data = {}
    for sym in symbols:
        df = fetch_data(sym, years=BACKTEST_YEARS, log=log)
        if df is not None:
            symbol_data[sym] = df

    if len(symbol_data) < 3:
        log.error(f"  Only {len(symbol_data)} symbols have data (need 3+)")
        update_idea_status(IDEAS_FILE, name, "failed")
        return "FAIL"

    log.info(f"  Data ready for {len(symbol_data)} symbols: {list(symbol_data.keys())}")

    # Step 3: Grid-search optimal TP/SL across all symbols
    log.info(f"\n  PHASE 1: Parameter Optimization")
    log.info(f"  {'─'*50}")

    best_combo, grid_results = find_optimal_params(
        strategy_class, symbol_data, tp_range, sl_range, log, lenient=lenient
    )

    save_optimization_grid(name, grid_results)

    if best_combo is None:
        log.error(f"  No valid parameter combination found")
        update_idea_status(IDEAS_FILE, name, "failed")
        return "FAIL"

    opt_tp, opt_sl = best_combo
    log.info(f"\n  Optimal params: TP={opt_tp}% / SL={opt_sl}%")

    # Show top 5 combos
    if grid_results:
        sorted_grid = sorted(grid_results, key=lambda x: (-x['symbols_passed'], -x['avg_return']))
        log.info(f"\n  Top parameter combos:")
        log.info(f"  {'TP':>5} {'SL':>5} {'PassRate':>9} {'AvgRet':>8} {'Trades':>7}")
        for row in sorted_grid[:5]:
            log.info(
                f"  {row['tp_pct']:>5.1f} {row['sl_pct']:>5.1f} "
                f"{row['pass_rate']:>8.1f}% {row['avg_return']:>+7.1f}% "
                f"{row['total_trades']:>7}"
            )

    # Step 4: Final validation with optimal params
    log.info(f"\n  PHASE 2: Validation (TP={opt_tp}% / SL={opt_sl}%)")
    log.info(f"  {'─'*50}")
    if lenient:
        log.info(f"  Criteria: trades>={MIN_TRADES} | WR>={MIN_WIN_RATE}% | PF>={MIN_PROFIT_FACTOR} | return>0%")
        log.info(f"  (B&H check: advisory only)")
    else:
        log.info(f"  Criteria: trades>={MIN_TRADES} | WR>={MIN_WIN_RATE}% | PF>={MIN_PROFIT_FACTOR} | beat B&H")

    verdict, results = validate_with_params(
        strategy_class, symbol_data, opt_tp, opt_sl, log, lenient=lenient
    )

    save_detail_results(name, results)

    # Regime analysis
    regime_analysis(results, log)

    # Step 5: Verdict
    metrics_count = sum(1 for r in results if r['metrics_passed'])
    bh_count = sum(1 for r in results if r['beat_buy_hold'])
    total_count = len(results)

    log.info(f"\n  {'='*50}")
    if verdict == "STRONG_PASS":
        log.info(f"  VERDICT: ★ STRONG PASS ★")
        log.info(f"  Optimal: TP={opt_tp}% / SL={opt_sl}%")
        log.info(f"  Metrics: {metrics_count}/{total_count} | Beat B&H: {bh_count}/{total_count}")
        update_idea_status(IDEAS_FILE, name, "validated")
        write_winner(name, opt_tp, opt_sl, results, verdict, regime=regime)
        log.info(f"  Recommendation: DEPLOY")
        log.info(f"  >> Written to winners.csv")
    elif verdict == "PASS":
        log.info(f"  VERDICT: PASS (metrics OK, B&H not beaten)")
        log.info(f"  Optimal: TP={opt_tp}% / SL={opt_sl}%")
        log.info(f"  Metrics: {metrics_count}/{total_count} | Beat B&H: {bh_count}/{total_count}")
        update_idea_status(IDEAS_FILE, name, "validated_no_bh")
        write_winner(name, opt_tp, opt_sl, results, verdict, regime=regime)
        log.info(f"  Recommendation: DEPLOY_CAUTIOUS (paper trade first, monitor vs B&H)")
        log.info(f"  >> Written to winners.csv")
    else:
        log.info(f"  VERDICT: FAILED")
        log.info(f"  Best was TP={opt_tp}% / SL={opt_sl}%")
        log.info(f"  Metrics: {metrics_count}/{total_count} | Beat B&H: {bh_count}/{total_count}")
        update_idea_status(IDEAS_FILE, name, "failed")
    log.info(f"  {'='*50}")

    return verdict


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RBI Agent v2 — Research, Backtest, Implement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 rbi_agent.py                  Test new/testing strategies
  python3 rbi_agent.py --idea momentum  Test a specific strategy
  python3 rbi_agent.py --all            Re-test everything
  python3 rbi_agent.py --all --lenient  Lenient B&H validation
  python3 rbi_agent.py --symbols TSLA,AMD,NVDA
        """,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--idea", type=str, help="Test a specific idea (partial name match)")
    parser.add_argument("--all", action="store_true", help="Test ALL ideas including validated/failed")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols to test on (overrides default)")
    parser.add_argument("--dry-run", action="store_true", help="Parse ideas without running backtests")
    parser.add_argument("--lenient", action="store_true",
                        help="Relax B&H requirement (flag but don't fail if strategy doesn't beat buy & hold)")
    args = parser.parse_args()

    log = setup_logging(args.verbose)

    # Header
    mode_str = "LENIENT" if args.lenient else "STRICT"
    log.info("")
    log.info("=" * 65)
    log.info("  RBI AGENT v2 — Research, Backtest, Implement")
    log.info("  MoonDev Methodology | Multi-Symbol Validation")
    log.info("=" * 65)
    log.info(f"  Data:       {BACKTEST_YEARS} years daily OHLCV via yfinance")
    log.info(f"  Cash:       ${BACKTEST_CASH:,} | Commission: {BACKTEST_COMMISSION*100:.1f}%")
    log.info(f"  Min Trades: {MIN_TRADES} | Min WR: {MIN_WIN_RATE}% | Min PF: {MIN_PROFIT_FACTOR}")
    log.info(f"  Pass Rate:  {MIN_SYMBOL_PASS_RATE*100:.0f}% of symbols must pass")
    log.info(f"  Mode:       {mode_str}")
    if args.lenient:
        log.info(f"  B&H Check:  Advisory only (won't cause FAIL)")
    log.info("")

    # Parse symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    else:
        symbols = DEFAULT_SYMBOLS
    log.info(f"  Test symbols: {symbols}")

    # Parse ideas
    if not IDEAS_FILE.exists():
        log.error(f"ideas.txt not found: {IDEAS_FILE}")
        return 1

    ideas = parse_ideas(IDEAS_FILE)
    log.info(f"  Parsed {len(ideas)} ideas from ideas.txt\n")

    for idea in ideas:
        status = idea.get('status', '?')
        log.info(f"    [{status:<16s}] {idea['name']}")

    # Filter
    if args.idea:
        targets = [i for i in ideas if args.idea.lower() in i['name'].lower()]
        if not targets:
            log.error(f"\nNo idea matching '{args.idea}'")
            log.info(f"Available: {[i['name'] for i in ideas]}")
            return 1
    elif args.all:
        targets = ideas
    else:
        # Default: new + testing
        targets = [i for i in ideas if i.get('status', '').strip() in ('new', 'testing')]

    if not targets:
        log.info("\nNo ideas to test. Use --all to re-test everything.")
        return 0

    log.info(f"\nWill backtest {len(targets)} idea(s):")
    for t in targets:
        log.info(f"  → {t['name']} [{t.get('status', '?')}]")

    if args.dry_run:
        log.info("\n[DRY RUN] Stopping before backtests.")
        return 0

    # Process each idea
    strong_count = 0
    pass_count = 0
    failed_count = 0
    results_summary = []

    for idea in targets:
        try:
            verdict = process_idea(idea, symbols, log, lenient=args.lenient)
            if verdict == "STRONG_PASS":
                strong_count += 1
                results_summary.append((idea['name'], 'STRONG_PASS'))
            elif verdict == "PASS":
                pass_count += 1
                results_summary.append((idea['name'], 'PASS'))
            else:
                failed_count += 1
                results_summary.append((idea['name'], 'FAILED'))
        except Exception as e:
            log.error(f"\nError processing '{idea['name']}': {e}")
            import traceback
            log.debug(traceback.format_exc())
            failed_count += 1
            results_summary.append((idea['name'], 'ERROR'))

    # Final summary
    log.info(f"\n{'='*65}")
    log.info(f"  RBI AGENT COMPLETE")
    log.info(f"{'='*65}")
    log.info(f"  Strategies tested: {len(targets)}")
    log.info(f"  Strong Pass: {strong_count}")
    log.info(f"  Pass:        {pass_count}")
    log.info(f"  Failed:      {failed_count}")
    log.info("")

    for name, status in results_summary:
        if status == "STRONG_PASS":
            marker = "STRONG"
        elif status == "PASS":
            marker = " PASS "
        else:
            marker = " FAIL "
        log.info(f"    [{marker}] {name}")

    log.info("")
    if WINNERS_FILE.exists():
        log.info(f"  Winners:      {WINNERS_FILE}")
    log.info(f"  Optimization: {OPT_DIR}/")
    log.info(f"  Results:      {RESULTS_DIR}/")
    log.info(f"  Logs:         {LOGS_DIR}/")
    log.info("")

    return 0


if __name__ == "__main__":
    sys.exit(main())

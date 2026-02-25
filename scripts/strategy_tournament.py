#!/usr/bin/env python3
"""
Strategy Tournament — Stock Bot Evaluation Framework

Evaluates all stock bots by backtesting against historical data from Alpaca.

Commands:
    --scan      List all bots, parameters, and lifecycle status
    --evaluate  Backtest all bots on 30 days of hourly data
    --review    Review live bots with >= 20 trades; promote/demote

Grading:
    A: return >= 0 AND profit_factor >= 1.0
    D: marginal (PF < 1.0 but return > -2%)
    F: losing (return <= -2%)

Usage:
    python scripts/strategy_tournament.py --scan
    python scripts/strategy_tournament.py --evaluate
    python scripts/strategy_tournament.py --evaluate --days 14
    python scripts/strategy_tournament.py --review
"""

import argparse
import importlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from src.utils.stock_journal import StockJournal

# Bot registry: module_path -> class_name
BOT_REGISTRY = {
    'bots.breakout_bot': 'BreakoutBot',
    'bots.momentum_bot': 'MomentumBot',
    'bots.bb_bounce_bot': 'BBBounceBot',
    'bots.macd_bot': 'MACDBot',
    'bots.mean_reversion_bot': 'MeanReversionBot',
}

# Tournament config
CONFIG = {
    'promote_wr': 40,
    'promote_pf': 1.0,
    'demote_wr': 30,
    'min_trades_review': 20,
}


def load_bot_class(module_path, class_name):
    """Dynamically import a bot class."""
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except Exception as e:
        print(f"  Error loading {module_path}.{class_name}: {e}")
        return None


def get_bot_info(bot_cls):
    """Extract bot metadata without instantiation."""
    return {
        'bot_name': bot_cls.BOT_NAME,
        'strategy': bot_cls.STRATEGY,
        'symbols': bot_cls.DEFAULT_SYMBOLS,
        'tp_pct': bot_cls.TP_PCT,
        'sl_pct': bot_cls.SL_PCT,
    }


def fetch_hourly_candles(symbol, days=30):
    """Fetch hourly candles from Alpaca for backtesting."""
    try:
        import alpaca_trade_api as tradeapi
        from src.utils.order_utils import get_api

        api = get_api()
        cal_days = days * 2 + 5  # Extra for weekends/holidays

        end = datetime.now(timezone.utc) - timedelta(hours=1)
        start = end - timedelta(days=cal_days)

        bars = api.get_bars(
            symbol,
            tradeapi.TimeFrame.Hour,
            start=start.isoformat(),
            end=end.isoformat(),
            feed='iex'
        ).df

        if bars.empty:
            return []

        candles = []
        for _, row in bars.iterrows():
            candles.append({
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume']),
            })

        return candles
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return []


def simulate_bot(bot_cls, candles_by_symbol, lookback=50):
    """
    Simulate a bot's check_signal over historical candles.

    Iterates through candles, feeding lookback windows to check_signal,
    then simulates bracket exits based on TP/SL percentages.

    Returns list of simulated trade dicts.
    """
    # We need a mock bot instance that doesn't connect to Alpaca
    # Instead, we'll call check_signal statically with enough context

    tp_pct = bot_cls.TP_PCT
    sl_pct = bot_cls.SL_PCT
    trades = []

    for symbol, candles in candles_by_symbol.items():
        if len(candles) < lookback + 10:
            continue

        in_trade = False
        entry_price = 0
        entry_idx = 0
        direction = 'LONG'

        for i in range(lookback, len(candles)):
            price = candles[i]['close']
            window = candles[i - lookback:i]
            prices = [c['close'] for c in window]

            if in_trade:
                # Check TP/SL
                high = candles[i]['high']
                low = candles[i]['low']

                if direction == 'LONG':
                    tp_price = entry_price * (1 + tp_pct / 100)
                    sl_price = entry_price * (1 - sl_pct / 100)
                    if high >= tp_price:
                        pnl_pct = tp_pct
                        exit_reason = 'TP'
                        in_trade = False
                    elif low <= sl_price:
                        pnl_pct = -sl_pct
                        exit_reason = 'SL'
                        in_trade = False
                else:  # SHORT
                    tp_price = entry_price * (1 - tp_pct / 100)
                    sl_price = entry_price * (1 + sl_pct / 100)
                    if low <= tp_price:
                        pnl_pct = tp_pct
                        exit_reason = 'TP'
                        in_trade = False
                    elif high >= sl_price:
                        pnl_pct = -sl_pct
                        exit_reason = 'SL'
                        in_trade = False

                if not in_trade:
                    trades.append({
                        'symbol': symbol,
                        'direction': direction,
                        'entry_price': entry_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': exit_reason,
                        'bars_held': i - entry_idx,
                    })
                continue

            # Check for signal using a lightweight approach
            # We can't instantiate the bot (needs Alpaca), so we replicate
            # the signal logic for each bot type
            should_enter, metadata = _check_signal_static(
                bot_cls, symbol, price, prices, window
            )

            if should_enter:
                in_trade = True
                entry_price = price
                entry_idx = i
                direction = metadata.get('direction', 'LONG')

    return trades


def _check_signal_static(bot_cls, symbol, price, prices, candles):
    """
    Static signal check that mimics bot.check_signal() without Alpaca.
    Uses the indicator helpers from base_bot.
    """
    from bots.base_bot import calc_rsi, calc_sma, calc_ema, calc_bollinger_bands, calc_macd

    strategy = bot_cls.STRATEGY

    if strategy == 'MOMENTUM':
        if len(prices) < 20:
            return False, {}
        sma_20 = calc_sma(prices, 20)
        ema_9 = calc_ema(prices, 9)
        rsi = calc_rsi(prices, 14)
        above_sma = price > sma_20
        rsi_ok = 40 <= rsi <= 80
        ema_distance = abs(price - ema_9) / price if price > 0 else 1.0
        at_ema = ema_distance <= 0.03
        conditions = [above_sma, rsi_ok, at_ema, True]  # volume always True
        passed = sum(conditions)
        if passed >= 3 and above_sma:
            return True, {'direction': 'LONG', 'signal_strength': f'{passed}/4'}
        return False, {}

    elif strategy == 'BREAKOUT':
        lookback = 24
        if len(candles) < lookback:
            return False, {}
        recent = candles[-lookback:]
        high_24h = max(c['high'] for c in recent)
        buffer = high_24h * 0.001
        if price > high_24h + buffer:
            return True, {'direction': 'LONG'}
        return False, {}

    elif strategy == 'BB_BOUNCE':
        if len(prices) < 20:
            return False, {}
        upper, middle, lower = calc_bollinger_bands(prices, 20, 2.0)
        rsi = calc_rsi(prices, 14)
        if lower is not None and price <= lower and rsi < 30:
            return True, {'direction': 'LONG'}
        return False, {}

    elif strategy == 'MACD':
        if len(prices) < 35:
            return False, {}
        macd_line, signal_line, histogram = calc_macd(prices, 12, 26, 9)
        if macd_line is None:
            return False, {}
        # Also need previous values for crossover detection
        prev_prices = prices[:-1]
        if len(prev_prices) < 35:
            return False, {}
        prev_macd, prev_signal, _ = calc_macd(prev_prices, 12, 26, 9)
        if prev_macd is None:
            return False, {}
        # Bullish crossover: MACD crosses above signal
        if prev_macd <= prev_signal and macd_line > signal_line:
            return True, {'direction': 'LONG'}
        return False, {}

    elif strategy == 'MEAN_REV':
        if len(prices) < 20:
            return False, {}
        rsi = calc_rsi(prices, 14)
        sma_20 = calc_sma(prices, 20)
        below_sma = price < sma_20 * 0.98  # 2% below SMA
        if rsi < 30 and below_sma:
            return True, {'direction': 'LONG'}
        return False, {}

    return False, {}


def grade_results(trades):
    """Grade a set of simulated trades."""
    if not trades:
        return 'NO_SIGNALS', {}

    pnls = [t['pnl_pct'] for t in trades]
    total_return = sum(pnls)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    win_rate = len(winners) / len(pnls) * 100
    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe approximation
    if len(pnls) > 1:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252) if np.std(pnls) > 0 else 0
    else:
        sharpe = 0

    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = max(drawdown) if len(drawdown) > 0 else 0

    metrics = {
        'total_trades': len(pnls),
        'return_pct': total_return,
        'win_rate': win_rate,
        'profit_factor': pf,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_pnl_pct': np.mean(pnls),
    }

    # Grade
    if total_return >= 0 and pf >= 1.0:
        grade = 'A'
    elif total_return > -2:
        grade = 'D'
    else:
        grade = 'F'

    return grade, metrics


def cmd_scan():
    """List all bots and their parameters."""
    print("\n" + "=" * 70)
    print("  STRATEGY TOURNAMENT — Bot Scan")
    print("=" * 70)

    journal = StockJournal()

    for module_path, class_name in BOT_REGISTRY.items():
        bot_cls = load_bot_class(module_path, class_name)
        if bot_cls is None:
            continue

        info = get_bot_info(bot_cls)
        print(f"\n  {info['bot_name']} ({info['strategy']})")
        print(f"    TP: {info['tp_pct']}% | SL: {info['sl_pct']}%")
        print(f"    Symbols: {', '.join(info['symbols'])}")

        # Check lifecycle status
        conn = journal._get_conn()
        try:
            row = conn.execute("""
                SELECT stage, timestamp FROM strategy_lifecycle
                WHERE bot_name = ? ORDER BY id DESC LIMIT 1
            """, (info['bot_name'],)).fetchone()
            if row:
                print(f"    Stage: {row['stage']} ({row['timestamp']})")
            else:
                print(f"    Stage: UNKNOWN (no lifecycle data)")
        finally:
            conn.close()

        # Check live trade count
        stats = journal.get_stats(strategy=info['strategy'])
        if stats:
            print(f"    Live: {stats['total_trades']} trades, "
                  f"WR {stats['win_rate']:.1f}%, PnL ${stats['total_pnl']:+.2f}")

    print("\n" + "=" * 70 + "\n")


def cmd_evaluate(days=30):
    """Backtest all bots on historical data."""
    print("\n" + "=" * 70)
    print(f"  STRATEGY TOURNAMENT — Evaluate ({days}d backtest)")
    print("=" * 70)

    results = []

    for module_path, class_name in BOT_REGISTRY.items():
        bot_cls = load_bot_class(module_path, class_name)
        if bot_cls is None:
            continue

        info = get_bot_info(bot_cls)
        print(f"\n  Evaluating {info['bot_name']}...")

        # Fetch candles for each symbol
        candles_by_symbol = {}
        for symbol in info['symbols'][:5]:  # Limit to 5 symbols for speed
            print(f"    Fetching {symbol}...", end=" ", flush=True)
            candles = fetch_hourly_candles(symbol, days=days)
            if candles:
                candles_by_symbol[symbol] = candles
                print(f"{len(candles)} bars")
            else:
                print("FAILED")

        if not candles_by_symbol:
            print(f"    No data available, skipping")
            results.append({
                'bot_name': info['bot_name'],
                'strategy': info['strategy'],
                'grade': 'NO_DATA',
                'metrics': {},
            })
            continue

        # Simulate
        trades = simulate_bot(bot_cls, candles_by_symbol)
        grade, metrics = grade_results(trades)

        results.append({
            'bot_name': info['bot_name'],
            'strategy': info['strategy'],
            'grade': grade,
            'metrics': metrics,
        })

        if metrics:
            pf_str = f"{metrics['profit_factor']:.2f}" if metrics['profit_factor'] != float('inf') else "inf"
            print(f"    Grade {grade}: {metrics['total_trades']} trades, "
                  f"{metrics['return_pct']:+.2f}%, WR {metrics['win_rate']:.1f}%, "
                  f"PF {pf_str}, Sharpe {metrics.get('sharpe', 0):.2f}")
        else:
            print(f"    Grade {grade}: No signals generated")

    # Summary table
    print("\n" + "=" * 70)
    print("  TOURNAMENT RESULTS")
    print("=" * 70)
    print(f"\n  {'Bot':<25} {'Grade':>6} {'Trades':>7} {'Return':>8} {'WR':>6} {'PF':>6}")
    print(f"  {'-'*25} {'-'*6} {'-'*7} {'-'*8} {'-'*6} {'-'*6}")

    for r in sorted(results, key=lambda x: {'A': 0, 'D': 1, 'F': 2}.get(x['grade'], 3)):
        m = r['metrics']
        if m:
            pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != float('inf') else "inf"
            print(f"  {r['bot_name']:<25} {r['grade']:>6} {m['total_trades']:>7} "
                  f"{m['return_pct']:>+7.2f}% {m['win_rate']:>5.1f}% {pf_str:>6}")
        else:
            print(f"  {r['bot_name']:<25} {r['grade']:>6}       —        —      —      —")

    print("\n" + "=" * 70 + "\n")
    return results


def cmd_review():
    """Review live bots and promote/demote based on real performance."""
    print("\n" + "=" * 70)
    print("  STRATEGY TOURNAMENT — Live Review")
    print("=" * 70)

    journal = StockJournal()

    for module_path, class_name in BOT_REGISTRY.items():
        bot_cls = load_bot_class(module_path, class_name)
        if bot_cls is None:
            continue

        info = get_bot_info(bot_cls)
        stats = journal.get_stats(strategy=info['strategy'])

        if not stats or stats['total_trades'] < CONFIG['min_trades_review']:
            trade_count = stats['total_trades'] if stats else 0
            print(f"\n  {info['bot_name']}: {trade_count} trades "
                  f"(need {CONFIG['min_trades_review']})")
            continue

        wr = stats['win_rate']
        pf = stats['profit_factor']

        # Determine action
        if wr >= CONFIG['promote_wr'] and pf >= CONFIG['promote_pf']:
            action = 'PROMOTED'
            reason = f"WR {wr:.1f}% >= {CONFIG['promote_wr']}%, PF {pf:.2f} >= {CONFIG['promote_pf']}"
        elif wr < CONFIG['demote_wr']:
            action = 'DEMOTED'
            reason = f"WR {wr:.1f}% < {CONFIG['demote_wr']}%"
        else:
            action = 'TESTING'
            reason = f"WR {wr:.1f}%, PF {pf:.2f} — needs more data"

        pf_str = f"{pf:.2f}" if pf != float('inf') else "inf"
        print(f"\n  {info['bot_name']}: {stats['total_trades']} trades, "
              f"WR {wr:.1f}%, PF {pf_str}, PnL ${stats['total_pnl']:+.2f}")
        print(f"    → {action}: {reason}")

        # Log lifecycle event
        metrics_json = json.dumps({
            'trades': stats['total_trades'],
            'win_rate': wr,
            'profit_factor': pf,
            'total_pnl': stats['total_pnl'],
        })
        journal.log_lifecycle_event(
            bot_name=info['bot_name'],
            strategy=info['strategy'],
            stage=action,
            reason=reason,
            metrics=metrics_json,
        )

    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Stock Strategy Tournament")
    parser.add_argument('--scan', action='store_true', help='List all bots and status')
    parser.add_argument('--evaluate', action='store_true', help='Backtest all bots')
    parser.add_argument('--review', action='store_true', help='Review live performance')
    parser.add_argument('--days', type=int, default=30, help='Backtest days (default: 30)')
    args = parser.parse_args()

    if args.scan:
        cmd_scan()
    elif args.evaluate:
        cmd_evaluate(days=args.days)
    elif args.review:
        cmd_review()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Skill Updater — Stock Trading Self-Improving Loop

Reads closed_trades from SQLite, computes per-strategy stats, regime matrix,
hourly/daily patterns, and parameter recommendations. Writes the skill file
at skills/strategy_performance.md for consumption by the regime gating in
base_bot.py.

Pattern types discovered:
  - bad_regime: WR < 40% with >= 5 trades in a regime
  - bad_hour: WR < 35% with >= 3 trades in an hour (ET)
  - bad_day: WR < 35% with >= 3 trades on a day of week
  - high_slippage: avg slippage > 0.5% (from r_multiple data)
  - consecutive_losses: 5+ consecutive losses detected
  - edge_decay: win rate declining over last 3 windows

Usage:
    python scripts/skill_updater.py                # normal run (min 10 trades)
    python scripts/skill_updater.py --force        # run even with few trades
    python scripts/skill_updater.py --dry-run      # print but don't write
    python scripts/skill_updater.py --threshold 5  # lower min trade threshold
"""

import argparse
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.stock_journal import StockJournal

SKILL_FILE = PROJECT_ROOT / 'skills' / 'strategy_performance.md'
SKILL_FILE.parent.mkdir(parents=True, exist_ok=True)

# Bot module names → class-level params
BOT_MODULES = {
    'breakout_bot': 'bots.breakout_bot',
    'momentum_bot': 'bots.momentum_bot',
    'bb_bounce_bot': 'bots.bb_bounce_bot',
    'macd_bot': 'bots.macd_bot',
    'mean_reversion_bot': 'bots.mean_reversion_bot',
}

# Mapping from bot_name to strategy name (for display)
BOT_STRATEGIES = {
    'breakout_bot': 'BREAKOUT',
    'momentum_bot': 'MOMENTUM',
    'bb_bounce_bot': 'BB_BOUNCE',
    'macd_bot': 'MACD',
    'mean_reversion_bot': 'MEAN_REV',
}

ET_OFFSET = -5  # UTC-5 for ET (EST; -4 for EDT)


def utc_to_et_hour(epoch_ts):
    """Convert epoch timestamp to ET hour (0-23)."""
    if not epoch_ts:
        return None
    dt = datetime.fromtimestamp(epoch_ts, tz=timezone.utc)
    et_hour = (dt.hour + ET_OFFSET) % 24
    return et_hour


def epoch_to_weekday(epoch_ts):
    """Convert epoch timestamp to weekday name."""
    if not epoch_ts:
        return None
    dt = datetime.fromtimestamp(epoch_ts, tz=timezone.utc)
    return dt.strftime('%A')


def load_bot_params():
    """Load current TP/SL from bot modules."""
    params = {}
    for bot_name, module_path in BOT_MODULES.items():
        try:
            mod = importlib.import_module(module_path)
            # Find the Bot class
            for attr_name in dir(mod):
                obj = getattr(mod, attr_name)
                if (isinstance(obj, type) and hasattr(obj, 'TP_PCT')
                        and hasattr(obj, 'SL_PCT') and attr_name != 'BaseBot'):
                    params[bot_name] = {
                        'tp_pct': obj.TP_PCT,
                        'sl_pct': obj.SL_PCT,
                    }
                    break
        except Exception:
            pass
    return params


def compute_strategy_stats(trades):
    """Compute per-strategy statistics."""
    by_strategy = defaultdict(list)
    for t in trades:
        strat = t.get('strategy') or t.get('bot_name') or 'UNKNOWN'
        by_strategy[strat].append(t)

    stats = {}
    for strat, strat_trades in by_strategy.items():
        pnls = [t['pnl_usd'] for t in strat_trades if t.get('pnl_usd') is not None]
        if not pnls:
            continue

        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0

        durations = [t['duration_seconds'] for t in strat_trades
                     if t.get('duration_seconds') is not None and t['duration_seconds'] > 0]
        avg_duration_h = (sum(durations) / len(durations) / 3600) if durations else 0

        stats[strat] = {
            'trades': len(pnls),
            'win_rate': len(winners) / len(pnls) * 100,
            'total_pnl': sum(pnls),
            'avg_pnl': sum(pnls) / len(pnls),
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
            'best': max(pnls),
            'worst': min(pnls),
            'avg_duration_h': avg_duration_h,
        }

    return stats


def compute_regime_matrix(trades):
    """Compute WR/trades per (strategy, regime)."""
    matrix = defaultdict(lambda: defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0}))
    for t in trades:
        strat = t.get('strategy') or 'UNKNOWN'
        regime = t.get('regime_at_entry') or 'unknown'
        pnl = t.get('pnl_usd') or 0
        matrix[strat][regime]['trades'] += 1
        if pnl > 0:
            matrix[strat][regime]['wins'] += 1
        matrix[strat][regime]['pnl'] += pnl
    return matrix


def compute_hourly_stats(trades):
    """Compute WR by hour of day (ET) per strategy."""
    by_hour = defaultdict(lambda: defaultdict(lambda: {'trades': 0, 'wins': 0}))
    for t in trades:
        strat = t.get('strategy') or 'UNKNOWN'
        hour = utc_to_et_hour(t.get('entry_time'))
        if hour is None:
            continue
        by_hour[strat][hour]['trades'] += 1
        if (t.get('pnl_usd') or 0) > 0:
            by_hour[strat][hour]['wins'] += 1
    return by_hour


def compute_daily_stats(trades):
    """Compute WR by day of week per strategy."""
    by_day = defaultdict(lambda: defaultdict(lambda: {'trades': 0, 'wins': 0}))
    for t in trades:
        strat = t.get('strategy') or 'UNKNOWN'
        day = epoch_to_weekday(t.get('entry_time'))
        if day is None:
            continue
        by_day[strat][day]['trades'] += 1
        if (t.get('pnl_usd') or 0) > 0:
            by_day[strat][day]['wins'] += 1
    return by_day


def discover_patterns(trades, regime_matrix, hourly_stats, daily_stats):
    """Discover actionable patterns / learned rules."""
    rules = []

    # Bad regimes
    for strat, regimes in regime_matrix.items():
        for regime, data in regimes.items():
            if data['trades'] >= 5:
                wr = data['wins'] / data['trades'] * 100
                if wr < 40:
                    rules.append({
                        'type': 'bad_regime',
                        'strategy': strat,
                        'regime': regime,
                        'detail': f"WR {wr:.0f}% in {regime} ({data['trades']} trades)",
                    })

    # Bad hours
    for strat, hours in hourly_stats.items():
        for hour, data in hours.items():
            if data['trades'] >= 3:
                wr = data['wins'] / data['trades'] * 100
                if wr < 35:
                    rules.append({
                        'type': 'bad_hour',
                        'strategy': strat,
                        'hour': hour,
                        'detail': f"WR {wr:.0f}% at {hour}:00 ET ({data['trades']} trades)",
                    })

    # Bad days
    for strat, days in daily_stats.items():
        for day, data in days.items():
            if data['trades'] >= 3:
                wr = data['wins'] / data['trades'] * 100
                if wr < 35:
                    rules.append({
                        'type': 'bad_day',
                        'strategy': strat,
                        'day': day,
                        'detail': f"WR {wr:.0f}% on {day} ({data['trades']} trades)",
                    })

    # Consecutive losses per strategy
    by_strat = defaultdict(list)
    for t in sorted(trades, key=lambda x: x.get('exit_time') or 0):
        strat = t.get('strategy') or 'UNKNOWN'
        by_strat[strat].append(t)

    for strat, strat_trades in by_strat.items():
        max_consec = 0
        current = 0
        for t in strat_trades:
            if (t.get('pnl_usd') or 0) <= 0:
                current += 1
                max_consec = max(max_consec, current)
            else:
                current = 0
        if max_consec >= 5:
            rules.append({
                'type': 'consecutive_losses',
                'strategy': strat,
                'detail': f"Max {max_consec} consecutive losses detected",
            })

    # Edge decay — compare WR of first half vs second half
    for strat, strat_trades in by_strat.items():
        if len(strat_trades) < 10:
            continue
        mid = len(strat_trades) // 2
        first_half = strat_trades[:mid]
        second_half = strat_trades[mid:]

        wr1 = len([t for t in first_half if (t.get('pnl_usd') or 0) > 0]) / len(first_half) * 100
        wr2 = len([t for t in second_half if (t.get('pnl_usd') or 0) > 0]) / len(second_half) * 100

        if wr1 - wr2 > 15:
            rules.append({
                'type': 'edge_decay',
                'strategy': strat,
                'detail': f"WR dropped from {wr1:.0f}% to {wr2:.0f}% (first→second half)",
            })

    return rules


def analyze_parameters(trades, bot_params):
    """Analyze TP/SL effectiveness, generate recommendations."""
    recommendations = []

    by_strat = defaultdict(list)
    for t in trades:
        strat = t.get('strategy') or 'UNKNOWN'
        by_strat[strat].append(t)

    for strat, strat_trades in by_strat.items():
        if len(strat_trades) < 10:
            continue

        # Find the matching bot name for this strategy
        matching_bot = None
        for bot_name, bot_strat in BOT_STRATEGIES.items():
            if bot_strat == strat:
                matching_bot = bot_name
                break

        if not matching_bot or matching_bot not in bot_params:
            continue

        current_tp = bot_params[matching_bot]['tp_pct']
        current_sl = bot_params[matching_bot]['sl_pct']

        # SL hit rate
        sl_hits = [t for t in strat_trades
                   if t.get('exit_reason') and 'stop' in t['exit_reason'].lower()]
        sl_rate = len(sl_hits) / len(strat_trades) * 100 if strat_trades else 0

        # TP capture rate
        tp_hits = [t for t in strat_trades
                   if t.get('exit_reason') and ('take profit' in t['exit_reason'].lower()
                                                 or 'limit' in t['exit_reason'].lower())]
        tp_rate = len(tp_hits) / len(strat_trades) * 100 if strat_trades else 0

        # If SL hit too often (>60%), suggest widening
        if sl_rate > 60 and len(strat_trades) >= 15:
            new_sl = min(current_sl * 1.15, current_sl * 1.2)  # +15-20%
            recommendations.append({
                'strategy': strat,
                'param': 'sl_pct',
                'current': current_sl,
                'recommended': round(new_sl, 2),
                'reason': f"SL hit rate {sl_rate:.0f}% — widening SL",
            })

        # If TP rarely hit (<20%) and most exits are SL, suggest tightening TP
        if tp_rate < 20 and sl_rate > 40 and len(strat_trades) >= 15:
            new_tp = max(current_tp * 0.85, current_tp * 0.8)  # -15-20%
            recommendations.append({
                'strategy': strat,
                'param': 'tp_pct',
                'current': current_tp,
                'recommended': round(new_tp, 2),
                'reason': f"TP hit rate only {tp_rate:.0f}% — tightening TP",
            })

    return recommendations


def write_skill_file(stats, regime_matrix, hourly_stats, daily_stats,
                     rules, recommendations, cumulative, dry_run=False):
    """Write the skill file markdown."""
    lines = []
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M UTC')

    lines.append("# Stock Strategy Performance")
    lines.append(f"_Auto-generated by skill_updater.py — {now_str}_\n")

    # Active Strategies
    lines.append("## Active Strategies\n")
    lines.append("| Strategy | Trades | Win Rate | Total PnL | Avg PnL | PF | Avg Hold |")
    lines.append("|----------|--------|----------|-----------|---------|----|----------|")
    for strat, s in sorted(stats.items()):
        pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] != float('inf') else "inf"
        lines.append(
            f"| {strat} | {s['trades']} | {s['win_rate']:.1f}% | "
            f"${s['total_pnl']:+.2f} | ${s['avg_pnl']:+.2f} | "
            f"{pf_str} | {s['avg_duration_h']:.1f}h |"
        )

    # Regime Matrix
    lines.append("\n## Regime Matrix\n")
    regimes_seen = set()
    for strat_regimes in regime_matrix.values():
        regimes_seen.update(strat_regimes.keys())
    regimes_sorted = sorted(regimes_seen)

    header = "| Strategy | " + " | ".join(regimes_sorted) + " |"
    sep = "|----------|" + "|".join(["--------"] * len(regimes_sorted)) + "|"
    lines.append(header)
    lines.append(sep)

    for strat in sorted(regime_matrix.keys()):
        cells = []
        for regime in regimes_sorted:
            data = regime_matrix[strat].get(regime, {'trades': 0, 'wins': 0})
            if data['trades'] == 0:
                cells.append("—")
            else:
                wr = data['wins'] / data['trades'] * 100
                cells.append(f"{wr:.0f}% (n={data['trades']})")
        lines.append(f"| {strat} | " + " | ".join(cells) + " |")

    # Time-of-Day (ET)
    lines.append("\n## Time-of-Day (ET)\n")
    market_hours = range(10, 17)  # 10 AM to 4 PM ET
    h_header = "| Strategy | " + " | ".join(f"{h}:00" for h in market_hours) + " |"
    h_sep = "|----------|" + "|".join(["------"] * len(list(market_hours))) + "|"
    lines.append(h_header)
    lines.append(h_sep)

    for strat in sorted(hourly_stats.keys()):
        cells = []
        for h in market_hours:
            data = hourly_stats[strat].get(h, {'trades': 0, 'wins': 0})
            if data['trades'] == 0:
                cells.append("—")
            else:
                wr = data['wins'] / data['trades'] * 100
                cells.append(f"{wr:.0f}%/{data['trades']}")
        lines.append(f"| {strat} | " + " | ".join(cells) + " |")

    # Day-of-Week
    lines.append("\n## Day-of-Week\n")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    d_header = "| Strategy | " + " | ".join(days_order) + " |"
    d_sep = "|----------|" + "|".join(["--------"] * len(days_order)) + "|"
    lines.append(d_header)
    lines.append(d_sep)

    for strat in sorted(daily_stats.keys()):
        cells = []
        for day in days_order:
            data = daily_stats[strat].get(day, {'trades': 0, 'wins': 0})
            if data['trades'] == 0:
                cells.append("—")
            else:
                wr = data['wins'] / data['trades'] * 100
                cells.append(f"{wr:.0f}%/{data['trades']}")
        lines.append(f"| {strat} | " + " | ".join(cells) + " |")

    # Learned Rules
    lines.append("\n## Learned Rules\n")
    if rules:
        for r in rules:
            lines.append(f"- **{r['type']}** [{r.get('strategy', '')}]: {r['detail']}")
    else:
        lines.append("_No patterns discovered yet (need more trades)._")

    # Parameter Recommendations
    lines.append("\n## Parameter Recommendations\n")
    if recommendations:
        lines.append("| Strategy | Param | Current | Recommended | Reason |")
        lines.append("|----------|-------|---------|-------------|--------|")
        for r in recommendations:
            lines.append(
                f"| {r['strategy']} | {r['param']} | {r['current']} | "
                f"{r['recommended']} | {r['reason']} |"
            )
    else:
        lines.append("_No parameter changes recommended._")

    # Cumulative Stats
    lines.append("\n## Cumulative Stats\n")
    lines.append(f"- Total trades: {cumulative['total_trades']}")
    lines.append(f"- Total PnL: ${cumulative['total_pnl']:+.2f}")
    lines.append(f"- Overall Win Rate: {cumulative['win_rate']:.1f}%")
    lines.append(f"- Best trade: ${cumulative['best']:+.2f}")
    lines.append(f"- Worst trade: ${cumulative['worst']:+.2f}")

    # Update History
    lines.append("\n## Update History\n")
    lines.append(f"- {now_str}: {cumulative['total_trades']} trades analyzed, "
                 f"{len(rules)} rules, {len(recommendations)} param recs")

    content = "\n".join(lines) + "\n"

    if dry_run:
        print(content)
        return content

    SKILL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SKILL_FILE, 'w') as f:
        f.write(content)
    print(f"Wrote skill file: {SKILL_FILE}")
    return content


def main():
    parser = argparse.ArgumentParser(description="Stock Skill Updater")
    parser.add_argument('--force', action='store_true',
                        help='Run even with fewer than threshold trades')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print output but do not write files')
    parser.add_argument('--threshold', type=int, default=10,
                        help='Minimum trades to run analysis (default: 10)')
    args = parser.parse_args()

    journal = StockJournal()
    trades = journal.get_closed_trades()

    if not trades:
        print("No closed trades found in SQLite journal.")
        if not args.force:
            print("Run scripts/migrate_csv.py first, or use --force.")
            return
        trades = []

    print(f"Loaded {len(trades)} closed trades")

    if len(trades) < args.threshold and not args.force:
        print(f"Need at least {args.threshold} trades (have {len(trades)}). Use --force to override.")
        return

    # Compute all stats
    stats = compute_strategy_stats(trades)
    regime_matrix = compute_regime_matrix(trades)
    hourly_stats = compute_hourly_stats(trades)
    daily_stats = compute_daily_stats(trades)
    rules = discover_patterns(trades, regime_matrix, hourly_stats, daily_stats)

    # Parameter analysis
    bot_params = load_bot_params()
    recommendations = analyze_parameters(trades, bot_params)

    # Cumulative stats
    pnls = [t['pnl_usd'] for t in trades if t.get('pnl_usd') is not None]
    cumulative = {
        'total_trades': len(pnls),
        'total_pnl': sum(pnls) if pnls else 0,
        'win_rate': len([p for p in pnls if p > 0]) / len(pnls) * 100 if pnls else 0,
        'best': max(pnls) if pnls else 0,
        'worst': min(pnls) if pnls else 0,
    }

    # Write skill file
    write_skill_file(stats, regime_matrix, hourly_stats, daily_stats,
                     rules, recommendations, cumulative, dry_run=args.dry_run)

    # Store recommendations in SQLite
    if recommendations and not args.dry_run:
        for rec in recommendations:
            journal.add_parameter_recommendation(
                strategy=rec['strategy'],
                param_name=rec['param'],
                current_value=rec['current'],
                recommended_value=rec['recommended'],
                reason=rec['reason'],
            )
        print(f"Stored {len(recommendations)} parameter recommendations")

    # Log the skill update
    if not args.dry_run:
        journal.log_skill_update(
            trades_analyzed=len(trades),
            rules_discovered=len(rules),
            params_recommended=len(recommendations),
            summary=f"{len(stats)} strategies, {len(rules)} rules, {len(recommendations)} param recs",
        )

    print(f"\nSummary: {len(stats)} strategies, {len(rules)} rules, "
          f"{len(recommendations)} recommendations")


if __name__ == '__main__':
    main()

"""
Trade Journal Analyzer — Enhanced Daily & Weekly Reports

Reads trade_journal.csv and produces:
- Daily report with per-bot breakdown, alerts, and recommendations
- Weekly summary with strategy comparison and RBI correlation
- Filtered analysis by bot, strategy, symbol, or date range

Reports saved to: ~/trading-bot/reports/daily_YYYY-MM-DD.md
                  ~/trading-bot/reports/weekly_YYYY-WXX.md

Usage:
    python3 src/analysis/journal_analyzer.py                  # Daily report for today
    python3 src/analysis/journal_analyzer.py --weekly         # Weekly summary
    python3 src/analysis/journal_analyzer.py --date 2026-01-29  # Specific day
    python3 src/analysis/journal_analyzer.py --bot momentum_bot # Single bot analysis
    python3 src/analysis/journal_analyzer.py --days 7         # Last N days
    python3 src/analysis/journal_analyzer.py --output-report  # Legacy markdown output
    python3 src/analysis/journal_analyzer.py --daily          # Generate + save daily report
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta, date
from pathlib import Path

from termcolor import cprint

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
JOURNAL_PATH = PROJECT_ROOT / 'csvs' / 'trade_journal.csv'
REPORT_DIR = PROJECT_ROOT / 'reports'

# Backtest expectations (from RBI / ideas.txt) — used for alerts
BACKTEST_EXPECTATIONS = {
    'MOMENTUM':  {'win_rate': 50, 'label': 'momentum_bot'},
    'MEAN_REV':  {'win_rate': 50, 'label': 'mean_reversion_bot'},
    'BREAKOUT':  {'win_rate': 40, 'label': 'breakout_bot'},
    'MACD':      {'win_rate': 50, 'label': 'macd_bot'},
    'BB_BOUNCE': {'win_rate': 50, 'label': 'bb_bounce_bot'},
    'GAP_GO':    {'win_rate': 50, 'label': 'gap_and_go_bot'},
}


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def load_trades(days=None, strategy=None, symbol=None, bot_name=None,
                date_str=None, date_start=None, date_end=None):
    """Load trades from journal CSV with optional filters."""
    if not JOURNAL_PATH.exists():
        cprint("No trade journal found at: " + str(JOURNAL_PATH), "red")
        return []

    cutoff = None
    if days:
        cutoff = datetime.now() - timedelta(days=days)

    # Date-specific filter
    target_date = None
    if date_str:
        try:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            cprint(f"Invalid date format: {date_str} (use YYYY-MM-DD)", "red")
            return []

    trades = []
    with open(JOURNAL_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['_timestamp'] = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            except (ValueError, KeyError):
                continue

            if cutoff and row['_timestamp'] < cutoff:
                continue
            if target_date and row['_timestamp'].date() != target_date:
                continue
            if date_start and row['_timestamp'].date() < date_start:
                continue
            if date_end and row['_timestamp'].date() > date_end:
                continue
            if strategy and row.get('strategy', '') != strategy:
                continue
            if symbol and row.get('symbol', '') != symbol:
                continue
            if bot_name and row.get('bot_name', '') != bot_name:
                continue

            # Parse numeric fields
            for field in ('price', 'shares', 'entry_price', 'exit_price',
                          'stop_loss', 'take_profit', 'pnl', 'pnl_pct',
                          'confidence', 'slippage', 'r_multiple'):
                val = row.get(field, '')
                if val and val.strip():
                    try:
                        row[field] = float(val)
                    except ValueError:
                        row[field] = None
                else:
                    row[field] = None

            trades.append(row)

    return trades


def split_entries_exits(trades):
    """Split trades into entries and exits."""
    entries = [t for t in trades if t.get('action') == 'ENTRY']
    exits = [t for t in trades if t.get('action') == 'EXIT']
    return entries, exits


# ─────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────

def overall_stats(exits):
    """Compute overall performance metrics."""
    if not exits:
        return None

    pnls = [t['pnl'] for t in exits if t['pnl'] is not None]
    pnl_pcts = [t['pnl_pct'] for t in exits if t['pnl_pct'] is not None]
    r_mults = [t['r_multiple'] for t in exits if t.get('r_multiple') is not None]

    if not pnls:
        return None

    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]

    return {
        'total_trades': len(pnls),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(pnls) * 100 if pnls else 0,
        'total_pnl': sum(pnls),
        'avg_pnl': sum(pnls) / len(pnls),
        'avg_pnl_pct': sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0,
        'best_trade': max(pnls),
        'worst_trade': min(pnls),
        'best_pct': max(pnl_pcts) if pnl_pcts else 0,
        'worst_pct': min(pnl_pcts) if pnl_pcts else 0,
        'avg_r_multiple': sum(r_mults) / len(r_mults) if r_mults else None,
        'profit_factor': (
            sum(winners) / abs(sum(losers))
            if losers and sum(losers) != 0 else float('inf') if winners else 0
        ),
    }


def stats_by_group(exits, group_key):
    """Compute win rate and P&L grouped by a field (strategy, symbol, regime, bot_name)."""
    groups = defaultdict(list)
    for t in exits:
        key = t.get(group_key, 'UNKNOWN') or 'UNKNOWN'
        if t['pnl'] is not None:
            groups[key].append(t)

    results = {}
    for key, trades in sorted(groups.items()):
        pnls = [t['pnl'] for t in trades]
        pnl_pcts = [t['pnl_pct'] for t in trades if t['pnl_pct'] is not None]
        r_mults = [t['r_multiple'] for t in trades if t.get('r_multiple') is not None]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        results[key] = {
            'trades': len(pnls),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(pnls) * 100 if pnls else 0,
            'total_pnl': sum(pnls),
            'avg_pnl': sum(pnls) / len(pnls) if pnls else 0,
            'avg_pnl_pct': sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0,
            'avg_r': sum(r_mults) / len(r_mults) if r_mults else None,
            'profit_factor': (
                sum(winners) / abs(sum(losers))
                if losers and sum(losers) != 0 else float('inf') if winners else 0
            ),
        }

    return results


def generate_alerts(strat_stats):
    """Generate alerts comparing live performance vs backtest expectations."""
    alerts = []
    for strategy, expected in BACKTEST_EXPECTATIONS.items():
        live = strat_stats.get(strategy)
        if live is None or live['trades'] < 3:
            continue

        expected_wr = expected['win_rate']
        live_wr = live['win_rate']
        bot_label = expected['label']

        if live_wr < expected_wr - 10:
            alerts.append({
                'type': 'warning',
                'msg': f"{bot_label} win rate ({live_wr:.0f}%) below backtest expectation ({expected_wr}%)",
            })
        elif live_wr >= expected_wr:
            alerts.append({
                'type': 'ok',
                'msg': f"{bot_label} performing as expected ({live_wr:.0f}% WR vs {expected_wr}% expected)",
            })

        if live['total_pnl'] < 0 and live['trades'] >= 5:
            alerts.append({
                'type': 'warning',
                'msg': f"{bot_label} net negative (${live['total_pnl']:+,.2f}) over {live['trades']} trades",
            })

    return alerts


def generate_recommendations(strat_stats, overall):
    """Generate recommendations based on performance data."""
    recs = []

    if not overall or not strat_stats:
        return recs

    # Find consistent winners (WR > 60% and PF > 1.5 with 5+ trades)
    for name, s in strat_stats.items():
        if s['trades'] >= 5 and s['win_rate'] > 60 and s['profit_factor'] > 1.5:
            recs.append(f"Consider increasing position size for {name} (consistent winner: {s['win_rate']:.0f}% WR, {s['profit_factor']:.1f} PF)")

    # Find underperformers (WR < 40% with 5+ trades)
    for name, s in strat_stats.items():
        if s['trades'] >= 5 and s['win_rate'] < 40:
            recs.append(f"Review {name} entry conditions (underperforming: {s['win_rate']:.0f}% WR over {s['trades']} trades)")

    # No trades from a bot
    active_strategies = set(strat_stats.keys())
    for strategy in BACKTEST_EXPECTATIONS:
        if strategy not in active_strategies:
            label = BACKTEST_EXPECTATIONS[strategy]['label']
            recs.append(f"{label} has 0 trades — verify signal conditions are reachable")

    return recs


# ─────────────────────────────────────────────────────────────
# Terminal output
# ─────────────────────────────────────────────────────────────

def print_report(trades, entries, exits, title="TRADE JOURNAL ANALYSIS"):
    """Print analysis report to terminal."""
    cprint("\n" + "=" * 70, "cyan")
    cprint(f"  {title}", "cyan", attrs=["bold"])
    cprint(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}", "cyan")
    cprint("=" * 70, "cyan")

    stats = overall_stats(exits)
    if not stats:
        cprint("\n  No completed trades (exits) found.\n", "yellow")
        cprint(f"  Total entries logged: {len(entries)}", "white")
        cprint("=" * 70 + "\n", "cyan")
        return stats

    cprint("\n  OVERALL PERFORMANCE", "white", attrs=["bold"])
    cprint("  " + "-" * 50, "white")

    color = "green" if stats['total_pnl'] > 0 else "red"
    cprint(f"  Total P&L:        ${stats['total_pnl']:>+10,.2f}", color)
    cprint(f"  Total Trades:     {stats['total_trades']:>10}", "white")
    cprint(f"  Win Rate:         {stats['win_rate']:>9.1f}%  ({stats['winners']}W / {stats['losers']}L)", color)
    cprint(f"  Avg P&L/Trade:    ${stats['avg_pnl']:>+10,.2f}  ({stats['avg_pnl_pct']:+.2f}%)", color)
    cprint(f"  Best Trade:       ${stats['best_trade']:>+10,.2f}  ({stats['best_pct']:+.2f}%)", "green")
    cprint(f"  Worst Trade:      ${stats['worst_trade']:>+10,.2f}  ({stats['worst_pct']:+.2f}%)", "red")

    pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
    cprint(f"  Profit Factor:    {pf_str:>10}", "green" if stats['profit_factor'] > 1 else "red")

    if stats['avg_r_multiple'] is not None:
        r_color = "green" if stats['avg_r_multiple'] > 0 else "red"
        cprint(f"  Avg R-Multiple:   {stats['avg_r_multiple']:>+10.2f}R", r_color)
    else:
        cprint(f"  Avg R-Multiple:         N/A", "yellow")

    # By Strategy
    strat_stats = stats_by_group(exits, 'strategy')
    if strat_stats:
        cprint("\n  BY STRATEGY", "white", attrs=["bold"])
        cprint("  " + "-" * 66, "white")
        cprint(f"  {'Strategy':<12} {'Trades':>6} {'WR':>7} {'Total P&L':>12} {'Avg P&L':>10} {'PF':>6}", "white")
        cprint("  " + "-" * 66, "white")
        for name, s in sorted(strat_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            c = "green" if s['total_pnl'] > 0 else "red"
            pf = f"{s['profit_factor']:.1f}" if s['profit_factor'] != float('inf') else "INF"
            cprint(
                f"  {name:<12} {s['trades']:>6} {s['win_rate']:>6.1f}% "
                f"${s['total_pnl']:>+10,.2f} ${s['avg_pnl']:>+8,.2f} {pf:>6}",
                c,
            )

    # By Bot Name
    bot_stats = stats_by_group(exits, 'bot_name')
    if bot_stats and any(k != 'UNKNOWN' for k in bot_stats):
        cprint("\n  BY BOT", "white", attrs=["bold"])
        cprint("  " + "-" * 66, "white")
        cprint(f"  {'Bot':<22} {'Trades':>6} {'WR':>7} {'Total P&L':>12} {'Avg P&L':>10}", "white")
        cprint("  " + "-" * 66, "white")
        for name, s in sorted(bot_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            c = "green" if s['total_pnl'] > 0 else "red"
            cprint(
                f"  {name:<22} {s['trades']:>6} {s['win_rate']:>6.1f}% "
                f"${s['total_pnl']:>+10,.2f} ${s['avg_pnl']:>+8,.2f}",
                c,
            )

    # By Symbol
    sym_stats = stats_by_group(exits, 'symbol')
    if sym_stats:
        cprint("\n  BY SYMBOL", "white", attrs=["bold"])
        cprint("  " + "-" * 66, "white")
        for name, s in sorted(sym_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            c = "green" if s['total_pnl'] > 0 else "red"
            cprint(
                f"  {name:<8} {s['trades']:>6} {s['win_rate']:>6.1f}% "
                f"${s['total_pnl']:>+10,.2f} {s['avg_pnl_pct']:>+7.2f}%",
                c,
            )

    # Alerts
    alerts = generate_alerts(strat_stats)
    if alerts:
        cprint("\n  ALERTS", "white", attrs=["bold"])
        cprint("  " + "-" * 50, "white")
        for a in alerts:
            if a['type'] == 'warning':
                cprint(f"  !! {a['msg']}", "yellow")
            else:
                cprint(f"  OK {a['msg']}", "green")

    # Recommendations
    recs = generate_recommendations(strat_stats, stats)
    if recs:
        cprint("\n  RECOMMENDATIONS", "white", attrs=["bold"])
        cprint("  " + "-" * 50, "white")
        for r in recs:
            cprint(f"  -> {r}", "cyan")

    cprint("\n" + "=" * 70 + "\n", "cyan")
    return stats


# ─────────────────────────────────────────────────────────────
# Daily Report (markdown)
# ─────────────────────────────────────────────────────────────

def generate_daily_report(target_date=None):
    """Generate daily markdown report for a specific date."""
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()

    date_str = target_date.strftime('%Y-%m-%d')
    trades = load_trades(date_str=date_str)
    entries, exits = split_entries_exits(trades)

    stats = overall_stats(exits)

    lines = [
        f"# Daily Trading Report - {date_str}",
        "",
    ]

    # Summary
    lines.append("## Summary")
    if stats:
        lines.extend([
            f"- Total P&L: ${stats['total_pnl']:+,.2f}",
            f"- Trades: {stats['total_trades']} ({stats['winners']} wins, {stats['losers']} losses)",
            f"- Win Rate: {stats['win_rate']:.0f}%",
            f"- Profit Factor: {stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else f"- Profit Factor: INF",
            "",
        ])
    else:
        lines.extend([
            f"- Total P&L: $0.00",
            f"- Trades: 0 exits ({len(entries)} entries pending)",
            "",
        ])

    # By Bot / Strategy
    strat_stats = stats_by_group(exits, 'strategy')
    bot_stats = stats_by_group(exits, 'bot_name')

    # Prefer bot_name grouping if available, fall back to strategy
    use_bot = bot_stats and any(k != 'UNKNOWN' for k in bot_stats)
    group_data = bot_stats if use_bot else strat_stats
    group_label = "Bot" if use_bot else "Strategy"

    if group_data:
        lines.extend([
            f"## By {group_label}",
            "",
            f"| {group_label} | Trades | Wins | Losses | P&L | Win Rate |",
            f"|{'---' * 5}|--------|------|--------|-----|----------|",
        ])
        for name, s in sorted(group_data.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            lines.append(
                f"| {name} | {s['trades']} | {s['winners']} | {s['losers']} | "
                f"${s['total_pnl']:+,.2f} | {s['win_rate']:.0f}% |"
            )
        lines.append("")

    # Alerts
    alerts = generate_alerts(strat_stats)
    if alerts:
        lines.append("## Alerts")
        for a in alerts:
            if a['type'] == 'warning':
                lines.append(f"- !! {a['msg']}")
            else:
                lines.append(f"- OK {a['msg']}")
        lines.append("")

    # Recommendations
    recs = generate_recommendations(strat_stats, stats)
    if recs:
        lines.append("## Recommendations")
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")

    # Trade Log
    if trades:
        lines.extend([
            "## Trade Log",
            "",
            "| Time | Action | Bot | Symbol | Strategy | Direction | Price | P&L |",
            "|------|--------|-----|--------|----------|-----------|-------|-----|",
        ])
        for t in trades:
            time_str = t['_timestamp'].strftime('%H:%M:%S')
            action = t.get('action', '')
            bot = t.get('bot_name', '') or ''
            sym = t.get('symbol', '')
            strat = t.get('strategy', '')
            direction = t.get('direction', '')
            price = t['price']
            price_str = f"${price:.2f}" if price else ''
            pnl = t['pnl']
            pnl_str = f"${pnl:+,.2f}" if pnl is not None else ''
            lines.append(
                f"| {time_str} | {action} | {bot} | {sym} | {strat} | "
                f"{direction} | {price_str} | {pnl_str} |"
            )
        lines.append("")

    lines.extend([
        "---",
        f"*Generated by journal_analyzer.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ])

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Weekly Report (markdown)
# ─────────────────────────────────────────────────────────────

def generate_weekly_report(target_date=None):
    """Generate weekly summary report (Mon-Fri of the given week)."""
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = datetime.strptime(target_date, '%Y-%m-%d').date()

    # Get Monday-Friday of the week
    weekday = target_date.weekday()  # 0=Mon
    monday = target_date - timedelta(days=weekday)
    friday = monday + timedelta(days=4)

    iso_year, iso_week, _ = target_date.isocalendar()
    week_label = f"{iso_year}-W{iso_week:02d}"

    trades = load_trades(date_start=monday, date_end=friday)
    entries, exits = split_entries_exits(trades)
    stats = overall_stats(exits)

    lines = [
        f"# Weekly Trading Summary - {week_label}",
        f"",
        f"**Period:** {monday.strftime('%Y-%m-%d')} to {friday.strftime('%Y-%m-%d')}  ",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Week summary
    lines.append("## Week Summary")
    if stats:
        pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
        lines.extend([
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total P&L | ${stats['total_pnl']:+,.2f} |",
            f"| Trades | {stats['total_trades']} ({stats['winners']}W / {stats['losers']}L) |",
            f"| Win Rate | {stats['win_rate']:.1f}% |",
            f"| Avg P&L/Trade | ${stats['avg_pnl']:+,.2f} ({stats['avg_pnl_pct']:+.2f}%) |",
            f"| Best Trade | ${stats['best_trade']:+,.2f} ({stats['best_pct']:+.2f}%) |",
            f"| Worst Trade | ${stats['worst_trade']:+,.2f} ({stats['worst_pct']:+.2f}%) |",
            f"| Profit Factor | {pf_str} |",
            "",
        ])
    else:
        lines.extend([
            f"- No completed trades this week ({len(entries)} entries pending)",
            "",
        ])

    # Strategy comparison
    strat_stats = stats_by_group(exits, 'strategy')
    if strat_stats:
        lines.extend([
            "## Strategy Comparison",
            "",
            "| Strategy | Trades | Win Rate | Total P&L | Avg P&L | PF |",
            "|----------|--------|----------|-----------|---------|-----|",
        ])
        for name, s in sorted(strat_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            pf = f"{s['profit_factor']:.1f}" if s['profit_factor'] != float('inf') else "INF"
            lines.append(
                f"| {name} | {s['trades']} | {s['win_rate']:.0f}% | "
                f"${s['total_pnl']:+,.2f} | ${s['avg_pnl']:+,.2f} | {pf} |"
            )
        lines.append("")

    # Bot comparison
    bot_stats = stats_by_group(exits, 'bot_name')
    if bot_stats and any(k != 'UNKNOWN' for k in bot_stats):
        lines.extend([
            "## Bot Comparison",
            "",
            "| Bot | Trades | Win Rate | Total P&L | Avg P&L |",
            "|-----|--------|----------|-----------|---------|",
        ])
        for name, s in sorted(bot_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            lines.append(
                f"| {name} | {s['trades']} | {s['win_rate']:.0f}% | "
                f"${s['total_pnl']:+,.2f} | ${s['avg_pnl']:+,.2f} |"
            )
        lines.append("")

    # Symbol performance
    sym_stats = stats_by_group(exits, 'symbol')
    if sym_stats:
        lines.extend([
            "## Symbol Performance",
            "",
            "| Symbol | Trades | Win Rate | Total P&L | Avg % |",
            "|--------|--------|----------|-----------|-------|",
        ])
        for name, s in sorted(sym_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
            lines.append(
                f"| {name} | {s['trades']} | {s['win_rate']:.0f}% | "
                f"${s['total_pnl']:+,.2f} | {s['avg_pnl_pct']:+.2f}% |"
            )
        lines.append("")

    # Daily breakdown
    daily_data = defaultdict(lambda: {'pnl': 0.0, 'trades': 0, 'wins': 0})
    for t in exits:
        if t['pnl'] is not None:
            day = t['_timestamp'].strftime('%A %m/%d')
            daily_data[day]['pnl'] += t['pnl']
            daily_data[day]['trades'] += 1
            if t['pnl'] > 0:
                daily_data[day]['wins'] += 1

    if daily_data:
        lines.extend([
            "## Daily Breakdown",
            "",
            "| Day | Trades | Wins | P&L |",
            "|-----|--------|------|-----|",
        ])
        for day, d in daily_data.items():
            wr = d['wins'] / d['trades'] * 100 if d['trades'] > 0 else 0
            lines.append(
                f"| {day} | {d['trades']} | {d['wins']} ({wr:.0f}%) | ${d['pnl']:+,.2f} |"
            )
        lines.append("")

    # RBI correlation
    rbi_ideas_path = PROJECT_ROOT / 'rbi' / 'ideas.txt'
    if rbi_ideas_path.exists():
        lines.extend([
            "## RBI Backtest Correlation",
            "",
            "| Strategy | Backtest Status | Live Trades | Live WR | Aligned? |",
            "|----------|----------------|-------------|---------|----------|",
        ])
        # Parse ideas.txt for status
        idea_status = _parse_ideas_status(rbi_ideas_path)
        for strategy, expected in BACKTEST_EXPECTATIONS.items():
            live = strat_stats.get(strategy)
            bt_status = idea_status.get(strategy.lower(), 'unknown')
            if live:
                expected_wr = expected['win_rate']
                aligned = "Yes" if live['win_rate'] >= expected_wr - 10 else "No"
                lines.append(
                    f"| {strategy} | {bt_status} | {live['trades']} | "
                    f"{live['win_rate']:.0f}% | {aligned} |"
                )
            else:
                lines.append(
                    f"| {strategy} | {bt_status} | 0 | N/A | N/A |"
                )
        lines.append("")

    # Alerts & Recommendations
    alerts = generate_alerts(strat_stats)
    recs = generate_recommendations(strat_stats, stats)

    if alerts:
        lines.append("## Alerts")
        for a in alerts:
            prefix = "!!" if a['type'] == 'warning' else "OK"
            lines.append(f"- {prefix} {a['msg']}")
        lines.append("")

    if recs:
        lines.append("## Recommendations for Next Week")
        for r in recs:
            lines.append(f"- {r}")
        lines.append("")

    lines.extend([
        "---",
        f"*Generated by journal_analyzer.py on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ])

    return "\n".join(lines)


def _parse_ideas_status(ideas_path):
    """Parse strategy statuses from rbi/ideas.txt."""
    statuses = {}
    current_name = None
    try:
        with open(ideas_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('name:'):
                    current_name = line.split(':', 1)[1].strip()
                elif line.startswith('status:') and current_name:
                    status = line.split(':', 1)[1].strip()
                    # Map idea names to strategy keys
                    name_map = {
                        'momentum_pullback': 'momentum',
                        'mean_reversion': 'mean_rev',
                        'breakout_moondev': 'breakout',
                        'macd_bullish': 'macd',
                        'bb_bounce': 'bb_bounce',
                        'gap_and_go': 'gap_go',
                    }
                    key = name_map.get(current_name, current_name)
                    statuses[key] = status
                    current_name = None
    except Exception:
        pass
    return statuses


# ─────────────────────────────────────────────────────────────
# Legacy markdown (backwards compatible)
# ─────────────────────────────────────────────────────────────

def generate_markdown_legacy(trades, entries, exits, days=None):
    """Generate the legacy weekly_analysis.md report."""
    period = f"Last {days} days" if days else "All time"
    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    lines = [
        f"# Trade Journal Analysis",
        f"",
        f"**Period:** {period}  ",
        f"**Generated:** {now}  ",
        f"**Source:** `{JOURNAL_PATH}`",
        f"",
    ]

    stats = overall_stats(exits)
    if not stats:
        lines.append("No completed trades found.")
        return "\n".join(lines)

    pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float('inf') else "INF"
    r_str = f"{stats['avg_r_multiple']:+.2f}R" if stats['avg_r_multiple'] is not None else "N/A"

    lines.extend([
        "## Overall Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total P&L | ${stats['total_pnl']:+,.2f} |",
        f"| Total Trades | {stats['total_trades']} |",
        f"| Win Rate | {stats['win_rate']:.1f}% ({stats['winners']}W / {stats['losers']}L) |",
        f"| Avg P&L/Trade | ${stats['avg_pnl']:+,.2f} ({stats['avg_pnl_pct']:+.2f}%) |",
        f"| Best Trade | ${stats['best_trade']:+,.2f} ({stats['best_pct']:+.2f}%) |",
        f"| Worst Trade | ${stats['worst_trade']:+,.2f} ({stats['worst_pct']:+.2f}%) |",
        f"| Profit Factor | {pf_str} |",
        f"| Avg R-Multiple | {r_str} |",
        "",
    ])

    for group_key, title in [('strategy', 'Strategy'), ('symbol', 'Symbol'), ('regime', 'Regime')]:
        group_stats = stats_by_group(exits, group_key)
        if group_stats:
            if group_key == 'regime':
                lines.extend([
                    f"## Performance by {title}",
                    "",
                    f"| {title} | Trades | Win Rate | Total P&L |",
                    f"|{'---'*5}|--------|----------|-----------|",
                ])
                for name, s in sorted(group_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
                    lines.append(f"| {name} | {s['trades']} | {s['win_rate']:.1f}% | ${s['total_pnl']:+,.2f} |")
            else:
                r_header = "Avg R" if group_key == 'strategy' else "Avg %"
                lines.extend([
                    f"## Performance by {title}",
                    "",
                    f"| {title} | Trades | Win Rate | Total P&L | Avg P&L | {r_header} |",
                    f"|{'---'*5}|--------|----------|-----------|---------|-------|",
                ])
                for name, s in sorted(group_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True):
                    if group_key == 'strategy':
                        r_val = f"{s['avg_r']:+.1f}R" if s['avg_r'] is not None else "N/A"
                    else:
                        r_val = f"{s['avg_pnl_pct']:+.2f}%"
                    lines.append(
                        f"| {name} | {s['trades']} | {s['win_rate']:.1f}% | "
                        f"${s['total_pnl']:+,.2f} | ${s['avg_pnl']:+,.2f} | {r_val} |"
                    )
            lines.append("")

    lines.extend(["---", f"*Generated by journal_analyzer.py on {now}*"])
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Trade Journal Analyzer")
    parser.add_argument("--days", type=int, default=None, help="Analyze last N days (default: all)")
    parser.add_argument("--date", type=str, default=None, help="Specific date YYYY-MM-DD")
    parser.add_argument("--strategy", type=str, default=None, help="Filter by strategy")
    parser.add_argument("--symbol", type=str, default=None, help="Filter by symbol")
    parser.add_argument("--bot", type=str, default=None, help="Filter by bot name")
    parser.add_argument("--daily", action="store_true", help="Generate daily report (saves to reports/)")
    parser.add_argument("--weekly", action="store_true", help="Generate weekly report (saves to reports/)")
    parser.add_argument("--output-report", action="store_true", help="Legacy: write weekly_analysis.md")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Daily report mode
    if args.daily:
        target = args.date or date.today().strftime('%Y-%m-%d')
        md = generate_daily_report(target)
        out_path = REPORT_DIR / f"daily_{target}.md"
        with open(out_path, 'w') as f:
            f.write(md)
        cprint(f"Daily report saved: {out_path}", "green")

        # Also print to terminal
        trades = load_trades(date_str=target)
        entries, exits = split_entries_exits(trades)
        print_report(trades, entries, exits, title=f"DAILY REPORT — {target}")
        return 0

    # Weekly report mode
    if args.weekly:
        target = args.date or date.today().strftime('%Y-%m-%d')
        target_dt = datetime.strptime(target, '%Y-%m-%d').date()
        iso_year, iso_week, _ = target_dt.isocalendar()
        week_label = f"{iso_year}-W{iso_week:02d}"

        md = generate_weekly_report(target)
        out_path = REPORT_DIR / f"weekly_{week_label}.md"
        with open(out_path, 'w') as f:
            f.write(md)
        cprint(f"Weekly report saved: {out_path}", "green")

        # Also print to terminal
        weekday = target_dt.weekday()
        monday = target_dt - timedelta(days=weekday)
        friday = monday + timedelta(days=4)
        trades = load_trades(date_start=monday, date_end=friday)
        entries, exits = split_entries_exits(trades)
        print_report(trades, entries, exits, title=f"WEEKLY SUMMARY — {week_label}")
        return 0

    # Standard analysis mode
    trades = load_trades(
        days=args.days,
        strategy=args.strategy,
        symbol=args.symbol,
        bot_name=args.bot,
        date_str=args.date,
    )
    if not trades:
        cprint("No trades found matching criteria.", "red")
        return 1

    entries, exits = split_entries_exits(trades)
    title = "TRADE JOURNAL ANALYSIS"
    if args.bot:
        title = f"BOT ANALYSIS — {args.bot}"
    elif args.strategy:
        title = f"STRATEGY ANALYSIS — {args.strategy}"
    elif args.date:
        title = f"DAILY REPORT — {args.date}"

    print_report(trades, entries, exits, title=title)

    # Legacy markdown report
    if args.output_report:
        md = generate_markdown_legacy(trades, entries, exits, days=args.days)
        legacy_path = REPORT_DIR / 'weekly_analysis.md'
        with open(legacy_path, 'w') as f:
            f.write(md)
        cprint(f"Report saved to: {legacy_path}", "green")

    return 0


if __name__ == "__main__":
    sys.exit(main())

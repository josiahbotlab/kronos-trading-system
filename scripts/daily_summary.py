#!/usr/bin/env python3
"""
Daily Telegram Summary — Stock Trading

Sends a daily summary of trades, open positions, regime, and cumulative
stats to Telegram. Run at market close (4 PM ET / 21:00 UTC).

Usage:
    python scripts/daily_summary.py              # send summary
    python scripts/daily_summary.py --dry-run    # print only
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from src.utils.stock_journal import StockJournal
from src.utils.telegram_notifier import TelegramNotifier

ET_OFFSET = -5  # UTC-5 for EST


def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if not seconds or seconds <= 0:
        return "—"
    hours = seconds / 3600
    if hours >= 24:
        return f"{int(hours / 24)}d {int(hours % 24)}h"
    return f"{int(hours)}h {int((hours % 1) * 60)}m"


def build_summary(journal, dry_run=False):
    """Build the daily summary message."""
    # Get trades from last trading day (~24h)
    recent_trades = journal.get_closed_trades(days=1)
    open_positions = journal.get_open_positions()
    all_trades = journal.get_closed_trades()

    # Current regime
    try:
        from src.utils.regime_detector import detect_market_regime
        regime = detect_market_regime()
    except Exception:
        regime = "unknown"

    lines = []
    lines.append("\U0001f4c8 <b>STOCKS — Daily Summary</b>")
    lines.append(f"\U0001f30d Regime: <b>{regime}</b>")
    lines.append(f"\U0001f4c5 {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    lines.append("")

    # Today's trades
    if recent_trades:
        lines.append(f"\U0001f4ca <b>Trades (24h): {len(recent_trades)}</b>")

        # Per-strategy breakdown
        by_strat = defaultdict(list)
        for t in recent_trades:
            strat = t.get('strategy') or 'UNKNOWN'
            by_strat[strat].append(t)

        total_pnl = 0
        for strat, trades in sorted(by_strat.items()):
            pnls = [t['pnl_usd'] for t in trades if t.get('pnl_usd') is not None]
            strat_pnl = sum(pnls)
            total_pnl += strat_pnl
            wins = len([p for p in pnls if p > 0])
            wr = wins / len(pnls) * 100 if pnls else 0
            emoji = "\u2705" if strat_pnl >= 0 else "\u274c"
            lines.append(
                f"  {emoji} {strat}: {len(pnls)} trades, "
                f"${strat_pnl:+.2f}, WR {wr:.0f}%"
            )

        # Individual trades
        lines.append("")
        for t in recent_trades[:10]:  # Limit to 10 most recent
            pnl = t.get('pnl_usd') or 0
            emoji = "\U0001f7e2" if pnl >= 0 else "\U0001f534"
            dur = format_duration(t.get('duration_seconds'))
            reg = t.get('regime_at_entry') or ''
            lines.append(
                f"  {emoji} {t['symbol']} {t['direction']} "
                f"${pnl:+.2f} ({t.get('pnl_pct', 0):+.1f}%) "
                f"[{dur}] {reg}"
            )

        lines.append(f"\n\U0001f4b0 <b>Day P&L: ${total_pnl:+.2f}</b>")
    else:
        lines.append("\U0001f634 <i>Quiet day — no trades closed.</i>")

    # Open positions
    lines.append("")
    if open_positions:
        lines.append(f"\U0001f4bc <b>Open Positions: {len(open_positions)}</b>")
        for p in open_positions:
            entry_price = p.get('entry_price') or 0
            lines.append(
                f"  \u2022 {p['symbol']} {p['direction']} "
                f"@ ${entry_price:.2f} ({p.get('strategy', '?')})"
            )
    else:
        lines.append("\U0001f4bc Open Positions: 0")

    # Cumulative stats
    if all_trades:
        pnls = [t['pnl_usd'] for t in all_trades if t.get('pnl_usd') is not None]
        if pnls:
            total = sum(pnls)
            wins = len([p for p in pnls if p > 0])
            wr = wins / len(pnls) * 100
            lines.append("")
            lines.append("\U0001f4c8 <b>All-Time Stats</b>")
            lines.append(
                f"  Trades: {len(pnls)} | "
                f"WR: {wr:.1f}% | "
                f"PnL: ${total:+.2f}"
            )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Stock Daily Summary")
    parser.add_argument('--dry-run', action='store_true',
                        help='Print summary but do not send to Telegram')
    args = parser.parse_args()

    journal = StockJournal()
    summary = build_summary(journal, dry_run=args.dry_run)

    if args.dry_run:
        # Strip HTML tags for console display
        import re
        clean = re.sub(r'<[^>]+>', '', summary)
        print(clean)
        return

    notifier = TelegramNotifier()
    if not notifier.enabled:
        print("Telegram not configured. Printing to console:")
        import re
        print(re.sub(r'<[^>]+>', '', summary))
        return

    notifier.send(summary)
    print(f"Daily summary sent ({len(summary)} chars)")


if __name__ == '__main__':
    main()

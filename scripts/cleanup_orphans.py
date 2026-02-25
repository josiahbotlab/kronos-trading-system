#!/usr/bin/env python3
"""
Cleanup orphan open_positions entries.

Compares journal open_positions against real Alpaca positions.
Keeps entries that match real positions (closest entry price).
Moves all others to closed_trades with exit_reason='orphan_reconciled'
and zero P&L.

Usage:
    python3 scripts/cleanup_orphans.py
    python3 scripts/cleanup_orphans.py --dry-run
"""

import argparse
import os
import sqlite3
import sys
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

DB_PATH = PROJECT_ROOT / 'data' / 'trade_journal.db'


def get_alpaca_positions():
    """Fetch real positions from Alpaca."""
    from alpaca_trade_api import REST

    key = os.environ.get('ALPACA_API_KEY') or os.environ.get('APCA_API_KEY_ID')
    secret = os.environ.get('ALPACA_SECRET_KEY') or os.environ.get('APCA_API_SECRET_KEY')
    base = os.environ.get('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    api = REST(key, secret, base_url=base)

    positions = {}
    for p in api.list_positions():
        positions[p.symbol] = {
            'side': p.side.upper(),
            'qty': float(p.qty),
            'avg_entry_price': float(p.avg_entry_price),
        }
    return positions


def main():
    parser = argparse.ArgumentParser(description="Cleanup orphan open positions")
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    # Get real positions
    real = get_alpaca_positions()
    print(f"Real Alpaca positions: {len(real)}")
    for sym, info in real.items():
        print(f"  {sym} {info['side']} {info['qty']} @ ${info['avg_entry_price']:.2f}")

    # Get journal entries
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")

    rows = conn.execute(
        "SELECT * FROM open_positions ORDER BY symbol, entry_time"
    ).fetchall()
    print(f"\nJournal open positions: {len(rows)}")

    # Match: for each real position, keep the journal entry with closest price
    matched_ids = set()
    for sym, rp in real.items():
        candidates = [
            r for r in rows
            if r['symbol'] == sym and (r['direction'] or '').upper() == rp['side']
        ]
        if not candidates:
            print(f"  WARNING: {sym} on Alpaca but not in journal")
            continue
        best = min(candidates, key=lambda r: abs((r['entry_price'] or 0) - rp['avg_entry_price']))
        matched_ids.add(best['position_id'])
        print(f"  KEEP: {sym} journal @ ${best['entry_price']:.2f} "
              f"(Alpaca @ ${rp['avg_entry_price']:.2f})")

    # Everything else is an orphan
    orphans = [r for r in rows if r['position_id'] not in matched_ids]
    print(f"\nOrphans to reconcile: {len(orphans)}")

    if args.dry_run:
        print("\n[DRY RUN] Would close these orphans:")
        for r in orphans:
            print(f"  {r['symbol']} {r['direction']} @ ${r['entry_price']:.2f} ({r['strategy']})")
        conn.close()
        return

    now = time.time()
    for r in orphans:
        trade_id = str(uuid.uuid4())[:12]
        entry_time = r['entry_time'] or now
        conn.execute("""
            INSERT INTO closed_trades
            (trade_id, position_id, bot_name, strategy, symbol, direction,
             entry_time, entry_price, shares, notional_usd,
             stop_loss, take_profit, signal_strength,
             regime_at_entry, reasoning,
             exit_time, exit_price, duration_seconds,
             pnl_usd, pnl_pct, exit_reason, regime_at_exit,
             r_multiple, alpaca_order_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (trade_id, r['position_id'], r['bot_name'],
              r['strategy'], r['symbol'], r['direction'],
              entry_time, r['entry_price'], r['shares'], r['notional_usd'],
              r['stop_loss'], r['take_profit'], r['signal_strength'],
              r['regime_at_entry'], r['reasoning'],
              now, 0, now - entry_time,
              0, 0,
              'orphan_reconciled', None,
              None, None))
        conn.execute("DELETE FROM open_positions WHERE position_id = ?",
                     (r['position_id'],))

    conn.commit()

    # Verify
    remaining = conn.execute("SELECT COUNT(*) FROM open_positions").fetchone()[0]
    orphan_ct = conn.execute(
        "SELECT COUNT(*) FROM closed_trades WHERE exit_reason = 'orphan_reconciled'"
    ).fetchone()[0]
    total_ct = conn.execute("SELECT COUNT(*) FROM closed_trades").fetchone()[0]
    conn.close()

    print(f"\n=== After cleanup ===")
    print(f"Open positions remaining: {remaining}")
    print(f"Orphans closed: {orphan_ct}")
    print(f"Total closed trades: {total_ct}")


if __name__ == '__main__':
    main()

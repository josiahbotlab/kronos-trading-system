#!/usr/bin/env python3
"""
CSV → SQLite Migration

One-time script to migrate trade_journal.csv into the SQLite database.
Matches ENTRY/EXIT pairs by (symbol, strategy, direction), oldest first.
Unmatched ENTRYs are checked against Alpaca positions:
  - Still open → open_positions
  - Gone → closed_trades with exit_reason='unknown_reconciled'

Usage:
    python scripts/migrate_csv.py
    python scripts/migrate_csv.py --csv csvs/trade_journal.csv
    python scripts/migrate_csv.py --dry-run
"""

import argparse
import csv
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

from src.utils.stock_journal import StockJournal

CSV_PATH = PROJECT_ROOT / 'csvs' / 'trade_journal.csv'


def parse_timestamp(ts_str):
    """Parse CSV timestamp to epoch. CSV timestamps are UTC."""
    try:
        from datetime import timezone as tz
        dt = datetime.strptime(ts_str.strip(), '%Y-%m-%d %H:%M:%S')
        dt = dt.replace(tzinfo=tz.utc)
        return dt.timestamp()
    except Exception:
        return time.time()


def parse_float(val):
    """Safely parse float from CSV."""
    try:
        return float(val) if val and val.strip() else None
    except (ValueError, TypeError):
        return None


def get_alpaca_positions():
    """Get current Alpaca positions as {symbol: position_dict}."""
    try:
        from src.utils.order_utils import get_api
        api = get_api()
        positions = api.list_positions()
        result = {}
        for p in positions:
            result[p.symbol] = {
                'qty': float(p.qty),
                'avg_entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'side': p.side,
            }
        return result
    except Exception as e:
        print(f"  Warning: Could not fetch Alpaca positions: {e}")
        return {}


def migrate(csv_path, dry_run=False):
    """Run the migration."""
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    journal = StockJournal()

    # Read all rows
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    print(f"Read {len(rows)} rows from CSV")

    # Separate entries and exits
    entries = []
    exits = []
    for row in rows:
        if row.get('action') == 'ENTRY':
            entries.append(row)
        elif row.get('action') == 'EXIT':
            exits.append(row)

    print(f"  Entries: {len(entries)}")
    print(f"  Exits:  {len(exits)}")

    # Match EXIT to ENTRY by (symbol, direction) — oldest unmatched first
    matched_pairs = []
    unmatched_entries = list(entries)  # copy

    for exit_row in exits:
        symbol = exit_row.get('symbol', '')
        direction = exit_row.get('direction', '')
        exit_entry_price = parse_float(exit_row.get('entry_price'))

        best_match = None
        best_idx = None

        for i, entry in enumerate(unmatched_entries):
            if entry.get('symbol') != symbol:
                continue
            if entry.get('direction') != direction:
                continue

            # Prefer matching by entry_price if available
            if exit_entry_price is not None:
                entry_price = parse_float(entry.get('price'))
                if entry_price is not None and abs(entry_price - exit_entry_price) < 0.02:
                    best_match = entry
                    best_idx = i
                    break

            # Otherwise match by strategy too
            if entry.get('strategy') == exit_row.get('strategy'):
                if best_match is None:
                    best_match = entry
                    best_idx = i

        if best_match is not None:
            matched_pairs.append((best_match, exit_row))
            unmatched_entries.pop(best_idx)

    print(f"  Matched pairs: {len(matched_pairs)}")
    print(f"  Unmatched entries: {len(unmatched_entries)}")

    if dry_run:
        print("\n[DRY RUN] Would insert:")
        print(f"  {len(matched_pairs)} closed trades")
        print(f"  Up to {len(unmatched_entries)} open positions")

        # Show a few samples
        for i, (entry, exit_row) in enumerate(matched_pairs[:3]):
            pnl = parse_float(exit_row.get('pnl')) or 0
            print(f"    Trade {i+1}: {entry['symbol']} {entry['direction']} "
                  f"@ ${parse_float(entry['price']):.2f} → "
                  f"${parse_float(exit_row['price']):.2f} "
                  f"PnL: ${pnl:+.2f}")
        return

    # Insert matched pairs into closed_trades
    conn = journal._get_conn()
    closed_count = 0
    try:
        for entry, exit_row in matched_pairs:
            entry_price = parse_float(entry.get('price')) or 0
            exit_price = parse_float(exit_row.get('price')) or 0
            shares = parse_float(entry.get('shares')) or parse_float(exit_row.get('shares')) or 0
            entry_time = parse_timestamp(entry.get('timestamp', ''))
            exit_time = parse_timestamp(exit_row.get('timestamp', ''))
            duration = exit_time - entry_time if exit_time > entry_time else 0

            direction = entry.get('direction', 'LONG')
            if direction == 'LONG':
                pnl_usd = (exit_price - entry_price) * shares
                pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price else 0
            else:
                pnl_usd = (entry_price - exit_price) * shares
                pnl_pct = (entry_price - exit_price) / entry_price * 100 if entry_price else 0

            # Override with CSV values if present
            csv_pnl = parse_float(exit_row.get('pnl'))
            csv_pnl_pct = parse_float(exit_row.get('pnl_pct'))
            if csv_pnl is not None:
                pnl_usd = csv_pnl
            if csv_pnl_pct is not None:
                pnl_pct = csv_pnl_pct

            sl = parse_float(entry.get('stop_loss'))
            tp = parse_float(entry.get('take_profit'))

            r_multiple = parse_float(exit_row.get('r_multiple'))
            if r_multiple is None and sl and entry_price:
                risk = abs(entry_price - sl)
                if risk > 0:
                    reward = abs(exit_price - entry_price)
                    r_multiple = reward / risk if pnl_usd >= 0 else -(reward / risk)

            trade_id = str(uuid.uuid4())[:12]
            position_id = f"csv_{str(uuid.uuid4())[:8]}"

            bot_name = entry.get('bot_name') or exit_row.get('bot_name') or ''
            reasoning = exit_row.get('reasoning') or ''

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
            """, (trade_id, position_id, bot_name,
                  entry.get('strategy', ''), entry.get('symbol', ''),
                  direction, entry_time, entry_price, shares,
                  entry_price * shares if entry_price and shares else 0,
                  sl, tp,
                  entry.get('entry_signal_strength', ''),
                  entry.get('regime', ''),
                  entry.get('reasoning', ''),
                  exit_time, exit_price, duration,
                  round(pnl_usd, 2), round(pnl_pct, 2),
                  reasoning, exit_row.get('regime', ''),
                  round(r_multiple, 2) if r_multiple is not None else None,
                  None))
            closed_count += 1

        conn.commit()
    finally:
        conn.close()

    print(f"  Inserted {closed_count} closed trades")

    # Handle unmatched entries
    alpaca_positions = get_alpaca_positions()
    open_count = 0
    reconciled_count = 0

    conn = journal._get_conn()
    try:
        for entry in unmatched_entries:
            symbol = entry.get('symbol', '')
            direction = entry.get('direction', 'LONG')
            entry_price = parse_float(entry.get('price')) or 0
            shares = parse_float(entry.get('shares')) or 0
            entry_time = parse_timestamp(entry.get('timestamp', ''))

            if symbol in alpaca_positions:
                # Position still open
                position_id = str(uuid.uuid4())[:12]
                sl = parse_float(entry.get('stop_loss'))
                tp = parse_float(entry.get('take_profit'))

                conn.execute("""
                    INSERT INTO open_positions
                    (position_id, bot_name, strategy, symbol, direction,
                     entry_time, entry_price, shares, notional_usd,
                     stop_loss, take_profit, signal_strength,
                     regime_at_entry, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (position_id, entry.get('bot_name', ''),
                      entry.get('strategy', ''), symbol, direction,
                      entry_time, entry_price, shares,
                      entry_price * shares if entry_price and shares else 0,
                      sl, tp,
                      entry.get('entry_signal_strength', ''),
                      entry.get('regime', ''),
                      entry.get('reasoning', '')))
                open_count += 1
            else:
                # Position gone — insert as closed with unknown exit
                trade_id = str(uuid.uuid4())[:12]
                position_id = f"csv_{str(uuid.uuid4())[:8]}"
                exit_time = entry_time + 86400  # estimate: 1 day later

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
                """, (trade_id, position_id, entry.get('bot_name', ''),
                      entry.get('strategy', ''), symbol, direction,
                      entry_time, entry_price, shares,
                      entry_price * shares if entry_price and shares else 0,
                      parse_float(entry.get('stop_loss')),
                      parse_float(entry.get('take_profit')),
                      entry.get('entry_signal_strength', ''),
                      entry.get('regime', ''),
                      entry.get('reasoning', ''),
                      exit_time, 0, 86400,
                      0, 0,
                      'unknown_reconciled', '',
                      None, None))
                reconciled_count += 1

        conn.commit()
    finally:
        conn.close()

    print(f"  Inserted {open_count} open positions")
    print(f"  Inserted {reconciled_count} unknown-reconciled trades")
    print(f"\nMigration complete!")
    print(f"  Total closed: {closed_count + reconciled_count}")
    print(f"  Total open:   {open_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Migrate CSV trade journal to SQLite")
    parser.add_argument('--csv', type=str, default=str(CSV_PATH),
                        help='Path to trade_journal.csv')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be migrated without writing')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    migrate(csv_path, dry_run=args.dry_run)

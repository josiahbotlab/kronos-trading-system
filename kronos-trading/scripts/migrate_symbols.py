#!/usr/bin/env python3
"""
Migrate symbol format in Kronos databases
==========================================
Converts old Hyperliquid format (BTC/USDC:USDC) to new Coinbase format (BTC-USD)

This updates:
- prices.db (ohlcv table)
- prices.db (fetch_status table)
- Any other tables with symbol references

Usage:
    python scripts/migrate_symbols.py
    python scripts/migrate_symbols.py --dry-run    # Preview changes only
"""

import argparse
import sqlite3
import sys
from pathlib import Path

# Symbol mapping: old -> new
SYMBOL_MAP = {
    "BTC/USDC:USDC": "BTC-USD",
    "ETH/USDC:USDC": "ETH-USD",
    "SOL/USDC:USDC": "SOL-USD",
    "DOGE/USDC:USDC": "DOGE-USD",
    "XRP/USDC:USDC": "XRP-USD",
    "ADA/USDC:USDC": "ADA-USD",
    "AVAX/USDC:USDC": "AVAX-USD",
    "LINK/USDC:USDC": "LINK-USD",
    "DOT/USDC:USDC": "DOT-USD",
    "ARB/USDC:USDC": "ARB-USD",
    "OP/USDC:USDC": "OP-USD",
    "SUI/USDC:USDC": "SUI-USD",
    "APT/USDC:USDC": "APT-USD",
    "NEAR/USDC:USDC": "NEAR-USD",
    "FIL/USDC:USDC": "FIL-USD",
    "ATOM/USDC:USDC": "ATOM-USD",
    "UNI/USDC:USDC": "UNI-USD",
}


def migrate_prices_db(db_path: Path, dry_run: bool = False):
    """Migrate prices.db symbol format."""
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return

    print(f"\n📊 Migrating {db_path}")
    conn = sqlite3.connect(str(db_path))

    # Check what symbols exist
    symbols = conn.execute("SELECT DISTINCT symbol FROM ohlcv").fetchall()
    symbols = [s[0] for s in symbols]

    print(f"\n   Found {len(symbols)} unique symbols:")
    for sym in symbols:
        new_sym = SYMBOL_MAP.get(sym, sym)
        if new_sym != sym:
            print(f"   {sym:20s} → {new_sym}")
        else:
            print(f"   {sym:20s} (no change)")

    if dry_run:
        print("\n   🔍 DRY RUN - No changes made")
        conn.close()
        return

    # Migrate ohlcv table
    print("\n   Updating ohlcv table...")
    total_updated = 0
    for old_sym, new_sym in SYMBOL_MAP.items():
        result = conn.execute(
            "UPDATE ohlcv SET symbol = ? WHERE symbol = ?",
            (new_sym, old_sym)
        )
        count = result.rowcount
        if count > 0:
            print(f"   ✓ {old_sym} → {new_sym}: {count:,} candles")
            total_updated += count

    # Migrate fetch_status table
    print("\n   Updating fetch_status table...")
    for old_sym, new_sym in SYMBOL_MAP.items():
        result = conn.execute(
            "UPDATE fetch_status SET symbol = ? WHERE symbol = ?",
            (new_sym, old_sym)
        )
        count = result.rowcount
        if count > 0:
            print(f"   ✓ {old_sym} → {new_sym}: {count} entries")

    conn.commit()
    conn.close()

    print(f"\n   ✅ Migration complete! Updated {total_updated:,} candles")


def migrate_execution_db(db_path: Path, dry_run: bool = False):
    """Migrate execution.db symbol format (if exists)."""
    if not db_path.exists():
        print(f"\n   (execution.db not found, skipping)")
        return

    print(f"\n📊 Migrating {db_path}")
    conn = sqlite3.connect(str(db_path))

    # Check if trades table exists
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    tables = [t[0] for t in tables]

    if "trades" in tables:
        symbols = conn.execute("SELECT DISTINCT symbol FROM trades").fetchall()
        symbols = [s[0] for s in symbols]

        print(f"\n   Found {len(symbols)} unique symbols in trades")
        for sym in symbols:
            new_sym = SYMBOL_MAP.get(sym, sym)
            if new_sym != sym:
                print(f"   {sym:20s} → {new_sym}")

        if not dry_run:
            print("\n   Updating trades table...")
            for old_sym, new_sym in SYMBOL_MAP.items():
                result = conn.execute(
                    "UPDATE trades SET symbol = ? WHERE symbol = ?",
                    (new_sym, old_sym)
                )
                count = result.rowcount
                if count > 0:
                    print(f"   ✓ {old_sym} → {new_sym}: {count} trades")

    if "equity_snapshots" in tables:
        # No symbol migration needed for equity snapshots
        pass

    if not dry_run:
        conn.commit()

    conn.close()


def migrate_portfolio_db(db_path: Path, dry_run: bool = False):
    """Migrate portfolio.db symbol format (Week 5)."""
    if not db_path.exists():
        print(f"\n   (portfolio.db not found, skipping)")
        return

    print(f"\n📊 Migrating {db_path}")
    conn = sqlite3.connect(str(db_path))

    # Check tables
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    tables = [t[0] for t in tables]

    if "live_trades" in tables:
        # Update coin column (Week 5 uses "coin" not "symbol")
        coins = conn.execute("SELECT DISTINCT coin FROM live_trades").fetchall()
        coins = [c[0] for c in coins]

        print(f"\n   Found {len(coins)} unique coins in live_trades")
        for coin in coins:
            new_coin = SYMBOL_MAP.get(coin, coin)
            if new_coin != coin:
                print(f"   {coin:20s} → {new_coin}")

        if not dry_run:
            print("\n   Updating live_trades table...")
            for old_sym, new_sym in SYMBOL_MAP.items():
                result = conn.execute(
                    "UPDATE live_trades SET coin = ? WHERE coin = ?",
                    (new_sym, old_sym)
                )
                count = result.rowcount
                if count > 0:
                    print(f"   ✓ {old_sym} → {new_sym}: {count} trades")

    if "signal_log" in tables:
        if not dry_run:
            print("\n   Updating signal_log table...")
            for old_sym, new_sym in SYMBOL_MAP.items():
                result = conn.execute(
                    "UPDATE signal_log SET coin = ? WHERE coin = ?",
                    (new_sym, old_sym)
                )
                count = result.rowcount
                if count > 0:
                    print(f"   ✓ {old_sym} → {new_sym}: {count} signals")

    if not dry_run:
        conn.commit()

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate Kronos symbol format")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Data directory (default: data)"
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / args.data_dir

    print("=" * 60)
    print("  KRONOS SYMBOL MIGRATION")
    print("=" * 60)
    print(f"\n  Data directory: {data_dir}")
    print(f"  Mode: {'DRY RUN (preview only)' if args.dry_run else 'LIVE (will modify)'}")

    if not args.dry_run:
        confirm = input("\n  ⚠️  This will modify your databases. Continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("  Cancelled.")
            return

    # Migrate each database
    migrate_prices_db(data_dir / "prices.db", args.dry_run)
    migrate_execution_db(data_dir / "execution.db", args.dry_run)
    migrate_portfolio_db(data_dir / "portfolio.db", args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("  DRY RUN COMPLETE - No changes made")
        print("  Run without --dry-run to apply changes")
    else:
        print("  ✅ MIGRATION COMPLETE")
        print("  All symbols updated to Coinbase format (BTC-USD, etc.)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

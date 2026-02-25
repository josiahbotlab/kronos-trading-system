#!/usr/bin/env python3
"""
Kronos Data Inspector
=====================
Quick utility to check collected data status and query the databases.

Usage:
    python inspect_data.py                  # Full status report
    python inspect_data.py --liqs 20        # Last 20 liquidations
    python inspect_data.py --cascades       # Recent cascade events
    python inspect_data.py --prices BTC     # Price data status for BTC
"""

import argparse
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

def fmt_usd(val):
    if val >= 1_000_000:
        return f"${val/1_000_000:.2f}M"
    if val >= 1_000:
        return f"${val/1_000:.1f}K"
    return f"${val:.0f}"

def status_report():
    """Print full status of all databases."""
    print("=" * 60)
    print("  KRONOS DATA STATUS REPORT")
    print("=" * 60)

    # --- Liquidations ---
    liq_db = DATA_DIR / "liquidations.db"
    if liq_db.exists():
        conn = sqlite3.connect(str(liq_db))
        total = conn.execute("SELECT COUNT(*) FROM liquidations").fetchone()[0]
        if total > 0:
            row = conn.execute("""
                SELECT
                    MIN(timestamp_utc), MAX(timestamp_utc),
                    SUM(usd_value), AVG(usd_value), MAX(usd_value),
                    SUM(CASE WHEN side='BUY' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN side='SELL' THEN 1 ELSE 0 END),
                    COUNT(DISTINCT symbol)
                FROM liquidations
            """).fetchone()

            print(f"\n📊 LIQUIDATIONS")
            print(f"   Total events:  {total:,}")
            print(f"   Date range:    {row[0]} → {row[1]}")
            print(f"   Total USD:     {fmt_usd(row[2])}")
            print(f"   Avg size:      {fmt_usd(row[3])}")
            print(f"   Largest:       {fmt_usd(row[4])}")
            print(f"   Short liqs:    {row[5]:,} (shorts got rekt)")
            print(f"   Long liqs:     {row[6]:,} (longs got rekt)")
            print(f"   Symbols:       {row[7]}")

            # Top symbols by volume
            top = conn.execute("""
                SELECT symbol, COUNT(*) as cnt, SUM(usd_value) as vol
                FROM liquidations
                GROUP BY symbol ORDER BY vol DESC LIMIT 5
            """).fetchall()
            print(f"\n   Top symbols by liquidation volume:")
            for sym, cnt, vol in top:
                print(f"     {sym:12s} {cnt:>6,} events  {fmt_usd(vol):>10s}")
        else:
            print("\n📊 LIQUIDATIONS: Empty (collector running?)")
        conn.close()
    else:
        print("\n📊 LIQUIDATIONS: No database yet")

    # --- Prices ---
    price_db = DATA_DIR / "prices.db"
    if price_db.exists():
        conn = sqlite3.connect(str(price_db))
        total = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
        if total > 0:
            symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM ohlcv").fetchone()[0]
            tfs = conn.execute("SELECT DISTINCT timeframe FROM ohlcv ORDER BY timeframe").fetchall()

            print(f"\n📈 PRICES (OHLCV)")
            print(f"   Total candles: {total:,}")
            print(f"   Symbols:       {symbols}")
            print(f"   Timeframes:    {', '.join(t[0] for t in tfs)}")

            # Per-timeframe breakdown
            for tf in tfs:
                cnt = conn.execute(
                    "SELECT COUNT(*) FROM ohlcv WHERE timeframe=?", tf
                ).fetchone()[0]
                print(f"     {tf[0]:6s}: {cnt:>8,} candles")
        else:
            print("\n📈 PRICES: Empty (collector running?)")
        conn.close()
    else:
        print("\n📈 PRICES: No database yet")

    # --- Positions ---
    pos_db = DATA_DIR / "positions.db"
    if pos_db.exists():
        conn = sqlite3.connect(str(pos_db))
        oi = conn.execute("SELECT COUNT(*) FROM open_interest").fetchone()[0]
        ls = conn.execute("SELECT COUNT(*) FROM long_short_ratio").fetchone()[0]
        tt = conn.execute("SELECT COUNT(*) FROM top_trader_ratio").fetchone()[0]
        tv = conn.execute("SELECT COUNT(*) FROM taker_volume").fetchone()[0]

        print(f"\n📊 POSITIONS / OPEN INTEREST")
        print(f"   OI snapshots:        {oi:,}")
        print(f"   L/S ratio entries:    {ls:,}")
        print(f"   Top trader ratios:    {tt:,}")
        print(f"   Taker volume entries: {tv:,}")
        conn.close()
    else:
        print("\n📊 POSITIONS: No database yet")

    # DB file sizes
    print(f"\n💾 DATABASE SIZES")
    for db_file in DATA_DIR.glob("*.db*"):
        size_mb = db_file.stat().st_size / (1024 * 1024)
        print(f"   {db_file.name:25s} {size_mb:.1f} MB")

    print()


def show_recent_liqs(n: int = 20):
    """Show most recent liquidation events."""
    liq_db = DATA_DIR / "liquidations.db"
    if not liq_db.exists():
        print("No liquidation database found.")
        return

    conn = sqlite3.connect(str(liq_db))
    rows = conn.execute(
        """SELECT timestamp_utc, symbol, side, usd_value, price
           FROM liquidations ORDER BY timestamp_ms DESC LIMIT ?""",
        (n,),
    ).fetchall()

    if not rows:
        print("No liquidation events found.")
        return

    print(f"\n{'Time (UTC)':20s} {'Symbol':12s} {'Side':6s} {'USD Value':>12s} {'Price':>12s}")
    print("-" * 66)
    for ts, sym, side, usd, price in rows:
        side_label = "SHORT" if side == "BUY" else "LONG"
        print(f"{ts:20s} {sym:12s} {side_label:6s} {fmt_usd(usd):>12s} {price:>12,.2f}")

    conn.close()


def show_cascades():
    """Show recent cascade events (high liquidation activity windows)."""
    liq_db = DATA_DIR / "liquidations.db"
    if not liq_db.exists():
        print("No liquidation database found.")
        return

    conn = sqlite3.connect(str(liq_db))
    rows = conn.execute("""
        SELECT
            datetime(window_start_ms/1000, 'unixepoch') as window_time,
            symbol, event_count, total_usd,
            short_liqs, long_liqs, max_single_liq
        FROM cascade_1m
        WHERE event_count >= 5
        ORDER BY window_start_ms DESC
        LIMIT 20
    """).fetchall()

    if not rows:
        print("No cascade events found (need 5+ liqs in 1 min).")
        return

    print(f"\n{'Window (UTC)':20s} {'Symbol':12s} {'Events':>7s} {'Total USD':>12s} {'S/L':>6s} {'Max Single':>12s}")
    print("-" * 73)
    for ts, sym, cnt, total, shorts, longs, max_s in rows:
        print(f"{ts:20s} {sym:12s} {cnt:>7d} {fmt_usd(total):>12s} {shorts}/{longs:>3d} {fmt_usd(max_s):>12s}")

    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Kronos Data Inspector")
    parser.add_argument("--liqs", type=int, nargs="?", const=20, help="Show recent liquidations")
    parser.add_argument("--cascades", action="store_true", help="Show cascade events")
    args = parser.parse_args()

    if args.liqs:
        show_recent_liqs(args.liqs)
    elif args.cascades:
        show_cascades()
    else:
        status_report()


if __name__ == "__main__":
    main()

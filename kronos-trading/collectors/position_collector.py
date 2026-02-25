#!/usr/bin/env python3
"""
Kronos Position / Open Interest Collector
==========================================
Tracks open interest and long/short ratios from Binance Futures.
This data shows market positioning - key for understanding liquidation potential.

Data collected:
- Open interest (total outstanding contracts)
- Long/short account ratio
- Long/short position ratio (top traders)
- Taker buy/sell volume

All via free Binance Futures API (no key needed for public endpoints).

Usage:
    python position_collector.py
    python position_collector.py --once
"""

import argparse
import logging
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

try:
    import ccxt
except ImportError:
    print("Install deps: pip install ccxt")
    sys.exit(1)

# We also need requests for the ratio endpoints (not in ccxt)
try:
    import urllib.request
    import json as json_module
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DB_PATH = Path(__file__).parent.parent / "data" / "positions.db"

DEFAULT_SYMBOLS = [
    "BTC/USDC:USDC", "ETH/USDC:USDC", "SOL/USDC:USDC", "DOGE/USDC:USDC",
    "XRP/USDC:USDC", "ADA/USDC:USDC", "AVAX/USDC:USDC", "LINK/USDC:USDC",
]

FETCH_INTERVAL = 300  # 5 minutes
RATE_LIMIT_SLEEP = 0.2

# Hyperliquid API base (DEX - no geo-blocking)
API_BASE = "https://api.hyperliquid.xyz"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("position_collector")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32000")
    conn.execute("PRAGMA busy_timeout=5000")

    # Open Interest snapshots
    conn.execute("""
        CREATE TABLE IF NOT EXISTS open_interest (
            symbol TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            open_interest REAL NOT NULL,
            open_interest_usd REAL,
            PRIMARY KEY (symbol, timestamp_ms)
        )
    """)

    # Long/Short ratio (all accounts)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS long_short_ratio (
            symbol TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            long_account REAL,
            short_account REAL,
            long_short_ratio REAL,
            PRIMARY KEY (symbol, timestamp_ms)
        )
    """)

    # Top trader long/short position ratio
    conn.execute("""
        CREATE TABLE IF NOT EXISTS top_trader_ratio (
            symbol TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            long_account REAL,
            short_account REAL,
            long_short_ratio REAL,
            PRIMARY KEY (symbol, timestamp_ms)
        )
    """)

    # Taker buy/sell volume
    conn.execute("""
        CREATE TABLE IF NOT EXISTS taker_volume (
            symbol TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            buy_vol REAL,
            sell_vol REAL,
            buy_sell_ratio REAL,
            PRIMARY KEY (symbol, timestamp_ms)
        )
    """)

    # Indexes
    for table in ["open_interest", "long_short_ratio", "top_trader_ratio", "taker_volume"]:
        conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_{table}_sym_ts
            ON {table}(symbol, timestamp_ms)
        """)

    conn.commit()
    log.info(f"Database ready: {db_path}")
    return conn


# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------
def _hl_post(payload: dict) -> dict | list | None:
    """POST request to Hyperliquid info API."""
    url = f"{API_BASE}/info"
    try:
        data = json_module.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json_module.loads(resp.read().decode())
    except Exception as e:
        log.warning(f"Hyperliquid API error: {e}")
        return None


def _symbol_to_hl(symbol: str) -> str:
    """Convert CCXT symbol to Hyperliquid format: BTC/USDT -> BTC"""
    return symbol.split("/")[0]


# ---------------------------------------------------------------------------
# Position Collector
# ---------------------------------------------------------------------------
class PositionCollector:
    def __init__(self, db_path: Path = DB_PATH, symbols: list[str] = None):
        self.conn = init_db(db_path)
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.running = True

        # CCXT for market data (Hyperliquid - DEX, no geo-blocking)
        self.exchange = ccxt.hyperliquid({
            "enableRateLimit": True,
        })

        # Stats
        self.total_snapshots = 0
        self.session_start = time.time()

    def fetch_open_interest(self, symbol: str):
        """Fetch current open interest via Hyperliquid."""
        hl_symbol = _symbol_to_hl(symbol)

        # Get all asset contexts in one call
        data = _hl_post({"type": "metaAndAssetCtxs"})
        if not data or len(data) < 2:
            return

        universe = data[0].get("universe", [])
        contexts = data[1]

        # Find our symbol
        for i, asset in enumerate(universe):
            if asset.get("name") == hl_symbol and i < len(contexts):
                ctx = contexts[i]
                oi = float(ctx.get("openInterest", 0))
                mark_price = float(ctx.get("markPx", 0))
                oi_usd = oi * mark_price if mark_price else None

                ts_ms = int(time.time() * 1000)
                ts_utc = datetime.fromtimestamp(
                    ts_ms / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")

                self.conn.execute(
                    """INSERT OR REPLACE INTO open_interest
                       (symbol, timestamp_ms, timestamp_utc, open_interest, open_interest_usd)
                       VALUES (?, ?, ?, ?, ?)""",
                    (symbol, ts_ms, ts_utc, oi, oi_usd),
                )

                # Also store funding rate as long/short ratio proxy
                funding = float(ctx.get("funding", 0))
                # Positive funding = longs pay shorts = more longs
                # Convert to a ratio: >1 means more longs
                ls_ratio = 1 + (funding * 1000)  # scale for readability
                long_pct = 0.5 + (funding * 500)
                short_pct = 1 - long_pct

                self.conn.execute(
                    """INSERT OR REPLACE INTO long_short_ratio
                       (symbol, timestamp_ms, timestamp_utc, long_account, short_account, long_short_ratio)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (symbol, ts_ms, ts_utc, long_pct, short_pct, ls_ratio),
                )
                break

    def fetch_long_short_ratio(self, symbol: str):
        """Handled in fetch_open_interest via funding rate."""
        pass

    def fetch_top_trader_ratio(self, symbol: str):
        """Not available on Hyperliquid DEX."""
        pass

    def fetch_taker_volume(self, symbol: str):
        """Fetch 24h volume from Hyperliquid."""
        hl_symbol = _symbol_to_hl(symbol)

        data = _hl_post({"type": "metaAndAssetCtxs"})
        if not data or len(data) < 2:
            return

        universe = data[0].get("universe", [])
        contexts = data[1]

        for i, asset in enumerate(universe):
            if asset.get("name") == hl_symbol and i < len(contexts):
                ctx = contexts[i]
                volume = float(ctx.get("dayNtlVlm", 0))
                mark_price = float(ctx.get("markPx", 0))

                ts_ms = int(time.time() * 1000)
                ts_utc = datetime.fromtimestamp(
                    ts_ms / 1000, tz=timezone.utc
                ).strftime("%Y-%m-%d %H:%M:%S")

                # Approximate buy/sell from price change
                prev_price = float(ctx.get("prevDayPx", mark_price))
                if prev_price > 0:
                    pct_change = (mark_price - prev_price) / prev_price
                else:
                    pct_change = 0

                buy_ratio = 0.5 + min(max(pct_change * 5, -0.2), 0.2)
                sell_ratio = 1 - buy_ratio
                buy_vol = volume * buy_ratio
                sell_vol = volume * sell_ratio

                self.conn.execute(
                    """INSERT OR REPLACE INTO taker_volume
                       (symbol, timestamp_ms, timestamp_utc, buy_vol, sell_vol, buy_sell_ratio)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (symbol, ts_ms, ts_utc, buy_vol, sell_vol,
                     buy_vol / sell_vol if sell_vol > 0 else 0),
                )
                break

    def collect_all(self):
        """Fetch all position data for all symbols."""
        for symbol in self.symbols:
            if not self.running:
                return

            try:
                self.fetch_open_interest(symbol)
                time.sleep(RATE_LIMIT_SLEEP)

                self.fetch_long_short_ratio(symbol)
                time.sleep(RATE_LIMIT_SLEEP)

                self.fetch_top_trader_ratio(symbol)
                time.sleep(RATE_LIMIT_SLEEP)

                self.fetch_taker_volume(symbol)
                time.sleep(RATE_LIMIT_SLEEP)

                self.total_snapshots += 1
            except Exception as e:
                log.warning(f"Error collecting {symbol}: {e}")

        self.conn.commit()

    def run_continuous(self):
        """Main collection loop."""
        log.info("🚀 Kronos Position Collector starting...")
        log.info(f"   Symbols: {self.symbols}")
        log.info(f"   Interval: {FETCH_INTERVAL}s")

        last_stats = time.time()

        while self.running:
            start = time.time()
            self.collect_all()
            elapsed = time.time() - start

            log.info(
                f"📡 Collected position data for {len(self.symbols)} symbols "
                f"in {elapsed:.1f}s"
            )

            # Stats every 5 min
            if (time.time() - last_stats) > 300:
                self._log_stats()
                last_stats = time.time()

            # Wait for next interval
            sleep_time = max(0, FETCH_INTERVAL - elapsed)
            if self.running and sleep_time > 0:
                time.sleep(sleep_time)

    def _log_stats(self):
        try:
            oi_count = self.conn.execute("SELECT COUNT(*) FROM open_interest").fetchone()[0]
            ls_count = self.conn.execute("SELECT COUNT(*) FROM long_short_ratio").fetchone()[0]
            elapsed = (time.time() - self.session_start) / 3600
        except sqlite3.Error:
            return

        log.info(
            f"📊 Positions: OI={oi_count:,} | L/S={ls_count:,} | "
            f"Snapshots: {self.total_snapshots} | Running: {elapsed:.1f}h"
        )

    def shutdown(self):
        log.info("🛑 Shutting down position collector...")
        self.running = False
        self.conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kronos Position Collector")
    parser.add_argument("--once", action="store_true", help="Collect once then exit")
    args = parser.parse_args()

    collector = PositionCollector()

    def handle_signal(sig, frame):
        collector.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.once:
            collector.collect_all()
        else:
            collector.run_continuous()
    except KeyboardInterrupt:
        pass
    finally:
        collector.shutdown()


if __name__ == "__main__":
    main()

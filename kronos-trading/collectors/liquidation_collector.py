#!/usr/bin/env python3
"""
Kronos Liquidation Collector
=============================
Connects to Binance Futures WebSocket and captures ALL forced liquidation events.
Stores in SQLite for backtesting liquidation cascade strategies.

Data source: wss://fstream.binance.com/ws/!forceOrder@arr
This is the alpha edge - liquidation data drives Moon Dev's best strategies.

Usage:
    python liquidation_collector.py
    python liquidation_collector.py --config ../config/kronos.json
"""

import asyncio
import json
import logging
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Install deps: pip install websockets")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_WS_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"
RECONNECT_DELAY = 5          # seconds between reconnect attempts
LOG_INTERVAL = 300            # print stats every 5 min
DB_PATH = Path(__file__).parent.parent / "data" / "liquidations.db"
BATCH_SIZE = 50               # flush to DB every N events
BATCH_TIMEOUT = 10            # or every N seconds, whichever first

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("liq_collector")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db(db_path: Path) -> sqlite3.Connection:
    """Create/open the liquidations database with optimized settings."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    # Performance optimizations for write-heavy workload
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    conn.execute("PRAGMA busy_timeout=5000")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS liquidations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp_ms INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            time_in_force TEXT,
            quantity REAL NOT NULL,
            price REAL NOT NULL,
            avg_price REAL NOT NULL,
            status TEXT NOT NULL,
            last_filled_qty REAL,
            accumulated_qty REAL,
            usd_value REAL NOT NULL,
            collected_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # Indexes for common query patterns
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_liq_timestamp
        ON liquidations(timestamp_ms)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_liq_symbol_time
        ON liquidations(symbol, timestamp_ms)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_liq_usd_value
        ON liquidations(usd_value)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_liq_side_time
        ON liquidations(side, timestamp_ms)
    """)

    # Cascades view - aggregated liquidation events per time window
    conn.execute("""
        CREATE VIEW IF NOT EXISTS cascade_1m AS
        SELECT
            (timestamp_ms / 60000) * 60000 AS window_start_ms,
            symbol,
            COUNT(*) AS event_count,
            SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) AS short_liqs,
            SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) AS long_liqs,
            SUM(usd_value) AS total_usd,
            SUM(CASE WHEN side = 'BUY' THEN usd_value ELSE 0 END) AS short_liq_usd,
            SUM(CASE WHEN side = 'SELL' THEN usd_value ELSE 0 END) AS long_liq_usd,
            MAX(usd_value) AS max_single_liq,
            AVG(usd_value) AS avg_liq_size
        FROM liquidations
        GROUP BY window_start_ms, symbol
    """)

    conn.commit()
    log.info(f"Database ready: {db_path}")
    return conn


# ---------------------------------------------------------------------------
# WebSocket Collector
# ---------------------------------------------------------------------------
class LiquidationCollector:
    def __init__(self, db_path: Path = DB_PATH, ws_url: str = DEFAULT_WS_URL):
        self.ws_url = ws_url
        self.conn = init_db(db_path)
        self.running = True
        self.batch: list[tuple] = []
        self.last_flush = time.time()

        # Stats
        self.total_events = 0
        self.session_events = 0
        self.session_usd = 0.0
        self.session_start = time.time()
        self.last_log = time.time()
        self.largest_liq = 0.0
        self.largest_liq_symbol = ""

    def parse_event(self, data: dict) -> tuple | None:
        """Parse a Binance forceOrder event into a DB row."""
        try:
            order = data.get("o", {})
            symbol = order.get("s", "")           # e.g. BTCUSDT
            side = order.get("S", "")              # BUY = short liquidated, SELL = long liquidated
            order_type = order.get("o", "")        # LIMIT
            tif = order.get("f", "")               # Time in force
            qty = float(order.get("q", 0))         # Original quantity
            price = float(order.get("p", 0))       # Price
            avg_price = float(order.get("ap", 0))  # Average price
            status = order.get("X", "")            # Order status
            last_qty = float(order.get("l", 0))    # Last filled quantity
            acc_qty = float(order.get("z", 0))     # Accumulated filled quantity
            ts_ms = int(order.get("T", 0))         # Trade time

            # Calculate USD value
            use_price = avg_price if avg_price > 0 else price
            usd_value = qty * use_price

            ts_utc = datetime.fromtimestamp(
                ts_ms / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")

            return (
                ts_ms, ts_utc, symbol, side, order_type, tif,
                qty, price, avg_price, status, last_qty, acc_qty, usd_value
            )
        except (KeyError, ValueError, TypeError) as e:
            log.warning(f"Failed to parse event: {e} | data={data}")
            return None

    def flush_batch(self):
        """Write buffered events to SQLite."""
        if not self.batch:
            return
        try:
            self.conn.executemany(
                """INSERT INTO liquidations
                   (timestamp_ms, timestamp_utc, symbol, side, order_type,
                    time_in_force, quantity, price, avg_price, status,
                    last_filled_qty, accumulated_qty, usd_value)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                self.batch,
            )
            self.conn.commit()
            self.batch.clear()
            self.last_flush = time.time()
        except sqlite3.Error as e:
            log.error(f"DB write failed: {e}")

    def log_stats(self):
        """Print periodic stats."""
        elapsed = time.time() - self.session_start
        hrs = elapsed / 3600
        rate = self.session_events / max(elapsed, 1) * 60  # events/min

        # Get total count from DB
        try:
            total = self.conn.execute(
                "SELECT COUNT(*) FROM liquidations"
            ).fetchone()[0]
        except sqlite3.Error:
            total = "?"

        log.info(
            f"📊 Session: {self.session_events:,} events | "
            f"${self.session_usd:,.0f} total | "
            f"{rate:.1f}/min | "
            f"Largest: ${self.largest_liq:,.0f} ({self.largest_liq_symbol}) | "
            f"DB total: {total:,} | "
            f"Running: {hrs:.1f}h"
        )
        self.last_log = time.time()

    async def collect(self):
        """Main collection loop with auto-reconnect."""
        log.info(f"🚀 Kronos Liquidation Collector starting...")
        log.info(f"   WebSocket: {self.ws_url}")
        log.info(f"   Database: {DB_PATH}")

        while self.running:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    log.info("✅ Connected to Binance Futures WebSocket")

                    async for raw_msg in ws:
                        if not self.running:
                            break

                        try:
                            data = json.loads(raw_msg)
                        except json.JSONDecodeError:
                            continue

                        row = self.parse_event(data)
                        if row is None:
                            continue

                        usd_value = row[12]  # usd_value field
                        symbol = row[2]
                        side = row[3]

                        self.batch.append(row)
                        self.session_events += 1
                        self.session_usd += usd_value

                        if usd_value > self.largest_liq:
                            self.largest_liq = usd_value
                            self.largest_liq_symbol = f"{symbol} ({'SHORT' if side == 'BUY' else 'LONG'})"

                        # Flush batch if full or timed out
                        now = time.time()
                        if (
                            len(self.batch) >= BATCH_SIZE
                            or (now - self.last_flush) > BATCH_TIMEOUT
                        ):
                            self.flush_batch()

                        # Periodic stats
                        if (now - self.last_log) > LOG_INTERVAL:
                            self.log_stats()

            except websockets.exceptions.ConnectionClosed as e:
                log.warning(f"WebSocket closed: {e}. Reconnecting in {RECONNECT_DELAY}s...")
            except Exception as e:
                log.error(f"WebSocket error: {e}. Reconnecting in {RECONNECT_DELAY}s...")

            # Flush any remaining before reconnect
            self.flush_batch()

            if self.running:
                await asyncio.sleep(RECONNECT_DELAY)

    def shutdown(self):
        """Graceful shutdown."""
        log.info("🛑 Shutting down...")
        self.running = False
        self.flush_batch()
        self.log_stats()
        self.conn.close()
        log.info("Database closed. Goodbye.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    collector = LiquidationCollector()

    # Handle graceful shutdown
    loop = asyncio.new_event_loop()

    def handle_signal(sig, frame):
        collector.shutdown()
        loop.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        loop.run_until_complete(collector.collect())
    except (KeyboardInterrupt, RuntimeError):
        pass
    finally:
        collector.shutdown()
        loop.close()


if __name__ == "__main__":
    main()

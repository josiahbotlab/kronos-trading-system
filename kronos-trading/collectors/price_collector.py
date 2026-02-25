#!/usr/bin/env python3
"""
Kronos Price Collector
======================
Fetches OHLCV (candlestick) data from Binance Futures via CCXT.
No API key needed - public data only.

Features:
- Multi-symbol, multi-timeframe collection
- Automatic backfill on first run (configurable days)
- Incremental updates (only fetches new candles)
- Rate limit aware
- SQLite storage with proper indexing

Usage:
    python price_collector.py                    # Run continuous collection
    python price_collector.py --backfill 90      # Backfill 90 days then collect
    python price_collector.py --once             # Single fetch then exit
"""

import argparse
import json
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent.parent / "config" / "kronos.json"
DB_PATH = Path(__file__).parent.parent / "data" / "prices.db"

DEFAULT_SYMBOLS = [
    "BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD",
    "XRP-USD", "ADA-USD", "AVAX-USD", "LINK-USD",
    "DOT-USD", "ARB-USD", "OP-USD",
    "SUI-USD", "APT-USD", "NEAR-USD",
    "ATOM-USD", "UNI-USD",
]

DEFAULT_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

# How often to fetch each timeframe (seconds)
FETCH_INTERVALS = {
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}

BACKFILL_DAYS = 90
RATE_LIMIT_SLEEP = 0.15  # seconds between API calls (Binance allows ~1200/min)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("price_collector")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def init_db(db_path: Path) -> sqlite3.Connection:
    """Create/open the prices database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA busy_timeout=5000")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            timestamp_ms INTEGER NOT NULL,
            timestamp_utc TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (symbol, timeframe, timestamp_ms)
        )
    """)

    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ohlcv_sym_tf_ts
        ON ohlcv(symbol, timeframe, timestamp_ms)
    """)

    # Track last fetch time per symbol/timeframe
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_status (
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            last_timestamp_ms INTEGER NOT NULL,
            last_fetched_at TEXT NOT NULL,
            candle_count INTEGER DEFAULT 0,
            PRIMARY KEY (symbol, timeframe)
        )
    """)

    conn.commit()
    log.info(f"Database ready: {db_path}")
    return conn


# ---------------------------------------------------------------------------
# Price Collector
# ---------------------------------------------------------------------------
class PriceCollector:
    def __init__(
        self,
        db_path: Path = DB_PATH,
        symbols: list[str] = None,
        timeframes: list[str] = None,
    ):
        self.conn = init_db(db_path)
        self.symbols = symbols or DEFAULT_SYMBOLS
        self.timeframes = timeframes or DEFAULT_TIMEFRAMES
        self.running = True

        # Initialize exchange (Coinbase - US-friendly, spot trading)
        # Using Coinbase for both price data and execution
        self.exchange = ccxt.coinbase({
            "enableRateLimit": True,
        })

        # Symbol mapping: our format (BTC-USD) to CCXT format (BTC/USD)
        self.symbol_map = {
            "BTC-USD": "BTC/USD",
            "ETH-USD": "ETH/USD",
            "SOL-USD": "SOL/USD",
            "DOGE-USD": "DOGE/USD",
            "XRP-USD": "XRP/USD",
            "ADA-USD": "ADA/USD",
            "AVAX-USD": "AVAX/USD",
            "LINK-USD": "LINK/USD",
            "DOT-USD": "DOT/USD",
            "ARB-USD": "ARB/USD",
            "OP-USD": "OP/USD",
            "SUI-USD": "SUI/USD",
            "APT-USD": "APT/USD",
            "NEAR-USD": "NEAR/USD",
            "ATOM-USD": "ATOM/USD",
            "UNI-USD": "UNI/USD",
        }

        # Track when each symbol/tf was last fetched this session
        self.last_fetch_time: dict[str, float] = {}

        # Stats
        self.total_candles_stored = 0
        self.session_start = time.time()

    def _get_last_timestamp(self, symbol: str, timeframe: str) -> int | None:
        """Get the last stored timestamp for a symbol/timeframe."""
        row = self.conn.execute(
            "SELECT last_timestamp_ms FROM fetch_status WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        ).fetchone()
        return row[0] if row else None

    def _update_fetch_status(self, symbol: str, timeframe: str, last_ts: int, count: int):
        """Update the fetch status tracker."""
        self.conn.execute(
            """INSERT OR REPLACE INTO fetch_status
               (symbol, timeframe, last_timestamp_ms, last_fetched_at, candle_count)
               VALUES (?, ?, ?, datetime('now'), ?)""",
            (symbol, timeframe, last_ts, count),
        )

    def fetch_ohlcv(self, symbol: str, timeframe: str, since_ms: int | None = None, limit: int = 1000) -> int:
        """Fetch OHLCV data for a single symbol/timeframe and store it."""
        # Map symbol to CCXT format
        ccxt_symbol = self.symbol_map.get(symbol, symbol)

        try:
            candles = self.exchange.fetch_ohlcv(
                ccxt_symbol, timeframe, since=since_ms, limit=limit
            )
        except ccxt.BaseError as e:
            log.warning(f"CCXT error fetching {symbol} {timeframe}: {e}")
            return 0

        if not candles:
            return 0

        rows = []
        for c in candles:
            ts_ms = int(c[0])
            ts_utc = datetime.fromtimestamp(
                ts_ms / 1000, tz=timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")
            rows.append((symbol, timeframe, ts_ms, ts_utc, c[1], c[2], c[3], c[4], c[5]))

        self.conn.executemany(
            """INSERT OR REPLACE INTO ohlcv
               (symbol, timeframe, timestamp_ms, timestamp_utc, open, high, low, close, volume)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )

        last_ts = int(candles[-1][0])
        total = self.conn.execute(
            "SELECT COUNT(*) FROM ohlcv WHERE symbol=? AND timeframe=?",
            (symbol, timeframe),
        ).fetchone()[0]

        self._update_fetch_status(symbol, timeframe, last_ts, total)
        self.conn.commit()

        self.total_candles_stored += len(rows)
        return len(rows)

    def backfill(self, days: int = BACKFILL_DAYS):
        """Backfill historical data for all symbols and timeframes."""
        since_ms = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)

        total_tasks = len(self.symbols) * len(self.timeframes)
        completed = 0

        log.info(f"📥 Starting backfill: {days} days, {len(self.symbols)} symbols, {len(self.timeframes)} timeframes")

        for symbol in self.symbols:
            for tf in self.timeframes:
                if not self.running:
                    return

                # Check if we already have data
                last_ts = self._get_last_timestamp(symbol, tf)
                fetch_since = last_ts + 1 if last_ts and last_ts > since_ms else since_ms

                total_fetched = 0
                current_since = fetch_since

                while self.running:
                    count = self.fetch_ohlcv(symbol, tf, since_ms=current_since)
                    total_fetched += count
                    time.sleep(RATE_LIMIT_SLEEP)

                    if count < 1000:
                        break  # No more data

                    # Move forward
                    last = self._get_last_timestamp(symbol, tf)
                    if last and last > current_since:
                        current_since = last + 1
                    else:
                        break

                completed += 1
                if total_fetched > 0:
                    log.info(
                        f"  [{completed}/{total_tasks}] {symbol} {tf}: "
                        f"+{total_fetched} candles"
                    )

        log.info(f"✅ Backfill complete: {self.total_candles_stored:,} total candles stored")

    def collect_incremental(self):
        """Fetch only new candles for all symbols/timeframes."""
        for symbol in self.symbols:
            for tf in self.timeframes:
                if not self.running:
                    return

                key = f"{symbol}_{tf}"
                now = time.time()
                interval = FETCH_INTERVALS.get(tf, 3600)

                # Skip if we fetched recently
                last = self.last_fetch_time.get(key, 0)
                if (now - last) < interval:
                    continue

                last_ts = self._get_last_timestamp(symbol, tf)
                since = last_ts + 1 if last_ts else None

                count = self.fetch_ohlcv(symbol, tf, since_ms=since)
                self.last_fetch_time[key] = now

                if count > 0:
                    log.debug(f"  {symbol} {tf}: +{count} candles")

                time.sleep(RATE_LIMIT_SLEEP)

    def run_continuous(self):
        """Main loop: backfill then collect incrementally."""
        log.info("🚀 Kronos Price Collector starting...")
        log.info(f"   Symbols: {len(self.symbols)}")
        log.info(f"   Timeframes: {self.timeframes}")

        # Backfill first
        self.backfill()

        # Then collect new data continuously
        log.info("📡 Switching to incremental collection mode...")
        last_stats = time.time()

        while self.running:
            self.collect_incremental()

            # Print stats every 5 min
            if (time.time() - last_stats) > 300:
                self._log_stats()
                last_stats = time.time()

            time.sleep(10)  # Check for new data every 10s

    def _log_stats(self):
        """Print collection stats."""
        try:
            total = self.conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()[0]
            symbols = self.conn.execute(
                "SELECT COUNT(DISTINCT symbol) FROM ohlcv"
            ).fetchone()[0]
            elapsed = (time.time() - self.session_start) / 3600
        except sqlite3.Error:
            return

        log.info(
            f"📊 Prices: {total:,} candles | "
            f"{symbols} symbols | "
            f"Session: +{self.total_candles_stored:,} | "
            f"Running: {elapsed:.1f}h"
        )

    def shutdown(self):
        """Graceful shutdown."""
        log.info("🛑 Shutting down price collector...")
        self.running = False
        self.conn.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kronos Price Collector")
    parser.add_argument("--backfill", type=int, default=BACKFILL_DAYS,
                        help=f"Days to backfill (default: {BACKFILL_DAYS})")
    parser.add_argument("--once", action="store_true",
                        help="Fetch once then exit")
    args = parser.parse_args()

    collector = PriceCollector()

    def handle_signal(sig, frame):
        collector.shutdown()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        if args.once:
            collector.backfill(args.backfill)
        else:
            collector.run_continuous()
    except KeyboardInterrupt:
        pass
    finally:
        collector.shutdown()


if __name__ == "__main__":
    main()

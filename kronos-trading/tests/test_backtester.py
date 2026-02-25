#!/usr/bin/env python3
"""
Kronos Backtester Integration Test
====================================
Tests the full pipeline: data loading, strategy execution,
metrics calculation, and robustness testing.

Uses synthetic price data so we don't need real DB files.
"""

import sys
import math
import sqlite3
import logging
import random
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.metrics import Trade, calculate_metrics, compare_reports
from core.backtester import Backtester
from core.robustness import RobustnessTestSuite
from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal
from strategies.momentum.sma_crossover import SMACrossover

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def create_synthetic_data(
    db_path: Path,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    n_candles: int = 2000,
    start_price: float = 40000.0,
    trend: float = 0.0001,      # slight uptrend
    volatility: float = 0.02,
):
    """Create synthetic OHLCV data in SQLite for testing."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    conn.execute("PRAGMA journal_mode=WAL")
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

    # Generate price series with trend + mean reversion + noise
    np.random.seed(42)
    base_ts = 1704067200000  # 2024-01-01 00:00:00 UTC
    tf_ms = 3600000  # 1h

    price = start_price
    rows = []

    for i in range(n_candles):
        ts = base_ts + i * tf_ms

        # Random walk with trend
        ret = trend + volatility * np.random.randn()
        # Add some mean reversion for realism
        if i > 50:
            mean_price = start_price * (1 + trend * i)
            ret += 0.001 * (mean_price - price) / price

        open_price = price
        close_price = price * (1 + ret)

        # Generate realistic high/low
        intra_vol = abs(ret) + volatility * 0.3 * abs(np.random.randn())
        high_price = max(open_price, close_price) * (1 + intra_vol * 0.5)
        low_price = min(open_price, close_price) * (1 - intra_vol * 0.5)

        volume = 100 + 900 * abs(np.random.randn())  # random volume

        from datetime import datetime, timezone
        ts_utc = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        rows.append((symbol, timeframe, ts, ts_utc,
                      open_price, high_price, low_price, close_price, volume))

        price = close_price

    conn.executemany(
        """INSERT OR REPLACE INTO ohlcv
           (symbol, timeframe, timestamp_ms, timestamp_utc, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )

    # Also create fetch_status table (required by schema)
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
    conn.close()

    log.info(f"Created {n_candles} synthetic candles at {db_path}")
    log.info(f"  Price range: ${rows[0][4]:,.0f} -> ${rows[-1][7]:,.0f}")
    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_metrics():
    """Test performance metrics calculation."""
    print("\n" + "=" * 50)
    print("  TEST 1: Metrics Calculation")
    print("=" * 50)

    trades = [
        Trade(entry_time=1000, exit_time=2000, symbol="BTC", side="long",
              entry_price=40000, exit_price=41000, quantity=0.1,
              pnl=100, pnl_pct=2.5),
        Trade(entry_time=3000, exit_time=4000, symbol="BTC", side="long",
              entry_price=41000, exit_price=40500, quantity=0.1,
              pnl=-50, pnl_pct=-1.22),
        Trade(entry_time=5000, exit_time=6000, symbol="BTC", side="short",
              entry_price=40500, exit_price=39500, quantity=0.1,
              pnl=100, pnl_pct=2.47),
        Trade(entry_time=7000, exit_time=8000, symbol="BTC", side="long",
              entry_price=39500, exit_price=40800, quantity=0.1,
              pnl=130, pnl_pct=3.29),
    ]

    report = calculate_metrics(trades, initial_capital=10000)

    assert report.total_trades == 4, f"Expected 4 trades, got {report.total_trades}"
    assert report.winning_trades == 3, f"Expected 3 winners, got {report.winning_trades}"
    assert report.losing_trades == 1, f"Expected 1 loser, got {report.losing_trades}"
    assert report.win_rate_pct == 75.0, f"Expected 75% win rate, got {report.win_rate_pct}"
    assert report.total_return_usd > 0, "Expected positive return"

    print(report.summary())
    print("✅ Metrics test PASSED")


def test_backtester():
    """Test backtester with synthetic data and SMA crossover."""
    print("\n" + "=" * 50)
    print("  TEST 2: Backtester Engine")
    print("=" * 50)

    # Create synthetic data with uptrend (SMA crossover should profit)
    db_path = Path("/tmp/kronos_test/prices.db")
    create_synthetic_data(db_path, n_candles=2000, trend=0.0002)

    bt = Backtester(
        symbol="BTC/USDT",
        timeframe="1h",
        initial_capital=10000,
        fee_rate=0.0006,
        use_liquidation_data=False,
        prices_db=db_path,
    )

    strategy = SMACrossover(fast_period=10, slow_period=30)
    report = bt.run(strategy)

    print(report.summary())

    assert report.total_trades > 0, "Expected some trades"
    print(f"✅ Backtester test PASSED ({report.total_trades} trades)")

    return bt, report


def test_compare():
    """Test strategy comparison."""
    print("\n" + "=" * 50)
    print("  TEST 3: Strategy Comparison")
    print("=" * 50)

    db_path = Path("/tmp/kronos_test/prices.db")

    bt = Backtester(
        symbol="BTC/USDT",
        timeframe="1h",
        initial_capital=10000,
        use_liquidation_data=False,
        prices_db=db_path,
    )

    reports = {}
    for fast, slow in [(5, 20), (10, 30), (20, 50)]:
        name = f"SMA_{fast}_{slow}"
        strategy = SMACrossover(fast_period=fast, slow_period=slow)
        reports[name] = bt.run(strategy)

    print(compare_reports(reports))
    print("✅ Comparison test PASSED")


def test_robustness():
    """Test robustness suite."""
    print("\n" + "=" * 50)
    print("  TEST 4: Robustness Test Suite")
    print("=" * 50)

    db_path = Path("/tmp/kronos_test/prices.db")

    bt = Backtester(
        symbol="BTC/USDT",
        timeframe="1h",
        initial_capital=10000,
        use_liquidation_data=False,
        prices_db=db_path,
    )

    suite = RobustnessTestSuite(
        backtester=bt,
        strategy_class=SMACrossover,
        strategy_params={"fast_period": 10, "slow_period": 30},
    )

    # Run all 5 tests
    results = suite.run_all(
        param_ranges={
            "fast_period": [5, 8, 10, 12, 15, 20],
            "slow_period": [20, 25, 30, 40, 50, 60],
        },
        n_monte_carlo=50,
        n_walk_windows=5,
        quarter_days=90,
    )

    print(results.summary())
    print(f"✅ Robustness test PASSED ({results.tests_passed}/{results.tests_total})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀 KRONOS BACKTESTER - INTEGRATION TESTS")
    print("=" * 50)

    test_metrics()
    test_backtester()
    test_compare()
    test_robustness()

    print("\n" + "=" * 50)
    print("  ALL TESTS COMPLETE ✅")
    print("=" * 50)

    # Cleanup
    import shutil
    shutil.rmtree("/tmp/kronos_test", ignore_errors=True)

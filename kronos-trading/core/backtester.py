#!/usr/bin/env python3
"""
Kronos Backtester
=================
Core backtesting engine for running strategies against historical data.

Features:
- Loads OHLCV + liquidation data from SQLite
- Runs any BaseStrategy subclass
- Handles position management (long/short)
- Tracks P&L with fees
- Produces PerformanceReport

Usage:
    from core.backtester import Backtester
    from strategies.momentum.cascade_ride import CascadeRide

    bt = Backtester(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date="2025-01-01",
        end_date="2025-12-31",
    )
    report = bt.run(CascadeRide(threshold=50000))
    print(report.summary())
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.metrics import Trade, PerformanceReport, calculate_metrics
from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal

log = logging.getLogger("backtester")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_FEES = 0.0006  # 0.06% taker fee (Binance Futures)


@dataclass
class Position:
    """Active position state."""
    side: str            # "long" or "short"
    entry_price: float
    entry_time: int      # timestamp ms
    quantity: float
    tag: str = ""


class Backtester:
    """
    Core backtesting engine.

    Loads data from SQLite, feeds candles to strategy,
    manages positions, and tracks performance.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        start_date: Optional[str] = None,   # "YYYY-MM-DD"
        end_date: Optional[str] = None,     # "YYYY-MM-DD"
        initial_capital: float = 10000.0,
        fee_rate: float = DEFAULT_FEES,
        leverage: float = 1.0,
        use_liquidation_data: bool = True,
        prices_db: Optional[Path] = None,
        liquidations_db: Optional[Path] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.leverage = leverage
        self.use_liquidation_data = use_liquidation_data

        self.prices_db = prices_db or (DATA_DIR / "prices.db")
        self.liquidations_db = liquidations_db or (DATA_DIR / "liquidations.db")

        # Convert dates to timestamp ms
        self.start_ms = self._date_to_ms(start_date) if start_date else None
        self.end_ms = self._date_to_ms(end_date) if end_date else None

        # State
        self.position: Optional[Position] = None
        self.equity = initial_capital
        self.trades: list[Trade] = []

    @staticmethod
    def _date_to_ms(date_str: str) -> int:
        dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    def load_candles(self) -> list[CandleData]:
        """Load OHLCV data from SQLite."""
        if not self.prices_db.exists():
            raise FileNotFoundError(f"Price database not found: {self.prices_db}")

        conn = sqlite3.connect(str(self.prices_db))

        query = """
            SELECT timestamp_ms, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND timeframe = ?
        """
        params = [self.symbol, self.timeframe]

        if self.start_ms:
            query += " AND timestamp_ms >= ?"
            params.append(self.start_ms)
        if self.end_ms:
            query += " AND timestamp_ms <= ?"
            params.append(self.end_ms)

        query += " ORDER BY timestamp_ms ASC"

        rows = conn.execute(query, params).fetchall()
        conn.close()

        candles = []
        for row in rows:
            candles.append(CandleData(
                timestamp_ms=row[0],
                open=row[1],
                high=row[2],
                low=row[3],
                close=row[4],
                volume=row[5],
            ))

        log.info(f"Loaded {len(candles)} candles for {self.symbol} {self.timeframe}")
        return candles

    def enrich_with_liquidations(self, candles: list[CandleData]) -> list[CandleData]:
        """Attach liquidation data to candles (matching by time window)."""
        if not self.use_liquidation_data or not self.liquidations_db.exists():
            return candles

        # Map symbol format: BTC/USDC:USDC -> BTCUSDC, BTC/USDT -> BTCUSDT
        bn_symbol = self.symbol.split(":")[0].replace("/", "")

        conn = sqlite3.connect(str(self.liquidations_db))

        # Get timeframe duration in ms
        tf_ms = self._timeframe_to_ms(self.timeframe)

        for candle in candles:
            window_start = candle.timestamp_ms
            window_end = window_start + tf_ms

            row = conn.execute("""
                SELECT
                    COALESCE(SUM(usd_value), 0),
                    COALESCE(SUM(CASE WHEN side='BUY' THEN usd_value ELSE 0 END), 0),
                    COALESCE(SUM(CASE WHEN side='SELL' THEN usd_value ELSE 0 END), 0),
                    COUNT(*)
                FROM liquidations
                WHERE symbol = ?
                  AND timestamp_ms >= ?
                  AND timestamp_ms < ?
            """, (bn_symbol, window_start, window_end)).fetchone()

            if row:
                candle.liquidation_usd = row[0]
                candle.short_liq_usd = row[1]
                candle.long_liq_usd = row[2]
                candle.liq_count = row[3]

        conn.close()
        log.info(f"Enriched candles with liquidation data")
        return candles

    @staticmethod
    def _timeframe_to_ms(tf: str) -> int:
        """Convert timeframe string to milliseconds."""
        multipliers = {
            "1m": 60_000,
            "3m": 180_000,
            "5m": 300_000,
            "15m": 900_000,
            "30m": 1_800_000,
            "1h": 3_600_000,
            "2h": 7_200_000,
            "4h": 14_400_000,
            "6h": 21_600_000,
            "8h": 28_800_000,
            "12h": 43_200_000,
            "1d": 86_400_000,
        }
        return multipliers.get(tf, 3_600_000)

    def _open_position(self, candle: CandleData, signal: Signal):
        """Open a new position."""
        side = "long" if signal.direction == 1 else "short"
        price = candle.close

        # Position size based on equity, leverage, and signal strength
        notional = self.equity * self.leverage * signal.strength
        quantity = notional / price

        # Entry fee
        fee = notional * self.fee_rate
        self.equity -= fee

        self.position = Position(
            side=side,
            entry_price=price,
            entry_time=candle.timestamp_ms,
            quantity=quantity,
            tag=signal.tag,
        )

    def _close_position(self, candle: CandleData) -> Trade:
        """Close current position and record trade."""
        pos = self.position
        exit_price = candle.close

        # Calculate P&L
        if pos.side == "long":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        pnl_pct = (pnl / (pos.entry_price * pos.quantity)) * 100

        # Exit fee
        notional = exit_price * pos.quantity
        fee = notional * self.fee_rate
        pnl -= fee

        self.equity += pnl

        trade = Trade(
            entry_time=pos.entry_time,
            exit_time=candle.timestamp_ms,
            symbol=self.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fees=fee * 2,  # entry + exit
            tag=pos.tag,
        )

        self.trades.append(trade)
        self.position = None
        return trade

    def run(self, strategy: BaseStrategy) -> PerformanceReport:
        """
        Run a strategy against historical data.

        Args:
            strategy: BaseStrategy subclass instance

        Returns:
            PerformanceReport with full metrics
        """
        # Reset state
        self.position = None
        self.equity = self.initial_capital
        self.trades = []

        # Load data
        candles = self.load_candles()
        if not candles:
            log.warning(f"No candle data found for {self.symbol} {self.timeframe}")
            return PerformanceReport(initial_capital=self.initial_capital)

        candles = self.enrich_with_liquidations(candles)

        # Initialize strategy
        strategy.on_init()

        log.info(
            f"Running {strategy.name} on {self.symbol} {self.timeframe} | "
            f"{len(candles)} candles | Capital: ${self.initial_capital:,.0f}"
        )

        # --- Main loop ---
        for candle in candles:
            # Update strategy history
            strategy._update_history(candle)

            # Get signal
            signal = strategy.on_candle(candle)

            if signal is None or signal.direction is None:
                continue  # Hold

            # --- Position management ---
            if signal.direction == 0:
                # Close signal
                if self.position:
                    trade = self._close_position(candle)
                    strategy.on_trade(trade.pnl, trade.pnl_pct)

            elif signal.direction == 1:
                # Long signal
                if self.position and self.position.side == "short":
                    trade = self._close_position(candle)
                    strategy.on_trade(trade.pnl, trade.pnl_pct)
                if not self.position:
                    self._open_position(candle, signal)

            elif signal.direction == -1:
                # Short signal
                if self.position and self.position.side == "long":
                    trade = self._close_position(candle)
                    strategy.on_trade(trade.pnl, trade.pnl_pct)
                if not self.position:
                    self._open_position(candle, signal)

        # Close any open position at end
        if self.position and candles:
            trade = self._close_position(candles[-1])
            strategy.on_trade(trade.pnl, trade.pnl_pct)

        # Calculate metrics
        report = calculate_metrics(
            self.trades,
            initial_capital=self.initial_capital,
        )

        log.info(
            f"Completed: {report.total_trades} trades | "
            f"Return: {report.total_return_pct:+.2f}% | "
            f"DD: {report.max_drawdown_pct:.2f}% | "
            f"Sharpe: {report.sharpe_ratio:.2f}"
        )

        return report

    def run_multi_symbol(
        self,
        strategy_class: type,
        symbols: list[str],
        strategy_params: Optional[dict] = None,
    ) -> dict[str, PerformanceReport]:
        """
        Run same strategy across multiple symbols.

        Args:
            strategy_class: BaseStrategy subclass (not instance)
            symbols: List of symbols to test
            strategy_params: Params to pass to strategy constructor

        Returns:
            Dict of symbol -> PerformanceReport
        """
        results = {}
        params = strategy_params or {}

        for symbol in symbols:
            self.symbol = symbol
            strategy = strategy_class(**params)
            results[symbol] = self.run(strategy)

        return results

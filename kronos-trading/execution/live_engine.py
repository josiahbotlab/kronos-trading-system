#!/usr/bin/env python3
"""
Kronos Live Trading Engine
============================
Main execution loop that runs strategies in real-time.

Modes:
- Paper: Simulated trading with real market data
- Live: Real execution on Hyperliquid (future)

Architecture:
1. Fetch latest candle data from price DB
2. Enrich with latest liquidation data
3. Feed to strategy -> get signal
4. Risk check -> position manager
5. Execute via exchange connector
6. Alert via Telegram
7. Sleep until next candle

Usage:
    # Paper trading with liq_bb_combo on BTC 5m
    python live_engine.py --strategy liq_bb_combo --symbol BTC-USD --timeframe 5m

    # Multi-strategy
    python live_engine.py --strategy cascade_ride,liq_bb_combo --symbol BTC-USD

    # With custom capital and risk
    python live_engine.py --strategy liq_bb_combo --capital 5000 --max-dd 10

    # Status check
    python live_engine.py --status
"""

import argparse
import importlib
import json
import logging
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.position_manager import PositionManager, RiskConfig, LivePosition
from execution.exchange_connector import ExchangeConnector
from execution.telegram_notifier import TelegramNotifier
from strategies.templates.base_strategy import BaseStrategy, CandleData, Signal

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent.parent / "data"
STRATEGY_DIRS = [
    Path(__file__).parent.parent / "strategies" / "momentum",
    Path(__file__).parent.parent / "strategies" / "reversal",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("engine")

# Timeframe to seconds mapping
TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}
TF_MS = {k: v * 1000 for k, v in TF_SECONDS.items()}


# ---------------------------------------------------------------------------
# Strategy Loading
# ---------------------------------------------------------------------------
def load_strategy(name: str, params: dict = None) -> BaseStrategy:
    """Load a strategy by name from the strategies directory."""
    for sdir in STRATEGY_DIRS:
        for py_file in sdir.glob("*.py"):
            if py_file.stem == "__init__":
                continue
            module_name = f"strategies.{sdir.stem}.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseStrategy)
                        and attr is not BaseStrategy
                    ):
                        instance = attr(**(params or {}))
                        if instance.name == name:
                            return instance
            except Exception:
                continue

    raise ValueError(f"Strategy '{name}' not found")


def list_strategies() -> list[str]:
    """List all available strategy names."""
    names = []
    for sdir in STRATEGY_DIRS:
        for py_file in sdir.glob("*.py"):
            if py_file.stem == "__init__":
                continue
            module_name = f"strategies.{sdir.stem}.{py_file.stem}"
            try:
                module = importlib.import_module(module_name)
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseStrategy)
                        and attr is not BaseStrategy
                    ):
                        names.append(attr().name)
            except Exception:
                continue
    return names


# ---------------------------------------------------------------------------
# Data Loading (from collectors' DBs)
# ---------------------------------------------------------------------------
def load_recent_candles(
    symbol: str,
    timeframe: str,
    n_candles: int = 500,
) -> list[CandleData]:
    """Load the most recent N candles from prices.db."""
    db_path = DATA_DIR / "prices.db"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))

    # Try new format first (BTC-USD)
    rows = conn.execute(
        """SELECT timestamp_ms, open, high, low, close, volume
           FROM ohlcv
           WHERE symbol = ? AND timeframe = ?
           ORDER BY timestamp_ms DESC
           LIMIT ?""",
        (symbol, timeframe, n_candles),
    ).fetchall()

    # If no data found, try old format (BTC/USDC:USDC) for backward compatibility
    if not rows:
        old_symbol_map = {
            "BTC-USD": "BTC/USDC:USDC",
            "ETH-USD": "ETH/USDC:USDC",
            "SOL-USD": "SOL/USDC:USDC",
            "DOGE-USD": "DOGE/USDC:USDC",
            "XRP-USD": "XRP/USDC:USDC",
            "ADA-USD": "ADA/USDC:USDC",
            "AVAX-USD": "AVAX/USDC:USDC",
            "LINK-USD": "LINK/USDC:USDC",
        }
        old_symbol = old_symbol_map.get(symbol)
        if old_symbol:
            log.warning(f"No data for {symbol}, trying old format {old_symbol}")
            rows = conn.execute(
                """SELECT timestamp_ms, open, high, low, close, volume
                   FROM ohlcv
                   WHERE symbol = ? AND timeframe = ?
                   ORDER BY timestamp_ms DESC
                   LIMIT ?""",
                (old_symbol, timeframe, n_candles),
            ).fetchall()

    conn.close()

    candles = []
    for row in reversed(rows):  # oldest first
        candles.append(CandleData(
            timestamp_ms=row[0],
            open=row[1], high=row[2], low=row[3],
            close=row[4], volume=row[5],
        ))
    return candles


def enrich_candle_liquidations(candle: CandleData, symbol: str, tf_ms: int):
    """Attach liquidation data to a single candle."""
    db_path = DATA_DIR / "liquidations.db"
    if not db_path.exists():
        return

    # Map symbol: BTC-USD -> BTCUSDT (liqs from Binance)
    base = symbol.split("-")[0]
    bn_symbol = f"{base}USDT"

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        """SELECT
            COALESCE(SUM(usd_value), 0),
            COALESCE(SUM(CASE WHEN side='BUY' THEN usd_value ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN side='SELL' THEN usd_value ELSE 0 END), 0),
            COUNT(*)
        FROM liquidations
        WHERE symbol = ? AND timestamp_ms >= ? AND timestamp_ms < ?""",
        (bn_symbol, candle.timestamp_ms, candle.timestamp_ms + tf_ms),
    ).fetchone()
    conn.close()

    if row:
        candle.liquidation_usd = row[0]
        candle.short_liq_usd = row[1]
        candle.long_liq_usd = row[2]
        candle.liq_count = row[3]


def enrich_candles_batch(candles: list[CandleData], symbol: str, tf_ms: int):
    """Enrich all candles with liquidation data."""
    db_path = DATA_DIR / "liquidations.db"
    if not db_path.exists():
        return

    base = symbol.split("-")[0]
    bn_symbol = f"{base}USDT"

    conn = sqlite3.connect(str(db_path))
    for candle in candles:
        row = conn.execute(
            """SELECT
                COALESCE(SUM(usd_value), 0),
                COALESCE(SUM(CASE WHEN side='BUY' THEN usd_value ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN side='SELL' THEN usd_value ELSE 0 END), 0),
                COUNT(*)
            FROM liquidations
            WHERE symbol = ? AND timestamp_ms >= ? AND timestamp_ms < ?""",
            (bn_symbol, candle.timestamp_ms, candle.timestamp_ms + tf_ms),
        ).fetchone()
        if row:
            candle.liquidation_usd = row[0]
            candle.short_liq_usd = row[1]
            candle.long_liq_usd = row[2]
            candle.liq_count = row[3]
    conn.close()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class LiveEngine:
    """
    Main trading engine loop.

    Runs one or more strategies against real-time data,
    manages positions through the position manager,
    and executes via the exchange connector.
    """

    def __init__(
        self,
        strategies: list[tuple[str, BaseStrategy]],  # (symbol, strategy) pairs
        timeframe: str = "5m",
        capital: float = 1000.0,
        paper: bool = True,
        risk_config: RiskConfig = None,
        status_interval: int = 3600,  # send status every hour
    ):
        self.strategies = strategies
        self.timeframe = timeframe
        self.tf_ms = TF_MS.get(timeframe, 300000)
        self.tf_seconds = TF_SECONDS.get(timeframe, 300)
        self.paper = paper
        self.running = False

        # Components
        self.position_mgr = PositionManager(
            initial_capital=capital,
            risk_config=risk_config or RiskConfig(),
            paper=paper,
        )
        self.exchange = ExchangeConnector(paper=paper)
        self.telegram = TelegramNotifier()

        # State
        self.last_candle_ts: dict[str, int] = {}  # symbol -> last processed candle ts
        self.strategy_histories: dict[str, list[CandleData]] = {}  # key -> candle history
        self.last_status_time = 0.0
        self.status_interval = status_interval
        self.cycle_count = 0

    def _strategy_key(self, symbol: str, strategy: BaseStrategy) -> str:
        return f"{strategy.name}_{symbol}"

    def start(self):
        """Start the engine."""
        mode = "PAPER" if self.paper else "LIVE"
        log.info(f"{'='*60}")
        log.info(f"  KRONOS TRADING ENGINE [{mode}]")
        log.info(f"{'='*60}")
        log.info(f"  Capital: ${self.position_mgr.equity:,.2f}")
        log.info(f"  Timeframe: {self.timeframe}")
        log.info(f"  Strategies:")
        for symbol, strategy in self.strategies:
            log.info(f"    {strategy.name} on {symbol}")
        log.info(f"{'='*60}")

        # Send startup notification
        strats = ", ".join(f"{s.name}({sym})" for sym, s in self.strategies)
        self.telegram.send_alert(
            f"Engine Started [{mode}]",
            f"Capital: ${self.position_mgr.equity:,.2f}\n"
            f"Timeframe: {self.timeframe}\n"
            f"Strategies: {strats}",
        )

        # Initialize strategies with historical data
        self._warmup_strategies()

        # Main loop
        self.running = True
        try:
            self._run_loop()
        except KeyboardInterrupt:
            log.info("Keyboard interrupt received")
        except Exception as e:
            log.error(f"Engine error: {e}", exc_info=True)
            self.telegram.send_error(str(e))
        finally:
            self.stop()

    def stop(self):
        """Graceful shutdown."""
        self.running = False

        # Close all positions
        for pos_id in list(self.position_mgr.positions.keys()):
            pos = self.position_mgr.positions[pos_id]
            price = self.exchange.get_price(pos.symbol)
            if price:
                trade = self.position_mgr.close_position(pos_id, price, "engine_shutdown")
                if trade:
                    self.telegram.send_trade_close({**trade, "reason": "engine_shutdown"})

        # Final status
        status = self.position_mgr.get_status()
        self.telegram.send_status(status)

        log.info(f"Engine stopped. Final equity: ${status['equity']:,.2f} "
                 f"({status['total_return_pct']:+.2f}%)")

    def _warmup_strategies(self):
        """Load historical candles to initialize strategy state."""
        log.info("Warming up strategies with historical data...")

        seen_symbols = set()
        for symbol, strategy in self.strategies:
            key = self._strategy_key(symbol, strategy)
            candles = load_recent_candles(symbol, self.timeframe, n_candles=500)

            if candles:
                enrich_candles_batch(candles, symbol, self.tf_ms)

            strategy.on_init()

            # Feed history to strategy (without acting on signals)
            for candle in candles:
                strategy._update_history(candle)
                strategy.on_candle(candle)  # warm up indicators

            self.strategy_histories[key] = candles
            if candles:
                self.last_candle_ts[symbol] = candles[-1].timestamp_ms

            log.info(f"  {strategy.name} on {symbol}: {len(candles)} candles loaded")
            seen_symbols.add(symbol)

    def _run_loop(self):
        """Main trading loop."""
        while self.running:
            try:
                self.cycle_count += 1
                cycle_start = time.time()

                # 1. Check for new candles
                new_candles = self._check_new_candles()

                # 2. Process new candles through strategies
                if new_candles:
                    self._process_candles(new_candles)

                # 3. Update positions with current prices
                self._update_positions()

                # 4. Periodic status
                if time.time() - self.last_status_time > self.status_interval:
                    self._send_status()
                    self.last_status_time = time.time()

                # 5. Sleep until next check
                # Check more frequently than candle period for price updates
                sleep_time = min(self.tf_seconds / 3, 60)
                elapsed = time.time() - cycle_start

                if elapsed < sleep_time:
                    time.sleep(sleep_time - elapsed)

            except Exception as e:
                log.error(f"Cycle error: {e}", exc_info=True)
                time.sleep(30)

    def _check_new_candles(self) -> dict[str, list[CandleData]]:
        """Check for new candles since last processed."""
        new_candles = {}
        seen = set()

        for symbol, _ in self.strategies:
            if symbol in seen:
                continue
            seen.add(symbol)

            last_ts = self.last_candle_ts.get(symbol, 0)

            # Load candles newer than what we've seen
            db_path = DATA_DIR / "prices.db"
            if not db_path.exists():
                continue

            conn = sqlite3.connect(str(db_path))
            rows = conn.execute(
                """SELECT timestamp_ms, open, high, low, close, volume
                   FROM ohlcv
                   WHERE symbol = ? AND timeframe = ? AND timestamp_ms > ?
                   ORDER BY timestamp_ms ASC""",
                (symbol, self.timeframe, last_ts),
            ).fetchall()
            conn.close()

            if rows:
                candles = []
                for row in rows:
                    c = CandleData(
                        timestamp_ms=row[0],
                        open=row[1], high=row[2], low=row[3],
                        close=row[4], volume=row[5],
                    )
                    enrich_candle_liquidations(c, symbol, self.tf_ms)
                    candles.append(c)

                new_candles[symbol] = candles
                self.last_candle_ts[symbol] = candles[-1].timestamp_ms

                if self.cycle_count % 10 == 0:  # don't spam logs
                    log.info(f"New candles: {symbol} +{len(candles)}")

        return new_candles

    def _process_candles(self, new_candles: dict[str, list[CandleData]]):
        """Process new candles through strategies and generate signals."""
        for symbol, strategy in self.strategies:
            if symbol not in new_candles:
                continue

            for candle in new_candles[symbol]:
                strategy._update_history(candle)
                signal = strategy.on_candle(candle)

                if signal is None or signal.direction is None:
                    continue

                self._handle_signal(symbol, strategy, signal, candle)

    def _handle_signal(
        self,
        symbol: str,
        strategy: BaseStrategy,
        signal: Signal,
        candle: CandleData,
    ):
        """Process a strategy signal."""
        key = self._strategy_key(symbol, strategy)

        # Check if we have a position from this strategy
        existing_pos = None
        for pos in self.position_mgr.positions.values():
            if pos.strategy == strategy.name and pos.symbol == symbol:
                existing_pos = pos
                break

        if signal.direction == 0:
            # CLOSE signal
            if existing_pos:
                price = self.exchange.get_price(symbol) or candle.close
                trade = self.position_mgr.close_position(
                    existing_pos.id, price, f"signal_close|{signal.tag}"
                )
                if trade:
                    self.telegram.send_trade_close({**trade, "reason": signal.tag})

        elif signal.direction in (1, -1):
            # LONG or SHORT signal
            side = "long" if signal.direction == 1 else "short"

            # Close opposite position first
            if existing_pos and existing_pos.side != side:
                price = self.exchange.get_price(symbol) or candle.close
                trade = self.position_mgr.close_position(
                    existing_pos.id, price, "signal_reverse"
                )
                if trade:
                    self.telegram.send_trade_close({**trade, "reason": "reverse"})
                existing_pos = None

            # Open new position if not already positioned
            if not existing_pos:
                entry_price = self.exchange.get_price(symbol) or candle.close

                # Calculate position size
                notional = self.position_mgr.calculate_position_size(
                    signal_strength=signal.strength,
                )
                quantity = notional / entry_price

                # Calculate stop/take profit from signal metadata
                stop_loss = signal.metadata.get("stop_loss") if signal.metadata else None
                take_profit = signal.metadata.get("take_profit") if signal.metadata else None
                trailing_pct = signal.metadata.get("trailing_stop_pct") if signal.metadata else None

                # Execute order
                if side == "long":
                    result = self.exchange.market_buy(symbol, quantity)
                else:
                    result = self.exchange.market_sell(symbol, quantity)

                if result.success:
                    pos = self.position_mgr.open_position(
                        strategy=strategy.name,
                        symbol=symbol,
                        side=side,
                        entry_price=result.fill_price,
                        quantity=result.fill_quantity,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        trailing_stop_pct=trailing_pct,
                        tag=signal.tag,
                    )

                    if pos:
                        self.telegram.send_trade_open({
                            "side": side,
                            "symbol": symbol,
                            "strategy": strategy.name,
                            "entry_price": result.fill_price,
                            "notional_usd": result.fill_price * result.fill_quantity,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                            "tag": signal.tag,
                            "mode": "paper" if self.paper else "live",
                        })

                    log.info(
                        f"Signal: {strategy.name} -> {side.upper()} {symbol} | "
                        f"Strength: {signal.strength:.2f} | Tag: {signal.tag}"
                    )
                else:
                    log.warning(f"Order failed: {result.error}")

    def _update_positions(self):
        """Update positions with current prices, check stops."""
        if not self.position_mgr.positions:
            return

        prices = self.exchange.get_all_prices()
        if not prices:
            return

        to_close = self.position_mgr.update_prices(prices)
        for pos_id, price, reason in to_close:
            trade = self.position_mgr.close_position(pos_id, price, reason)
            if trade:
                self.telegram.send_trade_close({**trade, "reason": reason})
                log.info(f"Auto-closed {pos_id}: {reason} @ {price:.2f}")

    def _send_status(self):
        """Send periodic status update."""
        status = self.position_mgr.get_status()
        self.position_mgr.save_equity_snapshot()

        # Log to console
        log.info(
            f"Status | Equity: ${status['equity']:,.2f} "
            f"({status['total_return_pct']:+.2f}%) | "
            f"DD: {status['drawdown_pct']:.2f}% | "
            f"Positions: {status['positions_count']} | "
            f"Daily PnL: ${status['daily_pnl']:+,.2f}"
        )

        # Send to Telegram
        self.telegram.send_status(status)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def show_status():
    """Show current engine status from DB."""
    db_path = DATA_DIR / "execution.db"
    if not db_path.exists():
        print("No execution database found. Engine hasn't run yet.")
        return

    conn = sqlite3.connect(str(db_path))

    # Recent trades
    trades = conn.execute(
        """SELECT strategy, symbol, side, entry_price, exit_price,
                  pnl_usd, pnl_pct, mode,
                  datetime(entry_time, 'unixepoch'), datetime(exit_time, 'unixepoch')
           FROM trades ORDER BY exit_time DESC LIMIT 10"""
    ).fetchall()

    # Summary
    summary = conn.execute(
        """SELECT mode, COUNT(*), SUM(pnl_usd),
                  SUM(CASE WHEN pnl_usd > 0 THEN 1 ELSE 0 END),
                  AVG(pnl_pct)
           FROM trades GROUP BY mode"""
    ).fetchall()

    # Latest equity
    equity = conn.execute(
        "SELECT equity, daily_pnl, mode FROM equity_snapshots ORDER BY timestamp DESC LIMIT 1"
    ).fetchone()

    conn.close()

    print("=" * 60)
    print("  KRONOS EXECUTION STATUS")
    print("=" * 60)

    if equity:
        print(f"\n💰 Current Equity: ${equity[0]:,.2f}")
        print(f"📅 Daily PnL: ${equity[1]:+,.2f}")
        print(f"📊 Mode: {equity[2].upper()}")

    if summary:
        for mode, count, total_pnl, wins, avg_pct in summary:
            wr = (wins / count * 100) if count > 0 else 0
            print(f"\n📊 {mode.upper()} Summary:")
            print(f"   Trades: {count} | Win Rate: {wr:.1f}%")
            print(f"   Total PnL: ${total_pnl:+,.2f} | Avg: {avg_pct:+.2f}%")

    if trades:
        print(f"\n📋 Recent Trades:")
        print(f"{'Strategy':15s} {'Symbol':16s} {'Side':6s} {'PnL':>10s} {'%':>8s} {'Mode':>6s}")
        print("-" * 65)
        for t in trades:
            pnl_str = f"${t[5]:+,.2f}" if t[5] else "$0.00"
            pct_str = f"{t[6]:+.2f}%" if t[6] else "0.00%"
            print(f"{t[0]:15s} {t[1]:16s} {t[2]:6s} {pnl_str:>10s} {pct_str:>8s} {t[7]:>6s}")
    else:
        print("\nNo trades recorded yet.")


def main():
    parser = argparse.ArgumentParser(description="Kronos Live Trading Engine")
    parser.add_argument("--strategy", type=str, help="Strategy name(s), comma-separated")
    parser.add_argument("--symbol", type=str, default="BTC-USD",
                        help="Trading symbol (default: BTC-USD)")
    parser.add_argument("--timeframe", type=str, default="5m",
                        help="Candle timeframe (default: 5m)")
    parser.add_argument("--capital", type=float, default=1000.0,
                        help="Starting capital (default: 1000)")
    parser.add_argument("--max-dd", type=float, default=15.0,
                        help="Max drawdown %% before circuit breaker (default: 15)")
    parser.add_argument("--max-pos", type=int, default=5,
                        help="Max concurrent positions (default: 5)")
    parser.add_argument("--params", type=str, default=None,
                        help="Strategy params as JSON string")
    parser.add_argument("--status", action="store_true",
                        help="Show current execution status")
    parser.add_argument("--list", action="store_true",
                        help="List available strategies")
    parser.add_argument("--live", action="store_true",
                        help="Enable live trading (default: paper)")

    args = parser.parse_args()

    if args.status:
        show_status()
        return

    if args.list:
        print("Available strategies:")
        for name in list_strategies():
            print(f"  {name}")
        return

    if not args.strategy:
        parser.print_help()
        return

    # Parse strategy params
    params = json.loads(args.params) if args.params else {}

    # Build strategy list
    strategy_pairs = []
    symbols = [s.strip() for s in args.symbol.split(",")]
    strategy_names = [s.strip() for s in args.strategy.split(",")]

    for strat_name in strategy_names:
        for symbol in symbols:
            strategy = load_strategy(strat_name, params)
            strategy_pairs.append((symbol, strategy))

    # Risk config
    risk = RiskConfig(
        max_drawdown_pct=args.max_dd,
        max_positions=args.max_pos,
    )

    # Create and start engine
    engine = LiveEngine(
        strategies=strategy_pairs,
        timeframe=args.timeframe,
        capital=args.capital,
        paper=not args.live,
        risk_config=risk,
    )

    # Handle signals
    def handle_signal(sig, frame):
        log.info("Shutdown signal received...")
        engine.running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    engine.start()


if __name__ == "__main__":
    main()

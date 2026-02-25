"""
Kronos Trading - Portfolio Manager
Orchestrates multiple strategies through the full trade lifecycle.
Connects: Strategy signals → Risk Manager → Executor → Tracking

This is the main entry point for live/paper trading.

Updated to use Coinbase Advanced Trade API instead of Hyperliquid.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("kronos.portfolio")


@dataclass
class Signal:
    """Trading signal from a strategy."""
    strategy: str
    coin: str
    side: str            # "long" or "short"
    entry_px: float
    stop_px: float
    take_profit_px: Optional[float] = None
    confidence: float = 0.5    # 0-1 scale
    win_rate: Optional[float] = None   # From backtest
    avg_rr: Optional[float] = None     # From backtest
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class LiveTrade:
    """Active trade being managed."""
    trade_id: int
    strategy: str
    coin: str
    side: str
    size: float
    entry_px: float
    stop_px: float
    take_profit_px: Optional[float] = None
    stop_oid: Optional[int] = None
    tp_oid: Optional[int] = None
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "open"


class PortfolioManager:
    """
    Manages the full lifecycle of trades across multiple strategies.

    Usage:
        from execution.coinbase_executor import CoinbaseExecutor
        from execution.risk_manager import RiskManager
        from execution.portfolio import PortfolioManager, Signal

        executor = CoinbaseExecutor.from_config()
        risk_mgr = RiskManager()
        portfolio = PortfolioManager(executor, risk_mgr)
        
        # Strategy generates a signal
        signal = Signal(
            strategy="cascade_p99",
            coin="BTC",
            side="long",
            entry_px=95000,
            stop_px=93000,
            take_profit_px=100000,
            win_rate=0.55,
            avg_rr=2.5,
        )
        
        # Portfolio handles everything
        trade = portfolio.execute_signal(signal)
        
        # Periodic management
        portfolio.manage_positions()
    """

    def __init__(self, executor, risk_manager, db_path: str = "data/portfolio.db"):
        self.executor = executor
        self.risk = risk_manager
        self.db_path = db_path
        self.active_trades: dict[int, LiveTrade] = {}
        self._init_db()
        self._load_active_trades()

    def _init_db(self):
        """Initialize portfolio tracking database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS live_trades (
                trade_id INTEGER PRIMARY KEY,
                strategy TEXT NOT NULL,
                coin TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_px REAL NOT NULL,
                stop_px REAL,
                take_profit_px REAL,
                stop_oid INTEGER,
                tp_oid INTEGER,
                opened_at TEXT NOT NULL,
                closed_at TEXT,
                exit_px REAL,
                pnl REAL,
                status TEXT DEFAULT 'open'
            );
            
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                coin TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_px REAL,
                stop_px REAL,
                confidence REAL,
                action TEXT NOT NULL,
                reason TEXT
            );
        """)
        conn.close()

    def _load_active_trades(self):
        """Load open trades from DB on startup."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM live_trades WHERE status = 'open'"
        ).fetchall()
        conn.close()

        for r in rows:
            trade = LiveTrade(
                trade_id=r[0], strategy=r[1], coin=r[2], side=r[3],
                size=r[4], entry_px=r[5], stop_px=r[6],
                take_profit_px=r[7], stop_oid=r[8], tp_oid=r[9],
                status="open",
            )
            self.active_trades[trade.trade_id] = trade

        if self.active_trades:
            logger.info(f"📂 Loaded {len(self.active_trades)} active trades from DB")

    # ─── Signal Execution ────────────────────────────────────────

    def execute_signal(self, signal: Signal) -> Optional[LiveTrade]:
        """
        Execute a trading signal through the full pipeline:
        1. Pre-trade risk check
        2. Position sizing
        3. Order execution
        4. SL/TP placement
        5. Trade recording
        """
        logger.info(f"📡 Signal: {signal.strategy} → {signal.side} {signal.coin} @ {signal.entry_px}")

        # Get current state
        try:
            state = self.executor.get_account_state()
        except Exception as e:
            self._log_signal(signal, "rejected", f"Failed to get account state: {e}")
            return None

        # Update peak equity
        self.risk.update_peak_equity(state.equity)

        # 1. Pre-trade risk check
        total_exposure = sum(abs(p.size * p.entry_px) for p in state.positions)
        check = self.risk.pre_trade_check(
            strategy=signal.strategy,
            coin=signal.coin,
            side=signal.side,
            equity=state.equity,
            open_positions=len(state.positions),
            total_exposure_usd=total_exposure,
        )

        if not check.passed:
            logger.warning(f"🚫 Risk check failed: {check.blocked_reason}")
            self._log_signal(signal, "blocked", check.blocked_reason)
            return None

        # 2. Position sizing
        sizing = self.risk.calculate_position_size(
            strategy=signal.strategy,
            coin=signal.coin,
            entry_px=signal.entry_px,
            stop_px=signal.stop_px,
            equity=state.equity,
            win_rate=signal.win_rate,
            avg_rr=signal.avg_rr,
        )

        if not sizing.approved:
            logger.warning(f"🚫 Sizing rejected: {sizing.reason}")
            self._log_signal(signal, "sizing_rejected", sizing.reason)
            return None

        logger.info(
            f"📏 Size: {sizing.size} {signal.coin} (${sizing.size_usd:.2f}, "
            f"risk ${sizing.risk_usd:.2f}, method={sizing.method})"
        )

        # 3. Set leverage
        self.executor.set_leverage(signal.coin, sizing.leverage)

        # 4. Execute market order
        is_buy = signal.side == "long"
        result = self.executor.market_open(
            coin=signal.coin,
            is_buy=is_buy,
            size=sizing.size,
        )

        if not result.success:
            logger.error(f"❌ Order failed: {result.error}")
            self._log_signal(signal, "execution_failed", result.error)
            return None

        entry_px = result.avg_px or signal.entry_px
        logger.info(f"✅ Filled: {sizing.size} {signal.coin} @ {entry_px}")

        # 5. Record trade in risk manager
        trade_id = self.risk.record_trade_open(
            strategy=signal.strategy,
            coin=signal.coin,
            side=signal.side,
            size=sizing.size,
            entry_px=entry_px,
        )

        # 6. Place SL/TP orders
        trade = LiveTrade(
            trade_id=trade_id,
            strategy=signal.strategy,
            coin=signal.coin,
            side=signal.side,
            size=sizing.size,
            entry_px=entry_px,
            stop_px=signal.stop_px,
            take_profit_px=signal.take_profit_px,
        )

        # Stop loss
        sl_result = self.executor.stop_loss(
            coin=signal.coin,
            is_buy=not is_buy,  # Opposite side to close
            size=sizing.size,
            trigger_px=signal.stop_px,
        )
        if sl_result.success:
            trade.stop_oid = sl_result.oid
            logger.info(f"🛡️ SL set @ {signal.stop_px}")

        # Take profit
        if signal.take_profit_px:
            tp_result = self.executor.take_profit(
                coin=signal.coin,
                is_buy=not is_buy,
                size=sizing.size,
                trigger_px=signal.take_profit_px,
            )
            if tp_result.success:
                trade.tp_oid = tp_result.oid
                logger.info(f"🎯 TP set @ {signal.take_profit_px}")

        # 7. Save to DB and memory
        self._save_trade(trade)
        self.active_trades[trade_id] = trade
        self._log_signal(signal, "executed", f"Trade #{trade_id}, size={sizing.size}")

        return trade

    # ─── Position Management ─────────────────────────────────────

    def manage_positions(self):
        """
        Periodic position management loop. Call this on a timer.
        Checks if positions were closed by SL/TP and records P&L.
        """
        if not self.active_trades:
            return

        try:
            state = self.executor.get_account_state()
        except Exception as e:
            logger.error(f"Failed to get state for position management: {e}")
            return

        self.risk.update_peak_equity(state.equity)

        # Build map of current positions
        position_map = {p.coin: p for p in state.positions}

        # Check each active trade
        closed_ids = []
        for trade_id, trade in self.active_trades.items():
            if trade.coin not in position_map:
                # Position no longer exists - was closed (SL/TP hit or manual)
                self._handle_trade_closed(trade, position_map)
                closed_ids.append(trade_id)
            else:
                # Position still open - check if size changed
                pos = position_map[trade.coin]
                if abs(pos.size) < abs(trade.size) * 0.1:  # Nearly closed
                    self._handle_trade_closed(trade, position_map)
                    closed_ids.append(trade_id)

        # Remove closed trades from active
        for tid in closed_ids:
            del self.active_trades[tid]

    def _handle_trade_closed(self, trade: LiveTrade, position_map: dict):
        """Handle a trade that was closed (by SL/TP or manually)."""
        # Try to get exit price from fills
        fills = self.executor.get_user_fills(limit=20)
        exit_px = None
        for fill in fills:
            if fill.get("coin") == trade.coin:
                exit_px = float(fill.get("px", 0))
                break

        if exit_px is None:
            # Estimate from current price
            mid = self.executor.get_mid_price(trade.coin)
            exit_px = mid or trade.entry_px

        # Calculate P&L
        if trade.side == "long":
            pnl = (exit_px - trade.entry_px) * trade.size
        else:
            pnl = (trade.entry_px - exit_px) * trade.size

        # Record in risk manager
        self.risk.record_trade_close(trade.trade_id, exit_px, pnl)

        # Update DB
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE live_trades SET
                closed_at = ?, exit_px = ?, pnl = ?, status = 'closed'
            WHERE trade_id = ?
        """, (datetime.now(timezone.utc).isoformat(), exit_px, pnl, trade.trade_id))
        conn.commit()
        conn.close()

        emoji = "💰" if pnl > 0 else "💸"
        logger.info(
            f"{emoji} Trade #{trade.trade_id} closed: {trade.side} {trade.size} {trade.coin} "
            f"@ {exit_px:.2f} | P&L: ${pnl:.2f}"
        )

    # ─── Manual Controls ─────────────────────────────────────────

    def close_trade(self, trade_id: int) -> bool:
        """Manually close a trade."""
        trade = self.active_trades.get(trade_id)
        if not trade:
            logger.warning(f"Trade #{trade_id} not found in active trades")
            return False

        # Cancel any open SL/TP orders
        if trade.stop_oid:
            self.executor.cancel_order(trade.coin, trade.stop_oid)
        if trade.tp_oid:
            self.executor.cancel_order(trade.coin, trade.tp_oid)

        # Close position
        result = self.executor.market_close(trade.coin)
        if result.success:
            logger.info(f"✅ Manually closed trade #{trade_id}")
            # manage_positions will pick up the closure
            return True
        else:
            logger.error(f"Failed to close trade #{trade_id}: {result.error}")
            return False

    def close_all(self) -> int:
        """Emergency: close all positions."""
        logger.warning("🚨 CLOSING ALL POSITIONS")
        closed = 0

        # Cancel all orders first
        self.executor.cancel_all()

        # Close each position
        state = self.executor.get_account_state()
        for pos in state.positions:
            result = self.executor.market_close(pos.coin)
            if result.success:
                closed += 1

        # Run management to record closures
        time.sleep(2)
        self.manage_positions()

        logger.info(f"Closed {closed} positions")
        return closed

    # ─── Status & Reporting ──────────────────────────────────────

    def get_status(self) -> dict:
        """Get full portfolio status."""
        try:
            state = self.executor.get_account_state()
            risk_summary = self.risk.get_risk_summary(state.equity)
        except Exception:
            state = None
            risk_summary = self.risk.get_risk_summary(0)

        return {
            "account": {
                "equity": state.equity if state else 0,
                "available": state.available_balance if state else 0,
                "positions": len(state.positions) if state else 0,
            } if state else None,
            "active_trades": len(self.active_trades),
            "trades": [
                {
                    "id": t.trade_id,
                    "strategy": t.strategy,
                    "coin": t.coin,
                    "side": t.side,
                    "size": t.size,
                    "entry": t.entry_px,
                    "stop": t.stop_px,
                    "tp": t.take_profit_px,
                }
                for t in self.active_trades.values()
            ],
            "risk": risk_summary,
            "testnet": self.executor.testnet,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_trade_history(self, limit: int = 50) -> list:
        """Get closed trade history."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT * FROM live_trades WHERE status = 'closed'
            ORDER BY closed_at DESC LIMIT ?
        """, (limit,)).fetchall()
        conn.close()

        return [
            {
                "trade_id": r[0], "strategy": r[1], "coin": r[2],
                "side": r[3], "size": r[4], "entry": r[5],
                "stop": r[6], "tp": r[7], "exit": r[12],
                "pnl": r[13], "opened": r[10], "closed": r[11],
            }
            for r in rows
        ]

    # ─── Helpers ─────────────────────────────────────────────────

    def _save_trade(self, trade: LiveTrade):
        """Save trade to database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO live_trades 
            (trade_id, strategy, coin, side, size, entry_px, stop_px, 
             take_profit_px, stop_oid, tp_oid, opened_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """, (
            trade.trade_id, trade.strategy, trade.coin, trade.side,
            trade.size, trade.entry_px, trade.stop_px,
            trade.take_profit_px, trade.stop_oid, trade.tp_oid,
            trade.opened_at.isoformat(),
        ))
        conn.commit()
        conn.close()

    def _log_signal(self, signal: Signal, action: str, reason: str = ""):
        """Log a signal and its disposition."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO signal_log 
            (timestamp, strategy, coin, side, entry_px, stop_px, confidence, action, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal.timestamp.isoformat(), signal.strategy, signal.coin,
            signal.side, signal.entry_px, signal.stop_px,
            signal.confidence, action, reason,
        ))
        conn.commit()
        conn.close()

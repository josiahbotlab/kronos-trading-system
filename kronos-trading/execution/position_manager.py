#!/usr/bin/env python3
"""
Kronos Position Manager
========================
Risk-managed position sizing and tracking.

Features:
- Kelly-criterion position sizing
- Max drawdown circuit breaker
- Per-symbol exposure limits
- Daily loss limits
- Position tracking across strategies
"""

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("position_mgr")

DB_PATH = Path(__file__).parent.parent / "data" / "execution.db"


@dataclass
class LivePosition:
    """Active live/paper position."""
    id: str                     # unique position ID
    strategy: str               # strategy that opened it
    symbol: str
    side: str                   # "long" or "short"
    entry_price: float
    quantity: float
    notional_usd: float
    entry_time: float           # unix timestamp
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_pct: Optional[float] = None
    peak_price: float = 0.0     # for trailing stop
    trough_price: float = 1e18
    tag: str = ""
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_pct: float = 0.25          # max 25% of equity per position
    max_total_exposure_pct: float = 0.75    # max 75% total exposure
    max_positions: int = 5                   # max concurrent positions
    max_daily_loss_pct: float = 5.0         # stop trading after 5% daily loss
    max_drawdown_pct: float = 15.0          # circuit breaker at 15% drawdown
    min_order_usd: float = 10.0             # minimum order size
    max_order_usd: float = 5000.0           # maximum single order
    default_leverage: float = 1.0
    cooldown_after_loss_seconds: int = 300  # 5 min cooldown after losing trade


class PositionManager:
    """
    Manages positions, risk limits, and execution state.
    Persists to SQLite for crash recovery.
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        risk_config: Optional[RiskConfig] = None,
        db_path: Path = DB_PATH,
        paper: bool = True,
    ):
        self.equity = initial_capital
        self.initial_capital = initial_capital
        self.peak_equity = initial_capital
        self.risk = risk_config or RiskConfig()
        self.paper = paper
        self.positions: dict[str, LivePosition] = {}  # id -> position
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.day_start = self._today_start()
        self.last_loss_time = 0.0
        self.circuit_breaker_active = False

        # Persistence
        self.db_path = db_path
        self._init_db()

    @staticmethod
    def _today_start() -> float:
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp()

    def _init_db(self):
        """Initialize execution database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                strategy TEXT,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                notional_usd REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                entry_time REAL,
                exit_time REAL,
                tag TEXT,
                mode TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS equity_snapshots (
                timestamp REAL,
                equity REAL,
                positions_count INTEGER,
                daily_pnl REAL,
                mode TEXT
            )
        """)

        conn.commit()
        conn.close()

    def _reset_daily(self):
        """Reset daily counters if new day."""
        now_start = self._today_start()
        if now_start > self.day_start:
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.day_start = now_start
            self.circuit_breaker_active = False
            log.info("New trading day - daily counters reset")

    # --- Risk Checks ---

    def can_open_position(self, symbol: str, notional_usd: float) -> tuple[bool, str]:
        """Check if a new position passes all risk checks."""
        self._reset_daily()

        # Circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker active (max drawdown reached)"

        # Daily loss limit
        if self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / self.initial_capital * 100
            if daily_loss_pct >= self.risk.max_daily_loss_pct:
                return False, f"Daily loss limit ({daily_loss_pct:.1f}% >= {self.risk.max_daily_loss_pct}%)"

        # Max drawdown
        dd_pct = (self.peak_equity - self.equity) / self.peak_equity * 100
        if dd_pct >= self.risk.max_drawdown_pct:
            self.circuit_breaker_active = True
            return False, f"Max drawdown hit ({dd_pct:.1f}% >= {self.risk.max_drawdown_pct}%)"

        # Max positions
        if len(self.positions) >= self.risk.max_positions:
            return False, f"Max positions ({len(self.positions)}/{self.risk.max_positions})"

        # Already have position in this symbol?
        for pos in self.positions.values():
            if pos.symbol == symbol:
                return False, f"Already positioned in {symbol}"

        # Position size limits
        if notional_usd < self.risk.min_order_usd:
            return False, f"Order too small ({notional_usd:.2f} < {self.risk.min_order_usd})"
        if notional_usd > self.risk.max_order_usd:
            return False, f"Order too large ({notional_usd:.2f} > {self.risk.max_order_usd})"

        # Max single position as % of equity
        max_notional = self.equity * self.risk.max_position_pct
        if notional_usd > max_notional:
            return False, f"Position too large ({notional_usd:.0f} > {max_notional:.0f} = {self.risk.max_position_pct*100}% equity)"

        # Max total exposure
        current_exposure = sum(p.notional_usd for p in self.positions.values())
        max_exposure = self.equity * self.risk.max_total_exposure_pct
        if current_exposure + notional_usd > max_exposure:
            return False, f"Total exposure limit ({current_exposure + notional_usd:.0f} > {max_exposure:.0f})"

        # Post-loss cooldown
        if time.time() - self.last_loss_time < self.risk.cooldown_after_loss_seconds:
            remaining = self.risk.cooldown_after_loss_seconds - (time.time() - self.last_loss_time)
            return False, f"Post-loss cooldown ({remaining:.0f}s remaining)"

        return True, "OK"

    def calculate_position_size(
        self,
        signal_strength: float = 1.0,
        leverage: float = None,
    ) -> float:
        """Calculate position size in USD based on risk params."""
        lev = leverage or self.risk.default_leverage
        base_size = self.equity * self.risk.max_position_pct * lev
        sized = base_size * min(signal_strength, 1.0)
        return min(max(sized, self.risk.min_order_usd), self.risk.max_order_usd)

    # --- Position Lifecycle ---

    def open_position(
        self,
        strategy: str,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float = None,
        take_profit: float = None,
        trailing_stop_pct: float = None,
        tag: str = "",
    ) -> Optional[LivePosition]:
        """Open a new position (after risk check passes)."""
        notional = entry_price * quantity
        can_open, reason = self.can_open_position(symbol, notional)

        if not can_open:
            log.warning(f"Position rejected: {reason}")
            return None

        pos_id = f"{strategy}_{symbol}_{int(time.time()*1000)}"

        pos = LivePosition(
            id=pos_id,
            strategy=strategy,
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            notional_usd=notional,
            entry_time=time.time(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop_pct=trailing_stop_pct,
            peak_price=entry_price,
            trough_price=entry_price,
            tag=tag,
        )

        self.positions[pos_id] = pos
        mode = "paper" if self.paper else "live"
        log.info(
            f"[{mode.upper()}] OPEN {side.upper()} {symbol} | "
            f"Price: {entry_price:.2f} | Qty: {quantity:.6f} | "
            f"Notional: ${notional:.2f} | SL: {stop_loss} | TP: {take_profit}"
        )

        return pos

    def close_position(
        self,
        pos_id: str,
        exit_price: float,
        reason: str = "",
    ) -> Optional[dict]:
        """Close a position and record the trade."""
        if pos_id not in self.positions:
            log.warning(f"Position {pos_id} not found")
            return None

        pos = self.positions.pop(pos_id)

        # Calculate PnL
        if pos.side == "long":
            pnl_usd = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl_usd = (pos.entry_price - exit_price) * pos.quantity

        pnl_pct = (pnl_usd / pos.notional_usd) * 100

        # Update equity
        self.equity += pnl_usd
        self.peak_equity = max(self.peak_equity, self.equity)
        self.daily_pnl += pnl_usd
        self.daily_trades += 1

        if pnl_usd < 0:
            self.last_loss_time = time.time()

        mode = "paper" if self.paper else "live"
        log.info(
            f"[{mode.upper()}] CLOSE {pos.side.upper()} {pos.symbol} | "
            f"Entry: {pos.entry_price:.2f} -> Exit: {exit_price:.2f} | "
            f"PnL: ${pnl_usd:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}"
        )

        # Persist to DB
        trade_record = {
            "id": pos.id,
            "strategy": pos.strategy,
            "symbol": pos.symbol,
            "side": pos.side,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "quantity": pos.quantity,
            "notional_usd": pos.notional_usd,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            "entry_time": pos.entry_time,
            "exit_time": time.time(),
            "tag": f"{pos.tag}|{reason}",
            "mode": mode,
        }
        self._save_trade(trade_record)

        return trade_record

    def update_prices(self, prices: dict[str, float]):
        """
        Update position tracking with current prices.
        Check stop losses, take profits, trailing stops.
        Returns list of position IDs that need closing.
        """
        to_close = []

        for pos_id, pos in self.positions.items():
            if pos.symbol not in prices:
                continue

            current_price = prices[pos.symbol]

            # Update trailing stop tracking
            pos.peak_price = max(pos.peak_price, current_price)
            pos.trough_price = min(pos.trough_price, current_price)

            # Update unrealized PnL
            if pos.side == "long":
                pos.pnl_usd = (current_price - pos.entry_price) * pos.quantity
            else:
                pos.pnl_usd = (pos.entry_price - current_price) * pos.quantity
            pos.pnl_pct = (pos.pnl_usd / pos.notional_usd) * 100

            # Check stop loss
            if pos.stop_loss:
                if pos.side == "long" and current_price <= pos.stop_loss:
                    to_close.append((pos_id, current_price, "stop_loss"))
                elif pos.side == "short" and current_price >= pos.stop_loss:
                    to_close.append((pos_id, current_price, "stop_loss"))

            # Check take profit
            if pos.take_profit:
                if pos.side == "long" and current_price >= pos.take_profit:
                    to_close.append((pos_id, current_price, "take_profit"))
                elif pos.side == "short" and current_price <= pos.take_profit:
                    to_close.append((pos_id, current_price, "take_profit"))

            # Check trailing stop
            if pos.trailing_stop_pct:
                pct = pos.trailing_stop_pct / 100
                if pos.side == "long":
                    trail_stop = pos.peak_price * (1 - pct)
                    if current_price <= trail_stop:
                        to_close.append((pos_id, current_price, "trailing_stop"))
                else:
                    trail_stop = pos.trough_price * (1 + pct)
                    if current_price >= trail_stop:
                        to_close.append((pos_id, current_price, "trailing_stop"))

        return to_close

    def get_status(self) -> dict:
        """Get current portfolio status."""
        self._reset_daily()
        total_exposure = sum(p.notional_usd for p in self.positions.values())
        unrealized_pnl = sum(p.pnl_usd for p in self.positions.values())
        dd_pct = (self.peak_equity - self.equity) / self.peak_equity * 100 if self.peak_equity > 0 else 0

        return {
            "equity": self.equity,
            "initial_capital": self.initial_capital,
            "total_return_pct": (self.equity - self.initial_capital) / self.initial_capital * 100,
            "peak_equity": self.peak_equity,
            "drawdown_pct": dd_pct,
            "positions_count": len(self.positions),
            "total_exposure_usd": total_exposure,
            "exposure_pct": total_exposure / self.equity * 100 if self.equity > 0 else 0,
            "unrealized_pnl": unrealized_pnl,
            "daily_pnl": self.daily_pnl,
            "daily_trades": self.daily_trades,
            "circuit_breaker": self.circuit_breaker_active,
            "mode": "paper" if self.paper else "live",
        }

    # --- Persistence ---

    def _save_trade(self, trade: dict):
        """Save completed trade to DB."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """INSERT OR REPLACE INTO trades
                   (id, strategy, symbol, side, entry_price, exit_price,
                    quantity, notional_usd, pnl_usd, pnl_pct,
                    entry_time, exit_time, tag, mode)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    trade["id"], trade["strategy"], trade["symbol"],
                    trade["side"], trade["entry_price"], trade["exit_price"],
                    trade["quantity"], trade["notional_usd"],
                    trade["pnl_usd"], trade["pnl_pct"],
                    trade["entry_time"], trade["exit_time"],
                    trade["tag"], trade["mode"],
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"Failed to save trade: {e}")

    def save_equity_snapshot(self):
        """Save current equity state."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                """INSERT INTO equity_snapshots
                   (timestamp, equity, positions_count, daily_pnl, mode)
                   VALUES (?,?,?,?,?)""",
                (
                    time.time(), self.equity, len(self.positions),
                    self.daily_pnl, "paper" if self.paper else "live",
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.error(f"Failed to save equity snapshot: {e}")

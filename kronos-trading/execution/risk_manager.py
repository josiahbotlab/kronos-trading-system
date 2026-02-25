"""
Kronos Trading - Risk Manager
Position sizing (Kelly Criterion), drawdown limits, kill switches.
Enforces Moon Dev's incubation rules: $10-$100 per strategy max.

This is the LAST LINE OF DEFENSE before any order hits the exchange.
"""

import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("kronos.risk")

# ─── Risk Configuration Defaults ─────────────────────────────────

DEFAULT_RISK_CONFIG = {
    # Global limits
    "max_portfolio_drawdown_pct": 15.0,     # Kill switch: halt ALL trading
    "max_daily_loss_pct": 5.0,              # Pause trading for the day
    "max_open_positions": 5,                # Max concurrent positions
    "max_portfolio_exposure_pct": 50.0,     # Max % of equity in positions
    
    # Per-strategy limits (Moon Dev incubation)
    "default_strategy_budget": 100.0,       # $100 max per strategy during incubation
    "min_strategy_budget": 10.0,            # $10 minimum
    "max_strategy_drawdown_pct": 30.0,      # Kill individual strategy at 30% DD
    
    # Position sizing
    "default_risk_per_trade_pct": 1.0,      # Risk 1% of equity per trade
    "max_risk_per_trade_pct": 2.0,          # Hard cap on single trade risk
    "kelly_fraction": 0.25,                 # Use 1/4 Kelly (conservative)
    
    # Per-coin limits
    "max_position_size_usd": 500.0,         # Max single position value
    "max_leverage": 5,                      # Max leverage per position
    
    # Kill switch
    "kill_switch_active": False,             # Emergency halt
    "kill_switch_reason": "",
}


@dataclass
class SizingResult:
    """Output from position sizing calculation."""
    approved: bool
    size: float = 0.0              # Position size in base units
    size_usd: float = 0.0         # Position value in USD
    risk_usd: float = 0.0         # Dollar risk on this trade
    leverage: int = 1
    reason: str = ""
    method: str = ""               # "kelly", "fixed_fraction", "capped"


@dataclass
class RiskCheck:
    """Pre-trade risk validation result."""
    passed: bool
    checks: dict = field(default_factory=dict)
    blocked_reason: Optional[str] = None


class RiskManager:
    """
    Manages all risk controls for Kronos trading system.
    
    Usage:
        rm = RiskManager("config/kronos.json")
        
        # Before any trade:
        check = rm.pre_trade_check(strategy="cascade_p99", coin="BTC", side="long")
        if not check.passed:
            print(f"BLOCKED: {check.blocked_reason}")
            return
        
        # Size the position:
        sizing = rm.calculate_position_size(
            strategy="cascade_p99",
            coin="BTC",
            entry_px=95000,
            stop_px=93000,
            equity=10000,
            win_rate=0.55,    # from backtest
            avg_rr=2.0,       # from backtest
        )
        
        # After trade completes:
        rm.record_trade(strategy="cascade_p99", coin="BTC", pnl=45.50)
    """

    def __init__(self, config_path: str = "config/kronos.json", db_path: str = "data/risk.db"):
        self.config = self._load_config(config_path)
        self.db_path = db_path
        self._init_db()
        
        # Runtime state
        self._daily_pnl = 0.0
        self._daily_reset_date = datetime.now(timezone.utc).date()
        self._peak_equity = 0.0

    def _load_config(self, config_path: str) -> dict:
        """Load risk config, falling back to defaults."""
        config = DEFAULT_RISK_CONFIG.copy()
        path = Path(config_path)
        if path.exists():
            with open(path) as f:
                file_config = json.load(f)
            risk_config = file_config.get("risk", {})
            config.update(risk_config)
        else:
            logger.warning(f"No config at {config_path}, using defaults")
        return config

    def _init_db(self):
        """Initialize risk tracking database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                strategy TEXT NOT NULL,
                coin TEXT NOT NULL,
                side TEXT NOT NULL,
                size REAL NOT NULL,
                entry_px REAL,
                exit_px REAL,
                pnl REAL DEFAULT 0,
                status TEXT DEFAULT 'open'
            );
            
            CREATE TABLE IF NOT EXISTS strategy_budgets (
                strategy TEXT PRIMARY KEY,
                allocated_budget REAL NOT NULL,
                current_exposure REAL DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                peak_value REAL DEFAULT 0,
                max_drawdown REAL DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                win_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'incubating'
            );
            
            CREATE TABLE IF NOT EXISTS kill_switch_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                reason TEXT NOT NULL,
                equity_at_trigger REAL,
                action TEXT NOT NULL
            );
        """)
        conn.close()

    # ─── Pre-Trade Validation ────────────────────────────────────

    def pre_trade_check(
        self,
        strategy: str,
        coin: str,
        side: str,
        equity: float = 0,
        open_positions: int = 0,
        total_exposure_usd: float = 0,
    ) -> RiskCheck:
        """
        Run all pre-trade risk checks. Must pass before ANY order.
        
        Returns RiskCheck with passed=True/False and detailed check results.
        """
        checks = {}

        # 1. Kill switch
        if self.config["kill_switch_active"]:
            return RiskCheck(
                passed=False,
                checks={"kill_switch": False},
                blocked_reason=f"Kill switch active: {self.config['kill_switch_reason']}",
            )
        checks["kill_switch"] = True

        # 2. Daily loss limit
        self._check_daily_reset()
        max_daily = equity * (self.config["max_daily_loss_pct"] / 100)
        if self._daily_pnl < -max_daily and equity > 0:
            return RiskCheck(
                passed=False,
                checks={**checks, "daily_loss": False},
                blocked_reason=f"Daily loss limit hit: ${self._daily_pnl:.2f} (max -${max_daily:.2f})",
            )
        checks["daily_loss"] = True

        # 3. Max open positions
        if open_positions >= self.config["max_open_positions"]:
            return RiskCheck(
                passed=False,
                checks={**checks, "max_positions": False},
                blocked_reason=f"Max positions reached: {open_positions}/{self.config['max_open_positions']}",
            )
        checks["max_positions"] = True

        # 4. Portfolio exposure
        if equity > 0:
            exposure_pct = (total_exposure_usd / equity) * 100
            if exposure_pct >= self.config["max_portfolio_exposure_pct"]:
                return RiskCheck(
                    passed=False,
                    checks={**checks, "exposure": False},
                    blocked_reason=f"Portfolio exposure too high: {exposure_pct:.1f}% (max {self.config['max_portfolio_exposure_pct']}%)",
                )
        checks["exposure"] = True

        # 5. Strategy budget check
        budget = self._get_strategy_budget(strategy)
        if budget and budget["status"] == "killed":
            return RiskCheck(
                passed=False,
                checks={**checks, "strategy_budget": False},
                blocked_reason=f"Strategy '{strategy}' was killed (max drawdown exceeded)",
            )
        checks["strategy_budget"] = True

        # 6. Portfolio drawdown
        if equity > 0 and self._peak_equity > 0:
            dd_pct = ((self._peak_equity - equity) / self._peak_equity) * 100
            if dd_pct >= self.config["max_portfolio_drawdown_pct"]:
                self._activate_kill_switch(
                    f"Portfolio drawdown {dd_pct:.1f}% exceeds {self.config['max_portfolio_drawdown_pct']}%",
                    equity,
                )
                return RiskCheck(
                    passed=False,
                    checks={**checks, "portfolio_drawdown": False},
                    blocked_reason=f"Portfolio drawdown kill switch: {dd_pct:.1f}%",
                )
        checks["portfolio_drawdown"] = True

        return RiskCheck(passed=True, checks=checks)

    # ─── Position Sizing ─────────────────────────────────────────

    def calculate_position_size(
        self,
        strategy: str,
        coin: str,
        entry_px: float,
        stop_px: float,
        equity: float,
        win_rate: Optional[float] = None,
        avg_rr: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> SizingResult:
        """
        Calculate position size using Kelly Criterion (fractional).
        Falls back to fixed-fraction if no win_rate/avg_rr provided.
        
        Args:
            strategy: Strategy name (for budget lookup)
            entry_px: Expected entry price
            stop_px: Stop loss price
            equity: Current account equity
            win_rate: Historical win rate (0-1)
            avg_rr: Average reward:risk ratio
            current_price: Current market price (for unit conversion)
        """
        price = current_price or entry_px

        # Calculate risk per unit (distance to stop)
        risk_per_unit = abs(entry_px - stop_px)
        if risk_per_unit == 0:
            return SizingResult(approved=False, reason="Stop price equals entry price")

        # Determine risk budget
        if win_rate is not None and avg_rr is not None and win_rate > 0:
            # Kelly Criterion: f* = (p * b - q) / b
            # where p = win_rate, q = 1-p, b = avg reward/risk
            p = win_rate
            q = 1 - p
            b = avg_rr
            kelly_full = (p * b - q) / b if b > 0 else 0
            kelly_full = max(0, kelly_full)  # Never negative

            # Use fractional Kelly (conservative)
            kelly_frac = kelly_full * self.config["kelly_fraction"]
            risk_pct = min(kelly_frac * 100, self.config["max_risk_per_trade_pct"])
            method = f"kelly_{self.config['kelly_fraction']}"

            if kelly_full <= 0:
                return SizingResult(
                    approved=False,
                    reason=f"Kelly says don't trade (WR={win_rate:.1%}, RR={avg_rr:.1f})",
                    method="kelly_reject",
                )
        else:
            # Fixed fraction fallback
            risk_pct = self.config["default_risk_per_trade_pct"]
            method = "fixed_fraction"

        # Dollar risk
        risk_usd = equity * (risk_pct / 100)

        # Position size in base units
        size = risk_usd / risk_per_unit
        size_usd = size * price

        # Apply caps
        # Cap 1: Max position size
        if size_usd > self.config["max_position_size_usd"]:
            size_usd = self.config["max_position_size_usd"]
            size = size_usd / price
            risk_usd = size * risk_per_unit
            method += "_capped_max_pos"

        # Cap 2: Strategy budget (incubation)
        budget = self._get_strategy_budget(strategy)
        if budget:
            remaining = budget["allocated_budget"] - budget["current_exposure"]
            if size_usd > remaining:
                size_usd = max(0, remaining)
                size = size_usd / price if price > 0 else 0
                risk_usd = size * risk_per_unit
                method += "_capped_budget"

        if size <= 0 or size_usd <= 0:
            return SizingResult(
                approved=False,
                reason="Position size too small after caps",
                method=method,
            )

        # Determine leverage needed
        leverage = 1
        margin_needed = size_usd / leverage
        if margin_needed > equity * 0.2:  # If using > 20% of equity
            leverage = min(
                int(size_usd / (equity * 0.1)) + 1,
                self.config["max_leverage"],
            )

        return SizingResult(
            approved=True,
            size=round(size, 6),
            size_usd=round(size_usd, 2),
            risk_usd=round(risk_usd, 2),
            leverage=leverage,
            method=method,
            reason=f"Risk {risk_pct:.2f}% of equity",
        )

    # ─── Strategy Budget Management ──────────────────────────────

    def allocate_strategy(self, strategy: str, budget: float = None) -> dict:
        """Allocate budget for a new strategy incubation."""
        if budget is None:
            budget = self.config["default_strategy_budget"]
        budget = max(
            self.config["min_strategy_budget"],
            min(budget, self.config["default_strategy_budget"]),
        )

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO strategy_budgets 
            (strategy, allocated_budget, current_exposure, total_pnl, peak_value, status)
            VALUES (?, ?, 0, 0, ?, 'incubating')
        """, (strategy, budget, budget))
        conn.commit()
        conn.close()

        logger.info(f"💰 Allocated ${budget} budget for strategy '{strategy}'")
        return {"strategy": strategy, "budget": budget, "status": "incubating"}

    def _get_strategy_budget(self, strategy: str) -> Optional[dict]:
        """Get strategy budget info."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT * FROM strategy_budgets WHERE strategy = ?", (strategy,)
        ).fetchone()
        conn.close()

        if not row:
            return None
        return {
            "strategy": row[0],
            "allocated_budget": row[1],
            "current_exposure": row[2],
            "total_pnl": row[3],
            "peak_value": row[4],
            "max_drawdown": row[5],
            "trade_count": row[6],
            "win_count": row[7],
            "status": row[8],
        }

    def get_all_strategies(self) -> list:
        """Get all strategy budgets."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT * FROM strategy_budgets ORDER BY strategy").fetchall()
        conn.close()
        return [
            {
                "strategy": r[0], "budget": r[1], "exposure": r[2],
                "pnl": r[3], "peak": r[4], "max_dd": r[5],
                "trades": r[6], "wins": r[7], "status": r[8],
            }
            for r in rows
        ]

    # ─── Trade Recording ─────────────────────────────────────────

    def record_trade_open(
        self, strategy: str, coin: str, side: str, size: float, entry_px: float
    ) -> int:
        """Record a new trade opening. Returns trade ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            INSERT INTO trade_log (timestamp, strategy, coin, side, size, entry_px, status)
            VALUES (?, ?, ?, ?, ?, ?, 'open')
        """, (datetime.now(timezone.utc).isoformat(), strategy, coin, side, size, entry_px))
        trade_id = cursor.lastrowid

        # Update strategy exposure
        conn.execute("""
            UPDATE strategy_budgets 
            SET current_exposure = current_exposure + ?
            WHERE strategy = ?
        """, (size * entry_px, strategy))

        conn.commit()
        conn.close()
        logger.info(f"📝 Trade #{trade_id} opened: {side} {size} {coin} @ {entry_px}")
        return trade_id

    def record_trade_close(self, trade_id: int, exit_px: float, pnl: float):
        """Record a trade closing with P&L."""
        conn = sqlite3.connect(self.db_path)

        # Get trade info
        trade = conn.execute(
            "SELECT strategy, coin, side, size, entry_px FROM trade_log WHERE id = ?",
            (trade_id,),
        ).fetchone()

        if not trade:
            logger.warning(f"Trade #{trade_id} not found")
            conn.close()
            return

        strategy = trade[0]
        size = trade[3]
        entry_px = trade[4]

        # Update trade
        conn.execute("""
            UPDATE trade_log SET exit_px = ?, pnl = ?, status = 'closed'
            WHERE id = ?
        """, (exit_px, pnl, trade_id))

        # Update strategy stats
        conn.execute("""
            UPDATE strategy_budgets SET
                current_exposure = MAX(0, current_exposure - ?),
                total_pnl = total_pnl + ?,
                trade_count = trade_count + 1,
                win_count = win_count + CASE WHEN ? > 0 THEN 1 ELSE 0 END
            WHERE strategy = ?
        """, (size * entry_px, pnl, pnl, strategy))

        # Check strategy drawdown
        budget = self._get_strategy_budget(strategy)
        if budget:
            current_value = budget["allocated_budget"] + budget["total_pnl"] + pnl
            peak = max(budget["peak_value"], current_value)
            dd = ((peak - current_value) / peak) * 100 if peak > 0 else 0

            conn.execute("""
                UPDATE strategy_budgets SET
                    peak_value = MAX(peak_value, ?),
                    max_drawdown = MAX(max_drawdown, ?)
                WHERE strategy = ?
            """, (current_value, dd, strategy))

            # Kill strategy if drawdown too high
            if dd >= self.config["max_strategy_drawdown_pct"]:
                conn.execute(
                    "UPDATE strategy_budgets SET status = 'killed' WHERE strategy = ?",
                    (strategy,),
                )
                logger.warning(f"💀 Strategy '{strategy}' KILLED - drawdown {dd:.1f}%")

        conn.commit()
        conn.close()

        # Update daily P&L
        self._daily_pnl += pnl
        logger.info(f"📝 Trade #{trade_id} closed: P&L ${pnl:.2f} (daily: ${self._daily_pnl:.2f})")

    # ─── Kill Switch ─────────────────────────────────────────────

    def _activate_kill_switch(self, reason: str, equity: float):
        """Activate emergency kill switch."""
        self.config["kill_switch_active"] = True
        self.config["kill_switch_reason"] = reason

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO kill_switch_log (timestamp, reason, equity_at_trigger, action)
            VALUES (?, ?, ?, 'activated')
        """, (datetime.now(timezone.utc).isoformat(), reason, equity))
        conn.commit()
        conn.close()

        logger.critical(f"🚨 KILL SWITCH ACTIVATED: {reason}")

    def deactivate_kill_switch(self):
        """Manually deactivate kill switch after review."""
        self.config["kill_switch_active"] = False
        self.config["kill_switch_reason"] = ""

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO kill_switch_log (timestamp, reason, equity_at_trigger, action)
            VALUES (?, 'Manual deactivation', 0, 'deactivated')
        """, (datetime.now(timezone.utc).isoformat(),))
        conn.commit()
        conn.close()

        logger.info("✅ Kill switch deactivated")

    def update_peak_equity(self, equity: float):
        """Update peak equity for drawdown tracking."""
        if equity > self._peak_equity:
            self._peak_equity = equity

    # ─── Daily Reset ─────────────────────────────────────────────

    def _check_daily_reset(self):
        """Reset daily P&L counter at midnight UTC."""
        today = datetime.now(timezone.utc).date()
        if today != self._daily_reset_date:
            logger.info(f"📅 Daily reset: yesterday P&L was ${self._daily_pnl:.2f}")
            self._daily_pnl = 0.0
            self._daily_reset_date = today

    # ─── Reporting ───────────────────────────────────────────────

    def get_risk_summary(self, equity: float = 0) -> dict:
        """Get current risk state summary."""
        strategies = self.get_all_strategies()
        dd_pct = 0
        if equity > 0 and self._peak_equity > 0:
            dd_pct = ((self._peak_equity - equity) / self._peak_equity) * 100

        return {
            "kill_switch": self.config["kill_switch_active"],
            "daily_pnl": round(self._daily_pnl, 2),
            "peak_equity": round(self._peak_equity, 2),
            "current_drawdown_pct": round(dd_pct, 2),
            "strategies": {
                "total": len(strategies),
                "active": sum(1 for s in strategies if s["status"] == "incubating"),
                "killed": sum(1 for s in strategies if s["status"] == "killed"),
            },
            "config": {
                "max_dd": self.config["max_portfolio_drawdown_pct"],
                "max_daily_loss": self.config["max_daily_loss_pct"],
                "max_positions": self.config["max_open_positions"],
                "kelly_fraction": self.config["kelly_fraction"],
            },
        }


# ─── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rm = RiskManager()

    # Example: allocate strategy budgets
    rm.allocate_strategy("cascade_p99", 100)
    rm.allocate_strategy("double_decay_reversal", 50)

    # Example: size a trade
    sizing = rm.calculate_position_size(
        strategy="cascade_p99",
        coin="BTC",
        entry_px=95000,
        stop_px=93000,
        equity=10000,
        win_rate=0.55,
        avg_rr=2.0,
    )
    print(f"\nSizing: {sizing}")

    # Example: risk summary
    summary = rm.get_risk_summary(equity=10000)
    print(f"\nRisk Summary: {json.dumps(summary, indent=2)}")

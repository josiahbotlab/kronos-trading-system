"""
Kronos Trading - Incubation Tracker
Implements Moon Dev's incubation methodology:
  1. Deploy with $10-$100 per strategy
  2. Run for 30+ days
  3. Only scale if live matches backtest
  4. Kill losers fast, let winners run

Tracks every strategy through its incubation lifecycle.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("kronos.incubation")

INCUBATION_DAYS = 30
MIN_TRADES_FOR_EVAL = 10
BACKTEST_MATCH_THRESHOLD = 0.7  # Live must achieve 70% of backtest metrics


@dataclass
class IncubationStatus:
    strategy: str
    status: str              # "incubating", "passed", "failed", "scaled", "killed"
    days_active: int
    budget: float
    current_value: float
    total_pnl: float
    pnl_pct: float
    trade_count: int
    win_rate: float
    max_drawdown_pct: float
    sharpe_estimate: float
    # Backtest comparison
    bt_return_pct: Optional[float] = None
    bt_win_rate: Optional[float] = None
    bt_sharpe: Optional[float] = None
    live_vs_bt_ratio: Optional[float] = None
    # Recommendation
    recommendation: str = ""
    days_remaining: int = 0


class IncubationTracker:
    """
    Tracks strategies through Moon Dev's incubation process.
    
    Usage:
        tracker = IncubationTracker()
        
        # Start incubating a strategy
        tracker.start_incubation(
            strategy="cascade_p99",
            budget=100,
            backtest_metrics={"return_pct": 45, "win_rate": 0.55, "sharpe": 2.1}
        )
        
        # Record live results (called after each trade)
        tracker.record_result("cascade_p99", pnl=12.50)
        
        # Check status
        status = tracker.evaluate("cascade_p99")
        print(status.recommendation)
        
        # Get dashboard
        dashboard = tracker.get_dashboard()
    """

    def __init__(self, db_path: str = "data/incubation.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS incubations (
                strategy TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                budget REAL NOT NULL,
                current_value REAL NOT NULL,
                peak_value REAL NOT NULL,
                total_pnl REAL DEFAULT 0,
                trade_count INTEGER DEFAULT 0,
                win_count INTEGER DEFAULT 0,
                max_drawdown_pct REAL DEFAULT 0,
                status TEXT DEFAULT 'incubating',
                ended_at TEXT,
                
                -- Backtest metrics for comparison
                bt_return_pct REAL,
                bt_win_rate REAL,
                bt_sharpe REAL,
                bt_max_dd REAL,
                bt_trade_count INTEGER,
                
                -- Daily P&L tracking for Sharpe calculation
                daily_returns TEXT DEFAULT '[]'
            );
            
            CREATE TABLE IF NOT EXISTS incubation_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                pnl REAL NOT NULL,
                cumulative_pnl REAL NOT NULL
            );
        """)
        conn.close()

    def start_incubation(
        self,
        strategy: str,
        budget: float = 100,
        backtest_metrics: Optional[dict] = None,
    ):
        """Start incubating a strategy with a budget and backtest benchmarks."""
        now = datetime.now(timezone.utc).isoformat()
        bt = backtest_metrics or {}

        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO incubations
            (strategy, started_at, budget, current_value, peak_value, total_pnl,
             trade_count, win_count, max_drawdown_pct, status,
             bt_return_pct, bt_win_rate, bt_sharpe, bt_max_dd, bt_trade_count,
             daily_returns)
            VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0, 'incubating', ?, ?, ?, ?, ?, '[]')
        """, (
            strategy, now, budget, budget, budget,
            bt.get("return_pct"), bt.get("win_rate"), bt.get("sharpe"),
            bt.get("max_drawdown"), bt.get("trade_count"),
        ))
        conn.commit()
        conn.close()

        logger.info(f"🧪 Started incubation: {strategy} with ${budget} budget")

    def record_result(self, strategy: str, pnl: float):
        """Record a trade result for an incubating strategy."""
        conn = sqlite3.connect(self.db_path)

        # Get current state
        row = conn.execute(
            "SELECT current_value, peak_value, total_pnl, trade_count, win_count, daily_returns "
            "FROM incubations WHERE strategy = ?", (strategy,)
        ).fetchone()

        if not row:
            logger.warning(f"Strategy '{strategy}' not found in incubation tracker")
            conn.close()
            return

        current_value = row[0] + pnl
        peak_value = max(row[1], current_value)
        total_pnl = row[2] + pnl
        trade_count = row[3] + 1
        win_count = row[4] + (1 if pnl > 0 else 0)

        # Calculate drawdown
        dd_pct = ((peak_value - current_value) / peak_value * 100) if peak_value > 0 else 0

        # Track daily returns for Sharpe
        daily_returns = json.loads(row[5])
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if daily_returns and daily_returns[-1][0] == today:
            daily_returns[-1][1] += pnl
        else:
            daily_returns.append([today, pnl])

        # Update
        conn.execute("""
            UPDATE incubations SET
                current_value = ?,
                peak_value = ?,
                total_pnl = ?,
                trade_count = ?,
                win_count = ?,
                max_drawdown_pct = MAX(max_drawdown_pct, ?),
                daily_returns = ?
            WHERE strategy = ?
        """, (current_value, peak_value, total_pnl, trade_count, win_count,
              dd_pct, json.dumps(daily_returns), strategy))

        # Record trade
        conn.execute("""
            INSERT INTO incubation_trades (strategy, timestamp, pnl, cumulative_pnl)
            VALUES (?, ?, ?, ?)
        """, (strategy, datetime.now(timezone.utc).isoformat(), pnl, total_pnl))

        conn.commit()
        conn.close()

    def evaluate(self, strategy: str) -> Optional[IncubationStatus]:
        """Evaluate a strategy's incubation progress."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT * FROM incubations WHERE strategy = ?", (strategy,)).fetchone()
        conn.close()

        if not row:
            return None

        started_at = datetime.fromisoformat(row[1])
        days_active = (datetime.now(timezone.utc) - started_at).days
        days_remaining = max(0, INCUBATION_DAYS - days_active)

        budget = row[2]
        current_value = row[3]
        total_pnl = row[5]
        trade_count = row[6]
        win_count = row[7]
        max_dd = row[8]
        status = row[9]

        pnl_pct = (total_pnl / budget * 100) if budget > 0 else 0
        win_rate = (win_count / trade_count) if trade_count > 0 else 0

        # Estimate Sharpe from daily returns
        daily_returns = json.loads(row[16])
        sharpe = self._estimate_sharpe(daily_returns)

        # Backtest comparison
        bt_return = row[10]
        bt_wr = row[11]
        bt_sharpe = row[12]

        live_vs_bt = None
        if bt_return and bt_return > 0 and days_active >= INCUBATION_DAYS:
            # Annualize live returns for comparison
            annualized_live = (pnl_pct / days_active) * 365 if days_active > 0 else 0
            live_vs_bt = annualized_live / bt_return if bt_return != 0 else 0

        # Generate recommendation
        recommendation = self._generate_recommendation(
            days_active=days_active,
            days_remaining=days_remaining,
            trade_count=trade_count,
            pnl_pct=pnl_pct,
            win_rate=win_rate,
            max_dd=max_dd,
            sharpe=sharpe,
            live_vs_bt=live_vs_bt,
            status=status,
        )

        return IncubationStatus(
            strategy=strategy,
            status=status,
            days_active=days_active,
            budget=budget,
            current_value=current_value,
            total_pnl=total_pnl,
            pnl_pct=pnl_pct,
            trade_count=trade_count,
            win_rate=win_rate,
            max_drawdown_pct=max_dd,
            sharpe_estimate=sharpe,
            bt_return_pct=bt_return,
            bt_win_rate=bt_wr,
            bt_sharpe=bt_sharpe,
            live_vs_bt_ratio=live_vs_bt,
            recommendation=recommendation,
            days_remaining=days_remaining,
        )

    def _estimate_sharpe(self, daily_returns: list) -> float:
        """Estimate annualized Sharpe ratio from daily P&L."""
        if len(daily_returns) < 5:
            return 0.0

        returns = [r[1] for r in daily_returns]
        mean_r = sum(returns) / len(returns)
        if len(returns) < 2:
            return 0.0

        variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        std_r = variance ** 0.5

        if std_r == 0:
            return 0.0

        # Annualize: multiply daily Sharpe by sqrt(365) for crypto
        daily_sharpe = mean_r / std_r
        return round(daily_sharpe * (365 ** 0.5), 2)

    def _generate_recommendation(self, **kwargs) -> str:
        """Generate human-readable recommendation."""
        status = kwargs["status"]
        if status in ("passed", "failed", "killed", "scaled"):
            return f"Strategy already {status}"

        days = kwargs["days_active"]
        remaining = kwargs["days_remaining"]
        trades = kwargs["trade_count"]
        pnl = kwargs["pnl_pct"]
        wr = kwargs["win_rate"]
        dd = kwargs["max_dd"]
        sharpe = kwargs["sharpe"]
        ratio = kwargs["live_vs_bt"]

        # Early kill conditions
        if dd > 30:
            return "🔴 KILL - Max drawdown exceeded 30%"
        if trades >= 10 and wr < 0.25:
            return "🔴 KILL - Win rate below 25% after 10+ trades"
        if pnl < -20:
            return "🔴 KILL - Lost more than 20% of budget"

        # Not enough data yet
        if days < 7:
            return f"⏳ WAIT - Only {days} days in, need more data"
        if trades < MIN_TRADES_FOR_EVAL:
            return f"⏳ WAIT - Only {trades} trades, need {MIN_TRADES_FOR_EVAL}+"

        # Incubation period complete
        if remaining == 0:
            if ratio is not None and ratio >= BACKTEST_MATCH_THRESHOLD:
                if sharpe >= 1.0 and pnl > 0:
                    return "🟢 SCALE - Passed incubation! Live matches backtest."
                else:
                    return "🟡 EXTEND - Matches backtest but metrics marginal. Run 2 more weeks."
            elif pnl > 0 and sharpe > 0.5:
                return "🟡 EXTEND - Profitable but underperforming backtest. Investigate."
            else:
                return "🔴 FAIL - Did not match backtest expectations"

        # Mid-incubation check
        if pnl > 0 and wr > 0.4:
            return f"🟢 ON TRACK - {remaining} days left, looking good"
        elif pnl > -5:
            return f"🟡 NEUTRAL - {remaining} days left, monitoring"
        else:
            return f"🟡 CONCERNING - Down {pnl:.1f}%, {remaining} days left"

    # ─── Strategy Status Updates ─────────────────────────────────

    def mark_passed(self, strategy: str):
        """Mark strategy as having passed incubation."""
        self._update_status(strategy, "passed")
        logger.info(f"✅ Strategy '{strategy}' PASSED incubation")

    def mark_failed(self, strategy: str):
        """Mark strategy as having failed incubation."""
        self._update_status(strategy, "failed")
        logger.info(f"❌ Strategy '{strategy}' FAILED incubation")

    def mark_scaled(self, strategy: str):
        """Mark strategy as scaled to production sizing."""
        self._update_status(strategy, "scaled")
        logger.info(f"📈 Strategy '{strategy}' SCALED to production")

    def kill(self, strategy: str):
        """Kill a strategy immediately."""
        self._update_status(strategy, "killed")
        logger.warning(f"💀 Strategy '{strategy}' KILLED")

    def _update_status(self, strategy: str, status: str):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE incubations SET status = ?, ended_at = ?
            WHERE strategy = ?
        """, (status, datetime.now(timezone.utc).isoformat(), strategy))
        conn.commit()
        conn.close()

    # ─── Dashboard ───────────────────────────────────────────────

    def get_dashboard(self) -> dict:
        """Get full incubation dashboard."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT strategy FROM incubations ORDER BY started_at DESC").fetchall()
        conn.close()

        strategies = []
        total_budget = 0
        total_pnl = 0

        for (strategy,) in rows:
            status = self.evaluate(strategy)
            if status:
                strategies.append({
                    "strategy": status.strategy,
                    "status": status.status,
                    "days": status.days_active,
                    "remaining": status.days_remaining,
                    "budget": status.budget,
                    "value": round(status.current_value, 2),
                    "pnl": round(status.total_pnl, 2),
                    "pnl_pct": round(status.pnl_pct, 2),
                    "trades": status.trade_count,
                    "win_rate": round(status.win_rate * 100, 1),
                    "max_dd": round(status.max_drawdown_pct, 1),
                    "sharpe": status.sharpe_estimate,
                    "recommendation": status.recommendation,
                })
                total_budget += status.budget
                total_pnl += status.total_pnl

        return {
            "total_strategies": len(strategies),
            "total_budget_allocated": round(total_budget, 2),
            "total_pnl": round(total_pnl, 2),
            "strategies": strategies,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def print_dashboard(self):
        """Print formatted dashboard to terminal."""
        dash = self.get_dashboard()
        
        print("\n" + "=" * 70)
        print("  KRONOS INCUBATION DASHBOARD")
        print("=" * 70)
        print(f"  Strategies: {dash['total_strategies']}  |  "
              f"Budget: ${dash['total_budget_allocated']}  |  "
              f"Total P&L: ${dash['total_pnl']:.2f}")
        print("-" * 70)

        for s in dash["strategies"]:
            emoji = {"incubating": "🧪", "passed": "✅", "failed": "❌",
                     "killed": "💀", "scaled": "📈"}.get(s["status"], "❓")
            print(f"\n  {emoji} {s['strategy']}")
            print(f"     Status: {s['status']}  |  Day {s['days']}/{INCUBATION_DAYS}")
            print(f"     Budget: ${s['budget']}  →  ${s['value']}  ({s['pnl_pct']:+.1f}%)")
            print(f"     Trades: {s['trades']}  |  WR: {s['win_rate']:.1f}%  |  "
                  f"Max DD: {s['max_dd']:.1f}%  |  Sharpe: {s['sharpe']}")
            print(f"     → {s['recommendation']}")

        print("\n" + "=" * 70)


# ─── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    tracker = IncubationTracker()

    # Demo
    tracker.start_incubation("cascade_p99", 100, {
        "return_pct": 425, "win_rate": 0.55, "sharpe": 2.98,
    })
    tracker.start_incubation("double_decay_reversal", 50, {
        "return_pct": 34, "win_rate": 0.48, "sharpe": 1.8,
    })

    # Simulate some trades
    for pnl in [5.2, -2.1, 8.3, -1.5, 3.7, 6.1, -4.2, 2.8]:
        tracker.record_result("cascade_p99", pnl)
    for pnl in [1.5, -3.2, 2.1, -0.8, 1.9]:
        tracker.record_result("double_decay_reversal", pnl)

    tracker.print_dashboard()

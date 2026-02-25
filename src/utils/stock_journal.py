"""
Stock Trade Journal — SQLite Backend

Structured trade journal for the stock trading system. Stores entries,
exits, and performance data in SQLite (WAL mode) for analysis by the
skill updater and daily summary scripts.

Usage:
    from src.utils.stock_journal import StockJournal

    journal = StockJournal()
    journal.log_entry(bot_name='momentum_bot', strategy='MOMENTUM',
                      symbol='AMD', direction='LONG', entry_price=150.0,
                      shares=3, stop_loss=145.5, take_profit=159.0)
    journal.log_exit(symbol='AMD', direction='LONG', exit_price=158.0,
                     exit_reason='Take profit filled')
"""

import sqlite3
import time
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_PATH = PROJECT_ROOT / 'data' / 'trade_journal.db'


class StockJournal:
    """SQLite trade journal for stock bots."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS open_positions (
                    position_id TEXT PRIMARY KEY,
                    bot_name TEXT,
                    strategy TEXT,
                    symbol TEXT,
                    direction TEXT,
                    entry_time REAL,
                    entry_price REAL,
                    shares REAL,
                    notional_usd REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    signal_strength TEXT,
                    regime_at_entry TEXT,
                    reasoning TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS closed_trades (
                    trade_id TEXT PRIMARY KEY,
                    position_id TEXT,
                    bot_name TEXT,
                    strategy TEXT,
                    symbol TEXT,
                    direction TEXT,
                    entry_time REAL,
                    entry_price REAL,
                    shares REAL,
                    notional_usd REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    signal_strength TEXT,
                    regime_at_entry TEXT,
                    reasoning TEXT,
                    exit_time REAL,
                    exit_price REAL,
                    duration_seconds REAL,
                    pnl_usd REAL,
                    pnl_pct REAL,
                    exit_reason TEXT,
                    regime_at_exit TEXT,
                    r_multiple REAL,
                    alpaca_order_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_closed_strategy ON closed_trades(strategy);
                CREATE INDEX IF NOT EXISTS idx_closed_regime ON closed_trades(regime_at_entry);
                CREATE INDEX IF NOT EXISTS idx_closed_exit_time ON closed_trades(exit_time);
                CREATE INDEX IF NOT EXISTS idx_closed_bot_name ON closed_trades(bot_name);

                CREATE TABLE IF NOT EXISTS skill_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    trades_analyzed INTEGER,
                    rules_discovered INTEGER,
                    params_recommended INTEGER,
                    summary TEXT
                );

                CREATE TABLE IF NOT EXISTS parameter_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT,
                    param_name TEXT,
                    current_value REAL,
                    recommended_value REAL,
                    reason TEXT,
                    status TEXT DEFAULT 'pending',
                    applied_at TEXT,
                    reverted_at TEXT
                );

                CREATE TABLE IF NOT EXISTS parameter_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT,
                    param_name TEXT,
                    old_value REAL,
                    new_value REAL,
                    source TEXT,
                    recommendation_id INTEGER
                );

                CREATE TABLE IF NOT EXISTS strategy_lifecycle (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    bot_name TEXT,
                    strategy TEXT,
                    stage TEXT,
                    reason TEXT,
                    metrics TEXT
                );
            """)
            conn.commit()
        finally:
            conn.close()

    def log_entry(self, bot_name, strategy, symbol, direction, entry_price,
                  shares, stop_loss=None, take_profit=None,
                  signal_strength=None, regime=None, reasoning=None):
        """Log a new position entry."""
        try:
            position_id = str(uuid.uuid4())[:12]
            entry_time = time.time()
            notional = entry_price * shares if entry_price and shares else 0

            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO open_positions
                    (position_id, bot_name, strategy, symbol, direction,
                     entry_time, entry_price, shares, notional_usd,
                     stop_loss, take_profit, signal_strength,
                     regime_at_entry, reasoning)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (position_id, bot_name, strategy, symbol, direction,
                      entry_time, entry_price, shares, notional,
                      stop_loss, take_profit, signal_strength,
                      regime, reasoning))
                conn.commit()
            finally:
                conn.close()
            return position_id
        except Exception:
            pass

    def log_exit(self, symbol, direction, exit_price, exit_reason,
                 regime=None, alpaca_order_id=None):
        """Find matching open position and move to closed_trades."""
        try:
            conn = self._get_conn()
            try:
                # Find the oldest matching open position
                row = conn.execute("""
                    SELECT * FROM open_positions
                    WHERE symbol = ? AND direction = ?
                    ORDER BY entry_time ASC LIMIT 1
                """, (symbol, direction)).fetchone()

                if not row:
                    return None

                entry_time = row['entry_time']
                entry_price = row['entry_price']
                shares = row['shares']
                exit_time = time.time()
                duration = exit_time - entry_time if entry_time else 0

                # Compute PnL
                if direction == 'LONG':
                    pnl_usd = (exit_price - entry_price) * shares
                    pnl_pct = (exit_price - entry_price) / entry_price * 100 if entry_price else 0
                else:
                    pnl_usd = (entry_price - exit_price) * shares
                    pnl_pct = (entry_price - exit_price) / entry_price * 100 if entry_price else 0

                # R-multiple
                r_multiple = None
                sl = row['stop_loss']
                if sl and sl > 0 and entry_price:
                    risk = abs(entry_price - sl)
                    if risk > 0:
                        reward = abs(exit_price - entry_price)
                        r_multiple = reward / risk if pnl_usd >= 0 else -(reward / risk)

                trade_id = str(uuid.uuid4())[:12]

                conn.execute("""
                    INSERT INTO closed_trades
                    (trade_id, position_id, bot_name, strategy, symbol, direction,
                     entry_time, entry_price, shares, notional_usd,
                     stop_loss, take_profit, signal_strength,
                     regime_at_entry, reasoning,
                     exit_time, exit_price, duration_seconds,
                     pnl_usd, pnl_pct, exit_reason, regime_at_exit,
                     r_multiple, alpaca_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (trade_id, row['position_id'], row['bot_name'],
                      row['strategy'], symbol, direction,
                      entry_time, entry_price, shares, row['notional_usd'],
                      row['stop_loss'], row['take_profit'], row['signal_strength'],
                      row['regime_at_entry'], row['reasoning'],
                      exit_time, exit_price, duration,
                      round(pnl_usd, 2), round(pnl_pct, 2),
                      exit_reason, regime,
                      round(r_multiple, 2) if r_multiple is not None else None,
                      alpaca_order_id))

                conn.execute("DELETE FROM open_positions WHERE position_id = ?",
                             (row['position_id'],))
                conn.commit()
                return trade_id
            finally:
                conn.close()
        except Exception:
            return None

    def get_stats(self, strategy=None, bot_name=None, regime=None,
                  symbol=None, days=None):
        """Get trade statistics with optional filters."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM closed_trades WHERE 1=1"
            params = []

            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            if bot_name:
                query += " AND bot_name = ?"
                params.append(bot_name)
            if regime:
                query += " AND regime_at_entry = ?"
                params.append(regime)
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            if days:
                cutoff = time.time() - (days * 86400)
                query += " AND exit_time >= ?"
                params.append(cutoff)

            rows = conn.execute(query, params).fetchall()

            if not rows:
                return None

            pnls = [r['pnl_usd'] for r in rows if r['pnl_usd'] is not None]
            if not pnls:
                return None

            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p <= 0]
            gross_profit = sum(winners) if winners else 0
            gross_loss = abs(sum(losers)) if losers else 0

            # By-regime breakdown
            regime_stats = {}
            for r in rows:
                reg = r['regime_at_entry'] or 'unknown'
                if reg not in regime_stats:
                    regime_stats[reg] = {'trades': 0, 'wins': 0, 'pnl': 0}
                regime_stats[reg]['trades'] += 1
                if r['pnl_usd'] and r['pnl_usd'] > 0:
                    regime_stats[reg]['wins'] += 1
                regime_stats[reg]['pnl'] += r['pnl_usd'] or 0

            return {
                'total_trades': len(pnls),
                'win_rate': len(winners) / len(pnls) * 100 if pnls else 0,
                'avg_pnl': sum(pnls) / len(pnls),
                'total_pnl': sum(pnls),
                'profit_factor': gross_profit / gross_loss if gross_loss > 0 else float('inf'),
                'winners': len(winners),
                'losers': len(losers),
                'best_trade': max(pnls) if pnls else 0,
                'worst_trade': min(pnls) if pnls else 0,
                'by_regime': regime_stats,
            }
        finally:
            conn.close()

    def get_open_positions(self):
        """Get all open positions as list of dicts."""
        conn = self._get_conn()
        try:
            rows = conn.execute("SELECT * FROM open_positions ORDER BY entry_time").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_closed_trades(self, days=None, strategy=None):
        """Get closed trades with optional filters."""
        conn = self._get_conn()
        try:
            query = "SELECT * FROM closed_trades WHERE 1=1"
            params = []
            if days:
                cutoff = time.time() - (days * 86400)
                query += " AND exit_time >= ?"
                params.append(cutoff)
            if strategy:
                query += " AND strategy = ?"
                params.append(strategy)
            query += " ORDER BY exit_time DESC"
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def log_skill_update(self, trades_analyzed, rules_discovered,
                         params_recommended, summary):
        """Log a skill file update for audit trail."""
        try:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO skill_updates
                    (trades_analyzed, rules_discovered, params_recommended, summary)
                    VALUES (?, ?, ?, ?)
                """, (trades_analyzed, rules_discovered, params_recommended, summary))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def add_parameter_recommendation(self, strategy, param_name,
                                     current_value, recommended_value, reason):
        """Add a parameter tuning recommendation."""
        try:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO parameter_recommendations
                    (strategy, param_name, current_value, recommended_value, reason)
                    VALUES (?, ?, ?, ?, ?)
                """, (strategy, param_name, current_value, recommended_value, reason))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def get_pending_recommendations(self):
        """Get all pending parameter recommendations."""
        conn = self._get_conn()
        try:
            rows = conn.execute("""
                SELECT * FROM parameter_recommendations
                WHERE status = 'pending'
                ORDER BY timestamp
            """).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_recommendation_status(self, rec_id, status):
        """Update recommendation status (applied/rejected/reverted)."""
        try:
            conn = self._get_conn()
            try:
                ts_field = 'applied_at' if status == 'applied' else 'reverted_at'
                conn.execute(f"""
                    UPDATE parameter_recommendations
                    SET status = ?, {ts_field} = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (status, rec_id))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def log_parameter_change(self, strategy, param_name, old_value,
                             new_value, source, recommendation_id=None):
        """Log a parameter change for audit trail."""
        try:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO parameter_changes
                    (strategy, param_name, old_value, new_value, source, recommendation_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (strategy, param_name, old_value, new_value, source, recommendation_id))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

    def log_lifecycle_event(self, bot_name, strategy, stage, reason, metrics=None):
        """Log strategy lifecycle event (CANDIDATE, TESTING, PROMOTED, DEMOTED)."""
        try:
            conn = self._get_conn()
            try:
                conn.execute("""
                    INSERT INTO strategy_lifecycle
                    (bot_name, strategy, stage, reason, metrics)
                    VALUES (?, ?, ?, ?, ?)
                """, (bot_name, strategy, stage, reason, metrics))
                conn.commit()
            finally:
                conn.close()
        except Exception:
            pass

"""
Kronos Trading - Performance Monitor
Terminal-based dashboard + optional Telegram reporting.
Shows live portfolio status, risk metrics, and incubation progress.

Run standalone: python monitoring/dashboard.py
Or import and call: monitor.report()
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger("kronos.monitor")


class PerformanceMonitor:
    """
    Monitors and reports on the Kronos trading system.
    
    Features:
    - Terminal dashboard
    - Telegram alerts for key events
    - Periodic status reports
    - Trade notifications
    """

    def __init__(
        self,
        portfolio=None,
        risk_manager=None,
        incubation_tracker=None,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        config_path: str = "config/kronos.json",
    ):
        self.portfolio = portfolio
        self.risk = risk_manager
        self.tracker = incubation_tracker

        # Load Telegram config
        self.tg_token = telegram_token
        self.tg_chat_id = telegram_chat_id

        if not self.tg_token:
            self._load_telegram_config(config_path)

    def _load_telegram_config(self, config_path: str):
        """Load Telegram bot credentials from config."""
        try:
            with open(config_path) as f:
                config = json.load(f)
            tg = config.get("telegram", {})
            self.tg_token = tg.get("bot_token")
            self.tg_chat_id = tg.get("chat_id")
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    # ─── Terminal Dashboard ──────────────────────────────────────

    def print_dashboard(self):
        """Print full terminal dashboard."""
        os.system("clear" if os.name != "nt" else "cls")

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n{'=' * 70}")
        print(f"  🦞 KRONOS TRADING SYSTEM - LIVE DASHBOARD")
        print(f"  {now}")
        print(f"{'=' * 70}")

        # Account Status
        if self.portfolio:
            status = self.portfolio.get_status()
            acct = status.get("account") or {}
            env = "🧪 TESTNET" if status.get("testnet") else "💰 MAINNET"

            print(f"\n  {env}")
            print(f"  {'─' * 40}")
            print(f"  Equity:    ${acct.get('equity', 0):>12,.2f}")
            print(f"  Available: ${acct.get('available', 0):>12,.2f}")
            print(f"  Positions: {acct.get('positions', 0)}")

            # Active trades
            trades = status.get("trades", [])
            if trades:
                print(f"\n  📊 Active Trades ({len(trades)})")
                print(f"  {'─' * 55}")
                for t in trades:
                    print(f"  #{t['id']:>3} | {t['strategy']:20s} | "
                          f"{t['side']:5s} {t['size']:.4f} {t['coin']:4s} | "
                          f"Entry: {t['entry']:.2f}")

        # Risk Status
        if self.risk:
            try:
                equity = self.portfolio.get_status()["account"]["equity"] if self.portfolio else 0
                risk = self.risk.get_risk_summary(equity)
            except Exception:
                risk = self.risk.get_risk_summary(0)

            kill = "🔴 ACTIVE" if risk["kill_switch"] else "🟢 OFF"
            print(f"\n  🛡️ Risk Controls")
            print(f"  {'─' * 40}")
            print(f"  Kill Switch:    {kill}")
            print(f"  Daily P&L:      ${risk['daily_pnl']:>+10,.2f}")
            print(f"  Peak Equity:    ${risk['peak_equity']:>10,.2f}")
            print(f"  Drawdown:       {risk['current_drawdown_pct']:>9.1f}%")
            print(f"  Strategies:     {risk['strategies']['active']} active, "
                  f"{risk['strategies']['killed']} killed")

        # Incubation Status
        if self.tracker:
            self.tracker.print_dashboard()

        print()

    # ─── Telegram Alerts ─────────────────────────────────────────

    def send_telegram(self, message: str) -> bool:
        """Send a message via Telegram bot."""
        if not self.tg_token or not self.tg_chat_id:
            logger.debug("Telegram not configured, skipping alert")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.tg_token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id": self.tg_chat_id,
                "text": message,
                "parse_mode": "HTML",
            }, timeout=10)
            return resp.ok
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def alert_trade_opened(self, strategy: str, coin: str, side: str, size: float, entry: float):
        """Alert on new trade."""
        msg = (
            f"📊 <b>Trade Opened</b>\n"
            f"Strategy: {strategy}\n"
            f"{side.upper()} {size} {coin} @ ${entry:,.2f}\n"
            f"Time: {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        )
        self.send_telegram(msg)

    def alert_trade_closed(self, strategy: str, coin: str, pnl: float, pnl_pct: float = 0):
        """Alert on trade close."""
        emoji = "💰" if pnl > 0 else "💸"
        msg = (
            f"{emoji} <b>Trade Closed</b>\n"
            f"Strategy: {strategy}\n"
            f"{coin}: ${pnl:+,.2f} ({pnl_pct:+.1f}%)\n"
        )
        self.send_telegram(msg)

    def alert_risk_event(self, event: str, details: str = ""):
        """Alert on risk events (kill switch, daily limit, etc)."""
        msg = f"🚨 <b>Risk Alert</b>\n{event}\n{details}"
        self.send_telegram(msg)

    def send_daily_report(self):
        """Send daily portfolio summary via Telegram."""
        if not self.portfolio:
            return

        status = self.portfolio.get_status()
        acct = status.get("account") or {}
        risk = status.get("risk", {})

        # Get trade history for today
        history = self.portfolio.get_trade_history(limit=20)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_trades = [t for t in history if t.get("closed", "").startswith(today)]

        today_pnl = sum(t.get("pnl", 0) for t in today_trades)
        wins = sum(1 for t in today_trades if t.get("pnl", 0) > 0)
        losses = len(today_trades) - wins

        msg = (
            f"📋 <b>Kronos Daily Report</b>\n"
            f"{'─' * 30}\n"
            f"Equity: ${acct.get('equity', 0):,.2f}\n"
            f"Daily P&L: ${today_pnl:+,.2f}\n"
            f"Trades: {len(today_trades)} ({wins}W / {losses}L)\n"
            f"Drawdown: {risk.get('current_drawdown_pct', 0):.1f}%\n"
            f"Active: {status.get('active_trades', 0)} positions\n"
        )

        # Add incubation summary
        if self.tracker:
            dash = self.tracker.get_dashboard()
            msg += (
                f"\n🧪 <b>Incubation</b>\n"
                f"Strategies: {dash['total_strategies']}\n"
                f"Budget: ${dash['total_budget_allocated']}\n"
                f"P&L: ${dash['total_pnl']:+,.2f}\n"
            )

        self.send_telegram(msg)

    # ─── Monitoring Loop ─────────────────────────────────────────

    def run_loop(self, interval_seconds: int = 60, dashboard: bool = True):
        """
        Main monitoring loop. Updates dashboard and checks for alerts.
        
        Args:
            interval_seconds: How often to refresh (default 60s)
            dashboard: Whether to print terminal dashboard
        """
        logger.info(f"Starting monitor loop (interval={interval_seconds}s)")
        last_daily_report = None

        try:
            while True:
                # Update dashboard
                if dashboard:
                    self.print_dashboard()

                # Manage positions (check for SL/TP fills)
                if self.portfolio:
                    self.portfolio.manage_positions()

                # Daily report at 00:00 UTC
                now = datetime.now(timezone.utc)
                if now.hour == 0 and last_daily_report != now.date():
                    self.send_daily_report()
                    last_daily_report = now.date()

                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print("\n\n  Monitoring stopped. Kronos still running on VPS.")


# ─── Standalone Dashboard ────────────────────────────────────────

def run_standalone():
    """Run dashboard without live trading (status only)."""
    import sys
    sys.path.insert(0, ".")

    monitor = PerformanceMonitor()

    # Try to load components
    try:
        from monitoring.incubation_tracker import IncubationTracker
        monitor.tracker = IncubationTracker()
    except ImportError:
        pass

    if monitor.tracker:
        monitor.tracker.print_dashboard()
    else:
        print("No components loaded. Run with portfolio manager for full dashboard.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_standalone()

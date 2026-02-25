#!/usr/bin/env python3
"""
Kronos Telegram Notifier
=========================
Sends trade alerts and status updates to Telegram.

Uses urllib only (no external deps). Reads bot token from env or config.

Usage:
    notifier = TelegramNotifier()
    notifier.send_trade_alert(trade_info)
    notifier.send_status(portfolio_status)
"""

import json
import logging
import os
import urllib.request
from pathlib import Path

log = logging.getLogger("telegram")

CONFIG_PATH = Path(__file__).parent.parent / "config" / "kronos.json"


class TelegramNotifier:
    """Sends messages to Telegram via Bot API."""

    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.environ.get("KRONOS_TG_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("KRONOS_TG_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            log.info("Telegram notifications disabled (no token/chat_id)")

    def send(self, text: str, parse_mode: str = "HTML"):
        """Send a message to Telegram."""
        if not self.enabled:
            return

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode())
                if not result.get("ok"):
                    log.warning(f"Telegram send failed: {result}")
        except Exception as e:
            log.warning(f"Telegram error: {e}")

    def send_trade_open(self, trade: dict):
        """Alert for new position opened."""
        side_emoji = "🟢" if trade.get("side") == "long" else "🔴"
        mode = trade.get("mode", "paper").upper()

        msg = (
            f"{side_emoji} <b>[{mode}] {trade['side'].upper()} {trade['symbol']}</b>\n"
            f"📊 Strategy: {trade.get('strategy', '?')}\n"
            f"💰 Entry: ${trade['entry_price']:,.2f}\n"
            f"📏 Size: ${trade.get('notional_usd', 0):,.2f}\n"
        )

        if trade.get("stop_loss"):
            msg += f"🛑 Stop: ${trade['stop_loss']:,.2f}\n"
        if trade.get("take_profit"):
            msg += f"🎯 Target: ${trade['take_profit']:,.2f}\n"
        if trade.get("tag"):
            msg += f"🏷 Signal: {trade['tag']}\n"

        self.send(msg)

    def send_trade_close(self, trade: dict):
        """Alert for position closed."""
        pnl = trade.get("pnl_usd", 0)
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        mode = trade.get("mode", "paper").upper()

        msg = (
            f"{pnl_emoji} <b>[{mode}] CLOSED {trade['side'].upper()} {trade['symbol']}</b>\n"
            f"📊 Strategy: {trade.get('strategy', '?')}\n"
            f"💰 Entry: ${trade['entry_price']:,.2f} → Exit: ${trade['exit_price']:,.2f}\n"
            f"{'📈' if pnl >= 0 else '📉'} PnL: ${pnl:+,.2f} ({trade.get('pnl_pct', 0):+.2f}%)\n"
            f"📝 Reason: {trade.get('reason', '?')}\n"
        )

        self.send(msg)

    def send_status(self, status: dict):
        """Send portfolio status update."""
        mode = status.get("mode", "paper").upper()
        eq = status.get("equity", 0)
        ret = status.get("total_return_pct", 0)
        dd = status.get("drawdown_pct", 0)

        msg = (
            f"📊 <b>KRONOS [{mode}] Status</b>\n"
            f"💰 Equity: ${eq:,.2f}\n"
            f"{'📈' if ret >= 0 else '📉'} Return: {ret:+.2f}%\n"
            f"📉 Drawdown: {dd:.2f}%\n"
            f"🔢 Positions: {status.get('positions_count', 0)}\n"
            f"💵 Exposure: ${status.get('total_exposure_usd', 0):,.2f} "
            f"({status.get('exposure_pct', 0):.1f}%)\n"
            f"📅 Daily PnL: ${status.get('daily_pnl', 0):+,.2f}\n"
        )

        if status.get("circuit_breaker"):
            msg += "🚨 <b>CIRCUIT BREAKER ACTIVE</b>\n"

        self.send(msg)

    def send_alert(self, title: str, message: str):
        """Send a generic alert."""
        self.send(f"🔔 <b>{title}</b>\n{message}")

    def send_error(self, error: str):
        """Send an error alert."""
        self.send(f"🚨 <b>KRONOS ERROR</b>\n<code>{error}</code>")

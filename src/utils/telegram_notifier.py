"""
Telegram Notifier for Stock Trading

Sends trade alerts, daily summaries, and status updates to Telegram.
Uses urllib only (no external deps). Reads bot token from env vars.

Uses same bot as Kronos: @jmurkedbot
Env vars: KRONOS_TG_BOT_TOKEN, KRONOS_TG_CHAT_ID

Usage:
    from src.utils.telegram_notifier import TelegramNotifier
    notifier = TelegramNotifier()
    notifier.send("Hello from stocks!")
"""

import json
import logging
import os
import urllib.request

log = logging.getLogger("telegram")


class TelegramNotifier:
    """Sends messages to Telegram via Bot API."""

    def __init__(self, bot_token=None, chat_id=None):
        self.bot_token = bot_token or os.environ.get("KRONOS_TG_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("KRONOS_TG_CHAT_ID", "")
        self.enabled = bool(self.bot_token and self.chat_id)

        if not self.enabled:
            log.info("Telegram notifications disabled (no token/chat_id)")

    def send(self, text, parse_mode="HTML"):
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

    def send_alert(self, title, message):
        """Send a generic alert."""
        self.send(f"\U0001f514 <b>{title}</b>\n{message}")

    def send_error(self, error):
        """Send an error alert."""
        self.send(f"\U0001f6a8 <b>STOCKS ERROR</b>\n<code>{error}</code>")

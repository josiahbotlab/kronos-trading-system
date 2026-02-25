"""
HyperLiquid Liquidation Trading Bot

Monitors liquidation data and enters short positions when thresholds are hit.
Uses config.py settings and nice_funcs_hl.py for order execution.

Usage:
    python src/agents/hyperliquid_bot.py              # Live trading
    python src/agents/hyperliquid_bot.py --dry-run   # Test mode (no real trades)
"""

import argparse
import random
import sys
import time
from datetime import datetime, timedelta
from collections import defaultdict

import requests
import schedule
from termcolor import cprint

sys.path.append('/Users/josiahgarcia/trading-bot')

from config import (
    SECRET_KEY,
    ORDER_USD_SIZE,
    LEVERAGE,
    SYMBOLS,
    SYMBOLS_DATA,
    MOONDEV_API_KEY
)
from src.nice_funcs_hl import (
    get_account,
    get_position,
    adjust_leverage_usd_size,
    market_sell,
    pnl_close,
    ask_bid
)


class LiquidationBot:
    """Bot that monitors liquidations and enters shorts on threshold breaches."""

    def __init__(self, dry_run=False, mock_mode=False):
        self.dry_run = dry_run
        self.mock_mode = mock_mode
        self.account = get_account(SECRET_KEY) if not mock_mode else None
        self.liquidation_history = defaultdict(list)
        self.cooldowns = {}
        self.cooldown_minutes = 30
        self.mock_cycle = 0

        if self.dry_run:
            cprint("*** DRY RUN MODE - No real trades will be executed ***", "yellow")
        if self.mock_mode:
            cprint("*** MOCK MODE - Using simulated liquidation data ***", "yellow")
            cprint("Bot initialized with mock account", "green")
        else:
            cprint(f"Bot initialized for address: {self.account.address}", "green")
        cprint(f"Trading symbols: {SYMBOLS}", "cyan")
        cprint(f"Order size: ${ORDER_USD_SIZE} | Leverage: {LEVERAGE}x", "cyan")

    def generate_mock_liquidation(self, symbol):
        """Generate mock liquidation data for testing."""
        if symbol not in SYMBOLS_DATA:
            return None

        threshold = SYMBOLS_DATA[symbol]['liquidations']

        # Generate varying amounts - occasionally spike above threshold
        # Every 3rd cycle, generate a spike for one random symbol
        if self.mock_cycle % 3 == 2 and random.random() < 0.4:
            # Spike: 80-120% of threshold
            amount = threshold * random.uniform(0.8, 1.2)
            cprint(f"[MOCK] Spike event for {symbol}: ${amount:,.0f}", "magenta")
        else:
            # Normal: 5-30% of threshold per cycle
            amount = threshold * random.uniform(0.05, 0.30)

        return {'amount': amount, 'symbol': symbol, 'mock': True}

    def fetch_liquidations(self, symbol):
        """Fetch recent liquidation data from MoonDev API or mock."""
        if self.mock_mode:
            return self.generate_mock_liquidation(symbol)

        try:
            url = f"https://api.moondev.com/liquidations/{symbol}"
            headers = {"Authorization": f"Bearer {MOONDEV_API_KEY}"}

            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                return response.json()
            else:
                cprint(f"API error for {symbol}: {response.status_code}", "yellow")
                return None

        except requests.exceptions.RequestException as e:
            cprint(f"Request failed for {symbol}: {e}", "red")
            return None

    def get_liquidations_in_window(self, symbol):
        """Calculate total liquidations within the configured time window."""
        if symbol not in SYMBOLS_DATA:
            return 0

        window_mins = SYMBOLS_DATA[symbol]['time_window_mins']
        cutoff = datetime.now() - timedelta(minutes=window_mins)

        # Filter liquidations within window
        recent = [
            liq for liq in self.liquidation_history[symbol]
            if liq['timestamp'] > cutoff
        ]

        # Update history to only keep recent data
        self.liquidation_history[symbol] = recent

        return sum(liq['amount'] for liq in recent)

    def record_liquidation(self, symbol, amount):
        """Record a liquidation event."""
        self.liquidation_history[symbol].append({
            'timestamp': datetime.now(),
            'amount': amount
        })

    def is_on_cooldown(self, symbol):
        """Check if symbol is on entry cooldown."""
        if symbol not in self.cooldowns:
            return False

        cooldown_end = self.cooldowns[symbol]
        if datetime.now() > cooldown_end:
            del self.cooldowns[symbol]
            return False

        return True

    def set_cooldown(self, symbol):
        """Set cooldown after entering a position."""
        self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.cooldown_minutes)

    def check_entry_signal(self, symbol):
        """Check if liquidation threshold is breached for entry."""
        if symbol not in SYMBOLS_DATA:
            return False

        threshold = SYMBOLS_DATA[symbol]['liquidations']
        total_liqs = self.get_liquidations_in_window(symbol)

        if total_liqs >= threshold:
            cprint(
                f"SIGNAL: {symbol} liquidations ${total_liqs:,.0f} >= ${threshold:,.0f}",
                "yellow"
            )
            return True

        return False

    def enter_short(self, symbol):
        """Enter a short position."""
        try:
            # Check cooldown first (works in all modes)
            if self.is_on_cooldown(symbol):
                cprint(f"{symbol} on cooldown, skipping entry", "yellow")
                return False

            # In mock mode, skip real position/size checks
            if self.mock_mode:
                price = self.get_mock_price(symbol)
                size = round((ORDER_USD_SIZE * LEVERAGE) / price, 4)
                cprint(f"[MOCK] Entering SHORT: {symbol} | Size: {size} @ ${price:,.2f}", "magenta")
                self.set_cooldown(symbol)
                return True

            # Check if already in position
            _, im_in_pos, _, _, _, _, _ = get_position(symbol, self.account)

            if im_in_pos:
                cprint(f"Already in position for {symbol}, skipping entry", "yellow")
                return False

            # Set leverage and calculate size
            _, size = adjust_leverage_usd_size(symbol, ORDER_USD_SIZE, LEVERAGE, self.account)

            if size <= 0:
                cprint(f"Invalid size calculated for {symbol}", "red")
                return False

            # Execute short (or simulate in dry-run mode)
            cprint(f"Entering SHORT: {symbol} | Size: {size}", "magenta")

            if self.dry_run:
                cprint(f"[DRY RUN] Would short {symbol} with size {size}", "cyan")
                self.set_cooldown(symbol)
                return True

            order = market_sell(symbol, size, self.account)

            if order:
                cprint(f"Short entered for {symbol}: {order}", "green")
                self.set_cooldown(symbol)
                return True
            else:
                cprint(f"Order failed for {symbol}", "red")
                return False

        except Exception as e:
            cprint(f"Error entering short for {symbol}: {e}", "red")
            return False

    def manage_positions(self):
        """Check and manage existing positions for TP/SL."""
        # Skip position management in mock mode (no real positions)
        if self.mock_mode:
            return

        for symbol in SYMBOLS:
            if symbol not in SYMBOLS_DATA:
                continue

            tp = SYMBOLS_DATA[symbol]['tp']
            sl = SYMBOLS_DATA[symbol]['sl']

            try:
                # Check position status
                _, im_in_pos, pos_size, _, entry_px, pnl_perc, long = get_position(symbol, self.account)

                if im_in_pos:
                    direction = "LONG" if long else "SHORT"
                    cprint(f"Position: {symbol} {direction} | PnL: {pnl_perc:.2f}%", "white")

                    if self.dry_run:
                        if pnl_perc >= tp:
                            cprint(f"[DRY RUN] Would close {symbol} at TP ({pnl_perc:.2f}% >= {tp}%)", "cyan")
                        elif pnl_perc <= sl:
                            cprint(f"[DRY RUN] Would close {symbol} at SL ({pnl_perc:.2f}% <= {sl}%)", "cyan")
                        continue

                closed = pnl_close(symbol, tp, sl, self.account)
                if closed:
                    cprint(f"Position closed for {symbol}", "green")
            except Exception as e:
                cprint(f"Error managing position for {symbol}: {e}", "red")

    def get_mock_price(self, symbol):
        """Get mock price for testing."""
        mock_prices = {
            'BTC': 95000 + random.uniform(-500, 500),
            'ETH': 3400 + random.uniform(-50, 50),
            'SOL': 180 + random.uniform(-5, 5),
            'WIF': 1.80 + random.uniform(-0.1, 0.1),
        }
        return mock_prices.get(symbol, 100.0)

    def check_symbols(self):
        """Main loop iteration - check all symbols."""
        cprint(f"\n--- Check cycle {self.mock_cycle} at {datetime.now().strftime('%H:%M:%S')} ---", "cyan")

        for symbol in SYMBOLS:
            # Fetch and record new liquidation data
            liq_data = self.fetch_liquidations(symbol)

            if liq_data and 'amount' in liq_data:
                self.record_liquidation(symbol, liq_data['amount'])

            # Check for entry signal
            if self.check_entry_signal(symbol):
                self.enter_short(symbol)

            # Log current state
            total_liqs = self.get_liquidations_in_window(symbol)
            threshold = SYMBOLS_DATA.get(symbol, {}).get('liquidations', 0)

            if self.mock_mode:
                price = self.get_mock_price(symbol)
            else:
                price, _, _ = ask_bid(symbol)
                price = price or 0

            cprint(
                f"{symbol}: ${price:,.2f} | Liqs: ${total_liqs:,.0f} / ${threshold:,.0f}",
                "white"
            )

        # Manage existing positions
        self.manage_positions()

        # Increment mock cycle counter
        self.mock_cycle += 1

    def run(self, interval_seconds=60):
        """Run the bot with scheduled checks."""
        cprint("\nStarting Liquidation Bot...", "green")
        cprint(f"Check interval: {interval_seconds}s", "cyan")

        # Initial check
        self.check_symbols()

        # Schedule recurring checks
        schedule.every(interval_seconds).seconds.do(self.check_symbols)

        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            cprint("\nBot stopped by user", "yellow")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="HyperLiquid Liquidation Trading Bot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in test mode without executing real trades"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock liquidation data (no API calls, no wallet needed)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    bot = LiquidationBot(dry_run=args.dry_run, mock_mode=args.mock)
    bot.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()

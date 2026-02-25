"""
HyperLiquid Breakout Trading Bot

Momentum strategy that enters positions on 24-hour high/low breakouts.
- LONG: Price breaks above 24-hour high
- SHORT: Price breaks below 24-hour low
- Stop Loss: 1.5x ATR
- Take Profit: 2:1 risk/reward ratio

Usage:
    python src/agents/breakout_bot.py              # Live trading
    python src/agents/breakout_bot.py --dry-run   # Test mode (no real trades)
    python src/agents/breakout_bot.py --mock      # Mock data (no API needed)
"""

import argparse
import random
import sys
import time
from datetime import datetime, timedelta
from collections import defaultdict

import schedule
from termcolor import cprint
from hyperliquid.info import Info
from hyperliquid.utils import constants

sys.path.append('/Users/josiahgarcia/trading-bot')

from config import (
    SECRET_KEY,
    ORDER_USD_SIZE,
    LEVERAGE,
    SYMBOLS,
)
from src.nice_funcs_hl import (
    get_account,
    get_position,
    adjust_leverage_usd_size,
    market_buy,
    market_sell,
    ask_bid
)


# ATR and risk parameters
ATR_PERIOD = 14
ATR_MULTIPLIER = 1.5
RISK_REWARD_RATIO = 2.0


class BreakoutBot:
    """Bot that trades 24-hour high/low breakouts with ATR-based stops."""

    def __init__(self, dry_run=False, mock_mode=False):
        self.dry_run = dry_run
        self.mock_mode = mock_mode
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True) if not mock_mode else None

        # Only initialize account if we have a key and not in mock mode
        self.account = None
        if not mock_mode and SECRET_KEY:
            try:
                self.account = get_account(SECRET_KEY)
            except Exception as e:
                cprint(f"Warning: Could not initialize wallet: {e}", "yellow")

        # Track positions and levels
        self.active_trades = {}  # symbol -> {entry, stop_loss, take_profit, direction}
        self.cooldowns = {}
        self.cooldown_minutes = 60
        self.mock_cycle = 0

        # Mock data state
        self.mock_prices = {}
        self.mock_candles = {}

        if self.dry_run:
            cprint("*** DRY RUN MODE - No real trades will be executed ***", "yellow")
        if self.mock_mode:
            cprint("*** MOCK MODE - Using simulated price data ***", "yellow")
            cprint("Bot initialized with mock account", "green")
            self._init_mock_data()
        elif self.account:
            cprint(f"Bot initialized for address: {self.account.address}", "green")
        else:
            cprint("Bot initialized without wallet (price monitoring only)", "yellow")

        cprint(f"Trading symbols: {SYMBOLS}", "cyan")
        cprint(f"Order size: ${ORDER_USD_SIZE} | Leverage: {LEVERAGE}x", "cyan")
        cprint(f"ATR period: {ATR_PERIOD} | SL multiplier: {ATR_MULTIPLIER}x | RR ratio: {RISK_REWARD_RATIO}", "cyan")

    def _init_mock_data(self):
        """Initialize mock price data."""
        base_prices = {
            'BTC': 95000,
            'ETH': 3400,
            'SOL': 180,
            'WIF': 1.80,
        }

        for symbol in SYMBOLS:
            base = base_prices.get(symbol, 100)
            self.mock_prices[symbol] = base
            # Generate 24 hours of hourly candles
            self.mock_candles[symbol] = self._generate_mock_candles(base, 24)

    def _generate_mock_candles(self, base_price, num_candles):
        """Generate mock OHLCV candles."""
        candles = []
        price = base_price * random.uniform(0.97, 1.0)

        for i in range(num_candles):
            volatility = base_price * 0.01  # 1% volatility per candle
            open_price = price
            high = open_price + random.uniform(0, volatility * 2)
            low = open_price - random.uniform(0, volatility * 2)
            close = open_price + random.uniform(-volatility, volatility)

            candles.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': random.uniform(1000, 10000)
            })
            price = close

        return candles

    def fetch_candles(self, symbol, hours=24):
        """Fetch hourly candles from HyperLiquid."""
        if self.mock_mode:
            return self.mock_candles.get(symbol, [])

        try:
            end_time = int(datetime.now().timestamp() * 1000)
            start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)

            raw_candles = self.info.candles_snapshot(symbol, "1h", start_time, end_time)

            candles = []
            for c in raw_candles:
                candles.append({
                    'open': float(c['o']),
                    'high': float(c['h']),
                    'low': float(c['l']),
                    'close': float(c['c']),
                    'volume': float(c['v'])
                })

            return candles

        except Exception as e:
            cprint(f"Error fetching candles for {symbol}: {e}", "red")
            return []

    def calculate_24h_range(self, candles):
        """Calculate 24-hour high and low from candles."""
        if not candles:
            return None, None

        high_24h = max(c['high'] for c in candles)
        low_24h = min(c['low'] for c in candles)

        return high_24h, low_24h

    def calculate_atr(self, candles, period=ATR_PERIOD):
        """Calculate Average True Range."""
        if len(candles) < period + 1:
            # Not enough data, estimate from recent volatility
            if candles:
                ranges = [c['high'] - c['low'] for c in candles]
                return sum(ranges) / len(ranges) if ranges else 0
            return 0

        true_ranges = []
        for i in range(1, len(candles)):
            high = candles[i]['high']
            low = candles[i]['low']
            prev_close = candles[i-1]['close']

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        # Use simple moving average of true ranges
        recent_tr = true_ranges[-period:]
        atr = sum(recent_tr) / len(recent_tr)

        return atr

    def get_current_price(self, symbol):
        """Get current price."""
        if self.mock_mode:
            # Simulate price movement
            base = self.mock_prices.get(symbol, 100)
            movement = base * random.uniform(-0.005, 0.005)
            self.mock_prices[symbol] = base + movement
            return self.mock_prices[symbol]

        price, _, _ = ask_bid(symbol)
        return price

    def is_on_cooldown(self, symbol):
        """Check if symbol is on entry cooldown."""
        if symbol not in self.cooldowns:
            return False

        if datetime.now() > self.cooldowns[symbol]:
            del self.cooldowns[symbol]
            return False

        return True

    def set_cooldown(self, symbol):
        """Set cooldown after trade."""
        self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.cooldown_minutes)

    def check_breakout(self, symbol, current_price, high_24h, low_24h):
        """Check for breakout signals."""
        if current_price is None or high_24h is None or low_24h is None:
            return None

        # Small buffer to avoid false breakouts (0.1%)
        buffer = high_24h * 0.001

        if current_price > high_24h + buffer:
            return 'LONG'
        elif current_price < low_24h - buffer:
            return 'SHORT'

        return None

    def calculate_levels(self, entry_price, atr, direction):
        """Calculate stop loss and take profit levels."""
        stop_distance = atr * ATR_MULTIPLIER
        profit_distance = stop_distance * RISK_REWARD_RATIO

        if direction == 'LONG':
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + profit_distance
        else:  # SHORT
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - profit_distance

        return stop_loss, take_profit

    def enter_position(self, symbol, direction, entry_price, stop_loss, take_profit):
        """Enter a position."""
        try:
            if self.is_on_cooldown(symbol):
                cprint(f"{symbol} on cooldown, skipping entry", "yellow")
                return False

            # Mock mode - simulate entry
            if self.mock_mode:
                size = round((ORDER_USD_SIZE * LEVERAGE) / entry_price, 4)
                cprint(f"[MOCK] Entering {direction}: {symbol} | Size: {size} @ ${entry_price:,.2f}", "magenta")
                cprint(f"[MOCK] SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}", "cyan")

                self.active_trades[symbol] = {
                    'entry': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'direction': direction,
                    'size': size
                }
                self.set_cooldown(symbol)
                return True

            # Check if already in position (only if we have an account)
            if self.account:
                _, im_in_pos, _, _, _, _, _ = get_position(symbol, self.account)

                if im_in_pos:
                    cprint(f"Already in position for {symbol}, skipping entry", "yellow")
                    return False

            # Calculate size
            if self.account:
                _, size = adjust_leverage_usd_size(symbol, ORDER_USD_SIZE, LEVERAGE, self.account)
            else:
                # Estimate size without account
                size = round((ORDER_USD_SIZE * LEVERAGE) / entry_price, 4)

            if size <= 0:
                cprint(f"Invalid size calculated for {symbol}", "red")
                return False

            cprint(f"Entering {direction}: {symbol} | Size: {size} @ ${entry_price:,.2f}", "magenta")
            cprint(f"SL: ${stop_loss:,.2f} | TP: ${take_profit:,.2f}", "cyan")

            if self.dry_run:
                cprint(f"[DRY RUN] Would enter {direction} {symbol} with size {size}", "cyan")
                self.active_trades[symbol] = {
                    'entry': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'direction': direction,
                    'size': size
                }
                self.set_cooldown(symbol)
                return True

            # Execute trade
            if direction == 'LONG':
                order = market_buy(symbol, size, self.account)
            else:
                order = market_sell(symbol, size, self.account)

            if order:
                cprint(f"{direction} entered for {symbol}: {order}", "green")
                self.active_trades[symbol] = {
                    'entry': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'direction': direction,
                    'size': size
                }
                self.set_cooldown(symbol)
                return True
            else:
                cprint(f"Order failed for {symbol}", "red")
                return False

        except Exception as e:
            cprint(f"Error entering position for {symbol}: {e}", "red")
            return False

    def manage_position(self, symbol, current_price):
        """Check and manage active position for SL/TP."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        direction = trade['direction']
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']
        entry = trade['entry']

        should_close = False
        close_reason = None

        if direction == 'LONG':
            if current_price <= stop_loss:
                should_close = True
                close_reason = f"SL hit (${current_price:,.2f} <= ${stop_loss:,.2f})"
            elif current_price >= take_profit:
                should_close = True
                close_reason = f"TP hit (${current_price:,.2f} >= ${take_profit:,.2f})"
        else:  # SHORT
            if current_price >= stop_loss:
                should_close = True
                close_reason = f"SL hit (${current_price:,.2f} >= ${stop_loss:,.2f})"
            elif current_price <= take_profit:
                should_close = True
                close_reason = f"TP hit (${current_price:,.2f} <= ${take_profit:,.2f})"

        if should_close:
            pnl_pct = ((current_price - entry) / entry * 100) if direction == 'LONG' else ((entry - current_price) / entry * 100)

            if self.mock_mode or self.dry_run:
                cprint(f"[{'MOCK' if self.mock_mode else 'DRY RUN'}] Closing {symbol} {direction}: {close_reason} | PnL: {pnl_pct:+.2f}%", "green" if pnl_pct > 0 else "red")
                del self.active_trades[symbol]
                return

            # Close real position
            try:
                if self.account:
                    _, im_in_pos, pos_size, _, _, _, is_long = get_position(symbol, self.account)

                    if im_in_pos:
                        if is_long:
                            market_sell(symbol, abs(pos_size), self.account)
                        else:
                            market_buy(symbol, abs(pos_size), self.account)

                cprint(f"Closed {symbol} {direction}: {close_reason} | PnL: {pnl_pct:+.2f}%", "green" if pnl_pct > 0 else "red")
                del self.active_trades[symbol]

            except Exception as e:
                cprint(f"Error closing position for {symbol}: {e}", "red")

    def update_mock_candles(self):
        """Update mock candles with new data to simulate price movement."""
        for symbol in SYMBOLS:
            if symbol in self.mock_candles and self.mock_candles[symbol]:
                last_close = self.mock_candles[symbol][-1]['close']
                volatility = last_close * 0.01

                # Occasionally create breakout conditions
                if random.random() < 0.15:  # 15% chance of larger move
                    volatility *= 3

                new_candle = {
                    'open': last_close,
                    'high': last_close + random.uniform(0, volatility * 2),
                    'low': last_close - random.uniform(0, volatility * 2),
                    'close': last_close + random.uniform(-volatility, volatility),
                    'volume': random.uniform(1000, 10000)
                }

                # Keep last 24 candles
                self.mock_candles[symbol] = self.mock_candles[symbol][-23:] + [new_candle]
                self.mock_prices[symbol] = new_candle['close']

    def check_symbols(self):
        """Main loop - check all symbols for breakouts."""
        cprint(f"\n--- Check cycle {self.mock_cycle} at {datetime.now().strftime('%H:%M:%S')} ---", "cyan")

        if self.mock_mode:
            self.update_mock_candles()

        for symbol in SYMBOLS:
            # Fetch candles and calculate levels
            candles = self.fetch_candles(symbol, hours=24)

            if not candles:
                cprint(f"{symbol}: No candle data available", "yellow")
                continue

            high_24h, low_24h = self.calculate_24h_range(candles)
            atr = self.calculate_atr(candles)
            current_price = self.get_current_price(symbol)

            if current_price is None:
                continue

            # Log current state
            range_pct = ((high_24h - low_24h) / low_24h * 100) if low_24h else 0
            cprint(
                f"{symbol}: ${current_price:,.2f} | 24h: ${low_24h:,.2f} - ${high_24h:,.2f} ({range_pct:.1f}%) | ATR: ${atr:,.2f}",
                "white"
            )

            # Check for active position management
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                pnl_pct = ((current_price - trade['entry']) / trade['entry'] * 100) if trade['direction'] == 'LONG' else ((trade['entry'] - current_price) / trade['entry'] * 100)
                cprint(f"  Active {trade['direction']}: Entry ${trade['entry']:,.2f} | PnL: {pnl_pct:+.2f}%", "magenta")
                self.manage_position(symbol, current_price)
                continue

            # Check for breakout
            signal = self.check_breakout(symbol, current_price, high_24h, low_24h)

            if signal:
                stop_loss, take_profit = self.calculate_levels(current_price, atr, signal)
                cprint(f"  BREAKOUT {signal}: {symbol} @ ${current_price:,.2f}", "yellow")
                self.enter_position(symbol, signal, current_price, stop_loss, take_profit)

        self.mock_cycle += 1

    def run(self, interval_seconds=60):
        """Run the bot with scheduled checks."""
        cprint("\nStarting Breakout Bot...", "green")
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
        description="HyperLiquid Breakout Trading Bot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in test mode without executing real trades"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock price data (no API calls, no wallet needed)"
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
    bot = BreakoutBot(dry_run=args.dry_run, mock_mode=args.mock)
    bot.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()

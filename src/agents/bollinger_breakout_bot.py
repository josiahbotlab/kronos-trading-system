"""
Bollinger Band Breakout Bot

Moon Dev's volatility breakout strategy:
- Bollinger Bands: 21 period, 2.7 standard deviations
- Entry: Price CLOSES ABOVE upper band (buying strength, not weakness)
- Confirmation: Heikin Ashi candle must be GREEN (bullish trend)
- Exit: Price returns to middle band (21 SMA) OR stop loss hit
- Stop Loss: 2% below entry

This is a BREAKOUT strategy - we buy strength when volatility expands.
NOT mean reversion - we're not buying at lower band.

Usage:
    python src/agents/bollinger_breakout_bot.py              # Live trading
    python src/agents/bollinger_breakout_bot.py --dry-run    # Test mode
    python src/agents/bollinger_breakout_bot.py --mock       # Mock data
    python src/agents/bollinger_breakout_bot.py --use-scanner --duration 60
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import schedule
from termcolor import cprint
from dotenv import load_dotenv

sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Import agents
from src.agents.risk_agent import check_risk_verbose, update_position, close_position, get_risk_status
from src.agents.exit_agent import manage_exit_verbose, reset_position, get_position_status
from src.agents.scanner import get_breakout_candidates

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    cprint("Warning: alpaca-trade-api not installed", "yellow")


# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Trading parameters
SYMBOLS = ['TSLA', 'AMD', 'NVDA', 'AAPL', 'SPY']
ORDER_USD_SIZE = 500

# Bollinger Band parameters (Moon Dev's specs)
BB_PERIOD = 21
BB_STD_DEV = 2.7
STOP_LOSS_PCT = 2.0

# Scanner config
SCANNER_REFRESH_MINUTES = 30
SCANNER_TOP_N = 10


def calculate_bollinger_bands(prices, period=BB_PERIOD, std_dev=BB_STD_DEV):
    """
    Calculate Bollinger Bands.

    Returns: (middle_band, upper_band, lower_band)
    - Middle: 21-period SMA
    - Upper: Middle + 2.7 * std dev
    - Lower: Middle - 2.7 * std dev
    """
    if len(prices) < period:
        return None, None, None

    prices_arr = np.array(prices[-period:])
    middle = np.mean(prices_arr)
    std = np.std(prices_arr)

    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)

    return middle, upper, lower


def calculate_heikin_ashi(candles):
    """
    Calculate Heikin Ashi candles from regular OHLC.

    Heikin Ashi smooths price action:
    - HA Close = (Open + High + Low + Close) / 4
    - HA Open = (prev HA Open + prev HA Close) / 2
    - HA High = max(High, HA Open, HA Close)
    - HA Low = min(Low, HA Open, HA Close)

    Returns list of HA candles with 'color' field ('green' or 'red')
    """
    if not candles or len(candles) < 2:
        return []

    ha_candles = []

    for i, candle in enumerate(candles):
        ha_close = (candle['open'] + candle['high'] + candle['low'] + candle['close']) / 4

        if i == 0:
            ha_open = (candle['open'] + candle['close']) / 2
        else:
            ha_open = (ha_candles[i-1]['open'] + ha_candles[i-1]['close']) / 2

        ha_high = max(candle['high'], ha_open, ha_close)
        ha_low = min(candle['low'], ha_open, ha_close)

        color = 'green' if ha_close > ha_open else 'red'

        ha_candles.append({
            'open': ha_open,
            'high': ha_high,
            'low': ha_low,
            'close': ha_close,
            'color': color
        })

    return ha_candles


class BollingerBreakoutBot:
    """
    Bollinger Band Breakout Bot.

    Moon Dev says: "Buy strength, not weakness. When price breaks above
    the upper band with momentum, that's expansion - ride the wave."
    """

    def __init__(self, dry_run=False, mock_mode=False, use_scanner=False):
        self.dry_run = dry_run
        self.mock_mode = mock_mode
        self.use_scanner = use_scanner
        self.api = None

        # Initialize Alpaca API
        if not mock_mode:
            if not ALPACA_AVAILABLE:
                raise RuntimeError("alpaca-trade-api package not installed")
            if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required in .env")

            base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
            self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

        # Track positions
        self.active_trades = {}
        self.cooldowns = {}
        self.cooldown_minutes = 60
        self.cycle_count = 0

        # Scanner state
        self.current_symbols = SYMBOLS.copy()
        self.last_scan_time = None

        # Mock data
        self.mock_prices = {}
        self.mock_candles = {}

        if self.dry_run:
            cprint("*** DRY RUN MODE - No real trades will be executed ***", "yellow")
        if self.mock_mode:
            cprint("*** MOCK MODE - Using simulated price data ***", "yellow")
            self._init_mock_data()
        else:
            account = self.api.get_account()
            cprint(f"Connected to Alpaca ({'PAPER' if ALPACA_PAPER else 'LIVE'})", "green")
            cprint(f"Equity: ${float(account.equity):,.2f}", "green")

        self._print_banner()

        if self.use_scanner:
            cprint("*** SCANNER MODE - Dynamic symbol selection ***", "yellow")
            self._refresh_symbols()
        else:
            cprint(f"Trading symbols: {self.current_symbols}", "cyan")

    def _print_banner(self):
        """Print startup banner."""
        cprint("\n" + "=" * 60, "magenta")
        cprint("  BOLLINGER BAND BREAKOUT BOT", "magenta", attrs=['bold'])
        cprint("  Moon Dev's Volatility Breakout Strategy", "magenta")
        cprint("=" * 60, "magenta")
        cprint("\n  STRATEGY RULES:", "white")
        cprint("  ─────────────────────────────────────", "white")
        cprint(f"  Bollinger Bands: {BB_PERIOD} period, {BB_STD_DEV} std dev", "cyan")
        cprint("  Entry: Price CLOSES ABOVE upper band", "green")
        cprint("  Confirmation: Heikin Ashi must be GREEN", "green")
        cprint("  Exit: Price returns to middle band (21 SMA)", "yellow")
        cprint(f"  Stop Loss: {STOP_LOSS_PCT}% below entry", "red")
        cprint("\n  PHILOSOPHY:", "white")
        cprint("  'Buy strength, not weakness. Ride the expansion.'", "cyan")
        cprint("=" * 60 + "\n", "magenta")

        risk_status = get_risk_status()
        cprint(f"Order size: ${ORDER_USD_SIZE} | Daily loss limit: ${-risk_status['daily_loss_limit']:,.0f}", "white")

    def _init_mock_data(self):
        """Initialize mock price data."""
        base_prices = {
            'SPY': 590.0, 'TSLA': 410.0, 'AMD': 125.0,
            'NVDA': 140.0, 'AAPL': 240.0, 'QQQ': 520.0
        }

        for symbol in self.current_symbols:
            base = base_prices.get(symbol, 100 + random.uniform(0, 200))
            self.mock_prices[symbol] = base
            self.mock_candles[symbol] = self._generate_mock_candles(base, 50)

    def _generate_mock_candles(self, base_price, num_candles):
        """Generate mock OHLCV candles with trending behavior."""
        candles = []
        price = base_price * random.uniform(0.95, 1.0)

        for i in range(num_candles):
            # Occasional volatility expansion
            if random.random() < 0.1:
                volatility = base_price * 0.025  # 2.5% move
            else:
                volatility = base_price * 0.008

            open_price = price
            close = open_price + random.uniform(-volatility, volatility * 1.5)  # Slight upward bias
            high = max(open_price, close) + random.uniform(0, volatility * 0.5)
            low = min(open_price, close) - random.uniform(0, volatility * 0.5)

            candles.append({
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': random.uniform(100000, 1000000)
            })
            price = close

        return candles

    def _refresh_symbols(self):
        """Refresh symbol list from scanner."""
        if not self.use_scanner:
            return

        now = datetime.now()
        if self.last_scan_time:
            elapsed = (now - self.last_scan_time).total_seconds() / 60
            if elapsed < SCANNER_REFRESH_MINUTES:
                return

        cprint(f"\n--- Refreshing symbols from scanner ---", "yellow")

        try:
            candidates = get_breakout_candidates(
                top_n=SCANNER_TOP_N,
                threshold_pct=2.0,
                mock_mode=self.mock_mode
            )

            if candidates:
                new_symbols = [c['symbol'] for c in candidates]

                for symbol in self.active_trades.keys():
                    if symbol not in new_symbols:
                        new_symbols.append(symbol)

                self.current_symbols = new_symbols
                self.last_scan_time = now

                if self.mock_mode:
                    for symbol in new_symbols:
                        if symbol not in self.mock_prices:
                            base = 100 + random.uniform(0, 300)
                            self.mock_prices[symbol] = base
                            self.mock_candles[symbol] = self._generate_mock_candles(base, 50)

                cprint(f"Scanner found {len(candidates)} candidates", "green")
        except Exception as e:
            cprint(f"Scanner error: {e}", "red")

    def fetch_candles(self, symbol, hours=50):
        """Fetch hourly candles."""
        if self.mock_mode:
            return self.mock_candles.get(symbol, [])

        try:
            end = datetime.now()
            start = end - timedelta(hours=hours + 8)

            bars = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=hours
            ).df

            if bars.empty:
                return []

            return [
                {
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                for _, row in bars.iterrows()
            ]
        except Exception as e:
            cprint(f"Error fetching candles for {symbol}: {e}", "red")
            return []

    def get_current_price(self, symbol):
        """Get current price."""
        if self.mock_mode:
            if symbol in self.mock_prices:
                base = self.mock_prices[symbol]
                movement = base * random.uniform(-0.004, 0.004)
                self.mock_prices[symbol] = base + movement
                return self.mock_prices[symbol]
            return None

        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except:
            return None

    def is_on_cooldown(self, symbol):
        """Check if symbol is on cooldown."""
        if symbol not in self.cooldowns:
            return False
        if datetime.now() > self.cooldowns[symbol]:
            del self.cooldowns[symbol]
            return False
        return True

    def set_cooldown(self, symbol):
        """Set cooldown after trade."""
        self.cooldowns[symbol] = datetime.now() + timedelta(minutes=self.cooldown_minutes)

    def check_entry_signal(self, symbol, candles):
        """
        Check for Bollinger Band breakout entry.

        Entry conditions:
        1. Current close is ABOVE upper Bollinger Band
        2. Heikin Ashi candle is GREEN (bullish confirmation)
        """
        if len(candles) < BB_PERIOD + 1:
            return False, None, None, None, None

        # Get prices for Bollinger calculation
        prices = [c['close'] for c in candles]
        current_price = prices[-1]

        # Calculate Bollinger Bands
        middle, upper, lower = calculate_bollinger_bands(prices)
        if middle is None:
            return False, None, None, None, None

        # Calculate Heikin Ashi
        ha_candles = calculate_heikin_ashi(candles)
        if not ha_candles:
            return False, None, None, None, None

        current_ha = ha_candles[-1]

        # Check entry conditions
        price_above_upper = current_price > upper
        ha_is_green = current_ha['color'] == 'green'

        if price_above_upper and ha_is_green:
            return True, current_price, middle, upper, lower

        return False, current_price, middle, upper, lower

    def check_exit_signal(self, symbol, current_price, entry_price, candles):
        """
        Check for exit conditions.

        Exit when:
        1. Price returns to middle band (21 SMA) - profit target
        2. Stop loss hit (2% below entry)
        """
        if len(candles) < BB_PERIOD:
            return False, None

        prices = [c['close'] for c in candles]
        middle, upper, lower = calculate_bollinger_bands(prices)

        if middle is None:
            return False, None

        stop_price = entry_price * (1 - STOP_LOSS_PCT / 100)

        # Check stop loss
        if current_price <= stop_price:
            return True, f"Stop loss hit (${current_price:.2f} <= ${stop_price:.2f})"

        # Check if returned to middle band
        if current_price <= middle:
            return True, f"Returned to middle band (${current_price:.2f} <= ${middle:.2f})"

        return False, None

    def enter_position(self, symbol, entry_price, middle, upper):
        """Enter a long position."""
        if self.is_on_cooldown(symbol):
            cprint(f"    {symbol} on cooldown, skipping", "yellow")
            return False

        shares = round(ORDER_USD_SIZE / entry_price, 4)
        if shares < 0.0001:
            cprint(f"    Order size too small for {symbol}", "yellow")
            return False

        # Risk check
        position_value = shares * entry_price
        risk_allowed, risk_reason = check_risk_verbose(symbol, position_value)
        if not risk_allowed:
            cprint(f"    RISK BLOCKED: {risk_reason}", "red")
            return False

        stop_loss = entry_price * (1 - STOP_LOSS_PCT / 100)
        take_profit = middle  # Target is middle band

        if self.mock_mode or self.dry_run:
            mode = 'MOCK' if self.mock_mode else 'DRY RUN'
            cprint(f"[{mode}] BB BREAKOUT LONG: {symbol} | {shares} shares @ ${entry_price:.2f}", "magenta")
            cprint(f"    Upper Band: ${upper:.2f} | Target (Middle): ${middle:.2f} | SL: ${stop_loss:.2f}", "cyan")

            self.active_trades[symbol] = {
                'entry': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'direction': 'LONG',
                'shares': shares,
                'upper_at_entry': upper,
                'middle_at_entry': middle
            }
            update_position(symbol, position_value)
            self.set_cooldown(symbol)
            return True

        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='day'
            )

            cprint(f"BB BREAKOUT LONG: {symbol} | {shares} shares @ ${entry_price:.2f}", "magenta")
            cprint(f"    Order ID: {order.id}", "green")

            self.active_trades[symbol] = {
                'entry': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'direction': 'LONG',
                'shares': shares,
                'order_id': order.id
            }
            update_position(symbol, position_value)
            self.set_cooldown(symbol)
            return True

        except Exception as e:
            cprint(f"    Error entering {symbol}: {e}", "red")
            return False

    def close_position(self, symbol, current_price, reason):
        """Close a position."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        entry = trade['entry']
        shares = trade['shares']

        pnl_pct = ((current_price - entry) / entry) * 100
        pnl_dollars = (current_price - entry) * shares

        if self.mock_mode or self.dry_run:
            mode = 'MOCK' if self.mock_mode else 'DRY RUN'
            color = "green" if pnl_pct > 0 else "red"
            cprint(f"[{mode}] CLOSE {symbol}: {reason} | PnL: {pnl_pct:+.2f}% (${pnl_dollars:+.2f})", color)

            close_position(symbol, pnl_dollars)
            reset_position(symbol)
            del self.active_trades[symbol]
            return

        try:
            self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side='sell',
                type='market',
                time_in_force='day'
            )

            color = "green" if pnl_pct > 0 else "red"
            cprint(f"CLOSE {symbol}: {reason} | PnL: {pnl_pct:+.2f}% (${pnl_dollars:+.2f})", color)

            close_position(symbol, pnl_dollars)
            reset_position(symbol)
            del self.active_trades[symbol]

        except Exception as e:
            cprint(f"Error closing {symbol}: {e}", "red")

    def manage_position(self, symbol, current_price, candles):
        """Manage an active position."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        entry = trade['entry']

        # Check exit conditions
        should_exit, exit_reason = self.check_exit_signal(symbol, current_price, entry, candles)

        if should_exit:
            self.close_position(symbol, current_price, exit_reason)
            return

        # Also check exit agent for trailing stops
        entry_time = trade.get('entry_time', datetime.now())
        stop_loss = trade['stop_loss']
        take_profit = trade['take_profit']

        exit_decision, exit_reason_agent, new_stop = manage_exit_verbose(
            symbol, entry, entry_time, current_price,
            stop_loss, take_profit, 'LONG'
        )

        if exit_decision == 'CLOSE_FULL':
            self.close_position(symbol, current_price, exit_reason_agent)
        elif exit_decision == 'MOVE_STOP':
            self.active_trades[symbol]['stop_loss'] = new_stop
            cprint(f"    {symbol}: {exit_reason_agent}", "green")

    def update_mock_data(self):
        """Update mock candles."""
        for symbol in self.current_symbols:
            if symbol in self.mock_candles and self.mock_candles[symbol]:
                last = self.mock_candles[symbol][-1]
                volatility = last['close'] * 0.01

                # Occasionally create breakout conditions
                if random.random() < 0.08:
                    volatility *= 3

                new_close = last['close'] + random.uniform(-volatility, volatility * 1.2)
                new_candle = {
                    'open': last['close'],
                    'high': max(last['close'], new_close) + random.uniform(0, volatility * 0.3),
                    'low': min(last['close'], new_close) - random.uniform(0, volatility * 0.3),
                    'close': new_close,
                    'volume': random.uniform(100000, 500000)
                }

                self.mock_candles[symbol] = self.mock_candles[symbol][-49:] + [new_candle]
                self.mock_prices[symbol] = new_close

    def is_market_open(self):
        """Check if market is open."""
        if self.mock_mode:
            return True
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except:
            return False

    def check_symbols(self):
        """Main loop - check all symbols for Bollinger breakouts."""
        if self.use_scanner:
            self._refresh_symbols()

        risk_status = get_risk_status()
        cprint(f"\n{'─'*60}", "magenta")
        cprint(f"  Cycle {self.cycle_count} | {datetime.now().strftime('%H:%M:%S')} | Daily P&L: ${risk_status['daily_pnl']:+,.2f}", "magenta")
        cprint(f"{'─'*60}", "magenta")

        if not risk_status['trading_allowed']:
            cprint("TRADING HALTED: Daily loss limit reached!", "red")
            self.cycle_count += 1
            return

        if not self.mock_mode and not self.is_market_open():
            cprint("Market is closed", "yellow")
            self.cycle_count += 1
            return

        if self.mock_mode:
            self.update_mock_data()

        for symbol in self.current_symbols:
            candles = self.fetch_candles(symbol, hours=50)
            current_price = self.get_current_price(symbol)

            if current_price is None or len(candles) < BB_PERIOD:
                continue

            # Calculate indicators for display
            prices = [c['close'] for c in candles]
            middle, upper, lower = calculate_bollinger_bands(prices)
            ha_candles = calculate_heikin_ashi(candles)
            ha_color = ha_candles[-1]['color'] if ha_candles else 'unknown'

            if middle:
                bb_width = ((upper - lower) / middle) * 100
                price_vs_upper = ((current_price - upper) / upper) * 100

                cprint(f"\n{symbol}: ${current_price:.2f} | BB Width: {bb_width:.1f}% | HA: {ha_color.upper()}", "white")
                cprint(f"    Upper: ${upper:.2f} ({price_vs_upper:+.1f}%) | Middle: ${middle:.2f} | Lower: ${lower:.2f}", "cyan")

            # Manage existing position
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                pnl_pct = ((current_price - trade['entry']) / trade['entry']) * 100
                cprint(f"    Active LONG: Entry ${trade['entry']:.2f} | PnL: {pnl_pct:+.2f}%", "magenta")
                self.manage_position(symbol, current_price, candles)
                continue

            # Check for entry
            if self.is_on_cooldown(symbol):
                continue

            should_enter, price, mid, up, low = self.check_entry_signal(symbol, candles)

            if should_enter:
                cprint(f"    BREAKOUT SIGNAL: Price ${price:.2f} > Upper ${up:.2f} + HA GREEN", "yellow")
                self.enter_position(symbol, price, mid, up)
            else:
                if ha_color == 'red' and current_price > upper:
                    cprint(f"    Near breakout but HA is RED - waiting for confirmation", "yellow")

        self.cycle_count += 1

    def run(self, interval_seconds=60, duration_minutes=None):
        """Run the bot."""
        cprint("\nStarting Bollinger Breakout Bot...", "green")
        cprint(f"Check interval: {interval_seconds}s", "cyan")
        if duration_minutes:
            cprint(f"Duration: {duration_minutes} minutes", "cyan")

        start_time = datetime.now()
        self.check_symbols()

        schedule.every(interval_seconds).seconds.do(self.check_symbols)

        try:
            while True:
                if duration_minutes:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= duration_minutes:
                        cprint(f"\nDuration of {duration_minutes} minutes reached.", "yellow")
                        break
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            cprint("\nBot stopped by user", "yellow")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Bollinger Band Breakout Bot")
    parser.add_argument("--dry-run", action="store_true", help="Test mode without real trades")
    parser.add_argument("--mock", action="store_true", help="Use mock price data")
    parser.add_argument("--interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--duration", type=int, default=None, help="Stop after X minutes")
    parser.add_argument("--use-scanner", action="store_true", help="Dynamic symbol selection")
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    bot = BollingerBreakoutBot(
        dry_run=args.dry_run,
        mock_mode=args.mock,
        use_scanner=args.use_scanner
    )
    bot.run(interval_seconds=args.interval, duration_minutes=args.duration)


if __name__ == "__main__":
    main()

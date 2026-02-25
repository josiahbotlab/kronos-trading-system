"""
Mean Reversion Trading Bot

Strategy: Buy oversold conditions, sell on mean reversion
- Entry: RSI < 30 (oversold) AND price > 2% below 20-period SMA
- Exit: RSI > 50 OR price returns to SMA
- Stop Loss: 3% below entry

Uses same infrastructure as breakout bot:
- Risk agent for position sizing and daily loss limits
- Exit agent for trailing stops and partial profits
- Scanner for dynamic symbol selection

Usage:
    python src/agents/mean_reversion_bot.py              # Live trading
    python src/agents/mean_reversion_bot.py --dry-run    # Test mode
    python src/agents/mean_reversion_bot.py --mock       # Mock data
    python src/agents/mean_reversion_bot.py --use-scanner --duration 60
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

# Risk management
from src.agents.risk_agent import check_risk_verbose, update_position, close_position, get_risk_status

# Exit management
from src.agents.exit_agent import manage_exit_verbose, reset_position, get_position_status

# Scanner for dynamic symbol selection
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

# Trading parameters - Mean Reversion Strategy
SYMBOLS = ['SPY', 'QQQ', 'IWM', 'DIA']  # ETFs work well for mean reversion
ORDER_USD_SIZE = 500  # Per-trade size in USD

# Strategy parameters
RSI_PERIOD = 14
RSI_OVERSOLD = 30       # Buy when RSI < 30
RSI_EXIT = 50           # Exit when RSI > 50
SMA_PERIOD = 20
SMA_DEVIATION = 0.02    # 2% below SMA to enter
STOP_LOSS_PCT = 3.0     # 3% stop loss

# Scanner configuration
SCANNER_REFRESH_MINUTES = 30
SCANNER_TOP_N = 10


def calculate_rsi(prices, period=RSI_PERIOD):
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return 50  # Neutral if not enough data

    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_sma(prices, period=SMA_PERIOD):
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return np.mean(prices) if prices else 0
    return np.mean(prices[-period:])


class MeanReversionBot:
    """Bot that trades mean reversion setups on stocks/ETFs."""

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

        # Track positions and levels
        self.active_trades = {}
        self.cooldowns = {}
        self.cooldown_minutes = 60
        self.cycle_count = 0

        # Scanner state
        self.current_symbols = SYMBOLS.copy()
        self.last_scan_time = None

        # Mock data state
        self.mock_prices = {}
        self.mock_price_history = {}

        if self.dry_run:
            cprint("*** DRY RUN MODE - No real trades will be executed ***", "yellow")
        if self.mock_mode:
            cprint("*** MOCK MODE - Using simulated price data ***", "yellow")
            self._init_mock_data()
        else:
            account = self.api.get_account()
            cprint(f"Connected to Alpaca ({'PAPER' if ALPACA_PAPER else 'LIVE'})", "green")
            cprint(f"Account: {account.account_number}", "green")
            cprint(f"Equity: ${float(account.equity):,.2f}", "green")
            cprint(f"Buying Power: ${float(account.buying_power):,.2f}", "cyan")

        cprint(f"\n{'='*50}", "cyan")
        cprint("  MEAN REVERSION STRATEGY", "cyan")
        cprint(f"{'='*50}", "cyan")
        cprint(f"  Entry: RSI < {RSI_OVERSOLD} AND price > {SMA_DEVIATION*100:.0f}% below SMA({SMA_PERIOD})", "white")
        cprint(f"  Exit:  RSI > {RSI_EXIT} OR price >= SMA", "white")
        cprint(f"  Stop:  {STOP_LOSS_PCT}% below entry", "white")
        cprint(f"{'='*50}\n", "cyan")

        if self.use_scanner:
            cprint("*** SCANNER MODE - Dynamic symbol selection enabled ***", "yellow")
            self._refresh_symbols()
        else:
            cprint(f"Trading symbols: {self.current_symbols}", "cyan")

        cprint(f"Order size: ${ORDER_USD_SIZE}", "cyan")

        # Display risk limits
        risk_status = get_risk_status()
        cprint(f"Risk limits: Daily loss ${-risk_status['daily_loss_limit']:,.0f} | Max position $5,000 | Max exposure $20,000", "cyan")

    def _init_mock_data(self):
        """Initialize mock price data with trending behavior."""
        base_prices = {
            'SPY': 590.0,
            'QQQ': 520.0,
            'IWM': 220.0,
            'DIA': 430.0,
            'AAPL': 240.0,
            'TSLA': 410.0,
            'NVDA': 140.0,
        }

        for symbol in self.current_symbols:
            base = base_prices.get(symbol, 100 + random.uniform(0, 200))
            self.mock_prices[symbol] = base
            # Generate price history for indicators
            self.mock_price_history[symbol] = self._generate_price_history(base, 50)

    def _generate_price_history(self, base_price, num_bars):
        """Generate mock price history with mean-reverting behavior."""
        prices = []
        price = base_price * random.uniform(0.95, 1.05)

        for i in range(num_bars):
            # Add some trend and mean reversion
            trend = random.uniform(-0.003, 0.003)
            mean_reversion = (base_price - price) / base_price * 0.1
            noise = random.uniform(-0.005, 0.005)

            price = price * (1 + trend + mean_reversion + noise)
            prices.append(price)

        return prices

    def _refresh_symbols(self):
        """Refresh symbol list from scanner."""
        if not self.use_scanner:
            return

        now = datetime.now()
        if self.last_scan_time:
            elapsed_mins = (now - self.last_scan_time).total_seconds() / 60
            if elapsed_mins < SCANNER_REFRESH_MINUTES:
                return

        cprint(f"\n--- Refreshing symbols from scanner ---", "yellow")

        try:
            candidates = get_breakout_candidates(
                top_n=SCANNER_TOP_N,
                threshold_pct=3.0,
                mock_mode=self.mock_mode
            )

            if candidates:
                new_symbols = [c['symbol'] for c in candidates]

                # Keep symbols with active positions
                for symbol in self.active_trades.keys():
                    if symbol not in new_symbols:
                        new_symbols.append(symbol)

                self.current_symbols = new_symbols
                self.last_scan_time = now

                # Initialize mock data for new symbols
                if self.mock_mode:
                    for symbol in new_symbols:
                        if symbol not in self.mock_prices:
                            base = 100 + random.uniform(0, 300)
                            self.mock_prices[symbol] = base
                            self.mock_price_history[symbol] = self._generate_price_history(base, 50)

                cprint(f"Scanner found {len(candidates)} candidates", "green")
            else:
                cprint("Scanner returned no candidates, keeping current symbols", "yellow")

        except Exception as e:
            cprint(f"Scanner error: {e}, keeping current symbols", "red")

    def fetch_price_history(self, symbol, bars=50):
        """Fetch price history for indicator calculation."""
        if self.mock_mode:
            return self.mock_price_history.get(symbol, [])

        try:
            end = datetime.now()
            start = end - timedelta(hours=bars + 8)

            bars_data = self.api.get_bars(
                symbol,
                tradeapi.TimeFrame.Hour,
                start=start.isoformat(),
                end=end.isoformat(),
                limit=bars
            ).df

            if bars_data.empty:
                return []

            return bars_data['close'].tolist()

        except Exception as e:
            cprint(f"Error fetching history for {symbol}: {e}", "red")
            return []

    def get_current_price(self, symbol):
        """Get current price."""
        if self.mock_mode:
            if symbol in self.mock_prices:
                # Update price with some randomness
                base = self.mock_prices[symbol]
                movement = base * random.uniform(-0.002, 0.002)
                self.mock_prices[symbol] = base + movement

                # Update price history
                if symbol in self.mock_price_history:
                    self.mock_price_history[symbol].append(self.mock_prices[symbol])
                    self.mock_price_history[symbol] = self.mock_price_history[symbol][-50:]

                return self.mock_prices[symbol]
            return None

        try:
            trade = self.api.get_latest_trade(symbol)
            return float(trade.price)
        except Exception as e:
            cprint(f"Error getting price for {symbol}: {e}", "red")
            return None

    def check_entry_signal(self, symbol, current_price, prices):
        """Check for mean reversion entry signal."""
        if len(prices) < max(RSI_PERIOD + 1, SMA_PERIOD):
            return False, None, None

        rsi = calculate_rsi(prices)
        sma = calculate_sma(prices)

        # Calculate how far below SMA
        sma_deviation = (sma - current_price) / sma if sma > 0 else 0

        # Entry conditions:
        # 1. RSI < 30 (oversold)
        # 2. Price > 2% below SMA
        rsi_oversold = rsi < RSI_OVERSOLD
        below_sma = sma_deviation >= SMA_DEVIATION

        if rsi_oversold and below_sma:
            return True, rsi, sma

        return False, rsi, sma

    def check_exit_signal(self, symbol, current_price, prices, entry_price):
        """Check for mean reversion exit signal."""
        if len(prices) < max(RSI_PERIOD + 1, SMA_PERIOD):
            return False, None

        rsi = calculate_rsi(prices)
        sma = calculate_sma(prices)

        # Exit conditions:
        # 1. RSI > 50 (returned to neutral)
        # 2. Price >= SMA (mean reverted)
        # 3. Stop loss hit (3% below entry)
        rsi_exit = rsi > RSI_EXIT
        price_at_sma = current_price >= sma
        stop_loss_price = entry_price * (1 - STOP_LOSS_PCT / 100)
        stop_hit = current_price <= stop_loss_price

        if stop_hit:
            return True, f"Stop loss hit (${current_price:.2f} <= ${stop_loss_price:.2f})"
        if rsi_exit:
            return True, f"RSI exit ({rsi:.1f} > {RSI_EXIT})"
        if price_at_sma:
            return True, f"Mean reverted (${current_price:.2f} >= SMA ${sma:.2f})"

        return False, None

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

    def get_position(self, symbol):
        """Get current position for symbol."""
        if self.mock_mode:
            return None

        try:
            position = self.api.get_position(symbol)
            return {
                'qty': float(position.qty),
                'side': 'long' if float(position.qty) > 0 else 'short',
                'entry': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc) * 100
            }
        except:
            return None

    def enter_position(self, symbol, entry_price, rsi, sma):
        """Enter a long position."""
        try:
            if self.is_on_cooldown(symbol):
                cprint(f"  {symbol} on cooldown, skipping entry", "yellow")
                return False

            # Calculate shares (fractional shares supported)
            shares = round(ORDER_USD_SIZE / entry_price, 4)
            if shares < 0.0001:
                cprint(f"  Order size too small for {symbol}", "yellow")
                return False

            # Risk check before entry
            position_value = shares * entry_price
            risk_allowed, risk_reason = check_risk_verbose(symbol, position_value)
            if not risk_allowed:
                cprint(f"  RISK BLOCKED: {risk_reason}", "red")
                return False

            # Calculate levels
            stop_loss = entry_price * (1 - STOP_LOSS_PCT / 100)
            take_profit = sma  # Target is mean reversion to SMA

            # Mock mode
            if self.mock_mode:
                cprint(f"[MOCK] LONG: {symbol} | {shares} shares @ ${entry_price:.2f}", "magenta")
                cprint(f"[MOCK] RSI: {rsi:.1f} | SMA: ${sma:.2f} | SL: ${stop_loss:.2f}", "cyan")
                self.active_trades[symbol] = {
                    'entry': entry_price,
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'direction': 'LONG',
                    'shares': shares,
                    'rsi_entry': rsi,
                    'sma_entry': sma
                }
                update_position(symbol, position_value)
                self.set_cooldown(symbol)
                return True

            # Check existing position
            existing = self.get_position(symbol)
            if existing:
                cprint(f"  Already in position for {symbol}, skipping entry", "yellow")
                return False

            cprint(f"LONG: {symbol} | {shares} shares @ ${entry_price:.2f}", "magenta")
            cprint(f"RSI: {rsi:.1f} | SMA: ${sma:.2f} | SL: ${stop_loss:.2f}", "cyan")

            if self.dry_run:
                cprint(f"[DRY RUN] Would enter LONG {symbol} with {shares} shares", "cyan")
                self.active_trades[symbol] = {
                    'entry': entry_price,
                    'entry_time': datetime.now(),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'direction': 'LONG',
                    'shares': shares,
                    'rsi_entry': rsi,
                    'sma_entry': sma
                }
                update_position(symbol, position_value)
                self.set_cooldown(symbol)
                return True

            # Execute trade
            order = self.api.submit_order(
                symbol=symbol,
                qty=shares,
                side='buy',
                type='market',
                time_in_force='day'
            )

            cprint(f"Order submitted: {order.id}", "green")
            self.active_trades[symbol] = {
                'entry': entry_price,
                'entry_time': datetime.now(),
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'direction': 'LONG',
                'shares': shares,
                'rsi_entry': rsi,
                'sma_entry': sma,
                'order_id': order.id
            }
            update_position(symbol, position_value)
            self.set_cooldown(symbol)
            return True

        except Exception as e:
            cprint(f"Error entering position for {symbol}: {e}", "red")
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
            order = self.api.submit_order(
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
            cprint(f"Error closing position for {symbol}: {e}", "red")

    def manage_position(self, symbol, current_price, prices):
        """Manage an active position."""
        if symbol not in self.active_trades:
            return

        trade = self.active_trades[symbol]
        entry = trade['entry']

        # Check mean reversion exit conditions
        should_exit, exit_reason = self.check_exit_signal(symbol, current_price, prices, entry)

        if should_exit:
            self.close_position(symbol, current_price, exit_reason)
            return

        # Also check exit agent for trailing stops etc
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
            cprint(f"  {symbol}: {exit_reason_agent}", "green")

    def update_mock_prices(self):
        """Update mock prices with mean-reverting behavior."""
        for symbol in self.current_symbols:
            if symbol in self.mock_prices:
                base = self.mock_prices[symbol]
                history = self.mock_price_history.get(symbol, [base])

                # Calculate current SMA as target
                sma = np.mean(history[-SMA_PERIOD:]) if len(history) >= SMA_PERIOD else np.mean(history)

                # Mean reversion tendency + noise
                reversion = (sma - base) / sma * 0.05
                noise = random.uniform(-0.008, 0.008)

                # Sometimes create oversold conditions for demo
                if random.random() < 0.05:
                    noise = -0.02  # Drop to create opportunity

                new_price = base * (1 + reversion + noise)
                self.mock_prices[symbol] = new_price

                # Update history
                history.append(new_price)
                self.mock_price_history[symbol] = history[-50:]

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
        """Main loop - check all symbols for mean reversion setups."""
        if self.use_scanner:
            self._refresh_symbols()

        risk_status = get_risk_status()
        cprint(f"\n--- Cycle {self.cycle_count} at {datetime.now().strftime('%H:%M:%S')} | Daily P&L: ${risk_status['daily_pnl']:+,.2f} | Positions: {len(self.active_trades)} ---", "cyan")

        if not risk_status['trading_allowed']:
            cprint("TRADING HALTED: Daily loss limit reached!", "red")
            self.cycle_count += 1
            return

        if not self.mock_mode and not self.is_market_open():
            cprint("Market is closed", "yellow")
            self.cycle_count += 1
            return

        if self.mock_mode:
            self.update_mock_prices()

        for symbol in self.current_symbols:
            prices = self.fetch_price_history(symbol, bars=50)
            current_price = self.get_current_price(symbol)

            if current_price is None or len(prices) < SMA_PERIOD:
                continue

            rsi = calculate_rsi(prices)
            sma = calculate_sma(prices)
            sma_dev = ((sma - current_price) / sma * 100) if sma > 0 else 0

            cprint(
                f"{symbol}: ${current_price:.2f} | RSI: {rsi:.1f} | SMA: ${sma:.2f} ({sma_dev:+.1f}%)",
                "white"
            )

            # Check active position management
            if symbol in self.active_trades:
                trade = self.active_trades[symbol]
                pnl_pct = ((current_price - trade['entry']) / trade['entry']) * 100
                cprint(f"  Active LONG: Entry ${trade['entry']:.2f} | {trade['shares']} shares | PnL: {pnl_pct:+.2f}%", "magenta")
                self.manage_position(symbol, current_price, prices)
                continue

            # Check for entry signal
            should_enter, entry_rsi, entry_sma = self.check_entry_signal(symbol, current_price, prices)

            if should_enter:
                cprint(f"  OVERSOLD SIGNAL: RSI {entry_rsi:.1f} < {RSI_OVERSOLD}, {sma_dev:.1f}% below SMA", "yellow")
                self.enter_position(symbol, current_price, entry_rsi, entry_sma)

        self.cycle_count += 1

    def run(self, interval_seconds=60, duration_minutes=None):
        """Run the bot with scheduled checks."""
        cprint("\nStarting Mean Reversion Bot...", "green")
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
                        cprint(f"\nDuration of {duration_minutes} minutes reached. Stopping bot.", "yellow")
                        break
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            cprint("\nBot stopped by user", "yellow")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mean Reversion Trading Bot"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in test mode without executing real trades"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock price data (no API calls needed)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Stop the bot after X minutes (default: run indefinitely)"
    )
    parser.add_argument(
        "--use-scanner",
        action="store_true",
        help="Use scanner to dynamically select symbols"
    )
    return parser.parse_args()


def main():
    """Entry point."""
    args = parse_args()
    bot = MeanReversionBot(
        dry_run=args.dry_run,
        mock_mode=args.mock,
        use_scanner=args.use_scanner
    )
    bot.run(interval_seconds=args.interval, duration_minutes=args.duration)


if __name__ == "__main__":
    main()

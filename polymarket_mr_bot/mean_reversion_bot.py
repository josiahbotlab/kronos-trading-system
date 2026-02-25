"""
Polymarket Mean Reversion Bot

Trades prediction markets using mean reversion strategy.
Buys when price deviates significantly below the mean,
sells when price deviates significantly above the mean.

Strategy:
1. Track rolling price history for each market
2. Calculate mean and standard deviation
3. Enter when price is >N std devs from mean (oversold/overbought)
4. Exit when price reverts toward mean

Usage:
    python -m polymarket_mr_bot.mean_reversion_bot
    python -m polymarket_mr_bot.mean_reversion_bot --dry-run
    python -m polymarket_mr_bot.mean_reversion_bot --scan-only
"""

import argparse
import csv
import signal
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from termcolor import cprint

# Import execution functions
from .nice_funcs_pm import (
    get_client,
    check_connection,
    get_markets,
    get_market,
    get_midpoint,
    get_spread,
    get_liquidity,
    get_positions,
    get_position_size,
    get_position_pnl,
    buy,
    sell,
    close_position,
    cancel_orders_for_token,
    get_balance,
    is_valid_price,
    has_sufficient_liquidity,
    format_price,
    calculate_shares,
    pnl_close,
    print_status,
)

# Import configuration
from .config import (
    ORDER_SIZE_USDC,
    MAX_POSITION_SIZE_USDC,
    MAX_OPEN_POSITIONS,
    MEAN_LOOKBACK_HOURS,
    STD_LOOKBACK_HOURS,
    ENTRY_ZSCORE_THRESHOLD,
    EXIT_ZSCORE_THRESHOLD,
    ENTRY_DEVIATION_PCT,
    EXIT_DEVIATION_PCT,
    STRATEGY_MODE,
    TAKE_PROFIT_PCT,
    STOP_LOSS_PCT,
    DAILY_LOSS_LIMIT_USDC,
    MIN_PRICE,
    MAX_PRICE,
    MIN_LIQUIDITY_USDC,
    MIN_VOLUME_24H_USDC,
    WHITELIST_MARKETS,
    BLACKLIST_MARKETS,
    MAX_DAYS_TO_EXPIRY,
    LOOP_INTERVAL_SECONDS,
    PRICE_UPDATE_INTERVAL,
    MIN_TRADE_INTERVAL,
    DATA_DIR,
    PRICE_HISTORY_CSV,
    TRADE_LOG_CSV,
    POSITION_LOG_CSV,
    DRY_RUN,
    VERBOSE_TRADES,
    VERBOSE_PRICES,
    MEAN_REVERSION_PARAMS,
)


# ============================================================================
# PRICE HISTORY TRACKING
# ============================================================================

class PriceHistory:
    """Track price history for mean reversion calculations."""

    def __init__(self, max_hours: int = 48):
        self.max_hours = max_hours
        # {token_id: [(timestamp, price), ...]}
        self.history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)

    def add_price(self, token_id: str, price: float, timestamp: Optional[datetime] = None):
        """Add a price observation."""
        if timestamp is None:
            timestamp = datetime.now()

        self.history[token_id].append((timestamp, price))
        self._cleanup(token_id)

    def _cleanup(self, token_id: str):
        """Remove old price data."""
        cutoff = datetime.now() - timedelta(hours=self.max_hours)
        self.history[token_id] = [
            (ts, px) for ts, px in self.history[token_id]
            if ts > cutoff
        ]

    def get_prices(self, token_id: str, hours: Optional[int] = None) -> List[float]:
        """Get price array for a token."""
        if hours is None:
            hours = self.max_hours

        cutoff = datetime.now() - timedelta(hours=hours)
        return [px for ts, px in self.history[token_id] if ts > cutoff]

    def get_mean(self, token_id: str, hours: Optional[int] = None) -> Optional[float]:
        """Calculate rolling mean."""
        prices = self.get_prices(token_id, hours)
        if len(prices) < 2:
            return None
        return np.mean(prices)

    def get_std(self, token_id: str, hours: Optional[int] = None) -> Optional[float]:
        """Calculate rolling standard deviation."""
        prices = self.get_prices(token_id, hours)
        if len(prices) < 2:
            return None
        return np.std(prices)

    def get_zscore(self, token_id: str, current_price: float) -> Optional[float]:
        """
        Calculate z-score of current price vs historical mean.

        Z-score = (price - mean) / std
        - Positive z-score: price above mean
        - Negative z-score: price below mean
        """
        mean = self.get_mean(token_id, MEAN_LOOKBACK_HOURS)
        std = self.get_std(token_id, STD_LOOKBACK_HOURS)

        if mean is None or std is None or std == 0:
            return None

        return (current_price - mean) / std

    def get_deviation_pct(self, token_id: str, current_price: float) -> Optional[float]:
        """
        Calculate percentage deviation from mean.

        Returns positive value if above mean, negative if below.
        """
        mean = self.get_mean(token_id, MEAN_LOOKBACK_HOURS)

        if mean is None or mean == 0:
            return None

        return ((current_price - mean) / mean) * 100

    def get_bollinger_bands(
        self,
        token_id: str,
        period: int = 20,
        num_std: float = 2.0
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Calculate Bollinger Bands.

        Returns:
            (lower_band, middle_band, upper_band)
        """
        prices = self.get_prices(token_id)
        if len(prices) < period:
            return None, None, None

        recent_prices = prices[-period:]
        middle = np.mean(recent_prices)
        std = np.std(recent_prices)

        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        return lower, middle, upper

    def data_points(self, token_id: str) -> int:
        """Get number of data points for a token."""
        return len(self.history.get(token_id, []))

    def save_to_csv(self, filepath: Path = PRICE_HISTORY_CSV):
        """Save price history to CSV."""
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['token_id', 'timestamp', 'price'])

            for token_id, prices in self.history.items():
                for ts, px in prices:
                    writer.writerow([token_id, ts.isoformat(), px])

    def load_from_csv(self, filepath: Path = PRICE_HISTORY_CSV):
        """Load price history from CSV."""
        if not filepath.exists():
            return

        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    token_id = row['token_id']
                    timestamp = datetime.fromisoformat(row['timestamp'])
                    price = float(row['price'])
                    self.history[token_id].append((timestamp, price))

            # Cleanup old data
            for token_id in self.history:
                self._cleanup(token_id)

            cprint(f"[DATA] Loaded price history from {filepath}", "green")

        except Exception as e:
            cprint(f"[ERROR] Failed to load price history: {e}", "red")


# ============================================================================
# TRADE LOGGING
# ============================================================================

def log_trade(
    action: str,
    token_id: str,
    market_name: str,
    side: str,
    size: float,
    price: float,
    reason: str,
    pnl: Optional[float] = None,
):
    """Log trade to CSV."""
    TRADE_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to write header
    write_header = not TRADE_LOG_CSV.exists()

    with open(TRADE_LOG_CSV, 'a', newline='') as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                'timestamp', 'action', 'token_id', 'market_name',
                'side', 'size', 'price', 'reason', 'pnl'
            ])

        writer.writerow([
            datetime.now().isoformat(),
            action,
            token_id,
            market_name[:50],
            side,
            f"{size:.4f}",
            f"{price:.4f}",
            reason,
            f"{pnl:.2f}" if pnl is not None else "",
        ])


# ============================================================================
# MEAN REVERSION BOT
# ============================================================================

class MeanReversionBot:
    """
    Mean Reversion Trading Bot for Polymarket.

    Identifies markets where prices have deviated significantly from
    their rolling mean and trades the expected reversion.
    """

    def __init__(
        self,
        dry_run: bool = DRY_RUN,
        scan_only: bool = False,
    ):
        self.dry_run = dry_run
        self.scan_only = scan_only
        self.running = False

        # Price history tracker
        self.price_history = PriceHistory(max_hours=48)

        # Active positions tracking
        self.active_positions: Dict[str, Dict] = {}

        # Market cache
        self.markets: List[Dict] = []
        self.market_tokens: Dict[str, Dict] = {}  # {token_id: market_info}

        # Daily P&L tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.last_trade_times: Dict[str, datetime] = {}

        # Timing
        self.last_price_update = datetime.min
        self.last_market_refresh = datetime.min

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        cprint("\n" + "=" * 50, "red")
        cprint("  SHUTDOWN SIGNAL RECEIVED", "red", attrs=['bold'])
        cprint("=" * 50, "red")

        self.running = False

        # Save price history
        self.price_history.save_to_csv()
        cprint("[SHUTDOWN] Price history saved", "green")

        # Print final status
        self._print_summary()

        sys.exit(0)

    # =========================================================================
    # MARKET DISCOVERY
    # =========================================================================

    def refresh_markets(self):
        """Refresh list of tradeable markets."""
        cprint("\n[MARKETS] Refreshing market list...", "cyan")

        all_markets = get_markets(limit=200)

        if not all_markets:
            cprint("[MARKETS] No markets found", "yellow")
            return

        # Filter markets
        filtered = []
        for market in all_markets:
            if self._is_market_eligible(market):
                filtered.append(market)

        self.markets = filtered
        cprint(f"[MARKETS] Found {len(filtered)} eligible markets", "green")

        # Build token -> market mapping
        self.market_tokens = {}
        for market in self.markets:
            tokens = market.get('tokens', [])
            for token in tokens:
                token_id = token.get('token_id')
                if token_id:
                    self.market_tokens[token_id] = {
                        'market': market,
                        'outcome': token.get('outcome', 'Unknown'),
                        'token_id': token_id,
                    }

        self.last_market_refresh = datetime.now()

    def _is_market_eligible(self, market: Dict) -> bool:
        """Check if market meets trading criteria."""
        condition_id = market.get('condition_id', '')
        question = market.get('question', '')

        # Whitelist/blacklist check
        if WHITELIST_MARKETS and condition_id not in WHITELIST_MARKETS:
            return False

        if condition_id in BLACKLIST_MARKETS:
            return False

        # Check expiry
        if MAX_DAYS_TO_EXPIRY > 0:
            end_date_str = market.get('end_date_iso')
            if end_date_str:
                try:
                    end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                    days_to_expiry = (end_date - datetime.now(end_date.tzinfo)).days
                    if days_to_expiry > MAX_DAYS_TO_EXPIRY or days_to_expiry < 1:
                        return False
                except:
                    pass

        # Must have tokens
        tokens = market.get('tokens', [])
        if len(tokens) < 2:
            return False

        # Check if active
        if not market.get('active', True):
            return False

        return True

    # =========================================================================
    # PRICE UPDATES
    # =========================================================================

    def update_prices(self):
        """Update price history for all tracked markets."""
        now = datetime.now()

        # Rate limit price updates
        if (now - self.last_price_update).total_seconds() < 30:
            return

        updated = 0
        for token_id, info in self.market_tokens.items():
            try:
                price = get_midpoint(token_id)
                if price is not None and 0 < price < 1:
                    self.price_history.add_price(token_id, price, now)
                    updated += 1

                    if VERBOSE_PRICES:
                        market = info.get('market', {})
                        outcome = info.get('outcome', '')
                        cprint(
                            f"[PRICE] {market.get('question', '')[:30]}... "
                            f"({outcome}): {format_price(price)}",
                            "white"
                        )

            except Exception as e:
                if VERBOSE_PRICES:
                    cprint(f"[ERROR] Price update failed for {token_id[:10]}...: {e}", "red")

        self.last_price_update = now

        if updated > 0 and not VERBOSE_PRICES:
            cprint(f"[PRICE] Updated {updated} prices", "cyan")

    # =========================================================================
    # SIGNAL GENERATION
    # =========================================================================

    def scan_for_signals(self) -> List[Dict]:
        """
        Scan markets for mean reversion signals.

        Returns:
            List of signal dicts with token_id, direction, strength, etc.
        """
        signals = []

        for token_id, info in self.market_tokens.items():
            signal = self._check_signal(token_id, info)
            if signal:
                signals.append(signal)

        # Sort by signal strength
        signals.sort(key=lambda x: abs(x.get('strength', 0)), reverse=True)

        return signals

    def _check_signal(self, token_id: str, info: Dict) -> Optional[Dict]:
        """Check if token has a valid trading signal."""
        # Need enough price history
        if self.price_history.data_points(token_id) < 10:
            return None

        # Get current price
        current_price = get_midpoint(token_id)
        if current_price is None:
            return None

        # Check price bounds
        if not is_valid_price(current_price):
            return None

        # Check liquidity
        if not has_sufficient_liquidity(token_id):
            return None

        # Check trade cooldown
        last_trade = self.last_trade_times.get(token_id)
        if last_trade:
            elapsed = (datetime.now() - last_trade).total_seconds()
            if elapsed < MIN_TRADE_INTERVAL:
                return None

        # Calculate signal based on strategy mode
        if STRATEGY_MODE == 'zscore':
            return self._check_zscore_signal(token_id, info, current_price)
        else:
            return self._check_pct_signal(token_id, info, current_price)

    def _check_zscore_signal(
        self,
        token_id: str,
        info: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """Check for z-score based signal."""
        zscore = self.price_history.get_zscore(token_id, current_price)
        if zscore is None:
            return None

        market = info.get('market', {})
        outcome = info.get('outcome', '')

        # Oversold signal (z-score < -threshold) -> BUY
        if zscore < -ENTRY_ZSCORE_THRESHOLD:
            return {
                'token_id': token_id,
                'market': market,
                'outcome': outcome,
                'direction': 'BUY',
                'current_price': current_price,
                'zscore': zscore,
                'strength': abs(zscore),
                'mean': self.price_history.get_mean(token_id),
                'reason': f"Oversold: z={zscore:.2f} < -{ENTRY_ZSCORE_THRESHOLD}",
            }

        # Overbought signal (z-score > threshold) -> SELL (if we have position)
        if zscore > ENTRY_ZSCORE_THRESHOLD:
            # Only signal sell if we have a long position
            pos_size = get_position_size(token_id)
            if pos_size > 0:
                return {
                    'token_id': token_id,
                    'market': market,
                    'outcome': outcome,
                    'direction': 'SELL',
                    'current_price': current_price,
                    'zscore': zscore,
                    'strength': abs(zscore),
                    'mean': self.price_history.get_mean(token_id),
                    'reason': f"Overbought: z={zscore:.2f} > {ENTRY_ZSCORE_THRESHOLD}",
                }

        return None

    def _check_pct_signal(
        self,
        token_id: str,
        info: Dict,
        current_price: float
    ) -> Optional[Dict]:
        """Check for percentage deviation signal."""
        deviation = self.price_history.get_deviation_pct(token_id, current_price)
        if deviation is None:
            return None

        market = info.get('market', {})
        outcome = info.get('outcome', '')

        # Oversold signal (deviation < -threshold) -> BUY
        if deviation < -ENTRY_DEVIATION_PCT:
            return {
                'token_id': token_id,
                'market': market,
                'outcome': outcome,
                'direction': 'BUY',
                'current_price': current_price,
                'deviation_pct': deviation,
                'strength': abs(deviation),
                'mean': self.price_history.get_mean(token_id),
                'reason': f"Oversold: {deviation:.1f}% < -{ENTRY_DEVIATION_PCT}%",
            }

        # Overbought signal
        if deviation > ENTRY_DEVIATION_PCT:
            pos_size = get_position_size(token_id)
            if pos_size > 0:
                return {
                    'token_id': token_id,
                    'market': market,
                    'outcome': outcome,
                    'direction': 'SELL',
                    'current_price': current_price,
                    'deviation_pct': deviation,
                    'strength': abs(deviation),
                    'mean': self.price_history.get_mean(token_id),
                    'reason': f"Overbought: {deviation:.1f}% > {ENTRY_DEVIATION_PCT}%",
                }

        return None

    # =========================================================================
    # EXIT SIGNAL CHECK
    # =========================================================================

    def check_exit_signals(self) -> List[Dict]:
        """Check for exit signals on existing positions."""
        exits = []

        positions = get_positions()
        for pos in positions:
            token_id = pos.get('asset', pos.get('token_id'))
            if not token_id:
                continue

            current_price = get_midpoint(token_id)
            if current_price is None:
                continue

            # Check mean reversion exit
            if STRATEGY_MODE == 'zscore':
                zscore = self.price_history.get_zscore(token_id, current_price)
                if zscore is not None and abs(zscore) < EXIT_ZSCORE_THRESHOLD:
                    exits.append({
                        'token_id': token_id,
                        'reason': f"Mean reversion: z={zscore:.2f}",
                        'current_price': current_price,
                    })
            else:
                deviation = self.price_history.get_deviation_pct(token_id, current_price)
                if deviation is not None and abs(deviation) < EXIT_DEVIATION_PCT:
                    exits.append({
                        'token_id': token_id,
                        'reason': f"Mean reversion: {deviation:.1f}%",
                        'current_price': current_price,
                    })

        return exits

    # =========================================================================
    # TRADE EXECUTION
    # =========================================================================

    def execute_signal(self, signal: Dict) -> bool:
        """Execute a trading signal."""
        token_id = signal['token_id']
        direction = signal['direction']
        price = signal['current_price']
        market = signal.get('market', {})
        outcome = signal.get('outcome', '')
        reason = signal.get('reason', '')

        market_name = market.get('question', 'Unknown')[:50]

        # Check if we can open more positions
        if direction == 'BUY':
            positions = get_positions()
            if len(positions) >= MAX_OPEN_POSITIONS:
                cprint(f"[SKIP] Max positions ({MAX_OPEN_POSITIONS}) reached", "yellow")
                return False

            # Check max position size
            current_size = get_position_size(token_id)
            if current_size * price >= MAX_POSITION_SIZE_USDC:
                cprint(f"[SKIP] Max position size reached for {token_id[:10]}...", "yellow")
                return False

            # Check daily loss limit
            if self.daily_pnl <= -DAILY_LOSS_LIMIT_USDC:
                cprint(f"[SKIP] Daily loss limit (${DAILY_LOSS_LIMIT_USDC}) reached", "red")
                return False

        # Calculate order size
        shares = calculate_shares(ORDER_SIZE_USDC, price)

        cprint(f"\n{'─' * 50}", "cyan")
        cprint(f"[SIGNAL] {direction} {outcome}", "cyan", attrs=['bold'])
        cprint(f"  Market: {market_name}", "white")
        cprint(f"  Price: {format_price(price)}", "white")
        cprint(f"  Size: {shares:.2f} shares (${ORDER_SIZE_USDC:.2f})", "white")
        cprint(f"  Reason: {reason}", "white")
        cprint(f"{'─' * 50}", "cyan")

        if self.scan_only:
            cprint("[SCAN ONLY] Would execute trade", "magenta")
            return False

        # Execute trade
        if direction == 'BUY':
            result = buy(token_id, shares, price)
        else:
            result = sell(token_id, shares, price)

        if result:
            # Log trade
            log_trade(
                action='ENTRY' if direction == 'BUY' else 'EXIT',
                token_id=token_id,
                market_name=market_name,
                side=direction,
                size=shares,
                price=price,
                reason=reason,
            )

            # Update tracking
            self.last_trade_times[token_id] = datetime.now()
            self.daily_trades += 1

            if VERBOSE_TRADES:
                cprint(f"[TRADE] {direction} executed successfully", "green")

            return True

        return False

    def execute_exit(self, exit_signal: Dict) -> bool:
        """Execute an exit signal."""
        token_id = exit_signal['token_id']
        reason = exit_signal['reason']
        price = exit_signal['current_price']

        # Get position details
        pnl_usdc, pnl_pct = get_position_pnl(token_id)

        cprint(f"\n{'─' * 50}", "yellow")
        cprint(f"[EXIT] Closing position", "yellow", attrs=['bold'])
        cprint(f"  Token: {token_id[:20]}...", "white")
        cprint(f"  Price: {format_price(price)}", "white")
        cprint(f"  P&L: ${pnl_usdc:+.2f} ({pnl_pct:+.1f}%)", "green" if pnl_usdc >= 0 else "red")
        cprint(f"  Reason: {reason}", "white")
        cprint(f"{'─' * 50}", "yellow")

        if self.scan_only:
            cprint("[SCAN ONLY] Would close position", "magenta")
            return False

        result = close_position(token_id)

        if result:
            # Update daily P&L
            self.daily_pnl += pnl_usdc

            # Log trade
            market_info = self.market_tokens.get(token_id, {})
            market = market_info.get('market', {})
            log_trade(
                action='EXIT',
                token_id=token_id,
                market_name=market.get('question', 'Unknown')[:50],
                side='SELL',
                size=get_position_size(token_id),
                price=price,
                reason=reason,
                pnl=pnl_usdc,
            )

            cprint(f"[EXIT] Position closed", "green")
            return True

        return False

    # =========================================================================
    # P&L MANAGEMENT
    # =========================================================================

    def check_tp_sl_all(self):
        """Check TP/SL for all positions."""
        positions = get_positions()

        for pos in positions:
            token_id = pos.get('asset', pos.get('token_id'))
            if not token_id:
                continue

            closed = pnl_close(token_id, TAKE_PROFIT_PCT, STOP_LOSS_PCT)
            if closed:
                pnl_usdc, _ = get_position_pnl(token_id)
                self.daily_pnl += pnl_usdc

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    def run(self):
        """Main bot loop."""
        self.running = True

        cprint("\n" + "=" * 60, "cyan")
        cprint("  POLYMARKET MEAN REVERSION BOT", "cyan", attrs=['bold'])
        cprint("=" * 60, "cyan")

        mode = "DRY RUN" if self.dry_run else "SCAN ONLY" if self.scan_only else "LIVE"
        cprint(f"\n  Mode: {mode}", "white")
        cprint(f"  Strategy: {STRATEGY_MODE.upper()}", "white")
        cprint(f"  Order Size: ${ORDER_SIZE_USDC:.2f}", "white")
        cprint(f"  Entry Threshold: {ENTRY_ZSCORE_THRESHOLD} std devs" if STRATEGY_MODE == 'zscore'
               else f"  Entry Threshold: {ENTRY_DEVIATION_PCT}%", "white")
        cprint(f"  TP/SL: +{TAKE_PROFIT_PCT}% / {STOP_LOSS_PCT}%", "white")

        # Check connection
        if not check_connection():
            cprint("\n[ERROR] Failed to connect to Polymarket", "red")
            return

        # Print initial status
        print_status()

        # Load price history
        self.price_history.load_from_csv()

        # Initial market refresh
        self.refresh_markets()

        cprint("\n[BOT] Starting main loop...", "green")
        cprint(f"[BOT] Loop interval: {LOOP_INTERVAL_SECONDS}s", "white")
        cprint("[BOT] Press Ctrl+C to stop\n", "white")

        while self.running:
            try:
                loop_start = datetime.now()

                # Refresh markets periodically (every hour)
                if (loop_start - self.last_market_refresh).total_seconds() > 3600:
                    self.refresh_markets()

                # Update prices
                self.update_prices()

                # Check TP/SL on existing positions
                self.check_tp_sl_all()

                # Check for exit signals (mean reversion)
                exit_signals = self.check_exit_signals()
                for exit_sig in exit_signals:
                    self.execute_exit(exit_sig)

                # Scan for entry signals
                signals = self.scan_for_signals()

                if signals:
                    cprint(f"\n[SCAN] Found {len(signals)} signals:", "cyan")
                    for sig in signals[:5]:  # Show top 5
                        market = sig.get('market', {})
                        cprint(
                            f"  {sig['direction']:4} | {market.get('question', '')[:40]}... | "
                            f"{format_price(sig['current_price'])} | {sig['reason']}",
                            "green" if sig['direction'] == 'BUY' else "red"
                        )

                    # Execute best signal
                    if signals and not self.scan_only:
                        self.execute_signal(signals[0])

                # Sleep until next loop
                elapsed = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, LOOP_INTERVAL_SECONDS - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                break
            except Exception as e:
                cprint(f"[ERROR] Loop error: {e}", "red")
                time.sleep(10)

        self._handle_shutdown(None, None)

    # =========================================================================
    # STATUS & SUMMARY
    # =========================================================================

    def _print_summary(self):
        """Print daily summary."""
        cprint("\n" + "=" * 50, "cyan")
        cprint("  DAILY SUMMARY", "cyan", attrs=['bold'])
        cprint("=" * 50, "cyan")

        cprint(f"\n  Trades: {self.daily_trades}", "white")
        pnl_color = "green" if self.daily_pnl >= 0 else "red"
        cprint(f"  Daily P&L: ${self.daily_pnl:+.2f}", pnl_color)
        cprint(f"  Markets Tracked: {len(self.market_tokens)}", "white")
        cprint(f"  Price Points: {sum(self.price_history.data_points(t) for t in self.market_tokens)}", "white")

        cprint("\n" + "=" * 50 + "\n", "cyan")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Polymarket Mean Reversion Bot")
    parser.add_argument("--dry-run", action="store_true", help="Simulate trades without executing")
    parser.add_argument("--scan-only", action="store_true", help="Only scan for signals, no trading")
    parser.add_argument("--status", action="store_true", help="Print status and exit")

    args = parser.parse_args()

    if args.status:
        print_status()
        return

    bot = MeanReversionBot(
        dry_run=args.dry_run or DRY_RUN,
        scan_only=args.scan_only,
    )

    bot.run()


if __name__ == "__main__":
    main()

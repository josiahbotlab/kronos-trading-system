"""
Order Execution Utilities

Robust order management for Alpaca trading:
- Position checking (prevent duplicates)
- Order cancellation (prevent orphans)
- Bracket orders (atomic TP/SL)
- Decimal precision handling
- Daily loss limit tracking

Usage:
    from src.utils.order_utils import (
        check_existing_position,
        cancel_symbol_orders,
        place_bracket_order,
        round_to_valid_qty,
        check_daily_loss_limit,
        reset_daily_tracker,
        get_daily_pnl,
    )
"""

import os
import time
from datetime import datetime, date
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Tuple

from termcolor import cprint
from dotenv import load_dotenv

load_dotenv()

# Alpaca imports
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    cprint("Warning: alpaca-trade-api not installed", "yellow")

# Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Daily loss limit (percentage of account)
DAILY_LOSS_LIMIT_PCT = 3.0

# Global state
_api = None
_daily_tracker = {
    'date': None,
    'starting_equity': 0,
    'realized_pnl': 0,
    'trading_halted': False,
}


def get_api():
    """Get or create Alpaca API instance."""
    global _api
    if _api is None:
        if not ALPACA_AVAILABLE:
            raise RuntimeError("alpaca-trade-api not installed")
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY required")

        base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
        _api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')
    return _api


def check_existing_position(symbol: str) -> Tuple[bool, Optional[dict]]:
    """
    Check if a position already exists for a symbol.

    Args:
        symbol: Stock symbol to check

    Returns:
        Tuple of (has_position, position_info)
        position_info contains: qty, market_value, unrealized_pnl, side
    """
    try:
        api = get_api()
        positions = api.list_positions()

        for pos in positions:
            if pos.symbol == symbol:
                qty = float(pos.qty)
                position_info = {
                    'symbol': symbol,
                    'qty': abs(qty),
                    'side': 'long' if qty > 0 else 'short',
                    'market_value': float(pos.market_value),
                    'avg_entry_price': float(pos.avg_entry_price),
                    'unrealized_pnl': float(pos.unrealized_pl),
                    'unrealized_pnl_pct': float(pos.unrealized_plpc) * 100,
                }
                cprint(f"  [ORDER] Found existing position: {symbol} {position_info['side'].upper()} {position_info['qty']} shares @ ${position_info['avg_entry_price']:.2f}", "yellow")
                return True, position_info

        return False, None

    except APIError as e:
        if 'position does not exist' in str(e).lower():
            return False, None
        cprint(f"  [ORDER] Error checking position for {symbol}: {e}", "red")
        return False, None
    except Exception as e:
        cprint(f"  [ORDER] Error checking position for {symbol}: {e}", "red")
        return False, None


def cancel_symbol_orders(symbol: str) -> int:
    """
    Cancel all open orders for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        Number of orders cancelled
    """
    try:
        api = get_api()
        orders = api.list_orders(status='open', symbols=[symbol])

        cancelled_count = 0
        for order in orders:
            try:
                api.cancel_order(order.id)
                cprint(f"  [ORDER] Cancelled order {order.id} ({order.side} {order.qty} {symbol})", "yellow")
                cancelled_count += 1
            except APIError as e:
                cprint(f"  [ORDER] Failed to cancel order {order.id}: {e}", "red")

        if cancelled_count > 0:
            # Wait for cancellations to process
            time.sleep(0.5)
            cprint(f"  [ORDER] Cancelled {cancelled_count} open orders for {symbol}", "yellow")

        return cancelled_count

    except Exception as e:
        cprint(f"  [ORDER] Error cancelling orders for {symbol}: {e}", "red")
        return 0


def get_asset_info(symbol: str) -> dict:
    """
    Get asset trading info (min qty, increment, etc.)

    Returns:
        Dict with min_qty, qty_increment, price_increment, tradable
    """
    try:
        api = get_api()
        asset = api.get_asset(symbol)

        return {
            'symbol': symbol,
            'tradable': asset.tradable,
            'fractionable': asset.fractionable,
            'min_order_size': float(asset.min_order_size) if hasattr(asset, 'min_order_size') else 0.0001,
            'min_trade_increment': float(asset.min_trade_increment) if hasattr(asset, 'min_trade_increment') else 0.0001,
            'price_increment': 0.01,  # Standard for stocks
        }
    except Exception as e:
        cprint(f"  [ORDER] Error getting asset info for {symbol}: {e}", "red")
        # Return defaults
        return {
            'symbol': symbol,
            'tradable': True,
            'fractionable': True,
            'min_order_size': 0.0001,
            'min_trade_increment': 0.0001,
            'price_increment': 0.01,
        }


def round_to_valid_qty(symbol: str, qty: float) -> float:
    """
    Round quantity to valid increment for the asset.

    Args:
        symbol: Stock symbol
        qty: Desired quantity

    Returns:
        Rounded quantity
    """
    asset_info = get_asset_info(symbol)
    increment = asset_info['min_trade_increment']
    min_qty = asset_info['min_order_size']

    # Round down to nearest increment
    original_qty = qty
    decimal_qty = Decimal(str(qty))
    decimal_increment = Decimal(str(increment))

    rounded_qty = float((decimal_qty / decimal_increment).to_integral_value(ROUND_DOWN) * decimal_increment)

    # Ensure minimum
    if rounded_qty < min_qty:
        rounded_qty = min_qty

    # Log if significant change
    if original_qty > 0:
        change_pct = abs(original_qty - rounded_qty) / original_qty * 100
        if change_pct > 1:
            cprint(f"  [ORDER] Quantity rounded: {original_qty:.4f} -> {rounded_qty:.4f} ({change_pct:.1f}% change)", "yellow")

    return rounded_qty


def round_price(price: float) -> float:
    """Round price to 2 decimal places for stocks."""
    return round(price, 2)


def place_bracket_order(
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    order_type: str = 'market'
) -> Optional[str]:
    """
    Place a bracket order with atomic TP and SL.

    Args:
        symbol: Stock symbol
        side: 'buy' or 'sell'
        qty: Number of shares
        entry_price: Entry price (for limit orders)
        tp_price: Take profit price
        sl_price: Stop loss price
        order_type: 'market' or 'limit'

    Returns:
        Order ID if successful, None if failed
    """
    try:
        api = get_api()

        # Round values
        qty = round_to_valid_qty(symbol, qty)

        # CRITICAL: Alpaca bracket orders require WHOLE shares (no fractional)
        # "fractional orders must be simple orders" error if not whole
        if qty != int(qty):
            cprint(f"  [ORDER] Converting fractional qty {qty} to whole shares for bracket order", "yellow")
            qty = int(qty)
            if qty == 0:
                cprint(f"  [ORDER] ERROR: Qty rounds to 0, cannot place bracket order", "red")
                return None

        entry_price = round_price(entry_price)
        tp_price = round_price(tp_price)
        sl_price = round_price(sl_price)

        # Validate bracket order prices
        if side == 'buy':
            # Long position: TP > entry > SL
            if tp_price <= entry_price:
                cprint(f"  [ORDER] Warning: TP price ${tp_price} <= entry ${entry_price}, adjusting", "yellow")
                tp_price = round_price(entry_price * 1.03)
            if sl_price >= entry_price:
                cprint(f"  [ORDER] Warning: SL price ${sl_price} >= entry ${entry_price}, adjusting", "yellow")
                sl_price = round_price(entry_price * 0.97)
        else:
            # Short position: SL > entry > TP
            if tp_price >= entry_price:
                cprint(f"  [ORDER] Warning: TP price ${tp_price} >= entry ${entry_price}, adjusting", "yellow")
                tp_price = round_price(entry_price * 0.97)
            if sl_price <= entry_price:
                cprint(f"  [ORDER] Warning: SL price ${sl_price} <= entry ${entry_price}, adjusting", "yellow")
                sl_price = round_price(entry_price * 1.03)

        cprint(f"  [ORDER] Placing bracket order: {side.upper()} {qty} {symbol}", "cyan")
        cprint(f"  [ORDER]   Entry: ${entry_price:.2f} | TP: ${tp_price:.2f} | SL: ${sl_price:.2f}", "cyan")

        # Place bracket order
        # CRITICAL: Use GTC so TP/SL orders persist across trading days
        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type=order_type,
            time_in_force='gtc',
            order_class='bracket',
            take_profit={'limit_price': tp_price},
            stop_loss={'stop_price': sl_price}
        )

        cprint(f"  [ORDER] Bracket order placed: {order.id}", "green")
        cprint(f"  [ORDER]   Status: {order.status}", "green")

        return order.id

    except APIError as e:
        cprint(f"  [ORDER] API Error placing bracket order: {e}", "red")
        return None
    except Exception as e:
        cprint(f"  [ORDER] Error placing bracket order: {e}", "red")
        return None


def place_simple_order(
    symbol: str,
    side: str,
    qty: float,
    order_type: str = 'market',
    limit_price: float = None
) -> Optional[str]:
    """
    Place a simple order without bracket (for exits).

    Args:
        symbol: Stock symbol
        side: 'buy' or 'sell'
        qty: Number of shares
        order_type: 'market' or 'limit'
        limit_price: Price for limit orders

    Returns:
        Order ID if successful, None if failed
    """
    try:
        api = get_api()

        qty = round_to_valid_qty(symbol, qty)

        order_params = {
            'symbol': symbol,
            'qty': qty,
            'side': side,
            'type': order_type,
            'time_in_force': 'day',
        }

        if order_type == 'limit' and limit_price:
            order_params['limit_price'] = round_price(limit_price)

        order = api.submit_order(**order_params)

        cprint(f"  [ORDER] Order placed: {side.upper()} {qty} {symbol} ({order_type})", "green")
        cprint(f"  [ORDER]   Order ID: {order.id} | Status: {order.status}", "green")

        return order.id

    except APIError as e:
        cprint(f"  [ORDER] API Error: {e}", "red")
        return None
    except Exception as e:
        cprint(f"  [ORDER] Error: {e}", "red")
        return None


def close_position(symbol: str) -> bool:
    """
    Close an existing position.

    Args:
        symbol: Stock symbol

    Returns:
        True if closed successfully
    """
    try:
        api = get_api()
        api.close_position(symbol)
        cprint(f"  [ORDER] Position closed: {symbol}", "green")
        return True
    except APIError as e:
        if 'position does not exist' in str(e).lower():
            cprint(f"  [ORDER] No position to close for {symbol}", "yellow")
            return True
        cprint(f"  [ORDER] Error closing position {symbol}: {e}", "red")
        return False
    except Exception as e:
        cprint(f"  [ORDER] Error closing position {symbol}: {e}", "red")
        return False


# ─────────────────────────────────────────────────────────────
# DAILY LOSS LIMIT TRACKING
# ─────────────────────────────────────────────────────────────

def reset_daily_tracker():
    """Reset daily P&L tracker (call at market open)."""
    global _daily_tracker

    try:
        api = get_api()
        account = api.get_account()
        equity = float(account.equity)

        _daily_tracker = {
            'date': date.today(),
            'starting_equity': equity,
            'realized_pnl': 0,
            'trading_halted': False,
        }

        cprint(f"  [RISK] Daily tracker reset. Starting equity: ${equity:,.2f}", "cyan")
        cprint(f"  [RISK] Daily loss limit: ${equity * DAILY_LOSS_LIMIT_PCT / 100:,.2f} ({DAILY_LOSS_LIMIT_PCT}%)", "cyan")

    except Exception as e:
        cprint(f"  [RISK] Error resetting daily tracker: {e}", "red")


def update_daily_pnl(pnl: float):
    """
    Update daily realized P&L.

    Args:
        pnl: Realized P&L from a closed trade
    """
    global _daily_tracker

    # Check if we need to reset (new day)
    if _daily_tracker['date'] != date.today():
        reset_daily_tracker()

    _daily_tracker['realized_pnl'] += pnl

    # Check if we've hit the loss limit
    if _daily_tracker['starting_equity'] > 0:
        loss_limit = _daily_tracker['starting_equity'] * DAILY_LOSS_LIMIT_PCT / 100

        if _daily_tracker['realized_pnl'] < -loss_limit:
            _daily_tracker['trading_halted'] = True
            cprint(f"  [RISK] DAILY LOSS LIMIT REACHED!", "red", attrs=['bold'])
            cprint(f"  [RISK] Daily P&L: ${_daily_tracker['realized_pnl']:,.2f}", "red")
            cprint(f"  [RISK] Trading halted for the day", "red")


def get_daily_pnl() -> float:
    """Get current daily realized P&L."""
    if _daily_tracker['date'] != date.today():
        reset_daily_tracker()
    return _daily_tracker['realized_pnl']


def check_daily_loss_limit() -> bool:
    """
    Check if daily loss limit has been reached.

    Returns:
        True if trading is allowed, False if halted
    """
    global _daily_tracker

    # Reset if new day
    if _daily_tracker['date'] != date.today():
        reset_daily_tracker()
        return True

    if _daily_tracker['trading_halted']:
        cprint(f"  [RISK] Trading halted - daily loss limit reached (${_daily_tracker['realized_pnl']:,.2f})", "red")
        return False

    return True


def is_trading_halted() -> bool:
    """Check if trading is currently halted."""
    if _daily_tracker['date'] != date.today():
        return False
    return _daily_tracker['trading_halted']


def get_account_info() -> dict:
    """Get current account information."""
    try:
        api = get_api()
        account = api.get_account()

        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value),
            'day_trade_count': int(account.daytrade_count),
            'pattern_day_trader': account.pattern_day_trader,
        }
    except Exception as e:
        cprint(f"  [ORDER] Error getting account info: {e}", "red")
        return {}


def get_all_positions() -> list:
    """Get all current positions."""
    try:
        api = get_api()
        positions = api.list_positions()

        return [
            {
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'side': 'long' if float(pos.qty) > 0 else 'short',
                'market_value': float(pos.market_value),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'unrealized_pnl': float(pos.unrealized_pl),
                'unrealized_pnl_pct': float(pos.unrealized_plpc) * 100,
            }
            for pos in positions
        ]
    except Exception as e:
        cprint(f"  [ORDER] Error getting positions: {e}", "red")
        return []


def get_open_orders(symbol: str = None) -> list:
    """Get all open orders, optionally filtered by symbol."""
    try:
        api = get_api()

        if symbol:
            orders = api.list_orders(status='open', symbols=[symbol])
        else:
            orders = api.list_orders(status='open')

        return [
            {
                'id': order.id,
                'symbol': order.symbol,
                'side': order.side,
                'qty': float(order.qty),
                'type': order.type,
                'status': order.status,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
            }
            for order in orders
        ]
    except Exception as e:
        cprint(f"  [ORDER] Error getting open orders: {e}", "red")
        return []


def place_stop_order(
    symbol: str,
    qty: float,
    stop_price: float,
    side: str = 'sell'
) -> Optional[str]:
    """
    Place a standalone stop order for an existing position.

    Args:
        symbol: Stock symbol
        qty: Number of shares
        stop_price: Stop trigger price
        side: 'sell' for long positions, 'buy' for short positions

    Returns:
        Order ID if successful, None if failed
    """
    try:
        api = get_api()

        qty = round_to_valid_qty(symbol, qty)
        if qty != int(qty):
            qty = int(qty)
            if qty == 0:
                return None

        stop_price = round_price(stop_price)

        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='stop',
            stop_price=stop_price,
            time_in_force='gtc'
        )

        cprint(f"  [ORDER] Stop order placed: {side.upper()} {qty} {symbol} @ ${stop_price:.2f}", "green")
        return order.id

    except Exception as e:
        cprint(f"  [ORDER] Error placing stop order for {symbol}: {e}", "red")
        return None


def place_limit_order(
    symbol: str,
    qty: float,
    limit_price: float,
    side: str = 'sell'
) -> Optional[str]:
    """
    Place a standalone limit order (for TP on existing positions).

    Args:
        symbol: Stock symbol
        qty: Number of shares
        limit_price: Limit price
        side: 'sell' for long positions, 'buy' for short positions

    Returns:
        Order ID if successful, None if failed
    """
    try:
        api = get_api()

        qty = round_to_valid_qty(symbol, qty)
        if qty != int(qty):
            qty = int(qty)
            if qty == 0:
                return None

        limit_price = round_price(limit_price)

        order = api.submit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            type='limit',
            limit_price=limit_price,
            time_in_force='gtc'
        )

        cprint(f"  [ORDER] Limit order placed: {side.upper()} {qty} {symbol} @ ${limit_price:.2f}", "green")
        return order.id

    except Exception as e:
        cprint(f"  [ORDER] Error placing limit order for {symbol}: {e}", "red")
        return None


def ensure_exit_protection(
    positions: list = None,
    sl_pct: float = 3.0,
    tp_pct: float = 6.0
) -> dict:
    """
    Ensure all positions have active exit orders (TP or SL).

    IMPORTANT: Alpaca doesn't allow BOTH TP and SL orders for the same shares.
    This function places ONE exit order per position:
    - If position is profitable (>= 80% of TP target): place TP limit order
    - Otherwise: place SL stop order (downside protection)

    Args:
        positions: List of position dicts from get_all_positions().
                   If None, fetches all positions automatically.
        sl_pct: Stop-loss percentage from entry
        tp_pct: Take-profit percentage from entry

    Returns:
        Dict with 'protected', 'added_sl', 'added_tp', 'failed' counts
    """
    result = {'protected': 0, 'added_sl': 0, 'added_tp': 0, 'failed': 0}

    # Fetch positions if not provided
    if positions is None:
        positions = get_all_positions()

    for pos in positions:
        symbol = pos['symbol']
        qty = abs(pos['qty'])
        entry = pos['avg_entry_price']
        current = pos['current_price']
        is_long = pos['side'] == 'long'

        # Check for ANY existing exit order (stop, limit, trailing_stop)
        open_orders = get_open_orders(symbol)
        exit_side = 'sell' if is_long else 'buy'
        has_exit_order = any(
            o['side'] == exit_side and o['type'] in ('stop', 'limit', 'trailing_stop')
            for o in open_orders
        )

        if has_exit_order:
            result['protected'] += 1
            order_types = [o['type'] for o in open_orders if o['side'] == exit_side]
            cprint(f"  [SAFETY] {symbol}: Exit order active ({', '.join(order_types)})", "green")
            continue

        # Calculate TP and SL prices
        if is_long:
            tp_price = round_price(entry * (1 + tp_pct / 100))
            sl_price = round_price(entry * (1 - sl_pct / 100))
            pnl_pct = (current - entry) / entry * 100
        else:
            tp_price = round_price(entry * (1 - tp_pct / 100))
            sl_price = round_price(entry * (1 + sl_pct / 100))
            pnl_pct = (entry - current) / entry * 100

        # Decide: place TP if profitable (>= 80% of target), otherwise SL
        tp_threshold = tp_pct * 0.8  # 80% of the way to TP
        use_tp = pnl_pct >= tp_threshold

        if use_tp:
            cprint(f"  [SAFETY] {symbol}: Profitable ({pnl_pct:+.1f}%) — placing TP @ ${tp_price:.2f}", "cyan")
            order_id = place_limit_order(symbol, qty, tp_price, exit_side)
            if order_id:
                result['added_tp'] += 1
                cprint(f"  [SAFETY] {symbol}: TP limit order placed (ID: {order_id})", "green")
            else:
                result['failed'] += 1
                cprint(f"  [SAFETY] {symbol}: FAILED to place TP order!", "red")
        else:
            cprint(f"  [SAFETY] {symbol}: P&L {pnl_pct:+.1f}% — placing SL @ ${sl_price:.2f}", "yellow")
            order_id = place_stop_order(symbol, qty, sl_price, exit_side)
            if order_id:
                result['added_sl'] += 1
                cprint(f"  [SAFETY] {symbol}: SL stop order placed (ID: {order_id})", "green")
            else:
                result['failed'] += 1
                cprint(f"  [SAFETY] {symbol}: FAILED to place SL order!", "red")

    return result


# Keep old name as alias for backwards compatibility
def ensure_stop_loss_protection(positions: list = None, sl_pct: float = 3.0) -> dict:
    """Backwards-compatible alias for ensure_exit_protection."""
    result = ensure_exit_protection(positions, sl_pct=sl_pct)
    # Convert to old return format
    return {
        'protected': result['protected'],
        'added': result['added_sl'] + result['added_tp'],
        'failed': result['failed']
    }


# ─────────────────────────────────────────────────────────────
# COMPLETE ORDER FLOW
# ─────────────────────────────────────────────────────────────

def execute_entry(
    symbol: str,
    side: str,
    qty: float,
    entry_price: float,
    tp_price: float,
    sl_price: float,
    strategy: str = 'UNKNOWN',
    dry_run: bool = False
) -> Tuple[bool, Optional[str], str]:
    """
    Execute a complete entry with all safety checks.

    Args:
        symbol: Stock symbol
        side: 'buy' or 'sell'
        qty: Number of shares
        entry_price: Entry price
        tp_price: Take profit price
        sl_price: Stop loss price
        strategy: Strategy name for logging
        dry_run: If True, log but don't execute

    Returns:
        Tuple of (success, order_id, message)
    """
    cprint(f"\n  [ORDER] ═══════════════════════════════════════════", "cyan")
    cprint(f"  [ORDER] ENTRY FLOW: {side.upper()} {qty} {symbol} @ ${entry_price:.2f}", "cyan")
    cprint(f"  [ORDER] Strategy: {strategy} | TP: ${tp_price:.2f} | SL: ${sl_price:.2f}", "cyan")
    cprint(f"  [ORDER] ═══════════════════════════════════════════", "cyan")

    # Step 1: Check daily loss limit
    cprint(f"  [ORDER] Step 1: Checking daily loss limit...", "white")
    if not check_daily_loss_limit():
        msg = "Daily loss limit reached - trading halted"
        cprint(f"  [ORDER] BLOCKED: {msg}", "red")
        return False, None, msg
    cprint(f"  [ORDER]   ✓ Daily P&L OK (${get_daily_pnl():+,.2f})", "green")

    # Step 2: Check existing position
    cprint(f"  [ORDER] Step 2: Checking for existing position...", "white")
    has_position, position_info = check_existing_position(symbol)
    if has_position:
        msg = f"Already in position: {position_info['qty']} shares {position_info['side']}"
        cprint(f"  [ORDER] BLOCKED: {msg}", "yellow")
        return False, None, msg
    cprint(f"  [ORDER]   ✓ No existing position", "green")

    # Step 3: Cancel any open orders for this symbol
    cprint(f"  [ORDER] Step 3: Cancelling open orders...", "white")
    cancelled = cancel_symbol_orders(symbol)
    if cancelled > 0:
        cprint(f"  [ORDER]   Cancelled {cancelled} open orders", "yellow")
    else:
        cprint(f"  [ORDER]   ✓ No open orders to cancel", "green")

    # Step 4: Round quantity to valid increment
    cprint(f"  [ORDER] Step 4: Validating quantity...", "white")
    original_qty = qty
    qty = round_to_valid_qty(symbol, qty)
    cprint(f"  [ORDER]   ✓ Quantity: {original_qty:.4f} -> {qty:.4f}", "green")

    # Step 5: Dry run check
    if dry_run:
        msg = f"DRY RUN - Would place: {side.upper()} {qty} {symbol}"
        cprint(f"  [ORDER] {msg}", "magenta")
        cprint(f"  [ORDER]   Entry: ${entry_price:.2f} | TP: ${tp_price:.2f} | SL: ${sl_price:.2f}", "magenta")
        return True, "DRY_RUN", msg

    # Step 6: Place bracket order
    cprint(f"  [ORDER] Step 5: Placing bracket order...", "white")
    order_id = place_bracket_order(
        symbol=symbol,
        side=side,
        qty=qty,
        entry_price=entry_price,
        tp_price=tp_price,
        sl_price=sl_price,
        order_type='market'
    )

    if order_id:
        msg = f"Bracket order placed: {order_id}"
        cprint(f"  [ORDER] ✓ SUCCESS: {msg}", "green", attrs=['bold'])
        return True, order_id, msg
    else:
        msg = "Failed to place bracket order"
        cprint(f"  [ORDER] ✗ FAILED: {msg}", "red")
        return False, None, msg


def execute_exit(
    symbol: str,
    qty: float,
    reason: str,
    dry_run: bool = False
) -> Tuple[bool, Optional[str], str]:
    """
    Execute a position exit with all safety checks.

    Args:
        symbol: Stock symbol
        qty: Number of shares to close
        reason: Exit reason for logging
        dry_run: If True, log but don't execute

    Returns:
        Tuple of (success, order_id, message)
    """
    cprint(f"\n  [ORDER] ═══════════════════════════════════════════", "yellow")
    cprint(f"  [ORDER] EXIT FLOW: {symbol} ({reason})", "yellow")
    cprint(f"  [ORDER] ═══════════════════════════════════════════", "yellow")

    # Check if position exists
    has_position, position_info = check_existing_position(symbol)

    if not has_position:
        msg = f"No position to close for {symbol}"
        cprint(f"  [ORDER] {msg}", "yellow")
        return True, None, msg

    if dry_run:
        msg = f"DRY RUN - Would close: {symbol} {position_info['qty']} shares"
        cprint(f"  [ORDER] {msg}", "magenta")
        return True, "DRY_RUN", msg

    # Cancel any open orders first
    cancel_symbol_orders(symbol)

    # Close the position
    success = close_position(symbol)

    if success:
        msg = f"Position closed: {symbol}"
        return True, None, msg
    else:
        msg = f"Failed to close position: {symbol}"
        return False, None, msg


if __name__ == "__main__":
    # Test the utilities
    cprint("\n" + "=" * 60, "cyan")
    cprint("  ORDER UTILITIES TEST", "cyan")
    cprint("=" * 60, "cyan")

    # Test account info
    cprint("\nAccount Info:", "yellow")
    info = get_account_info()
    for k, v in info.items():
        cprint(f"  {k}: {v}", "white")

    # Test positions
    cprint("\nCurrent Positions:", "yellow")
    positions = get_all_positions()
    if positions:
        for pos in positions:
            cprint(f"  {pos['symbol']}: {pos['qty']} shares @ ${pos['avg_entry_price']:.2f}", "white")
    else:
        cprint("  No positions", "white")

    # Test open orders
    cprint("\nOpen Orders:", "yellow")
    orders = get_open_orders()
    if orders:
        for order in orders:
            cprint(f"  {order['id']}: {order['side']} {order['qty']} {order['symbol']}", "white")
    else:
        cprint("  No open orders", "white")

    # Test daily tracker
    cprint("\nDaily Tracker:", "yellow")
    reset_daily_tracker()
    cprint(f"  Daily P&L: ${get_daily_pnl():,.2f}", "white")
    cprint(f"  Trading allowed: {check_daily_loss_limit()}", "white")

    cprint("\n" + "=" * 60 + "\n", "cyan")

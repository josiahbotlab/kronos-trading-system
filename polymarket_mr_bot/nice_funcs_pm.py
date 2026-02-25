"""
Polymarket Trading Functions

Execution layer for Polymarket CLOB API.
Based on Moon Dev's nice_funcs pattern.

Requires: pip install py-clob-client web3==6.14.0
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

from termcolor import cprint

# Polymarket imports
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import (
        OrderArgs,
        OrderType,
        MarketOrderArgs,
        BalanceAllowanceParams,
        AssetType,
    )
    from py_clob_client.order_builder.constants import BUY, SELL
    CLOB_AVAILABLE = True
except ImportError as e:
    CLOB_AVAILABLE = False
    ClobClient = None  # Type stub for when library not installed
    OrderArgs = None
    OrderType = None
    MarketOrderArgs = None
    BalanceAllowanceParams = None
    AssetType = None
    BUY = "BUY"
    SELL = "SELL"
    cprint(f"Warning: py-clob-client not installed. Run: pip install py-clob-client web3==6.14.0", "yellow")

from .config import (
    POLYMARKET_HOST,
    CHAIN_ID,
    POLY_PRIVATE_KEY,
    POLY_FUNDER_ADDRESS,
    SIGNATURE_TYPE,
    DRY_RUN,
    MIN_PRICE,
    MAX_PRICE,
    MIN_LIQUIDITY_USDC,
)


# ============================================================================
# CLIENT INITIALIZATION
# ============================================================================

_client = None  # Will be ClobClient instance when initialized


def get_public_client():
    """
    Get a read-only client for public data (no authentication required).
    """
    if not CLOB_AVAILABLE:
        cprint("[ERROR] py-clob-client not available", "red")
        return None

    try:
        # Public client - no auth needed
        client = ClobClient(
            host=POLYMARKET_HOST,
            chain_id=CHAIN_ID,
        )
        return client
    except Exception as e:
        cprint(f"[ERROR] Failed to create public client: {e}", "red")
        return None


def get_client():
    """
    Get or create the Polymarket CLOB client.

    Returns:
        ClobClient instance or None if not configured
    """
    global _client

    if not CLOB_AVAILABLE:
        cprint("[ERROR] py-clob-client not available", "red")
        return None

    if _client is not None:
        return _client

    if not POLY_PRIVATE_KEY:
        cprint("[ERROR] POLY_PRIVATE_KEY not set in environment", "red")
        return None

    try:
        # Initialize client
        _client = ClobClient(
            host=POLYMARKET_HOST,
            key=POLY_PRIVATE_KEY,
            chain_id=CHAIN_ID,
            signature_type=SIGNATURE_TYPE,
            funder=POLY_FUNDER_ADDRESS if POLY_FUNDER_ADDRESS else None,
        )

        # Set API credentials
        _client.set_api_creds(_client.create_or_derive_api_creds())

        cprint("[INIT] Polymarket client initialized", "green")
        return _client

    except Exception as e:
        cprint(f"[ERROR] Failed to initialize client: {e}", "red")
        return None


def check_connection() -> bool:
    """Check if connected to Polymarket."""
    client = get_client()
    if client is None:
        return False

    try:
        # Simple health check
        client.get_ok()
        return True
    except Exception:
        return False


# ============================================================================
# MARKET DATA
# ============================================================================

def get_markets(limit: int = 100) -> List[Dict]:
    """
    Get list of available markets.

    Returns:
        List of market dicts with condition_id, question, outcomes, etc.
    """
    client = get_client()
    if client is None:
        return []

    try:
        markets = client.get_simplified_markets()
        return markets[:limit] if markets else []
    except Exception as e:
        cprint(f"[ERROR] Failed to get markets: {e}", "red")
        return []


def get_market(condition_id: str) -> Optional[Dict]:
    """
    Get details for a specific market.

    Args:
        condition_id: The market condition ID

    Returns:
        Market dict or None
    """
    client = get_client()
    if client is None:
        return None

    try:
        market = client.get_market(condition_id)
        return market
    except Exception as e:
        cprint(f"[ERROR] Failed to get market {condition_id}: {e}", "red")
        return None


def get_order_book(token_id: str) -> Optional[Dict]:
    """
    Get order book for a token.

    Args:
        token_id: The outcome token ID

    Returns:
        Dict with 'bids' and 'asks' arrays
    """
    client = get_client()
    if client is None:
        return None

    try:
        book = client.get_order_book(token_id)
        return book
    except Exception as e:
        cprint(f"[ERROR] Failed to get order book: {e}", "red")
        return None


def get_price(token_id: str, side: str = "BUY") -> Optional[float]:
    """
    Get current price for a token.

    Args:
        token_id: The outcome token ID
        side: 'BUY' or 'SELL'

    Returns:
        Price as float (0.0 to 1.0) or None
    """
    client = get_client()
    if client is None:
        return None

    try:
        price_data = client.get_price(token_id, side)
        if price_data and 'price' in price_data:
            return float(price_data['price'])
        return None
    except Exception as e:
        cprint(f"[ERROR] Failed to get price: {e}", "red")
        return None


def get_midpoint(token_id: str) -> Optional[float]:
    """
    Get midpoint price for a token.

    Args:
        token_id: The outcome token ID

    Returns:
        Midpoint price as float or None
    """
    client = get_client()
    if client is None:
        return None

    try:
        mid_data = client.get_midpoint(token_id)
        if mid_data and 'mid' in mid_data:
            return float(mid_data['mid'])
        return None
    except Exception as e:
        cprint(f"[ERROR] Failed to get midpoint: {e}", "red")
        return None


def get_spread(token_id: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Get bid, ask, and spread for a token.

    Returns:
        (bid, ask, spread) tuple or (None, None, None)
    """
    client = get_client()
    if client is None:
        return None, None, None

    try:
        book = client.get_order_book(token_id)
        if not book:
            return None, None, None

        bids = book.get('bids', [])
        asks = book.get('asks', [])

        best_bid = float(bids[0]['price']) if bids else None
        best_ask = float(asks[0]['price']) if asks else None

        spread = None
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid

        return best_bid, best_ask, spread

    except Exception as e:
        cprint(f"[ERROR] Failed to get spread: {e}", "red")
        return None, None, None


def get_liquidity(token_id: str) -> Tuple[float, float]:
    """
    Get total liquidity (USDC) on both sides of order book.

    Returns:
        (bid_liquidity, ask_liquidity) in USDC
    """
    client = get_client()
    if client is None:
        return 0.0, 0.0

    try:
        book = client.get_order_book(token_id)
        if not book:
            return 0.0, 0.0

        bid_liq = sum(float(b['price']) * float(b['size']) for b in book.get('bids', []))
        ask_liq = sum(float(a['price']) * float(a['size']) for a in book.get('asks', []))

        return bid_liq, ask_liq

    except Exception as e:
        cprint(f"[ERROR] Failed to get liquidity: {e}", "red")
        return 0.0, 0.0


# ============================================================================
# POSITION MANAGEMENT
# ============================================================================

def get_positions() -> List[Dict]:
    """
    Get all open positions.

    Returns:
        List of position dicts
    """
    client = get_client()
    if client is None:
        return []

    try:
        # Get balance/positions
        positions = client.get_positions()
        return positions if positions else []
    except Exception as e:
        cprint(f"[ERROR] Failed to get positions: {e}", "red")
        return []


def get_position(token_id: str) -> Optional[Dict]:
    """
    Get position for a specific token.

    Args:
        token_id: The outcome token ID

    Returns:
        Position dict or None if no position
    """
    positions = get_positions()

    for pos in positions:
        if pos.get('asset') == token_id or pos.get('token_id') == token_id:
            return pos

    return None


def get_position_size(token_id: str) -> float:
    """
    Get position size for a token.

    Returns:
        Position size (positive for long, negative for short, 0 if none)
    """
    pos = get_position(token_id)
    if pos is None:
        return 0.0

    return float(pos.get('size', 0))


def get_position_pnl(token_id: str) -> Tuple[float, float]:
    """
    Get P&L for a position.

    Returns:
        (pnl_usdc, pnl_percent) tuple
    """
    pos = get_position(token_id)
    if pos is None:
        return 0.0, 0.0

    size = float(pos.get('size', 0))
    avg_price = float(pos.get('avg_price', 0))
    current_price = get_midpoint(token_id)

    if current_price is None or size == 0 or avg_price == 0:
        return 0.0, 0.0

    cost_basis = size * avg_price
    current_value = size * current_price
    pnl_usdc = current_value - cost_basis
    pnl_pct = (pnl_usdc / cost_basis) * 100 if cost_basis > 0 else 0.0

    return pnl_usdc, pnl_pct


# ============================================================================
# ORDER EXECUTION
# ============================================================================

def buy(token_id: str, size: float, price: Optional[float] = None) -> Optional[Dict]:
    """
    Buy outcome tokens.

    Args:
        token_id: The outcome token ID
        size: Number of shares to buy
        price: Limit price (None for market order)

    Returns:
        Order response dict or None
    """
    if DRY_RUN:
        cprint(f"[DRY RUN] Would BUY {size:.2f} @ {price or 'MARKET'}", "magenta")
        return {'dry_run': True, 'side': 'BUY', 'size': size, 'price': price}

    client = get_client()
    if client is None:
        return None

    try:
        if price is not None:
            # Limit order
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=BUY,
            )
            signed_order = client.create_order(order_args)
            response = client.post_order(signed_order, OrderType.GTC)
        else:
            # Market order (by dollar amount)
            amount_usdc = size * (get_midpoint(token_id) or 0.5)
            market_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount_usdc,
            )
            response = client.create_market_order(market_args)

        cprint(f"[ORDER] BUY {size:.2f} @ {price or 'MARKET'} - {response}", "green")
        return response

    except Exception as e:
        cprint(f"[ERROR] Buy order failed: {e}", "red")
        return None


def sell(token_id: str, size: float, price: Optional[float] = None) -> Optional[Dict]:
    """
    Sell outcome tokens.

    Args:
        token_id: The outcome token ID
        size: Number of shares to sell
        price: Limit price (None for market order)

    Returns:
        Order response dict or None
    """
    if DRY_RUN:
        cprint(f"[DRY RUN] Would SELL {size:.2f} @ {price or 'MARKET'}", "magenta")
        return {'dry_run': True, 'side': 'SELL', 'size': size, 'price': price}

    client = get_client()
    if client is None:
        return None

    try:
        if price is not None:
            # Limit order
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=SELL,
            )
            signed_order = client.create_order(order_args)
            response = client.post_order(signed_order, OrderType.GTC)
        else:
            # Market order
            market_args = MarketOrderArgs(
                token_id=token_id,
                amount=size,
                side=SELL,
            )
            response = client.create_market_order(market_args)

        cprint(f"[ORDER] SELL {size:.2f} @ {price or 'MARKET'} - {response}", "red")
        return response

    except Exception as e:
        cprint(f"[ERROR] Sell order failed: {e}", "red")
        return None


def close_position(token_id: str) -> Optional[Dict]:
    """
    Close entire position for a token.

    Returns:
        Order response or None
    """
    size = get_position_size(token_id)

    if size == 0:
        cprint(f"[INFO] No position to close for {token_id}", "yellow")
        return None

    if size > 0:
        return sell(token_id, abs(size))
    else:
        return buy(token_id, abs(size))


# ============================================================================
# ORDER MANAGEMENT
# ============================================================================

def get_open_orders() -> List[Dict]:
    """Get all open orders."""
    client = get_client()
    if client is None:
        return []

    try:
        orders = client.get_orders()
        return orders if orders else []
    except Exception as e:
        cprint(f"[ERROR] Failed to get orders: {e}", "red")
        return []


def get_orders_for_token(token_id: str) -> List[Dict]:
    """Get open orders for a specific token."""
    orders = get_open_orders()
    return [o for o in orders if o.get('asset') == token_id or o.get('token_id') == token_id]


def cancel_order(order_id: str) -> bool:
    """
    Cancel a specific order.

    Returns:
        True if cancelled successfully
    """
    if DRY_RUN:
        cprint(f"[DRY RUN] Would cancel order {order_id}", "magenta")
        return True

    client = get_client()
    if client is None:
        return False

    try:
        client.cancel(order_id)
        cprint(f"[ORDER] Cancelled order {order_id}", "yellow")
        return True
    except Exception as e:
        cprint(f"[ERROR] Failed to cancel order: {e}", "red")
        return False


def cancel_all_orders() -> int:
    """
    Cancel all open orders.

    Returns:
        Number of orders cancelled
    """
    if DRY_RUN:
        orders = get_open_orders()
        cprint(f"[DRY RUN] Would cancel {len(orders)} orders", "magenta")
        return len(orders)

    client = get_client()
    if client is None:
        return 0

    try:
        client.cancel_all()
        cprint("[ORDER] Cancelled all orders", "yellow")
        return -1  # Unknown count
    except Exception as e:
        cprint(f"[ERROR] Failed to cancel all orders: {e}", "red")
        return 0


def cancel_orders_for_token(token_id: str) -> int:
    """
    Cancel all orders for a specific token.

    Returns:
        Number of orders cancelled
    """
    orders = get_orders_for_token(token_id)
    cancelled = 0

    for order in orders:
        if cancel_order(order.get('id', '')):
            cancelled += 1

    return cancelled


# ============================================================================
# BALANCE & ALLOWANCES
# ============================================================================

def get_balance() -> float:
    """
    Get USDC balance.

    Returns:
        Balance in USDC
    """
    client = get_client()
    if client is None:
        return 0.0

    try:
        # Get balance allowances
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        balance_data = client.get_balance_allowance(params)
        if balance_data and 'balance' in balance_data:
            return float(balance_data['balance'])
        return 0.0
    except Exception as e:
        cprint(f"[ERROR] Failed to get balance: {e}", "red")
        return 0.0


def check_allowances() -> Dict[str, bool]:
    """
    Check token allowances for trading.

    Returns:
        Dict with allowance status
    """
    client = get_client()
    if client is None:
        return {'usdc': False, 'conditional': False}

    try:
        # Check USDC allowance
        usdc_params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        usdc_data = client.get_balance_allowance(usdc_params)
        usdc_ok = float(usdc_data.get('allowance', 0)) > 0 if usdc_data else False

        # Check conditional token allowance
        cond_params = BalanceAllowanceParams(asset_type=AssetType.CONDITIONAL)
        cond_data = client.get_balance_allowance(cond_params)
        cond_ok = float(cond_data.get('allowance', 0)) > 0 if cond_data else False

        return {'usdc': usdc_ok, 'conditional': cond_ok}
    except Exception as e:
        cprint(f"[ERROR] Failed to check allowances: {e}", "red")
        return {'usdc': False, 'conditional': False}


# ============================================================================
# P&L MANAGEMENT
# ============================================================================

def check_tp_sl(
    token_id: str,
    take_profit_pct: float,
    stop_loss_pct: float
) -> Tuple[bool, str]:
    """
    Check if take profit or stop loss is hit.

    Args:
        token_id: Token to check
        take_profit_pct: TP threshold (e.g., 20.0 for +20%)
        stop_loss_pct: SL threshold (e.g., -15.0 for -15%)

    Returns:
        (should_close, reason) tuple
    """
    pnl_usdc, pnl_pct = get_position_pnl(token_id)

    if pnl_pct >= take_profit_pct:
        return True, f"TP hit: {pnl_pct:.2f}% >= {take_profit_pct}%"

    if pnl_pct <= stop_loss_pct:
        return True, f"SL hit: {pnl_pct:.2f}% <= {stop_loss_pct}%"

    return False, ""


def pnl_close(
    token_id: str,
    take_profit_pct: float,
    stop_loss_pct: float
) -> bool:
    """
    Close position if TP or SL is hit.

    Returns:
        True if position was closed
    """
    should_close, reason = check_tp_sl(token_id, take_profit_pct, stop_loss_pct)

    if should_close:
        cprint(f"[P&L] {reason} - Closing position", "yellow")
        close_position(token_id)
        return True

    return False


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def is_valid_price(price: float) -> bool:
    """Check if price is within valid trading range."""
    return MIN_PRICE <= price <= MAX_PRICE


def has_sufficient_liquidity(token_id: str) -> bool:
    """Check if market has sufficient liquidity."""
    bid_liq, ask_liq = get_liquidity(token_id)
    total_liq = bid_liq + ask_liq
    return total_liq >= MIN_LIQUIDITY_USDC


def format_price(price: float) -> str:
    """Format price for display (as cents)."""
    return f"{price * 100:.1f}c"


def calculate_shares(usdc_amount: float, price: float) -> float:
    """Calculate number of shares for a given USDC amount and price."""
    if price <= 0:
        return 0.0
    return usdc_amount / price


def calculate_cost(shares: float, price: float) -> float:
    """Calculate USDC cost for shares at price."""
    return shares * price


# ============================================================================
# STATUS & DEBUGGING
# ============================================================================

def print_status():
    """Print current account status."""
    balance = get_balance()
    positions = get_positions()
    orders = get_open_orders()

    cprint("\n" + "=" * 50, "cyan")
    cprint("  POLYMARKET ACCOUNT STATUS", "cyan", attrs=['bold'])
    cprint("=" * 50, "cyan")

    cprint(f"\n  USDC Balance: ${balance:,.2f}", "white")
    cprint(f"  Open Positions: {len(positions)}", "white")
    cprint(f"  Open Orders: {len(orders)}", "white")

    if positions:
        cprint("\n  POSITIONS:", "cyan")
        for pos in positions:
            token_id = pos.get('asset', pos.get('token_id', 'unknown'))
            size = float(pos.get('size', 0))
            avg_px = float(pos.get('avg_price', 0))
            cprint(f"    {token_id[:20]}... | {size:.2f} @ {format_price(avg_px)}", "white")

    allowances = check_allowances()
    cprint(f"\n  Allowances: USDC={allowances['usdc']}, Conditional={allowances['conditional']}", "white")
    cprint("=" * 50 + "\n", "cyan")


# ============================================================================
# TESTING
# ============================================================================

def test_connection():
    """Test Polymarket connection."""
    cprint("\n" + "=" * 50, "cyan")
    cprint("  POLYMARKET CONNECTION TEST", "cyan", attrs=['bold'])
    cprint("=" * 50, "cyan")

    # Check client initialization
    client = get_client()
    if client is None:
        cprint("\n  [FAIL] Could not initialize client", "red")
        cprint("  Check POLY_PRIVATE_KEY in environment", "yellow")
        return False

    cprint("\n  [OK] Client initialized", "green")

    # Check connection
    if check_connection():
        cprint("  [OK] Connected to Polymarket", "green")
    else:
        cprint("  [FAIL] Connection failed", "red")
        return False

    # Get balance
    balance = get_balance()
    cprint(f"  [OK] Balance: ${balance:,.2f} USDC", "green")

    # Get markets sample
    markets = get_markets(limit=5)
    cprint(f"  [OK] Fetched {len(markets)} markets", "green")

    if markets:
        cprint("\n  Sample Markets:", "cyan")
        for m in markets[:3]:
            question = m.get('question', 'Unknown')[:50]
            cprint(f"    - {question}...", "white")

    cprint("\n" + "=" * 50 + "\n", "cyan")
    return True


def test_public_api():
    """
    Test Polymarket public API (no authentication required).
    Useful for verifying the bot can fetch market data.
    """
    cprint("\n" + "=" * 60, "cyan")
    cprint("  POLYMARKET PUBLIC API TEST", "cyan", attrs=['bold'])
    cprint("  (No authentication required)", "cyan")
    cprint("=" * 60, "cyan")

    if not CLOB_AVAILABLE:
        cprint("\n  [FAIL] py-clob-client not installed", "red")
        return False

    # Get public client
    client = get_public_client()
    if client is None:
        cprint("\n  [FAIL] Could not create public client", "red")
        return False

    cprint("\n  [OK] Public client created", "green")

    # Test server health
    try:
        client.get_ok()
        cprint("  [OK] Server is healthy", "green")
    except Exception as e:
        cprint(f"  [WARN] Health check: {e}", "yellow")

    # Get markets
    try:
        markets_response = client.get_simplified_markets()
        # API returns {'data': [...], 'next_cursor': ...}
        markets = markets_response.get('data', []) if isinstance(markets_response, dict) else []
        if markets:
            cprint(f"  [OK] Fetched {len(markets)} markets", "green")

            # Show sample markets
            cprint("\n  SAMPLE MARKETS:", "cyan")
            cprint("  " + "-" * 56, "cyan")

            # Filter for active markets accepting orders
            active_markets = [m for m in markets if m.get('active') and m.get('accepting_orders')]
            sample_markets = active_markets[:5] if len(active_markets) >= 5 else active_markets[:5] if active_markets else markets[:5]

            for i, market in enumerate(sample_markets):
                condition_id = market.get('condition_id', '')[:20]
                tokens = market.get('tokens', [])
                is_active = market.get('accepting_orders', False)
                status = "ACTIVE" if is_active else "CLOSED"

                cprint(f"\n  {i+1}. Market {condition_id}... [{status}]", "white")

                # Show token outcomes and prices
                for token in tokens[:2]:
                    outcome = token.get('outcome', 'Unknown')
                    price = token.get('price', 0)
                    if price is not None:
                        price_pct = float(price) * 100
                        cprint(f"     {outcome}: {price_pct:.1f}c", "green" if price_pct > 50 else "red")
                    else:
                        cprint(f"     {outcome}: (no price)", "yellow")

        else:
            cprint("  [WARN] No markets returned", "yellow")

    except Exception as e:
        cprint(f"  [FAIL] Failed to fetch markets: {e}", "red")
        return False

    # Get order book for first market
    try:
        if markets and len(markets) > 0 and markets[0].get('tokens'):
            token_id = markets[0]['tokens'][0].get('token_id')
            if token_id:
                book = client.get_order_book(token_id)
                if book:
                    bids = book.get('bids', [])
                    asks = book.get('asks', [])
                    cprint(f"\n  [OK] Order book: {len(bids)} bids, {len(asks)} asks", "green")
    except Exception as e:
        cprint(f"  [WARN] Order book test: {e}", "yellow")

    cprint("\n" + "=" * 60, "cyan")
    cprint("  PUBLIC API TEST COMPLETE", "cyan", attrs=['bold'])
    cprint("=" * 60 + "\n", "cyan")

    return True


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--public':
        test_public_api()
    else:
        test_connection()

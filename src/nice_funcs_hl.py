"""
HyperLiquid Trading Functions
Based on Moon Dev's nice_funcs for HyperLiquid
"""

import eth_account
from eth_account.signers.local import LocalAccount
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
import pandas as pd
import time
import requests

def get_account(secret_key):
    """Initialize account from private key"""
    return eth_account.Account.from_key(secret_key)

def ask_bid(symbol):
    """Get current ask/bid prices for a symbol"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    mids = info.all_mids()
    
    if symbol in mids:
        mid = float(mids[symbol])
        return mid, mid * 0.999, mid * 1.001
    return None, None, None

def get_position(symbol, account):
    """Get current position for a symbol"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    user_state = info.user_state(account.address)
    
    positions = []
    im_in_pos = False
    pos_size = 0
    pos_sym = None
    entry_px = 0
    pnl_perc = 0
    long = None
    
    for position in user_state.get('assetPositions', []):
        pos = position['position']
        if pos['coin'] == symbol:
            pos_size = float(pos['szi'])
            entry_px = float(pos['entryPx'])
            unrealized_pnl = float(pos['unrealizedPnl'])
            
            if pos_size != 0:
                im_in_pos = True
                pos_sym = symbol
                long = pos_size > 0
                position_value = abs(pos_size) * entry_px
                pnl_perc = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
                
            positions.append({
                'symbol': symbol,
                'size': pos_size,
                'entry_px': entry_px,
                'pnl_perc': pnl_perc,
                'long': long
            })
    
    return positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long

def get_sz_px_decimals(symbol):
    """Get size and price decimals for a symbol"""
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    meta = info.meta()
    
    for asset in meta['universe']:
        if asset['name'] == symbol:
            sz_decimals = asset['szDecimals']
            px_decimals = max(0, 5 - sz_decimals)
            return sz_decimals, px_decimals
    
    return 2, 2

def adjust_leverage_usd_size(symbol, usd_size, leverage, account):
    """Set leverage and calculate position size"""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    exchange.update_leverage(leverage, symbol)
    
    price, _, _ = ask_bid(symbol)
    if price is None:
        return leverage, 0
    
    sz_decimals, _ = get_sz_px_decimals(symbol)
    size = round((usd_size * leverage) / price, sz_decimals)
    
    return leverage, size

def market_buy(symbol, size, account):
    """Place market buy order"""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    order = exchange.market_open(symbol, True, size, None)
    return order

def market_sell(symbol, size, account):
    """Place market sell order"""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    order = exchange.market_open(symbol, False, size, None)
    return order

def limit_order(coin, is_buy, sz, limit_px, reduce_only, account):
    """Place limit order"""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    order = exchange.order(coin, is_buy, sz, limit_px, {"limit": {"tif": "Gtc"}}, reduce_only=reduce_only)
    return order

def cancel_symbol_orders(account, symbol):
    """Cancel all open orders for a symbol"""
    exchange = Exchange(account, constants.MAINNET_API_URL)
    info = Info(constants.MAINNET_API_URL, skip_ws=True)
    
    open_orders = info.open_orders(account.address)
    for order in open_orders:
        if order['coin'] == symbol:
            exchange.cancel(symbol, order['oid'])

def pnl_close(symbol, tp, sl, account):
    """Close position if TP or SL hit"""
    positions, im_in_pos, pos_size, pos_sym, entry_px, pnl_perc, long = get_position(symbol, account)
    
    if not im_in_pos:
        return False
    
    if pnl_perc >= tp:
        print(f"TP hit for {symbol}: {pnl_perc:.2f}% >= {tp}%")
        if long:
            market_sell(symbol, abs(pos_size), account)
        else:
            market_buy(symbol, abs(pos_size), account)
        return True
    
    if pnl_perc <= sl:
        print(f"SL hit for {symbol}: {pnl_perc:.2f}% <= {sl}%")
        if long:
            market_sell(symbol, abs(pos_size), account)
        else:
            market_buy(symbol, abs(pos_size), account)
        return True
    
    return False

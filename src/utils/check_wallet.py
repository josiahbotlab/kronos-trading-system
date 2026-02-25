#!/usr/bin/env python3
"""
HyperLiquid Wallet Check Utility

Connects to HyperLiquid and displays:
- Account balance
- Open positions
- Recent trades
- Open orders

Usage:
    python src/utils/check_wallet.py
"""

import sys
sys.path.append('/Users/josiahgarcia/trading-bot')

from termcolor import cprint
from hyperliquid.info import Info
from hyperliquid.utils import constants

from config import SECRET_KEY
from src.nice_funcs_hl import get_account


def format_usd(value):
    """Format value as USD."""
    return f"${value:,.2f}"


def check_wallet():
    """Check wallet connection and display account info."""

    # Check if key is configured
    if not SECRET_KEY:
        cprint("ERROR: HL_SECRET_KEY not found in .env file", "red")
        cprint("\nTo set up your wallet:", "yellow")
        cprint("1. Export your private key from Coinbase Wallet or MetaMask", "white")
        cprint("2. Create a .env file: cp .env.example .env", "white")
        cprint("3. Add your key: HL_SECRET_KEY=your_private_key_here", "white")
        cprint("\nSee instructions below for exporting your key.", "yellow")
        return False

    try:
        # Initialize account
        cprint("Connecting to HyperLiquid...", "cyan")
        account = get_account(SECRET_KEY)
        info = Info(constants.MAINNET_API_URL, skip_ws=True)

        cprint(f"\nWallet Address: {account.address}", "green")

        # Get user state (balances and positions)
        user_state = info.user_state(account.address)

        # Display account balance
        cprint("\n=== ACCOUNT BALANCE ===", "cyan")

        margin_summary = user_state.get('marginSummary', {})
        account_value = float(margin_summary.get('accountValue', 0))
        total_margin = float(margin_summary.get('totalMarginUsed', 0))
        available = account_value - total_margin

        cprint(f"Account Value:    {format_usd(account_value)}", "white")
        cprint(f"Margin Used:      {format_usd(total_margin)}", "white")
        cprint(f"Available:        {format_usd(available)}", "green")

        # Cross margin details
        cross_summary = user_state.get('crossMarginSummary', {})
        if cross_summary:
            equity = float(cross_summary.get('accountValue', 0))
            cprint(f"Cross Equity:     {format_usd(equity)}", "white")

        # Display positions
        positions = user_state.get('assetPositions', [])
        active_positions = [p for p in positions if float(p['position']['szi']) != 0]

        cprint(f"\n=== OPEN POSITIONS ({len(active_positions)}) ===", "cyan")

        if active_positions:
            for pos_data in active_positions:
                pos = pos_data['position']
                symbol = pos['coin']
                size = float(pos['szi'])
                entry_px = float(pos['entryPx'])
                unrealized_pnl = float(pos['unrealizedPnl'])
                leverage = pos_data.get('leverage', {})
                lev_value = leverage.get('value', 'N/A')

                direction = "LONG" if size > 0 else "SHORT"
                position_value = abs(size) * entry_px
                pnl_pct = (unrealized_pnl / position_value * 100) if position_value > 0 else 0

                color = "green" if unrealized_pnl >= 0 else "red"
                cprint(f"\n{symbol} {direction}", "magenta")
                cprint(f"  Size:       {abs(size)} ({format_usd(position_value)})", "white")
                cprint(f"  Entry:      {format_usd(entry_px)}", "white")
                cprint(f"  Leverage:   {lev_value}x", "white")
                cprint(f"  PnL:        {format_usd(unrealized_pnl)} ({pnl_pct:+.2f}%)", color)
        else:
            cprint("No open positions", "white")

        # Get open orders
        open_orders = info.open_orders(account.address)

        cprint(f"\n=== OPEN ORDERS ({len(open_orders)}) ===", "cyan")

        if open_orders:
            for order in open_orders:
                symbol = order['coin']
                side = "BUY" if order['side'] == 'B' else "SELL"
                size = order['sz']
                price = order['limitPx']
                cprint(f"  {symbol}: {side} {size} @ ${price}", "white")
        else:
            cprint("No open orders", "white")

        # Get recent fills (last 10)
        fills = info.user_fills(account.address)
        recent_fills = fills[:5] if fills else []

        cprint(f"\n=== RECENT TRADES ({len(recent_fills)}) ===", "cyan")

        if recent_fills:
            for fill in recent_fills:
                symbol = fill['coin']
                side = fill['side']
                size = fill['sz']
                price = fill['px']
                time_str = fill.get('time', 'N/A')
                cprint(f"  {symbol}: {side} {size} @ ${price}", "white")
        else:
            cprint("No recent trades", "white")

        cprint("\n=== CONNECTION SUCCESSFUL ===", "green")
        return True

    except Exception as e:
        cprint(f"\nERROR: {e}", "red")
        cprint("\nTroubleshooting:", "yellow")
        cprint("1. Check your private key is correct (should start with 0x)", "white")
        cprint("2. Ensure you have funds deposited on HyperLiquid", "white")
        cprint("3. Visit https://app.hyperliquid.xyz to deposit funds", "white")
        return False


def print_wallet_setup_guide():
    """Print guide for setting up wallet."""
    cprint("\n" + "="*60, "cyan")
    cprint("WALLET SETUP GUIDE", "cyan")
    cprint("="*60, "cyan")

    cprint("\n[OPTION 1] Export from Coinbase Wallet:", "yellow")
    cprint("1. Open Coinbase Wallet app", "white")
    cprint("2. Go to Settings > Security", "white")
    cprint("3. Tap 'Show recovery phrase' or 'Export private key'", "white")
    cprint("4. Copy the private key (starts with 0x)", "white")

    cprint("\n[OPTION 2] Export from MetaMask:", "yellow")
    cprint("1. Open MetaMask", "white")
    cprint("2. Click the three dots > Account details", "white")
    cprint("3. Click 'Export Private Key'", "white")
    cprint("4. Enter password and copy the key", "white")

    cprint("\n[OPTION 3] Create new wallet for trading:", "yellow")
    cprint("1. Go to https://app.hyperliquid.xyz", "white")
    cprint("2. Connect with a fresh MetaMask wallet", "white")
    cprint("3. Export the private key from MetaMask", "white")
    cprint("4. Transfer only trading funds to this wallet", "white")

    cprint("\n[SECURITY TIPS]:", "red")
    cprint("- NEVER share your private key", "white")
    cprint("- Use a dedicated trading wallet with limited funds", "white")
    cprint("- Keep .env in .gitignore (already configured)", "white")
    cprint("- Consider using a hardware wallet for main funds", "white")

    cprint("\n[SETUP STEPS]:", "yellow")
    cprint("1. Copy .env.example to .env:", "white")
    cprint("   cp .env.example .env", "cyan")
    cprint("\n2. Edit .env and add your private key:", "white")
    cprint("   HL_SECRET_KEY=0x1234...your_key_here", "cyan")
    cprint("\n3. Run this script again to verify:", "white")
    cprint("   python src/utils/check_wallet.py", "cyan")

    cprint("\n" + "="*60, "cyan")


if __name__ == "__main__":
    success = check_wallet()

    if not success:
        print_wallet_setup_guide()

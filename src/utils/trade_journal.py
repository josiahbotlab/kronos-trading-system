"""
Trade Journal

Automatic logging of all trades with detailed metadata for analysis.

Usage:
    from src.utils.trade_journal import log_trade, view_journal

    # Log an entry
    log_trade(
        action='ENTRY',
        symbol='TSLA',
        strategy='BREAKOUT',
        direction='LONG',
        price=420.50,
        shares=10,
        confidence=85,
        regime='RANGING',
        reasoning='24h high breakout with volume confirmation'
    )

    # Log an exit
    log_trade(
        action='EXIT',
        symbol='TSLA',
        strategy='BREAKOUT',
        direction='LONG',
        price=432.00,
        shares=10,
        entry_price=420.50,
        pnl=115.00,
        pnl_pct=2.73,
        reasoning='Take profit hit'
    )

    # View recent trades
    view_journal(limit=20)
"""

import csv
import os
from datetime import datetime
from pathlib import Path

from termcolor import cprint

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CSV_DIR = PROJECT_ROOT / 'csvs'
JOURNAL_PATH = CSV_DIR / 'trade_journal.csv'

# CSV columns
COLUMNS = [
    'timestamp',
    'bot_name',             # Which bot placed the trade (momentum_bot, breakout_bot, etc.)
    'action',               # ENTRY or EXIT
    'symbol',
    'strategy',             # BREAKOUT, MEAN_REV, GAP_GO, MACD, BB_BOUNCE, MOMENTUM
    'direction',            # LONG or SHORT
    'price',                # Entry or exit price
    'shares',
    'entry_price',          # For exits: original entry price
    'exit_price',           # For exits: exit price
    'stop_loss',
    'take_profit',
    'pnl',                  # P&L in dollars (exits only)
    'pnl_pct',              # P&L percentage (exits only)
    'confidence',           # Entry confidence score
    'regime',               # Market regime at time of trade
    'reasoning',            # Why the trade was taken/exited
    # Enhanced columns (v2)
    'entry_signal_strength', # How many conditions met, e.g. "3/4"
    'time_in_trade',        # Duration held, e.g. "2h 15m" (exits only)
    'slippage',             # Expected vs actual fill difference (exits only)
    'r_multiple',           # actual return / risk taken (exits only)
]


def ensure_csv_exists():
    """Ensure the CSV file exists with correct headers. Migrates old CSVs."""
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    if not JOURNAL_PATH.exists():
        with open(JOURNAL_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
        return

    # Check if existing CSV needs new columns
    with open(JOURNAL_PATH, 'r') as f:
        reader = csv.reader(f)
        try:
            existing_headers = next(reader)
        except StopIteration:
            existing_headers = []

    missing = [c for c in COLUMNS if c not in existing_headers]
    if missing:
        # Re-read all rows, rewrite with new columns (blanks for old rows)
        rows = []
        with open(JOURNAL_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        with open(JOURNAL_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def log_trade(
    action,
    symbol,
    strategy,
    direction,
    price,
    shares=None,
    entry_price=None,
    exit_price=None,
    stop_loss=None,
    take_profit=None,
    pnl=None,
    pnl_pct=None,
    confidence=None,
    regime=None,
    reasoning=None,
    entry_signal_strength=None,
    time_in_trade=None,
    slippage=None,
    r_multiple=None,
    bot_name=None,
):
    """
    Log a trade entry or exit to the journal.

    Args:
        action: 'ENTRY' or 'EXIT'
        symbol: Stock symbol (e.g., 'TSLA')
        strategy: Strategy used ('BREAKOUT', 'MEAN_REV', 'GAP_GO')
        direction: 'LONG' or 'SHORT'
        price: Current price (entry price for entries, exit price for exits)
        shares: Number of shares
        entry_price: Original entry price (for exits)
        exit_price: Exit price (for exits)
        stop_loss: Stop loss price
        take_profit: Take profit price
        pnl: Profit/loss in dollars (for exits)
        pnl_pct: Profit/loss percentage (for exits)
        confidence: Entry confidence score (0-100)
        regime: Market regime ('RANGING', 'TREND_UP', etc.)
        reasoning: Human-readable explanation
        entry_signal_strength: Conditions met, e.g. "3/4"
        time_in_trade: Duration held, e.g. "2h 15m" (exits only)
        slippage: Expected vs actual fill price difference (exits only)
        r_multiple: Actual return / risk taken (exits only)
        bot_name: Which bot placed the trade (e.g. 'momentum_bot')

    Returns:
        dict: The logged trade record
    """
    ensure_csv_exists()

    # Build record
    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'bot_name': bot_name,
        'action': action,
        'symbol': symbol,
        'strategy': strategy,
        'direction': direction,
        'price': round(price, 2) if price else None,
        'shares': round(shares, 4) if shares else None,
        'entry_price': round(entry_price, 2) if entry_price else None,
        'exit_price': round(exit_price, 2) if exit_price else None,
        'stop_loss': round(stop_loss, 2) if stop_loss else None,
        'take_profit': round(take_profit, 2) if take_profit else None,
        'pnl': round(pnl, 2) if pnl else None,
        'pnl_pct': round(pnl_pct, 2) if pnl_pct else None,
        'confidence': confidence,
        'regime': regime,
        'reasoning': reasoning,
        'entry_signal_strength': entry_signal_strength,
        'time_in_trade': time_in_trade,
        'slippage': round(slippage, 4) if slippage is not None else None,
        'r_multiple': round(r_multiple, 2) if r_multiple is not None else None,
    }

    # Write to CSV
    with open(JOURNAL_PATH, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writerow(record)

    # Print confirmation
    if action == 'ENTRY':
        cprint(f"  [JOURNAL] Logged {direction} entry: {symbol} @ ${price:.2f}", "blue")
    else:
        color = "green" if pnl and pnl > 0 else "red"
        pnl_str = f"${pnl:+.2f}" if pnl else "N/A"
        cprint(f"  [JOURNAL] Logged exit: {symbol} | P&L: {pnl_str}", color)

    return record


def view_journal(limit=20, symbol=None, strategy=None):
    """
    View recent trades from the journal.

    Args:
        limit: Number of recent trades to show
        symbol: Filter by symbol (optional)
        strategy: Filter by strategy (optional)
    """
    if not JOURNAL_PATH.exists():
        cprint("No trade journal found.", "yellow")
        return []

    # Read all trades
    trades = []
    with open(JOURNAL_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Apply filters
            if symbol and row['symbol'] != symbol:
                continue
            if strategy and row['strategy'] != strategy:
                continue
            trades.append(row)

    if not trades:
        cprint("No trades found matching criteria.", "yellow")
        return []

    # Get recent trades
    recent = trades[-limit:]

    # Print header
    cprint("\n" + "=" * 100, "cyan")
    cprint("  TRADE JOURNAL", "cyan", attrs=['bold'])
    cprint("=" * 100, "cyan")

    # Calculate summary stats
    entries = [t for t in trades if t['action'] == 'ENTRY']
    exits = [t for t in trades if t['action'] == 'EXIT']

    total_pnl = sum(float(t['pnl']) for t in exits if t['pnl'])
    winners = len([t for t in exits if t['pnl'] and float(t['pnl']) > 0])
    losers = len([t for t in exits if t['pnl'] and float(t['pnl']) <= 0])
    win_rate = (winners / len(exits) * 100) if exits else 0

    cprint(f"\n  Total Entries: {len(entries)} | Total Exits: {len(exits)}", "white")
    cprint(f"  Total P&L: ${total_pnl:+,.2f} | Win Rate: {win_rate:.1f}% ({winners}W / {losers}L)",
           "green" if total_pnl > 0 else "red")

    # Strategy breakdown
    strategies = {}
    for t in exits:
        strat = t['strategy']
        if strat not in strategies:
            strategies[strat] = {'pnl': 0, 'count': 0, 'wins': 0}
        if t['pnl']:
            strategies[strat]['pnl'] += float(t['pnl'])
            strategies[strat]['count'] += 1
            if float(t['pnl']) > 0:
                strategies[strat]['wins'] += 1

    if strategies:
        cprint("\n  BY STRATEGY:", "yellow")
        for strat, stats in strategies.items():
            wr = (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0
            color = "green" if stats['pnl'] > 0 else "red"
            cprint(f"    {strat:<12} P&L: ${stats['pnl']:>+10,.2f} | Trades: {stats['count']:>3} | Win Rate: {wr:>5.1f}%", color)

    # Print recent trades table
    cprint("\n  RECENT TRADES:", "white")
    cprint("  " + "-" * 96, "white")
    cprint(f"  {'Time':<20} {'Action':<6} {'Symbol':<6} {'Strategy':<10} {'Dir':<5} {'Price':>10} {'P&L':>12} {'Regime':<10}", "white")
    cprint("  " + "-" * 96, "white")

    for t in recent:
        action = t['action']
        symbol_str = t['symbol']
        strategy_str = t['strategy'] or ''
        direction = t['direction'] or ''
        price = float(t['price']) if t['price'] else 0
        pnl = float(t['pnl']) if t['pnl'] else None
        regime = t['regime'] or ''
        timestamp = t['timestamp'][5:16]  # MM-DD HH:MM

        if action == 'ENTRY':
            color = "cyan"
            pnl_str = ""
        else:
            color = "green" if pnl and pnl > 0 else "red"
            pnl_str = f"${pnl:>+.2f}" if pnl else ""

        cprint(f"  {timestamp:<20} {action:<6} {symbol_str:<6} {strategy_str:<10} {direction:<5} ${price:>9.2f} {pnl_str:>12} {regime:<10}", color)

    cprint("  " + "-" * 96, "white")
    cprint("=" * 100 + "\n", "cyan")

    return recent


def get_trade_stats(days=30):
    """
    Get trade statistics for the last N days.

    Returns:
        dict: Statistics including win rate, P&L, best/worst trades
    """
    if not JOURNAL_PATH.exists():
        return None

    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=days)

    trades = []
    with open(JOURNAL_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trade_time = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            if trade_time >= cutoff:
                trades.append(row)

    exits = [t for t in trades if t['action'] == 'EXIT' and t['pnl']]

    if not exits:
        return None

    pnls = [float(t['pnl']) for t in exits]

    return {
        'total_trades': len(exits),
        'total_pnl': sum(pnls),
        'avg_pnl': sum(pnls) / len(pnls),
        'best_trade': max(pnls),
        'worst_trade': min(pnls),
        'win_rate': len([p for p in pnls if p > 0]) / len(pnls) * 100,
        'winners': len([p for p in pnls if p > 0]),
        'losers': len([p for p in pnls if p <= 0]),
    }


def reconcile_exits(hours=24, stock_journal=None):
    """
    Match Alpaca filled exit orders to journal ENTRY records and log EXIT rows.

    Queries Alpaca for filled orders in the last N hours, finds ENTRY records
    in the journal that have no corresponding EXIT, and creates EXIT records
    with realized P&L.

    Args:
        hours: How far back to look for filled orders (default 24h)
        stock_journal: Optional StockJournal instance for SQLite logging

    Returns:
        int: Number of exits reconciled
    """
    try:
        from src.utils.order_utils import get_api
    except ImportError:
        from order_utils import get_api

    from datetime import timedelta, timezone

    ensure_csv_exists()
    api = get_api()

    # 1. Get filled orders from Alpaca
    after = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    try:
        closed_orders = api.list_orders(status='closed', after=after, limit=200)
    except Exception as e:
        cprint(f"  [RECONCILE] Error fetching orders: {e}", "red")
        return 0

    filled_orders = [
        o for o in closed_orders
        if o.status == 'filled' and o.filled_avg_price is not None
    ]

    if not filled_orders:
        cprint("  [RECONCILE] No filled orders in last {hours}h", "yellow")
        return 0

    # 2. Read journal entries and exits
    entries = []
    existing_exits = set()
    reconciled_order_ids = set()

    with open(JOURNAL_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['action'] == 'ENTRY':
                entries.append(row)
            elif row['action'] == 'EXIT':
                # Track exits by (symbol, direction, entry_price) to avoid dupes
                key = (row['symbol'], row['direction'], row.get('entry_price', ''))
                existing_exits.add(key)
                # Extract Alpaca order ID from reasoning if present
                reasoning = row.get('reasoning', '') or ''
                if '[alpaca_order:' in reasoning:
                    oid = reasoning.split('[alpaca_order:')[1].split(']')[0]
                    reconciled_order_ids.add(oid)

    # 3. Match filled orders to unmatched entries
    reconciled = 0

    for order in filled_orders:
        symbol = order.symbol
        fill_price = float(order.filled_avg_price)
        fill_qty = float(order.filled_qty)
        fill_side = order.side  # 'buy' or 'sell'
        fill_time = order.filled_at

        # Exit orders: sell closes a LONG, buy closes a SHORT
        if fill_side == 'sell':
            exit_direction = 'LONG'
        else:
            exit_direction = 'SHORT'

        # Skip bracket parent orders — these are ENTRY orders, not exits.
        # Bracket parent: order_class='bracket' (the opening buy/sell).
        # Bracket children (TP/SL fills): order_class='' (empty).
        # Standalone orders (our manual TP/SL): order_class='' (empty).
        order_class = getattr(order, 'order_class', '') or ''
        if order_class == 'bracket':
            continue

        # Skip if this order was already reconciled in a previous run
        order_id = order.id
        if order_id in reconciled_order_ids:
            continue

        # Find matching unmatched ENTRY
        matched_entry = None
        for e in entries:
            if e['symbol'] != symbol:
                continue
            if e['direction'] != exit_direction:
                continue

            entry_price_str = e.get('price', '')
            if not entry_price_str:
                continue

            dupe_key = (symbol, exit_direction, entry_price_str)
            if dupe_key in existing_exits:
                continue

            matched_entry = e
            break  # Match oldest unmatched entry first

        if matched_entry is None:
            continue

        # 4. Calculate P&L
        entry_price = float(matched_entry['price'])
        shares = fill_qty
        entry_shares = float(matched_entry.get('shares', 0) or 0)

        # Use the smaller of fill qty and entry qty
        if entry_shares > 0:
            shares = min(fill_qty, entry_shares)

        if exit_direction == 'LONG':
            pnl = (fill_price - entry_price) * shares
            pnl_pct = (fill_price - entry_price) / entry_price * 100
        else:
            pnl = (entry_price - fill_price) * shares
            pnl_pct = (entry_price - fill_price) / entry_price * 100

        # Calculate time in trade
        time_in_trade = ''
        try:
            entry_time = datetime.strptime(matched_entry['timestamp'], '%Y-%m-%d %H:%M:%S')
            if fill_time:
                fill_dt = fill_time.replace(tzinfo=None) if hasattr(fill_time, 'replace') else fill_time
                if isinstance(fill_dt, str):
                    fill_dt = datetime.fromisoformat(fill_dt.replace('Z', '+00:00')).replace(tzinfo=None)
                delta = fill_dt - entry_time
                total_hours = delta.total_seconds() / 3600
                if total_hours >= 24:
                    time_in_trade = f"{int(total_hours / 24)}d {int(total_hours % 24)}h"
                else:
                    time_in_trade = f"{int(total_hours)}h {int((total_hours % 1) * 60)}m"
        except Exception:
            pass

        # Calculate R-multiple
        r_multiple = None
        sl = matched_entry.get('stop_loss')
        if sl and float(sl) > 0:
            risk = abs(entry_price - float(sl))
            if risk > 0:
                reward = abs(fill_price - entry_price)
                r_multiple = reward / risk if pnl >= 0 else -(reward / risk)

        # Determine exit reasoning from order type, embed order ID for dedup
        order_type = order.type  # 'limit', 'stop', 'market'
        if order_type == 'limit':
            reasoning = f'Take profit filled (Alpaca) [alpaca_order:{order_id}]'
        elif order_type == 'stop':
            reasoning = f'Stop loss filled (Alpaca) [alpaca_order:{order_id}]'
        elif order_type == 'market':
            reasoning = f'Market order exit (Alpaca) [alpaca_order:{order_id}]'
        else:
            reasoning = f'Exit filled (Alpaca, type={order_type}) [alpaca_order:{order_id}]'

        # 5. Log the EXIT (CSV)
        log_trade(
            action='EXIT',
            symbol=symbol,
            strategy=matched_entry.get('strategy', ''),
            direction=exit_direction,
            price=fill_price,
            shares=shares,
            entry_price=entry_price,
            exit_price=fill_price,
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
            reasoning=reasoning,
            time_in_trade=time_in_trade,
            r_multiple=r_multiple,
            bot_name=matched_entry.get('bot_name', ''),
        )

        # 5b. Log to SQLite journal
        if stock_journal:
            try:
                stock_journal.log_exit(
                    symbol=symbol,
                    direction=exit_direction,
                    exit_price=fill_price,
                    exit_reason=reasoning,
                    alpaca_order_id=str(order_id),
                )
            except Exception:
                pass

        # Mark as reconciled to prevent dupes in this run
        dupe_key = (symbol, exit_direction, matched_entry.get('price', ''))
        existing_exits.add(dupe_key)
        reconciled += 1

    if reconciled > 0:
        cprint(f"  [RECONCILE] Reconciled {reconciled} exit(s) from Alpaca fills", "green")
    else:
        cprint(f"  [RECONCILE] All exits already up to date", "white")

    return reconciled


if __name__ == "__main__":
    # Demo / test
    import argparse

    parser = argparse.ArgumentParser(description="Trade Journal Viewer")
    parser.add_argument('--limit', type=int, default=20, help='Number of trades to show')
    parser.add_argument('--symbol', type=str, help='Filter by symbol')
    parser.add_argument('--strategy', type=str, help='Filter by strategy')
    parser.add_argument('--reconcile', action='store_true', help='Reconcile exits from Alpaca')
    args = parser.parse_args()

    if args.reconcile:
        reconcile_exits()
    view_journal(limit=args.limit, symbol=args.symbol, strategy=args.strategy)

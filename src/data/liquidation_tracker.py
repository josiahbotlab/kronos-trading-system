"""
Liquidation Tracker Module

Connects to Binance and OKX liquidation websockets and tracks liquidation events.
Based on Moon Dev's licks.py pattern with robust reconnection logic.

Features:
- Real-time liquidation data from Binance Futures and OKX Swaps
- CSV storage with configurable path
- Robust outer/inner loop reconnection pattern
- Aggregation functions for time-based analysis
- Background running capability
- Contract value adjustment for accurate OKX USD calculations

Websockets:
- Binance: wss://fstream.binance.com/ws/!forceOrder@arr
- OKX: wss://ws.okx.com:8443/ws/v5/public (liquidation-orders channel)

Usage:
    # As standalone script
    python src/data/liquidation_tracker.py

    # As module
    from src.data.liquidation_tracker import get_liquidation_totals, start_tracker
    totals = get_liquidation_totals('15 minutes')
    start_tracker()  # Runs in background thread (both Binance + OKX)
"""

import asyncio
import csv
import json
import os
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from termcolor import cprint

# Add project root to path
sys.path.append('/Users/josiahgarcia/trading-bot')

# Try to import websockets
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    cprint("Warning: websockets not installed. Run: pip install websockets", "yellow")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Binance Futures Liquidation Websocket
BINANCE_LIQUIDATION_WS = "wss://fstream.binance.com/ws/!forceOrder@arr"

# OKX Liquidation Websocket
OKX_LIQUIDATION_WS = "wss://ws.okx.com:8443/ws/v5/public"
OKX_INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments?instType=SWAP"

# Data storage
CSV_PATH = Path("/Users/josiahgarcia/trading-bot/csvs/liquidation_data.csv")
OKX_CSV_PATH = Path("/Users/josiahgarcia/trading-bot/csvs/liquidation_data_okx.csv")
CSV_COLUMNS = ['symbol', 'side', 'position_side', 'price', 'quantity', 'usd_size', 'timestamp', 'source']

# Reconnection settings
RECONNECT_DELAY_BASE = 5      # Base delay between reconnection attempts (seconds)
RECONNECT_DELAY_MAX = 60      # Maximum delay between reconnection attempts
RECONNECT_DELAY_MULTIPLIER = 2  # Exponential backoff multiplier
PING_INTERVAL = 30            # Ping interval to keep connection alive (seconds)
PING_TIMEOUT = 10             # Ping timeout (seconds)

# Time frame mappings for aggregation
TIME_FRAME_MINUTES = {
    '1 minute': 1,
    '5 minutes': 5,
    '15 minutes': 15,
    '30 minutes': 30,
    '1 hour': 60,
    '2 hours': 120,
    '4 hours': 240,
    '6 hours': 360,
    '12 hours': 720,
    '24 hours': 1440,
    '1 day': 1440,
}

# Global state
_tracker_thread: Optional[threading.Thread] = None
_okx_tracker_thread: Optional[threading.Thread] = None
_tracker_running = False
_okx_tracker_running = False
_liquidation_cache: List[Dict] = []  # In-memory cache for recent liquidations
_cache_lock = threading.Lock()
CACHE_MAX_SIZE = 10000  # Maximum liquidations to keep in memory

# OKX contract values cache: {instId: ctVal}
# ctVal is the contract value (e.g., 0.01 for BTC-USDT-SWAP means 1 contract = 0.01 BTC)
_okx_contract_values: Dict[str, float] = {}
_okx_contract_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
# CSV STORAGE
# ─────────────────────────────────────────────────────────────

def ensure_csv_exists():
    """Ensure CSV file exists with proper headers."""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not CSV_PATH.exists():
        with open(CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        cprint(f"Created Binance liquidation CSV: {CSV_PATH}", "green")

    # Also ensure OKX CSV exists
    if not OKX_CSV_PATH.exists():
        with open(OKX_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
        cprint(f"Created OKX liquidation CSV: {OKX_CSV_PATH}", "green")


# ─────────────────────────────────────────────────────────────
# OKX CONTRACT VALUES
# ─────────────────────────────────────────────────────────────

def fetch_okx_contract_values() -> Dict[str, float]:
    """
    Fetch contract values (ctVal) for all OKX SWAP instruments.

    ctVal is critical for USD calculation:
    - BTC-USDT-SWAP: ctVal = 0.01 (1 contract = 0.01 BTC)
    - ETH-USDT-SWAP: ctVal = 0.1 (1 contract = 0.1 ETH)
    - Altcoins: varies (often 1, 10, or 100)

    USD = sz * ctVal * bkPx
    """
    global _okx_contract_values

    try:
        cprint("[OKX] Fetching contract specifications...", "cyan")
        response = requests.get(OKX_INSTRUMENTS_URL, timeout=10)
        response.raise_for_status()

        data = response.json()
        if data.get('code') != '0':
            cprint(f"[OKX] API error: {data.get('msg')}", "red")
            return {}

        instruments = data.get('data', [])
        contract_values = {}

        for inst in instruments:
            inst_id = inst.get('instId', '')
            ct_val = inst.get('ctVal', '1')

            try:
                contract_values[inst_id] = float(ct_val)
            except (ValueError, TypeError):
                contract_values[inst_id] = 1.0

        with _okx_contract_lock:
            _okx_contract_values = contract_values

        cprint(f"[OKX] Loaded {len(contract_values)} contract specifications", "green")

        # Log some examples
        examples = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP']
        for ex in examples:
            if ex in contract_values:
                cprint(f"[OKX]   {ex}: ctVal = {contract_values[ex]}", "white")

        return contract_values

    except requests.exceptions.RequestException as e:
        cprint(f"[OKX] Error fetching contract values: {e}", "red")
        return {}
    except Exception as e:
        cprint(f"[OKX] Unexpected error: {e}", "red")
        return {}


def get_okx_contract_value(inst_id: str) -> float:
    """Get contract value for an OKX instrument."""
    with _okx_contract_lock:
        return _okx_contract_values.get(inst_id, 1.0)


def append_liquidation(liq_data: Dict):
    """Append a liquidation record to CSV and cache."""
    global _liquidation_cache

    ensure_csv_exists()

    # Extract fields
    row = [
        liq_data.get('symbol', ''),
        liq_data.get('side', ''),
        liq_data.get('position_side', ''),
        liq_data.get('price', 0),
        liq_data.get('quantity', 0),
        liq_data.get('usd_size', 0),
        liq_data.get('timestamp', ''),
        liq_data.get('source', 'binance'),
    ]

    # Append to CSV
    try:
        with open(CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        cprint(f"Error writing to CSV: {e}", "red")

    # Add to in-memory cache
    with _cache_lock:
        _liquidation_cache.append(liq_data)
        # Trim cache if too large
        if len(_liquidation_cache) > CACHE_MAX_SIZE:
            _liquidation_cache = _liquidation_cache[-CACHE_MAX_SIZE:]


def load_liquidations_from_csv(since: Optional[datetime] = None, csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Load liquidations from CSV, optionally filtered by time."""
    ensure_csv_exists()

    if csv_path is None:
        csv_path = CSV_PATH

    try:
        if not csv_path.exists():
            return pd.DataFrame(columns=CSV_COLUMNS)

        df = pd.read_csv(csv_path)

        if df.empty:
            return df

        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by time if specified
        if since is not None:
            df = df[df['timestamp'] >= since]

        return df

    except Exception as e:
        cprint(f"Error loading CSV {csv_path}: {e}", "red")
        return pd.DataFrame(columns=CSV_COLUMNS)


# ─────────────────────────────────────────────────────────────
# LIQUIDATION PARSING
# ─────────────────────────────────────────────────────────────

def parse_binance_liquidation(msg: Dict) -> Optional[Dict]:
    """
    Parse Binance liquidation message.

    Binance forceOrder message format:
    {
        "e": "forceOrder",          // Event Type
        "E": 1568014460893,         // Event Time
        "o": {
            "s": "BTCUSDT",         // Symbol
            "S": "SELL",            // Side
            "o": "LIMIT",           // Order Type
            "f": "IOC",             // Time in Force
            "q": "0.014",           // Original Quantity
            "p": "9910",            // Price
            "ap": "9910",           // Average Price
            "X": "FILLED",          // Order Status
            "l": "0.014",           // Order Last Filled Quantity
            "z": "0.014",           // Order Filled Accumulated Quantity
            "T": 1568014460893,     // Order Trade Time
        }
    }
    """
    try:
        if msg.get('e') != 'forceOrder':
            return None

        order = msg.get('o', {})

        symbol = order.get('s', '')
        side = order.get('S', '')  # BUY or SELL
        price = float(order.get('p', 0))
        quantity = float(order.get('q', 0))
        trade_time = order.get('T', 0)

        # Calculate USD size
        usd_size = price * quantity

        # Determine position side (opposite of liquidation side)
        # If liquidation is SELL, the position was LONG
        # If liquidation is BUY, the position was SHORT
        position_side = 'LONG' if side == 'SELL' else 'SHORT'

        # Convert timestamp
        if trade_time:
            timestamp = datetime.fromtimestamp(trade_time / 1000).isoformat()
        else:
            timestamp = datetime.now().isoformat()

        return {
            'symbol': symbol,
            'side': side,
            'position_side': position_side,
            'price': price,
            'quantity': quantity,
            'usd_size': usd_size,
            'timestamp': timestamp,
            'source': 'binance',
        }

    except Exception as e:
        cprint(f"Error parsing Binance liquidation: {e}", "red")
        return None


def parse_okx_liquidation(msg: Dict) -> List[Dict]:
    """
    Parse OKX liquidation message.

    OKX liquidation-orders message format:
    {
        "arg": {"channel": "liquidation-orders", "instType": "SWAP"},
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "instType": "SWAP",
            "instFamily": "BTC-USDT",
            "uly": "BTC-USDT",
            "details": [{
                "bkLoss": "0",
                "bkPx": "67890.5",      // Bankruptcy price
                "ccy": "",
                "posSide": "long",       // Position side
                "side": "sell",          // Liquidation side
                "sz": "100",             // Size in CONTRACTS (not USD!)
                "ts": "1723892524781"    // Timestamp ms
            }]
        }]
    }

    IMPORTANT: sz is in contracts. To get USD:
    USD = sz * ctVal * bkPx

    Where ctVal is the contract value from OKX instruments API.
    """
    liquidations = []

    try:
        # Check if this is a liquidation message
        arg = msg.get('arg', {})
        if arg.get('channel') != 'liquidation-orders':
            return []

        data_list = msg.get('data', [])

        for data in data_list:
            inst_id = data.get('instId', '')
            details = data.get('details', [])

            # Get contract value for this instrument
            ct_val = get_okx_contract_value(inst_id)

            for detail in details:
                try:
                    bk_px = float(detail.get('bkPx', 0))
                    sz = float(detail.get('sz', 0))
                    pos_side = detail.get('posSide', '').upper()
                    side = detail.get('side', '').upper()
                    ts = detail.get('ts', 0)

                    # Calculate USD size correctly:
                    # sz is contracts, ct_val is contract value, bk_px is price
                    # USD = contracts * contract_value * price
                    usd_size = sz * ct_val * bk_px

                    # Map OKX side to standard format
                    position_side = 'LONG' if pos_side == 'LONG' else 'SHORT'

                    # Convert timestamp
                    if ts:
                        timestamp = datetime.fromtimestamp(int(ts) / 1000).isoformat()
                    else:
                        timestamp = datetime.now().isoformat()

                    # Convert instId to symbol format similar to Binance
                    # BTC-USDT-SWAP -> BTCUSDT
                    symbol = inst_id.replace('-SWAP', '').replace('-', '')

                    liquidations.append({
                        'symbol': symbol,
                        'side': side,
                        'position_side': position_side,
                        'price': bk_px,
                        'quantity': sz * ct_val,  # Convert to actual quantity
                        'usd_size': usd_size,
                        'timestamp': timestamp,
                        'source': 'okx',
                        'inst_id': inst_id,  # Keep original for reference
                        'ct_val': ct_val,    # For debugging
                    })

                except (ValueError, TypeError) as e:
                    cprint(f"Error parsing OKX detail: {e}", "red")
                    continue

    except Exception as e:
        cprint(f"Error parsing OKX liquidation: {e}", "red")

    return liquidations


def append_okx_liquidation(liq_data: Dict):
    """Append an OKX liquidation record to CSV and cache."""
    global _liquidation_cache

    ensure_csv_exists()

    # Extract fields (same format as Binance)
    row = [
        liq_data.get('symbol', ''),
        liq_data.get('side', ''),
        liq_data.get('position_side', ''),
        liq_data.get('price', 0),
        liq_data.get('quantity', 0),
        liq_data.get('usd_size', 0),
        liq_data.get('timestamp', ''),
        liq_data.get('source', 'okx'),
    ]

    # Append to OKX CSV
    try:
        with open(OKX_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    except Exception as e:
        cprint(f"Error writing to OKX CSV: {e}", "red")

    # Add to shared in-memory cache
    with _cache_lock:
        _liquidation_cache.append(liq_data)
        # Trim cache if too large
        if len(_liquidation_cache) > CACHE_MAX_SIZE:
            _liquidation_cache = _liquidation_cache[-CACHE_MAX_SIZE:]


# ─────────────────────────────────────────────────────────────
# BINANCE WEBSOCKET CONNECTION
# ─────────────────────────────────────────────────────────────

async def connect_and_stream():
    """
    Connect to Binance liquidation websocket with robust reconnection.

    Uses Moon Dev's outer/inner loop pattern:
    - Outer loop: Handles reconnection on connection drop
    - Inner loop: Processes messages while connected
    """
    global _tracker_running

    reconnect_delay = RECONNECT_DELAY_BASE
    connection_count = 0

    cprint("\n" + "=" * 60, "cyan")
    cprint("  LIQUIDATION TRACKER", "cyan", attrs=['bold'])
    cprint("  Binance Futures Real-Time Liquidations", "cyan")
    cprint("=" * 60, "cyan")
    cprint(f"  Websocket: {BINANCE_LIQUIDATION_WS}", "white")
    cprint(f"  CSV Path: {CSV_PATH}", "white")
    cprint("=" * 60 + "\n", "cyan")

    # ─── OUTER LOOP: Reconnection handling ───
    while _tracker_running:
        connection_count += 1

        try:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to Binance... (attempt #{connection_count})", "yellow")

            async with websockets.connect(
                BINANCE_LIQUIDATION_WS,
                ping_interval=PING_INTERVAL,
                ping_timeout=PING_TIMEOUT,
                close_timeout=10,
            ) as ws:

                cprint(f"[{datetime.now().strftime('%H:%M:%S')}] Connected to Binance liquidation stream", "green")

                # Reset reconnect delay on successful connection
                reconnect_delay = RECONNECT_DELAY_BASE

                # Track stats
                msg_count = 0
                total_usd = 0
                start_time = datetime.now()

                # ─── INNER LOOP: Message processing ───
                while _tracker_running:
                    try:
                        # Receive message with timeout
                        msg_raw = await asyncio.wait_for(ws.recv(), timeout=60)

                        # Parse JSON
                        msg = json.loads(msg_raw)

                        # Parse liquidation
                        liq = parse_binance_liquidation(msg)

                        if liq:
                            msg_count += 1
                            total_usd += liq['usd_size']

                            # Store to CSV and cache
                            append_liquidation(liq)

                            # Log significant liquidations (> $10k)
                            if liq['usd_size'] >= 10000:
                                color = "red" if liq['position_side'] == 'LONG' else "green"
                                cprint(
                                    f"  [{liq['timestamp'][-8:]}] {liq['symbol']:10} "
                                    f"{liq['position_side']:5} liquidated: "
                                    f"${liq['usd_size']:>12,.2f} @ ${liq['price']:,.2f}",
                                    color
                                )

                            # Periodic stats
                            if msg_count % 100 == 0:
                                elapsed = (datetime.now() - start_time).total_seconds() / 60
                                cprint(
                                    f"  [STATS] {msg_count} liquidations | "
                                    f"${total_usd:,.0f} total | "
                                    f"{elapsed:.1f} min uptime",
                                    "cyan"
                                )

                    except asyncio.TimeoutError:
                        # No message received, but connection is still alive
                        # This is normal during quiet periods
                        continue

                    except websockets.exceptions.ConnectionClosed as e:
                        cprint(f"[{datetime.now().strftime('%H:%M:%S')}] Connection closed: {e}", "yellow")
                        break

                    except json.JSONDecodeError as e:
                        cprint(f"JSON decode error: {e}", "red")
                        continue

                    except Exception as e:
                        cprint(f"Error processing message: {e}", "red")
                        continue

        except websockets.exceptions.InvalidStatusCode as e:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] Invalid status code: {e}", "red")

        except websockets.exceptions.WebSocketException as e:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] WebSocket error: {e}", "red")

        except ConnectionRefusedError:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] Connection refused", "red")

        except Exception as e:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] Unexpected error: {e}", "red")

        # ─── Reconnection delay with exponential backoff ───
        if _tracker_running:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] Reconnecting in {reconnect_delay}s...", "yellow")
            await asyncio.sleep(reconnect_delay)

            # Exponential backoff
            reconnect_delay = min(reconnect_delay * RECONNECT_DELAY_MULTIPLIER, RECONNECT_DELAY_MAX)

    cprint("\nBinance liquidation tracker stopped.", "yellow")


def _run_tracker_loop():
    """Run the Binance tracker in an asyncio event loop (for threading)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(connect_and_stream())
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────
# OKX WEBSOCKET CONNECTION
# ─────────────────────────────────────────────────────────────

async def connect_and_stream_okx():
    """
    Connect to OKX liquidation websocket with robust reconnection.

    Uses the same outer/inner loop pattern as Binance.
    """
    global _okx_tracker_running

    reconnect_delay = RECONNECT_DELAY_BASE
    connection_count = 0

    cprint("\n" + "=" * 60, "magenta")
    cprint("  OKX LIQUIDATION TRACKER", "magenta", attrs=['bold'])
    cprint("  OKX Perpetual Swap Liquidations", "magenta")
    cprint("=" * 60, "magenta")
    cprint(f"  Websocket: {OKX_LIQUIDATION_WS}", "white")
    cprint(f"  CSV Path: {OKX_CSV_PATH}", "white")
    cprint("=" * 60 + "\n", "magenta")

    # Fetch contract values before starting
    fetch_okx_contract_values()

    # ─── OUTER LOOP: Reconnection handling ───
    while _okx_tracker_running:
        connection_count += 1

        try:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] [OKX] Connecting... (attempt #{connection_count})", "yellow")

            async with websockets.connect(
                OKX_LIQUIDATION_WS,
                ping_interval=PING_INTERVAL,
                ping_timeout=PING_TIMEOUT,
                close_timeout=10,
            ) as ws:

                # Subscribe to liquidation-orders channel
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [{"channel": "liquidation-orders", "instType": "SWAP"}]
                }
                await ws.send(json.dumps(subscribe_msg))
                cprint(f"[{datetime.now().strftime('%H:%M:%S')}] [OKX] Subscribed to liquidation-orders", "green")

                # Reset reconnect delay on successful connection
                reconnect_delay = RECONNECT_DELAY_BASE

                # Track stats
                msg_count = 0
                total_usd = 0
                start_time = datetime.now()

                # ─── INNER LOOP: Message processing ───
                while _okx_tracker_running:
                    try:
                        # Receive message with timeout
                        msg_raw = await asyncio.wait_for(ws.recv(), timeout=60)

                        # Parse JSON
                        msg = json.loads(msg_raw)

                        # Skip subscription confirmation messages
                        if msg.get('event') == 'subscribe':
                            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] [OKX] Subscription confirmed", "green")
                            continue

                        # Parse liquidations (returns list since one message can have multiple)
                        liquidations = parse_okx_liquidation(msg)

                        for liq in liquidations:
                            msg_count += 1
                            total_usd += liq['usd_size']

                            # Store to CSV and cache
                            append_okx_liquidation(liq)

                            # Log significant liquidations (> $10k)
                            if liq['usd_size'] >= 10000:
                                color = "red" if liq['position_side'] == 'LONG' else "green"
                                cprint(
                                    f"  [OKX] [{liq['timestamp'][-8:]}] {liq['symbol']:10} "
                                    f"{liq['position_side']:5} liquidated: "
                                    f"${liq['usd_size']:>12,.2f} @ ${liq['price']:,.2f}",
                                    color
                                )

                        # Periodic stats
                        if msg_count > 0 and msg_count % 50 == 0:
                            elapsed = (datetime.now() - start_time).total_seconds() / 60
                            cprint(
                                f"  [OKX STATS] {msg_count} liquidations | "
                                f"${total_usd:,.0f} total | "
                                f"{elapsed:.1f} min uptime",
                                "magenta"
                            )

                    except asyncio.TimeoutError:
                        # No message received, connection still alive
                        continue

                    except websockets.exceptions.ConnectionClosed as e:
                        cprint(f"[{datetime.now().strftime('%H:%M:%S')}] [OKX] Connection closed: {e}", "yellow")
                        break

                    except json.JSONDecodeError as e:
                        cprint(f"[OKX] JSON decode error: {e}", "red")
                        continue

                    except Exception as e:
                        cprint(f"[OKX] Error processing message: {e}", "red")
                        continue

        except websockets.exceptions.InvalidStatusCode as e:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] [OKX] Invalid status code: {e}", "red")

        except websockets.exceptions.WebSocketException as e:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] [OKX] WebSocket error: {e}", "red")

        except ConnectionRefusedError:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] [OKX] Connection refused", "red")

        except Exception as e:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] [OKX] Unexpected error: {e}", "red")

        # ─── Reconnection delay with exponential backoff ───
        if _okx_tracker_running:
            cprint(f"[{datetime.now().strftime('%H:%M:%S')}] [OKX] Reconnecting in {reconnect_delay}s...", "yellow")
            await asyncio.sleep(reconnect_delay)

            # Exponential backoff
            reconnect_delay = min(reconnect_delay * RECONNECT_DELAY_MULTIPLIER, RECONNECT_DELAY_MAX)

            # Refresh contract values on reconnect
            fetch_okx_contract_values()

    cprint("\nOKX liquidation tracker stopped.", "yellow")


def _run_okx_tracker_loop():
    """Run the OKX tracker in an asyncio event loop (for threading)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(connect_and_stream_okx())
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def start_tracker(background: bool = True, include_okx: bool = True) -> Optional[threading.Thread]:
    """
    Start the liquidation trackers (Binance + optionally OKX).

    Args:
        background: If True, runs in background threads. If False, blocks.
        include_okx: If True, also starts OKX tracker.

    Returns:
        Binance thread object if background=True, None otherwise.
    """
    global _tracker_thread, _tracker_running, _okx_tracker_thread, _okx_tracker_running

    if not WEBSOCKETS_AVAILABLE:
        cprint("Cannot start tracker: websockets library not installed", "red")
        return None

    if _tracker_running:
        cprint("Binance tracker is already running", "yellow")
    else:
        _tracker_running = True
        ensure_csv_exists()

        if background:
            _tracker_thread = threading.Thread(target=_run_tracker_loop, daemon=True)
            _tracker_thread.start()
            cprint("Binance liquidation tracker started in background", "green")

    # Start OKX tracker if requested
    if include_okx and not _okx_tracker_running:
        _okx_tracker_running = True
        if background:
            _okx_tracker_thread = threading.Thread(target=_run_okx_tracker_loop, daemon=True)
            _okx_tracker_thread.start()
            cprint("OKX liquidation tracker started in background", "green")

    if not background:
        # Run Binance in foreground (blocking) - OKX runs in background
        asyncio.run(connect_and_stream())

    return _tracker_thread


def start_okx_tracker(background: bool = True) -> Optional[threading.Thread]:
    """Start only the OKX liquidation tracker."""
    global _okx_tracker_thread, _okx_tracker_running

    if not WEBSOCKETS_AVAILABLE:
        cprint("Cannot start OKX tracker: websockets library not installed", "red")
        return None

    if _okx_tracker_running:
        cprint("OKX tracker is already running", "yellow")
        return _okx_tracker_thread

    _okx_tracker_running = True
    ensure_csv_exists()

    if background:
        _okx_tracker_thread = threading.Thread(target=_run_okx_tracker_loop, daemon=True)
        _okx_tracker_thread.start()
        cprint("OKX liquidation tracker started in background", "green")
        return _okx_tracker_thread
    else:
        asyncio.run(connect_and_stream_okx())
        return None


def stop_tracker():
    """Stop all liquidation trackers."""
    global _tracker_running, _okx_tracker_running
    _tracker_running = False
    _okx_tracker_running = False
    cprint("Stopping all liquidation trackers...", "yellow")


def stop_okx_tracker():
    """Stop only the OKX tracker."""
    global _okx_tracker_running
    _okx_tracker_running = False
    cprint("Stopping OKX liquidation tracker...", "yellow")


def is_tracker_running() -> bool:
    """Check if any tracker is currently running."""
    return _tracker_running or _okx_tracker_running


def get_tracker_status() -> Dict[str, bool]:
    """Get status of all trackers."""
    return {
        'binance': _tracker_running,
        'okx': _okx_tracker_running,
    }


def get_liquidation_totals(time_frame: str = '15 minutes', symbol: Optional[str] = None, source: Optional[str] = None) -> Dict:
    """
    Get combined liquidation totals from Binance + OKX for a given time frame.

    Args:
        time_frame: Time window (e.g., '5 minutes', '1 hour', '24 hours')
        symbol: Optional symbol filter (e.g., 'BTCUSDT')
        source: Optional source filter ('binance', 'okx', or None for both)

    Returns:
        Dict with keys:
        - total_usd: Total liquidation value in USD (Binance + OKX)
        - long_usd: Long position liquidations in USD
        - short_usd: Short position liquidations in USD
        - total_count: Number of liquidations
        - long_count: Number of long liquidations
        - short_count: Number of short liquidations
        - binance_usd: Binance-only total
        - okx_usd: OKX-only total
        - time_frame: The time frame used
        - since: Start timestamp of the window
    """
    # Parse time frame
    minutes = TIME_FRAME_MINUTES.get(time_frame.lower())
    if minutes is None:
        # Try to parse custom format like "10 minutes"
        try:
            parts = time_frame.lower().split()
            minutes = int(parts[0])
        except:
            cprint(f"Unknown time frame: {time_frame}. Using 15 minutes.", "yellow")
            minutes = 15

    since = datetime.now() - timedelta(minutes=minutes)

    # First try in-memory cache (faster) - contains both Binance and OKX
    result = _get_totals_from_cache(since, symbol, source)

    # If cache is empty or insufficient, fall back to CSVs
    if result['total_count'] == 0:
        result = _get_totals_from_csv(since, symbol, source)

    result['time_frame'] = time_frame
    result['since'] = since.isoformat()

    return result


def _get_totals_from_cache(since: datetime, symbol: Optional[str] = None, source: Optional[str] = None) -> Dict:
    """Get totals from in-memory cache (contains both Binance and OKX)."""
    with _cache_lock:
        filtered = []
        for liq in _liquidation_cache:
            try:
                liq_time = datetime.fromisoformat(liq['timestamp'])
                if liq_time >= since:
                    if symbol is None or liq.get('symbol') == symbol:
                        if source is None or liq.get('source') == source:
                            filtered.append(liq)
            except:
                continue

        return _calculate_totals(filtered)


def _get_totals_from_csv(since: datetime, symbol: Optional[str] = None, source: Optional[str] = None) -> Dict:
    """Get totals from both CSV files (Binance + OKX)."""
    all_records = []

    # Load Binance CSV
    if source is None or source == 'binance':
        df_binance = load_liquidations_from_csv(since, csv_path=CSV_PATH)
        if not df_binance.empty:
            if symbol:
                df_binance = df_binance[df_binance['symbol'] == symbol]
            all_records.extend(df_binance.to_dict('records'))

    # Load OKX CSV
    if source is None or source == 'okx':
        df_okx = load_liquidations_from_csv(since, csv_path=OKX_CSV_PATH)
        if not df_okx.empty:
            if symbol:
                df_okx = df_okx[df_okx['symbol'] == symbol]
            all_records.extend(df_okx.to_dict('records'))

    return _calculate_totals(all_records)


def _calculate_totals(liquidations: List[Dict]) -> Dict:
    """Calculate totals from a list of liquidation records (Binance + OKX combined)."""
    total_usd = 0
    long_usd = 0
    short_usd = 0
    total_count = len(liquidations)
    long_count = 0
    short_count = 0

    # Track by source
    binance_usd = 0
    binance_count = 0
    okx_usd = 0
    okx_count = 0

    for liq in liquidations:
        usd = float(liq.get('usd_size', 0))
        position_side = liq.get('position_side', '')
        source = liq.get('source', 'binance')

        total_usd += usd

        if position_side == 'LONG':
            long_usd += usd
            long_count += 1
        elif position_side == 'SHORT':
            short_usd += usd
            short_count += 1

        # Track by source
        if source == 'okx':
            okx_usd += usd
            okx_count += 1
        else:
            binance_usd += usd
            binance_count += 1

    return {
        'total_usd': total_usd,
        'long_usd': long_usd,
        'short_usd': short_usd,
        'total_count': total_count,
        'long_count': long_count,
        'short_count': short_count,
        'long_pct': (long_usd / total_usd * 100) if total_usd > 0 else 0,
        'short_pct': (short_usd / total_usd * 100) if total_usd > 0 else 0,
        'binance_usd': binance_usd,
        'binance_count': binance_count,
        'okx_usd': okx_usd,
        'okx_count': okx_count,
    }


def get_liquidation_history(
    hours: int = 24,
    symbol: Optional[str] = None,
    min_usd: float = 0
) -> pd.DataFrame:
    """
    Get liquidation history as a DataFrame.

    Args:
        hours: Number of hours of history
        symbol: Optional symbol filter
        min_usd: Minimum USD size filter

    Returns:
        DataFrame with liquidation records
    """
    since = datetime.now() - timedelta(hours=hours)
    df = load_liquidations_from_csv(since)

    if df.empty:
        return df

    if symbol:
        df = df[df['symbol'] == symbol]

    if min_usd > 0:
        df = df[df['usd_size'] >= min_usd]

    return df.sort_values('timestamp', ascending=False)


def get_top_liquidated_symbols(time_frame: str = '1 hour', top_n: int = 10) -> List[Dict]:
    """
    Get top liquidated symbols by USD volume.

    Args:
        time_frame: Time window
        top_n: Number of symbols to return

    Returns:
        List of dicts with symbol and totals
    """
    minutes = TIME_FRAME_MINUTES.get(time_frame.lower(), 60)
    since = datetime.now() - timedelta(minutes=minutes)

    df = load_liquidations_from_csv(since)

    if df.empty:
        return []

    # Group by symbol
    grouped = df.groupby('symbol').agg({
        'usd_size': 'sum',
        'symbol': 'count'
    }).rename(columns={'symbol': 'count'})

    grouped = grouped.sort_values('usd_size', ascending=False).head(top_n)

    result = []
    for symbol, row in grouped.iterrows():
        result.append({
            'symbol': symbol,
            'total_usd': row['usd_size'],
            'count': int(row['count'])
        })

    return result


def print_liquidation_summary(time_frame: str = '15 minutes'):
    """Print a formatted liquidation summary with source breakdown."""
    totals = get_liquidation_totals(time_frame)

    cprint(f"\n{'─'*60}", "cyan")
    cprint(f"  LIQUIDATION SUMMARY ({time_frame})", "cyan", attrs=['bold'])
    cprint(f"  Combined: Binance + OKX", "cyan")
    cprint(f"{'─'*60}", "cyan")

    cprint(f"  Total:      ${totals['total_usd']:>15,.2f}  ({totals['total_count']} trades)", "white")

    long_color = "red" if totals['long_usd'] > totals['short_usd'] else "white"
    short_color = "green" if totals['short_usd'] > totals['long_usd'] else "white"

    cprint(f"  Longs:      ${totals['long_usd']:>15,.2f}  ({totals['long_pct']:.1f}%)", long_color)
    cprint(f"  Shorts:     ${totals['short_usd']:>15,.2f}  ({totals['short_pct']:.1f}%)", short_color)

    # Source breakdown
    cprint(f"{'─'*60}", "cyan")
    binance_usd = totals.get('binance_usd', 0)
    binance_count = totals.get('binance_count', 0)
    okx_usd = totals.get('okx_usd', 0)
    okx_count = totals.get('okx_count', 0)

    cprint(f"  Binance:    ${binance_usd:>15,.2f}  ({binance_count} trades)", "yellow")
    cprint(f"  OKX:        ${okx_usd:>15,.2f}  ({okx_count} trades)", "magenta")

    # Sentiment indicator
    if totals['total_usd'] > 0:
        if totals['long_pct'] > 60:
            cprint(f"\n  Sentiment:  BEARISH (longs getting liquidated)", "red")
        elif totals['short_pct'] > 60:
            cprint(f"\n  Sentiment:  BULLISH (shorts getting liquidated)", "green")
        else:
            cprint(f"\n  Sentiment:  NEUTRAL", "yellow")

    cprint(f"{'─'*60}\n", "cyan")


# ─────────────────────────────────────────────────────────────
# TEST FUNCTION
# ─────────────────────────────────────────────────────────────

def test_trackers(duration_seconds: int = 120):
    """
    Test both Binance and OKX trackers for comparison against CoinGlass.

    Runs both trackers for the specified duration, then prints a comparison.
    """
    cprint("\n" + "=" * 70, "cyan")
    cprint("  LIQUIDATION TRACKER TEST", "cyan", attrs=['bold'])
    cprint("  Testing Binance + OKX for comparison against CoinGlass", "cyan")
    cprint("=" * 70, "cyan")
    cprint(f"\n  Duration: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)", "white")
    cprint("  Compare results against: https://www.coinglass.com/LiquidationData", "white")
    cprint("\n  Starting trackers...\n", "yellow")

    # Clear any existing data in memory cache
    global _liquidation_cache
    with _cache_lock:
        _liquidation_cache = []

    # Start both trackers
    start_tracker(background=True, include_okx=True)

    # Wait for connections
    time.sleep(5)

    start_time = datetime.now()
    cprint(f"\n[TEST] Started at {start_time.strftime('%H:%M:%S')}", "green")
    cprint(f"[TEST] Running for {duration_seconds} seconds...\n", "yellow")

    # Show periodic updates
    update_interval = 30  # seconds
    elapsed = 0

    try:
        while elapsed < duration_seconds:
            time.sleep(min(update_interval, duration_seconds - elapsed))
            elapsed = (datetime.now() - start_time).total_seconds()

            cprint(f"\n[TEST] {elapsed:.0f}s elapsed...", "cyan")
            print_liquidation_summary(f"{int(elapsed/60)+1} minutes")

    except KeyboardInterrupt:
        cprint("\n[TEST] Interrupted by user", "yellow")

    # Final results
    stop_tracker()

    cprint("\n" + "=" * 70, "green")
    cprint("  TEST COMPLETE - FINAL RESULTS", "green", attrs=['bold'])
    cprint("=" * 70, "green")

    # Get totals for the test period
    test_minutes = int((datetime.now() - start_time).total_seconds() / 60) + 1
    totals = get_liquidation_totals(f"{test_minutes} minutes")

    cprint(f"\n  Test Duration: {test_minutes} minutes", "white")
    cprint(f"  Total Liquidations: ${totals['total_usd']:,.2f} ({totals['total_count']} trades)", "white")

    cprint(f"\n  BY SOURCE:", "cyan")
    cprint(f"    Binance: ${totals.get('binance_usd', 0):>15,.2f} ({totals.get('binance_count', 0)} trades)", "yellow")
    cprint(f"    OKX:     ${totals.get('okx_usd', 0):>15,.2f} ({totals.get('okx_count', 0)} trades)", "magenta")

    cprint(f"\n  BY POSITION:", "cyan")
    cprint(f"    Longs:   ${totals['long_usd']:>15,.2f} ({totals['long_pct']:.1f}%)", "red")
    cprint(f"    Shorts:  ${totals['short_usd']:>15,.2f} ({totals['short_pct']:.1f}%)", "green")

    cprint(f"\n  COMPARE AGAINST:", "yellow")
    cprint(f"    CoinGlass: https://www.coinglass.com/LiquidationData", "white")
    cprint(f"    Select '{test_minutes}m' or 'All' timeframe and compare totals", "white")

    cprint("\n" + "=" * 70 + "\n", "green")

    return totals


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    """Main entry point for standalone execution."""
    import argparse

    parser = argparse.ArgumentParser(description="Binance + OKX Liquidation Tracker")
    parser.add_argument("--background", action="store_true", help="Run in background mode")
    parser.add_argument("--summary", action="store_true", help="Print summary and exit")
    parser.add_argument("--time-frame", default="15 minutes", help="Time frame for summary")
    parser.add_argument("--test", type=int, metavar="SECONDS", help="Run test for N seconds")
    parser.add_argument("--okx-only", action="store_true", help="Run only OKX tracker")
    parser.add_argument("--binance-only", action="store_true", help="Run only Binance tracker")
    args = parser.parse_args()

    if args.summary:
        print_liquidation_summary(args.time_frame)
        return

    if args.test:
        test_trackers(args.test)
        return

    if args.okx_only:
        if args.background:
            start_okx_tracker(background=True)
            cprint("OKX tracker running in background. Press Ctrl+C to stop.", "yellow")
            try:
                while True:
                    time.sleep(60)
                    print_liquidation_summary(args.time_frame)
            except KeyboardInterrupt:
                stop_okx_tracker()
        else:
            start_okx_tracker(background=False)
        return

    include_okx = not args.binance_only

    if args.background:
        start_tracker(background=True, include_okx=include_okx)
        source_msg = "Binance + OKX" if include_okx else "Binance only"
        cprint(f"Tracker ({source_msg}) running in background. Press Ctrl+C to stop.", "yellow")
        try:
            while True:
                time.sleep(60)
                print_liquidation_summary(args.time_frame)
        except KeyboardInterrupt:
            stop_tracker()
    else:
        # Run in foreground
        start_tracker(background=False, include_okx=include_okx)


if __name__ == "__main__":
    main()

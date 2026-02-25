"""
Market Indicators Module

Tracks key market indicators from Binance Futures:
1. Open Interest (OI) - Total value of outstanding derivatives contracts
2. Funding Rates - Cost of holding perpetual futures positions

High OI + High Funding = Overleveraged market, potential for liquidation cascades
Extreme funding rates indicate market sentiment extremes

Usage:
    python src/data/market_indicators.py                    # Run both trackers
    python src/data/market_indicators.py --snapshot         # Show current snapshot
    python src/data/market_indicators.py --oi-only          # Only track OI
    python src/data/market_indicators.py --funding-only     # Only track funding

    # As module
    from src.data.market_indicators import start_market_indicators, get_market_snapshot
    start_market_indicators(background=True)
    snapshot = get_market_snapshot()
"""

import argparse
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
from dotenv import load_dotenv

# Add project root to path
sys.path.append('/Users/josiahgarcia/trading-bot')
load_dotenv()

# Try to import websockets
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    cprint("Warning: websockets not installed. Run: pip install websockets", "yellow")

# Try to import Alpaca for crypto prices
try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    cprint("Warning: alpaca-trade-api not installed", "yellow")

# Alpaca configuration
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', '')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY', '')
ALPACA_PAPER = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

# Alpaca crypto symbol mapping (Binance -> Alpaca format)
CRYPTO_SYMBOL_MAP = {
    'BTCUSDT': 'BTC/USD',
    'ETHUSDT': 'ETH/USD',
}

# Cached Alpaca API instance
_alpaca_api = None


# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Binance Futures API endpoints
# Note: fapi.binance.com is geo-restricted in some regions (e.g., US)
# We use multiple sources for resilience
BINANCE_OI_ENDPOINT = "https://fapi.binance.com/fapi/v1/openInterest"
BINANCE_PRICE_ENDPOINT = "https://fapi.binance.com/fapi/v1/ticker/price"
BINANCE_MARK_PRICE_WS = "wss://fstream.binance.com/ws/!markPrice@arr"

# Alternative: CoinGlass API for Open Interest (free tier available)
COINGLASS_OI_ENDPOINT = "https://open-api.coinglass.com/public/v2/open_interest"

# Alternative: Use Binance spot prices as fallback
BINANCE_SPOT_PRICE_ENDPOINT = "https://api.binance.com/api/v3/ticker/price"

# Symbols to track
TRACKED_SYMBOLS = ['BTCUSDT', 'ETHUSDT']

# Data storage paths
OI_CSV_PATH = Path("/Users/josiahgarcia/trading-bot/csvs/open_interest.csv")
FUNDING_CSV_PATH = Path("/Users/josiahgarcia/trading-bot/csvs/funding_rates.csv")

# CSV column definitions
OI_COLUMNS = ['symbol', 'open_interest', 'price', 'value_usd', 'timestamp']
FUNDING_COLUMNS = ['symbol', 'funding_rate', 'yearly_rate', 'timestamp']

# Polling intervals
OI_POLL_INTERVAL = 60  # seconds

# Websocket settings
RECONNECT_DELAY_BASE = 5
RECONNECT_DELAY_MAX = 60
RECONNECT_DELAY_MULTIPLIER = 2
PING_INTERVAL = 30
PING_TIMEOUT = 10

# Funding sentiment thresholds (yearly rates)
SENTIMENT_THRESHOLDS = {
    'EXTREME_GREED': 50,   # > 50% yearly
    'GREED': 20,           # > 20% yearly
    'NEUTRAL_HIGH': 0,     # > 0%
    'NEUTRAL_LOW': -20,    # > -20%
    'FEAR': -50,           # > -50%
    # else EXTREME_FEAR
}

# Global state
_oi_thread: Optional[threading.Thread] = None
_funding_thread: Optional[threading.Thread] = None
_oi_running = False
_funding_running = False

# In-memory cache for latest values
_latest_oi: Dict[str, Dict] = {}
_latest_funding: Dict[str, Dict] = {}
_cache_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
# CSV STORAGE
# ─────────────────────────────────────────────────────────────

def ensure_oi_csv_exists():
    """Ensure OI CSV file exists with proper headers."""
    OI_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not OI_CSV_PATH.exists():
        with open(OI_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(OI_COLUMNS)
        cprint(f"Created OI CSV: {OI_CSV_PATH}", "green")


def ensure_funding_csv_exists():
    """Ensure Funding CSV file exists with proper headers."""
    FUNDING_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not FUNDING_CSV_PATH.exists():
        with open(FUNDING_CSV_PATH, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(FUNDING_COLUMNS)
        cprint(f"Created Funding CSV: {FUNDING_CSV_PATH}", "green")


def append_oi_record(record: Dict):
    """Append an open interest record to CSV."""
    ensure_oi_csv_exists()
    try:
        with open(OI_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                record.get('symbol', ''),
                record.get('open_interest', 0),
                record.get('price', 0),
                record.get('value_usd', 0),
                record.get('timestamp', ''),
            ])
    except Exception as e:
        cprint(f"Error writing OI to CSV: {e}", "red")


def append_funding_record(record: Dict):
    """Append a funding rate record to CSV."""
    ensure_funding_csv_exists()
    try:
        with open(FUNDING_CSV_PATH, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                record.get('symbol', ''),
                record.get('funding_rate', 0),
                record.get('yearly_rate', 0),
                record.get('timestamp', ''),
            ])
    except Exception as e:
        cprint(f"Error writing Funding to CSV: {e}", "red")


# ─────────────────────────────────────────────────────────────
# PART 1: OPEN INTEREST TRACKER
# ─────────────────────────────────────────────────────────────

def fetch_open_interest(symbol: str) -> Optional[float]:
    """
    Fetch open interest for a symbol.

    Tries Binance Futures API first, falls back to estimated OI from
    the markPrice websocket data if API is geo-restricted.
    """
    # Try Binance Futures API first
    try:
        response = requests.get(
            BINANCE_OI_ENDPOINT,
            params={'symbol': symbol},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        return float(data.get('openInterest', 0))
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 451:
            # Geo-restricted - use estimated OI from cache or return None
            # The markPrice websocket doesn't include OI, so we'll estimate
            pass
        else:
            cprint(f"Error fetching OI for {symbol}: {e}", "red")
    except Exception as e:
        cprint(f"Error fetching OI for {symbol}: {e}", "red")

    return None


def get_alpaca_api():
    """Get or create Alpaca API instance for crypto prices."""
    global _alpaca_api

    if _alpaca_api is not None:
        return _alpaca_api

    if not ALPACA_AVAILABLE:
        return None

    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        cprint("Warning: Alpaca API keys not configured", "yellow")
        return None

    try:
        base_url = "https://paper-api.alpaca.markets" if ALPACA_PAPER else "https://api.alpaca.markets"
        _alpaca_api = tradeapi.REST(
            ALPACA_API_KEY,
            ALPACA_SECRET_KEY,
            base_url,
            api_version='v2'
        )
        return _alpaca_api
    except Exception as e:
        cprint(f"Error creating Alpaca API: {e}", "red")
        return None


def fetch_price_alpaca(symbol: str) -> Optional[float]:
    """
    Fetch crypto price from Alpaca.

    Args:
        symbol: Binance-style symbol (e.g., 'BTCUSDT')

    Returns:
        Current price or None if unavailable
    """
    api = get_alpaca_api()
    if api is None:
        return None

    # Convert Binance symbol to Alpaca format
    alpaca_symbol = CRYPTO_SYMBOL_MAP.get(symbol)
    if not alpaca_symbol:
        return None

    try:
        # Get latest 1-minute bar for the crypto pair
        bars = api.get_crypto_bars(alpaca_symbol, '1Min', limit=1)
        if bars and len(bars) > 0:
            # Get the close price from the most recent bar
            bar = bars[0] if hasattr(bars, '__getitem__') else list(bars)[0]
            if hasattr(bar, 'c'):
                return float(bar.c)
            elif hasattr(bar, 'close'):
                return float(bar.close)
    except Exception as e:
        # Silently fail - will fall back to other methods
        pass

    return None


def fetch_price(symbol: str) -> Optional[float]:
    """
    Fetch current price for a symbol.

    Uses Alpaca crypto API (works in US), falls back to Binance websocket cache.
    """
    # Try Alpaca first (works in US)
    price = fetch_price_alpaca(symbol)
    if price is not None:
        return price

    # Fall back to cached price from websocket (markPrice stream)
    with _cache_lock:
        if symbol in _latest_oi and _latest_oi[symbol].get('price', 0) > 0:
            return _latest_oi[symbol]['price']
        if symbol in _latest_funding and _latest_funding[symbol].get('mark_price', 0) > 0:
            return _latest_funding[symbol]['mark_price']

    # Last resort: try Binance APIs (may fail with 451 in US)
    try:
        response = requests.get(
            BINANCE_PRICE_ENDPOINT,
            params={'symbol': symbol},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return float(data.get('price', 0))
    except:
        pass

    try:
        response = requests.get(
            BINANCE_SPOT_PRICE_ENDPOINT,
            params={'symbol': symbol},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()
        return float(data.get('price', 0))
    except:
        pass

    return None


def poll_open_interest():
    """Poll open interest for all tracked symbols."""
    global _latest_oi

    timestamp = datetime.now().isoformat()

    for symbol in TRACKED_SYMBOLS:
        oi = fetch_open_interest(symbol)
        price = fetch_price(symbol)

        # Store price even if OI isn't available (useful for snapshot)
        if price is not None:
            # If OI is available, calculate full record
            if oi is not None:
                value_usd = oi * price

                record = {
                    'symbol': symbol,
                    'open_interest': oi,
                    'price': price,
                    'value_usd': value_usd,
                    'timestamp': timestamp,
                }

                # Store to CSV
                append_oi_record(record)

                # Log
                symbol_short = symbol.replace('USDT', '')
                cprint(
                    f"  [OI] {symbol_short}: {oi:,.2f} contracts × ${price:,.2f} = ${value_usd:,.0f}",
                    "cyan"
                )
            else:
                # OI not available (geo-restricted), just store price
                record = {
                    'symbol': symbol,
                    'open_interest': 0,
                    'price': price,
                    'value_usd': 0,
                    'timestamp': timestamp,
                }

                # Log price only
                symbol_short = symbol.replace('USDT', '')
                cprint(
                    f"  [Price] {symbol_short}: ${price:,.2f} (OI unavailable - geo-restricted)",
                    "yellow"
                )

            # Update cache
            with _cache_lock:
                _latest_oi[symbol] = record


def run_oi_tracker():
    """Run the Open Interest tracker loop."""
    global _oi_running

    cprint("\n[OI Tracker] Starting Open Interest polling...", "green")
    cprint(f"[OI Tracker] Polling interval: {OI_POLL_INTERVAL}s", "white")
    cprint(f"[OI Tracker] Tracking: {', '.join(TRACKED_SYMBOLS)}", "white")

    while _oi_running:
        try:
            poll_open_interest()
        except Exception as e:
            cprint(f"[OI Tracker] Error: {e}", "red")

        # Wait for next poll
        for _ in range(OI_POLL_INTERVAL):
            if not _oi_running:
                break
            time.sleep(1)

    cprint("[OI Tracker] Stopped", "yellow")


# ─────────────────────────────────────────────────────────────
# PART 2: FUNDING RATE TRACKER
# ─────────────────────────────────────────────────────────────

def parse_funding_message(msg: Dict) -> Optional[Dict]:
    """
    Parse Binance markPrice message for funding rate and price.

    Message format:
    {
        "e": "markPriceUpdate",     // Event type
        "E": 1562305380000,         // Event time
        "s": "BTCUSDT",             // Symbol
        "p": "11794.15000000",      // Mark price
        "i": "11784.62659091",      // Index price
        "P": "11784.25641265",      // Estimated Settle Price
        "r": "0.00038167",          // Funding rate
        "T": 1562306400000          // Next funding time
    }
    """
    global _latest_oi

    try:
        symbol = msg.get('s', '')

        # Only process tracked symbols
        if symbol not in TRACKED_SYMBOLS:
            return None

        funding_rate = float(msg.get('r', 0))
        mark_price = float(msg.get('p', 0))

        # Calculate yearly rate: funding_rate * 3 (times per day) * 365 (days) * 100 (percent)
        yearly_rate = funding_rate * 3 * 365 * 100

        timestamp = datetime.now().isoformat()

        # Also update price in OI cache if we don't have it
        with _cache_lock:
            if symbol not in _latest_oi or _latest_oi[symbol].get('price', 0) == 0:
                _latest_oi[symbol] = {
                    'symbol': symbol,
                    'open_interest': 0,
                    'price': mark_price,
                    'value_usd': 0,
                    'timestamp': timestamp,
                }

        return {
            'symbol': symbol,
            'funding_rate': funding_rate,
            'yearly_rate': yearly_rate,
            'mark_price': mark_price,
            'timestamp': timestamp,
        }

    except Exception as e:
        cprint(f"Error parsing funding message: {e}", "red")
        return None


async def connect_funding_websocket():
    """
    Connect to Binance funding rate websocket with robust reconnection.

    Uses the same outer/inner loop pattern as liquidation_tracker.
    """
    global _funding_running, _latest_funding

    reconnect_delay = RECONNECT_DELAY_BASE
    connection_count = 0

    cprint("\n[Funding Tracker] Starting funding rate stream...", "green")
    cprint(f"[Funding Tracker] Websocket: {BINANCE_MARK_PRICE_WS}", "white")
    cprint(f"[Funding Tracker] Tracking: {', '.join(TRACKED_SYMBOLS)}", "white")

    # Track last update time to avoid spamming CSV
    last_update: Dict[str, datetime] = {}
    update_interval = timedelta(seconds=10)  # Only store every 10 seconds

    # ─── OUTER LOOP: Reconnection handling ───
    while _funding_running:
        connection_count += 1

        try:
            cprint(f"[Funding Tracker] Connecting... (attempt #{connection_count})", "yellow")

            async with websockets.connect(
                BINANCE_MARK_PRICE_WS,
                ping_interval=PING_INTERVAL,
                ping_timeout=PING_TIMEOUT,
                close_timeout=10,
            ) as ws:

                cprint(f"[Funding Tracker] Connected to Binance", "green")
                reconnect_delay = RECONNECT_DELAY_BASE

                # ─── INNER LOOP: Message processing ───
                while _funding_running:
                    try:
                        msg_raw = await asyncio.wait_for(ws.recv(), timeout=60)
                        messages = json.loads(msg_raw)

                        # markPrice@arr returns an array
                        if not isinstance(messages, list):
                            messages = [messages]

                        for msg in messages:
                            record = parse_funding_message(msg)

                            if record:
                                symbol = record['symbol']
                                now = datetime.now()

                                # Update cache always
                                with _cache_lock:
                                    _latest_funding[symbol] = record

                                # Only write to CSV periodically
                                should_store = (
                                    symbol not in last_update or
                                    now - last_update[symbol] >= update_interval
                                )

                                if should_store:
                                    append_funding_record(record)
                                    last_update[symbol] = now

                                    # Log
                                    symbol_short = symbol.replace('USDT', '')
                                    rate_color = "green" if record['yearly_rate'] < 20 else "yellow" if record['yearly_rate'] < 50 else "red"
                                    cprint(
                                        f"  [Funding] {symbol_short}: {record['funding_rate']:.6f} "
                                        f"({record['yearly_rate']:+.2f}% yearly)",
                                        rate_color
                                    )

                    except asyncio.TimeoutError:
                        continue

                    except websockets.exceptions.ConnectionClosed as e:
                        cprint(f"[Funding Tracker] Connection closed: {e}", "yellow")
                        break

                    except json.JSONDecodeError as e:
                        cprint(f"[Funding Tracker] JSON error: {e}", "red")
                        continue

                    except Exception as e:
                        cprint(f"[Funding Tracker] Message error: {e}", "red")
                        continue

        except websockets.exceptions.InvalidStatusCode as e:
            cprint(f"[Funding Tracker] Invalid status: {e}", "red")

        except websockets.exceptions.WebSocketException as e:
            cprint(f"[Funding Tracker] WebSocket error: {e}", "red")

        except ConnectionRefusedError:
            cprint(f"[Funding Tracker] Connection refused", "red")

        except Exception as e:
            cprint(f"[Funding Tracker] Unexpected error: {e}", "red")

        # ─── Reconnection delay with exponential backoff ───
        if _funding_running:
            cprint(f"[Funding Tracker] Reconnecting in {reconnect_delay}s...", "yellow")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * RECONNECT_DELAY_MULTIPLIER, RECONNECT_DELAY_MAX)

    cprint("[Funding Tracker] Stopped", "yellow")


def _run_funding_loop():
    """Run the funding tracker in an asyncio event loop (for threading)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(connect_funding_websocket())
    finally:
        loop.close()


# ─────────────────────────────────────────────────────────────
# PART 3: COMBINED RUNNER & PUBLIC API
# ─────────────────────────────────────────────────────────────

def start_oi_tracker(background: bool = True) -> Optional[threading.Thread]:
    """Start the Open Interest tracker."""
    global _oi_thread, _oi_running

    if _oi_running:
        cprint("OI Tracker already running", "yellow")
        return _oi_thread

    _oi_running = True
    ensure_oi_csv_exists()

    if background:
        _oi_thread = threading.Thread(target=run_oi_tracker, daemon=True)
        _oi_thread.start()
        return _oi_thread
    else:
        run_oi_tracker()
        return None


def start_funding_tracker(background: bool = True) -> Optional[threading.Thread]:
    """Start the Funding Rate tracker."""
    global _funding_thread, _funding_running

    if not WEBSOCKETS_AVAILABLE:
        cprint("Cannot start funding tracker: websockets not installed", "red")
        return None

    if _funding_running:
        cprint("Funding Tracker already running", "yellow")
        return _funding_thread

    _funding_running = True
    ensure_funding_csv_exists()

    if background:
        _funding_thread = threading.Thread(target=_run_funding_loop, daemon=True)
        _funding_thread.start()
        return _funding_thread
    else:
        asyncio.run(connect_funding_websocket())
        return None


def start_market_indicators(background: bool = True) -> Tuple[Optional[threading.Thread], Optional[threading.Thread]]:
    """
    Start both Open Interest and Funding Rate trackers.

    Args:
        background: If True, runs in background threads

    Returns:
        Tuple of (oi_thread, funding_thread)
    """
    cprint("\n" + "=" * 60, "magenta")
    cprint("  MARKET INDICATORS", "magenta", attrs=['bold'])
    cprint("  Open Interest + Funding Rate Tracker", "magenta")
    cprint("=" * 60, "magenta")

    oi_thread = start_oi_tracker(background=background)
    funding_thread = start_funding_tracker(background=background)

    if background:
        cprint("\nMarket indicators started in background", "green")

    return oi_thread, funding_thread


def stop_oi_tracker():
    """Stop the Open Interest tracker."""
    global _oi_running
    _oi_running = False


def stop_funding_tracker():
    """Stop the Funding Rate tracker."""
    global _funding_running
    _funding_running = False


def stop_market_indicators():
    """Stop all market indicator trackers."""
    stop_oi_tracker()
    stop_funding_tracker()
    cprint("Stopping market indicators...", "yellow")


def get_total_open_interest() -> Dict:
    """
    Get total open interest for BTC + ETH.

    Returns:
        Dict with:
        - total_usd: Combined USD value
        - btc_usd: BTC open interest in USD
        - eth_usd: ETH open interest in USD
        - btc_contracts: BTC open interest in contracts
        - eth_contracts: ETH open interest in contracts
        - timestamp: Latest update timestamp
    """
    with _cache_lock:
        btc = _latest_oi.get('BTCUSDT', {})
        eth = _latest_oi.get('ETHUSDT', {})

    btc_usd = btc.get('value_usd', 0)
    eth_usd = eth.get('value_usd', 0)

    return {
        'total_usd': btc_usd + eth_usd,
        'btc_usd': btc_usd,
        'eth_usd': eth_usd,
        'btc_contracts': btc.get('open_interest', 0),
        'eth_contracts': eth.get('open_interest', 0),
        'btc_price': btc.get('price', 0),
        'eth_price': eth.get('price', 0),
        'timestamp': btc.get('timestamp') or eth.get('timestamp') or datetime.now().isoformat(),
    }


def get_funding_rates() -> Dict:
    """
    Get current funding rates for BTC and ETH.

    Returns:
        Dict with:
        - btc_rate: BTC funding rate (raw)
        - btc_yearly: BTC yearly rate (%)
        - eth_rate: ETH funding rate (raw)
        - eth_yearly: ETH yearly rate (%)
        - avg_yearly: Average yearly rate
        - timestamp: Latest update timestamp
    """
    with _cache_lock:
        btc = _latest_funding.get('BTCUSDT', {})
        eth = _latest_funding.get('ETHUSDT', {})

    btc_yearly = btc.get('yearly_rate', 0)
    eth_yearly = eth.get('yearly_rate', 0)

    # Calculate average (handle case where one might be missing)
    rates = [r for r in [btc_yearly, eth_yearly] if r != 0]
    avg_yearly = sum(rates) / len(rates) if rates else 0

    return {
        'btc_rate': btc.get('funding_rate', 0),
        'btc_yearly': btc_yearly,
        'eth_rate': eth.get('funding_rate', 0),
        'eth_yearly': eth_yearly,
        'avg_yearly': avg_yearly,
        'timestamp': btc.get('timestamp') or eth.get('timestamp') or datetime.now().isoformat(),
    }


def get_funding_sentiment(avg_yearly: float) -> str:
    """
    Determine funding sentiment from average yearly rate.

    Args:
        avg_yearly: Average yearly funding rate percentage

    Returns:
        Sentiment string: EXTREME_GREED, GREED, NEUTRAL, FEAR, or EXTREME_FEAR
    """
    if avg_yearly > SENTIMENT_THRESHOLDS['EXTREME_GREED']:
        return 'EXTREME_GREED'
    elif avg_yearly > SENTIMENT_THRESHOLDS['GREED']:
        return 'GREED'
    elif avg_yearly > SENTIMENT_THRESHOLDS['NEUTRAL_LOW']:
        return 'NEUTRAL'
    elif avg_yearly > SENTIMENT_THRESHOLDS['FEAR']:
        return 'FEAR'
    else:
        return 'EXTREME_FEAR'


def get_market_snapshot() -> Dict:
    """
    Get a complete market snapshot with OI and funding data.

    Returns:
        Dict with:
        - total_open_interest: Total OI in USD (BTC + ETH)
        - btc_funding_yearly: BTC yearly funding rate (%)
        - eth_funding_yearly: ETH yearly funding rate (%)
        - funding_sentiment: Sentiment string based on avg funding
        - btc_price: Current BTC price
        - eth_price: Current ETH price
        - timestamp: Snapshot timestamp
    """
    oi = get_total_open_interest()
    funding = get_funding_rates()

    sentiment = get_funding_sentiment(funding['avg_yearly'])

    return {
        'total_open_interest': oi['total_usd'],
        'btc_open_interest': oi['btc_usd'],
        'eth_open_interest': oi['eth_usd'],
        'btc_funding_yearly': funding['btc_yearly'],
        'eth_funding_yearly': funding['eth_yearly'],
        'avg_funding_yearly': funding['avg_yearly'],
        'funding_sentiment': sentiment,
        'btc_price': oi['btc_price'],
        'eth_price': oi['eth_price'],
        'timestamp': datetime.now().isoformat(),
    }


def print_market_snapshot():
    """Print a formatted market snapshot."""
    snapshot = get_market_snapshot()

    cprint(f"\n{'─'*60}", "magenta")
    cprint(f"  MARKET SNAPSHOT", "magenta", attrs=['bold'])
    cprint(f"{'─'*60}", "magenta")

    # Open Interest
    cprint(f"\n  OPEN INTEREST:", "cyan")
    cprint(f"    BTC:   ${snapshot['btc_open_interest']:>15,.0f}", "white")
    cprint(f"    ETH:   ${snapshot['eth_open_interest']:>15,.0f}", "white")
    cprint(f"    Total: ${snapshot['total_open_interest']:>15,.0f}", "cyan")

    # Funding Rates
    cprint(f"\n  FUNDING RATES (Yearly):", "cyan")

    btc_color = "green" if snapshot['btc_funding_yearly'] < 20 else "yellow" if snapshot['btc_funding_yearly'] < 50 else "red"
    eth_color = "green" if snapshot['eth_funding_yearly'] < 20 else "yellow" if snapshot['eth_funding_yearly'] < 50 else "red"

    cprint(f"    BTC:   {snapshot['btc_funding_yearly']:>+15.2f}%", btc_color)
    cprint(f"    ETH:   {snapshot['eth_funding_yearly']:>+15.2f}%", eth_color)
    cprint(f"    Avg:   {snapshot['avg_funding_yearly']:>+15.2f}%", "cyan")

    # Sentiment
    sentiment = snapshot['funding_sentiment']
    sentiment_colors = {
        'EXTREME_GREED': 'red',
        'GREED': 'yellow',
        'NEUTRAL': 'white',
        'FEAR': 'cyan',
        'EXTREME_FEAR': 'blue',
    }
    cprint(f"\n  FUNDING SENTIMENT: {sentiment}", sentiment_colors.get(sentiment, 'white'), attrs=['bold'])

    # Prices
    cprint(f"\n  PRICES:", "cyan")
    cprint(f"    BTC:   ${snapshot['btc_price']:>15,.2f}", "white")
    cprint(f"    ETH:   ${snapshot['eth_price']:>15,.2f}", "white")

    cprint(f"\n{'─'*60}\n", "magenta")


def load_oi_history(hours: int = 24) -> pd.DataFrame:
    """Load open interest history from CSV."""
    ensure_oi_csv_exists()
    try:
        df = pd.read_csv(OI_CSV_PATH)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            since = datetime.now() - timedelta(hours=hours)
            df = df[df['timestamp'] >= since]
        return df
    except Exception as e:
        cprint(f"Error loading OI history: {e}", "red")
        return pd.DataFrame(columns=OI_COLUMNS)


def load_funding_history(hours: int = 24) -> pd.DataFrame:
    """Load funding rate history from CSV."""
    ensure_funding_csv_exists()
    try:
        df = pd.read_csv(FUNDING_CSV_PATH)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            since = datetime.now() - timedelta(hours=hours)
            df = df[df['timestamp'] >= since]
        return df
    except Exception as e:
        cprint(f"Error loading funding history: {e}", "red")
        return pd.DataFrame(columns=FUNDING_COLUMNS)


# ─────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────

def main():
    """Main entry point for standalone execution."""
    parser = argparse.ArgumentParser(description="Market Indicators Tracker")
    parser.add_argument("--snapshot", action="store_true", help="Show current snapshot and exit")
    parser.add_argument("--oi-only", action="store_true", help="Only track Open Interest")
    parser.add_argument("--funding-only", action="store_true", help="Only track Funding Rates")
    parser.add_argument("--duration", type=int, default=None, help="Run for N seconds then exit")
    args = parser.parse_args()

    if args.snapshot:
        # Just show snapshot from cached/CSV data
        print_market_snapshot()
        return

    try:
        if args.oi_only:
            start_oi_tracker(background=True)
        elif args.funding_only:
            start_funding_tracker(background=True)
        else:
            start_market_indicators(background=True)

        cprint("\nTrackers running. Press Ctrl+C to stop.", "yellow")

        if args.duration:
            # Run for specified duration
            start_time = time.time()
            while time.time() - start_time < args.duration:
                time.sleep(10)
                print_market_snapshot()
        else:
            # Run indefinitely with periodic snapshots
            while True:
                time.sleep(30)
                print_market_snapshot()

    except KeyboardInterrupt:
        cprint("\nStopping...", "yellow")
    finally:
        stop_market_indicators()


if __name__ == "__main__":
    main()

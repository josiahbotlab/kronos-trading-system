#!/usr/bin/env python3
"""
Kronos Coinbase Advanced Trade Connector
==========================================
Interface to Coinbase Advanced Trade API for price feeds and order execution.

Supports:
- Real-time price fetching
- Paper trading (simulated fills)
- Live trading (Coinbase Advanced Trade API)
- Order book data
- Historical candles

Uses Coinbase Advanced Trade API v3 with JWT authentication.
Requires COINBASE_API_KEY and COINBASE_API_SECRET environment variables.
"""

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import ccxt
except ImportError:
    ccxt = None

log = logging.getLogger("coinbase")

COINBASE_API_BASE = "https://api.coinbase.com"
COINBASE_EXCHANGE_API = "https://api.exchange.coinbase.com"  # Public API (Coinbase Pro)

# Map our symbols to Coinbase product IDs
SYMBOL_MAP = {
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "SOL-USD": "SOL-USD",
    "DOGE-USD": "DOGE-USD",
    "XRP-USD": "XRP-USD",
    "ADA-USD": "ADA-USD",
    "AVAX-USD": "AVAX-USD",
    "LINK-USD": "LINK-USD",
    "DOT-USD": "DOT-USD",
    "ARB-USD": "ARB-USD",
    "OP-USD": "OP-USD",
    "SUI-USD": "SUI-USD",
    "APT-USD": "APT-USD",
    "NEAR-USD": "NEAR-USD",
    "FIL-USD": "FIL-USD",
    "ATOM-USD": "ATOM-USD",
}


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: str = ""
    fill_price: float = 0.0
    fill_quantity: float = 0.0
    fee: float = 0.0
    error: str = ""
    timestamp: float = 0.0


class CoinbaseConnector:
    """
    Coinbase Advanced Trade API interface supporting paper and live trading.

    Paper mode: Simulates fills at current market price + slippage
    Live mode: Executes on Coinbase Advanced Trade

    Authentication uses JWT signing with API key + secret.
    """

    def __init__(
        self,
        paper: bool = True,
        slippage_bps: float = 5.0,  # 5 basis points default slippage
        fee_rate: float = 0.006,    # 0.6% taker fee (Coinbase Advanced Trade)
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        self.paper = paper
        self.slippage_bps = slippage_bps
        self.fee_rate = fee_rate
        self._price_cache: dict[str, tuple[float, float]] = {}  # symbol -> (price, timestamp)
        self._cache_ttl = 2.0  # seconds

        # Load credentials from env if not provided
        self.api_key = api_key or os.getenv("COINBASE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET", "")

        if not self.paper and (not self.api_key or not self.api_secret):
            raise ValueError("Coinbase API key and secret required for live trading")

        # Initialize CCXT for data fetching (handles rate limits, retries, etc.)
        if ccxt:
            self._ccxt_exchange = ccxt.coinbase({
                "enableRateLimit": True,
            })
            if not self.paper and self.api_key and self.api_secret:
                self._ccxt_exchange.apiKey = self.api_key
                self._ccxt_exchange.secret = self.api_secret
        else:
            self._ccxt_exchange = None
            log.warning("CCXT not installed, some features may be limited")

        # Paper trade log
        self.paper_db_path = Path(__file__).parent.parent / "data" / "paper_trades.db"
        if self.paper:
            self._init_paper_db()

    def _init_paper_db(self):
        """Initialize SQLite database for paper trade logging."""
        self.paper_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.paper_db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                order_id TEXT,
                symbol TEXT,
                side TEXT,
                price REAL,
                quantity REAL,
                notional REAL,
                fee REAL
            )
        """)
        conn.commit()
        conn.close()

    def _generate_jwt(self, method: str, path: str, body: str = "") -> str:
        """Generate JWT token for Coinbase Advanced Trade API authentication."""
        timestamp = str(int(time.time()))

        # Create message to sign: timestamp + method + path + body
        message = timestamp + method.upper() + path + body

        # Sign with HMAC-SHA256
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        return signature

    def _api_request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        body: dict = None,
    ) -> dict:
        """Make an authenticated request to Coinbase Advanced Trade API."""
        url = f"{COINBASE_API_BASE}{endpoint}"

        # Add query params
        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url += f"?{query}"

        # Prepare body
        body_str = ""
        if body:
            body_str = json.dumps(body)

        # Generate signature
        timestamp = str(int(time.time()))
        path = endpoint
        if params:
            path += "?" + "&".join(f"{k}={v}" for k, v in params.items())

        signature = self._generate_jwt(method, path, body_str)

        # Create request
        headers = {
            "Content-Type": "application/json",
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
        }

        data = body_str.encode() if body_str else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            error_body = e.read().decode()
            log.error(f"Coinbase API error {e.code}: {error_body}")
            raise
        except Exception as e:
            log.error(f"Request failed: {e}")
            raise

    def _public_request(self, endpoint: str, params: dict = None) -> dict:
        """Make a public (unauthenticated) request to Coinbase.

        Note: Coinbase Advanced Trade API may require authentication even for
        public data. This method will try public access first, and fall back
        to a simple implementation that returns mock data for paper trading.
        """
        url = f"{COINBASE_API_BASE}{endpoint}"

        if params:
            query = "&".join(f"{k}={v}" for k, v in params.items())
            url += f"?{query}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            # If unauthorized and we're in paper mode, we can use alternative data sources
            if e.code == 401:
                log.debug(f"Public API requires auth, will need to use authenticated request")
                raise
            raise

    # --- Price Data ---

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current mid price for a symbol."""
        # Check cache
        if symbol in self._price_cache:
            price, ts = self._price_cache[symbol]
            if time.time() - ts < self._cache_ttl:
                return price

        # Fetch fresh using CCXT
        if self._ccxt_exchange:
            try:
                # CCXT uses "BTC/USD" format
                ccxt_symbol = symbol.replace("-", "/")
                ticker = self._ccxt_exchange.fetch_ticker(ccxt_symbol)
                if ticker and "last" in ticker:
                    price = float(ticker["last"])
                    self._price_cache[symbol] = (price, time.time())
                    return price
            except Exception as e:
                log.error(f"Failed to fetch price for {symbol}: {e}")
                return None

        return None

    def get_all_prices(self) -> dict[str, float]:
        """Fetch current prices for all configured symbols."""
        prices = {}
        for symbol in SYMBOL_MAP.keys():
            price = self.get_price(symbol)
            if price:
                prices[symbol] = price
        return prices

    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[dict]:
        """Get order book for a symbol."""
        if self._ccxt_exchange:
            try:
                # CCXT uses "BTC/USD" format
                ccxt_symbol = symbol.replace("-", "/")
                orderbook = self._ccxt_exchange.fetch_order_book(ccxt_symbol, limit=depth)

                bids = orderbook.get("bids", [])[:depth]
                asks = orderbook.get("asks", [])[:depth]

                if bids and asks:
                    bids_formatted = [{"price": float(b[0]), "size": float(b[1])} for b in bids]
                    asks_formatted = [{"price": float(a[0]), "size": float(a[1])} for a in asks]

                    return {
                        "bids": bids_formatted,
                        "asks": asks_formatted,
                        "spread": asks[0][0] - bids[0][0],
                        "mid": (asks[0][0] + bids[0][0]) / 2,
                    }
            except Exception as e:
                log.error(f"Failed to fetch orderbook for {symbol}: {e}")
                return None

        return None

    def get_candles(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 100
    ) -> list[dict]:
        """Get historical candles for a symbol using CCXT."""
        if self._ccxt_exchange:
            try:
                # CCXT uses "BTC/USD" format
                ccxt_symbol = symbol.replace("-", "/")
                ohlcv = self._ccxt_exchange.fetch_ohlcv(ccxt_symbol, timeframe, limit=limit)

                candles = []
                for c in ohlcv:
                    # CCXT returns: [timestamp_ms, open, high, low, close, volume]
                    candles.append({
                        "timestamp": int(c[0] / 1000),  # Convert to seconds
                        "open": float(c[1]),
                        "high": float(c[2]),
                        "low": float(c[3]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    })

                return candles
            except Exception as e:
                log.error(f"Failed to fetch candles for {symbol}: {e}")
                return []

        return []

    # --- Order Execution ---

    def place_market_order(
        self,
        symbol: str,
        side: str,
        size_usd: float
    ) -> OrderResult:
        """
        Place a market order.

        Args:
            symbol: Trading pair (e.g., "BTC-USD")
            side: "buy" or "sell"
            size_usd: Order size in USD
        """
        if self.paper:
            return self._paper_fill(symbol, side, size_usd)

        product_id = SYMBOL_MAP.get(symbol, symbol)

        try:
            # Get current price to calculate quantity
            price = self.get_price(symbol)
            if not price:
                return OrderResult(success=False, error=f"No price for {symbol}")

            quantity = size_usd / price

            # Place market order
            body = {
                "product_id": product_id,
                "side": side.upper(),
                "order_configuration": {
                    "market_market_ioc": {
                        "quote_size": str(size_usd) if side.lower() == "buy" else None,
                        "base_size": str(quantity) if side.lower() == "sell" else None,
                    }
                }
            }

            result = self._api_request("POST", "/api/v3/brokerage/orders", body=body)

            if result and result.get("success"):
                order_id = result.get("order_id", "")
                # Get order details
                order_details = self._api_request(
                    "GET",
                    f"/api/v3/brokerage/orders/historical/{order_id}"
                )

                fill_price = float(order_details.get("average_filled_price", price))
                fill_quantity = float(order_details.get("filled_size", quantity))
                fee = float(order_details.get("total_fees", 0))

                log.info(
                    f"[LIVE] {side.upper()} {symbol} | "
                    f"Price: {fill_price:.2f} | "
                    f"Qty: {fill_quantity:.6f} | Fee: ${fee:.4f}"
                )

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    fill_price=fill_price,
                    fill_quantity=fill_quantity,
                    fee=fee,
                    timestamp=time.time(),
                )
            else:
                error = result.get("error_response", {}).get("message", "Unknown error")
                return OrderResult(success=False, error=error)

        except Exception as e:
            log.error(f"Market order failed: {e}")
            return OrderResult(success=False, error=str(e))

    def place_limit_order(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float
    ) -> OrderResult:
        """Place a limit order."""
        if self.paper:
            # For paper mode, treat limit orders as market orders for simplicity
            return self._paper_fill(symbol, side, size * price)

        product_id = SYMBOL_MAP.get(symbol, symbol)

        try:
            body = {
                "product_id": product_id,
                "side": side.upper(),
                "order_configuration": {
                    "limit_limit_gtc": {
                        "base_size": str(size),
                        "limit_price": str(price),
                        "post_only": False,
                    }
                }
            }

            result = self._api_request("POST", "/api/v3/brokerage/orders", body=body)

            if result and result.get("success"):
                order_id = result.get("order_id", "")

                return OrderResult(
                    success=True,
                    order_id=order_id,
                    fill_price=price,
                    fill_quantity=size,
                    fee=0.0,  # Won't know until filled
                    timestamp=time.time(),
                )
            else:
                error = result.get("error_response", {}).get("message", "Unknown error")
                return OrderResult(success=False, error=error)

        except Exception as e:
            log.error(f"Limit order failed: {e}")
            return OrderResult(success=False, error=str(e))

    def _paper_fill(self, symbol: str, side: str, size_usd: float) -> OrderResult:
        """Simulate a market order fill with slippage."""
        price = self.get_price(symbol)
        if price is None:
            return OrderResult(success=False, error=f"No price available for {symbol}")

        # Apply slippage
        slip = price * (self.slippage_bps / 10000)
        if side == "buy":
            fill_price = price + slip
        else:
            fill_price = price - slip

        # Calculate quantity
        quantity = size_usd / fill_price

        # Calculate fee
        notional = fill_price * quantity
        fee = notional * self.fee_rate

        order_id = f"paper_{int(time.time()*1000)}"

        # Log to database
        try:
            conn = sqlite3.connect(str(self.paper_db_path))
            conn.execute(
                """INSERT INTO paper_trades
                   (timestamp, order_id, symbol, side, price, quantity, notional, fee)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (time.time(), order_id, symbol, side, fill_price, quantity, notional, fee)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning(f"Failed to log paper trade: {e}")

        log.info(
            f"[PAPER] {side.upper()} {symbol} | "
            f"Price: {fill_price:.2f} (slip: {slip:.2f}) | "
            f"Qty: {quantity:.6f} | Fee: ${fee:.4f}"
        )

        return OrderResult(
            success=True,
            order_id=order_id,
            fill_price=fill_price,
            fill_quantity=quantity,
            fee=fee,
            timestamp=time.time(),
        )

    # --- Order Management ---

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.paper:
            log.info(f"[PAPER] Cancel order {order_id}")
            return True

        try:
            result = self._api_request(
                "POST",
                "/api/v3/brokerage/orders/batch_cancel",
                body={"order_ids": [order_id]}
            )

            if result and result.get("results"):
                success = result["results"][0].get("success", False)
                return success
            return False
        except Exception as e:
            log.error(f"Cancel order failed: {e}")
            return False

    def get_open_orders(self, symbol: Optional[str] = None) -> list[dict]:
        """Get list of open orders."""
        if self.paper:
            return []  # No open orders in paper mode (market orders instant fill)

        try:
            params = {}
            if symbol:
                params["product_id"] = SYMBOL_MAP.get(symbol, symbol)

            result = self._api_request("GET", "/api/v3/brokerage/orders", params=params)

            if result and "orders" in result:
                return result["orders"]
            return []
        except Exception as e:
            log.error(f"Failed to get open orders: {e}")
            return []

    # --- Account Info ---

    def get_account_balance(self) -> dict[str, float]:
        """Get account balances for all assets."""
        if self.paper:
            return {"USD": 1000.0}  # Paper mode default balance

        try:
            result = self._api_request("GET", "/api/v3/brokerage/accounts")

            balances = {}
            if result and "accounts" in result:
                for account in result["accounts"]:
                    currency = account.get("currency", "")
                    available = float(account.get("available_balance", {}).get("value", 0))
                    if available > 0:
                        balances[currency] = available

            return balances
        except Exception as e:
            log.error(f"Failed to get account balance: {e}")
            return {}

    def health_check(self) -> dict:
        """Verify connectivity and return account summary."""
        try:
            if self.paper:
                # Test public API
                btc_price = self.get_price("BTC-USD")
                return {
                    "status": "healthy",
                    "mode": "paper",
                    "btc_price": btc_price,
                    "balance_usd": 1000.0,
                }

            # Test authenticated API
            balances = self.get_account_balance()
            btc_price = self.get_price("BTC-USD")

            return {
                "status": "healthy",
                "mode": "live",
                "btc_price": btc_price,
                "balances": balances,
                "balance_usd": balances.get("USD", 0),
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
            }

    # --- Compatibility Methods (for drop-in replacement) ---

    def market_buy(self, symbol: str, quantity: float) -> OrderResult:
        """Execute a market buy order (compatibility with old interface)."""
        price = self.get_price(symbol)
        if not price:
            return OrderResult(success=False, error=f"No price for {symbol}")
        size_usd = quantity * price
        return self.place_market_order(symbol, "buy", size_usd)

    def market_sell(self, symbol: str, quantity: float) -> OrderResult:
        """Execute a market sell order (compatibility with old interface)."""
        price = self.get_price(symbol)
        if not price:
            return OrderResult(success=False, error=f"No price for {symbol}")
        size_usd = quantity * price
        return self.place_market_order(symbol, "sell", size_usd)

    def get_orderbook(self, symbol: str, depth: int = 5) -> Optional[dict]:
        """Alias for get_order_book (compatibility)."""
        return self.get_order_book(symbol, depth)

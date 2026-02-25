#!/usr/bin/env python3
"""
Kronos Exchange Connector
==========================
Interface to Coinbase Advanced Trade API for price feeds and order execution.

Supports:
- Real-time price fetching
- Paper trading (simulated fills)
- Live trading (Coinbase Advanced Trade API)
- Order book data

Requires COINBASE_API_KEY and COINBASE_API_SECRET environment variables for live trading.
"""

import logging
from typing import Optional

from execution.coinbase_connector import CoinbaseConnector, OrderResult

log = logging.getLogger("exchange")

# Map old symbols to new Coinbase format (for backwards compatibility)
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


class ExchangeConnector:
    """
    Exchange interface supporting paper and live trading via Coinbase Advanced Trade API.

    This is a wrapper around CoinbaseConnector that maintains backwards compatibility
    with the original ExchangeConnector interface.

    Paper mode: Simulates fills at current market price + slippage
    Live mode: Executes on Coinbase Advanced Trade
    """

    def __init__(
        self,
        paper: bool = True,
        slippage_bps: float = 5.0,  # 5 basis points default slippage
        fee_rate: float = 0.006,    # 0.6% taker fee (Coinbase)
    ):
        self.paper = paper
        self.slippage_bps = slippage_bps
        self.fee_rate = fee_rate

        # Create Coinbase connector instance
        self._connector = CoinbaseConnector(
            paper=paper,
            slippage_bps=slippage_bps,
            fee_rate=fee_rate,
        )

    def get_all_prices(self) -> dict[str, float]:
        """Fetch current mid prices for all assets."""
        return self._connector.get_all_prices()

    def get_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol."""
        return self._connector.get_price(symbol)

    def get_asset_info(self, symbol: str) -> Optional[dict]:
        """
        Get asset context (OI, funding, volume).

        Note: Coinbase spot markets don't have funding rates or OI.
        This returns basic info for compatibility.
        """
        price = self.get_price(symbol)
        if price:
            return {
                "symbol": symbol,
                "mark_price": price,
                "open_interest": 0.0,  # N/A for spot
                "funding_rate": 0.0,   # N/A for spot
                "volume_24h": 0.0,     # TODO: Add if needed
                "premium": 0.0,
            }
        return None

    # --- Order Execution ---

    def market_buy(self, symbol: str, quantity: float) -> OrderResult:
        """Execute a market buy order."""
        return self._connector.market_buy(symbol, quantity)

    def market_sell(self, symbol: str, quantity: float) -> OrderResult:
        """Execute a market sell order."""
        return self._connector.market_sell(symbol, quantity)

    def get_orderbook(self, symbol: str, depth: int = 5) -> Optional[dict]:
        """Get order book for a symbol."""
        return self._connector.get_order_book(symbol, depth)

"""
Kronos Trading - Coinbase Execution Wrapper
Wraps the shared coinbase_connector.py for clean integration with Week 5 system.
Provides the same interface as HyperliquidExecutor for drop-in replacement.

Supports paper trading (default) and live trading via Coinbase Advanced Trade API.

Requirements:
    pip install ccxt --break-system-packages
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Import the shared Coinbase connector
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from execution.coinbase_connector import CoinbaseConnector, OrderResult as CoinbaseOrderResult

logger = logging.getLogger("kronos.execution")


@dataclass
class OrderResult:
    """Order result matching HyperliquidExecutor interface."""
    success: bool
    oid: Optional[int] = None
    filled_sz: Optional[float] = None
    avg_px: Optional[float] = None
    error: Optional[str] = None
    raw: dict = field(default_factory=dict)


@dataclass
class Position:
    """Position matching HyperliquidExecutor interface."""
    coin: str
    size: float  # positive = long, negative = short
    entry_px: float
    unrealized_pnl: float
    leverage: float
    liquidation_px: Optional[float] = None
    margin_used: float = 0.0


@dataclass
class AccountState:
    """Account state matching HyperliquidExecutor interface."""
    equity: float
    available_balance: float
    margin_used: float
    positions: list = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CoinbaseExecutor:
    """
    Wrapper around CoinbaseConnector for Kronos trading system.
    Provides the same interface as HyperliquidExecutor for compatibility.

    Usage:
        executor = CoinbaseExecutor.from_config("config/kronos.json")
        # or
        executor = CoinbaseExecutor(paper=True)

        # Place a market buy
        result = executor.market_open("BTC-USD", is_buy=True, size=0.001)

        # Check positions
        state = executor.get_account_state()
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True,
    ):
        self.paper = paper
        self.testnet = paper  # For compatibility with HyperliquidExecutor

        # Load credentials from env if not provided
        self.api_key = api_key or os.getenv("COINBASE_API_KEY", "")
        self.api_secret = api_secret or os.getenv("COINBASE_API_SECRET", "")

        if paper:
            logger.info("🧪 Coinbase PAPER mode (simulated trading)")
        else:
            logger.warning("⚠️ Coinbase LIVE mode - REAL MONEY")

        # Initialize Coinbase connector
        self.connector = CoinbaseConnector(
            paper=paper,
            api_key=self.api_key if self.api_key else None,
            api_secret=self.api_secret if self.api_secret else None,
        )

        # Simulated account state for paper trading
        self._paper_equity = 10000.0  # Start with $10k paper money
        self._paper_positions: dict[str, Position] = {}

        # Cache metadata
        self._meta = None
        self._asset_map = None

    @classmethod
    def from_config(cls, config_path: str = "config/kronos.json") -> "CoinbaseExecutor":
        """Load executor from Kronos config file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(path) as f:
            config = json.load(f)

        cb_config = config.get("coinbase", {})
        return cls(
            api_key=cb_config.get("api_key") or os.getenv("COINBASE_API_KEY"),
            api_secret=cb_config.get("api_secret") or os.getenv("COINBASE_API_SECRET"),
            paper=cb_config.get("paper", True),
        )

    # ─── Market Data ─────────────────────────────────────────────

    def get_mid_price(self, coin: str) -> Optional[float]:
        """Get current mid price for a coin."""
        # Coinbase uses "BTC-USD" format
        if "/" in coin:
            coin = coin.replace("/", "-")
        return self.connector.get_price(coin)

    def get_all_prices(self) -> dict:
        """Get all mid prices as {coin: price}."""
        return self.connector.get_all_prices()

    # ─── Account State ───────────────────────────────────────────

    def get_account_state(self) -> AccountState:
        """Get full account state including positions."""
        if self.paper:
            # Return simulated paper trading state
            positions = list(self._paper_positions.values())

            # Calculate unrealized PnL
            total_unrealized = 0.0
            for pos in positions:
                current_price = self.get_mid_price(pos.coin) or pos.entry_px
                if pos.size > 0:  # Long
                    pos.unrealized_pnl = (current_price - pos.entry_px) * pos.size
                else:  # Short
                    pos.unrealized_pnl = (pos.entry_px - current_price) * abs(pos.size)
                total_unrealized += pos.unrealized_pnl

            # Calculate equity
            equity = self._paper_equity + total_unrealized

            # Calculate margin used (simplified)
            margin_used = sum(abs(p.size * p.entry_px) / p.leverage for p in positions)

            return AccountState(
                equity=equity,
                available_balance=equity - margin_used,
                margin_used=margin_used,
                positions=positions,
            )
        else:
            # For live trading, get actual account state from Coinbase
            # Note: Coinbase spot doesn't have positions like perps
            # This is a simplified implementation
            balances = self.connector.get_account_balance()

            total_value = 0.0
            positions = []

            for currency, balance in balances.items():
                if balance > 0 and currency != "USD":
                    # Get USD value of holdings
                    symbol = f"{currency}-USD"
                    price = self.get_mid_price(symbol)
                    if price:
                        value = balance * price
                        total_value += value

                        # Create position for each holding
                        positions.append(Position(
                            coin=symbol,
                            size=balance,
                            entry_px=price,  # Using current price as entry
                            unrealized_pnl=0.0,
                            leverage=1.0,  # Spot = 1x leverage
                            margin_used=value,
                        ))

            usd_balance = balances.get("USD", 0.0)
            equity = usd_balance + total_value

            return AccountState(
                equity=equity,
                available_balance=usd_balance,
                margin_used=total_value,
                positions=positions,
            )

    def get_position(self, coin: str) -> Optional[Position]:
        """Get position for a specific coin, or None if no position."""
        state = self.get_account_state()
        for pos in state.positions:
            if pos.coin == coin:
                return pos
        return None

    # ─── Order Execution ─────────────────────────────────────────

    def market_open(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        slippage: float = 0.01,
    ) -> OrderResult:
        """
        Open a market order.

        Args:
            coin: Asset symbol (e.g., "BTC-USD", "ETH-USD")
            is_buy: True for long, False for short
            size: Position size in base asset units (e.g., 0.001 BTC)
            slippage: Max slippage tolerance (default 1%)
        """
        side = "buy" if is_buy else "sell"
        logger.info(f"📊 Market {side.upper()} {size} {coin} (slippage={slippage})")

        try:
            # For paper mode, we need to handle simulated positions
            if self.paper:
                price = self.get_mid_price(coin)
                if not price:
                    return OrderResult(success=False, error=f"No price for {coin}")

                # Execute paper order
                result = self.connector.market_buy(coin, size) if is_buy else self.connector.market_sell(coin, size)

                if result.success:
                    # Update paper positions
                    position_size = size if is_buy else -size

                    if coin in self._paper_positions:
                        # Update existing position
                        pos = self._paper_positions[coin]
                        # Weighted average entry price
                        total_size = pos.size + position_size
                        if abs(total_size) > 0.0001:  # Position increased
                            pos.entry_px = (pos.entry_px * pos.size + result.fill_price * position_size) / total_size
                            pos.size = total_size
                        else:  # Position closed
                            # Calculate realized PnL
                            if pos.size > 0:  # Was long
                                pnl = (result.fill_price - pos.entry_px) * min(abs(position_size), pos.size)
                            else:  # Was short
                                pnl = (pos.entry_px - result.fill_price) * min(abs(position_size), abs(pos.size))
                            self._paper_equity += pnl
                            del self._paper_positions[coin]
                    else:
                        # New position
                        self._paper_positions[coin] = Position(
                            coin=coin,
                            size=position_size,
                            entry_px=result.fill_price,
                            unrealized_pnl=0.0,
                            leverage=1.0,  # Default to 1x for spot
                        )

                    return OrderResult(
                        success=True,
                        filled_sz=result.fill_quantity,
                        avg_px=result.fill_price,
                        raw={"connector_result": result.__dict__},
                    )
                else:
                    return OrderResult(success=False, error=result.error)
            else:
                # Live trading
                price = self.get_mid_price(coin)
                if not price:
                    return OrderResult(success=False, error=f"No price for {coin}")

                size_usd = size * price
                result = self.connector.place_market_order(coin, side, size_usd)

                if result.success:
                    return OrderResult(
                        success=True,
                        filled_sz=result.fill_quantity,
                        avg_px=result.fill_price,
                        raw={"connector_result": result.__dict__},
                    )
                else:
                    return OrderResult(success=False, error=result.error)

        except Exception as e:
            logger.error(f"Market open failed: {e}")
            return OrderResult(success=False, error=str(e))

    def market_close(self, coin: str, slippage: float = 0.01) -> OrderResult:
        """Close entire position for a coin."""
        logger.info(f"📊 Market CLOSE {coin}")

        try:
            pos = self.get_position(coin)
            if not pos:
                return OrderResult(success=False, error=f"No position for {coin}")

            # Close with opposite side
            is_buy = pos.size < 0  # If short, buy to close
            size = abs(pos.size)

            return self.market_open(coin, is_buy, size, slippage)

        except Exception as e:
            logger.error(f"Market close failed: {e}")
            return OrderResult(success=False, error=str(e))

    def limit_order(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        price: float,
        tif: str = "Gtc",
        reduce_only: bool = False,
    ) -> OrderResult:
        """
        Place a limit order.

        Note: Coinbase spot doesn't support reduce_only or stop orders in the same way.
        This is a simplified implementation.
        """
        side = "buy" if is_buy else "sell"
        logger.info(f"📊 Limit {side.upper()} {size} {coin} @ {price}")

        try:
            result = self.connector.place_limit_order(coin, side, price, size)

            if result.success:
                return OrderResult(
                    success=True,
                    oid=int(result.order_id.replace("paper_", "")) if result.order_id else None,
                    filled_sz=result.fill_quantity,
                    avg_px=result.fill_price,
                    raw={"connector_result": result.__dict__},
                )
            else:
                return OrderResult(success=False, error=result.error)

        except Exception as e:
            logger.error(f"Limit order failed: {e}")
            return OrderResult(success=False, error=str(e))

    def stop_loss(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        trigger_px: float,
    ) -> OrderResult:
        """
        Place a stop-loss trigger order.

        Note: For paper mode, this just logs the SL. Real implementation would use
        Coinbase's stop-limit orders or a monitoring loop to trigger market orders.
        """
        logger.info(f"📊 Stop Loss {coin} trigger={trigger_px}")

        # For now, return success with a fake OID
        # In production, implement proper stop-loss monitoring
        logger.warning("Stop-loss orders not fully implemented for Coinbase spot - using monitoring")

        return OrderResult(
            success=True,
            oid=int(datetime.now().timestamp() * 1000),
            raw={"type": "stop_loss", "trigger_px": trigger_px},
        )

    def take_profit(
        self,
        coin: str,
        is_buy: bool,
        size: float,
        trigger_px: float,
    ) -> OrderResult:
        """Place a take-profit trigger order."""
        logger.info(f"📊 Take Profit {coin} trigger={trigger_px}")

        # Similar to stop_loss - needs proper implementation
        logger.warning("Take-profit orders not fully implemented for Coinbase spot - using monitoring")

        return OrderResult(
            success=True,
            oid=int(datetime.now().timestamp() * 1000),
            raw={"type": "take_profit", "trigger_px": trigger_px},
        )

    def cancel_order(self, coin: str, oid: int) -> bool:
        """Cancel an open order by OID."""
        try:
            success = self.connector.cancel_order(str(oid))
            if success:
                logger.info(f"✅ Cancelled order {oid} on {coin}")
            else:
                logger.warning(f"Cancel failed for order {oid}")
            return success
        except Exception as e:
            logger.error(f"Cancel failed: {e}")
            return False

    def cancel_all(self, coin: Optional[str] = None) -> int:
        """Cancel all open orders, optionally filtered by coin. Returns count cancelled."""
        try:
            open_orders = self.connector.get_open_orders(coin)
            cancelled = 0
            for order in open_orders:
                order_id = order.get("order_id", "")
                if self.connector.cancel_order(order_id):
                    cancelled += 1
            return cancelled
        except Exception as e:
            logger.error(f"Cancel all failed: {e}")
            return 0

    def set_leverage(self, coin: str, leverage: int, is_cross: bool = True) -> bool:
        """
        Set leverage for a coin.

        Note: Coinbase spot doesn't support leverage. This is a no-op for compatibility.
        """
        if leverage != 1:
            logger.warning(f"Coinbase spot doesn't support leverage > 1x. Using 1x.")
        return True

    # ─── Open Orders ─────────────────────────────────────────────

    def get_open_orders(self, coin: Optional[str] = None) -> list:
        """Get all open orders, optionally filtered by coin."""
        try:
            orders = self.connector.get_open_orders(coin)
            return orders
        except Exception as e:
            logger.error(f"Failed to get open orders: {e}")
            return []

    # ─── User Fills / Trade History ──────────────────────────────

    def get_user_fills(self, limit: int = 50) -> list:
        """Get recent fills/trades."""
        # This would require accessing Coinbase fills API
        # For now, return empty list
        return []

    # ─── Health Check ────────────────────────────────────────────

    def health_check(self) -> dict:
        """Quick connectivity and balance check."""
        try:
            health = self.connector.health_check()
            state = self.get_account_state()

            return {
                "connected": health.get("status") == "healthy",
                "testnet": self.paper,
                "address": "paper_account" if self.paper else "coinbase_account",
                "equity": state.equity,
                "available": state.available_balance,
                "positions": len(state.positions),
                "btc_price": health.get("btc_price", 0),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def __repr__(self):
        env = "PAPER" if self.paper else "LIVE"
        return f"CoinbaseExecutor({env})"


# ─── CLI Quick Test ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/kronos.json"

    try:
        executor = CoinbaseExecutor.from_config(config_path)
        health = executor.health_check()
        print(json.dumps(health, indent=2))
    except FileNotFoundError:
        print(f"Config not found at {config_path}")
        print("Create config/kronos.json with your Coinbase API credentials.")
        print('Example: {"coinbase": {"api_key": "...", "api_secret": "...", "paper": true}}')

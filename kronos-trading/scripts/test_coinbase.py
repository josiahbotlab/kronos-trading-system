#!/usr/bin/env python3
"""
Test Coinbase Connector
========================
Simple script to verify Coinbase integration works locally before deploying.

Usage:
    # Test with paper mode (no credentials needed)
    python test_coinbase.py --paper

    # Test with live mode (requires API credentials)
    export COINBASE_API_KEY=your_key
    export COINBASE_API_SECRET=your_secret
    python test_coinbase.py --live
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from execution.coinbase_connector import CoinbaseConnector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("test")


def test_prices(connector: CoinbaseConnector):
    """Test price fetching."""
    log.info("\n" + "=" * 60)
    log.info("Testing Price Fetching")
    log.info("=" * 60)

    # Test single price
    btc_price = connector.get_price("BTC-USD")
    if btc_price:
        log.info(f"✓ BTC-USD price: ${btc_price:,.2f}")
    else:
        log.error("✗ Failed to fetch BTC-USD price")
        return False

    # Test all prices
    prices = connector.get_all_prices()
    if prices:
        log.info(f"✓ Fetched {len(prices)} prices:")
        for symbol, price in list(prices.items())[:5]:
            log.info(f"  {symbol}: ${price:,.2f}")
    else:
        log.error("✗ Failed to fetch all prices")
        return False

    return True


def test_orderbook(connector: CoinbaseConnector):
    """Test order book fetching."""
    log.info("\n" + "=" * 60)
    log.info("Testing Order Book")
    log.info("=" * 60)

    ob = connector.get_order_book("BTC-USD", depth=3)
    if ob:
        log.info(f"✓ Order book fetched:")
        log.info(f"  Best bid: ${ob['bids'][0]['price']:,.2f} ({ob['bids'][0]['size']:.4f})")
        log.info(f"  Best ask: ${ob['asks'][0]['price']:,.2f} ({ob['asks'][0]['size']:.4f})")
        log.info(f"  Spread: ${ob['spread']:.2f}")
        log.info(f"  Mid: ${ob['mid']:,.2f}")
    else:
        log.error("✗ Failed to fetch order book")
        return False

    return True


def test_candles(connector: CoinbaseConnector):
    """Test historical candles."""
    log.info("\n" + "=" * 60)
    log.info("Testing Historical Candles")
    log.info("=" * 60)

    candles = connector.get_candles("BTC-USD", timeframe="5m", limit=10)
    if candles:
        log.info(f"✓ Fetched {len(candles)} candles")
        if candles:
            latest = candles[-1]
            log.info(f"  Latest: O={latest['open']:,.2f} H={latest['high']:,.2f} "
                    f"L={latest['low']:,.2f} C={latest['close']:,.2f}")
    else:
        log.error("✗ Failed to fetch candles")
        return False

    return True


def test_paper_order(connector: CoinbaseConnector):
    """Test paper trading order."""
    log.info("\n" + "=" * 60)
    log.info("Testing Paper Order Execution")
    log.info("=" * 60)

    # Test market buy
    result = connector.place_market_order("BTC-USD", "buy", 100.0)
    if result.success:
        log.info(f"✓ Paper BUY order executed:")
        log.info(f"  Order ID: {result.order_id}")
        log.info(f"  Fill price: ${result.fill_price:,.2f}")
        log.info(f"  Quantity: {result.fill_quantity:.6f}")
        log.info(f"  Fee: ${result.fee:.4f}")
    else:
        log.error(f"✗ Paper order failed: {result.error}")
        return False

    # Test market sell
    result = connector.place_market_order("ETH-USD", "sell", 50.0)
    if result.success:
        log.info(f"✓ Paper SELL order executed:")
        log.info(f"  Order ID: {result.order_id}")
        log.info(f"  Fill price: ${result.fill_price:,.2f}")
        log.info(f"  Quantity: {result.fill_quantity:.6f}")
        log.info(f"  Fee: ${result.fee:.4f}")
    else:
        log.error(f"✗ Paper order failed: {result.error}")
        return False

    return True


def test_health(connector: CoinbaseConnector):
    """Test health check."""
    log.info("\n" + "=" * 60)
    log.info("Testing Health Check")
    log.info("=" * 60)

    health = connector.health_check()
    if health.get("status") == "healthy":
        log.info(f"✓ Health check passed:")
        log.info(f"  Mode: {health.get('mode')}")
        btc_price = health.get('btc_price')
        if btc_price:
            log.info(f"  BTC price: ${btc_price:,.2f}")
        if health.get('mode') == 'live':
            log.info(f"  Balance USD: ${health.get('balance_usd', 0):,.2f}")
            log.info(f"  Balances: {health.get('balances', {})}")
    else:
        log.error(f"✗ Health check failed: {health.get('error')}")
        return False

    return True


def test_account_balance(connector: CoinbaseConnector):
    """Test account balance fetching (live mode only)."""
    if connector.paper:
        log.info("\n(Skipping account balance test in paper mode)")
        return True

    log.info("\n" + "=" * 60)
    log.info("Testing Account Balance")
    log.info("=" * 60)

    balances = connector.get_account_balance()
    if balances:
        log.info(f"✓ Account balances fetched:")
        for currency, amount in balances.items():
            log.info(f"  {currency}: {amount:,.4f}")
    else:
        log.error("✗ Failed to fetch account balance")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Test Coinbase connector")
    parser.add_argument("--live", action="store_true", help="Test in live mode (requires API creds)")
    parser.add_argument("--paper", action="store_true", help="Test in paper mode (default)")
    args = parser.parse_args()

    # Default to paper mode
    paper = not args.live

    log.info("=" * 60)
    log.info("KRONOS COINBASE CONNECTOR TEST")
    log.info("=" * 60)
    log.info(f"Mode: {'PAPER' if paper else 'LIVE'}")

    # Create connector
    try:
        connector = CoinbaseConnector(paper=paper)
        log.info("✓ Connector initialized")
    except Exception as e:
        log.error(f"✗ Failed to initialize connector: {e}")
        sys.exit(1)

    # Run tests
    tests = [
        ("Price Fetching", lambda: test_prices(connector)),
        ("Order Book", lambda: test_orderbook(connector)),
        ("Historical Candles", lambda: test_candles(connector)),
        ("Health Check", lambda: test_health(connector)),
    ]

    if paper:
        tests.append(("Paper Order Execution", lambda: test_paper_order(connector)))
    else:
        tests.append(("Account Balance", lambda: test_account_balance(connector)))

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            log.error(f"✗ {test_name} test failed with exception: {e}", exc_info=True)
            results.append((test_name, False))

    # Summary
    log.info("\n" + "=" * 60)
    log.info("TEST SUMMARY")
    log.info("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        log.info(f"{status} - {test_name}")

    log.info("-" * 60)
    log.info(f"Passed: {passed}/{total}")

    if passed == total:
        log.info("\n🎉 All tests passed! Ready to deploy.")
        sys.exit(0)
    else:
        log.error(f"\n❌ {total - passed} test(s) failed. Fix issues before deploying.")
        sys.exit(1)


if __name__ == "__main__":
    main()

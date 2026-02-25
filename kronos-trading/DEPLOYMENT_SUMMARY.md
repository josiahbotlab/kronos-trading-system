# Kronos Trading - Coinbase Migration Deployment Summary

## Overview
Successfully migrated Kronos Trading System from Hyperliquid to Coinbase Advanced Trade API.

## Changes Made

### 1. New Coinbase Connector (`execution/coinbase_connector.py`)
- JWT-based authentication for Coinbase Advanced Trade API
- Uses CCXT library for robust API interaction
- Supports both paper and live trading modes
- Price fetching, order book, historical candles
- Market and limit order execution
- Paper trade simulation with configurable slippage (5 bps default)
- SQLite logging for paper trades

### 2. Updated Exchange Connector (`execution/exchange_connector.py`)
- Now wraps CoinbaseConnector instead of calling Hyperliquid
- Maintains backward compatibility with existing interface
- Updated symbol mapping to new format (BTC-USD vs BTC/USDC:USDC)

### 3. Updated Live Engine (`execution/live_engine.py`)
- Changed symbol format throughout: `BTC/USDC:USDC` → `BTC-USD`
- Updated default `--symbol` argument to `BTC-USD`
- Updated liquidation mapping: `BTC-USD` → `BTCUSDT` (for Binance liq data)
- Maintains compatibility with existing strategy interface

### 4. Updated Price Collector (`collectors/price_collector.py`)
- Changed from Hyperliquid to Coinbase via CCXT
- Updated symbol format to `BTC-USD`, `ETH-USD`, etc.
- Symbol mapping: stores as `BTC-USD`, fetches as `BTC/USD` via CCXT

### 5. Updated Systemd Service (`config/kronos-engine.service`)
- Changed ExecStart command to use new symbol format

### 6. Created Deployment Script (`scripts/deploy_to_vps.sh`)
- Automated deployment with backup
- Environment variable configuration
- Service restart and verification

### 7. Created Test Script (`scripts/test_coinbase.py`)
- Comprehensive testing of all connector functions
- Both paper and live mode support

## Symbols Supported

### Initial Deployment
- `BTC-USD`
- `ETH-USD`
- `SOL-USD`

### Ready to Add
- `DOGE-USD`
- `XRP-USD`
- `ADA-USD`
- `AVAX-USD`
- `LINK-USD`
- `ATOM-USD`
- `ARB-USD`
- `OP-USD`
- `SUI-USD`
- `APT-USD`
- `NEAR-USD`

## Test Results (Local) ✅

All tests passed on 2026-02-11:

1. **Price Fetching** ✅
   - BTC-USD: $67,519.99
   - ETH-USD: $1,950.46
   - SOL-USD: $79.67
   - Total: 16 symbols fetched successfully

2. **Order Book** ✅
   - Best bid: $67,545.57
   - Best ask: $67,547.38
   - Spread: $1.81

3. **Historical Candles** ✅
   - Fetched 9 candles for BTC-USD
   - OHLC data correct

4. **Health Check** ✅
   - System healthy
   - BTC price: $67,531.01

5. **Paper Order Execution** ✅
   - BUY order: 0.001480 BTC @ $67,564.78 (with 5 bps slippage)
   - SELL order: 0.025652 ETH @ $1,949.14 (with slippage)
   - Fees calculated correctly (0.6% taker fee)

## Deployment Instructions

### Prerequisites
- SSH access to VPS: `agent@100.113.94.124`
- Coinbase Advanced Trade API credentials
- Existing Telegram bot token (already configured)

### Deploy to VPS

```bash
cd /Users/josiahgarcia/trading-bot/kronos-trading

# Method 1: Pass credentials as arguments
./scripts/deploy_to_vps.sh YOUR_COINBASE_API_KEY YOUR_COINBASE_API_SECRET

# Method 2: Use environment variables
export COINBASE_API_KEY=your_key_here
export COINBASE_API_SECRET=your_secret_here
./scripts/deploy_to_vps.sh
```

### What the Deployment Script Does

1. **Verifies SSH connection** to VPS
2. **Creates backup** at `/home/agent/kronos-backups/kronos-trading-TIMESTAMP`
3. **Stops services** gracefully
4. **Uploads code** via rsync (excludes data/, logs/, cache)
5. **Configures environment**:
   - Adds `COINBASE_API_KEY` to `~/.config/environment.d/kronos.conf`
   - Adds `COINBASE_API_SECRET`
   - Sets `KRONOS_PAPER=true` (paper trading mode)
   - Preserves existing Telegram credentials
6. **Installs dependencies**: `pip3 install ccxt` and others
7. **Restarts services**: `kronos-engine.service`
8. **Verifies** service is running

### Manual Verification

After deployment, verify everything is working:

```bash
# SSH to VPS
ssh agent@100.113.94.124

# Check service status
systemctl --user status kronos-engine.service

# Monitor logs
tail -f ~/kronos-trading/logs/engine.log

# Check Telegram for startup notification
# (You should receive a message from ClawdBot)

# Verify environment variables
cat ~/.config/environment.d/kronos.conf
```

### Current VPS State

**Running Services:**
- `kronos-liq-collector` - Binance liquidation data
- `kronos-price-collector` - Price/OHLCV data (now from Coinbase)
- `kronos-position-collector` - Open interest/positions
- `kronos-engine` - Main trading engine (currently BTC 5m liq_bb_combo)

**After Deployment:**
- All collectors keep running unchanged
- `kronos-engine` restarts with Coinbase connector
- Paper trading active (`KRONOS_PAPER=true`)
- Trading `BTC-USD` on 5m timeframe with `liq_bb_combo` strategy

## Important Notes

### Security
- ✅ API credentials stored as environment variables (not in code)
- ✅ Paper mode enabled by default (`KRONOS_PAPER=true`)
- ✅ No live trading until you explicitly disable paper mode
- ✅ Backup created before any changes

### Collectors
- **Liquidation collector**: Still fetches from Binance (alpha source)
- **Price collector**: Now fetches from Coinbase (matches execution exchange)
- **Position collector**: Still fetches from Binance

### Symbol Format
- **Database storage**: `BTC-USD`
- **CCXT format**: `BTC/USD`
- **Coinbase product ID**: `BTC-USD`
- **Binance liquidations**: Still map to `BTCUSDT` format

### Paper Trading
- Simulates fills at market price + 5 bps slippage
- 0.6% taker fee (Coinbase Advanced Trade rate)
- Logs to SQLite: `/home/agent/kronos-trading/data/paper_trades.db`
- Telegram notifications still work

### Going Live
When ready to switch from paper to live trading:

1. SSH to VPS
2. Edit `~/.config/environment.d/kronos.conf`
3. Change `KRONOS_PAPER=true` to `KRONOS_PAPER=false`
4. Verify you have sufficient balance on Coinbase
5. Restart service: `systemctl --user restart kronos-engine.service`
6. Monitor closely for first few trades

## Rollback Instructions

If anything goes wrong:

```bash
ssh agent@100.113.94.124

# Stop services
systemctl --user stop kronos-engine.service

# Find your backup
ls -la ~/kronos-backups/

# Restore from backup (replace TIMESTAMP with actual timestamp)
rm -rf ~/kronos-trading
cp -r ~/kronos-backups/kronos-trading-TIMESTAMP ~/kronos-trading

# Restart services
systemctl --user daemon-reload
systemctl --user start kronos-engine.service
```

## Next Steps

1. **Deploy to VPS** using the deployment script
2. **Monitor logs** for first 30 minutes
3. **Verify Telegram alerts** are working
4. **Check paper trades** in database
5. **Let it run** for 24-48 hours in paper mode
6. **Review performance** before considering live mode

## Support & Monitoring

- **Logs**: `/home/agent/kronos-trading/logs/engine.log`
- **Paper trades DB**: `/home/agent/kronos-trading/data/paper_trades.db`
- **Telegram**: ClawdBot sends trade alerts
- **Service status**: `systemctl --user status kronos-engine.service`

## Files Modified

```
execution/coinbase_connector.py         [NEW]
execution/exchange_connector.py         [MODIFIED]
execution/live_engine.py                [MODIFIED]
collectors/price_collector.py           [MODIFIED]
config/kronos-engine.service            [MODIFIED]
scripts/deploy_to_vps.sh               [NEW]
scripts/test_coinbase.py               [NEW]
```

## Files NOT Changed (as per brief)

```
execution/position_manager.py          [NO CHANGE]
execution/telegram_notifier.py         [NO CHANGE]
collectors/liquidation_collector.py     [NO CHANGE]
collectors/position_collector.py        [NO CHANGE]
strategies/*                           [NO CHANGE]
```

---

**Generated:** 2026-02-11
**System:** Kronos Trading - Coinbase Migration
**Status:** ✅ Ready to Deploy

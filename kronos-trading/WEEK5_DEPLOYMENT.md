# Week 5 Incubation Layer - Coinbase Migration Complete

## Overview
Successfully migrated both Kronos trading systems from Hyperliquid to Coinbase Advanced Trade API.

## System Architecture

### System A: Live Engine (Existing)
**Files:**
- `execution/live_engine.py` - Main trading loop
- `execution/exchange_connector.py` - Exchange wrapper
- `execution/coinbase_connector.py` - Shared Coinbase API connector
- `execution/position_manager.py` - Position/risk management
- `execution/telegram_notifier.py` - Telegram alerts

**Usage:**
```bash
# Run System A (live engine)
python execution/live_engine.py --strategy liq_bb_combo --symbol BTC-USD --timeframe 5m

# Via systemd (on VPS)
systemctl --user start kronos-engine.service
```

### System B: Week 5 Incubation Layer (NEW)
**Files:**
- `run_kronos.py` - Main entry point
- `execution/coinbase_executor.py` - Executor wrapper for Coinbase
- `execution/portfolio.py` - Portfolio manager
- `execution/risk_manager.py` - Advanced risk management
- `monitoring/incubation_tracker.py` - Strategy incubation tracking
- `monitoring/dashboard.py` - Performance monitoring

**Usage:**
```bash
# Health check
python run_kronos.py check

# Account status
python run_kronos.py status

# Start paper trading
python run_kronos.py paper --interval 60

# Show dashboard
python run_kronos.py dashboard

# Start strategy incubation
python run_kronos.py incubate cascade_p99 100 --backtest results.json

# Emergency close all
python run_kronos.py close-all
```

## Key Differences

| Feature | System A (Live Engine) | System B (Week 5) |
|---------|------------------------|-------------------|
| **Purpose** | Simple live trading | Strategy incubation & portfolio management |
| **Strategy Support** | Single strategy per run | Multiple strategies simultaneously |
| **Risk Management** | Basic (position_manager.py) | Advanced (risk_manager.py) |
| **Position Management** | Manual SL/TP in code | Automated via portfolio manager |
| **Monitoring** | Telegram only | Dashboard + Incubation tracker |
| **Deployment** | Systemd service | Manual or cron |
| **Best For** | Running proven strategies | Testing new strategies |

## Configuration

### Create config/kronos.json

Copy from example:
```bash
cp config/kronos.json.example config/kronos.json
```

Edit with your credentials:
```json
{
    "coinbase": {
        "api_key": "YOUR_COINBASE_API_KEY",
        "api_secret": "YOUR_COINBASE_API_SECRET",
        "paper": true
    },

    "risk": {
        "max_portfolio_drawdown_pct": 15.0,
        "max_daily_loss_pct": 5.0,
        "max_open_positions": 5,
        "max_portfolio_exposure_pct": 50.0,
        "default_strategy_budget": 100.0
    },

    "telegram": {
        "bot_token": "YOUR_TELEGRAM_BOT_TOKEN",
        "chat_id": "YOUR_TELEGRAM_CHAT_ID"
    },

    "trading": {
        "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "default_timeframe": "1h",
        "check_interval_seconds": 60,
        "enable_auto_trading": false
    }
}
```

**Note:** For VPS deployment, credentials are loaded from environment variables:
- `COINBASE_API_KEY`
- `COINBASE_API_SECRET`
- `KRONOS_PAPER=true`

## Deployment to VPS

### 1. Deploy Both Systems

The deployment script will upload all files:

```bash
cd /Users/josiahgarcia/trading-bot/kronos-trading

# Set your Coinbase API credentials
export COINBASE_API_KEY=your_key_here
export COINBASE_API_SECRET=your_secret_here

# Deploy
./scripts/deploy_to_vps.sh
```

### 2. Verify System A (Live Engine)

SSH to VPS and check:
```bash
ssh agent@100.113.94.124

# Check systemd service
systemctl --user status kronos-engine.service

# Monitor logs
tail -f ~/kronos-trading/logs/engine.log
```

### 3. Test System B (Week 5)

On VPS:
```bash
cd ~/kronos-trading

# Create config from environment variables
cat > config/kronos.json << 'EOF'
{
    "coinbase": {
        "api_key": null,
        "api_secret": null,
        "paper": true
    },
    "risk": {
        "max_portfolio_drawdown_pct": 15.0,
        "max_daily_loss_pct": 5.0,
        "max_open_positions": 5,
        "max_portfolio_exposure_pct": 50.0,
        "default_strategy_budget": 100.0
    },
    "trading": {
        "symbols": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "default_timeframe": "1h",
        "check_interval_seconds": 60
    }
}
EOF

# Test Week 5 system
python3 run_kronos.py check

# Run paper trading
python3 run_kronos.py paper --interval 300
```

## Local Testing (Already Passed)

### System A Tests ✅
```bash
cd /Users/josiahgarcia/trading-bot/kronos-trading
python3 scripts/test_coinbase.py --paper
```

Results:
- ✅ Price fetching: BTC @ $67,519.99
- ✅ Order book: $1.81 spread
- ✅ Historical candles: 9 fetched
- ✅ Health check: Healthy
- ✅ Paper orders: BUY/SELL simulated

### System B Tests ✅
```bash
python3 run_kronos.py check
```

Results:
- ✅ Coinbase PAPER connected
- ✅ Equity: $10,000.00
- ✅ BTC Price: $67,226.75
- ✅ Risk Manager loaded
- ✅ Incubation Tracker ready

## Workflow Examples

### Example 1: Incubate a New Strategy

```bash
# 1. Backtest the strategy (separate process)
# python backtest_strategy.py cascade_p99 --symbol BTC-USD > cascade_p99_results.json

# 2. Start incubation with $100 budget
python run_kronos.py incubate cascade_p99 100 --backtest cascade_p99_results.json

# 3. Monitor for 30 days
python run_kronos.py dashboard

# 4. If performs well, promote to System A
# If not, kill it and try another
```

### Example 2: Run Multiple Strategies

Week 5 can run multiple strategies simultaneously:

```bash
# Start portfolio manager
python run_kronos.py paper --interval 60

# In another session, check status
python run_kronos.py status
python run_kronos.py dashboard
```

### Example 3: Emergency Shutdown

```bash
# Close all positions immediately
python run_kronos.py close-all
```

## File Structure

```
kronos-trading/
├── execution/
│   ├── coinbase_connector.py      # Shared Coinbase API wrapper (System A & B)
│   ├── coinbase_executor.py       # Week 5 executor (System B)
│   ├── exchange_connector.py      # System A wrapper
│   ├── live_engine.py            # System A main loop
│   ├── position_manager.py       # System A risk
│   ├── portfolio.py              # System B portfolio manager
│   ├── risk_manager.py           # System B advanced risk
│   └── telegram_notifier.py      # Shared alerts
├── monitoring/
│   ├── dashboard.py              # Performance monitoring
│   └── incubation_tracker.py    # Strategy evaluation
├── collectors/                   # Data collection (unchanged)
├── strategies/                   # Strategy signals (unchanged)
├── config/
│   ├── kronos.json.example      # Config template
│   └── kronos-engine.service    # System A systemd service
├── scripts/
│   ├── deploy_to_vps.sh        # Deployment script
│   └── test_coinbase.py        # Test script
├── run_kronos.py               # System B entry point
└── DEPLOYMENT_SUMMARY.md       # System A deployment guide
```

## Migration Notes

### What Changed
- ✅ Hyperliquid → Coinbase Advanced Trade API
- ✅ Symbol format: `BTC/USDC:USDC` → `BTC-USD`
- ✅ Paper mode: Simulated trading with slippage
- ✅ Both systems use shared `coinbase_connector.py`

### What Stayed the Same
- ✅ Collectors still fetch from Binance (liquidation data)
- ✅ Strategies are exchange-agnostic
- ✅ Risk management principles unchanged
- ✅ Telegram notifications work as before

### Limitations
- Coinbase spot doesn't have leverage like Hyperliquid perps
- Stop-loss/take-profit require monitoring (not native to spot orders)
- Funding rates N/A for spot (strategies using funding rate need adjustment)

## Next Steps

1. **Deploy to VPS** ✓ Ready
2. **Run System A** in paper mode for 24h
3. **Test System B** with small strategy budgets
4. **Monitor both** systems for conflicts
5. **Gradually increase** allocation to successful strategies
6. **Consider live trading** only after 30+ days of paper success

## Troubleshooting

### System A won't start
- Check `~/.config/environment.d/kronos.conf` has Coinbase credentials
- Verify `systemctl --user status kronos-engine.service`
- Check logs: `journalctl --user -u kronos-engine.service -n 50`

### System B config error
- Make sure `config/kronos.json` exists or env vars are set
- Test: `python3 run_kronos.py check`

### Price fetching fails
- Verify internet connection
- Check if CCXT is installed: `pip3 list | grep ccxt`
- Test: `python3 scripts/test_coinbase.py --paper`

### Both systems conflict
- System A runs via systemd (always on)
- System B runs manually (on-demand)
- They can run simultaneously if using different symbols
- Both use same Coinbase account, so watch position conflicts

---

**Status:** ✅ Ready to Deploy
**Date:** 2026-02-11
**Systems:** A (Live Engine) + B (Week 5 Incubation)
**Exchange:** Coinbase Advanced Trade API

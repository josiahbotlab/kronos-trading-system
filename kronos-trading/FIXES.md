# Kronos Fixes - Symbol Format & Telegram

## Issue 1: Symbol Format Mismatch ✅ FIXED

### Problem
- Database has old format: `BTC/USDC:USDC`
- Engine expects new format: `BTC-USD`
- Result: 0 candles loaded on warmup

### Solution (3-part fix)

#### 1. Migration Script (run locally first)
Migrate existing database to new format:

```bash
cd /Users/josiahgarcia/trading-bot/kronos-trading

# Preview changes (dry run)
python3 scripts/migrate_symbols.py --dry-run

# Apply migration
python3 scripts/migrate_symbols.py
```

This updates:
- ✅ `prices.db` - OHLCV candles
- ✅ `execution.db` - Trade history (if exists)
- ✅ `portfolio.db` - Week 5 trades (if exists)

#### 2. Backward Compatibility (live_engine.py)
Added fallback in `load_recent_candles()`:
- First tries new format: `BTC-USD`
- If no data, tries old format: `BTC/USDC:USDC`
- Logs a warning if using old format

#### 3. Price Collector (already fixed)
`collectors/price_collector.py` now:
- ✅ Stores as `BTC-USD` format
- ✅ Maps to CCXT format `BTC/USD` for fetching
- ✅ Uses Coinbase exchange

---

## Issue 2: Telegram Credentials Lost ✅ FIXED

### Problem
- Deployment script overwrote `~/.config/environment.d/kronos.conf`
- Telegram credentials were lost
- Notifications showed as "disabled"

### Solution
Updated `scripts/deploy_to_vps.sh`:
- ✅ Reads existing Telegram credentials before overwriting
- ✅ Preserves `KRONOS_TG_BOT_TOKEN` and `KRONOS_TG_CHAT_ID`
- ✅ Only updates Coinbase credentials
- ✅ Backs up old config to `kronos.conf.backup`

---

## Deployment Steps (Updated)

### 1. Migrate Local Database (optional but recommended)

```bash
cd /Users/josiahgarcia/trading-bot/kronos-trading

# Migrate local database
python3 scripts/migrate_symbols.py
```

### 2. Deploy to VPS

```bash
# Set Coinbase credentials
export COINBASE_API_KEY=your_key
export COINBASE_API_SECRET=your_secret

# Deploy (Telegram credentials will be preserved)
./scripts/deploy_to_vps.sh
```

### 3. Migrate VPS Database

SSH to VPS and run migration:

```bash
ssh agent@100.113.94.124

cd ~/kronos-trading

# Check current data
python3 scripts/migrate_symbols.py --dry-run

# Migrate
python3 scripts/migrate_symbols.py

# Verify
python3 scripts/inspect_data.py  # Or check logs
```

### 4. Restart Services

```bash
# Restart engine to reload with new data
systemctl --user restart kronos-engine.service

# Check logs
tail -f ~/kronos-trading/logs/engine.log

# Should now show:
# - "Warming up strategies with historical data..."
# - "cascade_ride on BTC-USD: 500 candles loaded"  ← NOT 0!
# - Telegram notifications working
```

---

## Verification

### Check Symbol Format in Database

```bash
sqlite3 ~/kronos-trading/data/prices.db "SELECT DISTINCT symbol FROM ohlcv LIMIT 10"
```

Should show:
```
BTC-USD
ETH-USD
SOL-USD
```

NOT:
```
BTC/USDC:USDC   ← Old format
```

### Check Telegram Credentials

```bash
cat ~/.config/environment.d/kronos.conf
```

Should show:
```bash
KRONOS_TG_BOT_TOKEN=7123456789:AAH...  ← Your actual token
KRONOS_TG_CHAT_ID=-1001234567890       ← Your actual chat ID
COINBASE_API_KEY=...
COINBASE_API_SECRET=...
KRONOS_PAPER=true
```

### Check Live Engine Warmup

```bash
journalctl --user -u kronos-engine.service -n 100 | grep -E "candles loaded|warmup"
```

Should show:
```
Warming up strategies with historical data...
  liq_bb_combo on BTC-USD: 500 candles loaded
```

NOT:
```
  liq_bb_combo on BTC-USD: 0 candles loaded  ← Bad!
```

---

## Manual Fix (if migration fails)

### Option A: Fresh Data Collection

Delete old data and let price_collector rebuild:

```bash
# Backup old data
mv ~/kronos-trading/data/prices.db ~/kronos-trading/data/prices.db.old

# Start price collector (will create new DB with correct format)
systemctl --user restart kronos-price-collector.service

# Wait 5-10 minutes for data to accumulate
# Then restart engine
systemctl --user restart kronos-engine.service
```

### Option B: Manual SQL Update

```bash
sqlite3 ~/kronos-trading/data/prices.db

UPDATE ohlcv SET symbol = 'BTC-USD' WHERE symbol = 'BTC/USDC:USDC';
UPDATE ohlcv SET symbol = 'ETH-USD' WHERE symbol = 'ETH/USDC:USDC';
UPDATE ohlcv SET symbol = 'SOL-USD' WHERE symbol = 'SOL/USDC:USDC';

UPDATE fetch_status SET symbol = 'BTC-USD' WHERE symbol = 'BTC/USDC:USDC';
UPDATE fetch_status SET symbol = 'ETH-USD' WHERE symbol = 'ETH/USDC:USDC';
UPDATE fetch_status SET symbol = 'SOL-USD' WHERE symbol = 'SOL/USDC:USDC';

.quit
```

---

## Files Changed

```
scripts/migrate_symbols.py         [NEW] - Database migration tool
scripts/deploy_to_vps.sh           [MODIFIED] - Preserves Telegram creds
execution/live_engine.py           [MODIFIED] - Backward compatibility
collectors/price_collector.py      [ALREADY UPDATED] - New symbol format
```

---

## Summary

✅ **Issue 1 Fixed:** Symbol format migration tool + backward compatibility
✅ **Issue 2 Fixed:** Deployment script preserves Telegram credentials

**Next:** Run migration locally, deploy to VPS, verify both systems work!

"""
Polymarket Mean Reversion Bot Configuration

All constants and settings for the mean reversion strategy.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# API CONFIGURATION
# ============================================================================

# Polymarket CLOB API endpoint
POLYMARKET_HOST = "https://clob.polymarket.com"

# Chain ID (137 = Polygon Mainnet)
CHAIN_ID = 137

# Wallet configuration from environment
POLY_PRIVATE_KEY = os.getenv("POLY_PRIVATE_KEY", "")
POLY_FUNDER_ADDRESS = os.getenv("POLY_FUNDER_ADDRESS", "")

# Signature type:
# 0 = EOA (MetaMask, hardware wallet - direct signing)
# 1 = Polymarket Proxy (email/Magic wallet)
# 2 = Gnosis Safe
SIGNATURE_TYPE = int(os.getenv("POLY_SIGNATURE_TYPE", "0"))

# ============================================================================
# TRADING PARAMETERS
# ============================================================================

# Position sizing
ORDER_SIZE_USDC = 10.0          # Default order size in USDC
MAX_POSITION_SIZE_USDC = 100.0  # Max position size per market
MAX_OPEN_POSITIONS = 5          # Max concurrent positions

# Mean reversion thresholds
MEAN_LOOKBACK_HOURS = 24        # Hours to calculate rolling mean
STD_LOOKBACK_HOURS = 24         # Hours to calculate rolling std deviation
ENTRY_ZSCORE_THRESHOLD = 2.0    # Enter when price is this many stds from mean
EXIT_ZSCORE_THRESHOLD = 0.5     # Exit when price reverts within this range

# Alternative: percentage-based thresholds (if not using z-score)
ENTRY_DEVIATION_PCT = 15.0      # Enter when price deviates 15% from mean
EXIT_DEVIATION_PCT = 5.0        # Exit when price reverts within 5% of mean

# Strategy mode: 'zscore' or 'percentage'
STRATEGY_MODE = 'zscore'

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

# Take profit / Stop loss (as percentage of position)
TAKE_PROFIT_PCT = 20.0          # Close at +20% profit
STOP_LOSS_PCT = -15.0           # Close at -15% loss

# Max daily loss limit (USDC)
DAILY_LOSS_LIMIT_USDC = 50.0

# Min/max price bounds (avoid extremes)
MIN_PRICE = 0.05                # Don't buy below 5 cents
MAX_PRICE = 0.95                # Don't buy above 95 cents

# Min liquidity (USDC in order book)
MIN_LIQUIDITY_USDC = 1000.0

# Min volume (24h USDC volume)
MIN_VOLUME_24H_USDC = 5000.0

# ============================================================================
# MARKET FILTERS
# ============================================================================

# Markets to trade (condition_ids or slugs)
# Empty list = trade all markets meeting criteria
WHITELIST_MARKETS = []

# Markets to exclude
BLACKLIST_MARKETS = []

# Market categories to include (e.g., 'politics', 'crypto', 'sports')
# Empty list = all categories
ALLOWED_CATEGORIES = []

# Only trade markets expiring within N days (0 = no limit)
MAX_DAYS_TO_EXPIRY = 30

# ============================================================================
# TIMING
# ============================================================================

# Main loop interval (seconds)
LOOP_INTERVAL_SECONDS = 60

# Price history update interval (seconds)
PRICE_UPDATE_INTERVAL = 300     # 5 minutes

# Minimum time between trades on same market (seconds)
MIN_TRADE_INTERVAL = 3600       # 1 hour

# ============================================================================
# DATA STORAGE
# ============================================================================

# Data directory
DATA_DIR = Path(__file__).parent / "data"

# CSV files
PRICE_HISTORY_CSV = DATA_DIR / "price_history.csv"
TRADE_LOG_CSV = DATA_DIR / "trade_log.csv"
POSITION_LOG_CSV = DATA_DIR / "positions.csv"
DAILY_SUMMARY_CSV = DATA_DIR / "daily_summary.csv"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

# Log level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL = "INFO"

# Print trade notifications
VERBOSE_TRADES = True

# Print price updates
VERBOSE_PRICES = False

# ============================================================================
# DRY RUN MODE
# ============================================================================

# If True, simulate trades without executing
DRY_RUN = True

# ============================================================================
# MARKET WATCHLIST (Popular markets to track)
# ============================================================================

# Example markets (update with current active markets)
# Format: {'name': 'description', 'condition_id': '...', 'token_id_yes': '...', 'token_id_no': '...'}
WATCHLIST = [
    # Add markets here as they become relevant
    # {
    #     'name': 'Example Market',
    #     'condition_id': '0x...',
    #     'token_id_yes': '...',
    #     'token_id_no': '...',
    # },
]

# ============================================================================
# STRATEGY PARAMETERS
# ============================================================================

# Mean reversion specific
MEAN_REVERSION_PARAMS = {
    # Bollinger Band settings (alternative to z-score)
    'bb_period': 20,
    'bb_std': 2.0,

    # RSI settings (for confirmation)
    'rsi_period': 14,
    'rsi_oversold': 30,
    'rsi_overbought': 70,

    # Volume filter
    'volume_ma_period': 20,
    'volume_spike_threshold': 1.5,  # 1.5x average volume

    # Price smoothing
    'ema_fast': 12,
    'ema_slow': 26,
}

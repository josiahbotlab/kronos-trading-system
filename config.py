import os
from dotenv import load_dotenv

load_dotenv()

# HyperLiquid wallet private key (from .env)
SECRET_KEY = os.getenv("HL_SECRET_KEY")

# Trading parameters
ORDER_USD_SIZE = 10  # Start small
LEVERAGE = 3
TIMEFRAME = '4h'

# Symbols to trade
SYMBOLS = ['WIF', 'SOL', 'BTC', 'ETH']

# Per-symbol settings
SYMBOLS_DATA = {
    'BTC': {
        'liquidations': 900000,
        'time_window_mins': 24,
        'sl': -2,
        'tp': 1
    },
    'ETH': {
        'liquidations': 500000,
        'time_window_mins': 4,
        'sl': -2,
        'tp': 1
    },
    'SOL': {
        'liquidations': 300000,
        'time_window_mins': 4,
        'sl': -2,
        'tp': 1
    },
    'WIF': {
        'liquidations': 10000,
        'time_window_mins': 5,
        'sl': -6,
        'tp': 6
    }
}

# API Keys
MOONDEV_API_KEY = os.getenv("MOONDEV_API_KEY")

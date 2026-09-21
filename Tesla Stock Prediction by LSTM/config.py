"""
Configuration module for the Stock Price Analysis and Prediction application.
Loads environment variables and sets path and model constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Directory Paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure runtime directories exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Flask Settings
SECRET_KEY = os.getenv("SECRET_KEY", "stockai-predict-dev-key-change-in-production")
DEBUG = os.getenv("FLASK_DEBUG", "True").lower() in ("1", "true", "yes")
PORT = int(os.getenv("PORT", 5000))
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", 3600))  # 1 hour cache

# ML Model Defaults
DEFAULT_SEQUENCE_LENGTH = int(os.getenv("DEFAULT_SEQUENCE_LENGTH", 60))
TRAIN_SPLIT_RATIO = float(os.getenv("TRAIN_SPLIT_RATIO", 0.8))
DEFAULT_EPOCHS = int(os.getenv("DEFAULT_EPOCHS", 20))
DEFAULT_BATCH_SIZE = int(os.getenv("DEFAULT_BATCH_SIZE", 32))
EARLY_STOPPING_PATIENCE = 5
DEFAULT_TRAIN_PERIOD = "2y"

# Supported Options
SUPPORTED_PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
SUPPORTED_FORECAST_DAYS = [1, 7, 14, 30]

# Pre-curated Popular Stocks
POPULAR_STOCKS = {
    "indian": [
        {"ticker": "RELIANCE.NS", "name": "Reliance Industries Ltd", "exchange": "NSE"},
        {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "exchange": "NSE"},
        {"ticker": "INFY.NS", "name": "Infosys Ltd", "exchange": "NSE"},
        {"ticker": "HDFCBANK.NS", "name": "HDFC Bank Ltd", "exchange": "NSE"},
        {"ticker": "ICICIBANK.NS", "name": "ICICI Bank Ltd", "exchange": "NSE"},
        {"ticker": "SBIN.NS", "name": "State Bank of India", "exchange": "NSE"},
        {"ticker": "ITC.NS", "name": "ITC Ltd", "exchange": "NSE"},
    ],
    "us": [
        {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
        {"ticker": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
        {"ticker": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
        {"ticker": "AMZN", "name": "Amazon.com Inc.", "exchange": "NASDAQ"},
        {"ticker": "NVDA", "name": "NVIDIA Corporation", "exchange": "NASDAQ"},
        {"ticker": "TSLA", "name": "Tesla Inc.", "exchange": "NASDAQ"},
        {"ticker": "META", "name": "Meta Platforms Inc.", "exchange": "NASDAQ"},
    ],
}

# Known Indian Stock Symbols for convenient normalization
# If the user enters 'TCS', auto-normalize to 'TCS.NS'
COMMON_INDIAN_SYMBOLS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "ITC": "ITC.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "LT": "LT.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "MARUTI": "MARUTI.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "WIPRO": "WIPRO.NS",
    "ADANIENT": "ADANIENT.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
}

# Technical Indicator Features used in model
FEATURE_COLUMNS = [
    "Close",
    "Volume",
    "SMA_20",
    "EMA_12",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "BB_High",
    "BB_Low",
    "Daily_Return",
    "Volatility_20",
]
TARGET_COLUMN = "Close"

"""
Utility functions for ticker normalization, logging, and data formatting.
"""

import re
import logging
import numpy as np
import pandas as pd
from typing import Any
from config import COMMON_INDIAN_SYMBOLS

# Setup root-level logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance."""
    return logging.getLogger(name)


logger = get_logger("utils")


def normalize_ticker(ticker: str) -> str:
    """
    Normalizes a user-entered stock ticker symbol.
    - Strips whitespace
    - Converts to uppercase
    - If a known Indian stock symbol is provided without an exchange suffix (e.g., 'TCS'),
      it normalizes it to the NSE symbol (e.g., 'TCS.NS').
    - If already suffixed (e.g., 'TCS.NS', 'RELIANCE.BO') or a US ticker ('AAPL'),
      it preserves it.
    """
    if not ticker or not isinstance(ticker, str):
        return ""

    cleaned = ticker.strip().upper()

    # Check if user entered common Indian symbol without exchange suffix
    if cleaned in COMMON_INDIAN_SYMBOLS:
        normalized = COMMON_INDIAN_SYMBOLS[cleaned]
        logger.info(f"Auto-normalized Indian ticker '{cleaned}' -> '{normalized}'")
        return normalized

    return cleaned


def is_valid_ticker_format(ticker: str) -> bool:
    """
    Validates if the ticker string has a valid format.
    Length between 1 and 25 characters, valid characters only.
    """
    if not ticker:
        return False
    return bool(re.match(r"^[A-Z0-9.\-_^]{1,25}$", ticker))


def get_safe_ticker_folder(ticker: str) -> str:
    """
    Returns a filesystem-safe folder name for storing ticker models.
    Replaces characters like '^' with '_' if needed.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "_", ticker)


def to_python_type(val: Any) -> Any:
    """
    Converts numpy/pandas numeric types to native Python types for JSON serialization.
    """
    if isinstance(val, (np.floating, float)):
        return None if np.isnan(val) else round(float(val), 4)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, (pd.Timestamp, np.datetime64)):
        return str(val)[:10]
    if isinstance(val, (list, tuple, np.ndarray)):
        return [to_python_type(x) for x in val]
    if isinstance(val, dict):
        return {k: to_python_type(v) for k, v in val.items()}
    return val

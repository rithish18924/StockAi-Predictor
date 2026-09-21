"""
Market data fetching module.
Abstracts data retrieval behind a provider interface with caching and error handling.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import pandas as pd
import yfinance as yf

from config import RAW_DATA_DIR, CACHE_TTL_SECONDS, SUPPORTED_PERIODS
from src.utils import get_logger, normalize_ticker, is_valid_ticker_format, get_safe_ticker_folder

logger = get_logger("data_fetcher")


class BaseMarketDataProvider(ABC):
    """Abstract Base Class for market data providers."""

    @abstractmethod
    def fetch_history(self, ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Fetch historical OHLCV data."""
        pass

    @abstractmethod
    def fetch_stock_info(self, ticker: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Fetch company summary, latest price, exchange, and 52-week statistics."""
        pass


class YFinanceProvider(BaseMarketDataProvider):
    """
    Yahoo Finance market data provider.
    Includes caching to disk and robust error checking.
    """

    def __init__(self, cache_dir: Path = RAW_DATA_DIR, cache_ttl: int = CACHE_TTL_SECONDS):
        self.cache_dir = cache_dir
        self.cache_ttl = cache_ttl

    def _get_cache_path(self, ticker: str, period: str) -> Path:
        safe_ticker = get_safe_ticker_folder(ticker)
        return self.cache_dir / f"{safe_ticker}_{period}.json"

    def _is_cache_valid(self, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        file_mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
        return datetime.now() - file_mtime < timedelta(seconds=self.cache_ttl)

    def fetch_history(self, ticker: str, period: str = "1y") -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Fetches historical OHLCV data for a given ticker and period.
        Returns: (df, None) on success, or (None, error_message) on failure.
        """
        normalized_ticker = normalize_ticker(ticker)
        if not is_valid_ticker_format(normalized_ticker):
            return None, f"Invalid ticker symbol format: '{ticker}'."

        if period not in SUPPORTED_PERIODS:
            period = "1y"

        cache_file = self._get_cache_path(normalized_ticker, period)

        # Check Cache
        if self._is_cache_valid(cache_file):
            try:
                logger.info(f"Loading cached market data for {normalized_ticker} ({period})")
                with open(cache_file, "r", encoding="utf-8") as f:
                    cache_payload = json.load(f)
                df = pd.read_json(cache_payload["data"], orient="split")
                df.index = pd.to_datetime(df.index)
                return df, None
            except Exception as e:
                logger.warning(f"Failed to read cache for {normalized_ticker}: {e}")

        # Fetch from Yahoo Finance
        try:
            logger.info(f"Fetching live market data for {normalized_ticker} (period: {period})")
            stock = yf.Ticker(normalized_ticker)
            df = stock.history(period=period, auto_adjust=True)

            if df is None or df.empty:
                return None, f"No market data found for ticker '{normalized_ticker}'. Please verify the symbol."

            # Clean and standardize DataFrame
            # yfinance index can have timezone info; normalize to timezone-naive dates
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Ensure expected columns exist
            required_cols = ["Open", "High", "Low", "Close", "Volume"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return None, f"Market data for '{normalized_ticker}' is missing required columns: {missing_cols}"

            df = df[required_cols].copy()
            df.sort_index(ascending=True, inplace=True)
            df.dropna(subset=["Close"], inplace=True)

            # Save to Cache
            try:
                cache_payload = {
                    "ticker": normalized_ticker,
                    "period": period,
                    "timestamp": datetime.now().isoformat(),
                    "data": df.to_json(orient="split", date_format="iso"),
                }
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_payload, f)
            except Exception as e:
                logger.warning(f"Could not write cache file for {normalized_ticker}: {e}")

            return df, None

        except Exception as e:
            err_msg = f"Network or API error while fetching '{normalized_ticker}': {str(e)}"
            logger.error(err_msg)
            return None, err_msg

    def fetch_stock_info(self, ticker: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Fetches company overview, quote data, and 52-week metrics.
        Returns: (info_dict, None) on success, or (None, error_message) on failure.
        """
        normalized_ticker = normalize_ticker(ticker)
        if not is_valid_ticker_format(normalized_ticker):
            return None, f"Invalid ticker format: '{ticker}'."

        try:
            stock = yf.Ticker(normalized_ticker)
            fast_info = getattr(stock, "fast_info", None)
            info = {}
            try:
                info = stock.info or {}
            except Exception:
                info = {}

            # Fast info fallbacks
            current_price = None
            prev_close = None
            currency = "USD"
            exchange = "UNKNOWN"

            if fast_info:
                current_price = getattr(fast_info, "last_price", None)
                prev_close = getattr(fast_info, "previous_close", None)
                currency = getattr(fast_info, "currency", "USD") or "USD"
                exchange = getattr(fast_info, "exchange", "UNKNOWN") or "UNKNOWN"
                fifty_two_high = getattr(fast_info, "year_high", None)
                fifty_two_low = getattr(fast_info, "year_low", None)
                market_cap = getattr(fast_info, "market_cap", None)
            else:
                fifty_two_high = info.get("fiftyTwoWeekHigh")
                fifty_two_low = info.get("fiftyTwoWeekLow")
                market_cap = info.get("marketCap")

            if current_price is None:
                current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if prev_close is None:
                prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")

            # If still missing, attempt to extract from latest history
            if current_price is None or prev_close is None:
                df_short, _ = self.fetch_history(normalized_ticker, period="5d")
                if df_short is not None and len(df_short) >= 1:
                    current_price = float(df_short["Close"].iloc[-1])
                    if len(df_short) >= 2:
                        prev_close = float(df_short["Close"].iloc[-2])
                    else:
                        prev_close = current_price

            if current_price is None:
                return None, f"Unable to fetch current quote for '{normalized_ticker}'."

            prev_close = prev_close if prev_close else current_price
            day_change = current_price - prev_close
            day_change_pct = (day_change / prev_close) * 100 if prev_close else 0.0

            # Determine market based on ticker or exchange
            market = "US Market"
            if normalized_ticker.endswith(".NS"):
                market = "NSE (National Stock Exchange of India)"
                currency = "INR"
            elif normalized_ticker.endswith(".BO"):
                market = "BSE (Bombay Stock Exchange)"
                currency = "INR"
            elif "NASDAQ" in exchange.upper():
                market = "NASDAQ"
            elif "NYSE" in exchange.upper():
                market = "NYSE"

            company_name = info.get("longName") or info.get("shortName") or normalized_ticker

            payload = {
                "ticker": normalized_ticker,
                "name": company_name,
                "currency": currency,
                "exchange": exchange,
                "market": market,
                "current_price": round(float(current_price), 2),
                "previous_close": round(float(prev_close), 2),
                "day_change": round(float(day_change), 2),
                "day_change_pct": round(float(day_change_pct), 2),
                "fifty_two_high": round(float(fifty_two_high), 2) if fifty_two_high else None,
                "fifty_two_low": round(float(fifty_two_low), 2) if fifty_two_low else None,
                "market_cap": market_cap,
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
                "summary": info.get("longBusinessSummary", ""),
            }
            return payload, None

        except Exception as e:
            err_msg = f"Error fetching information for '{normalized_ticker}': {str(e)}"
            logger.error(err_msg)
            return None, err_msg


# Default Provider Factory
def get_market_data_provider() -> BaseMarketDataProvider:
    """Returns the configured market data provider."""
    return YFinanceProvider()

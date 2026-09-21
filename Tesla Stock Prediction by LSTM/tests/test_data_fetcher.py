"""
Unit tests for data fetcher and ticker normalization utilities.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

from src.utils import normalize_ticker, is_valid_ticker_format, get_safe_ticker_folder
from src.data_fetcher import YFinanceProvider


class TestDataFetcher(unittest.TestCase):

    def test_normalize_ticker(self):
        # Indian common tickers auto-normalized to NSE
        self.assertEqual(normalize_ticker("tcs"), "TCS.NS")
        self.assertEqual(normalize_ticker("reliance"), "RELIANCE.NS")
        self.assertEqual(normalize_ticker("infy"), "INFY.NS")
        self.assertEqual(normalize_ticker("hdfcbank"), "HDFCBANK.NS")

        # Explicit suffix preserved
        self.assertEqual(normalize_ticker("TCS.NS"), "TCS.NS")
        self.assertEqual(normalize_ticker("RELIANCE.BO"), "RELIANCE.BO")

        # US tickers preserved
        self.assertEqual(normalize_ticker("aapl"), "AAPL")
        self.assertEqual(normalize_ticker("NVDA"), "NVDA")
        self.assertEqual(normalize_ticker("TSLA"), "TSLA")

        # Empty / whitespace
        self.assertEqual(normalize_ticker("  msft  "), "MSFT")
        self.assertEqual(normalize_ticker(""), "")

    def test_is_valid_ticker_format(self):
        self.assertTrue(is_valid_ticker_format("AAPL"))
        self.assertTrue(is_valid_ticker_format("TCS.NS"))
        self.assertTrue(is_valid_ticker_format("BRK.B"))
        self.assertFalse(is_valid_ticker_format(""))
        self.assertFalse(is_valid_ticker_format("INVALID SYMBOL WITH SPACES"))
        self.assertFalse(is_valid_ticker_format("INVALID<SCRIPT>"))

    def test_get_safe_ticker_folder(self):
        self.assertEqual(get_safe_ticker_folder("TCS.NS"), "TCS.NS")
        self.assertEqual(get_safe_ticker_folder("^NSEI"), "_NSEI")

    @patch("yfinance.Ticker")
    def test_fetch_history_success(self, mock_ticker_cls):
        # Create synthetic OHLCV dataframe
        dates = pd.date_range("2024-01-01", periods=10, freq="B")
        mock_df = pd.DataFrame({
            "Open": [100.0 + i for i in range(10)],
            "High": [105.0 + i for i in range(10)],
            "Low": [98.0 + i for i in range(10)],
            "Close": [102.0 + i for i in range(10)],
            "Volume": [1000000 + i * 1000 for i in range(10)],
        }, index=dates)

        mock_instance = MagicMock()
        mock_instance.history.return_value = mock_df
        mock_ticker_cls.return_value = mock_instance

        provider = YFinanceProvider()
        df, err = provider.fetch_history("TEST_TICKER", period="1mo")

        self.assertIsNone(err)
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 10)
        self.assertIn("Close", df.columns)

    def test_fetch_history_invalid_ticker(self):
        provider = YFinanceProvider()
        df, err = provider.fetch_history("INVALID$$$%", period="1mo")
        self.assertIsNone(df)
        self.assertIn("Invalid ticker symbol", err)


if __name__ == "__main__":
    unittest.main()

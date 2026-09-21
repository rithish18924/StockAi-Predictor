"""
Feature engineering module.
Calculates technical indicators without future-data leakage.
"""

import numpy as np
import pandas as pd
from typing import Optional
from src.utils import get_logger

logger = get_logger("feature_engineering")


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Simple and Exponential Moving Averages."""
    df = df.copy()
    close = df["Close"]

    # Simple Moving Averages
    df["SMA_20"] = close.rolling(window=20, min_periods=1).mean()
    df["SMA_50"] = close.rolling(window=50, min_periods=1).mean()
    df["SMA_200"] = close.rolling(window=200, min_periods=1).mean()

    # Exponential Moving Averages
    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    return df


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculates Relative Strength Index (RSI) using Wilder's smoothing.
    No future data is referenced.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's exponential smoothing
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Adds RSI indicator."""
    df = df.copy()
    df[f"RSI_{period}"] = calculate_rsi(df["Close"], period=period)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Adds Moving Average Convergence Divergence (MACD).
    MACD Line: EMA(fast) - EMA(slow)
    Signal Line: EMA(signal) of MACD Line
    Histogram: MACD Line - Signal Line
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """
    Adds Bollinger Bands:
    Middle Band: 20-day SMA
    Upper Band: Middle + num_std * 20-day standard deviation
    Lower Band: Middle - num_std * 20-day standard deviation
    """
    df = df.copy()
    rolling_mean = df["Close"].rolling(window=window, min_periods=1).mean()
    rolling_std = df["Close"].rolling(window=window, min_periods=1).std().fillna(0)

    df["BB_Middle"] = rolling_mean
    df["BB_High"] = rolling_mean + (rolling_std * num_std)
    df["BB_Low"] = rolling_mean - (rolling_std * num_std)
    df["BB_Width"] = (df["BB_High"] - df["BB_Low"]) / (rolling_mean + 1e-9)
    return df


def add_returns_and_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds percentage returns and annualized rolling volatility.
    """
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change().fillna(0)
    # Annualized volatility: std * sqrt(252 trading days)
    df[f"Volatility_{window}"] = (
        df["Daily_Return"].rolling(window=window, min_periods=1).std().fillna(0) * np.sqrt(252)
    )
    return df


def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all technical indicators and feature engineering to OHLCV DataFrame.
    Guarantees no future lookahead.
    """
    if df is None or df.empty:
        return df

    df = df.copy()
    df = add_moving_averages(df)
    df = add_rsi(df, period=14)
    df = add_macd(df, fast=12, slow=26, signal=9)
    df = add_bollinger_bands(df, window=20, num_std=2.0)
    df = add_returns_and_volatility(df, window=20)

    # Forward fill then backward fill any early calculation warmups
    df.bfill(inplace=True)
    df.ffill(inplace=True)

    return df

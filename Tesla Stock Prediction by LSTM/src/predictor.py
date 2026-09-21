"""
Multi-step future prediction engine.
Performs autoregressive forecasting with dynamic feature updates and uncertainty bounds.
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

from config import (
    MODELS_DIR,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SUPPORTED_FORECAST_DAYS,
)
from src.data_fetcher import get_market_data_provider
from src.feature_engineering import engineer_all_features
from src.preprocessing import DataPreprocessor
from src.model import load_saved_model
from src.utils import get_logger, normalize_ticker, get_safe_ticker_folder, to_python_type

logger = get_logger("predictor")


class StockPredictor:
    """
    Generates multi-step price forecasts using trained LSTM models.
    """

    def __init__(self, ticker: str):
        self.ticker = normalize_ticker(ticker)
        self.safe_ticker = get_safe_ticker_folder(self.ticker)
        self.model_dir = MODELS_DIR / self.safe_ticker
        self.model_file = self.model_dir / "model.keras"
        self.scaler_file = self.model_dir / "scaler.pkl"
        self.metadata_file = self.model_dir / "metadata.json"

    def is_model_available(self) -> bool:
        """Checks if model, scaler, and metadata exist on disk."""
        return (
            self.model_file.exists()
            and self.scaler_file.exists()
            and self.metadata_file.exists()
        )

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Reads model metadata."""
        if not self.metadata_file.exists():
            return None
        try:
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def predict_future(
        self, days: int = 7
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Generates multi-step price forecasts for the specified number of days (1, 7, 14, 30).
        Returns: (forecast_payload, None) on success, or (None, error_message) on failure.
        """
        if days not in SUPPORTED_FORECAST_DAYS:
            days = 7

        if not self.is_model_available():
            return None, f"No trained model found for '{self.ticker}'. Please train the model first."

        # Load artifacts
        try:
            model = load_saved_model(self.model_file)
            preprocessor = DataPreprocessor.load_scalers(self.scaler_file)
            metadata = self.get_metadata() or {}
        except Exception as e:
            return None, f"Failed to load model artifacts for '{self.ticker}': {e}"

        seq_len = preprocessor.sequence_length
        residual_std = metadata.get("residual_std", 1.0)

        # Fetch recent historical data (at least sequence_length + 60 days to seed feature calculations)
        provider = get_market_data_provider()
        df_hist, err = provider.fetch_history(self.ticker, period="1y")
        if err or df_hist is None or len(df_hist) < seq_len:
            return None, f"Insufficient recent historical data to seed prediction for '{self.ticker}'."

        df_working = engineer_all_features(df_hist.copy())
        latest_date = df_working.index[-1]
        latest_close = float(df_working["Close"].iloc[-1])

        # Prepare business days for future dates (skipping Saturdays and Sundays)
        future_dates: List[datetime] = []
        curr_date = latest_date
        while len(future_dates) < days:
            curr_date += timedelta(days=1)
            if curr_date.weekday() < 5:  # Monday to Friday
                future_dates.append(curr_date)

        forecasted_prices: List[float] = []
        upper_bounds: List[float] = []
        lower_bounds: List[float] = []

        # Autoregressive multi-step prediction loop
        current_features_df = df_working.copy()

        for step_idx in range(days):
            # Take the most recent sequence_length rows of features
            recent_slice = current_features_df.iloc[-seq_len:]
            scaled_features = preprocessor.transform_features(recent_slice)

            # Shape: (1, seq_len, n_features)
            input_tensor = np.expand_dims(scaled_features, axis=0)

            # Predict next step scaled close price
            scaled_pred = model.predict(input_tensor, verbose=0).flatten()[0]
            pred_price = float(preprocessor.inverse_transform_targets(np.array([scaled_pred]))[0])

            # Guard against extreme negative or impossible prices
            pred_price = max(pred_price, 0.01)
            forecasted_prices.append(round(pred_price, 2))

            # Uncertainty interval expands with forecast horizon:
            # 1.96 * residual_std * sqrt(step + 1) for ~95% confidence interval
            uncertainty_margin = 1.96 * residual_std * np.sqrt(step_idx + 1)
            upper = round(pred_price + uncertainty_margin, 2)
            lower = round(max(pred_price - uncertainty_margin, 0.01), 2)
            upper_bounds.append(upper)
            lower_bounds.append(lower)

            # Autoregressive step: create a synthetic next day row to update technical indicators
            next_date = future_dates[step_idx]
            # Use estimated volume (rolling average) and estimated open/high/low based on predicted close
            last_vol = current_features_df["Volume"].iloc[-5:].mean()
            new_row = pd.DataFrame(
                {
                    "Open": [pred_price],
                    "High": [max(pred_price, upper)],
                    "Low": [min(pred_price, lower)],
                    "Close": [pred_price],
                    "Volume": [last_vol],
                },
                index=[next_date],
            )

            # Append synthetic row and re-engineer features
            concat_df = pd.concat([current_features_df[["Open", "High", "Low", "Close", "Volume"]], new_row])
            current_features_df = engineer_all_features(concat_df)

        # Build detailed prediction records
        predictions = []
        for i, dt in enumerate(future_dates):
            price = forecasted_prices[i]
            diff = round(price - latest_close, 2)
            pct = round((diff / latest_close) * 100, 2) if latest_close else 0.0
            predictions.append({
                "step": i + 1,
                "date": dt.strftime("%Y-%m-%d"),
                "predicted_price": price,
                "lower_bound": lower_bounds[i],
                "upper_bound": upper_bounds[i],
                "change_from_latest": diff,
                "percent_change": pct,
            })

        final_predicted = forecasted_prices[-1]
        overall_change = round(final_predicted - latest_close, 2)
        overall_pct = round((overall_change / latest_close) * 100, 2) if latest_close else 0.0

        # Extract recent historical tail (last 30 trading days) for the comparison chart
        hist_tail = []
        for dt, row in df_hist.iloc[-30:].iterrows():
            hist_tail.append({
                "date": dt.strftime("%Y-%m-%d"),
                "close": round(float(row["Close"]), 2),
            })

        payload = {
            "ticker": self.ticker,
            "forecast_days": days,
            "latest_historical_date": latest_date.strftime("%Y-%m-%d"),
            "latest_historical_close": round(latest_close, 2),
            "final_predicted_price": final_predicted,
            "overall_change": overall_change,
            "overall_percent_change": overall_pct,
            "predictions": predictions,
            "historical_tail": hist_tail,
            "metadata": metadata,
            "disclaimer": (
                "These forecasts are generated by a machine-learning model (LSTM) using historical market data. "
                "They are experimental estimates and are not guaranteed future prices or financial advice. "
                "Market conditions can change unexpectedly."
            ),
        }

        return to_python_type(payload), None


def predict_stock_prices(
    ticker: str, days: int = 7
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Convenience function to generate predictions."""
    predictor = StockPredictor(ticker=ticker)
    return predictor.predict_future(days=days)

"""
Flask Application Entrypoint for Stock Price Analysis and Prediction.
Serves interactive dashboard pages and RESTful API endpoints.
"""

import os
from pathlib import Path
from flask import Flask, render_template, request, jsonify, abort

from config import (
    PORT,
    DEBUG,
    SECRET_KEY,
    POPULAR_STOCKS,
    SUPPORTED_PERIODS,
    SUPPORTED_FORECAST_DAYS,
    DEFAULT_EPOCHS,
)
from src.utils import get_logger, normalize_ticker, is_valid_ticker_format
from src.data_fetcher import get_market_data_provider
from src.feature_engineering import engineer_all_features
from src.trainer import train_stock_model
from src.predictor import predict_stock_prices, StockPredictor
from src.evaluation import get_metric_explanations

logger = get_logger("app")

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
app.config["JSON_SORT_KEYS"] = False


# =====================================================================
# Frontend HTML Routes
# =====================================================================


@app.route("/")
def index():
    """Renders the landing page with popular stocks and search hero."""
    return render_template(
        "index.html",
        popular_stocks=POPULAR_STOCKS,
    )


@app.route("/dashboard")
def dashboard():
    """Renders the financial analytics and forecasting dashboard."""
    ticker_arg = request.args.get("ticker", "").strip()
    normalized = normalize_ticker(ticker_arg) if ticker_arg else "TCS.NS"

    return render_template(
        "dashboard.html",
        initial_ticker=normalized,
        popular_stocks=POPULAR_STOCKS,
        supported_periods=SUPPORTED_PERIODS,
        supported_forecast_days=SUPPORTED_FORECAST_DAYS,
        metric_explanations=get_metric_explanations(),
    )


# =====================================================================
# RESTful API Endpoints
# =====================================================================


@app.route("/api/stock/<path:ticker>", methods=["GET"])
def api_stock_info(ticker: str):
    """
    Returns company overview and quote statistics.
    Example: GET /api/stock/TCS.NS or GET /api/stock/AAPL
    """
    normalized = normalize_ticker(ticker)
    if not is_valid_ticker_format(normalized):
        return jsonify({"success": False, "error": f"Invalid ticker format: '{ticker}'"}), 400

    provider = get_market_data_provider()
    info, err = provider.fetch_stock_info(normalized)

    if err or info is None:
        logger.warning(f"Stock info failed for {normalized}: {err}")
        return jsonify({
            "success": False,
            "error": err or f"Unable to retrieve quote data for '{normalized}'. Verify symbol and try again."
        }), 404

    return jsonify({"success": True, "data": info})


@app.route("/api/history/<path:ticker>", methods=["GET"])
def api_stock_history(ticker: str):
    """
    Returns historical OHLCV data with technical indicators for charting.
    Query params: ?period=1y (default)
    """
    normalized = normalize_ticker(ticker)
    period = request.args.get("period", "1y").lower()

    if not is_valid_ticker_format(normalized):
        return jsonify({"success": False, "error": f"Invalid ticker format: '{ticker}'"}), 400

    if period not in SUPPORTED_PERIODS:
        period = "1y"

    provider = get_market_data_provider()
    df, err = provider.fetch_history(normalized, period=period)

    if err or df is None or df.empty:
        logger.warning(f"History failed for {normalized}: {err}")
        return jsonify({
            "success": False,
            "error": err or f"No historical market data found for '{normalized}'."
        }), 404

    # Calculate technical indicators
    df_featured = engineer_all_features(df)

    # Format for JSON chart serialization
    records = []
    for dt, row in df_featured.iterrows():
        records.append({
            "date": dt.strftime("%Y-%m-%d"),
            "open": round(float(row["Open"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "close": round(float(row["Close"]), 2),
            "volume": int(row["Volume"]),
            "sma_20": round(float(row["SMA_20"]), 2) if "SMA_20" in row and not row.isna()["SMA_20"] else None,
            "sma_50": round(float(row["SMA_50"]), 2) if "SMA_50" in row and not row.isna()["SMA_50"] else None,
            "ema_12": round(float(row["EMA_12"]), 2) if "EMA_12" in row and not row.isna()["EMA_12"] else None,
            "rsi_14": round(float(row["RSI_14"]), 2) if "RSI_14" in row and not row.isna()["RSI_14"] else None,
            "macd": round(float(row["MACD"]), 4) if "MACD" in row and not row.isna()["MACD"] else None,
            "macd_signal": round(float(row["MACD_Signal"]), 4) if "MACD_Signal" in row and not row.isna()["MACD_Signal"] else None,
            "macd_hist": round(float(row["MACD_Hist"]), 4) if "MACD_Hist" in row and not row.isna()["MACD_Hist"] else None,
            "bb_high": round(float(row["BB_High"]), 2) if "BB_High" in row and not row.isna()["BB_High"] else None,
            "bb_low": round(float(row["BB_Low"]), 2) if "BB_Low" in row and not row.isna()["BB_Low"] else None,
            "volatility": round(float(row["Volatility_20"] * 100), 2) if "Volatility_20" in row and not row.isna()["Volatility_20"] else None,
        })

    return jsonify({
        "success": True,
        "ticker": normalized,
        "period": period,
        "count": len(records),
        "data": records,
    })


@app.route("/api/model-status/<path:ticker>", methods=["GET"])
def api_model_status(ticker: str):
    """
    Checks if a trained LSTM model exists for the ticker and returns metadata.
    """
    normalized = normalize_ticker(ticker)
    predictor = StockPredictor(ticker=normalized)
    is_trained = predictor.is_model_available()
    metadata = predictor.get_metadata() if is_trained else None

    return jsonify({
        "success": True,
        "ticker": normalized,
        "is_trained": is_trained,
        "metadata": metadata,
    })


@app.route("/api/train", methods=["POST"])
def api_train_model():
    """
    Trains/retrains the LSTM model for a specific ticker.
    Payload: {"ticker": "TCS.NS", "epochs": 20}
    """
    data = request.get_json(silent=True) or {}
    raw_ticker = data.get("ticker", "")
    epochs = int(data.get("epochs", DEFAULT_EPOCHS))
    epochs = max(3, min(epochs, 50))  # Sanitize range

    normalized = normalize_ticker(raw_ticker)
    if not is_valid_ticker_format(normalized):
        return jsonify({"success": False, "error": f"Invalid ticker format: '{raw_ticker}'"}), 400

    logger.info(f"Received training request for {normalized} (epochs: {epochs})")
    metadata, err = train_stock_model(ticker=normalized, epochs=epochs)

    if err or metadata is None:
        logger.error(f"Training failed for {normalized}: {err}")
        return jsonify({
            "success": False,
            "error": err or f"Model training failed for '{normalized}'."
        }), 500

    return jsonify({
        "success": True,
        "message": f"Successfully trained LSTM model for {normalized}",
        "metadata": metadata,
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Generates multi-day forecast for the ticker.
    Payload: {"ticker": "AAPL", "days": 7}
    """
    data = request.get_json(silent=True) or {}
    raw_ticker = data.get("ticker", "")
    days = int(data.get("days", 7))

    if days not in SUPPORTED_FORECAST_DAYS:
        days = 7

    normalized = normalize_ticker(raw_ticker)
    if not is_valid_ticker_format(normalized):
        return jsonify({"success": False, "error": f"Invalid ticker format: '{raw_ticker}'"}), 400

    predictor = StockPredictor(ticker=normalized)
    if not predictor.is_model_available():
        return jsonify({
            "success": False,
            "needs_training": True,
            "error": f"No trained model found for '{normalized}'. Please train the model first.",
        }), 404

    payload, err = predictor.predict_future(days=days)
    if err or payload is None:
        return jsonify({
            "success": False,
            "error": err or f"Prediction failed for '{normalized}'."
        }), 500

    return jsonify({"success": True, "data": payload})


# =====================================================================
# Error Handlers
# =====================================================================


@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Endpoint or resource not found."}), 404
    return render_template("error.html", error_code=404, message="The requested page could not be found."), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Internal server error occurred."}), 500
    return render_template("error.html", error_code=500, message="An internal server error occurred."), 500


if __name__ == "__main__":
    logger.info(f"Starting StockAI Predictor web application on port {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=DEBUG)

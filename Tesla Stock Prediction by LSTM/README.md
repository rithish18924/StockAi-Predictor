# StockAI Predictor — Multi-Stock Price Analysis & LSTM Forecasting Platform

A production-grade financial analytics and machine learning web application built with **Flask**, **TensorFlow/Keras (LSTM)**, **pandas**, **scikit-learn**, and **Chart.js**.

The platform is designed from the ground up for **multi-market equity analysis**, supporting both Indian equities (NSE / BSE) and US equities (NASDAQ / NYSE). Users can dynamically enter any valid ticker symbol (e.g., `RELIANCE.NS`, `TCS.NS`, `INFY.NS`, `AAPL`, `MSFT`, `NVDA`, `TSLA`) to inspect historical market data, compute real-time technical indicators, train isolated LSTM neural networks without lookahead bias, and generate multi-day autoregressive price forecasts with uncertainty boundaries.

---

## Key Features

1. **Multi-Market Support (Indian & US)**:
   - **Indian Equities**: Full NSE (`.NS`) and BSE (`.BO`) support. Automatic normalization converts shorthand inputs like `TCS` or `RELIANCE` to their verified NSE symbols (`TCS.NS`, `RELIANCE.NS`).
   - **US Equities**: Wall Street equities on NASDAQ and NYSE (`AAPL`, `NVDA`, `TSLA`, `MSFT`, `GOOGL`, etc.).
   - No hardcoded tickers or single-stock assumptions anywhere in the pipeline.

2. **Strict Zero-Leakage Machine Learning**:
   - **Strict Chronological Splitting**: Chronological 80/20 train/test splitting without random shuffling.
   - **Scaler Isolation**: Feature scalers and target price scalers are fitted **strictly on training data**. Out-of-sample test data is only transformed, preventing future-data leakage.
   - **Lag-Only Technical Indicators**: Indicators (SMA, EMA, RSI, MACD, Bollinger Bands, Volatility) are computed solely using past prices.

3. **Stacked LSTM Neural Network**:
   - Two recurrent LSTM layers with dropout regularization (`0.2`).
   - Huber loss function for resilience against volatile market spikes.
   - `EarlyStopping` and `ReduceLROnPlateau` callbacks to prevent overfitting.
   - Artifacts saved per ticker in isolated directories (`models/<TICKER>/`).

4. **Multi-Step Autoregressive Forecasting**:
   - Select forecast horizons: **1 Day**, **7 Days**, **14 Days**, or **30 Days**.
   - Dynamic feature updating: future steps recursively update moving averages, RSI, and momentum indicators rather than repeating a static price.
   - Calendar date mapping skips non-trading weekends.

5. **Responsible AI & Uncertainty Estimates**:
   - Computes **95% uncertainty boundaries** scaled with the square root of the forecasting horizon ($1.96 \cdot \sigma_{\text{residual}} \cdot \sqrt{t}$).
   - Prominent disclaimer highlighting that forecasts are experimental estimates, not investment advice.

6. **Professional Financial Terminal UI**:
   - Obsidian dark theme with glassmorphism cards and Inter/JetBrains Mono typography.
   - Interactive charts: Price & Moving Averages, RSI with Overbought/Oversold thresholds, MACD Histogram, and Historical vs Forecast trajectory.
   - Real-time model evaluation scorecard: MAE, RMSE, MSE, MAPE, Directional Accuracy, and $R^2$.

---

## Project Structure

```text
stock-prediction-webapp/
│
├── app.py                     # Flask application entrypoint & REST API
├── config.py                  # Global settings, paths, stock presets & hyperparams
├── requirements.txt           # Pinned production dependencies
├── README.md                  # Complete documentation
├── .gitignore                 # Git ignore rules
├── .env.example               # Template environment configuration
│
├── data/
│   ├── raw/                   # Cached market data JSON files
│   └── processed/             # Processed datasets
│
├── models/                    # Per-ticker trained models, scalers, and metadata
│   └── [TICKER]/
│       ├── model.keras        # Saved Keras LSTM model
│       ├── scaler.pkl         # Serialized MinMaxScaler artifacts
│       └── metadata.json      # Training parameters & out-of-sample metrics
│
├── src/
│   ├── __init__.py
│   ├── utils.py               # Ticker normalization, logging, and type sanitization
│   ├── data_fetcher.py        # Abstract market data provider & yfinance integration
│   ├── feature_engineering.py # SMA, EMA, RSI, MACD, Bollinger Bands, Volatility
│   ├── preprocessing.py       # Chronological split, scaler fitting, sequence windowing
│   ├── model.py               # Stacked LSTM architecture & Keras callbacks
│   ├── trainer.py             # End-to-end training pipeline & evaluation
│   ├── predictor.py           # Autoregressive multi-step forecaster & uncertainty bounds
│   └── evaluation.py          # Regression error metrics & plain-English definitions
│
├── templates/
│   ├── index.html             # Hero landing page with search & popular stock pills
│   ├── dashboard.html         # Full analytics terminal with charts & forecast controls
│   ├── error.html             # User-friendly error page
│   └── components/
│       ├── navbar.html        # Persistent navigation & search bar
│       ├── stock_card.html    # Company summary, quote, & 52-week progress bar
│       └── prediction_card.html # Forecast controls, breakdown table, & disclaimer
│
├── static/
│   ├── css/
│   │   └── style.css          # Dark financial theme, glassmorphism, responsive grid
│   └── js/
│       ├── app.js             # Form auto-normalization & toast notifications
│       ├── charts.js          # Chart.js renderers for Candlesticks, RSI, MACD, Forecast
│       └── dashboard.js       # Dashboard state manager & API client
│
└── tests/
    ├── __init__.py
    ├── test_data_fetcher.py   # Tests for ticker normalization & data fetcher
    ├── test_preprocessing.py  # Tests verifying zero leakage & sequence dimensions
    ├── test_predictor.py      # Tests for multi-step predictions & error metrics
    └── test_app.py            # Flask route & REST API integration tests
```

---

## Technology Stack

- **Backend**: Python 3.11+ / 3.13, Flask 3.1
- **Machine Learning**: TensorFlow 2.21, Keras 3.15, scikit-learn 1.8, pandas, NumPy
- **Market Data**: yfinance 1.7 (extensible to AlphaVantage/Polygon)
- **Frontend**: HTML5, Vanilla CSS3 (Custom Financial Dark Theme), Bootstrap 5.3, Chart.js 4.4
- **Testing**: pytest

---

## Installation & Setup (Windows & VS Code)

### 1. Prerequisites
- Windows 10/11
- Python 3.11, 3.12, or 3.13 (64-bit)
- Visual Studio Code

### 2. Clone or Navigate to Project
Open PowerShell in the project directory:
```powershell
cd "c:\Users\sures\Downloads\Tesla Stock Prediction by LSTM"
```

### 3. Create and Activate Virtual Environment (Recommended)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 4. Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Setup Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

---

## How to Run

### Start the Flask Web Server
```powershell
python app.py
```
Or using the Flask CLI:
```powershell
flask run --port 5000
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## How to Use the Application

1. **Search Any Stock**:
   - On the Home page or the navigation bar, enter a ticker:
     - Indian stocks: `TCS`, `RELIANCE`, `INFY`, `HDFCBANK`, `ICICIBANK`, `ITC.NS`
     - US stocks: `AAPL`, `NVDA`, `TSLA`, `MSFT`, `AMZN`, `GOOGL`, `META`
   - Click **Analyze**.
2. **Inspect Historical Data & Indicators**:
   - Switch timeframes (**1M, 3M, 6M, 1Y, 5Y**) on the main chart.
   - Examine RSI momentum, MACD trends, and 52-week trading ranges.
3. **Train the LSTM Model**:
   - If the stock does not have a pre-trained model, the status badge will indicate `Needs Training`.
   - Click **Train Model**, select the number of epochs (e.g., 20), and click **Start Training**.
   - The server will train the model, save artifacts into `models/<TICKER>/`, and update the performance scorecard.
4. **Generate Future Predictions**:
   - Select a forecast horizon (**1 Day**, **7 Days**, **14 Days**, or **30 Days**).
   - Click **Generate Forecast**.
   - View predicted prices, expected percentage change, and 95% uncertainty envelope on the interactive chart.

---

## Running the Automated Test Suite

Run all unit and integration tests using pytest:
```powershell
python -m pytest tests/ -v
```

---

## Model Architecture & Design Decisions

### Why Per-Ticker Models (`models/<TICKER>/`)?
Individual stocks have vastly different volatilities, pricing scales (e.g., ₹3,500 for TCS vs $180 for AAPL), and cyclical patterns. Training isolated models guarantees tailored weights and dedicated scalers without cross-contamination.

### Zero-Leakage Preprocessing
In standard machine learning, fitting scalers across the entire dataset exposes the model to future minimum and maximum price levels, creating artificially optimistic evaluation scores. In this application:
- Only historical training samples $t \in [0, T_{\text{split}}]$ are provided to `.fit_transform()`.
- Future test samples $t > T_{\text{split}}$ are strictly processed with `.transform()`.

---

## Disclaimer

> **Important Financial Disclaimer**:
> This software is intended solely for educational, academic, and research purposes. Stock market prices are influenced by unforeseen geopolitical events, macroeconomic indicators, and market sentiment that cannot be captured purely by historical price sequences. The price forecasts and indicators generated by this application are **statistical estimates and are not financial advice or guarantees of future performance**. Do not base real financial investments on these predictions.

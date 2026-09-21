"""
Model training pipeline module.
Executes end-to-end training with strict leakage prevention and artifact persistence.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np

from config import (
    MODELS_DIR,
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_TRAIN_PERIOD,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_SPLIT_RATIO,
)
from src.data_fetcher import get_market_data_provider
from src.feature_engineering import engineer_all_features
from src.preprocessing import DataPreprocessor
from src.model import build_lstm_model, get_default_callbacks, save_keras_model
from src.evaluation import calculate_metrics
from src.utils import get_logger, normalize_ticker, get_safe_ticker_folder, to_python_type

logger = get_logger("trainer")


class TrainingPipeline:
    """
    Manages end-to-end training of the LSTM model for any stock ticker.
    """

    def __init__(
        self,
        ticker: str,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        epochs: int = DEFAULT_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        period: str = DEFAULT_TRAIN_PERIOD,
    ):
        self.ticker = normalize_ticker(ticker)
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.period = period
        self.safe_ticker = get_safe_ticker_folder(self.ticker)
        self.model_dir = MODELS_DIR / self.safe_ticker
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Executes the complete training workflow.
        Returns: (metrics_dict, None) on success, or (None, error_message) on failure.
        """
        logger.info(f"Starting training pipeline for {self.ticker}...")

        # 1. Fetch Market Data
        provider = get_market_data_provider()
        df_raw, err = provider.fetch_history(self.ticker, period=self.period)
        if err or df_raw is None or len(df_raw) < (self.sequence_length + 30):
            msg = err or f"Insufficient historical data ({len(df_raw) if df_raw is not None else 0} rows) to train model for {self.ticker}."
            logger.error(msg)
            return None, msg

        # 2. Feature Engineering
        df_featured = engineer_all_features(df_raw)

        # 3. Preprocessing (Chronological Split + Zero Leakage Scaler)
        preprocessor = DataPreprocessor(
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            sequence_length=self.sequence_length,
            train_split_ratio=TRAIN_SPLIT_RATIO,
        )

        df_clean = preprocessor.clean_data(df_featured)
        train_df, test_df = preprocessor.chronological_split(df_clean)

        # Fit scaler ONLY on train_df
        (
            scaled_train_x,
            scaled_train_y,
            scaled_test_x,
            scaled_test_y,
        ) = preprocessor.fit_and_scale(train_df, test_df)

        # 4. Window Sequence Generation
        X_train, y_train = preprocessor.create_sequences(scaled_train_x, scaled_train_y)
        X_test, y_test = preprocessor.create_sequences(scaled_test_x, scaled_test_y)

        logger.info(
            f"Prepared sequences for {self.ticker}: X_train {X_train.shape}, X_test {X_test.shape}"
        )

        # 5. Build and Compile LSTM
        n_features = len(FEATURE_COLUMNS)
        model = build_lstm_model(
            sequence_length=self.sequence_length,
            n_features=n_features,
            lstm_units=(64, 32),
            dropout_rate=0.2,
        )

        # 6. Train with EarlyStopping & Checkpoint
        checkpoint_path = self.model_dir / "best_checkpoint.keras"
        callbacks = get_default_callbacks(checkpoint_path=checkpoint_path, patience=5)

        history = model.fit(
            X_train,
            y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1,
            shuffle=False,  # Preserve chronological order within minibatches
        )

        # 7. Evaluate on Test Set
        scaled_preds = model.predict(X_test, verbose=0).flatten()
        actual_test_prices = preprocessor.inverse_transform_targets(y_test)
        pred_test_prices = preprocessor.inverse_transform_targets(scaled_preds)

        test_metrics = calculate_metrics(actual_test_prices, pred_test_prices)
        logger.info(f"Evaluation metrics for {self.ticker}: {test_metrics}")

        # Calculate residual standard deviation for prediction uncertainty bounds
        residuals = actual_test_prices - pred_test_prices
        residual_std = float(np.std(residuals))

        # 8. Save Artifacts
        model_file = self.model_dir / "model.keras"
        save_keras_model(model, model_file)

        scaler_file = self.model_dir / "scaler.pkl"
        preprocessor.save_scalers(scaler_file)

        # Training metadata
        metadata = {
            "ticker": self.ticker,
            "trained_at": datetime.now().isoformat(),
            "data_start": str(df_clean.index[0].date()),
            "data_end": str(df_clean.index[-1].date()),
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "sequence_length": self.sequence_length,
            "epochs_completed": len(history.history["loss"]),
            "features": FEATURE_COLUMNS,
            "test_metrics": test_metrics,
            "residual_std": round(residual_std, 4),
            "final_train_loss": round(float(history.history["loss"][-1]), 6),
            "final_val_loss": round(float(history.history["val_loss"][-1]), 6),
        }

        metadata_file = self.model_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # Remove temporary checkpoint file if exists
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
            except Exception:
                pass

        logger.info(f"Training completed and artifacts persisted for {self.ticker}")
        return to_python_type(metadata), None


def train_stock_model(
    ticker: str,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    period: str = DEFAULT_TRAIN_PERIOD,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Convenience function to trigger pipeline."""
    pipeline = TrainingPipeline(
        ticker=ticker,
        epochs=epochs,
        batch_size=batch_size,
        period=period,
    )
    return pipeline.run()

"""
LSTM Model Architecture and serialization module using TensorFlow/Keras.
"""

from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from config import DEFAULT_SEQUENCE_LENGTH
from src.utils import get_logger

logger = get_logger("model")


def build_lstm_model(
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    n_features: int = 1,
    lstm_units: Tuple[int, int] = (64, 32),
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
) -> Sequential:
    """
    Constructs a stacked LSTM architecture for time-series forecasting.

    Architecture:
    - Input: (sequence_length, n_features)
    - LSTM 1: lstm_units[0] units, return_sequences=True
    - Dropout: dropout_rate
    - LSTM 2: lstm_units[1] units, return_sequences=False
    - Dropout: dropout_rate
    - Dense: 16 units, ReLU activation
    - Output: 1 unit, Linear activation (scaled price)
    """
    model = Sequential([
        Input(shape=(sequence_length, n_features)),
        LSTM(units=lstm_units[0], return_sequences=True),
        Dropout(rate=dropout_rate),
        LSTM(units=lstm_units[1], return_sequences=False),
        Dropout(rate=dropout_rate),
        Dense(units=16, activation="relu"),
        Dense(units=1, activation="linear"),
    ])

    optimizer = Adam(learning_rate=learning_rate)
    # Huber loss is less sensitive to extreme price spikes/outliers than MSE
    model.compile(optimizer=optimizer, loss="huber", metrics=["mae", "mse"])
    return model


def get_default_callbacks(
    checkpoint_path: Optional[Path] = None, patience: int = 5
) -> list:
    """
    Returns standard training callbacks:
    - EarlyStopping: prevents overfitting and stops when validation loss stops improving.
    - ReduceLROnPlateau: adaptively lowers learning rate for finer convergence.
    - ModelCheckpoint: saves best weights (if path provided).
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            ModelCheckpoint(
                filepath=str(checkpoint_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )

    return callbacks


def save_keras_model(model: Sequential, filepath: Path) -> None:
    """Saves model to standard Keras format (.keras)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(filepath))
    logger.info(f"Saved model to {filepath}")


def load_saved_model(filepath: Path) -> Optional[Sequential]:
    """Loads model from .keras file."""
    if not filepath.exists():
        logger.warning(f"Model file does not exist: {filepath}")
        return None
    try:
        model = load_model(str(filepath))
        logger.info(f"Loaded model from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {e}")
        return None

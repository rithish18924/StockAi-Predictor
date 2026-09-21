"""
Data preprocessing module.
Ensures zero data leakage:
- Chronological train/test split (no shuffling)
- Scalers fitted strictly on training data only
- Sliding window sequence creation for LSTM
"""

import pickle
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config import DEFAULT_SEQUENCE_LENGTH, TRAIN_SPLIT_RATIO, FEATURE_COLUMNS, TARGET_COLUMN
from src.utils import get_logger

logger = get_logger("preprocessing")


class DataPreprocessor:
    """
    Handles time-series cleaning, scaling, and windowing without data leakage.
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        target_column: str = TARGET_COLUMN,
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
        train_split_ratio: float = TRAIN_SPLIT_RATIO,
    ):
        self.feature_columns = feature_columns or FEATURE_COLUMNS
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.train_split_ratio = train_split_ratio

        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans and chronologically sorts the raw DataFrame.
        Removes duplicates and handles missing values safely.
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None.")

        df_clean = df.copy()

        # Sort chronologically
        df_clean.sort_index(ascending=True, inplace=True)

        # Remove duplicate index timestamps if any
        df_clean = df_clean[~df_clean.index.duplicated(keep="first")]

        # Forward fill then backward fill for any zero/nan entries in prices
        df_clean.ffill(inplace=True)
        df_clean.bfill(inplace=True)

        return df_clean

    def chronological_split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically.
        STRICT RULE: NO RANDOM SHUFFLING!
        """
        total_rows = len(df)
        if total_rows < self.sequence_length + 10:
            raise ValueError(
                f"Insufficient data ({total_rows} rows). Need at least {self.sequence_length + 10} rows."
            )

        split_idx = int(total_rows * self.train_split_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        logger.info(
            f"Chronological split: {len(train_df)} train samples ({train_df.index[0].date()} to {train_df.index[-1].date()}), "
            f"{len(test_df)} test samples ({test_df.index[0].date()} to {test_df.index[-1].date()})"
        )
        return train_df, test_df

    def fit_and_scale(
        self, train_df: pd.DataFrame, test_df: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        STRICT ZERO-LEAKAGE RULE:
        Scalers are fitted ONLY on train_df.
        test_df is transformed using the fitted scaler, never fitted.
        """
        # Validate required columns exist
        for col in self.feature_columns:
            if col not in train_df.columns:
                raise ValueError(f"Feature column '{col}' missing from DataFrame.")

        train_features = train_df[self.feature_columns].values
        train_targets = train_df[[self.target_column]].values

        # Fit scalers strictly on training data
        scaled_train_features = self.feature_scaler.fit_transform(train_features)
        scaled_train_targets = self.target_scaler.fit_transform(train_targets).flatten()
        self.is_fitted = True

        scaled_test_features = None
        scaled_test_targets = None

        if test_df is not None and not test_df.empty:
            test_features = test_df[self.feature_columns].values
            test_targets = test_df[[self.target_column]].values

            # Transform using train scaler
            scaled_test_features = self.feature_scaler.transform(test_features)
            scaled_test_targets = self.target_scaler.transform(test_targets).flatten()

        return (
            scaled_train_features,
            scaled_train_targets,
            scaled_test_features,
            scaled_test_targets,
        )

    def create_sequences(
        self, features: np.ndarray, targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Creates rolling sequence windows:
        X[i] = features[i : i + sequence_length]
        y[i] = targets[i + sequence_length]
        """
        X, y = [], []
        num_samples = len(features) - self.sequence_length

        if num_samples <= 0:
            raise ValueError(
                f"Data length ({len(features)}) is too short for sequence length ({self.sequence_length})."
            )

        for i in range(num_samples):
            X.append(features[i : i + self.sequence_length])
            y.append(targets[i + self.sequence_length])

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def inverse_transform_targets(self, scaled_values: np.ndarray) -> np.ndarray:
        """Inverse transforms target values back to original stock price currency."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor scalers have not been fitted.")
        reshaped = np.array(scaled_values).reshape(-1, 1)
        return self.target_scaler.inverse_transform(reshaped).flatten()

    def transform_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """Transforms a DataFrame of features using the fitted feature scaler."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor scalers have not been fitted.")
        features = features_df[self.feature_columns].values
        return self.feature_scaler.transform(features)

    def save_scalers(self, filepath: Path) -> None:
        """Serializes scalers and metadata to disk."""
        payload = {
            "feature_scaler": self.feature_scaler,
            "target_scaler": self.target_scaler,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "sequence_length": self.sequence_length,
            "train_split_ratio": self.train_split_ratio,
            "is_fitted": self.is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"Saved scalers to {filepath}")

    @classmethod
    def load_scalers(cls, filepath: Path) -> "DataPreprocessor":
        """Deserializes scalers and returns an initialized DataPreprocessor instance."""
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found at {filepath}")

        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        instance = cls(
            feature_columns=payload["feature_columns"],
            target_column=payload["target_column"],
            sequence_length=payload["sequence_length"],
            train_split_ratio=payload["train_split_ratio"],
        )
        instance.feature_scaler = payload["feature_scaler"]
        instance.target_scaler = payload["target_scaler"]
        instance.is_fitted = payload["is_fitted"]
        logger.info(f"Loaded scalers from {filepath}")
        return instance

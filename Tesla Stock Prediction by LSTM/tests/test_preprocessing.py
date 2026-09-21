"""
Unit tests for data preprocessing and zero-leakage verification.
"""

import unittest
import numpy as np
import pandas as pd

from src.preprocessing import DataPreprocessor
from src.feature_engineering import engineer_all_features
from config import FEATURE_COLUMNS, TARGET_COLUMN


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        # Generate 150 days of synthetic price data
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=150, freq="B")
        base_price = 100.0
        returns = np.random.normal(0.001, 0.02, size=150)
        price_series = base_price * np.cumprod(1 + returns)

        df = pd.DataFrame({
            "Open": price_series * 0.99,
            "High": price_series * 1.02,
            "Low": price_series * 0.98,
            "Close": price_series,
            "Volume": np.random.randint(100000, 500000, size=150),
        }, index=dates)

        self.df_raw = df
        self.df_featured = engineer_all_features(df)

    def test_chronological_split(self):
        preprocessor = DataPreprocessor(sequence_length=20, train_split_ratio=0.8)
        clean_df = preprocessor.clean_data(self.df_featured)
        train_df, test_df = preprocessor.chronological_split(clean_df)

        # STRICT VERIFICATION: Train timestamps must strictly precede test timestamps
        self.assertLess(train_df.index[-1], test_df.index[0])
        self.assertEqual(len(train_df) + len(test_df), len(clean_df))
        self.assertEqual(len(train_df), int(len(clean_df) * 0.8))

    def test_zero_leakage_scaling(self):
        preprocessor = DataPreprocessor(
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            sequence_length=20,
            train_split_ratio=0.8
        )
        clean_df = preprocessor.clean_data(self.df_featured)
        train_df, test_df = preprocessor.chronological_split(clean_df)

        # Force an extreme outlier into the test set to verify train scaler is unaffected
        test_df_copy = test_df.copy()
        test_df_copy.iloc[-1, test_df_copy.columns.get_loc("Close")] = 999999.0

        train_x, train_y, test_x, test_y = preprocessor.fit_and_scale(train_df, test_df_copy)

        # Train targets should strictly be bounded [0, 1] because scaler fit on it
        self.assertAlmostEqual(train_y.min(), 0.0, places=5)
        self.assertAlmostEqual(train_y.max(), 1.0, places=5)

        # Test targets transformed with train scaler will exceed 1.0 because of the extreme test value
        # This PROVES test data was NOT used to fit the scaler!
        self.assertGreater(test_y.max(), 1.0)

    def test_create_sequences_shape(self):
        seq_len = 15
        preprocessor = DataPreprocessor(
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            sequence_length=seq_len,
        )
        clean_df = preprocessor.clean_data(self.df_featured)
        train_df, _ = preprocessor.chronological_split(clean_df)
        train_x, train_y, _, _ = preprocessor.fit_and_scale(train_df)

        X, y = preprocessor.create_sequences(train_x, train_y)

        expected_samples = len(train_x) - seq_len
        self.assertEqual(X.shape, (expected_samples, seq_len, len(FEATURE_COLUMNS)))
        self.assertEqual(y.shape, (expected_samples,))

    def test_inverse_transform(self):
        preprocessor = DataPreprocessor(
            feature_columns=FEATURE_COLUMNS,
            target_column=TARGET_COLUMN,
            sequence_length=20,
        )
        clean_df = preprocessor.clean_data(self.df_featured)
        train_df, _ = preprocessor.chronological_split(clean_df)
        _, train_y, _, _ = preprocessor.fit_and_scale(train_df)

        original_close = train_df[TARGET_COLUMN].values
        reconstructed = preprocessor.inverse_transform_targets(train_y)

        np.testing.assert_allclose(original_close, reconstructed, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()

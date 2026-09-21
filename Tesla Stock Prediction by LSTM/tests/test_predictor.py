"""
Unit tests for predictor engine and evaluation metrics.
"""

import unittest
import numpy as np
from src.predictor import StockPredictor
from src.evaluation import calculate_metrics, get_metric_explanations


class TestPredictorAndEvaluation(unittest.TestCase):

    def test_evaluation_metrics_exact(self):
        y_true = np.array([100.0, 102.0, 105.0, 110.0])
        y_pred = np.array([101.0, 101.0, 107.0, 108.0])

        metrics = calculate_metrics(y_true, y_pred)

        self.assertIn("mae", metrics)
        self.assertIn("mse", metrics)
        self.assertIn("rmse", metrics)
        self.assertIn("mape", metrics)
        self.assertIn("directional_accuracy", metrics)
        self.assertIn("r2_score", metrics)

        # MAE: (|100-101| + |102-101| + |105-107| + |110-108|) / 4 = (1 + 1 + 2 + 2) / 4 = 1.5
        self.assertAlmostEqual(metrics["mae"], 1.5, places=3)
        # MSE: (1^2 + 1^2 + 2^2 + 2^2) / 4 = 10 / 4 = 2.5
        self.assertAlmostEqual(metrics["mse"], 2.5, places=3)
        # RMSE: sqrt(2.5) ~= 1.5811
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(2.5), places=3)

    def test_metric_explanations_present(self):
        explanations = get_metric_explanations()
        self.assertIn("mae", explanations)
        self.assertIn("rmse", explanations)
        self.assertIn("mape", explanations)
        self.assertIn("directional_accuracy", explanations)

    def test_predictor_untrained_ticker(self):
        predictor = StockPredictor("UNTRAINED_RANDOM_STOCK_XYZ")
        self.assertFalse(predictor.is_model_available())

        payload, err = predictor.predict_future(days=7)
        self.assertIsNone(payload)
        self.assertIn("No trained model found", err)


if __name__ == "__main__":
    unittest.main()

"""
Model evaluation module.
Calculates error metrics and generates plain-language explanations.
"""

from typing import Dict, Any
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from src.utils import get_logger, to_python_type

logger = get_logger("evaluation")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Computes regression metrics comparing true vs predicted prices:
    - MAE: Mean Absolute Error
    - MSE: Mean Squared Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error (in %)
    - Directional Accuracy: % of times predicted price change direction matches actual
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    if len(y_true) != len(y_pred):
        raise ValueError(f"Lengths differ: y_true ({len(y_true)}) vs y_pred ({len(y_pred)})")

    if len(y_true) == 0:
        return {}

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    # Avoid division by zero in MAPE
    mask = y_true != 0
    if np.any(mask):
        mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100.0
    else:
        mape = 0.0

    # Directional Accuracy (comparing day-to-day sign changes)
    if len(y_true) > 1:
        actual_diff = np.diff(y_true)
        pred_diff = np.diff(y_pred)
        correct_directions = np.sum(np.sign(actual_diff) == np.sign(pred_diff))
        directional_acc = (correct_directions / len(actual_diff)) * 100.0
    else:
        directional_acc = 50.0

    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0.0

    return {
        "mae": round(float(mae), 4),
        "mse": round(float(mse), 4),
        "rmse": round(float(rmse), 4),
        "mape": round(float(mape), 2),
        "directional_accuracy": round(float(directional_acc), 1),
        "r2_score": round(float(r2), 4),
    }


def get_metric_explanations() -> Dict[str, str]:
    """Provides clear, plain-English definitions for financial dashboard users."""
    return {
        "mae": "Mean Absolute Error: Average dollar/rupee deviation between predicted price and actual price.",
        "rmse": "Root Mean Squared Error: Measures standard deviation of prediction errors, giving higher penalty to larger mistakes.",
        "mse": "Mean Squared Error: Average squared difference between predicted and actual prices.",
        "mape": "Mean Absolute Percentage Error: Average percentage error relative to actual stock price.",
        "directional_accuracy": "Percentage of test days where the model correctly anticipated whether the price moved up or down.",
        "r2_score": "Coefficient of determination: proportion of price variance captured by the model on test data.",
    }

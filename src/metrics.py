"""
Evaluation metrics module.

Provides functions to evaluate regression model performance
with standard and additional metrics.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(y_true, y_pred) -> dict:
    """
    Compute regression evaluation metrics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        Dictionary containing:
            - MAE   : Mean Absolute Error
            - RMSE  : Root Mean Squared Error
            - R2    : R-squared (coefficient of determination)
            - MAPE  : Mean Absolute Percentage Error (%)
            - CV_RMSE : Coefficient of Variation of RMSE (%)
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE (guard against division by zero)
    non_zero = y_true != 0
    if non_zero.sum() > 0:
        mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    else:
        mape = float("inf")

    # CV(RMSE) — RMSE normalized by the mean of actual values
    mean_actual = np.mean(y_true)
    cv_rmse = (rmse / mean_actual * 100) if mean_actual != 0 else float("inf")

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
        "CV_RMSE": round(cv_rmse, 2),
    }


def print_metrics(metrics: dict, model_name: str = "Model"):
    """Pretty-print evaluation metrics for a model."""
    print(f"\n{'=' * 40}")
    print(f"  {model_name}")
    print(f"{'=' * 40}")
    for key, value in metrics.items():
        unit = "%" if key in ("MAPE", "CV_RMSE") else ""
        print(f"  {key:<10}: {value}{unit}")

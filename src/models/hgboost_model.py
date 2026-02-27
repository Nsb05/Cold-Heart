"""
HGBoost (Hyperoptimized Gradient Boosting) model for energy prediction.

Uses the hgboost library which automatically performs hyperparameter
optimization via Bayesian optimization (hyperopt) across multiple
gradient boosting backends (XGBoost, LightGBM, CatBoost).
"""

from hgboost import hgboost

from config import HGBOOST_PARAMS


def train(X_train, y_train):
    """
    Train an HGBoost model with automatic hyperparameter tuning.

    HGBoost internally runs Bayesian optimization to find the
    best hyperparameters across gradient boosting frameworks.

    Args:
        X_train: Training features (DataFrame or array).
        y_train: Training target values (Series or array).

    Returns:
        Fitted hgboost model object.
    """
    model = hgboost(
        max_eval=HGBOOST_PARAMS["max_eval"],
        threshold=HGBOOST_PARAMS["threshold"],
        cv=HGBOOST_PARAMS["cv"],
        random_state=HGBOOST_PARAMS["random_state"],
        verbose=HGBOOST_PARAMS["verbose"],
    )

    import numpy as np
    model.xgboost_reg(X_train, np.array(y_train))

    return model


def predict(model, X_test):
    """
    Generate predictions using the trained HGBoost model.

    Args:
        model: Fitted hgboost model object.
        X_test: Test features (DataFrame or array).

    Returns:
        Array of predicted values.
    """
    result = model.predict(X_test)
    # hgboost.predict returns (y_pred, y_proba) tuple
    return result[0]

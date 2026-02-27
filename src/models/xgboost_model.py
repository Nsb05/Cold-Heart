"""
XGBoost model for energy consumption prediction.

Gradient boosted decision trees — typically the strongest
tabular ML model for structured/feature-engineered data.
"""

from xgboost import XGBRegressor

from config import XGBOOST_PARAMS


def train(X_train, y_train):
    """
    Train an XGBoost regressor with configured hyperparameters.

    Args:
        X_train: Training features.
        y_train: Training target values.

    Returns:
        Fitted XGBRegressor model.
    """
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Generate predictions using the trained XGBoost model.

    Args:
        model: Fitted XGBRegressor.
        X_test: Test features.

    Returns:
        Array of predicted values.
    """
    return model.predict(X_test)

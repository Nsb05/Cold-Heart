"""
Ridge Regression model for energy consumption prediction.

Uses L2 regularization to reduce overfitting compared to
standard Linear Regression, penalizing large coefficients.
"""

from sklearn.linear_model import Ridge
from config import RIDGE_PARAMS


def train(X_train, y_train):
    """
    Train a Ridge Regression model.

    Args:
        X_train: Training features.
        y_train: Training target values.

    Returns:
        Fitted Ridge model.
    """
    model = Ridge(**RIDGE_PARAMS)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Generate predictions using the trained model.

    Args:
        model: Fitted Ridge model.
        X_test: Test features.

    Returns:
        Array of predicted values.
    """
    return model.predict(X_test)

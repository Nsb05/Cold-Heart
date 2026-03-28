"""
Lasso Regression model for energy consumption prediction.

Uses L1 regularization for automatic feature selection and
sparsity, driving less important feature coefficients to zero.
"""

from sklearn.linear_model import Lasso
from config import LASSO_PARAMS


def train(X_train, y_train):
    """
    Train a Lasso Regression model.

    Args:
        X_train: Training features.
        y_train: Training target values.

    Returns:
        Fitted Lasso model.
    """
    model = Lasso(**LASSO_PARAMS)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Generate predictions using the trained model.

    Args:
        model: Fitted Lasso model.
        X_test: Test features.

    Returns:
        Array of predicted values.
    """
    return model.predict(X_test)

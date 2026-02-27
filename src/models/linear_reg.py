"""
Linear Regression model for energy consumption prediction.

Serves as the baseline model for comparison against more
complex ML and DL approaches.
"""

from sklearn.linear_model import LinearRegression


def train(X_train, y_train):
    """
    Train a Linear Regression model.

    Args:
        X_train: Training features.
        y_train: Training target values.

    Returns:
        Fitted LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Generate predictions using the trained model.

    Args:
        model: Fitted LinearRegression model.
        X_test: Test features.

    Returns:
        Array of predicted values.
    """
    return model.predict(X_test)

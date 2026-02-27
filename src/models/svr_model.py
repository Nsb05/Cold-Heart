"""
Support Vector Regression (SVR) model for energy prediction.

Uses StandardScaler internally since SVR is sensitive to
feature magnitudes. The scaler is fitted on training data only.
"""

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from config import SVR_PARAMS


def train(X_train, y_train):
    """
    Train an SVR model with feature scaling.

    Args:
        X_train: Training features (unscaled).
        y_train: Training target values.

    Returns:
        Tuple of (fitted SVR model, fitted StandardScaler).
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = SVR(**SVR_PARAMS)
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict(model_and_scaler, X_test):
    """
    Generate predictions using the trained SVR model.

    Applies the same scaling transform learned during training.

    Args:
        model_and_scaler: Tuple of (fitted SVR, fitted StandardScaler).
        X_test: Test features (unscaled).

    Returns:
        Array of predicted values.
    """
    model, scaler = model_and_scaler
    X_test_scaled = scaler.transform(X_test)
    return model.predict(X_test_scaled)

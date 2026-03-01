"""
Support Vector Regression (SVR) model for energy prediction.

Uses StandardScaler for both features and target since SVR
is sensitive to feature magnitudes. RBF kernel captures
non-linear patterns in the energy data.
"""

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from config import SVR_PARAMS


def train(X_train, y_train):
    """
    Train an SVR model with feature and target scaling.

    Args:
        X_train: Training features (unscaled).
        y_train: Training target values.

    Returns:
        Tuple of (fitted SVR model, fitted feature scaler, fitted target scaler).
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    model = SVR(**SVR_PARAMS)
    model.fit(X_train_scaled, y_train_scaled)

    return model, scaler_X, scaler_y


def predict(model_and_scalers, X_test):
    """
    Generate predictions using the trained SVR model.

    Applies the same scaling transform learned during training
    and inverse-transforms predictions back to original scale.

    Args:
        model_and_scalers: Tuple of (fitted SVR, feature scaler, target scaler).
        X_test: Test features (unscaled).

    Returns:
        Array of predicted values in original scale.
    """
    model, scaler_X, scaler_y = model_and_scalers
    X_test_scaled = scaler_X.transform(X_test)
    pred_scaled = model.predict(X_test_scaled)
    return scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

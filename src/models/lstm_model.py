"""
LSTM (Long Short-Term Memory) model for energy prediction.

Uses windowed sequences of past observations to predict
the next timestep's energy consumption. Includes internal
scaling for both features and target.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from config import LSTM_PARAMS


def build_model(input_shape):
    """
    Build and compile the LSTM architecture.

    Architecture:
        Input → LSTM(64) → Dense(32, ReLU) → Dense(1)

    Args:
        input_shape: Shape of a single input sample (sequence_length, n_features).

    Returns:
        Compiled Keras Sequential model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(LSTM_PARAMS["units"]),
        layers.Dense(LSTM_PARAMS["dense_units"], activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=LSTM_PARAMS["optimizer"],
        loss=LSTM_PARAMS["loss"],
    )

    return model


def train(X_train_seq, y_train_seq):
    """
    Scale data and train the LSTM model.

    Args:
        X_train_seq: 3D array of shape (n_samples, sequence_length, n_features).
        y_train_seq: 1D array of target values.

    Returns:
        Tuple of (trained model, feature scaler, target scaler).
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Scale features
    n_samples, seq_len, n_features = X_train_seq.shape
    X_scaled = scaler_X.fit_transform(
        X_train_seq.reshape(-1, n_features)
    ).reshape(n_samples, seq_len, n_features)

    # Scale target
    y_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1))

    # Build and train
    model = build_model(input_shape=(seq_len, n_features))
    model.fit(
        X_scaled, y_scaled,
        epochs=LSTM_PARAMS["epochs"],
        batch_size=LSTM_PARAMS["batch_size"],
        verbose=1,
    )

    return model, scaler_X, scaler_y


def predict(model_and_scalers, X_test_seq):
    """
    Generate and inverse-transform predictions.

    Args:
        model_and_scalers: Tuple of (model, scaler_X, scaler_y).
        X_test_seq: 3D array of test sequences.

    Returns:
        Tuple of (predictions, inverse-transformed ground truth placeholder).
    """
    model, scaler_X, scaler_y = model_and_scalers

    n_samples, seq_len, n_features = X_test_seq.shape
    X_scaled = scaler_X.transform(
        X_test_seq.reshape(-1, n_features)
    ).reshape(n_samples, seq_len, n_features)

    pred_scaled = model.predict(X_scaled)
    predictions = scaler_y.inverse_transform(pred_scaled).flatten()

    return predictions

"""
1D Convolutional Neural Network (CNN) model for energy prediction.

Uses convolutional layers to extract local temporal patterns
from windowed sequences. Shares scaling logic with LSTM.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

from config import CNN_PARAMS


def build_model(input_shape):
    """
    Build and compile the 1D CNN architecture.

    Architecture:
        Input → Conv1D(64, k=3) → MaxPool(2) → Flatten → Dense(32) → Dense(1)

    Args:
        input_shape: Shape of a single input sample (sequence_length, n_features).

    Returns:
        Compiled Keras Sequential model.
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(
            filters=CNN_PARAMS["filters"],
            kernel_size=CNN_PARAMS["kernel_size"],
            activation="relu",
        ),
        layers.MaxPooling1D(pool_size=CNN_PARAMS["pool_size"]),
        layers.Flatten(),
        layers.Dense(CNN_PARAMS["dense_units"], activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=CNN_PARAMS["optimizer"],
        loss=CNN_PARAMS["loss"],
    )

    return model


def train(X_train_seq, y_train_seq):
    """
    Scale data and train the CNN model.

    Args:
        X_train_seq: 3D array (n_samples, sequence_length, n_features).
        y_train_seq: 1D array of target values (already scaled).

    Returns:
        Tuple of (trained model, feature scaler, target scaler).
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    n_samples, seq_len, n_features = X_train_seq.shape
    X_scaled = scaler_X.fit_transform(
        X_train_seq.reshape(-1, n_features)
    ).reshape(n_samples, seq_len, n_features)

    y_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1))

    model = build_model(input_shape=(seq_len, n_features))
    model.fit(
        X_scaled, y_scaled,
        epochs=CNN_PARAMS["epochs"],
        batch_size=CNN_PARAMS["batch_size"],
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
        Array of inverse-transformed predictions.
    """
    model, scaler_X, scaler_y = model_and_scalers

    n_samples, seq_len, n_features = X_test_seq.shape
    X_scaled = scaler_X.transform(
        X_test_seq.reshape(-1, n_features)
    ).reshape(n_samples, seq_len, n_features)

    pred_scaled = model.predict(X_scaled)
    predictions = scaler_y.inverse_transform(pred_scaled).flatten()

    return predictions

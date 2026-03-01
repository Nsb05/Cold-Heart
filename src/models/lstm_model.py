"""
LSTM (Long Short-Term Memory) model for energy prediction.

Uses windowed sequences of past observations to predict
the next timestep's energy consumption. Includes internal
scaling, gradient clipping, and adaptive callbacks for
stable training across different data splits.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from config import LSTM_PARAMS

# Set seed for reproducibility
tf.random.set_seed(42)


def build_model(input_shape):
    """
    Build and compile the LSTM architecture.

    Architecture:
        Input → LSTM(64) → Dropout(0.2) → Dense(32, ReLU) → Dense(1)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(LSTM_PARAMS["units"]),
        layers.Dropout(0.2),
        layers.Dense(LSTM_PARAMS["dense_units"], activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3,
            clipnorm=1.0,   # gradient clipping for stable training
        ),
        loss=LSTM_PARAMS["loss"],
    )

    return model


def train(X_train_seq, y_train_seq):
    """
    Scale data and train the LSTM model with EarlyStopping
    and ReduceLROnPlateau callbacks.
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    n_samples, seq_len, n_features = X_train_seq.shape
    X_scaled = scaler_X.fit_transform(
        X_train_seq.reshape(-1, n_features)
    ).reshape(n_samples, seq_len, n_features)

    y_scaled = scaler_y.fit_transform(y_train_seq.reshape(-1, 1))

    model = build_model(input_shape=(seq_len, n_features))

    cb = [
        callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    model.fit(
        X_scaled, y_scaled,
        epochs=LSTM_PARAMS["epochs"],
        batch_size=LSTM_PARAMS["batch_size"],
        callbacks=cb,
        verbose=1,
    )

    return model, scaler_X, scaler_y


def predict(model_and_scalers, X_test_seq):
    """Generate and inverse-transform predictions."""
    model, scaler_X, scaler_y = model_and_scalers

    n_samples, seq_len, n_features = X_test_seq.shape
    X_scaled = scaler_X.transform(
        X_test_seq.reshape(-1, n_features)
    ).reshape(n_samples, seq_len, n_features)

    pred_scaled = model.predict(X_scaled)
    predictions = scaler_y.inverse_transform(pred_scaled).flatten()

    return predictions

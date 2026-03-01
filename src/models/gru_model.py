"""
GRU (Gated Recurrent Unit) model for energy prediction.

A lighter alternative to LSTM with fewer parameters,
using reset and update gates instead of LSTM's three gates.
Optimized with BatchNorm, Dropout, and adaptive learning rate.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from config import GRU_PARAMS


def build_model(input_shape):
    """
    Build and compile the GRU architecture.

    Architecture:
        Input → GRU(64) → BatchNorm → Dropout(0.2)
              → Dense(32, ReLU) → Dense(1)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(GRU_PARAMS["units"]),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(GRU_PARAMS["dense_units"], activation="relu"),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=GRU_PARAMS["loss"],
    )

    return model


def train(X_train_seq, y_train_seq):
    """
    Scale data and train the GRU model with EarlyStopping
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
        epochs=GRU_PARAMS["epochs"],
        batch_size=GRU_PARAMS["batch_size"],
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

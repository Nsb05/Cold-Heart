"""
1D Convolutional Neural Network (CNN) model for energy prediction.

Uses convolutional layers to extract local temporal patterns
from windowed sequences. Includes BatchNorm and Dropout for
regularization with adaptive learning rate.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from config import CNN_PARAMS


def build_model(input_shape):
    """
    Build and compile the 1D CNN architecture.

    Architecture:
        Input → Conv1D(64, k=3, same) → BatchNorm → ReLU
              → Conv1D(32, k=3, same) → BatchNorm → ReLU → MaxPool(2)
              → Flatten → Dense(32, ReLU) → Dropout(0.2) → Dense(1)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(CNN_PARAMS["filters"], kernel_size=CNN_PARAMS["kernel_size"],
                      padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv1D(CNN_PARAMS["filters"] // 2, kernel_size=CNN_PARAMS["kernel_size"],
                      padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=CNN_PARAMS["pool_size"]),
        layers.Flatten(),
        layers.Dense(CNN_PARAMS["dense_units"], activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=CNN_PARAMS["loss"],
    )

    return model


def train(X_train_seq, y_train_seq):
    """
    Scale data and train the CNN model with EarlyStopping
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
        epochs=CNN_PARAMS["epochs"],
        batch_size=CNN_PARAMS["batch_size"],
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

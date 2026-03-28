"""
Transfer Learning LSTM with Progressive Unfreezing.

Three-phase training strategy:
    Phase 1 — Pre-train the full model on source data (2019) to learn
              general temporal energy-consumption patterns.
    Phase 2 — Freeze the LSTM backbone, fine-tune only the dense head
              on target data (2021) so the head adapts to the new domain.
    Phase 3 — Unfreeze the entire model and fine-tune end-to-end with
              a very low learning rate for gentle full adaptation.

This progressive approach avoids catastrophic forgetting while still
allowing the LSTM features to adapt to the target distribution.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks

from config import LSTM_TRANSFER_PARAMS

# Reproducibility
tf.random.set_seed(42)


def build_model(input_shape):
    """
    Build and compile the LSTM architecture.

    Architecture:
        Input → LSTM(64) → Dropout(0.2) → Dense(32, ReLU) → Dense(1)
    """
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(LSTM_TRANSFER_PARAMS["units"], name="lstm_backbone"),
        layers.Dropout(0.2, name="dropout"),
        layers.Dense(LSTM_TRANSFER_PARAMS["dense_units"],
                     activation="relu", name="dense_head"),
        layers.Dense(1, name="output"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=1e-3,
            clipnorm=1.0,
        ),
        loss=LSTM_TRANSFER_PARAMS["loss"],
    )

    return model


def _make_callbacks(min_lr=1e-6):
    """Shared EarlyStopping + ReduceLROnPlateau callbacks."""
    return [
        callbacks.EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="loss", factor=0.5, patience=3, min_lr=min_lr
        ),
    ]


def train(X_source_seq, y_source_seq, X_target_seq, y_target_seq):
    """
    Three-phase progressive-unfreezing transfer learning.

    Phase 1: Pre-train full model on source data (lr=1e-3).
    Phase 2: Freeze LSTM, fine-tune dense layers on target (lr=5e-4).
    Phase 3: Unfreeze all layers, fine-tune full model on target (lr=1e-5).

    Returns:
        (model, scaler_X, scaler_y)
    """
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # --- Fit scalers on combined source + target -----------------------
    n_src, seq_len, n_feat = X_source_seq.shape
    n_tgt = X_target_seq.shape[0]

    combined_X = np.concatenate([
        X_source_seq.reshape(-1, n_feat),
        X_target_seq.reshape(-1, n_feat),
    ])
    combined_y = np.concatenate([
        y_source_seq.reshape(-1, 1),
        y_target_seq.reshape(-1, 1),
    ])

    scaler_X.fit(combined_X)
    scaler_y.fit(combined_y)

    # Scale source data
    X_src_scaled = scaler_X.transform(
        X_source_seq.reshape(-1, n_feat)
    ).reshape(n_src, seq_len, n_feat)
    y_src_scaled = scaler_y.transform(y_source_seq.reshape(-1, 1))

    # Scale target data
    X_tgt_scaled = scaler_X.transform(
        X_target_seq.reshape(-1, n_feat)
    ).reshape(n_tgt, seq_len, n_feat)
    y_tgt_scaled = scaler_y.transform(y_target_seq.reshape(-1, 1))

    # --- Build model ---------------------------------------------------
    model = build_model(input_shape=(seq_len, n_feat))
    P = LSTM_TRANSFER_PARAMS

    # ====== PHASE 1: Pre-train on source data ==========================
    print("   [Transfer] Phase 1/3 — Pre-training on source data ...")
    model.fit(
        X_src_scaled, y_src_scaled,
        epochs=P["pretrain_epochs"],
        batch_size=P["batch_size"],
        callbacks=_make_callbacks(min_lr=1e-6),
        verbose=1,
    )

    # ====== PHASE 2: Freeze LSTM, fine-tune dense head =================
    print("   [Transfer] Phase 2/3 — Fine-tuning dense head on target ...")

    model.get_layer("lstm_backbone").trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=P["finetune_head_lr"],
            clipnorm=1.0,
        ),
        loss=P["loss"],
    )

    model.fit(
        X_tgt_scaled, y_tgt_scaled,
        epochs=P["finetune_head_epochs"],
        batch_size=P["batch_size"],
        callbacks=_make_callbacks(min_lr=1e-7),
        verbose=1,
    )

    # ====== PHASE 3: Unfreeze all, gentle full fine-tune ===============
    print("   [Transfer] Phase 3/3 — Full model fine-tune (low LR) ...")

    model.get_layer("lstm_backbone").trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=P["finetune_full_lr"],
            clipnorm=1.0,
        ),
        loss=P["loss"],
    )

    model.fit(
        X_tgt_scaled, y_tgt_scaled,
        epochs=P["finetune_full_epochs"],
        batch_size=P["batch_size"],
        callbacks=_make_callbacks(min_lr=1e-8),
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

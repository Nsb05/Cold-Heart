"""
Feature engineering module.

Provides functions to create temporal features, lag features,
and windowed sequences for deep learning models.
"""

import numpy as np
import pandas as pd

from config import TARGET, SEQUENCE_LENGTH, LAG_FEATURES


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the DataFrame.

    Features created:
        - hour_sin, hour_cos : Cyclical encoding of the hour of day
        - lag_1              : Energy value 1 hour ago
        - lag_24             : Energy value 24 hours ago

    Args:
        df: DataFrame with a DatetimeIndex and a TARGET column.

    Returns:
        DataFrame with new feature columns (rows with NaN from lags are dropped).
    """
    df = df.copy()

    # Cyclical hour encoding
    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    # Lag features
    for lag in LAG_FEATURES:
        df[f"lag_{lag}"] = df[TARGET].shift(lag)

    df = df.dropna()
    return df


def create_sequences(df: pd.DataFrame):
    """
    Create sliding-window sequences for LSTM / CNN models.

    Each sample consists of `SEQUENCE_LENGTH` consecutive rows as input (X)
    and the TARGET value at the next timestep as output (y).

    Args:
        df: DataFrame with all features and the TARGET column.

    Returns:
        X: np.ndarray of shape (n_samples, SEQUENCE_LENGTH, n_features)
        y: np.ndarray of shape (n_samples,)
    """
    X, y = [], []
    values = df.values
    target_idx = df.columns.get_loc(TARGET)

    for i in range(SEQUENCE_LENGTH, len(df)):
        X.append(values[i - SEQUENCE_LENGTH : i])
        y.append(values[i, target_idx])

    return np.array(X), np.array(y)

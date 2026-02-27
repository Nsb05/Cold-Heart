"""
Data loading and preprocessing module.

Handles reading raw CSV files, time column detection,
datetime parsing, hourly resampling, and missing value imputation.
"""

import numpy as np
import pandas as pd

from config import RESAMPLE_FREQ


def load_data(path: str) -> pd.DataFrame:
    """
    Load smart meter data from a CSV file and preprocess it.

    Steps:
        1. Auto-detect the time/date column
        2. Parse datetime with day-first format
        3. Sort chronologically and set as index
        4. Keep only numeric columns
        5. Resample to hourly frequency
        6. Interpolate missing values

    Args:
        path: Path to the CSV file.

    Returns:
        Cleaned DataFrame with a DatetimeIndex and hourly resolution.
    """
    df = pd.read_csv(path)

    # Auto-detect time column
    time_col = [
        col for col in df.columns
        if "time" in col.lower() or "date" in col.lower()
    ][0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df = df.sort_values(time_col)
    df = df.set_index(time_col)

    # Keep numeric columns only
    df = df.select_dtypes(include=[np.number])

    # Resample to hourly and interpolate gaps
    df = df.resample(RESAMPLE_FREQ).mean()
    df = df.interpolate()

    return df

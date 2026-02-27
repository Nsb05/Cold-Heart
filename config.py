"""
Central configuration for the Cold-Start Energy Prediction pipeline.

All hyperparameters, file paths, and settings are defined here
so they can be easily tuned without modifying model code.
"""

import os

# ============================================================
# DATA PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

DATA_2019_PATH = os.path.join(DATA_DIR, "CEEW - Smart meter data Mathura 2019.csv")
DATA_2020_PATH = os.path.join(DATA_DIR, "CEEW - Smart meter data Mathura 2020.csv")

# ============================================================
# TARGET VARIABLE
# ============================================================

TARGET = "t_kWh"

# ============================================================
# FEATURE ENGINEERING
# ============================================================

SEQUENCE_LENGTH = 48       # Lookback window for LSTM / CNN
LAG_FEATURES = [1, 24]     # Lag periods (hours)
RESAMPLE_FREQ = "H"        # Hourly resampling

# ============================================================
# MODEL HYPERPARAMETERS
# ============================================================

# --- XGBoost ---
XGBOOST_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}

# --- SVR ---
SVR_PARAMS = {
    "kernel": "linear",
    "C": 100,
    "epsilon": 0.01,
    "gamma": "scale",
}

# --- LSTM ---
LSTM_PARAMS = {
    "units": 64,
    "dense_units": 32,
    "epochs": 30,
    "batch_size": 64,
    "optimizer": "adam",
    "loss": "mse",
}

# --- CNN ---
CNN_PARAMS = {
    "filters": 64,
    "kernel_size": 3,
    "pool_size": 2,
    "dense_units": 32,
    "epochs": 30,
    "batch_size": 64,
    "optimizer": "adam",
    "loss": "mse",
}

# --- GRU ---
GRU_PARAMS = {
    "units": 64,
    "dense_units": 32,
    "epochs": 30,
    "batch_size": 64,
    "optimizer": "adam",
    "loss": "mse",
}

# --- BiLSTM ---
BILSTM_PARAMS = {
    "units": 64,
    "dense_units": 32,
    "epochs": 30,
    "batch_size": 64,
    "optimizer": "adam",
    "loss": "mse",
}

# --- HGBoost ---
HGBOOST_PARAMS = {
    "max_eval": 250,
    "threshold": 0.5,
    "cv": 5,
    "random_state": 42,
    "verbose": 3,
}

# --- ARIMA ---
ARIMA_ORDER = (5, 1, 2)

# ============================================================
# PLOTTING
# ============================================================

PLOT_SAMPLES = 200         # Number of data points to display in comparison plot
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_DPI = 150

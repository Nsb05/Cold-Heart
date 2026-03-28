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
DATA_2021_PATH = os.path.join(DATA_DIR, "SM Cleaned Data MH2021.csv")

# Train / Test split ratio (fraction used for testing)
TEST_SIZE = 0.2

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
    "kernel": "rbf",
    "C": 10,
    "epsilon": 0.1,
    "gamma": "scale",
}

# --- Ridge Regression ---
RIDGE_PARAMS = {
    "alpha": 1.0,
    "random_state": 42,
}

# --- Lasso Regression ---
LASSO_PARAMS = {
    "alpha": 0.01,
    "max_iter": 10000,
    "random_state": 42,
}

# --- LSTM ---
LSTM_PARAMS = {
    "units": 64,
    "dense_units": 32,
    "epochs": 80,
    "batch_size": 64,
    "optimizer": "adam",
    "loss": "mse",
}

# --- LSTM Transfer Learning (Progressive Unfreezing) ---
LSTM_TRANSFER_PARAMS = {
    "units": 64,
    "dense_units": 32,
    "pretrain_epochs": 50,       # Phase 1: pre-train on source (2019)
    "finetune_head_epochs": 20,  # Phase 2: freeze LSTM, fine-tune dense on target
    "finetune_full_epochs": 20,  # Phase 3: unfreeze all, gentle full fine-tune
    "batch_size": 64,
    "optimizer": "adam",
    "loss": "mse",
    "finetune_head_lr": 5e-4,    # Phase 2 LR (dense layers only)
    "finetune_full_lr": 1e-5,    # Phase 3 LR (very low, full model)
}

# --- CNN ---
CNN_PARAMS = {
    "filters": 64,
    "kernel_size": 3,
    "pool_size": 2,
    "dense_units": 32,
    "epochs": 80,
    "batch_size": 64,
    "optimizer": "adam",
    "loss": "mse",
}

# --- GRU ---
GRU_PARAMS = {
    "units": 64,
    "dense_units": 32,
    "epochs": 80,
    "batch_size": 64,
    "optimizer": "adam",
    "loss": "mse",
}

# --- BiLSTM ---
BILSTM_PARAMS = {
    "units": 64,
    "dense_units": 32,
    "epochs": 80,
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

# --- ARIMA (auto order selection) ---
ARIMA_AUTO_PARAMS = {
    "start_p": 1,
    "start_q": 1,
    "max_p": 5,
    "max_q": 5,
    "d": None,           # auto-detect differencing order
    "seasonal": False,
    "stepwise": True,    # faster search
}

# ============================================================
# PLOTTING
# ============================================================

PLOT_SAMPLES = 200         # Number of data points to display in comparison plot
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_DPI = 150

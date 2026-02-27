import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from sklearn.svm import SVR

from statsmodels.tsa.arima.model import ARIMA

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==============================
# CONFIG
# ==============================

DATA_2019_PATH = "CEEW - Smart meter data Mathura 2019.csv"
DATA_2020_PATH = "CEEW - Smart meter data Mathura 2020.csv"

TARGET = "t_kWh"
HISTORY = 48  # for LSTM sequence length

# ==============================
# LOAD DATA
# ==============================

def load_data(path):

    df = pd.read_csv(path)

    # detect time column automatically
    time_col = [col for col in df.columns
                if "time" in col.lower() or "date" in col.lower()][0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=True)
    df = df.sort_values(time_col)
    df = df.set_index(time_col)

    # keep numeric only
    df = df.select_dtypes(include=[np.number])

    # hourly resample
    df = df.resample("H").mean()

    # interpolate missing
    df = df.interpolate()

    return df

# ==============================
# FEATURE ENGINEERING
# ==============================

def add_features(df):

    df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)

    df["lag_1"] = df[TARGET].shift(1)
    df["lag_24"] = df[TARGET].shift(24)

    df = df.dropna()

    return df

# ==============================
# LSTM SEQUENCE CREATION
# ==============================

def create_sequences(df):

    X, y = [], []
    values = df.values
    target_idx = df.columns.get_loc(TARGET)

    for i in range(HISTORY, len(df)):
        X.append(values[i - HISTORY:i])
        y.append(values[i, target_idx])

    return np.array(X), np.array(y)

# ==============================
# METRICS
# ==============================

def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2

# ==============================
# MAIN
# ==============================

def main():

    print("\nLoading data...")
    df_2019 = load_data(DATA_2019_PATH)
    df_2020 = load_data(DATA_2020_PATH)

    df_2019 = add_features(df_2019)
    df_2020 = add_features(df_2020)

    print("\nDistribution Check")
    print("2019 mean:", df_2019[TARGET].mean())
    print("2020 mean:", df_2020[TARGET].mean())

    # =============================
    # TRAIN = 2019
    # TEST  = 2020
    # =============================

    X_train = df_2019.drop(columns=[TARGET])
    y_train = df_2019[TARGET]

    X_test = df_2020.drop(columns=[TARGET])
    y_test = df_2020[TARGET]

    # =====================================
    # 1️ LINEAR REGRESSION
    # =====================================

    print("\n===== LINEAR REGRESSION =====")

    linear = LinearRegression()
    linear.fit(X_train, y_train)

    pred_linear = linear.predict(X_test)

    mae, rmse, r2 = evaluate(y_test, pred_linear)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    # =====================================
    # 2 XGBOOST
    # =====================================

    print("\n===== XGBOOST =====")

    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    xgb.fit(X_train, y_train)
    pred_xgb = xgb.predict(X_test)

    mae, rmse, r2 = evaluate(y_test, pred_xgb)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    # =====================================
    # 3 SVM (SVR)
    # =====================================

    print("\n===== SVM (SVR) =====")

    # Scale features (VERY important for SVM)
    scaler_svm = StandardScaler()

    X_train_scaled = scaler_svm.fit_transform(X_train)
    X_test_scaled = scaler_svm.transform(X_test)

    svr = SVR(
        kernel='linear',      
        C=100,
        epsilon=0.01,
        gamma='scale'
    )

    svr.fit(X_train_scaled, y_train)

    pred_svm = svr.predict(X_test_scaled)

    mae, rmse, r2 = evaluate(y_test, pred_svm)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    # =====================================
    # 4 LSTM
    # =====================================

    print("\n===== LSTM =====")

    X_train_seq, y_train_seq = create_sequences(df_2019)
    X_test_seq, y_test_seq = create_sequences(df_2020)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # Fit on TRAIN (2019) only
    X_train_seq = scaler_X.fit_transform(
        X_train_seq.reshape(-1, X_train_seq.shape[-1])
    ).reshape(X_train_seq.shape)

    X_test_seq = scaler_X.transform(
        X_test_seq.reshape(-1, X_test_seq.shape[-1])
    ).reshape(X_test_seq.shape)

    y_train_seq = scaler_y.fit_transform(y_train_seq.reshape(-1, 1))
    y_test_seq_scaled = scaler_y.transform(y_test_seq.reshape(-1, 1))

    model = keras.Sequential([
        layers.Input(shape=X_train_seq.shape[1:]),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(
        X_train_seq,
        y_train_seq,
        epochs=30,
        batch_size=64,
        verbose=1
    )

    pred_lstm_scaled = model.predict(X_test_seq)

    pred_lstm = scaler_y.inverse_transform(pred_lstm_scaled)
    y_test_inv = scaler_y.inverse_transform(y_test_seq_scaled)

    mae, rmse, r2 = evaluate(y_test_inv, pred_lstm)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))


    # =====================================
    # 5 CNN (1D Convolution)
    # =====================================

    print("\n===== CNN (1D) =====")

    X_train_cnn = X_train_seq
    y_train_cnn = y_train_seq
    X_test_cnn = X_test_seq

    cnn_model = keras.Sequential([
        layers.Input(shape=X_train_cnn.shape[1:]),
        layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    cnn_model.compile(optimizer='adam', loss='mse')

    cnn_model.fit(
        X_train_cnn,
        y_train_cnn,
        epochs=30,
        batch_size=64,
        verbose=1
    )

    pred_cnn_scaled = cnn_model.predict(X_test_cnn)

    pred_cnn = scaler_y.inverse_transform(pred_cnn_scaled)

    mae, rmse, r2 = evaluate(y_test_inv, pred_cnn)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))


    # =====================================
    # 6 ARIMA
    # =====================================

    print("\n===== ARIMA =====")

    # Train only on 2019 target
    arima_model = ARIMA(df_2019[TARGET], order=(5,1,2))
    arima_model_fit = arima_model.fit()

    # Forecast length = size of 2020 dataset
    forecast_steps = len(df_2020)

    arima_forecast = arima_model_fit.forecast(steps=forecast_steps)

    # Align sizes (ARIMA may differ slightly)
    arima_forecast = arima_forecast[:len(y_test)]

    mae, rmse, r2 = evaluate(y_test.values[:len(arima_forecast)], arima_forecast)
    print("MAE :", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R²  :", round(r2, 4))

    # =====================================
    # Plot comparison
    # =====================================

    plt.figure()
    plt.plot(y_test.values[:200], label="True")
    plt.plot(pred_linear[:200], label="Linear")
    plt.plot(pred_xgb[:200], label="XGB")
    plt.plot(pred_svm[:200], label="SVM(SVR)")
    plt.legend()
    plt.title("Cold-Start Prediction (Train: 2019 → Test: 2020)")
    plt.show()


if __name__ == "__main__":
    main()

"""
main.py — Cold-Start Energy Consumption Prediction Pipeline

Entry point that orchestrates:
    1. Data loading and preprocessing
    2. Feature engineering
    3. Model training (9 models)
    4. Evaluation and comparison
    5. Visualization

Usage:
    python main.py

All configuration is in config.py.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    DATA_2019_PATH, DATA_2020_PATH, TARGET,
    RESULTS_DIR, PLOT_SAMPLES, FIGURE_DPI,
)
from src.data_loader import load_data
from src.feature_engineering import add_features, create_sequences
from src.metrics import evaluate, print_metrics

from src.models import linear_reg, xgboost_model, svr_model
from src.models import lstm_model, cnn_model, arima_model
from src.models import gru_model, bilstm_model, hgboost_model

warnings.filterwarnings("ignore")


def print_header():
    """Print a stylized project header."""
    print("\n" + "=" * 60)
    print("  Cold-Start Energy Consumption Prediction")
    print("  Train: 2019 │ Test: 2020 │ Target: t_kWh")
    print("=" * 60)


def run_pipeline():
    """Execute the full prediction pipeline."""

    print_header()

    # --------------------------------------------------
    # 1. LOAD DATA
    # --------------------------------------------------
    print("\n.....Loading datasets.....")
    df_2019 = load_data(DATA_2019_PATH)
    df_2020 = load_data(DATA_2020_PATH)

    # --------------------------------------------------
    # 2. FEATURE ENGINEERING
    # --------------------------------------------------
    print("🔧 Engineering features...")
    df_2019 = add_features(df_2019)
    df_2020 = add_features(df_2020)

    print(f"\n   2019 — {len(df_2019)} samples, mean {TARGET}: {df_2019[TARGET].mean():.4f}")
    print(f"   2020 — {len(df_2020)} samples, mean {TARGET}: {df_2020[TARGET].mean():.4f}")

    # --------------------------------------------------
    # 3. PREPARE TRAIN / TEST SPLITS
    # --------------------------------------------------
    X_train = df_2019.drop(columns=[TARGET])
    y_train = df_2019[TARGET]
    X_test = df_2020.drop(columns=[TARGET])
    y_test = df_2020[TARGET]

    # Sequences for DL models
    X_train_seq, y_train_seq = create_sequences(df_2019)
    X_test_seq, y_test_seq = create_sequences(df_2020)

    # --------------------------------------------------
    # 4. TRAIN & EVALUATE ALL MODELS
    # --------------------------------------------------
    all_results = {}
    predictions = {}

    # ---- 1. Linear Regression ----
    print("\n🔹 Training Linear Regression...")
    lr_model = linear_reg.train(X_train, y_train)
    pred_lr = linear_reg.predict(lr_model, X_test)
    all_results["Linear Regression"] = evaluate(y_test, pred_lr)
    predictions["Linear"] = pred_lr
    print_metrics(all_results["Linear Regression"], "Linear Regression")

    # ---- 2. XGBoost ----
    print("\n🔹 Training XGBoost...")
    xgb_model = xgboost_model.train(X_train, y_train)
    pred_xgb = xgboost_model.predict(xgb_model, X_test)
    all_results["XGBoost"] = evaluate(y_test, pred_xgb)
    predictions["XGBoost"] = pred_xgb
    print_metrics(all_results["XGBoost"], "XGBoost")

    # ---- 3. SVR ----
    print("\n🔹 Training SVR...")
    svr_trained = svr_model.train(X_train, y_train)
    pred_svr = svr_model.predict(svr_trained, X_test)
    all_results["SVR"] = evaluate(y_test, pred_svr)
    predictions["SVR"] = pred_svr
    print_metrics(all_results["SVR"], "SVR")

    # ---- 4. LSTM ----
    print("\n🔹 Training LSTM...")
    lstm_trained = lstm_model.train(X_train_seq, y_train_seq)
    pred_lstm = lstm_model.predict(lstm_trained, X_test_seq)
    all_results["LSTM"] = evaluate(y_test_seq, pred_lstm)
    predictions["LSTM"] = pred_lstm
    print_metrics(all_results["LSTM"], "LSTM")

    # ---- 5. CNN ----
    print("\n🔹 Training CNN (1D)...")
    cnn_trained = cnn_model.train(X_train_seq, y_train_seq)
    pred_cnn = cnn_model.predict(cnn_trained, X_test_seq)
    all_results["CNN (1D)"] = evaluate(y_test_seq, pred_cnn)
    predictions["CNN"] = pred_cnn
    print_metrics(all_results["CNN (1D)"], "CNN (1D)")

    # ---- 6. GRU ----
    print("\n🔹 Training GRU...")
    gru_trained = gru_model.train(X_train_seq, y_train_seq)
    pred_gru = gru_model.predict(gru_trained, X_test_seq)
    all_results["GRU"] = evaluate(y_test_seq, pred_gru)
    predictions["GRU"] = pred_gru
    print_metrics(all_results["GRU"], "GRU")

    # ---- 7. BiLSTM ----
    print("\n🔹 Training BiLSTM...")
    bilstm_trained = bilstm_model.train(X_train_seq, y_train_seq)
    pred_bilstm = bilstm_model.predict(bilstm_trained, X_test_seq)
    all_results["BiLSTM"] = evaluate(y_test_seq, pred_bilstm)
    predictions["BiLSTM"] = pred_bilstm
    print_metrics(all_results["BiLSTM"], "BiLSTM")

    # ---- 8. HGBoost ----
    print("\n🔹 Training HGBoost...")
    hgb_model = hgboost_model.train(X_train, y_train)
    pred_hgb = hgboost_model.predict(hgb_model, X_test)
    all_results["HGBoost"] = evaluate(y_test, pred_hgb)
    predictions["HGBoost"] = pred_hgb
    print_metrics(all_results["HGBoost"], "HGBoost")

    # ---- 9. ARIMA ----
    print("\n🔹 Training ARIMA...")
    arima_fitted = arima_model.train(df_2019[TARGET])
    pred_arima = arima_model.predict(arima_fitted, len(df_2020))
    pred_arima = pred_arima[:len(y_test)]
    y_test_arima = y_test.values[:len(pred_arima)]
    all_results["ARIMA"] = evaluate(y_test_arima, pred_arima)
    print_metrics(all_results["ARIMA"], "ARIMA")

    # --------------------------------------------------
    # 5. SUMMARY TABLE
    # --------------------------------------------------
    print("\n\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    summary_df = pd.DataFrame(all_results).T
    summary_df.index.name = "Model"
    print(summary_df.to_string())
    print()

    # --------------------------------------------------
    # 6. SAVE PLOTS
    # --------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Plot 1: Prediction comparison (ML models)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=FIGURE_DPI)

    # ML models
    axes[0].plot(y_test.values[:PLOT_SAMPLES], label="Actual", color="#2c3e50", linewidth=2)
    colors_ml = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    for (name, pred), color in zip(
        [("Linear", pred_lr), ("XGBoost", pred_xgb), ("SVR", pred_svr), ("HGBoost", pred_hgb)],
        colors_ml
    ):
        axes[0].plot(pred[:PLOT_SAMPLES], label=name, alpha=0.8, color=color)
    axes[0].set_title("ML Models — Cold-Start Prediction (Train: 2019 → Test: 2020)", fontsize=13)
    axes[0].set_ylabel("Energy (kWh)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # DL models
    axes[1].plot(y_test_seq[:PLOT_SAMPLES], label="Actual", color="#2c3e50", linewidth=2)
    colors_dl = ["#9b59b6", "#e67e22", "#1abc9c", "#e91e63"]
    for (name, pred), color in zip(
        [("LSTM", pred_lstm), ("CNN", pred_cnn), ("GRU", pred_gru), ("BiLSTM", pred_bilstm)],
        colors_dl
    ):
        axes[1].plot(pred[:PLOT_SAMPLES], label=name, alpha=0.8, color=color)
    axes[1].set_title("Deep Learning Models — Cold-Start Prediction", fontsize=13)
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Energy (kWh)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "model_comparison.png")
    plt.savefig(plot_path, bbox_inches="tight")
    print(f" Comparison plot saved to: {plot_path}")
    plt.close()

    # Plot 2: Bar chart of metrics
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=FIGURE_DPI)

    model_names = list(all_results.keys())
    colors_bar = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22",
                  "#1abc9c", "#e91e63", "#f39c12", "#607d8b"]

    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        values = [all_results[m][metric] for m in model_names]
        bars = ax.bar(model_names, values, color=colors_bar)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val}", ha="center", va="bottom", fontsize=9
            )

    plt.tight_layout()

    bar_path = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    plt.savefig(bar_path, bbox_inches="tight")
    print(f" Metrics bar chart saved to: {bar_path}")
    plt.close()

    # Save results to CSV
    csv_path = os.path.join(RESULTS_DIR, "model_results.csv")
    summary_df.to_csv(csv_path)
    print(f" Results CSV saved to: {csv_path}")

    print("\n Pipeline complete!\n")


if __name__ == "__main__":
    run_pipeline()

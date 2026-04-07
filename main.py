"""
main.py — Cold-Start Energy Consumption Prediction Pipeline

Entry point that orchestrates:
    1. Data loading and preprocessing
    2. Feature engineering
    3. Cold-start train/test split (2019+2021 train → 2020 test)
    4. Model training (11 models)
    5. Evaluation and comparison
    6. Visualization

Usage:
    python main.py
"""

import os
import random
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from config import (
    DATA_2019_PATH, DATA_2020_PATH, DATA_2021_PATH,
    TARGET,
    RESULTS_DIR, PLOT_SAMPLES, FIGURE_DPI, BASE_DIR,
)

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
from src.data_loader import load_data
from src.feature_engineering import add_features, create_sequences
from src.metrics import evaluate, print_metrics

from src.models import linear_reg, xgboost_model, svr_model
from src.models import ridge_model, lasso_model
from src.models import lstm_model, cnn_model, arima_model
from src.models import gru_model, bilstm_model, hgboost_model
from src.models import lstm_transfer_model

warnings.filterwarnings("ignore")

# Reproducibility seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def print_header():
    """Print a stylized project header."""
    print("\n" + "=" * 60)
    print("  Cold-Start Energy Consumption Prediction")
    print("  Train: 2019 + 2021 | Test: 2020 (unseen year)")
    print(f"  Target: {TARGET}")
    print("=" * 60)


def run_pipeline():
    """Execute the full prediction pipeline."""

    print_header()

    # --------------------------------------------------
    # 1. LOAD DATASETS
    # --------------------------------------------------
    print("\n.....Loading datasets.....")
    df_2019 = load_data(DATA_2019_PATH)
    df_2020 = load_data(DATA_2020_PATH)
    df_2021 = load_data(DATA_2021_PATH)

    print(f"   2019 — {len(df_2019)} samples")
    print(f"   2020 — {len(df_2020)} samples")
    print(f"   2021 — {len(df_2021)} samples")

    # --------------------------------------------------
    # 2. FEATURE ENGINEERING
    # --------------------------------------------------
    print("\n...Engineering features...")
    df_2019 = add_features(df_2019)
    df_2020 = add_features(df_2020)
    df_2021 = add_features(df_2021)

    # --------------------------------------------------
    # 3. COLD-START SPLIT: Train on 2019+2021, Test on 2020
    # --------------------------------------------------
    df_train = pd.concat([df_2019, df_2021]).sort_index()
    df_test = df_2020

    print(f"\n   Train set — {len(df_train)} samples (2019 + 2021)")
    print(f"   Test  set — {len(df_test)} samples (2020 — unseen year)")
    print(f"   Mean {TARGET} (train): {df_train[TARGET].mean():.4f}")
    print(f"   Mean {TARGET} (test):  {df_test[TARGET].mean():.4f}")

    X_train = df_train.drop(columns=[TARGET])
    y_train = df_train[TARGET]
    X_test = df_test.drop(columns=[TARGET])
    y_test = df_test[TARGET]

    # Sequences for DL models
    X_train_seq, y_train_seq = create_sequences(df_train)
    X_test_seq, y_test_seq = create_sequences(df_test)

    # Separate sequences for Transfer Learning (source=2019, target=2021)
    X_2019_seq, y_2019_seq = create_sequences(df_2019)
    X_2021_seq, y_2021_seq = create_sequences(df_2021)

    # --------------------------------------------------
    # 4. TRAIN & EVALUATE ALL MODELS
    # --------------------------------------------------
    all_results = {}
    predictions = {}

    # ---- 1. Linear Regression ----
    print("\n...Training Linear Regression...")
    lr_model = linear_reg.train(X_train, y_train)
    pred_lr = linear_reg.predict(lr_model, X_test)
    all_results["Linear Regression"] = evaluate(y_test, pred_lr)
    predictions["Linear"] = pred_lr
    print_metrics(all_results["Linear Regression"], "Linear Regression")

    # ---- 1b. Ridge Regression ----
    print("\n...Training Ridge Regression...")
    ridge_trained = ridge_model.train(X_train, y_train)
    pred_ridge = ridge_model.predict(ridge_trained, X_test)
    all_results["Ridge Regression"] = evaluate(y_test, pred_ridge)
    predictions["Ridge"] = pred_ridge
    print_metrics(all_results["Ridge Regression"], "Ridge Regression")

    # ---- 1c. Lasso Regression ----
    print("\n...Training Lasso Regression...")
    lasso_trained = lasso_model.train(X_train, y_train)
    pred_lasso = lasso_model.predict(lasso_trained, X_test)
    all_results["Lasso Regression"] = evaluate(y_test, pred_lasso)
    predictions["Lasso"] = pred_lasso
    print_metrics(all_results["Lasso Regression"], "Lasso Regression")

    # ---- 2. XGBoost ----
    print("\n...Training XGBoost...")
    xgb_model = xgboost_model.train(X_train, y_train)
    pred_xgb = xgboost_model.predict(xgb_model, X_test)
    all_results["XGBoost"] = evaluate(y_test, pred_xgb)
    predictions["XGBoost"] = pred_xgb
    print_metrics(all_results["XGBoost"], "XGBoost")

    # ---- 3. SVR ----
    print("\n...Training SVR...")
    svr_trained = svr_model.train(X_train, y_train)
    pred_svr = svr_model.predict(svr_trained, X_test)
    all_results["SVR"] = evaluate(y_test, pred_svr)
    predictions["SVR"] = pred_svr
    print_metrics(all_results["SVR"], "SVR")

    # ---- 4. LSTM ----
    print("\n...Training LSTM...")
    lstm_trained = lstm_model.train(X_train_seq, y_train_seq)
    pred_lstm = lstm_model.predict(lstm_trained, X_test_seq)
    all_results["LSTM"] = evaluate(y_test_seq, pred_lstm)
    predictions["LSTM"] = pred_lstm
    print_metrics(all_results["LSTM"], "LSTM")

    # ---- 4b. Transfer LSTM ----
    print("\n...Training Transfer Learning LSTM...")
    tl_lstm_trained = lstm_transfer_model.train(
        X_2019_seq, y_2019_seq, X_2021_seq, y_2021_seq
    )
    pred_tl_lstm = lstm_transfer_model.predict(tl_lstm_trained, X_test_seq)
    all_results["Transfer LSTM"] = evaluate(y_test_seq, pred_tl_lstm)
    predictions["Transfer LSTM"] = pred_tl_lstm
    print_metrics(all_results["Transfer LSTM"], "Transfer LSTM")

    # ---- 5. CNN ----
    print("\n...Training CNN (1D)...")
    cnn_trained = cnn_model.train(X_train_seq, y_train_seq)
    pred_cnn = cnn_model.predict(cnn_trained, X_test_seq)
    all_results["CNN (1D)"] = evaluate(y_test_seq, pred_cnn)
    predictions["CNN"] = pred_cnn
    print_metrics(all_results["CNN (1D)"], "CNN (1D)")

    # ---- 6. GRU ----
    print("\n...Training GRU...")
    gru_trained = gru_model.train(X_train_seq, y_train_seq)
    pred_gru = gru_model.predict(gru_trained, X_test_seq)
    all_results["GRU"] = evaluate(y_test_seq, pred_gru)
    predictions["GRU"] = pred_gru
    print_metrics(all_results["GRU"], "GRU")

    # ---- 7. BiLSTM ----
    print("\n...Training BiLSTM...")
    bilstm_trained = bilstm_model.train(X_train_seq, y_train_seq)
    pred_bilstm = bilstm_model.predict(bilstm_trained, X_test_seq)
    all_results["BiLSTM"] = evaluate(y_test_seq, pred_bilstm)
    predictions["BiLSTM"] = pred_bilstm
    print_metrics(all_results["BiLSTM"], "BiLSTM")

    # ---- 8. HGBoost ----
    print("\n...Training HGBoost...")
    hgb_model = hgboost_model.train(X_train, y_train)
    pred_hgb = hgboost_model.predict(hgb_model, X_test)
    all_results["HGBoost"] = evaluate(y_test, pred_hgb)
    predictions["HGBoost"] = pred_hgb
    print_metrics(all_results["HGBoost"], "HGBoost")

    # ---- 9. ARIMA ----
    print("\n...Training ARIMA...")
    arima_fitted = arima_model.train(df_train[TARGET])
    pred_arima = arima_model.predict(arima_fitted, len(df_test))
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

    # Plot 1: Prediction comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), dpi=FIGURE_DPI)

    # ML models
    axes[0].plot(y_test.values[:PLOT_SAMPLES], label="Actual", color="#2c3e50", linewidth=2)
    colors_ml = ["#e74c3c", "#ff6f61", "#d4a017", "#3498db", "#2ecc71", "#f39c12"]
    for (name, pred), color in zip(
        [("Linear", pred_lr), ("Ridge", pred_ridge), ("Lasso", pred_lasso),
         ("XGBoost", pred_xgb), ("SVR", pred_svr), ("HGBoost", pred_hgb)],
        colors_ml
    ):
        axes[0].plot(pred[:PLOT_SAMPLES], label=name, alpha=0.8, color=color)
    axes[0].set_title("ML Models — Cold-Start Prediction (Train: 2019+2021 → Test: 2020)", fontsize=13)
    axes[0].set_ylabel("Energy (kWh)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    # DL models
    axes[1].plot(y_test_seq[:PLOT_SAMPLES], label="Actual", color="#2c3e50", linewidth=2)
    colors_dl = ["#9b59b6", "#e67e22", "#1abc9c", "#e91e63", "#00bcd4"]
    for (name, pred), color in zip(
        [("LSTM", pred_lstm), ("CNN", pred_cnn), ("GRU", pred_gru), ("BiLSTM", pred_bilstm), ("Transfer LSTM", pred_tl_lstm)],
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
    colors_bar = ["#e74c3c", "#ff6f61", "#d4a017", "#3498db", "#2ecc71",
                  "#9b59b6", "#00bcd4", "#e67e22", "#1abc9c", "#e91e63",
                  "#f39c12", "#607d8b"]

    for ax, metric in zip(axes, ["MAE", "RMSE", "R2"]):
        values = [all_results[m][metric] for m in model_names]
        bars = ax.bar(model_names, values, color=colors_bar)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xticklabels(model_names, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3)

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

    # --------------------------------------------------
    # Plot 3: Individual Actual vs Predicted for each model
    # --------------------------------------------------
    print("\n...Generating individual Actual vs Predicted plots...")

    individual_models = [
        ("Linear Regression", y_test.values,   pred_lr,      "#e74c3c"),
        ("Ridge Regression",  y_test.values,   pred_ridge,   "#ff6f61"),
        ("Lasso Regression",  y_test.values,   pred_lasso,   "#d4a017"),
        ("XGBoost",           y_test.values,   pred_xgb,     "#3498db"),
        ("SVR",               y_test.values,   pred_svr,     "#2ecc71"),
        ("LSTM",              y_test_seq,       pred_lstm,    "#9b59b6"),
        ("CNN (1D)",          y_test_seq,       pred_cnn,     "#e67e22"),
        ("GRU",               y_test_seq,       pred_gru,     "#1abc9c"),
        ("BiLSTM",            y_test_seq,       pred_bilstm,  "#e91e63"),
        ("Transfer LSTM",     y_test_seq,       pred_tl_lstm, "#00bcd4"),
        ("HGBoost",           y_test.values,   pred_hgb,     "#f39c12"),
        ("ARIMA",             y_test_arima,     pred_arima,   "#607d8b"),
    ]

    for model_name, actual, predicted, color in individual_models:
        actual_arr = np.asarray(actual).flatten()
        pred_arr = np.asarray(predicted).flatten()
        n = min(len(actual_arr), len(pred_arr))
        actual_arr = actual_arr[:n]
        pred_arr = pred_arr[:n]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=FIGURE_DPI)

        # Left: Time-series line comparison
        samples = min(PLOT_SAMPLES, n)
        axes[0].plot(actual_arr[:samples], label="Actual", color="#2c3e50",
                     linewidth=2, zorder=3)
        axes[0].plot(pred_arr[:samples], label="Predicted", color=color,
                     linewidth=1.8, alpha=0.85, zorder=2)
        axes[0].fill_between(
            range(samples),
            actual_arr[:samples], pred_arr[:samples],
            alpha=0.12, color=color
        )
        axes[0].set_title(f"{model_name} — Actual vs Predicted (Time Series)",
                          fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Sample Index")
        axes[0].set_ylabel("Energy (kWh)")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # Right: Scatter plot with ideal line
        axes[1].scatter(actual_arr, pred_arr, alpha=0.4, s=18, color=color,
                        edgecolors="white", linewidths=0.3, label="Predictions")
        lo = min(actual_arr.min(), pred_arr.min())
        hi = max(actual_arr.max(), pred_arr.max())
        margin = (hi - lo) * 0.05
        axes[1].plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                     "k--", linewidth=1.2, label="Ideal (y = x)")
        axes[1].set_xlim(lo - margin, hi + margin)
        axes[1].set_ylim(lo - margin, hi + margin)
        axes[1].set_title(f"{model_name} — Actual vs Predicted (Scatter)",
                          fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Actual Energy (kWh)")
        axes[1].set_ylabel("Predicted Energy (kWh)")
        axes[1].legend(loc="upper left")
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect("equal", adjustable="box")

        plt.tight_layout()
        safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
        ind_path = os.path.join(RESULTS_DIR, f"actual_vs_predicted_{safe_name}.png")
        plt.savefig(ind_path, bbox_inches="tight")
        plt.close()
        print(f"   {model_name} plot saved to: {ind_path}")

    # --------------------------------------------------
    # Plot 4: LR vs Ridge vs Lasso Comparison
    # --------------------------------------------------
    print("\n...Generating LR vs Ridge vs Lasso comparison plot...")

    lr_family = {
        "Linear Regression": all_results["Linear Regression"],
        "Ridge Regression":  all_results["Ridge Regression"],
        "Lasso Regression":  all_results["Lasso Regression"],
    }
    lr_names = list(lr_family.keys())
    lr_colors = ["#e74c3c", "#ff6f61", "#d4a017"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=FIGURE_DPI)

    # (a) Time-series overlay
    ax = axes[0, 0]
    samples = min(PLOT_SAMPLES, len(pred_lr))
    ax.plot(y_test.values[:samples], label="Actual", color="#2c3e50", linewidth=2)
    for (name, pred), color in zip(
        [("Linear", pred_lr), ("Ridge", pred_ridge), ("Lasso", pred_lasso)],
        lr_colors
    ):
        ax.plot(pred[:samples], label=name, alpha=0.85, color=color, linewidth=1.5)
    ax.set_title("LR vs Ridge vs Lasso — Time Series Comparison", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Energy (kWh)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # (b) Bar chart per metric
    metrics_to_plot = ["MAE", "RMSE", "R2"]
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[(idx + 1) // 2, (idx + 1) % 2]
        vals = [lr_family[m][metric] for m in lr_names]
        bars = ax.bar(lr_names, vals, color=lr_colors, edgecolor="white", linewidth=1.2)
        ax.set_title(metric, fontsize=14, fontweight="bold")
        ax.set_xticklabels(lr_names, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val}", ha="center", va="bottom", fontsize=11, fontweight="bold"
            )

    plt.suptitle("Linear Regression Family — Performance Comparison",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    lr_compare_path = os.path.join(RESULTS_DIR, "lr_ridge_lasso_comparison.png")
    plt.savefig(lr_compare_path, bbox_inches="tight")
    print(f"   LR family comparison saved to: {lr_compare_path}")
    plt.close()

    csv_path = os.path.join(RESULTS_DIR, "model_results.csv")
    summary_df.to_csv(csv_path)
    print(f" Results CSV saved to: {csv_path}")

    # --------------------------------------------------
    # 7. SAVE TRAINED MODELS & TEST DATA FOR GUI
    # --------------------------------------------------
    print("\n...Saving trained models for GUI...")
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

    np.savez(
        os.path.join(SAVED_MODELS_DIR, "test_data.npz"),
        X_test=X_test.values,
        y_test=y_test.values,
        X_test_seq=X_test_seq,
        y_test_seq=y_test_seq,
        y_test_arima=y_test_arima,
    )

    joblib.dump(list(X_test.columns), os.path.join(SAVED_MODELS_DIR, "feature_names.pkl"))

    joblib.dump(lr_model,   os.path.join(SAVED_MODELS_DIR, "linear_reg.pkl"))
    joblib.dump(ridge_trained, os.path.join(SAVED_MODELS_DIR, "ridge.pkl"))
    joblib.dump(lasso_trained, os.path.join(SAVED_MODELS_DIR, "lasso.pkl"))
    joblib.dump(xgb_model,  os.path.join(SAVED_MODELS_DIR, "xgboost.pkl"))
    joblib.dump(hgb_model,  os.path.join(SAVED_MODELS_DIR, "hgboost.pkl"))
    joblib.dump(svr_trained, os.path.join(SAVED_MODELS_DIR, "svr.pkl"))


    for name, model_tuple in [
        ("lstm", lstm_trained),
        ("gru", gru_trained),
        ("bilstm", bilstm_trained),
        ("cnn", cnn_trained),
        ("transfer_lstm", tl_lstm_trained),
    ]:
        keras_model, sx, sy = model_tuple
        keras_model.save(os.path.join(SAVED_MODELS_DIR, f"{name}_model.keras"))
        joblib.dump((sx, sy), os.path.join(SAVED_MODELS_DIR, f"{name}_scalers.pkl"))

    joblib.dump(arima_fitted, os.path.join(SAVED_MODELS_DIR, "arima.pkl"))

    joblib.dump(predictions, os.path.join(SAVED_MODELS_DIR, "predictions.pkl"))
    joblib.dump(all_results, os.path.join(SAVED_MODELS_DIR, "all_results.pkl"))

    joblib.dump(pred_arima, os.path.join(SAVED_MODELS_DIR, "pred_arima.pkl"))

    print(f"   All models saved to: {SAVED_MODELS_DIR}")
    print("\n Pipeline complete!\n")


if __name__ == "__main__":
    run_pipeline()

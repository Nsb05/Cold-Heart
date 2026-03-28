"""
gui.py — Cold-Start Energy Prediction GUI

Interactive Tkinter dashboard that loads pre-trained models
and lets you explore predictions instantly without re-running
the full 30-minute pipeline.

Usage:
    python gui.py
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(BASE_DIR, "saved_models")

# ---------------------------------------------------------------------------
# Color palette (matches project's existing plots)
# ---------------------------------------------------------------------------
BG           = "#1a1a2e"
BG_CARD      = "#16213e"
BG_SIDEBAR   = "#0f3460"
ACCENT       = "#e94560"
TEXT_PRIMARY  = "#ffffff"
TEXT_MUTED    = "#a8a8b3"
CARD_BORDER  = "#233554"
CHART_BG     = "#16213e"
GRID_COLOR   = "#233554"

MODEL_COLORS = {
    "Linear":        "#e74c3c",
    "Ridge":         "#ff6f61",
    "Lasso":         "#d4a017",
    "XGBoost":       "#3498db",
    "SVR":           "#2ecc71",
    "LSTM":          "#9b59b6",
    "CNN":           "#e67e22",
    "GRU":           "#1abc9c",
    "BiLSTM":        "#e91e63",
    "Transfer LSTM": "#00bcd4",
    "HGBoost":       "#f39c12",
    "ARIMA":         "#607d8b",
}

# Models that use sequences (DL) vs tabular (ML)
SEQ_MODELS = {"LSTM", "CNN", "GRU", "BiLSTM", "Transfer LSTM"}
ML_MODELS  = {"Linear", "Ridge", "Lasso", "XGBoost", "SVR", "HGBoost"}


def load_everything():
    """Load all saved predictions, results, and test data."""
    if not os.path.isdir(SAVED_DIR):
        messagebox.showerror(
            "Models Not Found",
            f"No saved models found at:\n{SAVED_DIR}\n\n"
            "Please run the full pipeline first:\n  python main.py",
        )
        sys.exit(1)

    data = {}

    # Test data
    npz = np.load(os.path.join(SAVED_DIR, "test_data.npz"))
    data["y_test"]       = npz["y_test"]
    data["y_test_seq"]   = npz["y_test_seq"]
    data["y_test_arima"] = npz["y_test_arima"]
    data["X_test"]       = npz["X_test"]
    data["X_test_seq"]   = npz["X_test_seq"]

    # Predictions dict  {"Linear": array, "Ridge": array, ...}
    data["predictions"] = joblib.load(os.path.join(SAVED_DIR, "predictions.pkl"))
    data["pred_arima"]  = joblib.load(os.path.join(SAVED_DIR, "pred_arima.pkl"))

    # Metrics dict  {"Linear Regression": {"MAE":..., "RMSE":..., ...}, ...}
    data["all_results"] = joblib.load(os.path.join(SAVED_DIR, "all_results.pkl"))

    # Feature names
    data["feature_names"] = joblib.load(os.path.join(SAVED_DIR, "feature_names.pkl"))

    return data


# ===================================================================
# GUI Application
# ===================================================================
class PredictionGUI:
    # Mapping from short display name → full results-dict key
    MODEL_DISPLAY = {
        "Linear":        "Linear Regression",
        "Ridge":         "Ridge Regression",
        "Lasso":         "Lasso Regression",
        "XGBoost":       "XGBoost",
        "SVR":           "SVR",
        "LSTM":          "LSTM",
        "CNN":           "CNN (1D)",
        "GRU":           "GRU",
        "BiLSTM":        "BiLSTM",
        "Transfer LSTM": "Transfer LSTM",
        "HGBoost":       "HGBoost",
        "ARIMA":         "ARIMA",
    }

    def __init__(self, root, data):
        self.root = root
        self.data = data
        self.root.title("Cold-Start Energy Prediction — Dashboard")
        self.root.configure(bg=BG)
        self.root.state("zoomed")  # fullscreen on Windows

        # Style configuration
        self._setup_styles()

        # Layout: sidebar + main content
        self._build_sidebar()
        self._build_main()

        # Show first model by default
        self.model_var.set("XGBoost")
        self._on_model_change()

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------
    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Sidebar.TFrame",   background=BG_SIDEBAR)
        style.configure("Main.TFrame",      background=BG)
        style.configure("Card.TFrame",      background=BG_CARD)

        style.configure(
            "Title.TLabel",
            background=BG_SIDEBAR, foreground=TEXT_PRIMARY,
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "Subtitle.TLabel",
            background=BG_SIDEBAR, foreground=TEXT_MUTED,
            font=("Segoe UI", 9),
        )
        style.configure(
            "SideHead.TLabel",
            background=BG_SIDEBAR, foreground=TEXT_PRIMARY,
            font=("Segoe UI", 11, "bold"),
        )
        style.configure(
            "MetricName.TLabel",
            background=BG_CARD, foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure(
            "MetricValue.TLabel",
            background=BG_CARD, foreground=TEXT_PRIMARY,
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "CardTitle.TLabel",
            background=BG_CARD, foreground=TEXT_PRIMARY,
            font=("Segoe UI", 12, "bold"),
        )
        style.configure(
            "Status.TLabel",
            background=BG, foreground=ACCENT,
            font=("Segoe UI", 10, "italic"),
        )
        style.configure(
            "Accent.TButton",
            font=("Segoe UI", 10, "bold"),
        )

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------
    def _build_sidebar(self):
        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame", width=280)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        pad = {"padx": 18, "pady": (0, 0)}

        # Title
        ttk.Label(sidebar, text="⚡ Cold-Start", style="Title.TLabel").pack(
            **{**pad, "pady": (24, 0)}, anchor="w",
        )
        ttk.Label(sidebar, text="Energy Prediction Dashboard", style="Subtitle.TLabel").pack(
            **{**pad, "pady": (0, 18)}, anchor="w",
        )

        # Separator
        sep = tk.Frame(sidebar, bg=CARD_BORDER, height=1)
        sep.pack(fill=tk.X, padx=18, pady=4)

        # Model selector
        ttk.Label(sidebar, text="SELECT MODEL", style="SideHead.TLabel").pack(
            **{**pad, "pady": (16, 6)}, anchor="w",
        )

        self.model_var = tk.StringVar()
        model_names = list(self.MODEL_DISPLAY.keys())

        combo = ttk.Combobox(
            sidebar, textvariable=self.model_var,
            values=model_names, state="readonly",
            font=("Segoe UI", 11), width=22,
        )
        combo.pack(padx=18, pady=(0, 12), anchor="w")
        combo.bind("<<ComboboxSelected>>", lambda _: self._on_model_change())

        # Sample slider
        ttk.Label(sidebar, text="DISPLAY SAMPLES", style="SideHead.TLabel").pack(
            **{**pad, "pady": (12, 6)}, anchor="w",
        )

        self.sample_var = tk.IntVar(value=200)
        self.sample_label = ttk.Label(
            sidebar, text="200 samples", style="Subtitle.TLabel",
        )
        self.sample_label.pack(padx=18, anchor="w")

        slider = tk.Scale(
            sidebar, from_=50, to=1000, orient=tk.HORIZONTAL,
            variable=self.sample_var, showvalue=False,
            bg=BG_SIDEBAR, fg=TEXT_PRIMARY, troughcolor=BG,
            highlightthickness=0, length=220,
            command=self._on_slider,
        )
        slider.pack(padx=18, pady=(0, 12), anchor="w")

        # Buttons
        btn_style = {"bg": ACCENT, "fg": TEXT_PRIMARY, "relief": "flat",
                     "font": ("Segoe UI", 10, "bold"), "cursor": "hand2",
                     "activebackground": "#c73550", "activeforeground": TEXT_PRIMARY,
                     "bd": 0, "padx": 14, "pady": 8}

        btn_compare = tk.Button(
            sidebar, text="📊  Compare All Models",
            command=self._show_comparison, **btn_style,
        )
        btn_compare.pack(padx=18, pady=(8, 6), fill=tk.X)

        btn_custom = tk.Button(
            sidebar, text="🔮  Custom Prediction",
            command=self._show_custom_input, **btn_style,
        )
        btn_custom.pack(padx=18, pady=(0, 6), fill=tk.X)

        # Spacer
        tk.Frame(sidebar, bg=BG_SIDEBAR).pack(fill=tk.BOTH, expand=True)

        # Footer
        ttk.Label(
            sidebar,
            text="Train: 2019 + 2021\nTest:  2020 (unseen year)",
            style="Subtitle.TLabel",
        ).pack(padx=18, pady=(0, 18), anchor="w")

    # ------------------------------------------------------------------
    # Main content area
    # ------------------------------------------------------------------
    def _build_main(self):
        self.main = ttk.Frame(self.root, style="Main.TFrame")
        self.main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Top: metrics row ---
        self.metrics_frame = tk.Frame(self.main, bg=BG)
        self.metrics_frame.pack(fill=tk.X, padx=16, pady=(16, 8))

        self.metric_cards = {}
        for metric in ["MAE", "RMSE", "R²", "MAPE", "CV_RMSE"]:
            card = self._make_metric_card(self.metrics_frame, metric)
            card.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
            self.metric_cards[metric] = card

        # --- Bottom: charts (2 columns) ---
        chart_frame = tk.Frame(self.main, bg=BG)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))

        # Left chart: time-series
        self.fig_ts = Figure(figsize=(7, 4), dpi=100, facecolor=CHART_BG)
        self.ax_ts = self.fig_ts.add_subplot(111)
        self.canvas_ts = FigureCanvasTkAgg(self.fig_ts, master=chart_frame)
        self.canvas_ts.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Right chart: scatter
        self.fig_sc = Figure(figsize=(5, 4), dpi=100, facecolor=CHART_BG)
        self.ax_sc = self.fig_sc.add_subplot(111)
        self.canvas_sc = FigureCanvasTkAgg(self.fig_sc, master=chart_frame)
        self.canvas_sc.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

    def _make_metric_card(self, parent, metric_name):
        card = tk.Frame(parent, bg=BG_CARD, highlightbackground=CARD_BORDER,
                        highlightthickness=1, padx=14, pady=10)

        lbl_name = tk.Label(card, text=metric_name, bg=BG_CARD,
                            fg=TEXT_MUTED, font=("Segoe UI", 10))
        lbl_name.pack(anchor="w")

        lbl_val = tk.Label(card, text="—", bg=BG_CARD,
                           fg=TEXT_PRIMARY, font=("Segoe UI", 22, "bold"))
        lbl_val.pack(anchor="w")

        card._value_label = lbl_val
        return card

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_slider(self, _val):
        n = self.sample_var.get()
        self.sample_label.config(text=f"{n} samples")
        self._update_charts()

    def _on_model_change(self):
        model_short = self.model_var.get()
        model_full = self.MODEL_DISPLAY.get(model_short, model_short)

        # Update metrics
        metrics = self.data["all_results"].get(model_full, {})
        mapping = {"MAE": "MAE", "RMSE": "RMSE", "R²": "R2",
                   "MAPE": "MAPE", "CV_RMSE": "CV_RMSE"}
        for display_name, key in mapping.items():
            val = metrics.get(key, "—")
            if val != "—":
                unit = "%" if key in ("MAPE", "CV_RMSE") else ""
                val = f"{val}{unit}"
            self.metric_cards[display_name]._value_label.config(text=val)

        self._update_charts()

    def _update_charts(self):
        model_short = self.model_var.get()
        model_full = self.MODEL_DISPLAY.get(model_short, model_short)
        color = MODEL_COLORS.get(model_short, ACCENT)
        samples = self.sample_var.get()

        # Get actual and predicted arrays
        if model_short == "ARIMA":
            actual = self.data["y_test_arima"]
            pred   = self.data["pred_arima"]
        elif model_short in SEQ_MODELS:
            actual = self.data["y_test_seq"]
            pred   = self.data["predictions"].get(model_short, np.array([]))
        else:
            actual = self.data["y_test"]
            pred   = self.data["predictions"].get(model_short, np.array([]))

        actual = np.asarray(actual).flatten()
        pred   = np.asarray(pred).flatten()
        n = min(len(actual), len(pred))
        actual = actual[:n]
        pred   = pred[:n]
        show   = min(samples, n)

        # --- Time-series plot ---
        ax = self.ax_ts
        ax.clear()
        ax.set_facecolor(CHART_BG)
        ax.plot(actual[:show], label="Actual", color="#ffffff", linewidth=1.8, alpha=0.9)
        ax.plot(pred[:show],   label="Predicted", color=color, linewidth=1.6, alpha=0.85)
        ax.fill_between(range(show), actual[:show], pred[:show],
                        alpha=0.10, color=color)
        ax.set_title(f"{model_full} — Actual vs Predicted", color=TEXT_PRIMARY,
                     fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Sample Index", color=TEXT_MUTED, fontsize=9)
        ax.set_ylabel("Energy (kWh)", color=TEXT_MUTED, fontsize=9)
        ax.legend(loc="upper right", facecolor=BG_CARD, edgecolor=CARD_BORDER,
                  labelcolor=TEXT_PRIMARY, fontsize=9)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax.grid(True, color=GRID_COLOR, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_color(CARD_BORDER)
        self.fig_ts.tight_layout()
        self.canvas_ts.draw()

        # --- Scatter plot ---
        ax2 = self.ax_sc
        ax2.clear()
        ax2.set_facecolor(CHART_BG)
        ax2.scatter(actual, pred, alpha=0.35, s=14, color=color,
                    edgecolors="white", linewidths=0.3, label="Predictions")
        lo = min(actual.min(), pred.min())
        hi = max(actual.max(), pred.max())
        margin = (hi - lo) * 0.05
        ax2.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                 "--", color="#ffffff", linewidth=1, alpha=0.6, label="Ideal (y=x)")
        ax2.set_xlim(lo - margin, hi + margin)
        ax2.set_ylim(lo - margin, hi + margin)
        ax2.set_title(f"{model_full} — Scatter", color=TEXT_PRIMARY,
                      fontsize=12, fontweight="bold", pad=10)
        ax2.set_xlabel("Actual (kWh)", color=TEXT_MUTED, fontsize=9)
        ax2.set_ylabel("Predicted (kWh)", color=TEXT_MUTED, fontsize=9)
        ax2.legend(loc="upper left", facecolor=BG_CARD, edgecolor=CARD_BORDER,
                   labelcolor=TEXT_PRIMARY, fontsize=9)
        ax2.tick_params(colors=TEXT_MUTED, labelsize=8)
        ax2.grid(True, color=GRID_COLOR, alpha=0.4)
        ax2.set_aspect("equal", adjustable="box")
        for spine in ax2.spines.values():
            spine.set_color(CARD_BORDER)
        self.fig_sc.tight_layout()
        self.canvas_sc.draw()

    # ------------------------------------------------------------------
    # Compare All Models (popup)
    # ------------------------------------------------------------------
    def _show_comparison(self):
        win = tk.Toplevel(self.root)
        win.title("All Models — Metric Comparison")
        win.configure(bg=BG)
        win.geometry("1200x600")

        results = self.data["all_results"]
        model_names = list(results.keys())
        colors = [MODEL_COLORS.get(k.split()[0], ACCENT) for k in model_names]

        fig = Figure(figsize=(14, 5), dpi=100, facecolor=BG)

        for idx, metric in enumerate(["MAE", "RMSE", "R2"]):
            ax = fig.add_subplot(1, 3, idx + 1)
            ax.set_facecolor(CHART_BG)
            vals = [results[m].get(metric, 0) for m in model_names]
            bars = ax.bar(range(len(model_names)), vals, color=colors, edgecolor=BG_CARD)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha="right",
                               fontsize=7, color=TEXT_MUTED)
            ax.set_title(metric, color=TEXT_PRIMARY, fontsize=13, fontweight="bold")
            ax.tick_params(colors=TEXT_MUTED, labelsize=8)
            ax.grid(axis="y", color=GRID_COLOR, alpha=0.4)
            for spine in ax.spines.values():
                spine.set_color(CARD_BORDER)

            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val}", ha="center", va="bottom", fontsize=7,
                        color=TEXT_PRIMARY, fontweight="bold")

        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        canvas.draw()

    # ------------------------------------------------------------------
    # Custom prediction (popup)
    # ------------------------------------------------------------------
    def _show_custom_input(self):
        win = tk.Toplevel(self.root)
        win.title("Custom Prediction — Enter Feature Values")
        win.configure(bg=BG)
        win.geometry("520x650")
        win.resizable(False, False)

        # Header
        tk.Label(win, text="🔮 Custom Prediction", bg=BG, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 14, "bold")).pack(pady=(18, 4))
        tk.Label(win, text="Enter feature values and pick a model to get an instant prediction.",
                 bg=BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(pady=(0, 12))

        # Scrollable frame for feature inputs
        container = tk.Frame(win, bg=BG)
        container.pack(fill=tk.BOTH, expand=True, padx=20)

        canvas_scroll = tk.Canvas(container, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas_scroll.yview)
        scroll_frame = tk.Frame(canvas_scroll, bg=BG)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )
        canvas_scroll.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        feature_names = self.data["feature_names"]
        entries = {}

        for feat in feature_names:
            row = tk.Frame(scroll_frame, bg=BG)
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=feat, bg=BG, fg=TEXT_MUTED,
                     font=("Segoe UI", 9), width=16, anchor="w").pack(side=tk.LEFT)
            ent = tk.Entry(row, bg=BG_CARD, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY,
                           font=("Segoe UI", 10), width=18, relief="flat",
                           highlightbackground=CARD_BORDER, highlightthickness=1)
            ent.insert(0, "0.0")
            ent.pack(side=tk.LEFT, padx=(8, 0))
            entries[feat] = ent

        # Model picker + predict button
        bottom = tk.Frame(win, bg=BG)
        bottom.pack(fill=tk.X, padx=20, pady=(8, 18))

        tk.Label(bottom, text="Model:", bg=BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(side=tk.LEFT)

        predict_model_var = tk.StringVar(value="XGBoost")
        ml_only = [k for k in self.MODEL_DISPLAY.keys() if k in ML_MODELS]
        combo = ttk.Combobox(bottom, textvariable=predict_model_var,
                             values=ml_only, state="readonly",
                             font=("Segoe UI", 10), width=14)
        combo.pack(side=tk.LEFT, padx=8)

        result_label = tk.Label(bottom, text="", bg=BG, fg=ACCENT,
                                font=("Segoe UI", 12, "bold"))
        result_label.pack(side=tk.RIGHT)

        def do_predict():
            try:
                vals = [float(entries[f].get()) for f in feature_names]
            except ValueError:
                messagebox.showwarning("Invalid Input", "Please enter numeric values for all features.")
                return

            model_short = predict_model_var.get()
            X_input = np.array(vals).reshape(1, -1)

            try:
                model_file_map = {
                    "Linear":  "linear_reg.pkl",
                    "Ridge":   "ridge.pkl",
                    "Lasso":   "lasso.pkl",
                    "XGBoost": "xgboost.pkl",
                    "HGBoost": "hgboost.pkl",
                    "SVR":     "svr.pkl",
                }
                fname = model_file_map[model_short]
                loaded = joblib.load(os.path.join(SAVED_DIR, fname))

                if model_short == "SVR":
                    model, sx, sy = loaded
                    X_scaled = sx.transform(X_input)
                    pred_scaled = model.predict(X_scaled)
                    prediction = sy.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()[0]
                elif model_short == "HGBoost":
                    import pandas as pd
                    X_df = pd.DataFrame(X_input, columns=feature_names)
                    result = loaded.predict(X_df)
                    prediction = result[0][0] if isinstance(result, tuple) else float(result[0])
                else:
                    prediction = loaded.predict(X_input)[0]

                result_label.config(text=f"Predicted: {prediction:.6f} kWh")

            except Exception as e:
                messagebox.showerror("Prediction Error", str(e))

        btn = tk.Button(bottom, text="Predict", bg=ACCENT, fg=TEXT_PRIMARY,
                        font=("Segoe UI", 10, "bold"), relief="flat",
                        cursor="hand2", command=do_predict,
                        activebackground="#c73550", padx=14, pady=4)
        btn.pack(side=tk.LEFT, padx=8)


# ===================================================================
# Entry point
# ===================================================================
def main():
    print("Loading saved models and data...")
    data = load_everything()
    print("Done! Launching GUI...\n")

    root = tk.Tk()
    _app = PredictionGUI(root, data)
    root.mainloop()


if __name__ == "__main__":
    main()

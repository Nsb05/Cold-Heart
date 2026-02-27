#  Cold-Start Energy Consumption Prediction for Smart Buildings

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-green)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine learning and deep learning framework for **predicting energy consumption in smart buildings under cold-start conditions** — where historical data from the target year is unavailable.

## 📌 Problem Statement

Traditional energy forecasting relies on having historical data from the **same building and time period**. In real-world scenarios such as:

- Newly installed smart meters
- New buildings with no usage history
- Post-renovation energy pattern changes

...this data simply doesn't exist. Our approach trains models on **2019 smart meter data** and tests directly on **2020 data**, simulating a true **cold-start scenario**.

##  Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│  Raw CSV ──► Time Parsing ──► Hourly Resampling ──► Imputation  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING                           │
│  Cyclical Encoding (hour_sin, hour_cos) + Lag Features (1, 24)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
     ┌──────────────┐ ┌─────────┐ ┌──────────────┐
     │  ML Models   │ │   DL    │ │ Statistical  │
     │              │ │ Models  │ │    Models     │
     │ • Linear Reg │ │ • LSTM  │ │ • ARIMA      │
     │ • XGBoost    │ │ • CNN   │ │   (5,1,2)    │
     │ • SVR        │ │  (1D)   │ │              │
     └──────┬───────┘ └────┬────┘ └──────┬───────┘
            │              │             │
            └──────────────┼─────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EVALUATION & COMPARISON                     │
│              MAE  ·  RMSE  ·  R²  ·  MAPE  ·  CV(RMSE)         │
└─────────────────────────────────────────────────────────────────┘
```

##  Models Implemented

| #  | Model               | Type          | Description                                      |
|----|----------------------|---------------|--------------------------------------------------|
| 1  | Linear Regression    | ML Baseline   | Simple linear model for baseline comparison            |
| 2  | XGBoost              | Ensemble ML   | Gradient boosted trees with tuned hyperparameters      |
| 3  | SVR                  | Kernel ML     | Support Vector Regression with linear kernel           |
| 4  | LSTM                 | Deep Learning | Long Short-Term Memory for sequence modeling           |
| 5  | CNN (1D)             | Deep Learning | 1D Convolutional Neural Network                        |
| 6  | GRU                  | Deep Learning | Gated Recurrent Unit — lighter LSTM alternative        |
| 7  | BiLSTM               | Deep Learning | Bidirectional LSTM with forward + backward processing  |
| 8  | HGBoost              | AutoML        | Hyperoptimized Gradient Boosting via Bayesian search   |
| 9  | ARIMA                | Statistical   | AutoRegressive Integrated Moving Average               |

##  Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Nsb05/Cold-Heart.git
cd Cold-Heart
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add data

Place the CEEW smart meter CSV files in the `data/` directory. See [`data/README.md`](data/README.md) for details.

### 4. Run the full pipeline

```bash
python main.py
```

This will train all 6 models, print a comparison table, and save plots to `results/`.

##  Project Structure

```
├── README.md                        # Project overview (you are here)
├── requirements.txt                 # Python dependencies
├── config.py                        # Centralized configuration
├── main.py                          # Entry point — runs the full pipeline
├── LICENSE                          # MIT License
├── .gitignore                       # Git ignore rules
│
├── src/                             # Source modules
│   ├── __init__.py
│   ├── data_loader.py               # Data loading and preprocessing
│   ├── feature_engineering.py       # Feature creation and sequences
│   ├── metrics.py                   # Evaluation metrics
│   └── models/                      # Individual model implementations
│       ├── __init__.py
│       ├── linear_reg.py
│       ├── xgboost_model.py
│       ├── svr_model.py
│       ├── lstm_model.py
│       ├── cnn_model.py
│       ├── gru_model.py
│       ├── bilstm_model.py
│       ├── hgboost_model.py
│       └── arima_model.py
│
├── notebooks/                       # Jupyter notebooks
│   └── exploratory_analysis.ipynb   # EDA and data visualization
│
├── data/                            # Dataset directory (gitignored)
│   └── README.md                    # Data sourcing instructions
│
└── results/                         # Output plots and metrics
    └── README.md
```

##  Dataset

- **Source**: CEEW Smart Meter Data — Mathura, India
- **Training**: 2019 hourly readings
- **Testing**: 2020 hourly readings
- **Target variable**: `t_kWh` (total energy in kilowatt-hours)

##  Configuration

All hyperparameters and settings are centralized in [`config.py`](config.py):

```python
# Adjust model hyperparameters
XGBOOST_PARAMS = {
    "n_estimators": 600,
    "learning_rate": 0.03,
    "max_depth": 4,
    ...
}

# Change sequence length for LSTM/CNN
SEQUENCE_LENGTH = 48
```

##  Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-model`)
3. Commit your changes (`git commit -m 'Add Random Forest model'`)
4. Push to the branch (`git push origin feature/new-model`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

##  Citation

If you use this work in your research, please cite:

```bibtex
@article{cold_start_energy_2026,
  title   = {Cold Start Energy Consumption Prediction for Smart Buildings
             Using Machine Learning and Deep Learning},
  year    = {2026},
  note    = {CEEW Smart Meter Data, Mathura, India}
}
```

---

<p align="center">
  Made with ❤️ for smarter, greener buildings
</p>

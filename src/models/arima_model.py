"""
ARIMA model with automatic order selection via pmdarima.

Uses auto_arima to find the optimal (p,d,q) order by
minimizing AIC, replacing the fixed-order approach.
"""

import numpy as np
import pmdarima as pm

from config import ARIMA_AUTO_PARAMS


def train(train_series):
    """
    Fit an ARIMA model with auto-selected order on the training target series.

    Args:
        train_series: pandas Series of training target values.

    Returns:
        Fitted auto_arima model.
    """
    model = pm.auto_arima(
        train_series,
        start_p=ARIMA_AUTO_PARAMS["start_p"],
        start_q=ARIMA_AUTO_PARAMS["start_q"],
        max_p=ARIMA_AUTO_PARAMS["max_p"],
        max_q=ARIMA_AUTO_PARAMS["max_q"],
        d=ARIMA_AUTO_PARAMS["d"],
        seasonal=ARIMA_AUTO_PARAMS["seasonal"],
        stepwise=ARIMA_AUTO_PARAMS["stepwise"],
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    print(f"   Auto-ARIMA selected order: {model.order}")
    return model


def predict(model_fit, forecast_steps):
    """
    Generate out-of-sample forecasts.

    Args:
        model_fit: Fitted auto_arima model.
        forecast_steps: Number of future steps to forecast.

    Returns:
        Array of forecasted values.
    """
    forecast = model_fit.predict(n_periods=forecast_steps)
    return np.array(forecast)

"""
ARIMA (AutoRegressive Integrated Moving Average) model.

A traditional statistical time-series model that serves
as a baseline from the forecasting domain. Unlike ML/DL
models, ARIMA operates only on the univariate target series.
"""

from statsmodels.tsa.arima.model import ARIMA

from config import ARIMA_ORDER


def train(train_series):
    """
    Fit an ARIMA model on the training target series.

    Args:
        train_series: pandas Series of training target values.

    Returns:
        Fitted ARIMA model results object.
    """
    model = ARIMA(train_series, order=ARIMA_ORDER)
    model_fit = model.fit()
    return model_fit


def predict(model_fit, forecast_steps):
    """
    Generate out-of-sample forecasts.

    Args:
        model_fit: Fitted ARIMA results object.
        forecast_steps: Number of future steps to forecast.

    Returns:
        Array of forecasted values.
    """
    forecast = model_fit.forecast(steps=forecast_steps)
    return forecast.values

"""SARIMA (seasonal ARIMA)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return {"aic": model.aic, "forecast": model.forecast(12).tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(120)
    series = pd.Series(50 + 0.2 * t + 8 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.5, 120))
    r = fit_sarima(series)
    print("AIC:", r["aic"])
    print("forecast (12 steps):", r["forecast"])

"""SES / Holt / Holt-Winters (ETS)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


def simple_smoothing(series, alpha=None):
    model = SimpleExpSmoothing(series, initialization_method="estimated").fit(smoothing_level=alpha)
    return {"alpha": model.params["smoothing_level"], "aic": model.aic, "forecast": model.forecast(5).tolist()}


def holt_winters(series, seasonal=None, seasonal_period=None, trend="add"):
    model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal,
                                 seasonal_periods=seasonal_period,
                                 initialization_method="estimated").fit()
    return {"aic": model.aic, "forecast": model.forecast(seasonal_period or 5).tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(96)
    series = pd.Series(50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 2, 96))
    print("SES:", simple_smoothing(series))
    print("Holt-Winters:", holt_winters(series, seasonal="add", seasonal_period=12))

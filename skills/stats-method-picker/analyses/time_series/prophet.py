"""Time series forecasting with seasonality (Facebook/Meta Prophet)."""
import numpy as np
import pandas as pd
from prophet import Prophet


def forecast_prophet(dates, values, future_periods=30, freq="D"):
    df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": values})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=future_periods, freq=freq)
    forecast = model.predict(future)
    return {
        "forecast": forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(future_periods).to_dict("records"),
        "components": [c for c in ("trend", "weekly", "yearly") if c in forecast.columns],
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=365, freq="D")
    t = np.arange(365)
    values = 100 + 0.1 * t + 15 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 3, 365)
    r = forecast_prophet(dates, values, future_periods=14)
    print("first forecasts:", r["forecast"][:2])

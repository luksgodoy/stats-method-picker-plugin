"""ARIMA, with order selection by AIC (small grid search)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import itertools


def fit_arima(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order).fit()
    return {"order": order, "aic": model.aic, "forecast": model.forecast(5).tolist(),
            "summary": model.summary().as_text()}


def select_arima_order(series, p_max=2, d_max=1, q_max=2):
    best = None
    for p, d, q in itertools.product(range(p_max + 1), range(d_max + 1), range(q_max + 1)):
        try:
            model = ARIMA(series, order=(p, d, q)).fit()
            if best is None or model.aic < best[1]:
                best = ((p, d, q), model.aic)
        except Exception:
            continue
    return {"best_order": best[0], "aic": best[1]}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 150)
    series = pd.Series(np.cumsum(noise) + 100)
    print("ARIMA(1,1,1):", {k: v for k, v in fit_arima(series).items() if k != "summary"})
    print("best order:", select_arima_order(series))

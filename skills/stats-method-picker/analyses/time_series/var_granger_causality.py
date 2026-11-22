"""VAR models and Granger causality."""
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests


def fit_var(df_series, max_lags=5):
    model = VAR(df_series)
    selection = model.select_order(max_lags)
    result = model.fit(selection.aic)
    return {"optimal_lag_aic": selection.aic, "aic": result.aic,
            "forecast": result.forecast(df_series.values[-selection.aic:], steps=5).tolist()}


def granger_causality(cause_series, effect_series, max_lag=4, alpha=0.05):
    df = pd.DataFrame({"effect": effect_series, "cause": cause_series})
    results = grangercausalitytests(df[["effect", "cause"]], maxlag=max_lag, verbose=False)
    p_values = {lag: res[0]["ssr_ftest"][1] for lag, res in results.items()}
    best_lag = min(p_values, key=p_values.get)
    return {"p_values_by_lag": p_values, "granger_causes": bool(p_values[best_lag] < alpha)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(2, n):
        x[i] = 0.5 * x[i - 1] + rng.normal(0, 1)
        y[i] = 0.3 * y[i - 1] + 0.6 * x[i - 1] + rng.normal(0, 1)
    df = pd.DataFrame({"x": x, "y": y})
    print("VAR:", {k: v for k, v in fit_var(df).items() if k != "forecast"})
    print("Granger x->y:", granger_causality(df["x"], df["y"]))

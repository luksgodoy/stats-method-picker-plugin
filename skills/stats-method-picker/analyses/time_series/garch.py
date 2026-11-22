"""GARCH (conditional volatility), via the arch package."""
import numpy as np
from arch import arch_model


def fit_garch(returns, p=1, q=1):
    model = arch_model(returns, vol="Garch", p=p, q=q, dist="normal")
    result = model.fit(disp="off")
    forecast = result.forecast(horizon=5)
    return {"parameters": result.params.to_dict(), "aic": result.aic,
            "forecast_variance": forecast.variance.values[-1].tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 1000
    returns = np.zeros(n)
    sigma2 = np.ones(n)
    for i in range(1, n):
        sigma2[i] = 0.05 + 0.1 * returns[i - 1] ** 2 + 0.85 * sigma2[i - 1]
        returns[i] = rng.normal(0, np.sqrt(sigma2[i]))
    r = fit_garch(returns * 100)
    print("parameters:", r["parameters"])

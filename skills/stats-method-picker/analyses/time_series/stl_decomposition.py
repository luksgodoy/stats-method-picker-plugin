"""STL decomposition (trend, seasonality, residual via loess)."""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL


def stl_decompose(series, period, robust=True):
    result = STL(series, period=period, robust=robust).fit()
    trend_strength = max(0, 1 - result.resid.var() / (result.trend + result.resid).var())
    seasonal_strength = max(0, 1 - result.resid.var() / (result.seasonal + result.resid).var())
    return {"trend": result.trend.tolist(), "seasonal": result.seasonal.tolist(),
            "residual": result.resid.tolist(), "trend_strength": trend_strength,
            "seasonal_strength": seasonal_strength}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    t = np.arange(120)
    series = pd.Series(50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 1.5, 120))
    r = stl_decompose(series, period=12)
    print("trend strength:", r["trend_strength"], "seasonal strength:", r["seasonal_strength"])

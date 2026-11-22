"""Poisson regression (count data)."""
import numpy as np
import statsmodels.api as sm


def poisson_regression(x, y):
    x_const = sm.add_constant(np.asarray(x, dtype=float))
    model = sm.GLM(y, x_const, family=sm.families.Poisson()).fit()
    rate_ratios = np.exp(model.params)
    dispersion = model.pearson_chi2 / model.df_resid
    return {"coefficients": model.params.tolist(), "rate_ratios": rate_ratios.tolist(),
            "p_values": model.pvalues.tolist(), "dispersion": float(dispersion),
            "overdispersion": bool(dispersion > 1.5)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (200, 2))
    rate = np.exp(0.5 + 0.8 * x[:, 0] - 0.3 * x[:, 1])
    y = rng.poisson(rate)
    r = poisson_regression(x, y)
    print("rate ratios:", r["rate_ratios"], "dispersion:", r["dispersion"])

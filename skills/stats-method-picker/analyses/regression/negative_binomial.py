"""Negative binomial regression (count data with overdispersion)."""
import numpy as np
import statsmodels.api as sm


def negative_binomial_regression(x, y):
    x_const = sm.add_constant(np.asarray(x, dtype=float))
    model = sm.GLM(y, x_const, family=sm.families.NegativeBinomial()).fit()
    return {"coefficients": model.params.tolist(), "p_values": model.pvalues.tolist(),
            "deviance": model.deviance, "aic": model.aic}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (200, 2))
    mean = np.exp(0.5 + 0.6 * x[:, 0])
    y = rng.negative_binomial(n=3, p=3 / (3 + mean))
    r = negative_binomial_regression(x, y)
    print("coefficients:", r["coefficients"], "aic:", r["aic"])

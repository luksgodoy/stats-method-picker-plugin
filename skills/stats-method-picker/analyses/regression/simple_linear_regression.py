"""Simple linear regression (OLS)."""
import numpy as np
import statsmodels.api as sm


def simple_linear_regression(x, y):
    x_const = sm.add_constant(np.asarray(x, dtype=float))
    model = sm.OLS(y, x_const).fit()
    return {"intercept": model.params[0], "coefficient": model.params[1],
            "r2": model.rsquared, "coef_p_value": model.pvalues[1],
            "summary": model.summary().as_text()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 50)
    y = 3 + 2 * x + rng.normal(0, 1.5, 50)
    r = simple_linear_regression(x, y)
    print("coef:", r["coefficient"], "r2:", r["r2"])

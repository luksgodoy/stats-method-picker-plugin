"""Multiple linear regression (OLS), with VIF for multicollinearity."""
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def multiple_linear_regression(x, y):
    x_const = sm.add_constant(np.asarray(x, dtype=float))
    model = sm.OLS(y, x_const).fit()
    vif = [variance_inflation_factor(x_const, i) for i in range(x_const.shape[1])]
    return {"coefficients": model.params.tolist(), "adjusted_r2": model.rsquared_adj,
            "p_values": model.pvalues.tolist(), "vif": vif}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (100, 3))
    y = 1 + 2 * x[:, 0] - 1.5 * x[:, 1] + 0.5 * x[:, 2] + rng.normal(0, 1, 100)
    r = multiple_linear_regression(x, y)
    print("coefficients:", r["coefficients"], "adjusted_r2:", r["adjusted_r2"])

"""Binary logistic regression."""
import numpy as np
import statsmodels.api as sm


def binary_logistic_regression(x, y):
    x_const = sm.add_constant(np.asarray(x, dtype=float))
    model = sm.Logit(y, x_const).fit(disp=0)
    odds_ratios = np.exp(model.params)
    return {"coefficients": model.params.tolist(), "odds_ratios": odds_ratios.tolist(),
            "p_values": model.pvalues.tolist(), "pseudo_r2": model.prsquared}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (200, 2))
    logit = 1.5 * x[:, 0] - 2 * x[:, 1]
    prob = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, prob)
    r = binary_logistic_regression(x, y)
    print("odds ratios:", r["odds_ratios"], "pseudo_r2:", r["pseudo_r2"])

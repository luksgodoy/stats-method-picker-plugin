"""Multinomial logistic regression."""
import numpy as np
import statsmodels.api as sm


def multinomial_logistic_regression(x, y):
    x_const = sm.add_constant(np.asarray(x, dtype=float))
    model = sm.MNLogit(y, x_const).fit(disp=0)
    return {"coefficients": np.asarray(model.params).tolist(), "pseudo_r2": model.prsquared,
            "classes": sorted(set(np.asarray(y).tolist()))}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 2))
    score_b = 1.2 * x[:, 0]
    score_c = -1.0 * x[:, 1]
    probs = np.column_stack([np.zeros(300), score_b, score_c])
    probs = np.exp(probs) / np.exp(probs).sum(axis=1, keepdims=True)
    y = np.array([rng.choice([0, 1, 2], p=p) for p in probs])
    r = multinomial_logistic_regression(x, y)
    print("pseudo_r2:", r["pseudo_r2"], "classes:", r["classes"])

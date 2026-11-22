"""Zero-inflated models (ZIP / ZINB)."""
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.count_model import ZeroInflatedPoisson, ZeroInflatedNegativeBinomialP


def zip_model(x, y, x_inflation=None):
    x_const = sm.add_constant(np.asarray(x, dtype=float))
    x_inf = sm.add_constant(np.asarray(x_inflation, dtype=float)) if x_inflation is not None else np.ones((len(y), 1))
    model = ZeroInflatedPoisson(y, x_const, exog_infl=x_inf, inflation="logit").fit(disp=0)
    return {"coefficients": model.params.tolist(), "aic": model.aic, "log_likelihood": model.llf}


def zinb_model(x, y, x_inflation=None):
    x_const = sm.add_constant(np.asarray(x, dtype=float))
    x_inf = sm.add_constant(np.asarray(x_inflation, dtype=float)) if x_inflation is not None else np.ones((len(y), 1))
    model = ZeroInflatedNegativeBinomialP(y, x_const, exog_infl=x_inf, inflation="logit").fit(disp=0, maxiter=200)
    return {"coefficients": model.params.tolist(), "aic": model.aic, "log_likelihood": model.llf}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(0, 1, (n, 1))
    inflated = rng.binomial(1, 0.3, n)
    counts = rng.poisson(np.exp(0.5 + 0.7 * x[:, 0]))
    y = np.where(inflated == 1, 0, counts)
    print("ZIP:", zip_model(x, y))

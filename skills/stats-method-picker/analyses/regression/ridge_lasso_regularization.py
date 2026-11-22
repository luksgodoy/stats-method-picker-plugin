"""Ridge / Lasso / Elastic Net regularization, with alpha chosen via CV."""
import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler


def ridge(x, y, alphas=np.logspace(-3, 3, 50), cv=5):
    x = StandardScaler().fit_transform(np.asarray(x, dtype=float))
    model = RidgeCV(alphas=alphas, cv=cv).fit(x, y)
    return {"optimal_alpha": model.alpha_, "coefficients": model.coef_.tolist(), "r2": model.score(x, y)}


def lasso(x, y, cv=5):
    x = StandardScaler().fit_transform(np.asarray(x, dtype=float))
    model = LassoCV(cv=cv, random_state=0).fit(x, y)
    return {"optimal_alpha": model.alpha_, "coefficients": model.coef_.tolist(),
            "selected_variables": int((model.coef_ != 0).sum()), "r2": model.score(x, y)}


def elastic_net(x, y, l1_ratios=(.1, .5, .7, .9, .95, 1), cv=5):
    x = StandardScaler().fit_transform(np.asarray(x, dtype=float))
    model = ElasticNetCV(l1_ratio=list(l1_ratios), cv=cv, random_state=0).fit(x, y)
    return {"optimal_alpha": model.alpha_, "optimal_l1_ratio": model.l1_ratio_,
            "coefficients": model.coef_.tolist(), "r2": model.score(x, y)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (150, 10))
    beta = np.array([3, -2, 0, 0, 1.5, 0, 0, 0, 0, 0.5])
    y = x @ beta + rng.normal(0, 1, 150)
    print("ridge:", {k: v for k, v in ridge(x, y).items() if k != "coefficients"})
    r_lasso = lasso(x, y)
    print("lasso selected variables:", r_lasso["selected_variables"], "r2:", r_lasso["r2"])
    print("elastic net l1_ratio:", elastic_net(x, y)["optimal_l1_ratio"])

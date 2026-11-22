"""Exploratory factor analysis (maximum likelihood extraction)."""
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


def factor_analysis(x, n_factors):
    x = StandardScaler().fit_transform(np.asarray(x, dtype=float))
    model = FactorAnalysis(n_components=n_factors, random_state=0).fit(x)
    communalities = (model.components_.T ** 2).sum(axis=1)
    return {
        "factor_loadings": model.components_.T.tolist(),
        "communalities": communalities.tolist(),
        "noise_variance": model.noise_variance_.tolist(),
        "log_likelihood": model.score(x),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    f1 = rng.normal(0, 1, 200)
    f2 = rng.normal(0, 1, 200)
    x = np.column_stack([
        0.8 * f1 + rng.normal(0, 0.3, 200), 0.7 * f1 + rng.normal(0, 0.3, 200),
        0.9 * f2 + rng.normal(0, 0.3, 200), 0.6 * f2 + rng.normal(0, 0.3, 200),
    ])
    r = factor_analysis(x, n_factors=2)
    print("communalities:", r["communalities"])

"""Gaussian Mixture Models (probabilistic clustering via EM)."""
import numpy as np
from sklearn.mixture import GaussianMixture


def gaussian_mixture(x, n_components, random_state=0):
    model = GaussianMixture(n_components=n_components, random_state=random_state).fit(x)
    return {
        "labels": model.predict(x).tolist(),
        "probabilities": model.predict_proba(x).tolist(),
        "means": model.means_.tolist(),
        "bic": model.bic(x),
        "aic": model.aic(x),
        "converged": bool(model.converged_),
    }


def choose_n_components(x, candidates=range(1, 7), random_state=0):
    bics = [GaussianMixture(n_components=k, random_state=random_state).fit(x).bic(x) for k in candidates]
    return {"optimal_n_by_bic": list(candidates)[int(np.argmin(bics))], "bics": bics}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([0, 0], 0.5, (40, 2)), rng.normal([5, 5], 0.5, (40, 2)), rng.normal([0, 5], 0.5, (40, 2))])
    r = gaussian_mixture(x, n_components=3)
    print("BIC:", r["bic"], "converged:", r["converged"])
    print(choose_n_components(x))

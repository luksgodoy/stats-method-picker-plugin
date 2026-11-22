"""Principal component analysis."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def principal_component_analysis(x, n_components=None, standardize=True):
    x = np.asarray(x, dtype=float)
    if standardize:
        x = StandardScaler().fit_transform(x)
    model = PCA(n_components=n_components).fit(x)
    return {
        "explained_variance": model.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": np.cumsum(model.explained_variance_ratio_).tolist(),
        "loadings": model.components_.tolist(),
        "scores": model.transform(x).tolist(),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, (100, 3))
    x = np.column_stack([base, base @ [1, 0.5, -0.5] + rng.normal(0, 0.1, 100)])
    r = principal_component_analysis(x, n_components=2)
    print("explained variance:", r["explained_variance"])

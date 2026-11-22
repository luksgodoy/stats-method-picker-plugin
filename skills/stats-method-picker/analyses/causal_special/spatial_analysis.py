"""Spatial analysis: spatial autocorrelation (Moran's I) and
density-based geographic clustering (DBSCAN on coordinates)."""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def moran_index(values, coordinates, n_neighbors=5):
    """Moran's I: measures spatial autocorrelation - nearby values tend to be similar
    (I > 0), dissimilar (I < 0) or random (I ~ 0)."""
    n = len(values)
    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(coordinates)
    _, indices = neighbors.kneighbors(coordinates)
    weights = np.zeros((n, n))
    for i, row in enumerate(indices):
        for j in row[1:]:
            weights[i, j] = 1.0

    mean = values.mean()
    deviations = values - mean
    numerator = sum(weights[i, j] * deviations[i] * deviations[j] for i in range(n) for j in range(n))
    denominator = (deviations ** 2).sum()
    weights_sum = weights.sum()
    morans_i = (n / weights_sum) * (numerator / denominator)
    return {"morans_i": float(morans_i),
            "interpretation": "positive clustering" if morans_i > 0.1 else
                              "negative clustering" if morans_i < -0.1 else "random"}


def spatial_clusters(coordinates, max_radius=1.0, min_samples=5):
    model = DBSCAN(eps=max_radius, min_samples=min_samples).fit(coordinates)
    return {"n_clusters": int(len(set(model.labels_)) - (1 if -1 in model.labels_ else 0)),
            "n_noise": int((model.labels_ == -1).sum())}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    center1 = rng.normal([0, 0], 0.3, (80, 2))
    center2 = rng.normal([5, 5], 0.3, (80, 2))
    coordinates = np.vstack([center1, center2])
    values = np.concatenate([rng.normal(10, 1, 80), rng.normal(50, 1, 80)])

    r_moran = moran_index(values, coordinates)
    print("Moran's I:", round(r_moran["morans_i"], 3), "-", r_moran["interpretation"])

    r_clusters = spatial_clusters(coordinates, max_radius=1.0, min_samples=5)
    print("spatial clusters found:", r_clusters["n_clusters"])

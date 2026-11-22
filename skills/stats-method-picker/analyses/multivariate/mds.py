"""Multidimensional scaling (metric and non-metric)."""
import numpy as np
from sklearn.manifold import MDS


def multidimensional_scaling(distance_matrix, n_components=2, metric=True, random_state=0):
    model = MDS(n_components=n_components, metric=metric, dissimilarity="precomputed",
                random_state=random_state, normalized_stress="auto" if not metric else False)
    coords = model.fit_transform(distance_matrix)
    return {"coordinates": coords.tolist(), "stress": float(model.stress_)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    points = rng.normal(0, 1, (15, 4))
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    r = multidimensional_scaling(d, n_components=2)
    print("stress:", r["stress"])

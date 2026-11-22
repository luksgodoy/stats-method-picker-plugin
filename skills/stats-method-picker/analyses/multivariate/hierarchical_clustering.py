"""Hierarchical (agglomerative) clustering."""
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist


def hierarchical_clustering(x, method="ward", metric="euclidean", n_clusters=None, cut_height=None):
    x = np.asarray(x, dtype=float)
    z = linkage(x, method=method, metric=metric)
    if n_clusters is not None:
        labels = fcluster(z, t=n_clusters, criterion="maxclust")
    elif cut_height is not None:
        labels = fcluster(z, t=cut_height, criterion="distance")
    else:
        labels = None
    return {"linkage": z, "labels": labels.tolist() if labels is not None else None}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([0, 0], 0.5, (20, 2)), rng.normal([5, 5], 0.5, (20, 2))])
    r = hierarchical_clustering(x, n_clusters=2)
    print("labels:", r["labels"])
    print("minimum pairwise distance:", pdist(x).min())

"""K-means and DBSCAN."""
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score


def kmeans(x, n_clusters, random_state=0):
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(x)
    sil = silhouette_score(x, model.labels_) if n_clusters > 1 else None
    return {"labels": model.labels_.tolist(), "centers": model.cluster_centers_.tolist(),
            "inertia": model.inertia_, "silhouette": sil}


def dbscan(x, eps, min_samples):
    model = DBSCAN(eps=eps, min_samples=min_samples).fit(x)
    return {"labels": model.labels_.tolist(), "n_clusters": len(set(model.labels_) - {-1}),
            "n_noise": int((model.labels_ == -1).sum())}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([0, 0], 0.5, (30, 2)), rng.normal([6, 6], 0.5, (30, 2))])
    print("k-means:", {k: v for k, v in kmeans(x, 2).items() if k != "centers"})
    print("dbscan:", dbscan(x, eps=1.0, min_samples=5))

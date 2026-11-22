"""Non-linear dimensionality reduction for visualization (t-SNE / UMAP)."""
import numpy as np
from sklearn.manifold import TSNE
import umap


def tsne(x, n_components=2, perplexity=30, random_state=0):
    model = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, init="pca")
    return {"embedding": model.fit_transform(x).tolist(), "kl_divergence": float(model.kl_divergence_)}


def umap_reduction(x, n_components=2, n_neighbors=15, min_dist=0.1, random_state=0):
    model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                      random_state=random_state)
    return {"embedding": model.fit_transform(x).tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([0, 0, 0], 0.5, (30, 3)), rng.normal([5, 5, 5], 0.5, (30, 3))])
    r_tsne = tsne(x, perplexity=10)
    print("t-SNE KL divergence:", r_tsne["kl_divergence"])
    r_umap = umap_reduction(x, n_neighbors=10)
    print("UMAP shape:", np.array(r_umap["embedding"]).shape)

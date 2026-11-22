"""Recommender systems: item-item collaborative filtering
(cosine similarity) and matrix factorization via SVD."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def item_item_recommendation(ratings_matrix, user_idx, target_item_idx, top_n_neighbors=3):
    """ratings_matrix: (n_users x n_items), 0 = not rated. Predicts a user's rating for an
    item via an average weighted by item-item similarity."""
    item_similarity = cosine_similarity(ratings_matrix.T)
    target_item_similarities = item_similarity[target_item_idx]

    user_ratings = ratings_matrix[user_idx]
    rated_items = np.where(user_ratings > 0)[0]
    rated_items = rated_items[rated_items != target_item_idx]

    if len(rated_items) == 0:
        return {"prediction": float(user_ratings[user_ratings > 0].mean())}

    relevant_similarities = target_item_similarities[rated_items]
    top_indices = np.argsort(relevant_similarities)[::-1][:top_n_neighbors]
    neighbor_items = rated_items[top_indices]
    weights = target_item_similarities[neighbor_items]

    if weights.sum() == 0:
        prediction = user_ratings[rated_items].mean()
    else:
        prediction = np.average(user_ratings[neighbor_items], weights=weights)
    return {"prediction": float(prediction)}


def svd_factorization_recommendation(ratings_matrix, n_factors=2):
    u, s, vt = np.linalg.svd(ratings_matrix, full_matrices=False)
    reconstruction = u[:, :n_factors] @ np.diag(s[:n_factors]) @ vt[:n_factors, :]
    return {"reconstructed_matrix": reconstruction.round(2).tolist()}


if __name__ == "__main__":
    matrix = np.array([
        [5, 4, 0, 1],
        [4, 0, 0, 1],
        [1, 1, 0, 5],
        [1, 0, 0, 4],
        [0, 1, 5, 4],
    ], dtype=float)

    r_item = item_item_recommendation(matrix, user_idx=0, target_item_idx=2)
    print("prediction (user 0, item 2):", round(r_item["prediction"], 2))

    r_svd = svd_factorization_recommendation(matrix, n_factors=2)
    print("reconstructed matrix (SVD, 2 factors):", r_svd["reconstructed_matrix"][0])

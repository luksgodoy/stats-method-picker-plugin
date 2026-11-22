"""Analytic Hierarchy Process (AHP): multicriteria prioritization via a pairwise
comparison matrix and principal eigenvector, with a consistency check."""
import numpy as np

_RANDOM_INDEX = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
                 9: 1.45, 10: 1.49}


def ahp(comparison_matrix):
    """comparison_matrix: square matrix of pairwise comparisons (Saaty scale, 1-9)
    between criteria or alternatives. Returns normalized weights and the consistency ratio (CR)."""
    comparison_matrix = np.asarray(comparison_matrix, dtype=float)
    n = comparison_matrix.shape[0]

    eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
    principal_index = int(np.argmax(eigenvalues.real))
    max_eigenvalue = eigenvalues[principal_index].real
    weights = np.abs(eigenvectors[:, principal_index].real)
    weights /= weights.sum()

    consistency_index = (max_eigenvalue - n) / (n - 1) if n > 1 else 0.0
    ri = _RANDOM_INDEX.get(n, 1.49)
    consistency_ratio = consistency_index / ri if ri > 0 else 0.0

    return {"weights": weights.tolist(), "consistency_ratio": float(consistency_ratio),
            "consistent": bool(consistency_ratio < 0.10)}


if __name__ == "__main__":
    # Prioritize 3 criteria (cost, quality, lead time) by pairwise comparison
    matrix = [
        [1, 3, 5],
        [1 / 3, 1, 2],
        [1 / 5, 1 / 2, 1],
    ]
    r = ahp(matrix)
    print("weights:", [round(p, 3) for p in r["weights"]])
    print("consistency ratio:", round(r["consistency_ratio"], 4), "consistent:", r["consistent"])

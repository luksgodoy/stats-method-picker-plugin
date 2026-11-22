"""Pearson / Spearman correlation with significance test."""
from scipy import stats
import numpy as np


def pearson_correlation(x, y, alpha=0.05):
    r, p = stats.pearsonr(x, y)
    return {"r": float(r), "p_value": float(p), "significant": bool(p < alpha)}


def spearman_correlation(x, y, alpha=0.05):
    rho, p = stats.spearmanr(x, y)
    return {"rho": float(rho), "p_value": float(p), "significant": bool(p < alpha)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 60)
    y = 2 * x + rng.normal(0, 0.5, 60)
    print("pearson:", pearson_correlation(x, y))
    print("spearman:", spearman_correlation(x, y))

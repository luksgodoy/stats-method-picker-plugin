"""Mann-Whitney, Wilcoxon, Kruskal-Wallis."""
from scipy import stats
import numpy as np


def mann_whitney(a, b, alpha=0.05):
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"U": float(u), "p_value": float(p), "reject_h0": bool(p < alpha)}


def paired_wilcoxon(a, b, alpha=0.05):
    w, p = stats.wilcoxon(a, b)
    return {"W": float(w), "p_value": float(p), "reject_h0": bool(p < alpha)}


def kruskal_wallis(groups, alpha=0.05):
    h, p = stats.kruskal(*groups)
    return {"H": float(h), "p_value": float(p), "reject_h0": bool(p < alpha)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.exponential(2, 30)
    b = rng.exponential(3, 30)
    print("mann-whitney:", mann_whitney(a, b))
    print("wilcoxon:", paired_wilcoxon(a, a + rng.normal(0.5, 0.2, 30)))
    c = rng.exponential(4, 30)
    print("kruskal-wallis:", kruskal_wallis([a, b, c]))

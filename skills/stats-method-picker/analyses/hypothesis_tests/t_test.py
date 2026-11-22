"""t-test (one sample, two samples, paired). Wrapper over scipy.stats."""
from scipy import stats
import numpy as np


def one_sample_t_test(sample, mu0, alpha=0.05):
    t, p = stats.ttest_1samp(sample, mu0)
    return {"t": float(t), "p_value": float(p), "reject_h0": bool(p < alpha)}


def two_sample_t_test(a, b, paired=False, welch=True, alpha=0.05):
    if paired:
        t, p = stats.ttest_rel(a, b)
    else:
        t, p = stats.ttest_ind(a, b, equal_var=not welch)
    return {"t": float(t), "p_value": float(p), "reject_h0": bool(p < alpha)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    group_a = rng.normal(50, 5, 40)
    group_b = rng.normal(53, 5, 40)
    print("one sample:", one_sample_t_test(group_a, 50))
    print("two samples (Welch):", two_sample_t_test(group_a, group_b))
    print("paired:", two_sample_t_test(group_a, group_a + rng.normal(1, 1, 40), paired=True))

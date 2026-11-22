"""Bootstrapping (confidence interval by resampling)."""
import numpy as np


def bootstrap_confidence_interval(sample, statistic=np.mean, n_resamples=10_000, confidence=0.95, seed=0):
    rng = np.random.default_rng(seed)
    sample = np.asarray(sample)
    n = len(sample)
    statistics = np.array([statistic(sample[rng.integers(0, n, n)]) for _ in range(n_resamples)])
    alpha = 1 - confidence
    ci = np.percentile(statistics, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"sample_statistic": float(statistic(sample)), "bootstrap_std_error": float(statistics.std()),
            "ci_lower": float(ci[0]), "ci_upper": float(ci[1])}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sample = rng.exponential(5, 60)
    print("bootstrap CI of the mean:", bootstrap_confidence_interval(sample, n_resamples=5000))
    print("bootstrap CI of the median:", bootstrap_confidence_interval(sample, np.median, n_resamples=5000))

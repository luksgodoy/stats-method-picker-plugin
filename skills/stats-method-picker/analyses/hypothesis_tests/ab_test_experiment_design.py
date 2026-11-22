"""A/B test (proportions) and experiment design (sample size)."""
import numpy as np
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def ab_test_proportions(conversions, sizes, alpha=0.05):
    """conversions/sizes: lists of 2 elements, [control, treatment]."""
    z, p = proportions_ztest(conversions, sizes)
    rate_a = conversions[0] / sizes[0]
    rate_b = conversions[1] / sizes[1]
    return {"z": float(z), "p_value": float(p), "significant": bool(p < alpha),
            "control_rate": rate_a, "treatment_rate": rate_b,
            "relative_lift": (rate_b - rate_a) / rate_a if rate_a else None}


def ab_sample_size(p1, p2, alpha=0.05, power=0.8):
    effect = proportion_effectsize(p1, p2)
    n = NormalIndPower().solve_power(effect_size=effect, alpha=alpha, power=power, ratio=1.0)
    return {"n_per_group": int(np.ceil(n))}


if __name__ == "__main__":
    result = ab_test_proportions([120, 150], [1000, 1000])
    print("A/B test:", result)
    print("required sample size:", ab_sample_size(0.12, 0.15))
    print("CI of control conversion:", proportion_confint(120, 1000, alpha=0.05))

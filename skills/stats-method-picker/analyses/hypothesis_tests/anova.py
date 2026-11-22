"""ANOVA (one-way / factorial) and post-hoc (Tukey)."""
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
import statsmodels.api as sm


def one_way_anova(groups, alpha=0.05):
    f, p = stats.f_oneway(*groups)
    return {"F": float(f), "p_value": float(p), "reject_h0": bool(p < alpha)}


def factorial_anova(df, formula):
    """df: long-format DataFrame; formula e.g. 'response ~ C(factor_a) * C(factor_b)'."""
    model = smf.ols(formula, data=df).fit()
    return sm.stats.anova_lm(model, typ=2)


def tukey_post_hoc(values, groups, alpha=0.05):
    res = pairwise_tukeyhsd(endog=values, groups=groups, alpha=alpha)
    return res


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    g1, g2, g3 = rng.normal(10, 2, 30), rng.normal(12, 2, 30), rng.normal(15, 2, 30)
    print("one-way anova:", one_way_anova([g1, g2, g3]))

    values = np.concatenate([g1, g2, g3])
    groups = np.array(["A"] * 30 + ["B"] * 30 + ["C"] * 30)
    print(tukey_post_hoc(values, groups))

    df = pd.DataFrame({
        "response": values,
        "factor_a": np.tile(["x", "y"], 45),
        "factor_b": groups,
    })
    print(factorial_anova(df, "response ~ C(factor_a) * C(factor_b)"))

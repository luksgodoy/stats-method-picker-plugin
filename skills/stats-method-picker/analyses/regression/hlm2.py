"""HLM2 multilevel model (2 levels: individuals nested in groups)."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def hlm2(df, fixed_formula, group_col, random_slope=None):
    """df: long format with group_col identifying level 2 (e.g. school, company)."""
    re_formula = f"~{random_slope}" if random_slope else None
    model = smf.mixedlm(fixed_formula, df, groups=df[group_col], re_formula=re_formula).fit()
    return {"fixed_effects": model.fe_params.to_dict(), "group_variance": model.cov_re.values.tolist(),
            "log_likelihood": model.llf, "summary": model.summary().as_text()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_groups, n_per_group = 20, 15
    group = np.repeat(np.arange(n_groups), n_per_group)
    group_effect = rng.normal(0, 2, n_groups)[group]
    x = rng.normal(0, 1, n_groups * n_per_group)
    y = 10 + 1.5 * x + group_effect + rng.normal(0, 1, n_groups * n_per_group)
    df = pd.DataFrame({"y": y, "x": x, "group": group})
    r = hlm2(df, "y ~ x", "group")
    print("fixed effects:", r["fixed_effects"])

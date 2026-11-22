"""HLM3 multilevel model (3 levels: individuals in groups in larger units),
via statsmodels nested variance components (vc_formula)."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def hlm3(df, fixed_formula, level2_col, level3_col):
    """Level 3 (e.g. company) is the main group; level 2 (e.g. branch) is a variance
    component nested within level 3."""
    vc = {"level2": f"0 + C({level2_col})"}
    model = smf.mixedlm(fixed_formula, df, groups=df[level3_col], vc_formula=vc).fit()
    return {"fixed_effects": model.fe_params.to_dict(), "log_likelihood": model.llf,
            "summary": model.summary().as_text()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n3, n2_per_3, n1_per_2 = 6, 4, 10
    level3 = np.repeat(np.arange(n3), n2_per_3 * n1_per_2)
    level2 = np.tile(np.repeat(np.arange(n2_per_3), n1_per_2), n3) + level3 * n2_per_3
    effect3 = rng.normal(0, 2, n3)[level3]
    effect2 = rng.normal(0, 1, level2.max() + 1)[level2]
    x = rng.normal(0, 1, len(level3))
    y = 5 + 1.2 * x + effect3 + effect2 + rng.normal(0, 1, len(level3))
    df = pd.DataFrame({"y": y, "x": x, "level2": level2, "level3": level3})
    r = hlm3(df, "y ~ x", "level2", "level3")
    print("fixed effects:", r["fixed_effects"])

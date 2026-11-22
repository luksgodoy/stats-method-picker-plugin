"""Differences-in-Differences (DiD): estimates the causal effect of a treatment by
comparing the change over time between a treated group and a control group."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def difference_in_differences(df, outcome_col, treated_col, post_period_col):
    """df must have: outcome_col (variable of interest), treated_col (1 = treated group,
    0 = control), post_period_col (1 = after the treatment, 0 = before)."""
    df = df.copy()
    df["interaction"] = df[treated_col] * df[post_period_col]
    formula = f"{outcome_col} ~ {treated_col} + {post_period_col} + interaction"
    model = smf.ols(formula, data=df).fit()
    return {"did_causal_effect": float(model.params["interaction"]),
            "p_value": float(model.pvalues["interaction"]),
            "r2": float(model.rsquared)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    treated = rng.integers(0, 2, n)
    post_period = rng.integers(0, 2, n)
    true_effect = 5.0
    outcome = (10 + 2 * treated + 3 * post_period + true_effect * treated * post_period
               + rng.normal(0, 2, n))
    df = pd.DataFrame({"outcome": outcome, "treated": treated, "post_period": post_period})
    r = difference_in_differences(df, "outcome", "treated", "post_period")
    print("estimated causal effect (DiD):", round(r["did_causal_effect"], 2), "(true: 5.0)")
    print("p-value:", r["p_value"])

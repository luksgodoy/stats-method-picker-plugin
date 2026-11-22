"""Conjoint analysis (part-worth utilities via regression on dummy attributes)."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def conjoint_analysis(df_profiles, preference_col, attribute_cols):
    """df_profiles: one profile (combination of attribute levels) per row, with the
    respondent's preference (rating/ranking). Returns part-worth utilities and the
    relative importance of each attribute."""
    dummies = pd.get_dummies(df_profiles[attribute_cols], drop_first=True).astype(float)
    x = sm.add_constant(dummies)
    y = df_profiles[preference_col].astype(float)
    model = sm.OLS(y, x).fit()

    part_worths = model.params.drop("const").to_dict()
    importance = {}
    for attribute in attribute_cols:
        levels = [c for c in part_worths if c.startswith(attribute + "_")]
        span = (max(part_worths[c] for c in levels) - min(0, *(part_worths[c] for c in levels))) if levels else 0
        importance[attribute] = span
    total = sum(importance.values()) or 1
    relative_importance = {k: v / total for k, v in importance.items()}
    return {"part_worths": part_worths, "relative_importance": relative_importance,
            "r2": model.rsquared}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    profiles = pd.DataFrame({
        "brand": rng.choice(["A", "B", "C"], 60),
        "price": rng.choice(["low", "medium", "high"], 60),
        "warranty": rng.choice(["1year", "2years"], 60),
    })
    base_utility = {"brand": {"A": 2, "B": 0, "C": -1}, "price": {"low": 3, "medium": 1, "high": -2},
                    "warranty": {"1year": 0, "2years": 1}}
    profiles["preference"] = (profiles["brand"].map(base_utility["brand"])
                              + profiles["price"].map(base_utility["price"])
                              + profiles["warranty"].map(base_utility["warranty"])
                              + rng.normal(0, 0.5, 60))
    r = conjoint_analysis(profiles, "preference", ["brand", "price", "warranty"])
    print("relative importance:", r["relative_importance"])

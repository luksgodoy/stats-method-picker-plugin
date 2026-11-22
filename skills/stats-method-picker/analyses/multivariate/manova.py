"""MANOVA (multiple response variables)."""
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA


def manova(df, formula):
    """df: long-format DataFrame; formula e.g. 'y1 + y2 ~ C(group)'."""
    model = MANOVA.from_formula(formula, data=df)
    return model.mv_test()


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    group = np.repeat(["A", "B", "C"], 30)
    y1 = rng.normal(0, 1, 90) + (group == "B") * 1.5 + (group == "C") * 3
    y2 = rng.normal(0, 1, 90) + (group == "B") * 0.5 + (group == "C") * 1.0
    df = pd.DataFrame({"y1": y1, "y2": y2, "group": group})
    print(manova(df, "y1 + y2 ~ C(group)"))

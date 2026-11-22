"""Multiple correspondence analysis, via CA on the indicator (one-hot) matrix."""
import numpy as np
import pandas as pd
from analyses.multivariate.anacor import anacor


def mca(df_categorical):
    """df_categorical: DataFrame with categorical columns (variables to analyze together)."""
    indicator = pd.get_dummies(df_categorical)
    result = anacor(indicator.values)
    result["indicator_columns"] = indicator.columns.tolist()
    return result


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "region": rng.choice(["north", "south", "east"], 100),
        "age_group": rng.choice(["young", "adult", "senior"], 100),
        "purchase_category": rng.choice(["A", "B", "C"], 100),
    })
    r = mca(df)
    print("explained inertia (first 2 dims):", r["explained_inertia"][:2])

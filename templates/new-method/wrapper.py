"""My Method: one-line summary of what the wrapper computes.

Longer description if useful: assumptions, what the returned dict contains.
"""
import numpy as np


def my_method(x, y):
    """Fit the method and return a dict of plain Python values (floats, lists)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))
    return {"slope": slope, "intercept": float(y.mean() - slope * x.mean())}


if __name__ == "__main__":
    # Minimal runnable example on simulated data (no files, no personal data).
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 100)
    y = 2 + 1.5 * x + rng.normal(0, 0.5, 100)
    print(my_method(x, y))

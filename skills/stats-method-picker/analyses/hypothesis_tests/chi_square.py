"""Chi-square test of independence and goodness of fit."""
from scipy import stats
import numpy as np


def chi_square_independence(contingency_table, alpha=0.05):
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p),
            "reject_h0": bool(p < alpha), "expected": expected.tolist()}


def chi_square_goodness_of_fit(observed, expected=None, alpha=0.05):
    chi2, p = stats.chisquare(observed, f_exp=expected)
    return {"chi2": float(chi2), "p_value": float(p), "reject_h0": bool(p < alpha)}


if __name__ == "__main__":
    table = np.array([[30, 10], [15, 25]])
    print("independence:", chi_square_independence(table))
    print("goodness of fit:", chi_square_goodness_of_fit([18, 22, 20, 20], [20, 20, 20, 20]))

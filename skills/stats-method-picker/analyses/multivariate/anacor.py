"""Simple correspondence analysis, via SVD of the standardized residuals matrix."""
import numpy as np


def anacor(contingency_table):
    n = np.asarray(contingency_table, dtype=float)
    p = n / n.sum()
    r = p.sum(axis=1)
    c = p.sum(axis=0)
    # Accelerate/BLAS on macOS emits a spurious RuntimeWarning in matmul with a large np.diag;
    # the result stays finite (checked below) - this is not a genuine division by zero r/c.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        d_r = np.diag(1 / np.sqrt(r))
        d_c = np.diag(1 / np.sqrt(c))
        s = d_r @ (p - np.outer(r, c)) @ d_c
        u, sv, vt = np.linalg.svd(s, full_matrices=False)
        inertia = sv ** 2
        row_coords = d_r @ u * sv
        col_coords = d_c @ vt.T * sv
    assert np.isfinite(s).all(), "standardized residuals matrix has a non-finite value"
    return {
        "total_inertia": float(inertia.sum()),
        "explained_inertia": (inertia / inertia.sum()).tolist(),
        "row_coordinates": row_coords.tolist(),
        "column_coordinates": col_coords.tolist(),
    }


if __name__ == "__main__":
    table = np.array([[30, 10, 5], [15, 25, 10], [5, 10, 20]])
    r = anacor(table)
    print("explained inertia:", r["explained_inertia"])

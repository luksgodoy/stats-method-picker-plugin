"""Simplex method implemented manually (tableau), for maximizing
Ax <= b, x >= 0 - illustrates the internal mechanism behind solvers like linprog."""
import numpy as np


def manual_simplex(objective_coefficients, constraint_matrix, constraint_bounds, max_iterations=100):
    """Maximizes objective_coefficients·x subject to constraint_matrix @ x <= constraint_bounds, x >= 0."""
    n_constraints, n_variables = constraint_matrix.shape
    tableau = np.zeros((n_constraints + 1, n_variables + n_constraints + 1))
    tableau[:n_constraints, :n_variables] = constraint_matrix
    tableau[:n_constraints, n_variables:n_variables + n_constraints] = np.eye(n_constraints)
    tableau[:n_constraints, -1] = constraint_bounds
    tableau[-1, :n_variables] = -objective_coefficients

    for _ in range(max_iterations):
        if np.all(tableau[-1, :-1] >= -1e-9):
            break
        pivot_col = int(np.argmin(tableau[-1, :-1]))
        ratios = np.full(n_constraints, np.inf)
        for i in range(n_constraints):
            if tableau[i, pivot_col] > 1e-9:
                ratios[i] = tableau[i, -1] / tableau[i, pivot_col]
        pivot_row = int(np.argmin(ratios))
        if not np.isfinite(ratios[pivot_row]):
            return {"success": False, "reason": "unbounded problem"}

        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        for i in range(n_constraints + 1):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

    solution = np.zeros(n_variables)
    for col in range(n_variables):
        vector = tableau[:-1, col]
        if np.count_nonzero(np.abs(vector - 1) < 1e-9) == 1 and np.count_nonzero(np.abs(vector) > 1e-9) == 1:
            row = int(np.argmax(np.abs(vector - 1) < 1e-9))
            solution[col] = tableau[row, -1]

    return {"success": True, "solution": solution.tolist(), "objective_value": float(tableau[-1, -1])}


if __name__ == "__main__":
    # Maximize 3x1 + 5x2, subject to: x1 <= 4; 2x2 <= 12; 3x1 + 2x2 <= 18
    objective_coefficients = np.array([3.0, 5.0])
    constraint_matrix = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 2.0]])
    constraint_bounds = np.array([4.0, 12.0, 18.0])
    r = manual_simplex(objective_coefficients, constraint_matrix, constraint_bounds)
    print("optimal solution:", r["solution"], "objective value:", r["objective_value"])

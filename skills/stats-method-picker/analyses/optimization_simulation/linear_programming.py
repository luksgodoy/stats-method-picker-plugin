"""Linear programming (optimizing a linear objective function subject to
linear constraints) via scipy.optimize.linprog."""
import numpy as np
from scipy.optimize import linprog


def optimize_production(costs, constraint_matrix, constraint_bounds, variable_bounds=None):
    """Minimizes costs·x subject to constraint_matrix @ x <= constraint_bounds.
    To maximize profit, pass costs as -profits (linprog only minimizes)."""
    result = linprog(costs, A_ub=constraint_matrix, b_ub=constraint_bounds,
                     bounds=variable_bounds, method="highs")
    return {"success": bool(result.success), "solution": result.x.tolist(),
            "objective_value": float(result.fun)}


if __name__ == "__main__":
    # Maximize profit = 3x1 + 5x2, subject to: x1 <= 4; 2x2 <= 12; 3x1 + 2x2 <= 18; x >= 0
    profits = np.array([3, 5])
    costs = -profits
    constraint_matrix = np.array([[1, 0], [0, 2], [3, 2]])
    constraint_bounds = np.array([4, 12, 18])
    r = optimize_production(costs, constraint_matrix, constraint_bounds, variable_bounds=[(0, None)] * 2)
    print("optimal solution:", r["solution"], "maximum profit:", -r["objective_value"])

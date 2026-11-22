"""Integer linear programming applied to the transportation problem (allocating
sources to destinations at minimum cost, with integer variables)."""
import numpy as np
from scipy.optimize import linprog


def solve_transport_problem(costs, supplies, demands):
    """costs: matrix (n_sources x n_destinations); supplies: capacity of each source;
    demands: requirement of each destination. Solves via the relaxed LP (the solution tends
    to be integer in balanced transportation problems, due to their totally unimodular structure)."""
    costs = np.asarray(costs, dtype=float)
    n_sources, n_destinations = costs.shape
    cost_vector = costs.ravel()

    equality_matrix = []
    equality_bounds = []
    for source in range(n_sources):
        row = np.zeros(n_sources * n_destinations)
        row[source * n_destinations:(source + 1) * n_destinations] = 1
        equality_matrix.append(row)
        equality_bounds.append(supplies[source])
    for destination in range(n_destinations):
        row = np.zeros(n_sources * n_destinations)
        row[destination::n_destinations] = 1
        equality_matrix.append(row)
        equality_bounds.append(demands[destination])

    result = linprog(cost_vector, A_eq=equality_matrix, b_eq=equality_bounds,
                     bounds=(0, None), method="highs")
    allocation = result.x.reshape(n_sources, n_destinations)
    return {"success": bool(result.success), "allocation": allocation.round(4).tolist(),
            "total_cost": float(result.fun)}


if __name__ == "__main__":
    costs = [[4, 6, 8], [5, 4, 7], [6, 5, 3]]
    supplies = [120, 150, 100]
    demands = [110, 130, 130]
    r = solve_transport_problem(costs, supplies, demands)
    print("minimum total cost:", r["total_cost"])
    print("allocation:", r["allocation"])

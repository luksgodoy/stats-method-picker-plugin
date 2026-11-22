"""Manual boosting: gradient boosting implemented from scratch,
successively fitting shallow trees to the residuals."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def manual_boosting(x, y, n_rounds=50, learning_rate=0.1, max_depth=2):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    prediction = np.full(len(y), y.mean())
    trees = []
    error_history = []
    for _ in range(n_rounds):
        residual = y - prediction
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(x, residual)
        prediction = prediction + learning_rate * tree.predict(x)
        trees.append(tree)
        error_history.append(float(np.mean((y - prediction) ** 2)))

    def predict(x_new, initial_mean=y.mean()):
        p = np.full(len(x_new), initial_mean)
        for tree in trees:
            p = p + learning_rate * tree.predict(x_new)
        return p

    return {"final_squared_error": error_history[-1], "error_history": error_history, "predict": predict}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (200, 3))
    y = 3 * x[:, 0] ** 2 - 2 * x[:, 1] + rng.normal(0, 0.5, 200)
    r = manual_boosting(x, y)
    print("initial vs final squared error:", r["error_history"][0], "->", r["final_squared_error"])

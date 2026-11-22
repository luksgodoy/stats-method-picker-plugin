"""Multilayer neural network (deep MLP) for regression and classification."""
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def mlp_classification(x, y, hidden_layers=(16, 8)):
    x = StandardScaler().fit_transform(x)
    model = MLPClassifier(hidden_layer_sizes=hidden_layers, activation="relu",
                          max_iter=3000, random_state=0)
    cv_accuracy = cross_val_score(model, x, y, cv=5).mean()
    return {"cv_accuracy": float(cv_accuracy)}


def mlp_regression(x, y, hidden_layers=(16, 8)):
    x = StandardScaler().fit_transform(x)
    model = MLPRegressor(hidden_layer_sizes=hidden_layers, activation="relu",
                         max_iter=3000, random_state=0)
    cv_r2 = cross_val_score(model, x, y, cv=5, scoring="r2").mean()
    return {"cv_r2": float(cv_r2)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (250, 4))
    y_clf = ((x[:, 0] ** 2 + x[:, 1] - x[:, 2] * x[:, 3]) > 1).astype(int)
    y_reg = np.sin(x[:, 0]) * 3 + x[:, 1] + rng.normal(0, 0.3, 250)
    print("classification, CV accuracy:", mlp_classification(x, y_clf))
    print("regression, CV r2:", mlp_regression(x, y_reg))

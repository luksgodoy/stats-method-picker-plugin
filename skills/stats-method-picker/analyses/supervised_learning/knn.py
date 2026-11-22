"""K-Nearest Neighbors (KNN) for classification and regression."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV


def knn_classification(x, y, k_candidates=(3, 5, 7, 9, 11)):
    x = StandardScaler().fit_transform(x)
    search = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": list(k_candidates)}, cv=5)
    search.fit(x, y)
    return {"best_k": int(search.best_params_["n_neighbors"]),
            "cv_accuracy": float(search.best_score_)}


def knn_regression(x, y, k_candidates=(3, 5, 7, 9, 11)):
    x = StandardScaler().fit_transform(x)
    search = GridSearchCV(KNeighborsRegressor(), {"n_neighbors": list(k_candidates)}, cv=5, scoring="r2")
    search.fit(x, y)
    return {"best_k": int(search.best_params_["n_neighbors"]),
            "cv_r2": float(search.best_score_)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 4))
    y_clf = ((x[:, 0] + x[:, 1]) > 0).astype(int)
    y_reg = x[:, 0] * 2 - x[:, 1] + rng.normal(0, 0.4, 300)
    print("classification:", knn_classification(x, y_clf))
    print("regression:", knn_regression(x, y_reg))

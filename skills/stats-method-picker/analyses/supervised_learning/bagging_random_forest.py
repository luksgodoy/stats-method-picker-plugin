"""Bagging and Random Forest."""
import numpy as np
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


def bagging(x, y, n_estimators=100):
    model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=n_estimators, random_state=0)
    cv_accuracy = cross_val_score(model, x, y, cv=5).mean()
    return {"cv_accuracy": float(cv_accuracy)}


def random_forest(x, y, n_estimators=200):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=0).fit(x, y)
    cv_accuracy = cross_val_score(model, x, y, cv=5).mean()
    return {"cv_accuracy": float(cv_accuracy), "feature_importances": model.feature_importances_.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 5))
    y = ((x[:, 0] + x[:, 1] - x[:, 2] + rng.normal(0, 0.5, 300)) > 0).astype(int)
    print("bagging:", bagging(x, y))
    print("random forest:", {k: v for k, v in random_forest(x, y).items() if k != "feature_importances"})

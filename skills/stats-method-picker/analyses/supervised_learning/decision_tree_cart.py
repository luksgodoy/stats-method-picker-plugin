"""Decision tree / regression tree (CART)."""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_text
from sklearn.model_selection import cross_val_score


def classification_tree(x, y, max_depth=4):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(x, y)
    cv_accuracy = cross_val_score(model, x, y, cv=5).mean()
    return {"cv_accuracy": float(cv_accuracy), "feature_importances": model.feature_importances_.tolist(),
            "tree_text": export_text(model)}


def regression_tree(x, y, max_depth=4):
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=0).fit(x, y)
    cv_r2 = cross_val_score(model, x, y, cv=5, scoring="r2").mean()
    return {"cv_r2": float(cv_r2), "feature_importances": model.feature_importances_.tolist()}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (200, 4))
    y_clf = (x[:, 0] + x[:, 1] > 0).astype(int)
    y_reg = 2 * x[:, 0] - x[:, 1] + rng.normal(0, 0.5, 200)
    print("classification, CV accuracy:", classification_tree(x, y_clf)["cv_accuracy"])
    print("regression, CV r2:", regression_tree(x, y_reg)["cv_r2"])

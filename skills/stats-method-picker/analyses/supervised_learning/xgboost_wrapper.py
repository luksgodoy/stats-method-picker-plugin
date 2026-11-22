"""XGBoost (optimized gradient boosting)."""
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score


def xgboost_classification(x, y, n_estimators=200, max_depth=4, learning_rate=0.1):
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                              learning_rate=learning_rate, random_state=0,
                              eval_metric="logloss")
    cv_accuracy = cross_val_score(model, x, y, cv=5).mean()
    model.fit(x, y)
    return {"cv_accuracy": float(cv_accuracy), "feature_importances": model.feature_importances_.tolist()}


def xgboost_regression(x, y, n_estimators=200, max_depth=4, learning_rate=0.1):
    model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate, random_state=0)
    cv_r2 = cross_val_score(model, x, y, cv=5, scoring="r2").mean()
    return {"cv_r2": float(cv_r2)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 5))
    y_clf = ((x[:, 0] + x[:, 1] - x[:, 2]) > 0).astype(int)
    y_reg = 2 * x[:, 0] - x[:, 1] + rng.normal(0, 0.5, 300)
    print("classification, CV accuracy:", xgboost_classification(x, y_clf)["cv_accuracy"])
    print("regression, CV r2:", xgboost_regression(x, y_reg)["cv_r2"])

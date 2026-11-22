"""Support Vector Machine (SVM) for classification and regression."""
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


def svm_classification(x, y, kernel="rbf", c=1.0):
    x = StandardScaler().fit_transform(x)
    model = SVC(kernel=kernel, C=c, random_state=0)
    cv_accuracy = cross_val_score(model, x, y, cv=5).mean()
    model.fit(x, y)
    return {"cv_accuracy": float(cv_accuracy), "n_support_vectors": model.n_support_.tolist()}


def svm_regression(x, y, kernel="rbf", c=1.0):
    x = StandardScaler().fit_transform(x)
    model = SVR(kernel=kernel, C=c)
    cv_r2 = cross_val_score(model, x, y, cv=5, scoring="r2").mean()
    return {"cv_r2": float(cv_r2)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 4))
    y_clf = ((x[:, 0] ** 2 + x[:, 1] ** 2) > 1.5).astype(int)
    y_reg = np.sin(x[:, 0]) * 2 + x[:, 1] + rng.normal(0, 0.3, 300)
    print("classification:", svm_classification(x, y_clf))
    print("regression:", svm_regression(x, y_reg))

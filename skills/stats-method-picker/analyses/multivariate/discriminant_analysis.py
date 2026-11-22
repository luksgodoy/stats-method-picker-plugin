"""Linear discriminant analysis (LDA)."""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score


def discriminant_analysis(x, y, n_components=None):
    x = np.asarray(x, dtype=float)
    model = LinearDiscriminantAnalysis(n_components=n_components)
    model.fit(x, y)
    cv_accuracy = cross_val_score(model, x, y, cv=5).mean()
    return {
        "train_accuracy": model.score(x, y),
        "cross_validation_accuracy": float(cv_accuracy),
        "discriminant_scores": model.transform(x).tolist(),
        "classes": model.classes_.tolist(),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = np.vstack([rng.normal([0, 0], 1, (50, 2)), rng.normal([4, 4], 1, (50, 2)), rng.normal([0, 4], 1, (50, 2))])
    y = np.repeat(["A", "B", "C"], 50)
    r = discriminant_analysis(x, y)
    print("cross-validation accuracy:", r["cross_validation_accuracy"])

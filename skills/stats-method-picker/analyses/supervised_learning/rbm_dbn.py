"""Restricted Boltzmann Machine (RBM) / Deep Belief Network for unsupervised feature
extraction, used as pre-training for a classifier."""
import numpy as np
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler


def rbm_feature_extraction(x, y, n_components=20):
    """Stacks an RBM (extracts binary latent features) with a linear classifier -
    approximates the role of a shallow Deep Belief Network (1 pre-training layer)."""
    x = MinMaxScaler().fit_transform(x)
    rbm = BernoulliRBM(n_components=n_components, learning_rate=0.05, n_iter=50, random_state=0)
    classifier = LogisticRegression(max_iter=1000)
    pipeline = Pipeline([("rbm", rbm), ("clf", classifier)])
    cv_accuracy = cross_val_score(pipeline, x, y, cv=5).mean()
    return {"cv_accuracy": float(cv_accuracy), "n_latent_components": n_components}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (300, 12))
    y = ((x[:, :6].sum(axis=1)) > 3).astype(int)
    r = rbm_feature_extraction(x, y)
    print("CV accuracy (RBM + logistic):", r["cv_accuracy"])

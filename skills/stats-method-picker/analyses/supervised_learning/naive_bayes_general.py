"""Naive Bayes (Gaussian, Multinomial and Bernoulli) for general classification
(outside the text-specific context - see analyses/text for the text case)."""
import numpy as np
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score


def gaussian_naive_bayes(x, y):
    model = GaussianNB()
    cv_accuracy = cross_val_score(model, x, y, cv=5).mean()
    return {"cv_accuracy": float(cv_accuracy), "variant": "gaussian"}


def multinomial_naive_bayes(x_counts, y):
    model = MultinomialNB()
    cv_accuracy = cross_val_score(model, x_counts, y, cv=5).mean()
    return {"cv_accuracy": float(cv_accuracy), "variant": "multinomial"}


def bernoulli_naive_bayes(x_binary, y):
    model = BernoulliNB()
    cv_accuracy = cross_val_score(model, x_binary, y, cv=5).mean()
    return {"cv_accuracy": float(cv_accuracy), "variant": "bernoulli"}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x_continuous = rng.normal(0, 1, (300, 4))
    y = ((x_continuous[:, 0] + x_continuous[:, 1]) > 0).astype(int)
    print("gaussian:", gaussian_naive_bayes(x_continuous, y))

    x_counts = rng.poisson(3, (300, 6))
    print("multinomial:", multinomial_naive_bayes(x_counts, y))

    x_binary = (rng.uniform(0, 1, (300, 6)) > 0.5).astype(int)
    print("bernoulli:", bernoulli_naive_bayes(x_binary, y))

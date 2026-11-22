"""Bayesian regression (BayesianRidge) and conjugate Beta-Binomial test."""
import numpy as np
from scipy import stats
from sklearn.linear_model import BayesianRidge


def bayesian_regression(x, y):
    model = BayesianRidge().fit(x, y)
    mean, std = model.predict(x, return_std=True)
    return {"coefficients": model.coef_.tolist(), "intercept": model.intercept_,
            "alpha_precision": model.alpha_, "lambda_precision": model.lambda_,
            "mean_prediction_std": float(std.mean())}


def beta_binomial_ab(successes_a, n_a, successes_b, n_b, prior_alpha=1, prior_beta=1, n_samples=200_000, seed=0):
    """Bayesian A/B test: Beta posterior for each conversion rate, P(B > A) by Monte Carlo."""
    rng = np.random.default_rng(seed)
    post_a = stats.beta(prior_alpha + successes_a, prior_beta + n_a - successes_a)
    post_b = stats.beta(prior_alpha + successes_b, prior_beta + n_b - successes_b)
    samples_a = post_a.rvs(n_samples, random_state=rng)
    samples_b = post_b.rvs(n_samples, random_state=rng)
    return {"posterior_mean_a": post_a.mean(), "posterior_mean_b": post_b.mean(),
            "p_b_greater_than_a": float((samples_b > samples_a).mean())}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (100, 3))
    y = 2 + x @ [1.5, -0.8, 0.3] + rng.normal(0, 1, 100)
    r = bayesian_regression(x, y)
    print("coefficients:", r["coefficients"])
    print("Bayesian A/B test:", beta_binomial_ab(120, 1000, 150, 1000))

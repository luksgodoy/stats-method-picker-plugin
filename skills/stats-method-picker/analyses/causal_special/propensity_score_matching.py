"""Propensity Score Matching (PSM): estimates the causal effect of a treatment by
matching treated and control units with a similar probability of treatment."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors


def propensity_score_matching(x_covariates, treatment, outcome, n_neighbors=1):
    """x_covariates: covariate matrix; treatment: binary array (1 = treated);
    outcome: variable of interest. Returns the Average Treatment Effect on the Treated (ATT)."""
    propensity_model = LogisticRegression(max_iter=1000)
    propensity_model.fit(x_covariates, treatment)
    scores = propensity_model.predict_proba(x_covariates)[:, 1].reshape(-1, 1)

    treated_indices = np.where(treatment == 1)[0]
    control_indices = np.where(treatment == 0)[0]

    neighbors = NearestNeighbors(n_neighbors=n_neighbors).fit(scores[control_indices])
    _, matches = neighbors.kneighbors(scores[treated_indices])

    differences = []
    for i, treated_idx in enumerate(treated_indices):
        matched_controls = control_indices[matches[i]]
        differences.append(outcome[treated_idx] - outcome[matched_controls].mean())

    return {"att": float(np.mean(differences)), "n_pairs": len(differences)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 500
    x = rng.normal(0, 1, (n, 3))
    true_propensity = 1 / (1 + np.exp(-(x[:, 0] + 0.5 * x[:, 1])))
    treatment = (rng.uniform(0, 1, n) < true_propensity).astype(int)
    true_effect = 4.0
    outcome = 10 + 2 * x[:, 0] - x[:, 1] + true_effect * treatment + rng.normal(0, 1, n)
    r = propensity_score_matching(x, treatment, outcome)
    print("estimated ATT:", round(r["att"], 2), "(true: 4.0)", "n_pairs:", r["n_pairs"])

"""Survival analysis (Kaplan-Meier and Cox regression)."""
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter


def kaplan_meier(durations, events):
    """durations: observed time; events: 1 if the event occurred, 0 if censored."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=events)
    return {"median_survival": float(kmf.median_survival_time_),
            "final_survival": float(kmf.survival_function_.iloc[-1, 0])}


def cox_regression(df, duration_col, event_col):
    """df: DataFrame with covariate columns + duration + event (1 = event, 0 = censored)."""
    cph = CoxPHFitter()
    cph.fit(df, duration_col=duration_col, event_col=event_col)
    return {"hazard_ratios": cph.hazard_ratios_.to_dict(),
            "concordance": float(cph.concordance_index_)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(0, 1, n)
    x2 = rng.uniform(0, 1, n)
    durations = rng.exponential(10 + 3 * x1 - 2 * x2)
    censoring = rng.uniform(0, 20, n)
    events = (durations <= censoring).astype(int)
    observed_durations = np.minimum(durations, censoring)

    r_km = kaplan_meier(observed_durations, events)
    print("Kaplan-Meier:", r_km)

    df = pd.DataFrame({"x1": x1, "x2": x2, "duration": observed_durations, "event": events})
    r_cox = cox_regression(df, "duration", "event")
    print("Cox - concordance:", r_cox["concordance"])

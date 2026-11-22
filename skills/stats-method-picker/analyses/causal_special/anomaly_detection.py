"""Anomaly/outlier detection via Isolation Forest and Local Outlier Factor."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def detect_anomalies_isolation_forest(x, contamination=0.05):
    model = IsolationForest(contamination=contamination, random_state=0)
    labels = model.fit_predict(x)
    return {"anomaly_indices": np.where(labels == -1)[0].tolist(),
            "n_anomalies": int((labels == -1).sum())}


def detect_anomalies_lof(x, contamination=0.05, n_neighbors=20):
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = model.fit_predict(x)
    return {"anomaly_indices": np.where(labels == -1)[0].tolist(),
            "n_anomalies": int((labels == -1).sum())}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x_normal = rng.normal(0, 1, (300, 3))
    x_anomalous = rng.normal(8, 1, (15, 3))
    x = np.vstack([x_normal, x_anomalous])
    r_if = detect_anomalies_isolation_forest(x)
    r_lof = detect_anomalies_lof(x)
    print("Isolation Forest - anomalies detected:", r_if["n_anomalies"])
    print("LOF - anomalies detected:", r_lof["n_anomalies"])

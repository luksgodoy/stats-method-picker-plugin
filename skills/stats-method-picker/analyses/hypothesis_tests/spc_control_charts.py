"""Control charts (X-bar/R) and process capability (Cp/Cpk)."""
import numpy as np

# Tabulated Shewhart constants for small subgroups (n=2..10).
_A2 = {2: 1.880, 3: 1.023, 4: 0.729, 5: 0.577, 6: 0.483, 7: 0.419, 8: 0.373, 9: 0.337, 10: 0.308}
_D3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0.076, 8: 0.136, 9: 0.184, 10: 0.223}
_D4 = {2: 3.267, 3: 2.574, 4: 2.282, 5: 2.114, 6: 2.004, 7: 1.924, 8: 1.864, 9: 1.816, 10: 1.777}


def xbar_r_chart(subgroups):
    """subgroups: array (k, n) - k subgroups of size n."""
    subgroups = np.asarray(subgroups, dtype=float)
    n = subgroups.shape[1]
    means = subgroups.mean(axis=1)
    ranges = subgroups.max(axis=1) - subgroups.min(axis=1)
    grand_mean = means.mean()
    r_bar = ranges.mean()
    a2, d3, d4 = _A2[n], _D3[n], _D4[n]
    return {
        "cl_x": grand_mean,
        "ucl_x": grand_mean + a2 * r_bar,
        "lcl_x": grand_mean - a2 * r_bar,
        "cl_r": r_bar,
        "ucl_r": d4 * r_bar,
        "lcl_r": d3 * r_bar,
        "out_of_control": [i for i, m in enumerate(means)
                           if m > grand_mean + a2 * r_bar or m < grand_mean - a2 * r_bar],
    }


def process_capability(data, lsl, usl):
    data = np.asarray(data, dtype=float)
    mean, std = data.mean(), data.std(ddof=1)
    cp = (usl - lsl) / (6 * std)
    cpk = min((usl - mean) / (3 * std), (mean - lsl) / (3 * std))
    return {"mean": mean, "std": std, "cp": cp, "cpk": cpk,
            "capable_process": bool(cpk >= 1.33)}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    subgroups = rng.normal(50, 1.5, size=(25, 5))
    print("X-bar/R chart:", xbar_r_chart(subgroups))
    print("capability:", process_capability(subgroups.ravel(), lsl=45, usl=55))

"""User's accuracy (UA), producer's accuracy (PA) and F1-score of the target class of a map."""
from __future__ import annotations

import numpy as np


def accuracy_stratified(stratum, mapped, reference, W_h):
    """UA, PA and F1 of one map and their standard errors, estimated from a stratified sample.

    With x = 1 if a unit is mapped as the target crop and y = 1 if its reference label is the
    target crop, each metric is a ratio M = sum_h W_h mean_h(num) / sum_h W_h mean_h(den):

        metric   num      den      residual e = num - M den
        UA       x y      x        x (y - UA)
        PA       x y      y        y (x - PA)
        F1       2 x y    x + y    2 x y - F1 (x + y)

    V(M) = sum_h W_h^2 s2_h(e) / n_h / D^2,  D = sum_h W_h mean_h(den).

    stratum, mapped, reference : arrays over the sample units (mapped / reference 0/1)
    W_h : {stratum: weight}; strata without sample units are skipped.
    Returns {"UA", "UA_SE", "PA", "PA_SE", "F1", "F1_SE"}.
    """
    h = np.asarray(stratum)
    x = np.asarray(mapped, dtype=float)
    y = np.asarray(reference, dtype=float)
    strata = [g for g in W_h if np.any(h == g)]

    def metric(num, den):
        D = sum(W_h[g] * den[h == g].mean() for g in strata)
        M = sum(W_h[g] * num[h == g].mean() for g in strata) / D
        e = num - M * den
        V = sum(W_h[g] ** 2 * np.var(e[h == g], ddof=1) / np.sum(h == g) for g in strata) / D ** 2
        return float(M), float(np.sqrt(V))

    out = {}
    for key, (num, den) in {"UA": (x * y, x), "PA": (x * y, y), "F1": (2 * x * y, x + y)}.items():
        out[key], out[f"{key}_SE"] = metric(num, den)
    return out


def accuracy_census(n_mapped_and_reference, n_mapped, n_reference):
    """UA, PA and F1 from census pixel counts (reference label known for every pixel)."""
    ua = n_mapped_and_reference / n_mapped
    pa = n_mapped_and_reference / n_reference
    return {"UA": float(ua), "PA": float(pa), "F1": float(2 * ua * pa / (ua + pa))}

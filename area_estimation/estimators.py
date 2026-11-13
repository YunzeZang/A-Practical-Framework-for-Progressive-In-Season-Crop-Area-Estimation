"""Area estimators written on cell counts.

Notation
--------
h          design stratum (class of the initial map)
k          class of the updated map
W_h        design-stratum weight (share of the study area)
w_{k|h}    share of stratum h that the updated map labels k (census)
W_k        share of the study area that the updated map labels k (census)
n_h        sample size in stratum h
n_hk       sample units in the cross-classified substratum (h, k)
m_hk       sample units in (h, k) whose reference label is the target crop

Every function accepts arrays with an optional leading replicate axis (shape (..., H) or
(..., H, K)), so the same code evaluates one sample or many Monte Carlo replicates at once.
Strata with n_h = 0 add nothing to the estimate or to the variance.  The finite population
correction is omitted.

    stratified       stratified estimator (initial map)
    within_stratum   within-stratum poststratified estimator; no estimate (NaN) when a
                     substratum with w_{k|h} > 0 has n_hk < 2
    combined_ratio   combined ratio poststratified estimator with the adjusted
                     linearization variance

Each function returns the estimated target-crop proportion and its variance; multiply by the
total area to obtain the area and its standard error.
"""
from __future__ import annotations

import numpy as np


def _safe_div(a, b):
    """Element-wise a / b, 0 where b == 0."""
    a, b = np.broadcast_arrays(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    out = np.zeros_like(a)
    np.divide(a, b, out=out, where=b != 0)
    return out


def _sample_variance(n, m):
    """Proportion p = m / n and binary sample variance s2 = n / (n - 1) p (1 - p) (0 if n < 2)."""
    p = _safe_div(m, n)
    s2 = np.where(n > 1, _safe_div(n, n - 1) * p * (1 - p), 0.0)
    return p, s2


def stratified(n_h, m_h, W):
    """p = sum_h W_h p_h,  V = sum_h W_h^2 s2_h / n_h."""
    n, m, W = (np.asarray(x, dtype=float) for x in (n_h, m_h, W))
    p_h, s2_h = _sample_variance(n, m)
    p = (W * p_h).sum(axis=-1)
    var = _safe_div(W ** 2 * s2_h, n).sum(axis=-1)
    return p, var


def within_stratum(n_hk, m_hk, W, w_cond, n_h):
    """p = sum_h W_h sum_k w_{k|h} p_hk,  V = sum_h W_h^2 sum_k w_{k|h}^2 s2_hk / n_hk.

    The variance needs s2_hk in every substratum, so a sample in which a substratum with
    w_{k|h} > 0 holds fewer than two units gives no estimate: p and V are NaN.
    Returns (p, var, failed).
    """
    n, m = np.asarray(n_hk, dtype=float), np.asarray(m_hk, dtype=float)
    W, w, n_h = (np.asarray(x, dtype=float) for x in (W, w_cond, n_h))
    p_hk, s2_hk = _sample_variance(n, m)
    p = (W * (w * p_hk).sum(axis=-1)).sum(axis=-1)
    var = (W ** 2 * (w ** 2 * np.where(n > 1, s2_hk / np.maximum(n, 1), 0.0)).sum(axis=-1)).sum(axis=-1)
    active = (w > 0) & (n_h > 0)[:, None]
    failed = (active & (n < 2)).any(axis=(-2, -1))
    return np.where(failed, np.nan, p), np.where(failed, np.nan, var), failed


def combined_ratio(n_hk, m_hk, W, W_k, n_h):
    """Combined ratio poststratified estimator and its adjusted linearization variance.

        d_h  = W_h / n_h                              weight of a unit in stratum h
        R_k  = sum_h d_h m_hk / sum_h d_h n_hk        target share of class k, pooled over h
        p    = sum_k W_k R_k
        r    = (W_k / W_k^sample) (y - R_k)           adjusted residual, W_k^sample = sum_h d_h n_hk
        V    = sum_h W_h^2 s2_h(r) / n_h

    y is binary, so within a stratum all units with the same (k, y) share one residual and
    s2_h(r) is computed from the cell counts.  Classes without sample units get R_k = 0.
    """
    n, m = np.asarray(n_hk, dtype=float), np.asarray(m_hk, dtype=float)
    W, W_k, n_h = (np.asarray(x, dtype=float) for x in (W, W_k, n_h))
    d_h = _safe_div(W, n_h)
    W_k_sample = (d_h[:, None] * n).sum(axis=-2)
    R_k = _safe_div((d_h[:, None] * m).sum(axis=-2), W_k_sample)
    p = (W_k * R_k).sum(axis=-1)
    g = _safe_div(W_k, W_k_sample)
    r1, r0 = g * (1 - R_k), -g * R_k
    total = (m * r1[..., None, :] + (n - m) * r0[..., None, :]).sum(axis=-1)
    mean_r = _safe_div(total, n_h)
    ss = (m * (r1[..., None, :] - mean_r[..., None]) ** 2
          + (n - m) * (r0[..., None, :] - mean_r[..., None]) ** 2).sum(axis=-1)
    s2_r = np.where(n_h > 1, ss / np.maximum(n_h - 1, 1), 0.0)
    var = _safe_div(W ** 2 * s2_r, n_h).sum(axis=-1)
    return p, var


def cell_counts(h, k, y, strata, classes):
    """(H, K) sample counts n_hk and target counts m_hk from per-unit labels."""
    h, k, y = (np.asarray(x) for x in (h, k, y))
    n = np.array([[np.sum((h == a) & (k == b)) for b in classes] for a in strata], dtype=float)
    m = np.array([[np.sum((h == a) & (k == b) & (y == 1)) for b in classes] for a in strata], dtype=float)
    return n, m

"""Monte Carlo evaluation of the estimators on a fully known binary population.

Every pixel has a design stratum h (class of the initial map, 0 or 1), a class k_t in each of
the T updated maps (0 or 1) and a reference label y (0 or 1).  A pixel is summarised by its code

    code = y + 2 k_1 + 4 k_2 + ... + 2^T k_T

and the population by the census table counts[h, code].  A stratified sample is described by a
table of the same shape (area_estimation.sampling), from which the cell counts n_hk and m_hk of
every updated map follow.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np

from area_estimation.accuracy import accuracy_census
from area_estimation.estimators import combined_ratio, stratified, within_stratum

Z95 = NormalDist().inv_cdf(0.975)


def pixel_codes(reference, updates):
    """Code of every pixel from its reference label and its classes in the updated maps."""
    code = np.asarray(reference).astype(np.uint16)
    for t, k in enumerate(updates, start=1):
        code |= np.asarray(k).astype(np.uint16) << t
    return code


def count_table(strata, codes, n_updates):
    """Census table counts[h, code] of a population with strata 0 and 1."""
    n_codes = 2 ** (n_updates + 1)
    return np.stack([np.bincount(codes[strata == s], minlength=n_codes) for s in (0, 1)]).astype(np.int64)


def n_updates_of(counts):
    return int(np.log2(np.asarray(counts).shape[-1])) - 1


def code_masks(n_updates):
    """Boolean masks over the codes: reference label y, and class k_t of every updated map."""
    idx = np.arange(2 ** (n_updates + 1))
    return (idx & 1).astype(bool), {t: ((idx >> t) & 1).astype(bool) for t in range(1, n_updates + 1)}


def census(counts, pixel_ha):
    """Stratum weights, target share of each stratum and, per updated map, the census weights."""
    T = n_updates_of(counts)
    y, k = code_masks(T)
    N_h = counts.sum(axis=1).astype(float)
    N = N_h.sum()
    cells = {}
    for t in range(1, T + 1):
        N_hk = np.stack([counts[:, ~k[t]].sum(axis=1), counts[:, k[t]].sum(axis=1)], axis=1).astype(float)
        cells[t] = dict(N_hk=N_hk, w_cond=N_hk / N_hk.sum(axis=1, keepdims=True), W_k=N_hk.sum(axis=0) / N)
    return dict(counts=counts, n_updates=T, W=N_h / N, p=counts[:, y].sum(axis=1) / N_h, N_h=N_h,
                A_tot=N * pixel_ha, true_area=counts[:, y].sum() * pixel_ha, cells=cells)


def map_accuracy(counts):
    """UA, PA and F1 of the target class of the initial map (t = 0) and of every updated map,
    evaluated on all pixels."""
    T = n_updates_of(counts)
    y, k = code_masks(T)
    c = counts.astype(float)
    n_ref = c[:, y].sum()
    rows = [dict(t=0, **accuracy_census(c[1, y].sum(), c[1].sum(), n_ref))]
    for t in range(1, T + 1):
        rows.append(dict(t=t, **accuracy_census(c[:, k[t] & y].sum(), c[:, k[t]].sum(), n_ref)))
    return rows


def cell_counts(tables, t):
    """Sample counts n_hk and target counts m_hk (replicate, h, k) of updated map t."""
    y, k = code_masks(n_updates_of(tables))
    n_hk = np.stack([tables[:, :, ~k[t]].sum(axis=2), tables[:, :, k[t]].sum(axis=2)], axis=2).astype(float)
    m_hk = np.stack([tables[:, :, ~k[t] & y].sum(axis=2), tables[:, :, k[t] & y].sum(axis=2)], axis=2).astype(float)
    return n_hk, m_hk


def estimate_replicates(tables, n_h, cen):
    """Area estimates (ha) and standard errors of every replicate.

    Returns {(t, estimator): (area, se)} for t = 0 (stratified) and t >= 1 (within_stratum,
    combined_ratio), and {t: (failed, n_hk)} for t >= 1.
    """
    n_h = np.asarray(n_h, dtype=float)
    y, _ = code_masks(cen["n_updates"])
    W, A = cen["W"], cen["A_tot"]
    m_h = tables[:, :, y].sum(axis=2).astype(float)
    p, var = stratified(n_h, m_h, W)
    est = {(0, "stratified"): (A * p, A * np.sqrt(var))}
    cells = {}
    for t in range(1, cen["n_updates"] + 1):
        c = cen["cells"][t]
        n_hk, m_hk = cell_counts(tables, t)
        p_ws, var_ws, failed = within_stratum(n_hk, m_hk, W, c["w_cond"], n_h)
        p_cr, var_cr = combined_ratio(n_hk, m_hk, W, c["W_k"], n_h)
        est[(t, "within_stratum")] = (A * p_ws, A * np.sqrt(var_ws))
        est[(t, "combined_ratio")] = (A * p_cr, A * np.sqrt(var_cr))
        cells[t] = (failed, n_hk)
    return est, cells


def summarise(estimate, se, truth):
    """Performance of one estimator over the replicates in which it gives an estimate: mean of
    the estimates, empirical SD, root mean square of the estimated SEs and coverage of the
    nominal 95 % confidence interval."""
    used = np.isfinite(estimate)
    out = dict(n_used=int(used.sum()))
    if used.sum() >= 2:
        a, s = estimate[used], se[used]
        out.update(mean=float(a.mean()), sd=float(a.std(ddof=1)), rms_se=float(np.sqrt((s ** 2).mean())),
                   coverage95=float((np.abs(a - truth) <= Z95 * s).mean()))
    return out

"""Numerical example of Appendix B: one replicate of the 5 % design (n = 237), initial map of
20 June (t = 0) and updated map of 30 June (t = 1).

Every intermediate quantity of the stratified estimator and of the within-stratum and combined
ratio poststratified estimators is computed from the sample units, following the equations of
Section 2.3, and the results are checked against area_estimation.estimators.

Reads   results/area_estimation/population_counts.npz   (01_build_population.py)
        data/sample_points/conf95_d05/rep_0075.csv       (04_export_sample_points.py)
Writes  results/numerical_example/appendixB_numbers.json
            design     Table B1 (N_h, W_h, n_h, m_h)
            t0         stratified estimator
            t1_cross   Table B2 (N_hk, W_hk, n_hk, m_hk, p_hk)
            t1_ws      within-stratum poststratified estimator
            t1_cr      combined ratio poststratified estimator (Tables B3 and B4)

Usage   python Story_County/experiments/08_numerical_example.py [--replicate 75]
"""
import argparse
import json

import numpy as np
import pandas as pd

import common as C
from area_estimation.estimators import cell_counts, combined_ratio, stratified, within_stratum

Z = 1.959964
DESIGN = C.design_name(0.05)
DATE_T1 = C.UPDATE_DATES[0]


def estimate(p, V, A):
    half = Z * V ** 0.5
    return dict(p=p, V=V, SE=V ** 0.5, half_width=half, area_ha=A * p, area_half_width_ha=A * half)


def main(replicate):
    cen = C.load_census()
    N_hk = cen["cells"][1]["N_hk"]
    N_h, N, A, W = cen["N_h"], cen["N_h"].sum(), cen["A_tot"], cen["W"]
    s = pd.read_csv(C.SAMPLE_DIR / DESIGN / f"rep_{replicate:04d}.csv")
    h, k, y = s["h"].to_numpy(), s[f"k_{DATE_T1}"].to_numpy(), s["reference"].to_numpy()
    n_h = np.array([(h == a).sum() for a in (0, 1)])
    m_h = np.array([y[h == a].sum() for a in (0, 1)])
    out = dict(replicate=replicate, design=DESIGN, A_ha=A, N=int(N), true_proportion=cen["true_area"] / A,
               design_strata=dict(N_h=N_h.tolist(), W_h=W.tolist(), n_h=n_h.tolist(), m_h=m_h.tolist()))

    # t = 0: stratified estimator
    p_h = m_h / n_h
    V_h = W ** 2 * p_h * (1 - p_h) / (n_h - 1)
    p0, V0 = (W * p_h).sum(), V_h.sum()
    out["t0"] = dict(p_h=p_h.tolist(), W_h_p_h=(W * p_h).tolist(), V_h=V_h.tolist(), **estimate(p0, V0, A))

    # t = 1: cross-classified substrata and the within-stratum poststratified estimator
    n_hk, m_hk = (x.astype(int) for x in cell_counts(h, k, y, (0, 1), (0, 1)))
    if (n_hk < 2).any():
        raise SystemExit(f"replicate {replicate}: a substratum holds fewer than two sample units "
                         f"(n_hk = {n_hk.tolist()}); the within-stratum estimator cannot be computed")
    W_hk = N_hk / N
    p_hk = m_hk / n_hk
    V_hk = W_hk ** 2 * p_hk * (1 - p_hk) / (n_hk - 1)
    p_ws, V_ws = (W_hk * p_hk).sum(), V_hk.sum()
    out["t1_cross"] = dict(N_hk=N_hk.tolist(), W_hk=W_hk.tolist(), n_hk=n_hk.tolist(), m_hk=m_hk.tolist(),
                           p_hk=p_hk.tolist())
    out["t1_ws"] = dict(W_hk_p_hk=(W_hk * p_hk).tolist(), V_hk=V_hk.tolist(), **estimate(p_ws, V_ws, A))

    # t = 1: combined ratio poststratified estimator
    N_k = N_hk.sum(0)
    W_k = N_k / N
    x_bar, y_bar = n_hk / n_h[:, None], m_hk / n_h[:, None]
    R_k = (N_h[:, None] * y_bar).sum(0) / (N_h[:, None] * x_bar).sum(0)
    p_cr = (W_k * R_k).sum()
    W_k_sample = (W[:, None] * x_bar).sum(0)
    g_k = W_k / W_k_sample
    r = g_k[k] * (y - R_k[k])
    s2_h = np.array([r[h == a].var(ddof=1) for a in (0, 1)])
    V_h = W ** 2 * s2_h / n_h
    V_cr = V_h.sum()
    residuals = {f"k{b}_y{v}": float(g_k[b] * (v - R_k[b])) for b in (0, 1) for v in (1, 0)}
    out["t1_cr"] = dict(N_k=N_k.tolist(), W_k=W_k.tolist(), x_bar_hk=x_bar.tolist(), y_bar_hk=y_bar.tolist(),
                        R_k=R_k.tolist(), W_k_sample=W_k_sample.tolist(), g_k=g_k.tolist(), residuals=residuals,
                        r_bar_h=[float(r[h == a].mean()) for a in (0, 1)], s2_h=s2_h.tolist(), V_h=V_h.tolist(),
                        **estimate(p_cr, V_cr, A))

    checks = [(stratified(n_h, m_h, W), (p0, V0)),
              (within_stratum(n_hk, m_hk, W, N_hk / N_h[:, None], n_h)[:2], (p_ws, V_ws)),
              (combined_ratio(n_hk, m_hk, W, W_k, n_h), (p_cr, V_cr))]
    for (pa, va), (pb, vb) in checks:
        assert np.isclose(pa, pb) and np.isclose(va, vb)

    C.EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    (C.EXAMPLE_DIR / "appendixB_numbers.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for key in ("t0", "t1_ws", "t1_cr"):
        e = out[key]
        print(f"{key}: p = {e['p']:.4f}, V = {e['V']:.3e}, area = {e['area_ha']:,.0f} +/- {e['area_half_width_ha']:,.0f} ha")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replicate", type=int, default=C.CFG["example_replicate"])
    main(ap.parse_args().replicate)

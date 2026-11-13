"""Progressive in-season estimation of the rapeseed area in Yanting County (Table 4, Fig. 9).

t = 0     stratified estimator on the design strata of the initial map (16 December 2024)
t = 1-3   within-stratum and combined ratio poststratified estimators on the updated maps of
          16 January, 25 March and 19 April 2025, with the same sample
Every date: UA, PA and F1-score of the rapeseed class of the map with standard errors, and the
area obtained by pixel counting.

Design strata h and updated-map classes k: 0 = uncultivated, 1 = non-rapeseed, 2 = rapeseed.
The uncultivated stratum has no sample units and adds nothing to the estimates or to the
variances.  The class of a sample unit in an updated map is the class of its pixel.

Reads   data/initial_map, data/updated_maps, data/sample_points
Writes  results/area_estimation/progressive_estimates.csv   one row per date:
            area_cr, se_cr, prop_se_cr   combined ratio estimator: area and SE (ha), SE of the proportion
            area_ws, se_ws, prop_se_ws   within-stratum estimator (both are the stratified estimator at t = 0)
            UA, PA, F1 and their SEs, pixel_count_ha, A_tot_ha, N_h, n_hk
        results/area_estimation/sample_unit_classes.csv     h, y and k at every date of each sample unit

Usage   python Yanting_County/experiments/02_progressive_estimation.py
"""
import geopandas as gpd
import numpy as np
import pandas as pd

import common as C
from area_estimation.accuracy import accuracy_stratified
from area_estimation.estimators import cell_counts, combined_ratio, stratified, within_stratum
from area_estimation.raster_utils import class_at_points, crosstab, read_class_raster

STRATA = (0, 1, 2)          # design strata h
CLASSES = (1, 2)            # updated-map classes k of the cropland strata
TARGET = 2                  # rapeseed


def main():
    units = gpd.read_file(C.SAMPLE_UNITS)
    h, y = units["h"].to_numpy(dtype=int), units["y"].to_numpy(dtype=int)
    table = pd.DataFrame({"id": units["id"], "h": h, "y": y})
    initial = C.map_path(C.INITIAL_DATE)
    pixel_ha = read_class_raster(initial)[3] / 1e4
    if not np.array_equal(class_at_points(initial, units), h):
        raise RuntimeError("the strata of the sample units differ from the initial map")

    # t = 0: stratified estimator
    N_h = crosstab(initial)[list(STRATA), 0].astype(float)
    A = N_h.sum() * pixel_ha
    W = N_h / N_h.sum()
    n_h = np.array([np.sum(h == s) for s in STRATA], dtype=float)
    m_h = np.array([np.sum((h == s) & (y == 1)) for s in STRATA], dtype=float)
    p, var = stratified(n_h, m_h, W)
    se = float(np.sqrt(var))
    acc = accuracy_stratified(h, h == TARGET, y == 1, dict(zip(STRATA, W)))
    common = dict(A_tot_ha=A, N_0=N_h[0], N_1=N_h[1], N_2=N_h[2])
    rows = [dict(t=0, date=C.INITIAL_DATE, label=C.date_label(C.INITIAL_DATE), area_cr=A * p, se_cr=A * se,
                 prop_se_cr=se, area_ws=A * p, se_ws=A * se, prop_se_ws=se, **acc,
                 pixel_count_ha=N_h[TARGET] * pixel_ha, **common)]
    print(f"county area A = {A:,.1f} ha;  W_h = {np.round(W, 4)};  n_h = {n_h.astype(int)};  m_h = {m_h.astype(int)}")
    print(f"t = 0  stratified {A * p:,.0f} +/- {1.96 * A * se:,.0f} ha (95 % CI)   F1 {acc['F1']:.3f}")

    # t >= 1: poststratified estimators on the updated maps
    for t, date in enumerate(C.UPDATE_DATES, start=1):
        updated = C.map_path(date)
        k = class_at_points(updated, units)
        if not np.isin(k, CLASSES).all():
            raise RuntimeError(f"{date}: sample units outside the cropland classes of the updated map")
        table[f"k_{date}"] = k
        N_hk = crosstab(initial, updated)[np.ix_(STRATA, STRATA)].astype(float)
        w_cond = np.zeros((len(STRATA), len(CLASSES)))
        w_cond[1:] = N_hk[1:, 1:] / N_hk[1:, 1:].sum(axis=1, keepdims=True)
        W_k = N_hk.sum(axis=0) / N_hk.sum()
        n_hk, m_hk = cell_counts(h, k, y, STRATA, CLASSES)
        p_ws, var_ws, _ = within_stratum(n_hk, m_hk, W, w_cond, n_h)
        n_all, m_all = cell_counts(h, k, y, STRATA, STRATA)
        p_cr, var_cr = combined_ratio(n_all, m_all, W, W_k, n_h)
        acc = accuracy_stratified(h, k == TARGET, y == 1, dict(zip(STRATA, W)))
        rows.append(dict(t=t, date=date, label=C.date_label(date),
                         area_cr=A * p_cr, se_cr=A * np.sqrt(var_cr), prop_se_cr=np.sqrt(var_cr),
                         area_ws=A * p_ws, se_ws=A * np.sqrt(var_ws), prop_se_ws=np.sqrt(var_ws), **acc,
                         pixel_count_ha=N_hk[:, TARGET].sum() * pixel_ha, **common,
                         n_11=int(n_hk[1, 0]), n_12=int(n_hk[1, 1]), n_21=int(n_hk[2, 0]), n_22=int(n_hk[2, 1])))
        print(f"t = {t}  combined ratio {A * p_cr:,.0f} +/- {1.96 * A * np.sqrt(var_cr):,.0f} ha | within-stratum "
              f"{A * p_ws:,.0f} +/- {1.96 * A * np.sqrt(var_ws):,.0f} ha (95 % CI)   F1 {acc['F1']:.3f}")

    C.AREA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(C.AREA_DIR / "progressive_estimates.csv", index=False)
    table.to_csv(C.AREA_DIR / "sample_unit_classes.csv", index=False)
    print("wrote progressive_estimates.csv and sample_unit_classes.csv")


if __name__ == "__main__":
    main()

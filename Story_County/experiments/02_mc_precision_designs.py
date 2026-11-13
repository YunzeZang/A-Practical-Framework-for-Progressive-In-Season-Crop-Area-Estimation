"""Monte Carlo simulation of the in-season area estimation under five sample sizes (Fig. 5).

Designs      Neyman allocation for a 95 % confidence interval half-width of 1 % ... 5 %
             (n = 5,925 ... 237); the anticipated corn proportions of the strata are the
             CDL-derived proportions
Replicates   M = 10,000 stratified random samples without replacement per design
Estimators   stratified estimator on the initial map (t = 0); within-stratum and combined ratio
             poststratified estimators on every updated map (t >= 1)

Reads   results/area_estimation/population_counts.npz   (01_build_population.py)
Writes  results/area_estimation/mc_precision_replicates.npz
            area, se   (design, date, estimator, replicate), ha; estimator 0 = within-stratum
                       (stratified at t = 0), 1 = combined ratio (NaN at t = 0); NaN where the
                       within-stratum estimator cannot be computed
            flips      (design, date, substratum, replicate): sample units in (0, 1) and (1, 0)
            margin, n, n_h, dates, true_area, county_area
        results/area_estimation/mc_precision_summary.csv
            mean, empirical SD, RMS of the estimated SEs, 95 % coverage and failure rate

Usage   python Story_County/experiments/02_mc_precision_designs.py
"""
from statistics import NormalDist

import numpy as np
import pandas as pd

import common as C
from area_estimation.sample_design import required_sample_size


def main():
    cen = C.load_census()
    W, p = cen["W"], cen["p"]
    S = np.sqrt(p * (1 - p))
    z = NormalDist().inv_cdf(1 - (1 - C.CFG["confidence"]) / 2)
    M = C.CFG["n_replicates"]
    area, se, flips, n, n_h, rows = [], [], [], [], [], []
    for margin in C.MARGINS:
        design = required_sample_size(W, S, z, margin, "neyman")
        a, s, f = C.simulate_design(cen, design["n_h"], C.design_seed(margin), M)
        area.append(a); se.append(s); flips.append(f); n.append(design["n"]); n_h.append(design["n_h"])
        key = dict(design=C.design_name(margin), margin=margin, n=design["n"],
                   n_h0=int(design["n_h"][0]), n_h1=int(design["n_h"][1]))
        rows += C.performance_table(a, s, f, cen, key)
        print(f"half-width {margin:.0%}: n = {design['n']:,} ({design['n_h'][0]} / {design['n_h'][1]})")
    np.savez_compressed(C.AREA_DIR / "mc_precision_replicates.npz", area=np.array(area), se=np.array(se),
                        flips=np.array(flips), margin=np.array(C.MARGINS), n=np.array(n), n_h=np.array(n_h),
                        dates=np.array(C.DATES), true_area=cen["true_area"], county_area=cen["A_tot"])
    pd.DataFrame(rows).to_csv(C.AREA_DIR / "mc_precision_summary.csv", index=False)
    print("wrote mc_precision_replicates.npz and mc_precision_summary.csv")


if __name__ == "__main__":
    main()

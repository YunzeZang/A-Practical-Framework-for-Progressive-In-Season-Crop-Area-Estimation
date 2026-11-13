"""Monte Carlo simulation of the sensitivity to the sample allocation (Appendix C, Fig. C1).

The total sample size is fixed at that of the 5 % design (n = 237) and divided among the two
strata by Neyman, proportional and equal allocation; M = 10,000 replicates per allocation.  The
Neyman allocation is the 5 % design of 02_mc_precision_designs.py (same replicates).

Reads   results/area_estimation/population_counts.npz   (01_build_population.py)
Writes  results/area_estimation/mc_allocation_replicates.npz
            area, se, flips   (allocation, date, estimator or substratum, replicate), as in
                              mc_precision_replicates.npz
            allocation, n_h, dates, true_area, county_area
        results/area_estimation/mc_allocation_summary.csv

Usage   python Story_County/experiments/03_mc_sample_allocation.py
"""
from statistics import NormalDist

import numpy as np
import pandas as pd

import common as C
from area_estimation.sample_design import allocation_shares, largest_remainder, required_sample_size


def main():
    cen = C.load_census()
    W, p = cen["W"], cen["p"]
    S = np.sqrt(p * (1 - p))
    z = NormalDist().inv_cdf(1 - (1 - C.CFG["confidence"]) / 2)
    margin = C.CFG["allocation_margin"]
    n = required_sample_size(W, S, z, margin, "neyman")["n"]
    M = C.CFG["n_replicates"]
    area, se, flips, n_hs, rows = [], [], [], [], []
    for allocation in C.CFG["allocations"]:
        n_h = largest_remainder(n, allocation_shares(W, S, allocation))
        a, s, f = C.simulate_design(cen, n_h, C.design_seed(margin), M)
        area.append(a); se.append(s); flips.append(f); n_hs.append(n_h)
        rows += C.performance_table(a, s, f, cen, dict(allocation=allocation, n=n, n_h0=int(n_h[0]), n_h1=int(n_h[1])))
        print(f"{allocation}: n = {n} ({n_h[0]} / {n_h[1]})")
    np.savez_compressed(C.AREA_DIR / "mc_allocation_replicates.npz", area=np.array(area), se=np.array(se),
                        flips=np.array(flips), allocation=np.array(C.CFG["allocations"]), n_h=np.array(n_hs),
                        dates=np.array(C.DATES), true_area=cen["true_area"], county_area=cen["A_tot"])
    pd.DataFrame(rows).to_csv(C.AREA_DIR / "mc_allocation_summary.csv", index=False)
    print("wrote mc_allocation_replicates.npz and mc_allocation_summary.csv")


if __name__ == "__main__":
    main()

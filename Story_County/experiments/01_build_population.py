"""Population of the simulation: census count table and accuracy of the in-season maps.

The population is every pixel of the county (N = 14,861,881 pixels of 10 m).  Each pixel has a
design stratum h (initial map of 20 June), a class k in every updated map and a reference label
y (2025 CDL), all coded 1 = corn and 0 = non-corn.  The pixels are summarised by the count
table of area_estimation.monte_carlo, which is all that the Monte Carlo simulation needs.

Reads   data/initial_map, data/updated_maps, data/reference
Writes  results/area_estimation/population_counts.npz   counts (stratum x pixel code), pixel_ha, N, dates
        results/area_estimation/map_accuracy.csv        UA, PA and F1 of every map, all pixels (Fig. 5a)

Usage   python Story_County/experiments/01_build_population.py
"""
import numpy as np
import pandas as pd

import common as C
from area_estimation import monte_carlo as mc


def main():
    pop = C.read_population()
    counts = mc.count_table(pop["strata"], pop["codes"], len(C.UPDATE_DATES))
    C.AREA_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(C.AREA_DIR / "population_counts.npz", counts=counts, pixel_ha=pop["pixel_ha"],
                        N=int(counts.sum()), dates=np.array(C.DATES))
    acc = pd.DataFrame(mc.map_accuracy(counts))
    acc.insert(1, "date", C.DATES)
    acc.to_csv(C.AREA_DIR / "map_accuracy.csv", index=False)
    cen = mc.census(counts, pop["pixel_ha"])
    print(f"N = {int(counts.sum()):,} pixels, A = {cen['A_tot']:,.1f} ha, corn (CDL) = {cen['true_area']:,.1f} ha")
    print(f"W_h = {cen['W'].round(6)}, corn proportion within strata = {cen['p'].round(6)}")
    print(acc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

"""Sample points of 100 replicates of every precision design, for the response-design simulation.

For each design, 100 replicates are chosen at random among replicates 1 ... 1,000 of
02_mc_precision_designs.py and the pixels of each of them are drawn: for every stratum and
pixel code the replicate holds a number of sample units (its count table), and that many
pixels are drawn without replacement among the pixels of that stratum and code.

Reads   data/initial_map, data/updated_maps, data/reference
Writes  data/sample_points/conf95_dXX/rep_NNNN.csv   one row per sample unit:
            id          number of the sample unit within the replicate
            h           design stratum (initial map): 1 = corn, 0 = non-corn
            x, y        pixel centre (EPSG:5070)
            reference   reference label y (2025 CDL): 1 = corn, 0 = non-corn
            k_YYYYMMDD  class in the updated map of that date: 1 = corn, 0 = non-corn
        data/sample_points/sample_points_index.csv    design, sample size and replicate of every file

Usage   python Story_County/experiments/04_export_sample_points.py
"""
from statistics import NormalDist

import numpy as np
import pandas as pd

import common as C
from area_estimation import monte_carlo as mc
from area_estimation.sample_design import required_sample_size
from area_estimation.sampling import draw_replicates, draw_sample_units, sort_population


def main():
    pop = C.read_population()
    n_updates = len(C.UPDATE_DATES)
    counts = mc.count_table(pop["strata"], pop["codes"], n_updates)
    cen = mc.census(counts, pop["pixel_ha"])
    members, offsets = sort_population(pop["strata"], pop["codes"], counts.shape[1])
    W, p = cen["W"], cen["p"]
    S = np.sqrt(p * (1 - p))
    z = NormalDist().inv_cdf(1 - (1 - C.CFG["confidence"]) / 2)
    n_base, n_export = C.CFG["n_base"], C.CFG["n_export"]
    a, b, c, d, e, f = tuple(pop["transform"])[:6]
    index = []
    for margin in C.MARGINS:
        design = required_sample_size(W, S, z, margin, "neyman")
        seed = C.design_seed(margin)
        tables = draw_replicates(counts, design["n_h"], seed, n_base, n_base)
        replicates = np.sort(np.random.default_rng([seed, 1]).choice(n_base, size=n_export, replace=False))
        rng = np.random.default_rng([seed, 2])
        folder = C.SAMPLE_DIR / C.design_name(margin)
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("rep_*.csv"):
            old.unlink()
        for r in replicates:
            pos, strata = draw_sample_units(tables[r], members, offsets, rng)
            rows, cols = np.divmod(pop["pix"][pos], pop["shape"][1])
            code = pop["codes"][pos].astype(np.int64)
            table = pd.DataFrame({"id": np.arange(1, pos.size + 1), "h": strata,
                                  "x": a * (cols + 0.5) + b * (rows + 0.5) + c,
                                  "y": d * (cols + 0.5) + e * (rows + 0.5) + f,
                                  "reference": code & 1})
            for t, date in enumerate(C.UPDATE_DATES, start=1):
                table[f"k_{date}"] = (code >> t) & 1
            table.to_csv(folder / f"rep_{r + 1:04d}.csv", index=False, float_format="%.2f")
            index.append(dict(design=C.design_name(margin), margin=margin, n=design["n"],
                              n_h0=int(design["n_h"][0]), n_h1=int(design["n_h"][1]), replicate=int(r + 1)))
        print(f"{C.design_name(margin)}: {n_export} replicates, n = {design['n']:,}")
    pd.DataFrame(index).to_csv(C.SAMPLE_DIR / "sample_points_index.csv", index=False)


if __name__ == "__main__":
    main()

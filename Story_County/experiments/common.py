"""Paths, configuration and data access shared by the Story County scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CFG = yaml.safe_load((SITE / "config.yaml").read_text(encoding="utf-8"))
DATA, RESULTS = SITE / "data", SITE / "results"
AREA_DIR = RESULTS / "area_estimation"
ROUTE_DIR = RESULTS / "response_design"
EXAMPLE_DIR = RESULTS / "numerical_example"
FIGURE_DIR = RESULTS / "figures"
SAMPLE_DIR = DATA / "sample_points"
ROADS = DATA / "roads" / "osm_roads_Story.shp"
REFERENCE = DATA / "reference" / "CDL_Story_2025.tif"

INITIAL_DATE = CFG["initial_date"]
UPDATE_DATES = list(CFG["update_dates"])
DATES = [INITIAL_DATE] + UPDATE_DATES
MARGINS = sorted(CFG["margins"])
MAP_CRS = "EPSG:5070"
NODATA = 255


def map_path(date: str) -> Path:
    folder = "initial_map" if date == INITIAL_DATE else "updated_maps"
    return DATA / folder / f"corn_map_Story_{date}.tif"


def design_name(margin: float) -> str:
    """Folder name of a precision design, e.g. conf95_d05 for a half-width of 5 %."""
    return f"conf{CFG['confidence'] * 100:.0f}_d{margin * 100:02.0f}"


DESIGNS = [design_name(m) for m in MARGINS]


def design_seed(margin: float) -> int:
    return CFG["seed"] + MARGINS.index(margin)


def load_census():
    """Census count table of the population and its summary (01_build_population.py)."""
    from area_estimation import monte_carlo as mc
    z = np.load(AREA_DIR / "population_counts.npz")
    return mc.census(z["counts"], float(z["pixel_ha"]))


def read_sample_points(design: str, replicate: int):
    """Sample units of one exported replicate as points in the map CRS."""
    import geopandas as gpd
    table = pd.read_csv(SAMPLE_DIR / design / f"rep_{replicate:04d}.csv", float_precision="round_trip")
    return gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(table["x"], table["y"]), crs=MAP_CRS)


def exported_replicates(design: str) -> list[int]:
    return sorted(int(f.stem.split("_")[1]) for f in (SAMPLE_DIR / design).glob("rep_*.csv"))


def read_population():
    """Stratum, pixel code and flat raster index of every pixel of the population."""
    from area_estimation import monte_carlo as mc
    from area_estimation.raster_utils import read_class_raster
    strata, transform, crs, pixel_m2 = read_class_raster(map_path(INITIAL_DATE))
    valid = strata != NODATA
    pix = np.flatnonzero(valid)
    reference = read_class_raster(REFERENCE)[0].ravel()[pix]
    updates = [read_class_raster(map_path(d))[0].ravel()[pix] for d in UPDATE_DATES]
    if any(np.any(a == NODATA) for a in [reference] + updates):
        raise ValueError("the maps do not cover the same population")
    codes = mc.pixel_codes(reference, updates)
    return dict(pix=pix, strata=strata.ravel()[pix], codes=codes, shape=strata.shape, transform=transform,
                pixel_ha=pixel_m2 / 1e4)


def simulate_design(cen, n_h, seed, n_rep):
    """Draw the replicate samples of one design and apply every estimator at every date.

    Returns area and se (date, estimator, replicate; ha) and flips (date, substratum, replicate):
    estimator 0 is the within-stratum poststratified estimator (the stratified estimator at
    t = 0) and estimator 1 the combined ratio poststratified estimator (NaN at t = 0); flips are
    the sample units in the substrata (0, 1) and (1, 0) of each updated map.
    """
    from area_estimation import monte_carlo as mc
    from area_estimation.sampling import draw_replicates
    tables = draw_replicates(cen["counts"], n_h, seed, n_rep, CFG["n_base"])
    est, cells = mc.estimate_replicates(tables, n_h, cen)
    T = cen["n_updates"] + 1
    area, se = np.full((T, 2, n_rep), np.nan), np.full((T, 2, n_rep), np.nan)
    flips = np.zeros((T, 2, n_rep), dtype=np.int32)
    for (t, name), (a, s) in est.items():
        e = 1 if name == "combined_ratio" else 0
        area[t, e], se[t, e] = a, s
    for t, (_, n_hk) in cells.items():
        flips[t, 0], flips[t, 1] = n_hk[:, 0, 1], n_hk[:, 1, 0]
    return area, se, flips


def performance_table(area, se, flips, cen, key):
    """Performance of the estimators of one design (proportions of the county area)."""
    from area_estimation import monte_carlo as mc
    A = cen["A_tot"]
    truth = cen["true_area"] / A
    rows = []
    for t, date in enumerate(DATES):
        for e, name in enumerate(("within_stratum", "combined_ratio")):
            if t == 0 and e == 1:
                continue
            row = dict(key, t=t, date=date, estimator="stratified" if t == 0 else name,
                       true_proportion=truth, **mc.summarise(area[t, e] / A, se[t, e] / A, truth))
            row["failure_rate"] = float(np.isnan(area[t, e]).mean())
            row["mean_n_01"], row["mean_n_10"] = float(flips[t, 0].mean()), float(flips[t, 1].mean())
            rows.append(row)
    return rows


# ----------------------------------------------------------------------------- response design
def routing_arguments(description: str):
    """Command-line options shared by the response-design scripts."""
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--designs", nargs="+", default=DESIGNS, help="designs to plan (default: all)")
    ap.add_argument("--replicates", nargs="+", type=int, default=None, help="replicates to plan (default: all exported)")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--overwrite", action="store_true", help="plan again the replicates that already have a plan")
    ap.add_argument("--gpkg", action="store_true", help="also write the plan layers to plan.gpkg")
    return ap.parse_args()


def routing_tasks(method: str, args) -> list[tuple]:
    """(design, replicate, output folder) of every plan to compute."""
    tasks = []
    for design in args.designs:
        for replicate in exported_replicates(design):
            out_dir = ROUTE_DIR / method / design / f"rep_{replicate:04d}"
            if args.replicates and replicate not in args.replicates:
                continue
            if (out_dir / "summary.json").exists() and not args.overwrite:
                continue
            tasks.append((design, replicate, out_dir))
    return tasks


def run_tasks(worker, tasks, workers: int, initializer=None) -> None:
    """Run worker(task) for every task and print one line per finished plan."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if not tasks:
        print("nothing to do: every requested plan exists (use --overwrite to plan again)")
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for i, future in enumerate(as_completed(futures), start=1):
            design, replicate, s = future.result()
            print(f"[{i}/{len(tasks)}] {design} rep_{replicate:04d}: {s['n_vehicle_stops']} stops, "
                  f"vehicle {s['vehicle_time_h']} h, UAV {s['uav_time_h']} h", flush=True)


def load_roads():
    """Road rules of the configuration and the roads of the study area in the working CRS (UTM)."""
    from vehicle_uav_routing import planner, road_network as rn
    rules = rn.RoadRules.from_config(CFG["routing"]["roads"])
    first = read_sample_points(DESIGNS[-1], exported_replicates(DESIGNS[-1])[0])
    return rules, rn.read_roads(ROADS, planner.utm_crs(first), rules)

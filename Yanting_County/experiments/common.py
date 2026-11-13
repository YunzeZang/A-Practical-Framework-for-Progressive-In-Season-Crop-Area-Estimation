"""Paths and configuration shared by the Yanting County scripts."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CFG = yaml.safe_load((SITE / "config.yaml").read_text(encoding="utf-8"))
DATA, RESULTS = SITE / "data", SITE / "results"
AREA_DIR = RESULTS / "area_estimation"
ROUTE_DIR = RESULTS / "response_design"
FIGURE_DIR = RESULTS / "figures"
SAMPLE_UNITS = DATA / "sample_points" / "sample_units_Yanting.shp"
ROADS = DATA / "roads" / "osm_roads_Yanting.shp"

INITIAL_DATE = CFG["initial_date"]
UPDATE_DATES = list(CFG["update_dates"])
DATES = [INITIAL_DATE] + UPDATE_DATES


def map_path(date: str) -> Path:
    folder = "initial_map" if date == INITIAL_DATE else "updated_maps"
    return DATA / folder / f"rapeseed_map_Yanting_{date}.tif"


def date_label(date: str) -> str:
    """Label of a map date in Fig. 9, e.g. "PS,2025/01/16"."""
    return f"{CFG['sensors'][date]},{date[:4]}/{date[4:6]}/{date[6:]}"

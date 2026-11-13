"""Park-and-launch sites (PLSs): candidates along the roads, location set covering problem (LSCP)
and assignment of the sample units to the selected sites."""
from __future__ import annotations

from typing import Iterable

import geopandas as gpd
import numpy as np
import pulp
from pulp import PULP_CBC_CMD
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiLineString, Point


def _line_parts(geom) -> Iterable[LineString]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, LineString):
        yield geom
    elif isinstance(geom, MultiLineString):
        yield from (part for part in geom.geoms if not part.is_empty)


def _deduplicate(points: list[Point], grid_m: float = 10.0) -> list[Point]:
    seen: set[tuple[int, int]] = set()
    result: list[Point] = []
    for point in points:
        key = (round(point.x / grid_m), round(point.y / grid_m))
        if key not in seen:
            seen.add(key)
            result.append(point)
    return result


def generate_candidates(roads: gpd.GeoDataFrame, spacing_m: float) -> gpd.GeoDataFrame:
    """Candidate PLSs every spacing_m along each road, plus the midpoint of every road 1-2 km long."""
    spacing = float(spacing_m)
    points: list[Point] = []
    for geom in roads.geometry:
        for part in _line_parts(geom):
            length = part.length
            distance = spacing
            while distance < length:
                points.append(part.interpolate(distance))
                distance += spacing
            if 1000.0 <= length < 2000.0:
                points.append(part.interpolate(0.5, normalized=True))
    points = _deduplicate(points)
    if not points:
        raise RuntimeError("no candidate park-and-launch site on the roads")
    return gpd.GeoDataFrame({"candidate_id": range(len(points))}, geometry=points, crs=roads.crs)


def _xy(frame: gpd.GeoDataFrame) -> np.ndarray:
    return np.column_stack([frame.geometry.x, frame.geometry.y])


def greedy_set_cover(candidates: gpd.GeoDataFrame, units: gpd.GeoDataFrame, radius_m: float) -> list[int]:
    """Greedy solution of the LSCP: repeatedly take the candidate covering most uncovered units."""
    tree = cKDTree(_xy(units))
    covers = [set(tree.query_ball_point(xy, radius_m)) for xy in _xy(candidates)]
    uncovered = set(range(len(units)))
    selected: list[int] = []
    while uncovered:
        gains = [len(cover & uncovered) for cover in covers]
        best = int(np.argmax(gains))
        if gains[best] == 0:
            raise RuntimeError("some sample units are farther than the coverage radius from every candidate")
        selected.append(best)
        uncovered -= covers[best]
    return selected


def solve_lscp_exact(candidates: gpd.GeoDataFrame, units: gpd.GeoDataFrame, radius_m: float,
                     time_limit_s: float) -> list[int] | None:
    """LSCP as a binary integer programme (PuLP / CBC): the fewest candidates such that every
    sample unit lies within radius_m of a selected candidate.  Returns None when the solver gives
    no covering solution within the time limit."""
    tree = cKDTree(_xy(units))
    covers = [tree.query_ball_point(xy, radius_m) for xy in _xy(candidates)]
    cover_by_unit: dict[int, list[int]] = {}
    for candidate_index, covered in enumerate(covers):
        for unit_index in covered:
            cover_by_unit.setdefault(int(unit_index), []).append(candidate_index)
    if len(cover_by_unit) < len(units):
        raise RuntimeError("some sample units are farther than the coverage radius from every candidate")

    problem = pulp.LpProblem("LSCP", pulp.LpMinimize)
    select = pulp.LpVariable.dicts("x", range(len(candidates)), cat="Binary")
    problem += pulp.lpSum(select[i] for i in range(len(candidates)))
    for unit_index in range(len(units)):
        problem += pulp.lpSum(select[i] for i in cover_by_unit[unit_index]) >= 1
    problem.solve(PULP_CBC_CMD(msg=False, timeLimit=float(time_limit_s)))
    selected = [i for i in range(len(candidates))
                if select[i].varValue is not None and select[i].varValue > 0.5]
    if not selected:
        return None
    reached = {int(j) for i in selected for j in covers[i]}
    if len(reached) < len(units):
        return None
    return selected


def select_pls(candidates: gpd.GeoDataFrame, units: gpd.GeoDataFrame, radius_m: float,
               time_limit_s: float = 300.0) -> tuple[gpd.GeoDataFrame, str]:
    """Selected PLSs (column pls_id) and the method that produced them ("exact" or "greedy")."""
    method = "exact"
    try:
        selected = solve_lscp_exact(candidates, units, radius_m, time_limit_s)
    except RuntimeError:
        raise
    except Exception:
        selected = None
    if selected is None:
        selected, method = greedy_set_cover(candidates, units, radius_m), "greedy"
    pls = candidates.iloc[selected].copy().reset_index(drop=True)
    pls["pls_id"] = np.arange(len(pls), dtype=int)
    return pls, method


def assign_units(units: gpd.GeoDataFrame, pls: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Each sample unit is assigned to its nearest PLS (columns pls_id and distance_m)."""
    distance, nearest = cKDTree(_xy(pls)).query(_xy(units))
    assigned = units.copy().reset_index(drop=True)
    assigned["pls_id"] = pls.iloc[nearest]["pls_id"].to_numpy()
    assigned["distance_m"] = distance
    return assigned

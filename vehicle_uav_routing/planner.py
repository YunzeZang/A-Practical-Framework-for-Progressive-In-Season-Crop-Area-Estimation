"""Response-design plans for one set of sample units.

    plan_vehicle_uav      PLS selection (LSCP), UAV sorties at every PLS and vehicle route over the
                          PLSs (vehicle-UAV optimization)
    replan_uav_sorties    UAV sorties of an existing plan re-optimized for another number of UAVs
    plan_vehicle_only     the vehicle stops at the road point nearest to every sample unit and the
                          UAV flies to the unit and back (vehicle optimization only)

Sample units are a GeoDataFrame of points with an integer column `id`, in a projected CRS in
metres (see to_utm).  The random generators of the ant colonies are seeded at the start of
every plan.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS
from shapely.geometry import LineString

from vehicle_uav_routing import road_network as rn
from vehicle_uav_routing.pls_selection import assign_units, generate_candidates, select_pls
from vehicle_uav_routing.uav_routing import plan_uav_sorties, uav_distances
from vehicle_uav_routing.vehicle_routing import aco_open_tour, plan_vehicle_route


def utm_crs(points: gpd.GeoDataFrame) -> CRS:
    """UTM zone of the centre of the points."""
    ll = points.to_crs(4326) if points.crs.to_epsg() != 4326 else points
    xmin, ymin, xmax, ymax = ll.total_bounds
    lon, lat = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    zone = int((lon + 180) / 6) % 60 + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def to_utm(points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return points.to_crs(utm_crs(points))


def _seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def _uav_summary(sorties, n_uavs, uav_speed_ms):
    flight_m, longest_m = uav_distances(sorties)
    km_per_h = uav_speed_ms * 3.6
    return {"n_uavs": int(n_uavs), "n_uav_sorties": int(len(sorties)),
            "uav_distance_km": round(flight_m / 1000.0, 3),
            "uav_flight_time_h": round(flight_m / 1000.0 / km_per_h, 3),
            "uav_time_h": round(longest_m / 1000.0 / km_per_h, 3)}


def plan_vehicle_uav(units: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, rules: rn.RoadRules, *,
                     coverage_radius_m: float, uav_range_m: float, uav_speed_ms: float, spacing_m: float,
                     n_uavs: int = 1, clip_buffer_m: float = 20_000.0, lscp_time_limit_s: float = 300.0,
                     seed: int | None = None) -> dict:
    """Vehicle-UAV plan.  `roads` are the roads of the study area in the CRS of `units`.

    Returns a dict with the GeoDataFrames pls (pls_id, visit_order), assigned (sample units with
    pls_id and distance_m), sorties and route (vehicle legs), and the summary.
    """
    _seed(seed)
    local = rn.clip_roads(rn.select_drivable(roads, rules), units, clip_buffer_m)
    candidates = generate_candidates(local, spacing_m)
    pls, lscp_method = select_pls(candidates, units, coverage_radius_m, lscp_time_limit_s)
    assigned = assign_units(units, pls)
    sorties = plan_uav_sorties(pls, assigned, uav_range_m, n_uavs)
    route, visit_order, vehicle_time_s, snap_max_m = plan_vehicle_route(pls, local, rules)
    pls["visit_order"] = pls["pls_id"].map(visit_order).astype(int)
    summary = {
        "method": "vehicle_uav", "n_sample_units": len(units), "n_candidate_pls": len(candidates),
        "n_pls": len(pls), "n_vehicle_stops": len(pls), "lscp_method": lscp_method,
        **_uav_summary(sorties, n_uavs, uav_speed_ms),
        "vehicle_distance_km": round(float(route["distance_m"].sum()) / 1000.0, 3),
        "vehicle_time_h": round(vehicle_time_s / 3600.0, 3),
        "pls_to_road_max_m": round(snap_max_m, 1), "seed": seed, "crs": units.crs.to_string(),
    }
    return {"pls": pls, "assigned": assigned, "sorties": sorties, "route": route, "summary": summary}


def replan_uav_sorties(plan: dict, n_uavs: int, *, uav_range_m: float, uav_speed_ms: float,
                       seed: int | None = None) -> dict:
    """UAV sorties of `plan` re-optimized for n_uavs UAVs operating in parallel at every PLS; the
    PLSs, the assignment of the sample units and the vehicle route are kept."""
    _seed(seed)
    pls = plan["pls"].sort_values("pls_id").reset_index(drop=True)
    sorties = plan_uav_sorties(pls, plan["assigned"], uav_range_m, n_uavs)
    summary = dict(plan["summary"])
    summary.update(_uav_summary(sorties, n_uavs, uav_speed_ms))
    summary["seed"] = seed
    return {"pls": pls, "assigned": plan["assigned"], "sorties": sorties, "route": plan.get("route"),
            "summary": summary}


def plan_vehicle_only(units: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, rules: rn.RoadRules, *,
                      uav_speed_ms: float, clip_buffer_m: float = 20_000.0, merge_m: float = 1.0,
                      seed: int | None = None) -> dict:
    """Vehicle optimization only: every sample unit is projected onto its nearest road segment,
    the vehicle visits all projected points (projections closer than merge_m on one segment share
    a stop) and the UAV flies from each stop to the sample unit and back.

    Returns a dict with stops (stop_id, visit_order), assigned (sample units with stop_id and
    offset_m), route (vehicle legs) and the summary.
    """
    _seed(seed)
    local = rn.clip_roads(rn.select_drivable(roads, rules), units, clip_buffer_m)
    graph, node_xy, edge_len, stops, assigned, _ = rn.build_network_with_stops(units, local, rules, merge_m)
    nodes = stops["node"].to_numpy()
    D = rn.travel_time_matrix(graph, nodes)
    order, vehicle_time_s = aco_open_tour(D)
    order = np.asarray(order)
    leg_time = D[order[:-1], order[1:]]
    legs = rn.route_legs(graph, nodes[order], leg_time, edge_len)

    offsets = assigned["offset_m"].to_numpy()
    xy = np.round(np.column_stack([assigned.geometry.x, assigned.geometry.y]), 2)
    _, first = np.unique(xy, axis=0, return_index=True)
    uav_m = 2.0 * offsets[first].sum()
    uav_h = uav_m / 1000.0 / (uav_speed_ms * 3.6)

    stop_ids = stops["stop"].to_numpy()
    visit_order = np.empty(len(stops), dtype=int)
    visit_order[order] = np.arange(1, len(stops) + 1)
    stops_frame = gpd.GeoDataFrame({"stop_id": stop_ids, "visit_order": visit_order, "n_units": stops["n_points"]},
                                   geometry=gpd.points_from_xy(stops["x"], stops["y"]), crs=units.crs)
    route = gpd.GeoDataFrame(
        {"leg_id": range(len(legs)), "from_stop": stop_ids[order[:-1]], "to_stop": stop_ids[order[1:]],
         "travel_time_s": leg_time, "distance_m": [d for _, d in legs]},
        geometry=[LineString(node_xy[seq]) for seq, _ in legs], crs=units.crs)
    summary = {
        "method": "vehicle_only", "n_sample_units": len(units), "n_vehicle_stops": len(stops),
        "n_uavs": 1, "n_uav_sorties": int(len(first)), "uav_distance_km": round(uav_m / 1000.0, 3),
        "uav_flight_time_h": round(uav_h, 3), "uav_time_h": round(uav_h, 3),
        "vehicle_distance_km": round(sum(d for _, d in legs) / 1000.0, 3),
        "vehicle_time_h": round(vehicle_time_s / 3600.0, 3), "seed": seed, "crs": units.crs.to_string(),
    }
    return {"stops": stops_frame, "assigned": assigned, "route": route, "summary": summary}


# ----------------------------------------------------------------------------- files
def _table(frame):
    return pd.DataFrame(frame.drop(columns="geometry"))


def write_plan(out_dir, plan: dict, gpkg: bool = False) -> None:
    """Write a plan as tables (summary.json and CSV files); gpkg = True also writes the point
    and line layers to plan.gpkg."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = {}
    if "pls" in plan:
        pls = plan["pls"]
        pd.DataFrame({"pls_id": pls["pls_id"], "visit_order": pls["visit_order"],
                      "x": pls.geometry.x, "y": pls.geometry.y}).to_csv(out_dir / "pls.csv", index=False)
        plan["assigned"][["id", "pls_id", "distance_m"]].to_csv(out_dir / "assignment.csv", index=False)
        layers.update(pls=pls, sample_units=plan["assigned"])
    if "stops" in plan:
        stops = plan["stops"]
        pd.DataFrame({"stop_id": stops["stop_id"], "visit_order": stops["visit_order"],
                      "x": stops.geometry.x, "y": stops.geometry.y}).to_csv(out_dir / "stops.csv", index=False)
        plan["assigned"][["id", "stop_id", "offset_m"]].to_csv(out_dir / "assignment.csv", index=False)
        layers.update(stops=stops, sample_units=plan["assigned"])
    if plan.get("sorties") is not None:
        _table(plan["sorties"]).to_csv(out_dir / "uav_sorties.csv", index=False)
        layers["uav_sorties"] = plan["sorties"]
    if plan.get("route") is not None:
        _table(plan["route"]).to_csv(out_dir / "vehicle_route.csv", index=False)
        layers["vehicle_route"] = plan["route"]
    if gpkg:
        for name, frame in layers.items():
            if len(frame):
                frame.to_file(out_dir / "plan.gpkg", layer=name, driver="GPKG")
    (out_dir / "summary.json").write_text(json.dumps(plan["summary"], indent=2), encoding="utf-8")


def read_vehicle_uav_plan(plan_dir, units: gpd.GeoDataFrame) -> dict:
    """Vehicle-UAV plan written by write_plan; `units` are the sample units of the plan in the
    CRS of the plan (summary["crs"])."""
    plan_dir = Path(plan_dir)
    summary = json.loads((plan_dir / "summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(plan_dir / "pls.csv", float_precision="round_trip")
    pls = gpd.GeoDataFrame(table[["pls_id", "visit_order"]], geometry=gpd.points_from_xy(table["x"], table["y"]),
                           crs=summary["crs"])
    assignment = pd.read_csv(plan_dir / "assignment.csv")
    if len(assignment) != len(units) or not np.array_equal(assignment["id"].to_numpy(), units["id"].to_numpy()):
        raise ValueError(f"{plan_dir}: the plan does not belong to these sample units")
    assigned = units.reset_index(drop=True).copy()
    assigned["pls_id"] = assignment["pls_id"].to_numpy()
    assigned["distance_m"] = assignment["distance_m"].to_numpy()
    return {"pls": pls, "assigned": assigned, "sorties": None, "route": None, "summary": summary}

"""Vehicle route over the stops: open travelling salesman problem on the road travel times,
solved with an ant colony."""
from __future__ import annotations

import random

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

from vehicle_uav_routing import road_network as rn

N_ANTS, N_ITERATIONS = 10, 100
ALPHA, BETA, EVAPORATION, DEPOSIT = 1, 5, 0.5, 100


def aco_open_tour(D: np.ndarray, n_ants=N_ANTS, n_iterations=N_ITERATIONS, alpha=ALPHA, beta=BETA,
                  evaporation=EVAPORATION, deposit=DEPOSIT):
    """Ant colony for the open tour over a travel-time matrix D.

    Each ant starts from a random stop and moves to an unvisited stop with probability
    proportional to pheromone^alpha * (1 / time)^beta.  After every iteration the pheromone
    evaporates and each ant deposits deposit / (tour time) on the arcs of its tour.  Returns the
    shortest tour found and its travel time.
    """
    n = len(D)
    if n == 1:
        return [0], 0.0
    with np.errstate(divide="ignore"):
        eta = np.where(D > 0, (1.0 / D) ** beta, 0.0)
    pher = np.ones((n, n))
    best_route, best_len = None, float("inf")
    for _ in range(n_iterations):
        routes, lengths = [], []
        for _ in range(n_ants):
            unvisited = np.ones(n, dtype=bool)
            cur = random.randint(0, n - 1)
            route = [cur]
            unvisited[cur] = False
            length = 0.0
            for _ in range(n - 1):
                idx = np.flatnonzero(unvisited)
                ph = pher[cur, idx]
                w = (ph if alpha == 1 else ph ** alpha) * eta[cur, idx]
                total = w.sum()
                if total <= 0:
                    nxt = int(random.choice(list(idx)))
                else:
                    cw = np.cumsum(w / total)
                    cw /= cw[-1]
                    k = min(int(np.searchsorted(cw, np.random.random(), side="right")), idx.size - 1)
                    nxt = int(idx[k])
                length += D[cur, nxt]
                route.append(nxt)
                unvisited[nxt] = False
                cur = nxt
            routes.append(route)
            lengths.append(length)
            if length < best_len:
                best_len, best_route = length, route
        pher *= (1 - evaporation)
        for r, length in zip(routes, lengths):
            dep = deposit / max(length, 1e-9)
            r = np.asarray(r)
            pher[r[:-1], r[1:]] += dep
            pher[r[-1], r[0]] += dep
    return best_route, float(best_len)


def plan_vehicle_route(pls: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, rules: rn.RoadRules,
                       n_ants: int = N_ANTS, n_iterations: int = N_ITERATIONS):
    """Vehicle route over the PLSs on the road network.

    Returns (legs, visit_order, travel_time_s, snap_max_m): legs has one row per leg (from_pls,
    to_pls, travel_time_s, distance_m, geometry), visit_order maps pls_id to its position in the
    route (1, 2, ...) and snap_max_m is the largest distance between a PLS and the road network.
    """
    columns = ["leg_id", "from_pls", "to_pls", "travel_time_s", "distance_m", "geometry"]
    ids = pls["pls_id"].astype(int).to_numpy()
    if len(pls) < 2:
        empty = gpd.GeoDataFrame({c: [] for c in columns[:-1]}, geometry=[], crs=pls.crs)
        return empty, {int(s): i + 1 for i, s in enumerate(ids)}, 0.0, 0.0
    graph, node_xy, edge_len, stops, assigned, _ = rn.build_network_with_stops(pls, roads, rules, merge_m=0.01)
    nodes = stops["node"].to_numpy()[assigned["stop_id"].to_numpy()]
    D = rn.travel_time_matrix(graph, nodes)
    order, time_s = aco_open_tour(D, n_ants, n_iterations)
    order = np.asarray(order)
    leg_time = D[order[:-1], order[1:]]
    legs = rn.route_legs(graph, nodes[order], leg_time, edge_len)
    route_ids = ids[order]
    frame = gpd.GeoDataFrame(
        {"leg_id": range(len(legs)), "from_pls": route_ids[:-1], "to_pls": route_ids[1:],
         "travel_time_s": leg_time, "distance_m": [d for _, d in legs]},
        geometry=[LineString(node_xy[seq]) for seq, _ in legs], crs=pls.crs)
    visit_order = {int(sid): i + 1 for i, sid in enumerate(route_ids)}
    return frame, visit_order, float(time_s), float(assigned["offset_m"].max())

"""UAV sorties at a park-and-launch site (PLS): endurance-constrained multi-salesman TSP solved
with a Min-Max ant colony.

Each ant builds a complete solution for all UAVs, which extend their routes in turn by one
sample unit.  A unit can be added to the current sortie only if the UAV can still return to the
PLS within its endurance; otherwise the UAV returns and starts a new sortie.  A solution is
evaluated by the longest cumulative flight distance among the UAVs.
"""
from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString

N_ANTS, N_ITERATIONS = 10, 100
ALPHA, BETA, EVAPORATION = 1.0, 2.0, 0.1


class MinMaxAntColony:
    """Node 0 is the PLS; nodes 1 ... n are the sample units assigned to it."""

    def __init__(self, nodes, n_uavs, max_distance, n_ants=N_ANTS, n_iterations=N_ITERATIONS,
                 alpha=ALPHA, beta=BETA, evaporation=EVAPORATION):
        self.n_nodes = len(nodes)
        self.n_uavs = n_uavs
        self.max_distance = max_distance
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.evaporation = evaporation
        xy = np.asarray(nodes, dtype=float)
        self.dist = np.zeros((self.n_nodes, self.n_nodes))
        for i in range(self.n_nodes):
            self.dist[i] = np.linalg.norm(xy - xy[i], axis=1)
        self.pheromone = np.ones((self.n_nodes, self.n_nodes))
        self.eta = (1.0 / (self.dist + 1e-9)) ** beta
        self.back = self.dist[:, 0].copy()
        self.best_paths = None
        self.best_cost = float("inf")

    def run(self):
        for _ in range(self.n_iterations):
            solutions = [self._construct() for _ in range(self.n_ants)]
            self.pheromone *= (1 - self.evaporation)
            for paths, cost in solutions:
                deposit = 1.0 / cost if cost > 0 else 0
                for path in paths:
                    for i in range(len(path) - 1):
                        self.pheromone[path[i]][path[i + 1]] += deposit
            for paths, cost in solutions:
                if cost < self.best_cost:
                    self.best_paths, self.best_cost = paths, cost
        return self.best_paths, self.best_cost

    def _construct(self):
        n, S, dist = self.n_nodes, self.n_uavs, self.dist
        paths = [[0] for _ in range(S)]
        unvisited = np.ones(n, dtype=bool)
        unvisited[0] = False
        n_left = n - 1
        uav_cost = [0.0] * S
        sortie_cost = [0.0] * S
        stalled = 0
        while n_left > 0:
            progress = False
            for s in range(S):
                if n_left == 0:
                    break
                cur = paths[s][-1]
                idx = np.flatnonzero(unvisited & (sortie_cost[s] + dist[cur] + self.back <= self.max_distance))
                if idx.size == 0:
                    if cur != 0:
                        uav_cost[s] += sortie_cost[s] + dist[cur][0]
                        paths[s].append(0)
                        sortie_cost[s] = 0.0
                    continue
                ph = self.pheromone[cur, idx]
                w = (ph if self.alpha == 1 else ph ** self.alpha) * self.eta[cur, idx]
                cw = np.cumsum(w / w.sum())
                cw /= cw[-1]
                k = min(int(np.searchsorted(cw, np.random.random(), side="right")), idx.size - 1)
                nxt = int(idx[k])
                paths[s].append(nxt)
                step = dist[cur][nxt]
                uav_cost[s] += step
                sortie_cost[s] += step
                unvisited[nxt] = False
                n_left -= 1
                progress = True
            stalled = 0 if progress else stalled + 1
            if stalled > S:
                for i in np.flatnonzero(unvisited):
                    s = int(np.argmin(uav_cost))
                    cur = paths[s][-1]
                    paths[s].append(int(i))
                    step = dist[cur][i]
                    uav_cost[s] += step
                    sortie_cost[s] += step
                unvisited[:] = False
                break
        for s in range(S):
            if paths[s][-1] != 0:
                uav_cost[s] += dist[paths[s][-1]][0]
                paths[s].append(0)
        return paths, max(uav_cost)


def split_sorties(path: list[int]) -> list[list[int]]:
    """Split the path of one UAV into its PLS-to-PLS sorties."""
    sorties: list[list[int]] = []
    current = [0]
    for node in path:
        if node == 0:
            if len(current) > 1:
                current.append(0)
                sorties.append(current)
            current = [0]
        else:
            current.append(node)
    if len(current) > 1:
        current.append(0)
        sorties.append(current)
    return sorties


def plan_uav_sorties(pls: gpd.GeoDataFrame, assigned: gpd.GeoDataFrame, max_range_m: float, n_uavs: int = 1,
                     n_ants: int = N_ANTS, n_iterations: int = N_ITERATIONS) -> gpd.GeoDataFrame:
    """Sorties of all PLSs: one row per sortie with pls_id, uav_id, distance_m, n_units, the ids
    of the visited sample units (unit_ids, comma separated) and the flight line."""
    rows = []
    sortie_id = 0
    for site in pls.itertuples():
        subset = assigned[assigned["pls_id"] == site.pls_id]
        nodes = [(float(site.geometry.x), float(site.geometry.y))]
        unit_ids: list[int | None] = [None]
        for row in subset.itertuples():
            nodes.append((float(row.geometry.x), float(row.geometry.y)))
            unit_ids.append(int(row.id))
        if len(nodes) < 2:
            continue
        best_paths, _ = MinMaxAntColony(nodes, n_uavs, max_range_m, n_ants, n_iterations).run()
        if not best_paths:
            continue
        for uav_id, path in enumerate(best_paths):
            for sortie in split_sorties(path):
                coords = [nodes[i] for i in sortie]
                distance = sum(float(math.hypot(a[0] - b[0], a[1] - b[1])) for a, b in zip(coords[:-1], coords[1:]))
                visited = [unit_ids[i] for i in sortie if i != 0]
                rows.append({"sortie_id": sortie_id, "pls_id": int(site.pls_id), "uav_id": int(uav_id),
                             "distance_m": distance, "n_units": len(visited),
                             "unit_ids": ",".join(str(x) for x in visited), "geometry": LineString(coords)})
                sortie_id += 1
    columns = ["sortie_id", "pls_id", "uav_id", "distance_m", "n_units", "unit_ids", "geometry"]
    return gpd.GeoDataFrame(rows, columns=columns, geometry="geometry", crs=pls.crs)


def uav_distances(sorties: gpd.GeoDataFrame) -> tuple[float, float]:
    """Total flight distance (m) and, summed over the PLSs, the flight distance of the UAV that
    flies longest at each PLS (m); the two are equal with one UAV."""
    if not len(sorties):
        return 0.0, 0.0
    per_uav = sorties.groupby(["pls_id", "uav_id"])["distance_m"].sum()
    return float(sorties["distance_m"].sum()), float(per_uav.groupby(level=0).max().sum())

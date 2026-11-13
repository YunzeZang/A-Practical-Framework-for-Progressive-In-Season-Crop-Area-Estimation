"""Road network of the vehicle: drivable roads, travel-time graph and stops on the roads.

* roads: lines with a road-class attribute; the classes that the vehicle can use and the
  driving speed of every class are given by RoadRules;
* graph: every road line is cut into straight segments between consecutive vertices and every
  vertex is a node, so roads that share a vertex are connected; only the largest connected
  network is kept; the edge weight is the travel time (s);
* stops: points are projected onto the nearest segment and inserted into the graph, so the
  vehicle stops exactly there.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from shapely.strtree import STRtree


@dataclass(frozen=True)
class RoadRules:
    """Road classes used by the vehicle and their driving speeds.

    class_field   attribute holding the road class
    fast_classes  classes driven at fast_kmh; every other class is driven at slow_kmh
    drivable      classes that the vehicle can use (None: all classes)
    excluded      classes removed from the network
    """
    class_field: str
    fast_classes: frozenset
    fast_kmh: float
    slow_kmh: float
    drivable: frozenset | None = None
    excluded: frozenset = frozenset()

    @classmethod
    def from_config(cls, cfg: dict) -> "RoadRules":
        return cls(class_field=cfg["class_field"], fast_classes=frozenset(cfg["fast_classes"]),
                   fast_kmh=float(cfg["fast_kmh"]), slow_kmh=float(cfg["slow_kmh"]),
                   drivable=frozenset(cfg["drivable"]) if cfg.get("drivable") else None,
                   excluded=frozenset(cfg.get("excluded") or ()))

    def speed_kmh(self, road_class: str) -> float:
        return self.fast_kmh if str(road_class).lower() in self.fast_classes else self.slow_kmh


def read_roads(path, crs, rules: RoadRules) -> gpd.GeoDataFrame:
    """Road lines with a road class, projected to `crs`."""
    roads = gpd.read_file(path)
    if roads.crs is None:
        raise ValueError(f"road file has no CRS: {path}")
    field = rules.class_field
    roads = roads[roads[field].notna() & (roads[field].astype(str) != "")]
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].copy()
    roads = roads[roads.geometry.geom_type.isin(["LineString", "MultiLineString"])]
    return roads.to_crs(crs).reset_index(drop=True)


def select_drivable(roads: gpd.GeoDataFrame, rules: RoadRules) -> gpd.GeoDataFrame:
    """Roads that the vehicle can use."""
    cls = roads[rules.class_field].astype(str)
    keep = cls.isin(rules.drivable) if rules.drivable is not None else pd.Series(True, index=roads.index)
    if rules.excluded:
        keep &= ~cls.str.lower().isin(rules.excluded)
    return roads[keep]


def clip_roads(roads: gpd.GeoDataFrame, points: gpd.GeoDataFrame, buffer_m: float) -> gpd.GeoDataFrame:
    """Roads within the convex hull of the points, enlarged by buffer_m."""
    boundary = points.geometry.union_all().convex_hull.buffer(buffer_m)
    roads = gpd.clip(roads, boundary)
    roads = roads[roads.geometry.notna() & ~roads.geometry.is_empty].reset_index(drop=True)
    if roads.empty:
        raise RuntimeError("no roads remain after clipping")
    return roads


def largest_network(roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Roads of the largest connected network (roads are connected where they share a vertex)."""
    lines = roads.explode(index_parts=False).reset_index(drop=True)
    lines = lines[lines.geom_type == "LineString"].reset_index(drop=True)
    coords, owner = shapely.get_coordinates(lines.geometry.to_numpy(), return_index=True)
    _, node = np.unique(np.round(coords, 1), axis=0, return_inverse=True)
    node = node.ravel()
    same = owner[:-1] == owner[1:]
    n = int(node.max()) + 1
    graph = coo_matrix((np.ones(int(same.sum())), (node[:-1][same], node[1:][same])), shape=(n, n))
    _, label = connected_components(graph, directed=False)
    first = np.unique(owner, return_index=True)[1]
    line_label = label[node[first]]
    length = np.bincount(line_label, weights=lines.length.to_numpy())
    return lines[line_label == length.argmax()].reset_index(drop=True)


def road_segments(roads: gpd.GeoDataFrame, rules: RoadRules) -> pd.DataFrame:
    """Every straight segment between consecutive vertices of the road lines."""
    lines = roads.geometry.explode(index_parts=False)
    lines = lines[lines.geom_type == "LineString"]
    road_class = roads[rules.class_field].reindex(lines.index).astype(str).to_numpy()
    coords, owner = shapely.get_coordinates(lines.to_numpy(), return_index=True)
    same = owner[:-1] == owner[1:]
    a, b = coords[:-1][same], coords[1:][same]
    cls = road_class[owner[:-1][same]]
    length = np.hypot(*(b - a).T)
    keep = length > 0
    speed_ms = np.array([rules.speed_kmh(c) / 3.6 for c in cls[keep]])
    return pd.DataFrame({"ax": a[keep, 0], "ay": a[keep, 1], "bx": b[keep, 0], "by": b[keep, 1],
                         "length": length[keep], "speed_ms": speed_ms})


def _node_ids(seg: pd.DataFrame):
    pts = np.vstack([seg[["ax", "ay"]].to_numpy(), seg[["bx", "by"]].to_numpy()])
    keys, inv = np.unique(np.round(pts, 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = len(seg)
    return keys, inv[:n], inv[n:]


def _csr(a, b, w, n):
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    f = pd.DataFrame({"lo": lo, "hi": hi, "w": w}).groupby(["lo", "hi"], as_index=False)["w"].min()
    return coo_matrix((f["w"].to_numpy(), (f["lo"].to_numpy(), f["hi"].to_numpy())), shape=(n, n)).tocsr()


def build_network_with_stops(points: gpd.GeoDataFrame, roads: gpd.GeoDataFrame, rules: RoadRules,
                             merge_m: float = 1.0):
    """Road graph (travel time, s) on all vertices, split at the projected points.

    Returns graph, node_xy, edge_len, stops, assigned, info: `stops` has one row per stop
    (stop, graph node, x, y, n_points); `assigned` is `points` with stop_id and the projection
    distance offset_m.  Projections closer than merge_m on the same segment share one stop.
    """
    seg = road_segments(roads, rules)
    keys, ua, ub = _node_ids(seg)
    base = _csr(ua, ub, seg["length"].to_numpy() / seg["speed_ms"].to_numpy(), len(keys))
    _, lab = connected_components(base, directed=False)
    main = np.bincount(lab).argmax()
    keep = lab[ua] == main
    seg, ua, ub = seg[keep].reset_index(drop=True), ua[keep], ub[keep]

    geoms = shapely.linestrings(np.stack([seg[["ax", "ay"]].to_numpy(), seg[["bx", "by"]].to_numpy()], axis=1))
    pts = points.geometry.to_numpy()
    (src, tgt), dist = STRtree(geoms).query_nearest(pts, return_distance=True, all_matches=False)
    order = np.argsort(src, kind="stable")
    src, tgt, dist = src[order], tgt[order], dist[order]
    first = np.unique(src, return_index=True)[1]
    seg_of = np.empty(len(pts), dtype=int)
    offset = np.empty(len(pts))
    seg_of[src[first]], offset[src[first]] = tgt[first], dist[first]
    along = shapely.line_locate_point(geoms[seg_of], pts)

    p = pd.DataFrame({"d": np.arange(len(pts)), "seg": seg_of, "s": along}).sort_values(["seg", "s"])
    new = (p["seg"].diff() != 0) | (p["s"].diff() > merge_m)
    p["stop"] = np.cumsum(new.to_numpy()) - 1
    stops = p.groupby("stop").agg(seg=("seg", "first"), s=("s", "mean")).reset_index()
    stops["node"] = len(keys) + stops["stop"].to_numpy()
    sxy = shapely.get_coordinates(shapely.line_interpolate_point(geoms[stops["seg"]], stops["s"]))
    stop_of = np.empty(len(pts), dtype=int)
    stop_of[p["d"].to_numpy()] = p["stop"].to_numpy()

    has = np.zeros(len(seg), dtype=bool)
    has[stops["seg"].unique()] = True
    L, V = seg["length"].to_numpy(), seg["speed_ms"].to_numpy()
    A, B, W, D = [ua[~has]], [ub[~has]], [L[~has] / V[~has]], [L[~has]]
    for k, grp in stops.groupby("seg"):
        nodes = np.concatenate([[ua[k]], grp["node"].to_numpy(), [ub[k]]])
        s = np.concatenate([[0.0], grp["s"].to_numpy(), [L[k]]])
        piece = np.maximum(np.diff(s), 1e-3)
        A.append(nodes[:-1]); B.append(nodes[1:]); W.append(piece / V[k]); D.append(piece)
    A, B, W, D = map(np.concatenate, (A, B, W, D))
    n_nodes = len(keys) + len(stops)
    graph = _csr(A, B, W, n_nodes)
    e = pd.DataFrame({"lo": np.minimum(A, B), "hi": np.maximum(A, B), "w": W, "len": D})
    e = e.sort_values("w").drop_duplicates(["lo", "hi"])
    edge_len = dict(zip(zip(e["lo"].to_numpy(), e["hi"].to_numpy()), e["len"].to_numpy()))
    node_xy = np.vstack([keys.astype(float), sxy])

    assigned = points.copy().reset_index(drop=True)
    assigned["stop_id"] = stop_of
    assigned["offset_m"] = offset
    stops = stops.assign(x=sxy[:, 0], y=sxy[:, 1], n_points=np.bincount(stop_of, minlength=len(stops)))
    info = {"n_road_segments": int(len(seg)), "n_graph_nodes": int(n_nodes)}
    return graph, node_xy, edge_len, stops, assigned, info


def travel_time_matrix(graph, nodes: np.ndarray, chunk: int = 64) -> np.ndarray:
    """Shortest travel time (s) between all stops."""
    n = len(nodes)
    out = np.empty((n, n))
    for i in range(0, n, chunk):
        out[i:i + chunk] = dijkstra(graph, directed=False, indices=nodes[i:i + chunk])[:, nodes]
    if not np.isfinite(out).all():
        raise RuntimeError("unreachable stops in the road network")
    return out


def route_legs(graph, path_nodes: np.ndarray, leg_time_s: np.ndarray, edge_len: dict, chunk: int = 64):
    """Node sequence and road distance (m) of every leg of a route."""
    legs = []
    for i in range(0, len(path_nodes) - 1, chunk):
        src = path_nodes[i:i + chunk]
        limit = float(leg_time_s[i:i + chunk].max()) * 1.001 + 1.0
        _, pred = dijkstra(graph, directed=False, indices=src, return_predecessors=True, limit=limit)
        for r, s in enumerate(src):
            k = i + r
            if k + 1 >= len(path_nodes):
                break
            seq = [int(path_nodes[k + 1])]
            while seq[-1] != s:
                seq.append(int(pred[r, seq[-1]]))
            seq.reverse()
            legs.append((seq, float(sum(edge_len[(min(u, v), max(u, v))] for u, v in zip(seq[:-1], seq[1:])))))
    return legs

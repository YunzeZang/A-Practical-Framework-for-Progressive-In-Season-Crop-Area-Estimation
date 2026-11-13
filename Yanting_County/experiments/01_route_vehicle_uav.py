"""Response design in Yanting County: vehicle-UAV plan for the 263 sample units.

Candidate park-and-launch sites (PLSs) every 1 km along the roads, plus the midpoint of every
road 1-2 km long; the fewest PLSs such that every sample unit lies within 5 km of a PLS (LSCP);
each sample unit assigned to its nearest PLS; UAV sorties at every PLS (Min-Max ant colony, one
UAV, 15 km endurance, 15 m/s); vehicle route over the PLSs (ant colony on the road travel
times, 60 km/h on primary and secondary roads and 30 km/h elsewhere, motorways excluded; only
the largest connected road network is used).

The plan followed in the field was computed before the survey with a commercial network solver
and candidate PLSs every 2 km.  This script repeats the workflow with open-source tools; with
the road layer of this repository one sample unit has no candidate within 5 km at a spacing of
2 km, so 1 km is used.  The result is close to, but not identical with, the field plan.

Reads   data/sample_points/sample_units_Yanting.shp, data/roads/osm_roads_Yanting.shp
Writes  results/response_design/
            summary.json        number of PLSs and sorties, vehicle travel time, UAV flight time
            pls.csv             selected PLSs (UTM zone 48N) and their order in the vehicle route
            assignment.csv      PLS of every sample unit and its distance to the PLS
            uav_sorties.csv     sample units and flight distance of every sortie
            vehicle_route.csv   travel time and distance of every leg of the route
            plan.gpkg           PLSs, sample units, sorties and vehicle route as map layers

Usage   python Yanting_County/experiments/01_route_vehicle_uav.py
"""
import json

import geopandas as gpd

import common as C
from vehicle_uav_routing import planner, road_network as rn

R = C.CFG["routing"]


def main():
    units = planner.to_utm(gpd.read_file(C.SAMPLE_UNITS))
    rules = rn.RoadRules.from_config(R["roads"])
    roads = rn.largest_network(rn.select_drivable(rn.read_roads(C.ROADS, units.crs, rules), rules))
    plan = planner.plan_vehicle_uav(units, roads, rules, coverage_radius_m=R["coverage_radius_m"],
                                    uav_range_m=R["uav_range_m"], uav_speed_ms=R["uav_speed_ms"],
                                    spacing_m=R["pls_spacing_m"], n_uavs=1, clip_buffer_m=R["clip_buffer_m"],
                                    seed=R["seed"])
    plan_file = C.ROUTE_DIR / "plan.gpkg"
    if plan_file.exists():
        plan_file.unlink()
    planner.write_plan(C.ROUTE_DIR, plan, gpkg=True)
    print(json.dumps(plan["summary"], indent=2))


if __name__ == "__main__":
    main()

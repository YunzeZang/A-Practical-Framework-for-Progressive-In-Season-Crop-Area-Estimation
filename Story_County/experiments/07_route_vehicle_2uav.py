"""Simulated response design (iii): vehicle-UAV optimization with two UAVs (Fig. 4).

The park-and-launch sites (PLSs), the assignment of the sample units and the vehicle route of
the one-UAV plans are kept.  The sorties at every PLS are re-optimized for two UAVs operating in
parallel (Min-Max ant colony with two salesmen).  The UAV time of a plan is the sum over the
PLSs of the flight time of the UAV that flies longest.

Reads   data/sample_points/conf95_dXX/rep_NNNN.csv
        results/response_design/vehicle_1uav/conf95_dXX/rep_NNNN/   (06_route_vehicle_1uav.py)
Writes  results/response_design/vehicle_2uav/conf95_dXX/rep_NNNN/
            summary.json        as the one-UAV plan, with the UAV figures of two UAVs
            uav_sorties.csv     UAV, sample units and flight distance of every sortie

Usage   python Story_County/experiments/07_route_vehicle_2uav.py [--designs conf95_d05] [--replicates 75]
                                                                 [--workers 4] [--overwrite] [--gpkg]
"""
import common as C
from vehicle_uav_routing import planner

R = C.CFG["routing"]
N_UAVS = 2


def plan_one(task):
    design, replicate, out_dir, gpkg = task
    source = C.ROUTE_DIR / "vehicle_1uav" / design / f"rep_{replicate:04d}"
    units = C.read_sample_points(design, replicate)
    one_uav = planner.read_vehicle_uav_plan(source, units.to_crs(planner.utm_crs(units)))
    plan = planner.replan_uav_sorties(one_uav, N_UAVS, uav_range_m=R["uav_range_m"],
                                      uav_speed_ms=R["uav_speed_ms"], seed=R["seed"])
    planner.write_plan(out_dir, {"sorties": plan["sorties"], "summary": plan["summary"]}, gpkg=gpkg)
    return design, replicate, plan["summary"]


def main():
    args = C.routing_arguments(__doc__.splitlines()[0])
    tasks = [(*t, args.gpkg) for t in C.routing_tasks("vehicle_2uav", args)
             if (C.ROUTE_DIR / "vehicle_1uav" / t[0] / f"rep_{t[1]:04d}" / "summary.json").exists()]
    C.run_tasks(plan_one, tasks, args.workers)


if __name__ == "__main__":
    main()

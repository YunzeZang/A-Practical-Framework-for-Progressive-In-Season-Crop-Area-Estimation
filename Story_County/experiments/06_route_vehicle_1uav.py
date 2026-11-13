"""Simulated response design (ii): vehicle-UAV optimization with one UAV (Fig. 4).

For every sample: candidate park-and-launch sites (PLSs) every 500 m along the drivable roads,
plus the midpoint of every road 1-2 km long; the fewest PLSs such that every sample unit lies
within 5 km of a PLS (LSCP); each sample unit assigned to its nearest PLS; UAV sorties at every
PLS (Min-Max ant colony, 15 km endurance, 15 m/s); vehicle route over the PLSs (ant colony on
the road travel times, 90 km/h on motorway, trunk and primary roads and 50 km/h elsewhere).

Reads   data/sample_points/conf95_dXX/rep_NNNN.csv   (04_export_sample_points.py), data/roads
Writes  results/response_design/vehicle_1uav/conf95_dXX/rep_NNNN/
            summary.json        number of PLSs and sorties, vehicle travel time, UAV flight time
            pls.csv             selected PLSs (working CRS) and their order in the vehicle route
            assignment.csv      PLS of every sample unit and its distance to the PLS
            uav_sorties.csv     sample units and flight distance of every sortie
            vehicle_route.csv   travel time and distance of every leg of the route

Usage   python Story_County/experiments/06_route_vehicle_1uav.py [--designs conf95_d05] [--replicates 75]
                                                                 [--workers 4] [--overwrite] [--gpkg]
        All 500 samples take several hours on one process.
"""
import common as C
from vehicle_uav_routing import planner

R = C.CFG["routing"]
STATE = {}


def initialise():
    STATE["rules"], STATE["roads"] = C.load_roads()


def plan_one(task):
    design, replicate, out_dir, gpkg = task
    units = planner.to_utm(C.read_sample_points(design, replicate))
    plan = planner.plan_vehicle_uav(units, STATE["roads"], STATE["rules"], coverage_radius_m=R["coverage_radius_m"],
                                    uav_range_m=R["uav_range_m"], uav_speed_ms=R["uav_speed_ms"],
                                    spacing_m=R["pls_spacing_m"], n_uavs=1, clip_buffer_m=R["clip_buffer_m"],
                                    seed=R["seed"])
    planner.write_plan(out_dir, plan, gpkg=gpkg)
    return design, replicate, plan["summary"]


def main():
    args = C.routing_arguments(__doc__.splitlines()[0])
    tasks = [(*t, args.gpkg) for t in C.routing_tasks("vehicle_1uav", args)]
    C.run_tasks(plan_one, tasks, args.workers, initialise)


if __name__ == "__main__":
    main()

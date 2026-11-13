"""Simulated response design (i): vehicle optimization only (Fig. 4).

No park-and-launch site is selected.  The vehicle stops at the road point nearest to each sample
unit, from which the UAV flies to the sample unit and back (15 m/s); the vehicle route over all
stops is optimized with the ant colony.  Roads: drivable OpenStreetMap classes, 90 km/h on
motorway, trunk and primary roads and 50 km/h elsewhere.

Reads   data/sample_points/conf95_dXX/rep_NNNN.csv   (04_export_sample_points.py), data/roads
Writes  results/response_design/vehicle_only/conf95_dXX/rep_NNNN/
            summary.json        number of stops, vehicle travel time, UAV flight time
            stops.csv           vehicle stops and their order in the route
            assignment.csv      stop of every sample unit and its distance to the stop
            vehicle_route.csv   travel time and distance of every leg of the route

Usage   python Story_County/experiments/05_route_vehicle_only.py [--designs conf95_d05] [--replicates 75]
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
    plan = planner.plan_vehicle_only(units, STATE["roads"], STATE["rules"], uav_speed_ms=R["uav_speed_ms"],
                                     clip_buffer_m=R["clip_buffer_m"], seed=R["seed"])
    planner.write_plan(out_dir, plan, gpkg=gpkg)
    return design, replicate, plan["summary"]


def main():
    args = C.routing_arguments(__doc__.splitlines()[0])
    tasks = [(*t, args.gpkg) for t in C.routing_tasks("vehicle_only", args)]
    C.run_tasks(plan_one, tasks, args.workers, initialise)


if __name__ == "__main__":
    main()

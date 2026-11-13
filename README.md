# A Practical Framework for Progressive In-Season Crop Area Estimation

Code and data of the study "A practical framework for progressive in-season crop area estimation"

## Contents

```
area_estimation/            general functions: area estimation and accuracy
    sample_design.py            sample size and allocation (Section 2.1)
    estimators.py               stratified, within-stratum and combined ratio estimators (Section 2.3)
    accuracy.py                 UA, PA, F1-score and their standard errors (Section 2.4)
    sampling.py                 stratified random sampling without replacement
    monte_carlo.py              evaluation of the estimators on a fully known population
    raster_utils.py             census cross-tabulation, map class at sample points
vehicle_uav_routing/        general functions: vehicle-UAV response design (Section 2.2, Appendix A)
    road_network.py             road graph, travel times, stops on the roads
    pls_selection.py            candidate park-and-launch sites (PLSs), LSCP, assignment
    vehicle_routing.py          vehicle route (ant colony, open TSP)
    uav_routing.py              UAV sorties (Min-Max ant colony, endurance-constrained)
    planner.py                  complete plans and their files
Story_County/               simulation experiment (corn, Iowa, USA, growing season 2025)
    config.yaml  data/  experiments/  figures/  results/
Yanting_County/             practical experiment (rapeseed, Yanting, China, growing season 2024-2025)
    config.yaml  data/  experiments/  figures/  results/
```

## Installation

Python 3.10 or later.

```
pip install -r requirements.txt
```

## Running the experiments

Every script runs on its own from the repository root and reads the results of the earlier
scripts from `results/`; all results are stored in the repository, so any script can be run
without running the others first.

### Story County

| Script | Content | Paper |
| `experiments/01_build_population.py` | census table of the population, accuracy of the maps | Section 3.1.1, Fig. 5a |
| `experiments/02_mc_precision_designs.py` | Monte Carlo simulation, five sample sizes, M = 10,000 | Fig. 5 | 
| `experiments/03_mc_sample_allocation.py` | Monte Carlo simulation, three allocations at n = 237 | Fig. C1 |
| `experiments/04_export_sample_points.py` | sample points of 100 replicates per design | Section 3.1.2.2 |
| `experiments/05_route_vehicle_only.py` | response design (i): vehicle optimization only | Fig. 4 |
| `experiments/06_route_vehicle_1uav.py` | response design (ii): vehicle-UAV optimization, one UAV | Fig. 4 |
| `experiments/07_route_vehicle_2uav.py` | response design (iii): vehicle-UAV optimization, two UAVs | Fig. 4 |
| `experiments/08_numerical_example.py` | numerical example of one replicate | Appendix B |
| `figures/fig4_time_cost.py` | time cost of the simulated field surveys | Fig. 4 |
| `figures/fig5_estimator_performance.py` | performance of the poststratified estimators | Fig. 5 |
| `figures/figC1_sample_allocation.py` | sensitivity to the sample allocation | Fig. C1 |

```
python Story_County/experiments/02_mc_precision_designs.py
python Story_County/figures/fig5_estimator_performance.py
```

### Yanting County

| Script | Content | Paper |
|---|---|---|
| `experiments/01_route_vehicle_uav.py` | vehicle-UAV plan for the 263 sample units | Section 4.2.2 |
| `experiments/02_progressive_estimation.py` | initial and progressively updated area estimates, map accuracy | Table 4, Fig. 9 |
| `figures/fig9_progressive_estimates.py` | in-season estimates of the rapeseed area | Fig. 9 |

## Data

### Story County (`Story_County/data`)

All rasters was projected to 10 m grid in NAD83 / Conus Albers (EPSG:5070).

| Folder | Content | Values |
| `initial_map/` | in-season corn map of 20 June 2025 (design strata *h*) | 1 corn, 0 non-corn, 255 outside the population |
| `updated_maps/` | in-season corn maps of 30 June ... 29 August 2025 (classes *k*) | 1 corn, 0 non-corn, 255 outside the population |
| `reference/` | 2025 USDA Cropland Data Layer (reference label *y*) | 1 corn, 0 non-corn, 255 outside the population |
| `roads/` | OpenStreetMap roads, attribute `highway` | |
| `sample_points/conf95_dXX/rep_NNNN.csv` | sample units of 100 replicates of each design (`d05` ... `d01`: half-width 5 % ... 1 %) | `h`, `reference`, `k_YYYYMMDD`: 1 corn, 0 non-corn; `x`, `y`: pixel centre in EPSG:5070 |

The population is the 14,861,881 pixels with a value in every map (148,618.8 ha, of which
65,123.1 ha are corn in the CDL).

### Yanting County (`Yanting_County/data`)

All rasters was projected to 3 m grid in an Albers equal-area conic projection centred on the county
(WGS 84; central meridian 105.48 E; standard parallels 31.08 N and 31.42 N; latitude of origin
31.25 N).

| Folder | Content | Values |
| `initial_map/` | initial map of 16 December 2024 (design strata *h*) | 0 uncultivated, 1 non-rapeseed, 2 rapeseed, 255 outside the county |
| `updated_maps/` | updated maps of 16 January, 25 March and 19 April 2025 (classes *k*) | 0 uncultivated, 1 non-rapeseed, 2 rapeseed, 255 outside the county |
| `sample_points/` | the 263 sample units | `h`: 1 non-rapeseed, 2 rapeseed; `y`: 1 rapeseed, 0 non-rapeseed |
| `roads/` | OpenStreetMap roads |

## Results

| File | Content |
|---|---|
| `Story_County/results/area_estimation/mc_precision_replicates.npz` | area estimate and SE of every replicate, design, date and estimator |
| `Story_County/results/area_estimation/mc_allocation_replicates.npz` | the same for the three allocations |
| `Story_County/results/area_estimation/mc_*_summary.csv` | mean, empirical SD, RMS of the estimated SEs, coverage, failure rate |
| `Story_County/results/response_design/<method>/conf95_dXX/rep_NNNN/` | plan of every sample: `summary.json` and tables |
| `Story_County/results/response_design/time_cost_summary.csv` | mean time cost per method and design |
| `Story_County/results/numerical_example/appendixB_numbers.json` | all numbers of Appendix B |
| `Yanting_County/results/area_estimation/progressive_estimates.csv` | estimates, SEs, map accuracy and pixel-counting area per date |
| `Yanting_County/results/response_design/` | vehicle-UAV plan computed by `01_route_vehicle_uav.py` |

## Data sources

USDA NASS Cropland Data Layer; Copernicus Sentinel-2; PlanetScope; road data (c) OpenStreetMap
contributors, available under the Open Database License.

## Use of AI tools

Claude (Anthropic) was used to assist in refactoring and documenting the code in this
repository. The methods, experiments and results are the authors' own; the authors reviewed
and tested the code and take full responsibility for its content.
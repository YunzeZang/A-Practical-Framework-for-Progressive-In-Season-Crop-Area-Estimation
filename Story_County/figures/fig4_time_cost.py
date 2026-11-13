"""Fig. 4: time cost of the simulated field surveys for vehicle optimization only and
vehicle-UAV optimization with one or two UAVs, under the five precision designs.

Each bar stacks the vehicle travel time, the UAV flight time and a fixed stopping time of 2 min
per vehicle stop.  Bars are means over the 100 replicates of a design, whiskers span the minimum
and the maximum, and the y axis is broken above the second-highest bar.

Reads   results/response_design/<method>/conf95_dXX/rep_NNNN/summary.json
        (05_route_vehicle_only.py, 06_route_vehicle_1uav.py, 07_route_vehicle_2uav.py)
Writes  results/figures/Fig4_time_cost.png
        results/response_design/time_cost_summary.csv

Usage   python Story_County/figures/fig4_time_cost.py
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

import plot_style as G
from plot_style import C

DESIGNS = C.DESIGNS[::-1]                                          # 5 % ... 1 %
STOP_MIN = float(C.CFG["routing"]["stop_minutes"])                 # fixed stopping time per vehicle stop
METHODS = [  # folder, label, hatch
    ("vehicle_only", "Vehicle optimization only", "////"),
    ("vehicle_1uav", "Vehicle\u2013UAV optimization (1 UAV)", ""),
    ("vehicle_2uav", "Vehicle\u2013UAV optimization (2 UAVs)", "xxx"),
]
VEHICLE_C, UAV_C, STOP_C = "#D35400", "#27AE60", "#9E9E9E"
FS = {"tick": 13.0, "label": 14.0, "annot": 11.0, "legend": 11.5}
BAR_W = 0.27
FIG_W, FIG_H = 7.2, 4.6                                            # inches


def load():
    """One row per plan: vehicle travel, UAV flight and stopping time (h)."""
    rows = []
    for method, label, _ in METHODS:
        for design in DESIGNS:
            for f in sorted((C.ROUTE_DIR / method / design).glob("rep_*/summary.json")):
                m = json.loads(f.read_text(encoding="utf-8"))
                vehicle, uav = float(m["vehicle_time_h"]), float(m["uav_time_h"])
                stop = int(m["n_vehicle_stops"]) * STOP_MIN / 60.0
                rows.append({"method": method, "label": label, "design": design, "replicate": f.parent.name,
                             "n_sample_units": int(m["n_sample_units"]), "n_stops": int(m["n_vehicle_stops"]),
                             "n_uav_sorties": int(m["n_uav_sorties"]), "vehicle_h": vehicle, "uav_h": uav,
                             "stop_h": stop, "total_h": vehicle + uav + stop})
    return pd.DataFrame(rows)


def bars(ax, df, label_range):
    x = np.arange(len(DESIGNS), dtype=float)
    offsets = (np.arange(len(METHODS)) - (len(METHODS) - 1) / 2) * BAR_W
    for off, (key, _, hatch) in zip(offsets, METHODS):
        for xi, d in zip(x + off, DESIGNS):
            s = df[(df.method == key) & (df.design == d)]
            if s.empty:
                continue
            vehicle, uav, stop, total = s.vehicle_h.mean(), s.uav_h.mean(), s.stop_h.mean(), s.total_h.mean()
            v = s.total_h.to_numpy(dtype=float)
            lo, hi = v.mean() - v.min(), v.max() - v.mean()          # whisker: minimum to maximum
            kw = dict(width=BAR_W, edgecolor="white", linewidth=0.4, hatch=hatch, zorder=2)
            ax.bar(xi, vehicle, color=VEHICLE_C, **kw)
            ax.bar(xi, uav, bottom=vehicle, color=UAV_C, **kw)
            ax.bar(xi, stop, bottom=vehicle + uav, color=STOP_C, **kw)
            ax.errorbar(xi, total, yerr=[[lo], [hi]], fmt="none", ecolor="0.25",
                        elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=3)
            if label_range[0] <= total < label_range[1]:
                ax.annotate(f"{int(round(total))}", (xi, total + hi), xytext=(0, 2),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=FS["annot"], color="0.15", annotation_clip=False)
    n_of = df.groupby("design")["n_sample_units"].first()
    ax.set_xticks(x)
    ax.set_xticklabels([f"{float(d.rsplit('d', 1)[1]):g}%\n({n_of.get(d, 0):,})" for d in DESIGNS])
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)


def plot(df, out_png):
    G.setup_rc(FS["tick"], FS["label"], **{"hatch.linewidth": 0.6})
    tops = df.groupby(["method", "design"])["total_h"].max().sort_values()
    low_top = np.ceil(float(tops.iloc[-2]) * 1.12 / 10) * 10
    hi_bot = np.floor(float(tops.iloc[-1]) * 0.90 / 10) * 10   
    hi_top = float(tops.iloc[-1]) * 1.06
    print(f"y axis broken between {low_top:g} and {hi_bot:g} h")
    fig, (ax_h, ax_a) = plt.subplots(2, 1, sharex=True, figsize=(FIG_W, FIG_H),
                                     gridspec_kw={"height_ratios": [1, 4], "hspace": 0.06})
    bars(ax_h, df, label_range=(low_top, np.inf))
    bars(ax_a, df, label_range=(-np.inf, low_top))
    ax_a.set_ylim(0, low_top)
    ax_h.set_ylim(hi_bot, hi_top)
    ax_a.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="upper"))
    ax_h.yaxis.set_major_locator(MaxNLocator(nbins=2))
    ax_h.spines["bottom"].set_visible(False)
    ax_h.tick_params(axis="x", which="both", length=0, labelbottom=False)
    d = 0.012                                         # break marks on the y axis
    kw = dict(color="k", clip_on=False, linewidth=0.9)
    ax_h.plot((-d, +d), (-d * 4, +d * 4), transform=ax_h.transAxes, **kw)
    ax_a.plot((-d, +d), (1 - d, 1 + d), transform=ax_a.transAxes, **kw)
    ax_a.set_xlabel("Half-width\n(sample size)", labelpad=3)
    fig.text(0.0, 0.55, "Time cost (h)", rotation=90, ha="left", va="center",
             fontsize=FS["label"], fontweight="bold")
    handles = [Patch(facecolor=VEHICLE_C, edgecolor="none", label="Vehicle travel"),
               Patch(facecolor=UAV_C, edgecolor="none", label="UAV flight"),
               Patch(facecolor=STOP_C, edgecolor="none", label=f"Vehicle stopping ({STOP_MIN:g} min per stop)")]
    handles += [Patch(facecolor="white", edgecolor="0.3", hatch=h, linewidth=0.6, label=lab)
                for _, lab, h in METHODS]
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.14)
    pos = ax_h.get_position()                         # key at the top left, drawn across the break
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(pos.x0 + 0.012, pos.y1), frameon=False,
               fontsize=FS["legend"], handlelength=1.6, handletextpad=0.5)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.14)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load()
    C.FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot(df, C.FIGURE_DIR / "Fig4_time_cost.png")
    table = (df.groupby(["design", "label"], sort=False)
               .agg(n_replicates=("total_h", "size"), n_sample_units=("n_sample_units", "first"),
                    n_stops=("n_stops", "mean"), n_uav_sorties=("n_uav_sorties", "mean"),
                    vehicle_h=("vehicle_h", "mean"), uav_h=("uav_h", "mean"), stop_h=("stop_h", "mean"),
                    total_h=("total_h", "mean"), total_min_h=("total_h", "min"), total_max_h=("total_h", "max"))
               .reset_index())
    table.round(3).to_csv(C.ROUTE_DIR / "time_cost_summary.csv", index=False)
    print(table.round(2).to_string(index=False))


if __name__ == "__main__":
    main()

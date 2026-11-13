"""Style and helpers shared by the box-plot figures of the simulation (Figs. 5 and C1)."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common as C  # noqa: E402

FS = {"tick": 10, "label": 10, "annot": 9, "small": 8}
BLUE, ORANGE = "#1F618D", "#D35400"                    # within-stratum, combined ratio
EST_COL = [BLUE, ORANGE]
GREY = "0.45"                                          # stratified estimator on the initial map
RED = "#E00000"                                        # empirical SD
ACC_COL = {"F1": "#D35400", "UA": "#1F618D", "PA": "#27AE60"}
ACC_ORDER = ("F1", "UA", "PA")
BAR_10, BAR_W = "0.72", 0.34                           # bars of the flipped substrata
OFF, BW = 0.19, 0.30                                   # box offset from the date and box width
FLIP_FLOOR, FLIP_CAP, FLIP_TOP = 0.1, 100.0, 10 ** 3.1
FAIL_TOP = 140.0
DASH = dict(color="0.6", lw=0.8, ls=(0, (4, 2)), zorder=0)
TRUTH = dict(color="0.1", lw=0.9, ls=(0, (4, 2)), zorder=10)
Z95 = 1.959963984540054
ACC_LO = 0.5
LABEL_GAP = 0.12                                       # gap (cm) between y-axis titles and tick labels
SERIF = ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "Liberation Serif", "DejaVu Serif"]


def setup_rc(size=FS["tick"], label_size=FS["label"], **extra):
    plt.rcParams.update({
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
        "font.size": size, "axes.labelsize": label_size, "axes.labelweight": "bold",
        "xtick.labelsize": size, "ytick.labelsize": size,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 300, "savefig.dpi": 300, **extra,
    })


def date_label(d):
    return f"{d[4:6]}-{d[6:]}"


def series_x(t):
    """x positions of (within-stratum, combined ratio); the stratified box at t = 0 is centred."""
    return (t, None) if t == 0 else (t - OFF, t + OFF)


def box(ax, x, lo, hi, wlo, whi, mid, color, mid_color, z=3, mid_lw=1.3):
    """Box from lo to hi, whiskers to wlo and whi, and a horizontal line at mid."""
    ax.add_patch(Rectangle((x - BW / 2, lo), BW, hi - lo, facecolor=color, alpha=0.40,
                           edgecolor=color, lw=0.8, zorder=z))
    ax.plot([x, x], [wlo, lo], color=color, lw=0.8, zorder=z)
    ax.plot([x, x], [hi, whi], color=color, lw=0.8, zorder=z)
    ax.plot([x - BW / 2, x + BW / 2], [mid, mid], color=mid_color, lw=mid_lw, zorder=z + 1,
            solid_capstyle="butt")


def finite(a):
    return a[np.isfinite(a)]


def fail_label(v):
    return "0" if round(v, 1) == 0 else (f"{v:.1f}" if v < 9.95 else f"{v:.0f}")

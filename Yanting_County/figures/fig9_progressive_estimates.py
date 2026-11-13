"""Fig. 9: in-season estimates of the rapeseed area in Yanting County.

    (a) combined ratio poststratified estimator and (b) within-stratum poststratified estimator:
        estimate +/- SE, 95 % and 99 % confidence intervals, pixel-counting area and official
        rapeseed area; the SE of the proportion (%) is printed for every date
    (c) UA, PA and F1-score of every map with 95 % confidence intervals

Reads   results/area_estimation/progressive_estimates.csv   (02_progressive_estimation.py)
Writes  results/figures/Fig9_progressive_estimates.png

Usage   python Yanting_County/figures/fig9_progressive_estimates.py
"""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import common as C  # noqa: E402

RESULTS_CSV = C.AREA_DIR / "progressive_estimates.csv"
OUT_PNG = C.FIGURE_DIR / "Fig9_progressive_estimates.png"

Z95, Z99 = 1.96, 2.576
CENSUS_HA = C.CFG["census_area_ha"]      # official rapeseed area, dashed reference line
TARGET_W_CM = 17.5                       # double-column width in the paper
FS = {"tick": 9, "label": 9, "panel": 9, "annot": 8}

C_CI99, C_CI95 = "red", "blue"
C_PIXEL        = "red"
C_BARS = {"F1": "#D35400", "UA": "#1F618D", "PA": "#27AE60"}

METHODS = [
    ("a", "Combined ratio estimator", "area_cr", "se_cr", "prop_se_cr"),
    ("b", "Within-stratum estimator", "area_ws", "se_ws", "prop_se_ws"),
]

# Vertical budget of the area panels (axes fraction): SE row at the bottom,
# data in the middle, series key at the top.
DATA_LO, DATA_HI = 0.22, 0.74


# ----------------------------------------------------------------------
def setup_rc():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "Liberation Serif",
                       "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": FS["tick"],
        "axes.labelsize": FS["label"],
        "axes.labelweight": "bold",
        "xtick.labelsize": FS["tick"],
        "ytick.labelsize": FS["tick"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    })


def check_font_sizes(fig_w_in, target_w_cm, sizes, warn_pt=7.5):
    scale = fig_w_in / (target_w_cm / 2.54)
    bad = [f"  '{k}': {pt}pt -> {pt / scale:.1f}pt printed"
           for k, pt in sizes.items() if pt / scale < warn_pt]
    if bad:
        raise ValueError("Font sizes too small when printed:\n" + "\n".join(bad))


def panel_label(ax, letter, y_top=1.11):
    ax.text(0.0, y_top, f"({letter})", transform=ax.transAxes,
            fontsize=FS["panel"], fontweight="bold", va="top", ha="left")


def style_x(ax, xs, labels, bottom_row):
    ax.set_xticks(xs)
    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    if bottom_row:
        ax.set_xticklabels(labels, rotation=60, ha="right", rotation_mode="anchor")
    else:
        ax.tick_params(axis="x", length=0, labelbottom=False)


# ----------------------------------------------------------------------
def draw_key(ax, rows):
    """Name all series once, in panel (a), as in-axes artists (two columns)."""
    tr = ax.transAxes
    for r, items in enumerate(rows):
        y = 0.955 - r * 0.075
        for c, (kind, color, alpha, text) in enumerate(items):
            x = 0.03 + c * 0.52
            if kind == "line":
                ax.plot([x, x + 0.08], [y, y], transform=tr, color=color,
                        alpha=alpha, lw=1.4, clip_on=False)
            elif kind == "dash":
                ax.plot([x, x + 0.08], [y, y], transform=tr, color=color,
                        lw=0.8, ls="--", clip_on=False)
            else:
                ax.add_patch(plt.Rectangle((x, y - 0.022), 0.08, 0.044,
                                           transform=tr, color=color,
                                           alpha=alpha, lw=0, clip_on=False))
            ax.text(x + 0.10, y, text, transform=tr, fontsize=FS["annot"],
                    ha="left", va="center")


KEY_ROWS = [
    [("line", "black", 1.0, "Estimate ± SE"), ("line", C_PIXEL, 0.5, "Pixel count")],
    [("patch", C_CI95, 0.22, "95% CI"),        ("patch", C_CI99, 0.18, "99% CI")],
    [("dash", "black", 1.0, "Census area")],
]


def draw_area_panel(ax, df, area_col, se_col, prop_se_col, ylim):
    xs = df.t.to_numpy()
    est = df[area_col].to_numpy()
    se = df[se_col].to_numpy()

    ax.axhline(CENSUS_HA, color="black", lw=0.8, ls="--", zorder=2)
    ax.fill_between(xs, est - Z99 * se, est + Z99 * se, color=C_CI99,
                    alpha=0.18, lw=0)
    ax.fill_between(xs, est - Z95 * se, est + Z95 * se, color=C_CI95,
                    alpha=0.22, lw=0)
    ax.plot(xs, df.pixel_count_ha, color=C_PIXEL, alpha=0.5, lw=1.4,
            marker="o", ms=3.5, zorder=3)
    ax.errorbar(xs, est, yerr=se, fmt="o-", color="black", lw=1.4, ms=3.5,
                capsize=2.5, elinewidth=1.0, zorder=4)

    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.set_ylabel("Rapeseed area (ha)")

    # SE of the proportion (%), one value per acquisition, along the bottom
    span = ylim[1] - ylim[0]
    y_row = ylim[0] + 0.045 * span
    for x, s in zip(xs, df[prop_se_col].to_numpy()):
        ax.text(x, y_row, f"{100 * s:.2f}", fontsize=FS["annot"],
                ha="center", va="bottom", color="0.2")
    ax.text(xs[0] - 0.45, y_row + 0.075 * span, "SE of proportion (%)",
            fontsize=FS["annot"], ha="left", va="bottom", color="0.2")


def draw_accuracy_panel(ax, df):
    xs = df.t.to_numpy()
    w = 0.26
    for off, key in zip((-w, 0.0, w), ("F1", "UA", "PA")):
        half = Z95 * df[f"{key}_SE"]                 # half-width of the 95 % CI
        ax.bar(xs + off, df[key], width=w, color=C_BARS[key], alpha=0.75, lw=0)
        ax.errorbar(xs + off, df[key], yerr=half, fmt="none", ecolor="0.2",
                    elinewidth=0.8, capsize=1.8, capthick=0.8)
        for x, v, e in zip(xs + off, df[key], half):  # value above the error bar (vertical)
            ax.text(x, v + e + 0.015, f"{v:.3f} ± {e:.3f}", rotation=90, ha="center", va="bottom",
                    fontsize=FS["annot"], color="0.15")
    ax.set_ylim(0, 1.55)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Accuracy")
    for x, key, name in ((0.03, "F1", "F1-score"), (0.30, "UA", "UA"),
                         (0.46, "PA", "PA")):
        ax.add_patch(plt.Rectangle((x, 0.935), 0.035, 0.04, transform=ax.transAxes,
                                   color=C_BARS[key], alpha=0.75, lw=0))
        ax.text(x + 0.05, 0.955, name, transform=ax.transAxes,
                fontsize=FS["annot"], ha="left", va="center")


# ----------------------------------------------------------------------
def main():
    df = pd.read_csv(RESULTS_CSV).sort_values("t")
    labels = df.label.tolist()
    xs = df.t.to_numpy()

    # one y range for (a)-(b): every 99 % CI band and the pixel-count area,
    # mapped into [DATA_LO, DATA_HI] of the axes height
    lo, hi = [], []
    for _, _, a_col, s_col, _ in METHODS:
        lo.append((df[a_col] - Z99 * df[s_col]).min())
        hi.append((df[a_col] + Z99 * df[s_col]).max())
    lo.append(df.pixel_count_ha.min()); hi.append(df.pixel_count_ha.max())
    lo.append(CENSUS_HA); hi.append(CENSUS_HA)
    d_lo, d_hi = min(lo), max(hi)
    span = (d_hi - d_lo) / (DATA_HI - DATA_LO)
    ylim = (d_lo - DATA_LO * span, d_lo - DATA_LO * span + span)

    # ---- absolute layout (cm): one row, three columns; (b) shares the y axis of (a) ----
    LEFT, RIGHT, TOP, BOT = 1.75, 0.25, 0.85, 1.55
    PW_AREA, PW_ACC, PH = 4.4, 5.5, 5.0      # accuracy panel wider: 12 bars + vertical labels
    GAP_AB, GAP_BC = 0.30, 1.45
    fig_w_cm = LEFT + 2 * PW_AREA + GAP_AB + GAP_BC + PW_ACC + RIGHT
    fig_h_cm = TOP + PH + BOT
    fig_w_in, fig_h_in = fig_w_cm / 2.54, fig_h_cm / 2.54
    check_font_sizes(fig_w_in, TARGET_W_CM, FS)

    fx = lambda cm: cm / fig_w_cm
    fy = lambda cm: cm / fig_h_cm
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    y0, ph = fy(BOT), fy(PH)
    xa = LEFT
    xb = xa + PW_AREA + GAP_AB
    xc = xb + PW_AREA + GAP_BC
    ax_a = fig.add_axes([fx(xa), y0, fx(PW_AREA), ph])
    ax_b = fig.add_axes([fx(xb), y0, fx(PW_AREA), ph])
    ax_c = fig.add_axes([fx(xc), y0, fx(PW_ACC), ph])

    for ax, (letter, title, a_col, s_col, p_col) in zip((ax_a, ax_b), METHODS):
        draw_area_panel(ax, df, a_col, s_col, p_col, ylim)
        panel_label(ax, letter)
    ax_b.tick_params(axis="y", labelleft=False)
    ax_b.set_ylabel("")
    draw_key(ax_a, KEY_ROWS)
    draw_accuracy_panel(ax_c, df)
    panel_label(ax_c, "c")

    for ax in (ax_a, ax_b, ax_c):
        style_x(ax, xs, labels, bottom_row=True)

    # one shared x label, centred under the three panels, just below the lowest tick label
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    y_min = min(t.get_window_extent(rnd).y0
                for ax in (ax_a, ax_b, ax_c) for t in ax.get_xticklabels())
    y_lab = fig.transFigure.inverted().transform((0, y_min))[1] - fy(0.12)
    fig.text(fx((xa + xc + PW_ACC) / 2), y_lab, "Date", ha="center", va="top",
             fontsize=FS["label"], fontweight="bold")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PNG.name}   (shared y range {ylim[0]:,.0f} - {ylim[1]:,.0f} ha)")


if __name__ == "__main__":
    setup_rc()
    main()

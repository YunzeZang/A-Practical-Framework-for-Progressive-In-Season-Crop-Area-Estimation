"""Fig. C1: effect of the sample allocation on the within-stratum and combined ratio poststratified
estimators, with the total sample size fixed at n = 237 (M = 10,000 replicates per allocation).

Columns: Neyman, proportional and equal allocation.
Rows
    (a) sample units per replicate in the flipped substrata (0, 1) and (1, 0) and failure rate of
        the within-stratum estimator
    (b) estimated corn proportion of the county area (%)
    (c) estimated SE of the proportion (%), RMS of the estimated SEs and empirical SD
    (d) coverage of the nominal 95 % confidence interval
Symbols as in Fig. 5 (fig5_estimator_performance.py).

Reads   results/area_estimation/mc_allocation_replicates.npz   (03_mc_sample_allocation.py)
Writes  results/figures/FigC1_sample_allocation.png

Usage   python Story_County/figures/figC1_sample_allocation.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator, FixedLocator, NullLocator
from matplotlib.legend_handler import HandlerTuple
from matplotlib.transforms import blended_transform_factory

import plot_style as G
from plot_style import (BLUE, BW, C, DASH, EST_COL, FAIL_TOP, FLIP_CAP, FLIP_FLOOR,
                        FLIP_TOP, FS, GREY, LABEL_GAP, ORANGE, RED, TRUTH, Z95, box, fail_label, finite, series_x)

ALLOC_NAME = {"neyman": "Neyman", "proportional": "Proportional", "equal": "Equal"}


def main():
    G.setup_rc()
    z = np.load(C.AREA_DIR / "mc_allocation_replicates.npz")
    truth, county = float(z["true_area"]), float(z["county_area"])
    pct = 100.0 / county
    area, se, flips = z["area"] * pct, z["se"] * pct, z["flips"]
    dates = [str(d) for d in z["dates"]]
    labels = [G.date_label(d) for d in dates]
    ts = np.arange(len(dates))
    order = list(range(len(z["allocation"])))
    conf = C.CFG["confidence"]

    ncol = len(order)
    LEFT, RIGHT, BOT = 2.15, 1.75, 2.10
    PW, CGAP = 2.95, 0.55
    LEG, LEG_DY = 0.95, 0.04
    PH = 1.2 * PW
    rows = [("flips", PW), ("area", PH), ("se", PH), ("cov", PW)]
    HEAD = 1.05
    gap_above = {"flips": HEAD, "area": 0.45, "se": 0.45, "cov": 0.45}
    XL = [LEFT + j * (PW + CGAP) for j in range(ncol)]
    W = XL[-1] + PW + RIGHT
    H = BOT + sum(h + LEG + gap_above[k] for k, h in rows)
    fx, fy = (lambda c: c / W), (lambda c: c / H)
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    y = H
    ybox = {}
    for key, h in rows:
        y -= gap_above[key] + LEG
        ybox[key] = (y - h, h)
        y -= h

    def legend_above(key, handles, names, ncol_leg, x0=XL[0], x1=XL[-1] + PW, dy=LEG_DY):
        b, h = ybox[key]
        fig.legend(handles, names, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3)},
                   loc="lower center", bbox_to_anchor=(fx((x0 + x1) / 2), fy(b + h + dy)),
                   ncol=ncol_leg, frameon=False, fontsize=FS["annot"], handlelength=1.8,
                   handletextpad=0.5, columnspacing=1.4, borderaxespad=0)

    def row_axes(key):
        b, h = ybox[key]
        return [fig.add_axes([fx(XL[j]), fy(b), fx(PW), fy(h)]) for j in range(ncol)]

    # column headings above the legend of row (a)
    b_, h_ = ybox["flips"]
    y_head = b_ + h_ + LEG + 0.12
    for j, d in enumerate(order):
        nh = z["n_h"][d]
        fig.text(fx(XL[j] + PW / 2), fy(y_head + 0.42), f"{ALLOC_NAME[str(z['allocation'][d])]} allocation",
                 ha="center", va="bottom", fontsize=FS["label"], fontweight="bold")
        fig.text(fx(XL[j] + PW / 2), fy(y_head), f"n = {int(nh.sum())}  ({int(nh[0])} / {int(nh[1])})",
                 ha="center", va="bottom", fontsize=FS["annot"], color="0.35")

    # ------------------------------------------------------------------ (b) proportion, (c) SE
    axes_c, axes_d = row_axes("area"), row_axes("se")
    for j, d in enumerate(order):
        axc, axd = axes_c[j], axes_d[j]
        axc.axhline(truth * pct, **TRUTH)
        for t in ts:
            for e, x in enumerate(series_x(t)):
                if x is None:
                    continue
                col = GREY if t == 0 else EST_COL[e]
                a = finite(area[d, t, e])
                m, s = a.mean(), a.std(ddof=1)
                q = np.percentile(a, [2.5, 25, 75, 97.5])
                box(axc, x, q[1], q[2], q[0], q[3], m, col, "0.1")
                v = finite(se[d, t, e])
                q = np.percentile(v, [2.5, 25, 75, 97.5])
                box(axd, x, q[1], q[2], q[0], q[3], np.sqrt((v ** 2).mean()), col, "0.05", mid_lw=0.9)
                axd.plot([x - BW / 2, x + BW / 2], [s] * 2, color=RED, lw=0.9, zorder=6,
                         solid_capstyle="butt")
    for axs, key, ylab in ((axes_c, "area", "(b) Corn\nproportion (%)"), (axes_d, "se", "(c) SE of the\nproportion (%)")):
        lo = min(ax.dataLim.y0 for ax in axs)
        hi = max(ax.dataLim.y1 for ax in axs)
        pad = 0.04 * (hi - lo)
        lo = lo - pad if key == "area" else 0.0
        for j, ax in enumerate(axs):
            ax.set_ylim(lo, hi + pad)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            if j > 0:
                ax.tick_params(axis="y", labelleft=False)
        axs[0].set_ylabel(ylab, labelpad=3)
    est_h = [(Patch(facecolor=c, alpha=0.4, edgecolor=c), lab)
             for c, lab in ((BLUE, "Within-stratum"), (ORANGE, "Combined ratio"), (GREY, "Stratified, initial map only"))]
    legend_above("area", [h_ for h_, _ in est_h] + [Line2D([0], [0], **{k: v for k, v in TRUTH.items() if k != "zorder"})],
                 [l_ for _, l_ in est_h] + ["True proportion (CDL)"], 2)
    legend_above("se", [Line2D([0], [0], color="0.05", lw=0.9), Line2D([0], [0], color=RED, lw=0.9)],
                 ["RMS of estimated SE", "Empirical SD"], 2)

    # ------------------------------------------------------------------ (d) coverage
    axes_e = row_axes("cov")
    for j, d in enumerate(order):
        ax = axes_e[j]
        ax.axhline(100 * conf, **DASH)
        for e in (0, 1):
            cov = []
            for t in ts:
                a, s = area[d, t, e], se[d, t, e]
                u = np.isfinite(a)
                cov.append(np.mean(np.abs(a[u] - truth * pct) <= Z95 * s[u]) if u.sum() else np.nan)
            cov = 100 * np.array(cov)
            if e == 0:
                ax.plot(0, cov[0], ls="none", marker="*", ms=6.5, color=GREY, zorder=5)
            ax.plot(ts[1:], cov[1:], color=EST_COL[e], lw=1.2, marker="oD"[e], ms=3.0)
    lo = min(ax.get_ylim()[0] for ax in axes_e); hi = max(ax.get_ylim()[1] for ax in axes_e)
    for j, ax in enumerate(axes_e):
        ax.set_ylim(min(lo, 93.5), max(hi, 96.5))
        ax.yaxis.set_major_locator(FixedLocator([94, 95, 96]))
        if j > 0:
            ax.tick_params(axis="y", labelleft=False)
    axes_e[0].set_ylabel(f"(d) {conf:.0%} CI\ncoverage (%)", labelpad=3)
    legend_above("cov", [Line2D([0], [0], color=BLUE, marker="o", lw=1.2, ms=3),
                         Line2D([0], [0], color=ORANGE, marker="D", lw=1.2, ms=3),
                         Line2D([0], [0], ls="none", marker="*", ms=6.5, color=GREY),
                         Line2D([0], [0], **{k: v for k, v in DASH.items() if k != "zorder"})],
                 ["Within-stratum", "Combined ratio", "Stratified, initial map only", "Nominal level"], 2)

    # ------------------------------------------------------------------ (a) flips and failure rate
    axes_b = row_axes("flips")
    for j, d in enumerate(order):
        ax = axes_b[j]
        x = ts[1:]
        ax.set_yscale("log")
        ax.set_ylim(FLIP_FLOOR, FLIP_TOP)
        for c, (dx, fc, ec) in enumerate(((-G.BAR_W / 2, "white", "0.25"), (G.BAR_W / 2, G.BAR_10, "0.45"))):
            f = flips[d, x, c].astype(float)
            mean = f.mean(axis=1)
            q = np.percentile(f, [2.5, 97.5], axis=1)
            top = np.minimum(mean, FLIP_CAP)
            ax.bar(x + dx, top - FLIP_FLOOR, width=G.BAR_W, bottom=FLIP_FLOOR, color=fc,
                   edgecolor=ec, linewidth=0.6, zorder=2)
            show = q[0] < FLIP_CAP
            lo_ = np.clip(q[0], FLIP_FLOOR, FLIP_CAP)
            hi_ = np.clip(q[1], FLIP_FLOOR, FLIP_CAP)
            ax.errorbar((x + dx)[show], top[show], yerr=[(top - lo_)[show], (hi_ - top)[show]],
                        fmt="none", ecolor="0.1", elinewidth=0.6, capsize=1.3, capthick=0.6, zorder=3)
        ax.yaxis.set_major_locator(FixedLocator([FLIP_FLOOR, 1, 10, FLIP_CAP]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda v, _: {FLIP_FLOOR: "0", 1: r"$10^0$", 10: r"$10^1$", FLIP_CAP: r">$\,10^2$"}.get(v, "")))
        if j > 0:
            ax.tick_params(axis="y", labelleft=False)
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(True)
        fr = 100 * np.isnan(z["area"][d, 1:, 0]).mean(axis=1)
        ax2.plot(x, fr, color=BLUE, lw=1.2, marker="o", ms=3.0, zorder=4)
        ax2.set_ylim(-4, FAIL_TOP)
        ax2.set_yticks([0, 50, 100])
        ax2.tick_params(axis="x", length=0, labelbottom=False)
        ax2.tick_params(axis="y", labelright=(j == ncol - 1))
        txt = [fail_label(v) for v in fr]
        stagger = any(len(s_) > 2 for s_ in txt)
        for k, (t, s_) in enumerate(zip(x, txt)):
            yy = 0.985 - (0.115 if stagger and k % 2 else 0.0)
            ax.text(t, yy, s_, transform=ax.get_xaxis_transform(), ha="center",
                    va="top", fontsize=FS["small"], color=BLUE)
        if j == ncol - 1:
            ax2.set_ylabel("Failure\nrate (%)", labelpad=4)
    axes_b[0].set_ylabel("(a) Flipped\nsample units", labelpad=3)
    legend_above("flips", [Patch(facecolor="white", edgecolor="0.25", lw=0.6),
                           Patch(facecolor=G.BAR_10, edgecolor="0.45", lw=0.6),
                           Line2D([0], [0], color=BLUE, marker="o", lw=1.2, ms=3)],
                 [r"Flips $0\rightarrow1$", r"Flips $1\rightarrow0$", "Failure rate of within-stratum"], 2)

    # ------------------------------------------------------------------ shared x axes
    for i, axs in enumerate((axes_b, axes_c, axes_d, axes_e)):
        for j, ax in enumerate(axs):
            ax.set_xticks(ts)
            ax.set_xlim(-0.5, ts[-1] + 0.5)
            if i == 3:
                ax.set_xticklabels(labels, rotation=60, ha="right", rotation_mode="anchor")
                ax.set_xlabel("Date", labelpad=2)
            else:
                ax.tick_params(axis="x", length=0, labelbottom=False)

    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    x_tick = min(inv.transform(t.get_window_extent())[0, 0]
                 for ax in (axes_b[0], axes_c[0], axes_d[0], axes_e[0]) for t in ax.get_yticklabels()
                 if t.get_visible() and t.get_text())
    for ax in (axes_b[0], axes_c[0], axes_d[0], axes_e[0]):
        ax.yaxis.set_label_coords(x_tick - fx(LABEL_GAP), 0.5,
                                  transform=blended_transform_factory(fig.transFigure, ax.transAxes))

    out = C.FIGURE_DIR / "FigC1_sample_allocation.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {Path(out).name}  ({W:.1f} x {H:.1f} cm)")


if __name__ == "__main__":
    main()

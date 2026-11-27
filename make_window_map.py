#!/usr/bin/env python3
"""
Where the ribbons live in the window (ROADMAP Stage 9z): two figures, written to figures/.

Input: dumps from generic_phase.py DUMP=... (they hold, for every tile, its outline, its window point in Z_0(S) and
the phason steps that remove it).

  ribbon_map_phases.png   the heat map of flips (number of step directions that remove a tile) for each phase.
  ribbon_map_window.png   the same tiles drawn in the window: every tile is a point of Z_0(O) or Z_0(G), coloured by
                          the same count. A tile is removed by a step when its window point leaves the window, so the
                          flipped tiles outline the window's boundary, and a tile removed by many directions sits where
                          the boundary surrounds it on many sides.

Test of the hub idea: the "fill" of a tile is the number of tiles of the same shape whose window point lies within EPS
of its own, divided by the median over all tiles. Deep inside the window the fill is about 1, at a straight piece of
boundary about 1/2, at a tip less. Reported: mean fill by number of removing directions.

Usage: python3 make_window_map.py dump1.json [dump2.json ...] [--tests]
"""
import json
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
EPS = 0.07
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titlelocation": "left", "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def load(path):
    d = json.load(open(path))[0]
    n = len(d["tiles"])
    count = np.zeros(n, int)
    for st in d["steps"]:
        count[st["removed"]] += 1
    shape = np.array([w[0] for w in d["window"]])
    z = np.array([[w[1], w[2]] for w in d["window"]])
    fill = np.zeros(n)
    for S in "OG":
        m = np.where(shape == S)[0]
        tr = cKDTree(z[m])
        c = np.array([len(x) - 1 for x in tr.query_ball_point(z[m], EPS)], float)
        fill[m] = c / np.median(c)
    return d, count, shape, z, fill


def bare(ax):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    data = [load(f) for f in files]
    nsteps = len(data[0][0]["steps"])
    cmap = plt.get_cmap("inferno_r", nsteps)
    norm = BoundaryNorm(np.arange(0.5, nsteps + 1.5), nsteps)
    out = dict(eps=EPS, phases=[])
    # ---- figure 1: heat maps in the plane ----
    fig, axs = plt.subplots(1, len(data), figsize=(7.2, 2.9), layout="constrained")
    axs = np.atleast_1d(axs)
    for ax, (d, count, shape, z, fill), f in zip(axs, data, files):
        polys = [np.array(p) for pl in d["tiles"] for p in pl[:1]]            # one outline per key (a Mystic is one 20-gon)
        cen = np.array([p.mean(0) for p in polys])
        inner = np.linalg.norm(cen, axis=1) < d["R"] - 6
        cold = [i for i in range(len(polys)) if inner[i] and count[i] == 0]
        hot = [i for i in range(len(polys)) if inner[i] and count[i] > 0]
        ax.add_collection(PolyCollection([polys[i] for i in cold], facecolors="#efefef", edgecolors="none"))
        pc = PolyCollection([polys[i] for i in hot], array=count[hot], cmap=cmap, norm=norm, edgecolors="none")
        ax.add_collection(pc); ax.autoscale_view(); bare(ax)
        top = np.where(count == count.max())[0]
        ax.scatter(cen[top, 0], cen[top, 1], s=60, facecolors="none", edgecolors="k", linewidths=0.8, zorder=5)
        ax.set_title("%s  phase %d" % ("abc"[len(out["phases"])], len(out["phases"]) + 1))
        out["phases"].append(dict(file=os.path.basename(f), keys=len(polys), y0=d["y0"], removed_by={str(k): int(v) for k, v in sorted(Counter(count[count > 0].tolist()).items())},
                                  max_count=int(count.max()), tiles_at_max=int(len(top)), shapes_at_max=dict(Counter(shape[top].tolist())),
                                  fill_by_count={str(k): [float(fill[count == k].mean()), int((count == k).sum())] for k in sorted(set(count.tolist()))},
                                  fill_at_max=[float(x) for x in fill[top]]))
    cb = fig.colorbar(pc, ax=axs, orientation="horizontal", fraction=0.05, pad=0.03, shrink=0.5, ticks=range(1, nsteps + 1))
    cb.set_label("phason step directions (of %d) that remove the tile; circles: the most-removed tiles" % nsteps)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "ribbon_map_phases.png")); plt.close(fig)
    # ---- figure 2: the same tiles in the window ----
    fig, axs = plt.subplots(2, len(data), figsize=(7.2, 5.2), layout="constrained", squeeze=False)
    for j, (d, count, shape, z, fill) in enumerate(data):
        for i, S in enumerate("OG"):
            ax = axs[i, j]
            m = shape == S
            c0 = m & (count == 0); c1 = m & (count > 0)
            ax.scatter(z[c0, 0], z[c0, 1], s=1.2, c="#d0d0d0", linewidths=0, rasterized=True)
            order = np.argsort(count[c1])
            sc = ax.scatter(z[c1, 0][order], z[c1, 1][order], s=5, c=count[c1][order], cmap=cmap, norm=norm, linewidths=0)
            top = np.where(m & (count == count[m].max()) & (count > 0))[0]
            ax.scatter(z[top, 0], z[top, 1], s=60, facecolors="none", edgecolors="k", linewidths=0.8, zorder=5)
            bare(ax)
            ax.set_title("%s  %s, phase %d" % ("abcdef"[i * len(data) + j], "single tiles: window $Z_0(O)$" if S == "O" else "Mystics: window $Z_0(G)$", j + 1))
    cb = fig.colorbar(sc, ax=axs, orientation="horizontal", fraction=0.04, pad=0.02, shrink=0.5, ticks=range(1, nsteps + 1))
    cb.set_label("phason step directions (of %d) that remove the tile (grey: none)" % nsteps)
    fig.savefig(os.path.join(FIG, "ribbon_map_window.png")); plt.close(fig)
    # ---- pooled fill by count ----
    allc = np.concatenate([x[1] for x in data]); allf = np.concatenate([x[4] for x in data])
    out["fill_by_count_pooled"] = {str(k): dict(mean=float(allf[allc == k].mean()), n=int((allc == k).sum())) for k in sorted(set(allc.tolist()))}
    for p in out["phases"]:
        print("%s: removed by k directions %s; most-removed: %d tiles at %d (%s), fill %s" % (p["file"], p["removed_by"], p["tiles_at_max"], p["max_count"], p["shapes_at_max"], ["%.2f" % x for x in p["fill_at_max"]]))
    print("mean fill by number of removing directions (pooled):", {k: "%.2f (n=%d)" % (v["mean"], v["n"]) for k, v in out["fill_by_count_pooled"].items()})
    json.dump(out, open(os.path.join(HERE, "make_window_map.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        f = out["fill_by_count_pooled"]
        assert f["0"]["mean"] > 0.9 and all(f[k]["mean"] < f["0"]["mean"] for k in f if k != "0")
        print("make_window_map: tests passed (flipped tiles have lower fill than unflipped ones: they sit at the window boundary)")


if __name__ == "__main__":
    main()

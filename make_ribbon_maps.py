#!/usr/bin/env python3
"""
Maps of the ribbon model (ROADMAP Stage 9y): two figures, written to figures/.

  ribbon_map_forest.png   a level-5 substitution patch.
        a  the forest of possible active ribbons (ribbon_trees.py): every link, coloured by the size of its tree,
           with width growing with the number of Mystics downstream; forks, roots and the links that the patch
           boundary cuts are marked.
        b  the same forest as a heat map on the tiles: each tile takes the downstream count of the chain Mystic of
           its level-1 supertile ("flow"); tiles whose supertile is not on a chain stay grey.
  ribbon_map_heat.png     a generic-phase patch (needs a dump from generic_phase.py DUMP=...).
        a  heat map of flips: in how many of the phason steps of the dump each tile is removed;
        b  the flipped tiles by kind: in a stack of flipped tiles (silent), at a pair site (active), other.

Usage: python3 make_ribbon_maps.py [dump.json] [--tests]
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import LogNorm, BoundaryNorm, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recognisability_radius as RR  # noqa: E402
import ribbon_trees as RT  # noqa: E402
import tile_stacks as TS  # noqa: E402

REF = TS.REF
FIG = os.path.join(HERE, "figures")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titlelocation": "left", "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "legend.frameon": False})
SIZE_COL = {11: "#9ecae1", 30: "#4292c6", 210: "#d95f02"}          # complete trees; larger ones (cut by the boundary) in red
BIG = "#b2182b"


def bare(ax):
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def forest_figure(level=5):
    levels, leaves = RR.build(level)
    allP = [REF @ T[:, :2].T + T[:, 2] for T, _, _ in leaves]
    G = [i for i, (T, lab, b) in enumerate(leaves) if lab == "Gamma1"]
    P = np.array([leaves[i][0][:, 2] for i in G]); L = np.array([leaves[i][0][:, :2] for i in G])
    cen = np.array([allP[i].mean(0) for i in G])
    sup1 = np.array([levels[0][leaves[i][2]][2] for i in G])            # level-1 supertile of each Mystic
    lab1 = [levels[1][a][1] for a in sup1]
    succ, pred = RT.links(P, L)
    n = len(G)
    down = np.zeros(n, int); tree_of = -np.ones(n, int); tsize = {}
    roots = [u for u in succ if u not in pred]
    for r in roots:
        order, stack = [], [r]
        while stack:
            x = stack.pop(); order.append(x); stack += list(succ.get(x, {}).values())
        for x in reversed(order):
            down[x] = 1 + sum(down[y] for y in succ.get(x, {}).values())
        for x in order:
            tree_of[x] = r
        tsize[r] = len(order)
    cut_in = [u for u in roots if lab1[u] not in ("Lambda", "Pi")]      # a root whose label says it has a predecessor
    cut_out = [u for u in pred if u not in succ and lab1[u] not in ("Psi", "Xi")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.6), layout="constrained")
    # a: the forest
    ax = axs[0]
    ax.add_collection(PolyCollection(allP, facecolors="#f3f3f3", edgecolors="none"))
    segs, cols, wid = [], [], []
    for u in succ:
        for k, v in succ[u].items():
            segs.append([cen[u], cen[v]]); cols.append(SIZE_COL.get(tsize[tree_of[u]], BIG)); wid.append(0.35 + 0.22 * math.sqrt(down[v]))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=wid, capstyle="round"))
    fk = [u for u in succ if len(succ[u]) == 2]
    ax.scatter(cen[fk, 0], cen[fk, 1], s=2.2, c="k", linewidths=0, zorder=3)
    ax.scatter(cen[roots, 0], cen[roots, 1], s=7, facecolors="white", edgecolors="k", linewidths=0.5, zorder=4)
    for u, mk in [(u, "v") for u in cut_in] + [(u, "^") for u in cut_out]:
        ax.scatter(*cen[u], s=42, marker=mk, facecolors="#ffd92f", edgecolors="k", linewidths=0.6, zorder=5)
    ax.autoscale_view(); bare(ax)
    ax.set_title("a  Possible active ribbons: a forest")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=SIZE_COL[11], lw=1.4, label="trees of 11 Mystics"), Line2D([], [], color=SIZE_COL[30], lw=1.4, label="of 30"),
          Line2D([], [], color=SIZE_COL[210], lw=1.4, label="of 210"), Line2D([], [], color=BIG, lw=1.4, label="largest"),
          Line2D([], [], marker="o", ls="", mfc="white", mec="k", ms=3.5, label="root"), Line2D([], [], marker="o", ls="", color="k", ms=2, label="fork"),
          Line2D([], [], marker="v", ls="", mfc="#ffd92f", mec="k", ms=5, label="cut by the boundary")]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=7, handlelength=1.4, columnspacing=1.0)
    # b: flow as a heat map on the tiles
    ax = axs[1]
    flow_of_sup = {int(sup1[u]): int(down[u]) for u in range(n) if tree_of[u] >= 0}
    val = np.array([flow_of_sup.get(int(levels[0][b][2]), 0) for _, _, b in leaves])
    on = val > 0
    ax.add_collection(PolyCollection([allP[i] for i in np.where(~on)[0]], facecolors="#e9e9e9", edgecolors="none"))
    pc = PolyCollection([allP[i] for i in np.where(on)[0]], array=val[on], cmap="magma_r", norm=LogNorm(1, val.max()), edgecolors="none")
    ax.add_collection(pc)
    ax.autoscale_view(); bare(ax)
    ax.set_title("b  Flow: Mystics downstream of each supertile")
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, shrink=0.85)
    cb.set_label("Mystics downstream on the chain (grey: not on a chain)")
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "ribbon_map_forest.png")); plt.close(fig)
    return dict(level=level, mystics=n, links=len(segs), trees=dict(sorted(Counter(tsize.values()).items())), forks=len(fk), roots=len(roots),
                cut_in=len(cut_in), cut_out=len(cut_out), tiles_on_chains=int(on.sum()), tiles=len(leaves), max_flow=int(val.max()))


def heat_figure(path):
    import stack_trace as ST
    dump = json.load(open(path))[0]
    d = ST.prepare(path)
    P, L, lab, nb = d["P"], d["L"], d["lab"], d["nb"]
    n = len(P)
    polys = [REF @ L[u].T + P[u] for u in range(n)]
    sp, u = defaultdict(list), 0
    for a, pl in enumerate(dump["tiles"]):
        for Q in pl:
            for _ in range(1 if len(Q) == 14 else 2):
                sp[a].append(u); u += 1
    count = np.zeros(n, int)
    for st in dump["steps"]:
        for a in st["removed"]:
            count[sp[a]] += 1
    nsteps = len(dump["steps"])
    flipped = np.where(count > 0)[0]
    fs = set(flipped.tolist())
    stacked = {t for s, _ in TS.stacks(P, L, flipped) for t in s}
    paired = ST.pair_tiles(d)
    kind = {t: ("silent" if t in stacked else "active" if t in paired else "other") for t in flipped}
    inner = np.linalg.norm(P, axis=1) < d["R"] - 6
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 4.4), layout="constrained")
    ax = axs[0]
    show = np.where(inner)[0]
    ax.add_collection(PolyCollection([polys[t] for t in show if count[t] == 0], facecolors="#efefef", edgecolors="none"))
    hot = [t for t in show if count[t] > 0]
    cmap = plt.get_cmap("inferno_r", nsteps)
    pc = PolyCollection([polys[t] for t in hot], array=count[hot], cmap=cmap, norm=BoundaryNorm(np.arange(0.5, nsteps + 1.5), nsteps), edgecolors="none")
    ax.add_collection(pc); ax.autoscale_view(); bare(ax)
    ax.set_title("a  Heat map of flips at one phase")
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, shrink=0.85, ticks=range(1, nsteps + 1))
    cb.set_label("phason step directions (of %d) that remove the tile" % nsteps)
    ax = axs[1]
    kc = {"silent": "#1f77b4", "active": "#d95f02", "other": "#9a9a9a"}
    ax.add_collection(PolyCollection([polys[t] for t in show if count[t] == 0], facecolors=["#f6e3cf" if t in paired else "#efefef" for t in show if count[t] == 0], edgecolors="none"))
    ax.add_collection(PolyCollection([polys[t] for t in hot], facecolors=[kc[kind[t]] for t in hot], edgecolors="none"))
    ax.autoscale_view(); bare(ax)
    ax.set_title("b  The flipped tiles by kind")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=kc["silent"], label="in a stack of flipped tiles (silent)"), Patch(color=kc["active"], label="at a pair site (active)"),
                       Patch(color=kc["other"], label="other flipped tile"), Patch(color="#f6e3cf", label="pair site, not flipped")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=7, handlelength=1.2, columnspacing=1.0)
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, "ribbon_map_heat.png")); plt.close(fig)
    kc_ = Counter(kind.values())
    hub = np.where(count == count.max())[0]
    return dict(file=os.path.basename(path), tiles=n, steps=nsteps, flipped=len(flipped), flipped_share=len(flipped) / n,
                by_count={str(k): int(v) for k, v in sorted(Counter(count[flipped].tolist()).items())}, kinds=dict(kc_),
                pair_site_share_all=len(paired) / n, pair_site_share_flipped=len(paired & fs) / len(fs),
                hottest=dict(steps=int(count.max()), tiles=int(len(hub)), centre=[float(x) for x in P[hub].mean(0)], labels=dict(Counter(lab[t] for t in hub))))


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    out = dict(forest=forest_figure())
    print("forest:", out["forest"])
    if files:
        out["heat"] = heat_figure(files[0])
        print("heat:", out["heat"])
    json.dump(out, open(os.path.join(HERE, "make_ribbon_maps.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        f = out["forest"]
        assert f["links"] == f["mystics"] - f["roots"] - (f["mystics"] - sum(k * v for k, v in f["trees"].items()))      # a forest: links = nodes on trees - trees
        print("make_ribbon_maps: tests passed (the links form a forest)")


if __name__ == "__main__":
    main()

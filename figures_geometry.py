"""
figures_geometry.py - two paper figures from the exact patches.
  fig_nesting.png    (a) the level-4 patch (4401 tiles) coloured by the nested levels 1..4 (9 < 71 < 559 < 4401);
                     (b) the 16-pair cluster of Result res:hopf-tiers: pairs of tier 2 (6), new in tier 3 (5), new in tier 4 (5),
                         with the dipole couplings between pair centres (width ~ |W|, solid W>0, dashed W<0).
  fig_window.png     largest pairwise concurrence along the Spectre-Hat flow for 6, 10, 12 central pairs of level 3.
Run:  python3 figures_geometry.py [outdir]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

import pair_hamiltonian as ph
import nested_tiers as nt
from quantumspectremodel import build_substitution_tier, tile_polygon_at_b

SURF, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#e9e8e4", "#9ec5f4", "#3987e5", "#0d366b"]       # level 4 (outer) ... level 1 (core)


def style(ax):
    ax.set_aspect("equal"); ax.axis("off"); ax.set_facecolor(SURF)


def nesting(out):
    b = 1.0
    patch = build_substitution_tier(4)
    polys = [np.array(tile_polygon_at_b(t, b)) for t in patch.tiles]
    level = np.zeros(len(polys), int)                      # 0 = new at level 4
    for k, lv in ((3, 1), (2, 2), (1, 3)):
        level[nt.embedding_to_top(k, 4)] = lv
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.2), facecolor=SURF, gridspec_kw=dict(width_ratios=[1, 1.05]))
    ax = axs[0]; style(ax)
    for lv in range(4):
        idx = np.where(level == lv)[0]
        pc = PatchCollection([Polygon(polys[i], closed=True) for i in idx], facecolor=RAMP[lv], edgecolor=SURF, linewidth=0.15)
        ax.add_collection(pc)
    allp = np.vstack(polys); ax.set_xlim(allp[:, 0].min(), allp[:, 0].max()); ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    c = np.mean(np.vstack([polys[i] for i in np.where(level == 3)[0]]), axis=0)
    ax.set_title("(a) level-4 patch, 4401 tiles, nested levels", color=INK, fontsize=10, loc="left")
    names = ["new at level 4", "level 3 (559)", "level 2 (71)", "level 1 (9)"]
    for lv, n in enumerate(names):
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=RAMP[lv], label=n))
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK2)
    axa = ax
    # (b) cluster
    fp = ph.flow_parameters(0.2)
    W, ranges, order = nt.nested_cluster((6, 5, 5), 1.0)
    shellcol = [BLUE, ORANGE, AQUA]
    ax = axs[1]; style(ax)
    pc = PatchCollection([Polygon(p, closed=True) for p in polys], facecolor="#efeeea", edgecolor=SURF, linewidth=0.3)
    ax.add_collection(pc)
    cent = []
    for s, r in enumerate(ranges):
        for q in r:
            i, j = order[q]
            for t in (i, j):
                ax.add_patch(Polygon(polys[t], closed=True, facecolor=shellcol[s], alpha=0.85, edgecolor=SURF, linewidth=0.4))
            cent.append(0.5 * (polys[i].mean(0) + polys[j].mean(0)))
    cent = np.array(cent)
    wmax = np.abs(W).max()
    for p in range(len(cent)):
        for q in range(p + 1, len(cent)):
            if abs(W[p, q]) > 0.12 * wmax:
                ax.plot(*zip(cent[p], cent[q]), color=INK, lw=2.2 * abs(W[p, q]) / wmax, ls="-" if W[p, q] > 0 else (0, (3, 2)), alpha=0.75, zorder=5)
    ax.scatter(cent[:, 0], cent[:, 1], s=14, color=INK, zorder=6)
    pad = 5.0
    ax.set_xlim(cent[:, 0].min() - pad, cent[:, 0].max() + pad); ax.set_ylim(cent[:, 1].min() - pad, cent[:, 1].max() + pad)
    ax.set_title("(b) 16 pair qubits from the nesting: shells of 6, 5, 5 pairs", color=INK, fontsize=10, loc="left")
    for col, n in zip(shellcol, ("shell 1: pairs of level 2", "shell 2: new at level 3", "shell 3: new at level 4")):
        ax.add_patch(plt.Rectangle((0, 0), 0, 0, facecolor=col, label=n))
    ax.plot([], [], color=INK, lw=1.5, label="dipole coupling W>0"); ax.plot([], [], color=INK, lw=1.5, ls=(0, (3, 2)), label="W<0 (width ~ |W|)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=8, labelcolor=INK2)
    axa.add_patch(plt.Rectangle((cent[:, 0].min() - pad, cent[:, 1].min() - pad), np.ptp(cent[:, 0]) + 2 * pad, np.ptp(cent[:, 1]) + 2 * pad, fill=False, edgecolor=ORANGE, lw=1.2))
    fig.tight_layout(); fig.savefig(out + "/fig_nesting.png", dpi=170, facecolor=SURF); plt.close(fig)


def window(out):
    taus = np.linspace(0.0, 0.4, 41)
    fig, ax = plt.subplots(figsize=(5.8, 3.6), facecolor=SURF)
    ax.set_facecolor(SURF)
    for n, col in ((6, BLUE), (10, ORANGE), (12, AQUA)):
        y = []
        for tau in taus:
            fp = ph.flow_parameters(float(tau)); Wm = fp.kappa * ph.central_couplings(3, n, fp.b)
            _, psi = ph.ground_state(Wm / fp.kappa, fp.J, fp.kappa)
            y.append(np.max(ph.pairwise_concurrence(psi, n)))
        ax.plot(taus, y, color=col, lw=2, label=f"{n} pairs")
        k = int(np.argmax(y)); ax.annotate(f"{n}", (taus[k], y[k]), textcoords="offset points", xytext=(0, 6), ha="center", color=INK2, fontsize=8)
    ax.set_xlabel("flow time τ (0 = Spectre, 1 = Hat)", color=INK2); ax.set_ylabel("largest pairwise concurrence", color=INK2)
    ax.set_title("Entanglement window crossed by the flow (level 3, central pairs)", color=INK, fontsize=10, loc="left")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2); ax.grid(color="#e6e5e0", lw=0.6); ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout(); fig.savefig(out + "/fig_window.png", dpi=170, facecolor=SURF); plt.close(fig)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    nesting(out); window(out)

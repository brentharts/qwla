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


if __name__ == "__main__" and not (len(sys.argv) > 2 and sys.argv[2] == "more"):
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    nesting(out); window(out)


# ---------------------------------------------------------------------------------------------------------------
# figures 3-5: Hopf structure, depth profile of the local temperature, relative entropy

def hopf(out):
    import hopf_pairs as hp
    fig = plt.figure(figsize=(10.5, 4.2), facecolor=SURF)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.1, 1.6], wspace=0.25)
    # (a) schematic: Bloch sphere, a point, its fibre
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURF); ax.set_aspect("equal"); ax.axis("off")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color=INK2, lw=1.2)
    ax.plot(np.cos(th), 0.28 * np.sin(th), color="#c9c8c2", lw=0.9, ls=(0, (3, 2)))
    ax.plot([0, 0], [-1, 1], color="#c9c8c2", lw=0.9)
    p = np.array([0.55, 0.62]); ax.scatter(*p, s=36, color=BLUE, zorder=5)
    ax.text(p[0] + 0.07, p[1] + 0.08, "$\\vec r$", color=BLUE, fontsize=10)
    ax.text(0, -1.25, "$S^2$: states of one qubit", ha="center", color=INK2, fontsize=8)
    c0 = np.array([2.0, 0.55]); ax.add_patch(plt.Circle(c0, 0.42, fill=False, edgecolor=ORANGE, lw=2.0))
    ax.annotate("", c0 + np.array([-0.42, 0]), p + np.array([0.03, 0]), arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.0, ls=(0, (3, 2))))
    ax.text(c0[0], c0[1] - 0.95, "fibre over $\\vec r$:\nphases $e^{i\\varphi}|\\psi\\rangle$\n($S^1\\subset S^3$)", ha="center", color=INK2, fontsize=8)
    ax.set_xlim(-1.2, 2.7); ax.set_ylim(-1.9, 1.35)
    ax.set_title("(a) one qubit: $S^1\\to S^3\\to S^2$", color=INK, fontsize=10, loc="left")
    # (b) schematic bar: pure two-qubit state  |r|^2 + C^2 = 1 ; mixed pair adds D
    ax = fig.add_subplot(gs[1]); ax.set_facecolor(SURF)
    ax.barh([2], [0.62], color=BLUE, height=0.5); ax.barh([2], [0.38], left=[0.62], color=ORANGE, height=0.5)
    ax.barh([1], [0.35], color=BLUE, height=0.5); ax.barh([1], [0.25], left=[0.35], color=ORANGE, height=0.5); ax.barh([1], [0.40], left=[0.60], color="#b9b8b2", height=0.5)
    ax.set_yticks([2, 1]); ax.set_yticklabels(["pure pair\n$S^7\\to S^4$", "pair inside a\nlarger state"], fontsize=8, color=INK2)
    ax.set_xlim(0, 1); ax.set_xticks([0, 1]); ax.set_ylim(0.4, 2.7)
    ax.text(0.31, 2, "$|\\vec r_1|^2$", ha="center", va="center", color="white", fontsize=9)
    ax.text(0.81, 2, "$C^2$", ha="center", va="center", color="white", fontsize=9)
    ax.text(0.175, 1, "$|\\vec r_p|^2$", ha="center", va="center", color="white", fontsize=9)
    ax.text(0.475, 1, "$C^2$", ha="center", va="center", color="white", fontsize=9)
    ax.text(0.80, 1, "$D$ (leak)", ha="center", va="center", color=INK, fontsize=9)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c2"); ax.tick_params(colors=INK2, left=False)
    ax.set_title("(b) unit budget on $S^4$ (schematic)", color=INK, fontsize=10, loc="left")
    # (c) data: strongest-partner pairs of 8 central pairs of level 3
    ax = fig.add_subplot(gs[2]); ax.set_facecolor(SURF)
    xpos, labels, off = [], [], 0
    for tau in (0.05, 0.2):
        fp = ph.flow_parameters(tau); Wm = ph.central_couplings(3, 8, fp.b); _, psi = ph.ground_state(Wm, fp.J, fp.kappa)
        pairs = set()
        for q in range(8):
            r = max((x for x in range(8) if x != q), key=lambda x: abs(Wm[q, x])); pairs.add((min(q, r), max(q, r)))
        for k, (a, b2) in enumerate(sorted(pairs)):
            d = hp.leak(ph.reduced_two_qubit(psi, 8, a, b2)); x = off + k
            ax.bar(x, d["bloch_len2"], color=BLUE, width=0.7); ax.bar(x, d["C"] ** 2, bottom=d["bloch_len2"], color=ORANGE, width=0.7)
            ax.bar(x, d["D"], bottom=d["bloch_len2"] + d["C"] ** 2, color="#b9b8b2", width=0.7)
        ax.text(off + (len(pairs) - 1) / 2, 1.04, f"flow $\\tau$ = {tau}", ha="center", color=INK2, fontsize=8)
        off += len(pairs) + 1
    ax.set_ylim(0, 1.12); ax.set_xticks([]); ax.set_ylabel("share of the unit budget", color=INK2, fontsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2)
    ax.set_title("(c) 8 central pairs, strongest partner", color=INK, fontsize=10, loc="left")
    fig.savefig(out + "/fig_hopf.png", dpi=170, facecolor=SURF, bbox_inches="tight"); plt.close(fig)


def depth(out):
    import relative_entropy as rel
    from scipy.stats import spearmanr
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.9), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.1, 1]))
    ax = axs[0]; ax.set_facecolor(SURF)
    data = {}
    for n, col in ((12, BLUE), (16, ORANGE)):
        sizes = list(range(3, n // 2 + 2 if n == 12 else n // 2 + 3))
        X, _ = rel.weight_profiles(3, n, 0.05, sizes)
        data[n] = X
        ms = sorted(set(X[:, 0].astype(int)))
        mean = [X[X[:, 0] == m, 4].mean() for m in ms]
        sd = [X[X[:, 0] == m, 4].std() for m in ms]
        ax.errorbar(np.array(ms) / n, mean, yerr=sd, color=col, lw=1.8, marker="o", ms=4, capsize=2, label=f"{n} pairs")
    ax.axhline(0, color="#c9c8c2", lw=0.8); ax.axvline(0.5, color="#c9c8c2", lw=0.8, ls=(0, (3, 2)))
    ax.text(0.505, ax.get_ylim()[1] * 0.9, "half the system", color=INK2, fontsize=8)
    ax.set_xlabel("region size / system size  $m/n$", color=INK2); ax.set_ylabel("mean local inverse temperature $\\beta_{loc}$", color=INK2)
    ax.set_title("(a) $\\beta_{loc}$ changes sign near half the system", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower left")
    ax = axs[1]; ax.set_facecolor(SURF)
    X = data[12]
    for m, col, mk in ((4, BLUE, "o"), (5, AQUA, "s"), (7, ORANGE, "^")):
        sel = X[:, 0] == m
        rho = spearmanr(X[sel, 1], X[sel, 4])[0]
        ax.scatter(X[sel, 1], X[sel, 4], s=14, color=col, marker=mk, alpha=0.8, label=f"region of {m} (Spearman {rho:+.2f})")
    ax.axhline(0, color="#c9c8c2", lw=0.8)
    ax.set_xlabel("depth: distance to the nearest pair outside the region", color=INK2); ax.set_ylabel("$\\beta_{loc}$", color=INK2)
    ax.set_title("(b) 12 pairs: grows with depth, then reverses", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    for a in axs:
        for s in ("top", "right"): a.spines[s].set_visible(False)
        for s in ("left", "bottom"): a.spines[s].set_color("#c9c8c2")
        a.tick_params(colors=INK2); a.grid(color="#e6e5e0", lw=0.6)
    fig.tight_layout(); fig.savefig(out + "/fig_depth.png", dpi=170, facecolor=SURF); plt.close(fig)


def relent(out):
    import relative_entropy as rel
    from entanglement_geometry import boundary_size, build_cluster, compact_regions, delaunay_links, random_regions
    n, gen, tau, delta = 10, 3, 0.05, 0.004
    _, _, mids, _ = build_cluster(gen, n)
    links = delaunay_links(mids)
    rows = []
    for label, regions in (("compact", compact_regions(mids, n // 2)), ("random", random_regions(n, range(2, n // 2 + 1), 250, 0))):
        y = rel.region_relative_entropies(gen, n, tau, delta, regions)
        for r, v in zip(regions, y):
            if np.isfinite(v):
                rows.append((label, len(r), boundary_size(r, links), v))
    fig, axs = plt.subplots(1, 3, figsize=(11.0, 3.6), facecolor=SURF)
    cols = {"compact": BLUE, "random": ORANGE}
    for ax, idx, xl in ((axs[0], 1, "volume: pairs in the region"), (axs[1], 2, "boundary: links leaving the region")):
        ax.set_facecolor(SURF)
        for lab in ("random", "compact"):
            sel = [r for r in rows if r[0] == lab]
            xs = np.array([r[idx] for r in sel], float) + (np.random.default_rng(1).uniform(-0.15, 0.15, len(sel)))
            ax.scatter(xs, [r[3] * 1e3 for r in sel], s=9, color=cols[lab], alpha=0.5, label=lab)
        yy = np.array([r[3] for r in rows]); xx = np.array([r[idx] for r in rows], float)
        ax.set_xlabel(xl, color=INK2); ax.set_ylabel("$S_{rel}$ for a flow step $\\delta=0.004$ ($10^{-3}$ nats)", color=INK2, fontsize=8)
        ax.set_title(("(a) " if idx == 1 else "(b) ") + f"correlation with $S_{{rel}}$: {np.corrcoef(xx, yy)[0, 1]:.2f}", color=INK, fontsize=10, loc="left")
    axs[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax = axs[2]; ax.set_facecolor(SURF)
    betas = [-20, -5, -2, -1, -0.5, -0.2, -0.05, 0.05, 0.2, 0.5, 1, 2, 5, 20]
    res, ratio = [], []
    for b in betas:
        t = rel.thermal_first_law(3, 6, 0.05, b, (0, 1, 2)); res.append(t["residual"]); ratio.append(t["regional_ratio"])
    ax.plot(np.abs(betas)[np.array(betas) > 0], np.array(res)[np.array(betas) > 0], color=BLUE, marker="o", ms=4, lw=1.8, label="$\\beta>0$")
    ax.plot(np.abs(betas)[np.array(betas) < 0], np.array(res)[np.array(betas) < 0], color=ORANGE, marker="s", ms=4, lw=1.8, ls=(0, (3, 2)), label="$\\beta<0$")
    ax.set_xscale("log"); ax.set_xlabel("$|\\beta|$ of the reference state", color=INK2); ax.set_ylabel("residual of $K_A\\approx\\beta_{eff}H_A$", color=INK2)
    ax.axvline(0.5, color="#c9c8c2", lw=0.8, ls=(0, (3, 2))); ax.text(0.55, 0.02, "hot", color=INK2, fontsize=8)
    ax.set_title("(c) local first law needs a hot state", color=INK, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    for a in axs:
        for s in ("top", "right"): a.spines[s].set_visible(False)
        for s in ("left", "bottom"): a.spines[s].set_color("#c9c8c2")
        a.tick_params(colors=INK2); a.grid(color="#e6e5e0", lw=0.6)
    fig.tight_layout(); fig.savefig(out + "/fig_relent.png", dpi=170, facecolor=SURF); plt.close(fig)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "more":
    out = sys.argv[1]
    hopf(out); depth(out); relent(out)

"""
Figures for the Spectre pair-qubit network. Writes PNGs to figures/.

    python3 make_figures.py            # all figures (about 1 minute)
    python3 make_figures.py tiers      # one of: tiers bit dynamics couplings counts

Every number drawn comes from the code in this repository; nothing is copied from a log.
"""
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.patches import Polygon

import quantumspectremodel as qm
from quantumspectremodel import A0, FLAT_MODEL
from state import State
from tri_spectre_qc import exact_expectation_z, trotter_step

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
BUMP, DENT, SINGLE, QUBIT = "#d1495b", "#2e86ab", "#e8e4d8", "#f2a93b"


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def _poly(tile):
    return np.array([np.asarray(v.position, float) for v in tile.vertices])


def draw_patch(ax, patch, tiles=None, lw=0.35):
    """Straight outlines; tiles of a pair coloured bump/dent, tiles of a qubit pair outlined in orange."""
    role = {}
    for p, (i, j) in enumerate(patch.pairs):
        role[i], role[j] = "A", "B"
    qtiles = {t for p in patch.pair_to_qubit for t in patch.pairs[p]}
    for i, t in enumerate(patch.tiles):
        face = {"A": BUMP, "B": DENT}.get(role.get(i), SINGLE)
        edge = QUBIT if i in qtiles else "#555555"
        ax.add_patch(Polygon(_poly(t), closed=True, facecolor=face, edgecolor=edge,
                             linewidth=1.6 if i in qtiles else lw, alpha=0.95))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.axis("off")


def fig_tiers():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, k, budget in zip(axes, (2, 3, 4), (4, 4, 4)):
        patch = qm.build_substitution_tier(k)
        qm.assign_qubits_to_patch(patch, max_qubits=budget)
        draw_patch(ax, patch, lw=0.3 if k < 4 else 0.08)
        ax.set_title(f"generation {k}: {patch.n_tiles} tiles, {patch.n_pairs} pairs", fontsize=10)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="#555") for c in (BUMP, DENT, SINGLE)]
    axes[1].legend(handles + [plt.Rectangle((0, 0), 1, 1, fc="none", ec=QUBIT, lw=2)],
                   ["pair tile A (bump when $\\langle Z\\rangle=+1$)", "pair tile B (dent)",
                    "tile in no pair", "pair carrying a qubit (4 nearest the centre)"],
                   loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)
    fig.suptitle("The three exact tiers of the network (edge-matched Spectre patches)", y=1.0)
    _save(fig, "fig_tiers.png")


def fig_bit():
    patch = qm.build_substitution_tier(2)
    qm.assign_qubits_to_patch(patch, max_qubits=6)
    A, B = next(iter(qm.pair_qubit_tiles(patch).values()))
    h = 0.35
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9), gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
    for ax, z in zip(axes[:3], (1.0, 0.0, -1.0)):
        qm.set_pair_bit(A, B, z, h)
        for t, c in ((A, BUMP), (B, DENT)):
            ax.add_patch(Polygon(_poly(t), closed=True, fc=c, ec="#333", alpha=0.85, lw=1.0))
            x, y = t.outline(60)
            ax.plot(x, y, color="#222", lw=0.8)
            p8 = np.asarray(t.vertices[FLAT_MODEL].position, float)
            ax.plot(*p8, "o", color="k", ms=4)
        pts = np.vstack([_poly(A), _poly(B)])
        ax.set_xlim(pts[:, 0].min() - 0.3, pts[:, 0].max() + 0.3)
        ax.set_ylim(pts[:, 1].min() - 0.3, pts[:, 1].max() + 0.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"$\\langle\\tau^z\\rangle={z:g}$:  $A_A={A.area():.3f}$,  $A_B={B.area():.3f}$", fontsize=9)
    zs = np.linspace(-1, 1, 41)
    aa, ab = [], []
    for z in zs:
        qm.set_pair_bit(A, B, z, h)
        aa.append(A.area())
        ab.append(B.area())
    ax = axes[3]
    ax.plot(zs, aa, color=BUMP, label="tile A")
    ax.plot(zs, ab, color=DENT, label="tile B")
    ax.text(0.5, 0.88, "A + B = %.3f for every $\\langle\\tau^z\\rangle$" % (aa[0] + ab[0]), transform=ax.transAxes,
            ha="center", fontsize=8)
    ax.axhline(A0, color="#999", lw=0.6)
    ax.set_xlabel("$\\langle\\tau^z\\rangle$")
    ax.set_ylabel("area")
    ax.legend(fontsize=8, frameon=False, loc="lower center")
    ax.set_title(f"area $=A_0\\pm a h\\langle\\tau^z\\rangle$, $h={h}$", fontsize=9)
    fig.suptitle("The bit moves the flat vertex $P_8$ of both tiles of a pair; the pair keeps its area", y=1.03)
    _save(fig, "fig_bit_geometry.png")


def fig_dynamics():
    patch = qm.build_substitution_tier(2)
    qm.assign_qubits_to_patch(patch, max_qubits=4)
    W = qm.pair_couplings(patch)
    # The geometric couplings are small (|W| <= 0.06 in units of (a h)^2), so to see them act the
    # tunnelling is taken small as well: J = 0.05, kappa = 1 (both are definitions of the model).
    J, kap, h = 0.05, 1.0, 0.3
    T, dt = 40.0, 0.25
    n_steps = int(round(T / dt))
    st = State(n_qubits=4)
    ts, trot = [0.0], [[st.expectation_z(p) for p in range(4)]]
    for s in range(n_steps):
        trotter_step(st, W, 0, J, kap, dt)
        ts.append((s + 1) * dt)
        trot.append([st.expectation_z(p) for p in range(4)])
    trot = np.array(trot)
    tex = np.linspace(0, T, 41)
    exact = np.array([exact_expectation_z(W, J, kap, t) for t in tex])
    err = np.abs(trot[::int(round(1.0 / dt))] - exact).max()
    cols = ["#d1495b", "#2e86ab", "#f2a93b", "#3d9970"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for p in range(4):
        axes[0].plot(ts, trot[:, p], color=cols[p], lw=1.3, label=f"pair {p}")
        axes[0].plot(tex, exact[:, p], "o", color=cols[p], ms=3.2, mfc="none")
    axes[0].set_xlabel("time $t$")
    axes[0].set_ylabel("$\\langle Z_p\\rangle$")
    axes[0].set_title(f"$J=0.05,\\ \\kappa=1$: Trotter (lines), exact (circles)\nmax difference {err:.1e}", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    # all-equal weak-coupling case for contrast
    st = State(n_qubits=4)
    zw = [[st.expectation_z(p) for p in range(4)]]
    for s in range(n_steps):
        trotter_step(st, W, 0, 1.0, 0.09, dt / 10)
        zw.append([st.expectation_z(p) for p in range(4)])
    zw = np.array(zw)
    tw = np.arange(n_steps + 1) * dt / 10
    for p in range(4):
        axes[1].plot(tw, zw[:, p], color=cols[p], lw=1.3, label=f"pair {p}")
    axes[1].set_xlabel("time $t$")
    axes[1].set_title("$J=1,\\ \\kappa=(ah)^2=0.09$ (default scale):\ncouplings $\\sim1\\%$ of $J$, the four curves coincide", fontsize=8)
    axes[1].legend(fontsize=8, frameon=False, ncol=2)
    # geometry feedback from the first run
    st = State(n_qubits=4)
    areas = {q: [] for q in range(4)}
    ts2 = []
    for s in range(n_steps + 1):
        ts2.append(s * dt)
        for q, (A, B) in qm.pair_qubit_tiles(patch).items():
            qm.set_pair_bit(A, B, st.expectation_z(q), h)
            areas[q].append(A.area() - A0)
        trotter_step(st, W, 0, J, kap, dt)
    for q in range(4):
        axes[2].plot(ts2, areas[q], color=cols[q], lw=1.3, label=f"tile A of pair {q}")
    axes[2].axhline(0, color="#999", lw=0.6)
    axes[2].set_xlabel("time $t$")
    axes[2].set_ylabel("$A_A-A_0$  (tile B: opposite sign)")
    axes[2].set_title(f"geometry feedback of the left panel, $h={h}$", fontsize=9)
    axes[2].legend(fontsize=8, frameon=False)
    fig.suptitle("Pair-qubit dynamics on the generation-2 patch, all four central pairs start with tile A as the bump", y=1.03)
    _save(fig, "fig_dynamics.png")


def fig_couplings():
    patch = qm.build_substitution_tier(4)
    qm.assign_qubits_to_patch(patch, max_qubits=None)
    # use the 120 pairs nearest the centre to keep the pair list quick
    keep = patch.pair_order[:120]
    patch.pair_to_qubit = {p: q for q, p in enumerate(keep)}
    W = qm.pair_couplings(patch)
    cents = {q: 0.5 * (qm.tile_centroid(A) + qm.tile_centroid(B)) for q, (A, B) in qm.pair_qubit_tiles(patch).items()}
    n = W.shape[0]
    r, w = [], []
    for p in range(n):
        for q in range(p + 1, n):
            r.append(np.linalg.norm(cents[p] - cents[q]))
            w.append(W[p, q])
    r, w = np.array(r), np.array(w)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    pos, neg = w > 0, w < 0
    axes[0].loglog(r[pos], w[pos], ".", ms=2.5, color=BUMP, label=f"$W>0$ ({pos.sum()}) antiferro")
    axes[0].loglog(r[neg], -w[neg], ".", ms=2.5, color=DENT, label=f"$W<0$ ({neg.sum()}) ferro")
    rr = np.linspace(r.min(), r.max(), 50)
    axes[0].loglog(rr, 4.0 * rr ** -3.0, "k--", lw=1, label="$\\propto r^{-3}$")
    axes[0].set_xlabel("distance between pair midpoints")
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0].set_ylabel("$|W_{pq}|$")
    mask = np.abs(w) > 0
    slope = np.polyfit(np.log(r[mask]), np.log(np.abs(w[mask])), 1)[0]
    axes[0].set_title(f"dipole couplings of {n} central pairs (level 4), fitted slope {slope:.2f}", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    im = axes[1].imshow(W, cmap="RdBu_r", vmin=-np.abs(W).max() * 0.3, vmax=np.abs(W).max() * 0.3)
    axes[1].set_title("$W_{pq}$ (pairs ordered from the centre)", fontsize=9)
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle("The couplings fixed by the geometry have both signs: the pair model is frustrated", y=1.02)
    _save(fig, "fig_couplings.png")


def fig_counts():
    ks = np.arange(0, 6)
    tiles, pairs = [], []
    for k in ks:
        p = qm.build_substitution_tier(int(k))
        tiles.append(p.n_tiles)
        pairs.append(p.n_pairs)
    tiles, pairs = np.array(tiles), np.array(pairs)
    rho0 = (5 - math.sqrt(15)) / 5
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].semilogy(ks, tiles, "o-", color="#555", label="tiles $N_k=8N_{k-1}-N_{k-2}$")
    axes[0].semilogy(ks[2:], pairs[2:], "s-", color=BUMP, label="bump-and-dent pairs $P_k$")
    axes[0].set_xlabel("generation $k$")
    axes[0].legend(fontsize=8, frameon=False)
    for k, t, p in zip(ks, tiles, pairs):
        axes[0].annotate(str(t), (k, t), textcoords="offset points", xytext=(0, 5), fontsize=7, ha="center")
        if p:
            axes[0].annotate(str(p), (k, p), textcoords="offset points", xytext=(0, -11), fontsize=7, ha="center")
    axes[1].plot(ks[2:], pairs[2:] / tiles[2:], "s-", color=BUMP, label="$P_k/N_k$")
    axes[1].axhline(rho0, color="k", ls="--", lw=1, label=f"$\\rho_0=(5-\\sqrt{{15}})/5={rho0:.4f}$")
    axes[1].set_xlabel("generation $k$")
    axes[1].set_ylim(0, 0.3)
    axes[1].legend(fontsize=8, frameon=False)
    fig.suptitle("Tiles and pairs per generation (counted from the exact patches)", y=1.02)
    _save(fig, "fig_pair_counts.png")


FIGS = {"tiers": fig_tiers, "bit": fig_bit, "dynamics": fig_dynamics, "couplings": fig_couplings, "counts": fig_counts}

if __name__ == "__main__":
    which = sys.argv[1:] or list(FIGS)
    for name in which:
        FIGS[name]()

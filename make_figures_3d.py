"""
3D figures for the Spectre pair-qubit network. Writes PNGs to figures/.

    python3 make_figures_3d.py             # all (about 1 minute)
    python3 make_figures_3d.py stack       # one of: stack spacetime relief lift

Every number drawn comes from the code in this repository.
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

import quantumspectremodel as qm
from quantumspectremodel import A0, FLAT_MODEL
from state import State
from tri_spectre_qc import trotter_step

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


def _lift(poly2, z, scale=1.0, centre=(0.0, 0.0)):
    p = (poly2 - np.asarray(centre)) * scale
    return np.column_stack([p, np.full(len(p), z)])


def _flat(ax):
    ax.set_box_aspect((1, 1, 0.9))
    for a in (ax.xaxis, ax.yaxis):
        a.set_ticklabels([])
    ax.tick_params(length=0)


def fig_stack():
    """The three tiers as stacked planes with the CX/CZ bridges between anchor pairs."""
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    anchors = []
    for lvl, (k, budget) in enumerate(zip((2, 3, 4), (4, 4, 4))):
        patch = qm.build_substitution_tier(k)
        qm.assign_qubits_to_patch(patch, max_qubits=budget)
        cents = np.array([qm.tile_centroid(t) for t in patch.tiles])
        centre = cents.mean(axis=0)
        radius = np.linalg.norm(_poly(patch.tiles[0]) - centre, axis=1).max()
        extent = max(np.linalg.norm(np.vstack([_poly(t) for t in patch.tiles]) - centre, axis=1))
        scale = 10.0 / extent
        role = {}
        for i, j in patch.pairs:
            role[i], role[j] = BUMP, DENT
        qtiles = {t for p in patch.pair_to_qubit for t in patch.pairs[p]}
        polys, cols, edges = [], [], []
        for i, t in enumerate(patch.tiles):
            polys.append(_lift(_poly(t), 3.2 * lvl, scale, centre))
            cols.append(role.get(i, SINGLE))
            edges.append(QUBIT if i in qtiles else "#777")
        pc = Poly3DCollection(polys, facecolors=cols, edgecolors=edges, linewidths=0.25 if k < 4 else 0.05, alpha=0.9)
        ax.add_collection3d(pc)
        anch = {}
        for q, (A, B) in qm.pair_qubit_tiles(patch).items():
            m = 0.5 * (qm.tile_centroid(A) + qm.tile_centroid(B))
            anch[q] = np.array([*((m - centre) * scale), 3.2 * lvl])
        anchors.append(anch)
        ax.text(11.5, 0, 3.2 * lvl, f"generation {k}\n{patch.n_tiles} tiles\n{patch.n_pairs} pairs", fontsize=9, va="center")
    # the bridges of TriSpectreNetwork.entangle_across_computers: CX(g0,g1), CX(g1,g2), CZ(g0,g2)
    for i in (0, 1):
        pts = [anchors[0][i], anchors[1][i], anchors[2][i]]
        ax.add_collection3d(Line3DCollection([[pts[0], pts[1]], [pts[1], pts[2]]], colors="#111", linewidths=2.0))
        ax.add_collection3d(Line3DCollection([[pts[0], pts[2]]], colors="#111", linewidths=1.2, linestyles="dashed"))
        for p in pts:
            ax.scatter(*p, color=QUBIT, edgecolor="k", s=45, zorder=10)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-0.5, 7.2)
    ax.view_init(elev=24, azim=-62)
    _flat(ax)
    ax.set_zticks([])
    ax.set_title("TriSpectreNetwork: three exact tiers (each scaled to the same width) and the cross-tier bridges\n"
                 "solid: CX between neighbouring tiers, dashed: CZ between the outer tiers; dots: anchor pair qubits", fontsize=9)
    _save(fig, "fig3d_tier_stack.png")


def _evolve(patch, J, kappa, T, dt, h, q_count):
    W = qm.pair_couplings(patch)
    st = State(n_qubits=q_count)
    ts, zs = [], []
    for s in range(int(round(T / dt)) + 1):
        ts.append(s * dt)
        zs.append([st.expectation_z(q) for q in range(q_count)])
        trotter_step(st, W, 0, J, kappa, dt)
    return np.array(ts), np.array(zs)


def fig_spacetime():
    """Time as the vertical axis: the pair qubits as worldlines over the patch, tile A outlines at four times."""
    patch = qm.build_substitution_tier(2)
    qm.assign_qubits_to_patch(patch, max_qubits=6)
    J, kap, h, T, dt = 0.05, 1.0, 0.35, 40.0, 0.5
    ts, zs = _evolve(patch, J, kap, T, dt, h, 6)
    fig = plt.figure(figsize=(10, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    cents = np.array([qm.tile_centroid(t) for t in patch.tiles])
    centre = cents.mean(axis=0)
    role = {}
    for i, j in patch.pairs:
        role[i], role[j] = BUMP, DENT
    ax.add_collection3d(Poly3DCollection([_lift(_poly(t), 0.0, 1.0, centre) for t in patch.tiles],
                                         facecolors=[role.get(i, SINGLE) for i in range(patch.n_tiles)],
                                         edgecolors="#888", linewidths=0.3, alpha=0.55))
    cmap = plt.get_cmap("coolwarm_r")
    pairs = qm.pair_qubit_tiles(patch)
    for q, (A, B) in pairs.items():
        m = 0.5 * (qm.tile_centroid(A) + qm.tile_centroid(B)) - centre
        pts = np.column_stack([np.full_like(ts, m[0]), np.full_like(ts, m[1]), ts / T * 12.0])
        segs = [[pts[i], pts[i + 1]] for i in range(len(ts) - 1)]
        lc = Line3DCollection(segs, colors=cmap((zs[:-1, q] + 1) / 2), linewidths=3)
        ax.add_collection3d(lc)
        ax.text(m[0], m[1], -0.9, f"{q}", fontsize=8, ha="center")
    # tile A of the most active pair as an outline at four times, with P8 displaced by its <Z>
    q_show = int(np.argmin(zs.min(axis=0)))
    A, B = pairs[q_show]
    for t_pick in (0.0, 10.0, 20.0, 30.0, 40.0):
        idx = int(round(t_pick / dt))
        qm.set_pair_bit(A, B, zs[idx, q_show], h)
        for tile, c in ((A, BUMP), (B, DENT)):
            P = _lift(_poly(tile), t_pick / T * 12.0, 1.0, centre)
            ax.add_collection3d(Poly3DCollection([P], facecolors="none", edgecolors=c, linewidths=1.4))
            p8 = P[FLAT_MODEL]
            ax.scatter(*p8, color="k", s=10, zorder=10)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1))
    cb = fig.colorbar(sm, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("$\\langle Z_p\\rangle$  (+1: tile A is the bump)")
    allp = np.vstack([_poly(t) - centre for t in patch.tiles])
    ax.set_xlim(allp[:, 0].min(), allp[:, 0].max())
    ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
    ax.set_zlim(-1, 12.5)
    ax.view_init(elev=21, azim=-58)
    _flat(ax)
    ax.set_zlabel("time $t$ (0 to 40)")
    ax.set_zticks([0, 12])
    ax.set_zticklabels(["0", "40"])
    ax.set_title(f"Pair-qubit worldlines over the generation-2 patch ($J={J}$, $\\kappa={kap:g}$)\n"
                 f"outlines: the two tiles of pair {q_show} at $t=0,10,20,30,40$, flat vertex $P_8$ dotted", fontsize=9)
    _save(fig, "fig3d_spacetime.png")


def fig_relief():
    """Tile area change as height: the bit raises tile A and lowers tile B of each pair."""
    patch = qm.build_substitution_tier(2)
    qm.assign_qubits_to_patch(patch, max_qubits=6)
    J, kap, h = 0.05, 1.0, 0.35
    times = (0.0, 14.0, 28.0)
    ts, zs = _evolve(patch, J, kap, 28.0, 0.5, h, 6)
    pairs = qm.pair_qubit_tiles(patch)
    cents = np.array([qm.tile_centroid(t) for t in patch.tiles])
    centre = cents.mean(axis=0)
    in_pair = {}
    for q, (A, B) in pairs.items():
        in_pair[id(A)] = (q, +1)
        in_pair[id(B)] = (q, -1)
    fig = plt.figure(figsize=(15, 5))
    norm = plt.Normalize(-h, h)
    cmap = plt.get_cmap("coolwarm_r")
    for k, t in enumerate(times):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        idx = int(round(t / 0.5))
        for q, (A, B) in pairs.items():
            qm.set_pair_bit(A, B, zs[idx, q], h)
        polys, cols, zl = [], [], []
        for tile in patch.tiles:
            dA = tile.area() - A0
            P = _lift(_poly(tile), dA * 6.0, 1.0, centre)
            polys.append(P)
            cols.append(cmap(norm(dA)) if id(tile) in in_pair else SINGLE)
            zl.append(dA)
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="#555", linewidths=0.25, alpha=0.95))
        allp = np.vstack([_poly(t) - centre for t in patch.tiles])
        ax.set_xlim(allp[:, 0].min(), allp[:, 0].max())
        ax.set_ylim(allp[:, 1].min(), allp[:, 1].max())
        ax.set_zlim(-3, 3)
        ax.view_init(elev=32, azim=-70)
        _flat(ax)
        ax.set_zticks([])
        ax.set_title(f"$t={t:g}$:  area change $A-A_0$ from {min(zl):+.2f} to {max(zl):+.2f}", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=fig.axes, shrink=0.6, pad=0.02)
    cb.set_label("$A-A_0$ of a pair tile (height is $6\\times$ this)")
    fig.suptitle("The bit as relief: each tile of a pair is lifted or lowered by its area change; tiles in no pair stay at 0", y=0.97)
    _save(fig, "fig3d_area_relief.png")


def fig_lift():
    """The vertices of the exact patch in the lift lattice Z^4, projected to (p, q, r) with s as colour."""
    patch = qm.build_substitution_tier(3)
    keys = sorted(set(k for t in patch.tiles for k in t.data["exact_vertices"]))
    arr = np.array(keys, float)
    flat = set()
    for i, j in patch.pairs:
        flat.add(patch.tiles[i].data["exact_vertices"][FLAT_MODEL])
    fl = np.array(sorted(flat), float)
    fig = plt.figure(figsize=(13, 5.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    sc = ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], c=arr[:, 3], s=1.2, cmap="viridis", alpha=0.45)
    ax.scatter(fl[:, 0], fl[:, 1], fl[:, 2], color=BUMP, s=14, edgecolor="k", linewidths=0.3,
               label=f"flat vertices $P_8$ of the {len(fl)} pairs")
    ax.set_xlabel("$p$")
    ax.set_ylabel("$q$")
    ax.set_zlabel("$r$")
    ax.view_init(elev=18, azim=-55)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_title("generation 3: lift coordinates $(p,q,r)$, colour $s$", fontsize=9)
    fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, label="$s$")
    # plane position vs the fourth lift coordinate: the tiling is a slice, the lattice is 4-dimensional
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    plane = np.array([[np.real(__import__("spectre_exact").to_c(k)), np.imag(__import__("spectre_exact").to_c(k))] for k in keys])
    ax2.scatter(plane[:, 0], plane[:, 1], arr[:, 3], c=arr[:, 3], s=1.2, cmap="viridis", alpha=0.45)
    flp = np.array([[np.real(__import__("spectre_exact").to_c(tuple(int(x) for x in k))),
                     np.imag(__import__("spectre_exact").to_c(tuple(int(x) for x in k)))] for k in fl])
    ax2.scatter(flp[:, 0], flp[:, 1], fl[:, 3], color=BUMP, s=14, edgecolor="k", linewidths=0.3)
    ax2.set_xlabel("$x$")
    ax2.set_ylabel("$y$")
    ax2.set_zlabel("$s$")
    ax2.view_init(elev=22, azim=-60)
    ax2.set_title("plane position $(x,y)$ against the lift coordinate $s$", fontsize=9)
    fig.suptitle("The patch as a set of points of the lift lattice $\\mathbb{Z}[\\zeta_{12}]\\cong\\mathbb{Z}^4$", y=1.0)
    _save(fig, "fig3d_lift.png")


FIGS = {"stack": fig_stack, "spacetime": fig_spacetime, "relief": fig_relief, "lift": fig_lift}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(FIGS):
        FIGS[name]()

#!/usr/bin/env python3
"""Figures for HAT_TIME.md. Needs brentharts/hexasphere as a sibling folder (or HEXASPHERE_DIR).
python3 make_hat_time_figures.py  ->  figures/hat_time_1_rings.png, hat_time_2_levels.png, hat_time_3_chirality.png"""
import collections
import itertools
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, [sys.argv[0]]
import ring_bits as RB  # noqa: E402
import qdeform_punctures as QD  # noqa: E402
import hat_chirality as HC  # noqa: E402
sys.argv = _argv

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8,
                     "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 400, "savefig.bbox": "tight",
                     "axes.titlelocation": "left", "xtick.direction": "out", "ytick.direction": "out"})
OUT = os.path.join(HERE, "figures")
ORANGE, BLUE, GREEN, VERM, GREY = "#E69F00", "#0072B2", "#009E73", "#D55E00", "#999999"
RINGCOL = ["#DCE9F5", "#E4F1E1", "#F3E9D2", "#EADDEA"]


def letter(ax, s, dx=-0.08, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def frame(z):
    z = z / np.linalg.norm(z)
    x = np.cross([0.3, 0.2, 1.0], z); x /= np.linalg.norm(x)
    return x, np.cross(z, x), z


def draw_ball(ax, fl, T, rings, colour4, view, flipped=(), mark=None):
    x, y, z = frame(view)
    pos = {}
    def cp(c):
        if c not in pos:
            pos[c] = fl.S.corner_position(c)
        return pos[c]
    ring_of = {k: r for r, (_, _, _, reg) in enumerate(rings) for k in reg}
    for i in T:
        ks = fl.placements[i]
        cen = np.mean([cp(c) for k in ks for c in fl.corners[k]], axis=0)
        if cen @ z < 0.08:
            continue
        r = ring_of[next(iter(ks))]
        col = (ORANGE if r in flipped else RINGCOL[colour4[r]])
        edges = collections.Counter()
        for k in ks:
            cs = fl.corners[k]
            q = np.array([cp(c) for c in cs])
            ax.add_patch(MPoly(np.c_[q @ x, q @ y], closed=True, fc=col, ec=col, lw=0.3))
            for a in range(4):
                edges[frozenset((cs[a], cs[(a + 1) % 4]))] += 1
        for e, n in edges.items():
            if n == 1:
                a, b = tuple(e)
                q = np.array([cp(a), cp(b)])
                ax.plot(q @ x, q @ y, color="#333333", lw=0.6, solid_capstyle="round")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), color="#BBBBBB", lw=0.5)
    ax.set_aspect("equal"); ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03); ax.axis("off")


def four_colouring(edges):
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    col = {}
    def go(v):
        if v == 12:
            return True
        for c in range(4):
            if all(col.get(u) != c for u in adj[v]):
                col[v] = c
                if go(v + 1):
                    return True
                del col[v]
        return False
    assert go(0)
    return col


def fig1(ball, fl, T0, rings, P, edges):
    c4 = four_colouring(edges)
    view = P[0] + 0.25 * P[[j for i, j in edges if i == 0][0]]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    b0 = (0,) * 12
    b1 = tuple(1 if i == 0 else 0 for i in range(12))
    nb = [j for i, j in edges if i == 0][0]
    b2 = tuple(1 if i in (0, nb) else 0 for i in range(12))
    for ax, b, t in zip(axs, (b0, b1, b2), ("None flipped: 80 vertices", "One ring flipped: 80", "Two neighbours flipped: 79")):
        T = RB.tiling_of(b, T0, rings)
        draw_ball(ax, fl, T, rings, c4, view, flipped=[i for i in range(12) if b[i]])
        ax.set_title(t, loc="left")
        assert fl.n_vertices(T) == int(t.split(":")[1].split()[0])
    for ax, l in zip(axs, "abc"):
        letter(ax, l, dx=-0.09, dy=1.0)
    fig.legend(handles=[Patch(fc=ORANGE, label="hats of a flipped ring"), Patch(fc=RINGCOL[0], ec="#999999", label="hats of an unflipped ring (tint = ring)"),
                        Line2D([], [], color="#333333", lw=0.8, label="hat outline")], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.savefig(os.path.join(OUT, "hat_time_1_rings.png")); plt.close(fig)


def fig2():
    V, E, F = QD.icosa()
    cyc = QD.hamiltonian_cycles(E, 2)[0]
    ks = [2, 3, 4, 5, 6, 8, 10, 14, 20, 30, 50, 100, 200]
    spec = {}
    for k in ks + [None]:
        m = QD.Model(1j if k is None else QD.A_of_level(k), cyc, E, F)
        dim, neg, ev, herm = m.physical(m.hamiltonian())
        spec[k] = np.sort(ev.real)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.55, width_ratios=[0.8, 1.25, 0.9]))
    ax = axs[0]
    kk = [1, 2, 3, 4, 5, 6, 7, 8]
    dims = [QD.verlinde(k) for k in kk]
    ax.plot(kk, dims, "o-", color=BLUE, ms=3.5, lw=1)
    ax.axhline(132, color=GREY, ls=":", lw=0.8)
    ax.text(8, 92, "132 classical", ha="right", va="top", fontsize=8, color="#555555")
    ax.annotate("32 = 2$^5$", (2, 32), (2.9, 18), fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"))
    ax.set_xlabel("level $k$"); ax.set_ylabel("number of states"); ax.set_ylim(-6, 145); ax.set_xticks(kk)
    ax.set_title("States per level", loc="left")
    ax = axs[1]
    xs = {k: 1.0 / k for k in ks}
    for k in ks:
        ax.plot(np.full(len(spec[k]), xs[k]), spec[k], "_", color="#B8CCE0", ms=4, mew=0.6)
        ax.plot([xs[k]], [spec[k][0]], "o", color=BLUE, ms=3)
    ax.plot(np.zeros(132), spec[None], "_", color="#F1D49A", ms=5, mew=0.6)
    ax.plot([0], [spec[None][0]], "o", color=VERM, ms=3.5)
    ax.plot([xs[k] for k in ks] + [0], [spec[k][0] for k in ks] + [spec[None][0]], "-", color=BLUE, lw=0.7, zorder=0)
    ax.annotate("classical ring bits\n$-6.188$", (0, spec[None][0]), (0.02, -11.9), va="top", fontsize=8, arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555"))
    ax.set_xlabel("1 / level"); ax.set_ylabel("energy  $\\sum_{edges}(1/4 - e/d)$")
    ax.set_xlim(-0.03, 0.53); ax.set_ylim(-15.5, 13)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=3, label="lowest level"), Line2D([], [], marker="_", ls="", color="#8FAACB", label="all levels")], loc="upper right")
    ax.set_title("Towards the classical spectrum", loc="left")
    ax = axs[2]
    lv = collections.Counter(np.round(spec[2], 4))
    for e, n in sorted(lv.items()):
        ax.hlines(e, 0, n, color=GREEN, lw=2)
        ax.text(n + 0.6, e, "%d" % n, va="center", fontsize=8)
    for m in (0, 2, 4, 6):
        ax.plot([-1.2], [-7.5 + 5 ** 0.5 / 2 * (6 - 2 * m)], ">", color="#333333", ms=3)
    ax.set_xlim(-2, 20); ax.set_xlabel("states in the level"); ax.set_ylabel("energy at level 2")
    ax.set_ylim(-15.5, 0.5)
    ax.text(19.5, -1.6, "arrows: six free\nmodes of energy $\\sqrt{5}$", ha="right", va="top", fontsize=8, color="#333333")
    ax.set_title("Level 2: free Majoranas", loc="left")
    for a, l in zip(axs, "abc"):
        letter(a, l, dx=-0.22)
    fig.savefig(os.path.join(OUT, "hat_time_2_levels.png")); plt.close(fig)
    return {str(k): [float(spec[k][0]), int(len(spec[k]))] for k in spec}


def fig3(ball, fl, T0, rings, P, edges):
    paths = HC.helical_paths(P, edges)
    adj = {i: set() for i in range(12)}
    for i, j in edges:
        adj[i].add(j); adj[j].add(i)
    a = 0
    d = int(np.argmin(P @ P[a]))
    seqs = []
    for b in sorted(adj[a]):
        for c in sorted(adj[b] & adj[d]):
            seqs.append(([a, b, c, d], paths[frozenset((a, b, c, d))]))
    b0 = seqs[0][0][1]
    left = [s for s, h in seqs if s[1] == b0 and h == 1][0]
    right = [s for s, h in seqs if s[1] == b0 and h == -1][0]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.45, width_ratios=[1.05, 0.9, 0.9]))
    ax = axs[0]
    view = P[left[1]] + P[left[2]] + P[right[2]] + 0.35 * (P[a] + P[d]) + 0.25 * np.cross(P[a], P[left[1]])
    x, y, z = frame(view)
    for i, j in edges:
        front = (P[i] + P[j]) @ z > 0
        ax.plot([P[i] @ x, P[j] @ x], [P[i] @ y, P[j] @ y], color="#BBBBBB" if front else "#E6E6E6", lw=0.8 if front else 0.5, zorder=1 if front else 0)
    for s, col, off in ((left, BLUE, 0.012), (right, VERM, -0.012)):
        q = P[s]
        ax.plot(q @ x + off, q @ y + off, "-", color=col, lw=2, zorder=3, solid_capstyle="round")
    for v, name in ((a, "a"), (left[1], "b"), (left[2], "c"), (right[2], "c$'$"), (d, "d")):
        ax.plot([P[v] @ x], [P[v] @ y], "o", color="#333333", ms=3.5, zorder=4)
        ax.text(P[v] @ x + 0.06, P[v] @ y + 0.05, name, fontsize=9, zorder=5)
    ax.set_aspect("equal"); ax.axis("off"); ax.margins(0.08)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="left-handed path, $h=+1$"), Line2D([], [], color=VERM, lw=2, label="right-handed path, $h=-1$")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02))
    ax.set_title("Paths have a hand", loc="left")
    ax = axs[1]
    r = json.load(open(os.path.join(HERE, "hat_chirality.json")))
    vals = [0, 0, r["wl_differs_from_mirror_pattern"]]
    ax.bar([1, 2, 3], vals, color=[GREY, GREY, BLUE], width=0.6)
    for xx, v in zip([1, 2, 3], vals):
        ax.text(xx, v + 25, str(v), ha="center", fontsize=8)
    ax.axhline(1320, color=GREY, ls=":", lw=0.8)
    ax.set_xticks([1, 2, 3]); ax.set_xlabel("neighbourhood radius (panels)"); ax.set_ylabel("chiral patterns told from mirror")
    ax.set_ylim(0, 1500)
    ax.set_title("Visible at radius 3", loc="left")
    ax = axs[2]
    c = json.load(open(os.path.join(HERE, "hat_chirality_chi4.json")))["chi4_values"]
    ks = [-1, 0, 1]
    ax.bar(ks, [c[str(k)] for k in ks], color=[VERM, GREY, BLUE], width=0.6)
    for k in ks:
        ax.text(k, c[str(k)] + 60, str(c[str(k)]), ha="center", fontsize=8)
    ax.set_xticks(ks); ax.set_xlabel("$\\chi_4$ of the ring pattern"); ax.set_ylabel("tilings (of 4096)"); ax.set_ylim(0, 3500)
    ax.set_title("The 4-ring chiral term", loc="left")
    for a_, l in zip(axs, "abc"):
        letter(a_, l, dx=-0.2 if l != "a" else -0.12)
    fig.savefig(os.path.join(OUT, "hat_time_3_chirality.png")); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    ball, fl, T0, rings, P, edges = RB.setup()
    fig1(ball, fl, T0, rings, P, edges)
    print("fig 1 done")
    print(fig2())
    fig3(ball, fl, T0, rings, P, edges)
    print("fig 3 done")

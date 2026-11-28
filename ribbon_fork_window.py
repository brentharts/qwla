#!/usr/bin/env python3
"""
What in the phase decides the branch at a fork?  (ROADMAP Stage 10e.)

A fork is the Mystic of a Gamma supertile; the active links x and y leave it (Stage 9w). Stage 9w found that the
branch a ribbon takes belongs to the phase and not to the step direction. Here the phase is made explicit: every
Mystic has a window point in Z_0(G) (dumps of generic_phase.py), and for each fork inside a flipped active ribbon the
branch taken (x, y, or both, over the steps of the dump) is set against the window point of the fork.

Test: leave-one-out nearest-neighbour prediction of the branch from the window point alone. If the branch is a
function of the position in the window, neighbours in the window take the same branch.
Figure: figures/ribbon_fork_window.png, the Mystic window with the forks coloured by branch.

Usage: python3 ribbon_fork_window.py dump.json [...] [--tests]
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import stack_trace as ST  # noqa: E402
import ribbon_trees as RT  # noqa: E402


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    pts, allG, rim = [], [], []
    for f in files:
        dump = json.load(open(f))[0]
        d = ST.prepare(f)
        P, L, lab, nb = d["P"], d["L"], d["lab"], d["nb"]
        sp, owner, u = defaultdict(list), [], 0
        for a, pl in enumerate(dump["tiles"]):
            for Q in pl:
                for _ in range(1 if len(Q) == 14 else 2):
                    sp[a].append(u); owner.append(a); u += 1
        win = dump["window"]
        G = [i for i in range(len(P)) if lab[i] == "Gamma1"]
        s_, _ = RT.links(P[G], L[G])
        succ = {G[a]: {k: G[b] for k, b in v.items()} for a, v in s_.items()}
        allG += [win[a][1:] for a in range(len(win)) if win[a][0] == "G"]
        count = np.zeros(len(win), int)
        taken = defaultdict(set)
        for st in dump["steps"]:
            count[st["removed"]] += 1
            R = {v for a in st["removed"] for v in sp[a]}
            theta = lambda v: any(w in R and lab[w] == "Theta" for w in nb.get(v, []))
            for i, s in succ.items():
                if len(s) == 2 and i in R:
                    for k in "xy":
                        if s[k] in R and theta(s[k]):
                            taken[i].add(k)
        rim += [win[a][1:] for a in range(len(win)) if win[a][0] == "G" and count[a] > 0]
        for i, br in taken.items():
            pts.append((win[owner[i]][1], win[owner[i]][2], "".join(sorted(br)), os.path.basename(f)))
    Z = np.array([[p[0], p[1]] for p in pts]); B = [p[2] for p in pts]
    n = len(pts)
    # leave-one-out nearest neighbour, among forks with a single branch
    single = [i for i in range(n) if B[i] in ("x", "y")]
    hit = 0; dists = []
    for i in single:
        others = [j for j in single if j != i]
        dd = np.linalg.norm(Z[others] - Z[i], axis=1)
        j = others[int(np.argmin(dd))]
        hit += B[j] == B[i]; dists.append(float(dd.min()))
    base = max(Counter(B[i] for i in single).values()) / len(single)
    # a linear separator in the window (least squares on +-1), as a simple summary
    y = np.array([1.0 if B[i] == "x" else -1.0 for i in single])
    A = np.c_[Z[single], np.ones(len(single))]
    w = np.linalg.lstsq(A, y, rcond=None)[0]
    lin = float(np.mean(np.sign(A @ w) == y))
    out = dict(forks=n, branches=dict(Counter(B)), single=len(single), loo_nearest_neighbour_accuracy=hit / len(single), majority_baseline=base,
               mean_nn_distance=float(np.mean(dists)), linear_separator=[float(x) for x in w], linear_accuracy=lin,
               mean_window_point={k: [float(x) for x in Z[[i for i in range(n) if B[i] == k]].mean(0)] for k in set(B)})
    print(out)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.titlelocation": "left", "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False})
    fig, ax = plt.subplots(figsize=(3.5, 3.0), layout="constrained")
    aG = np.array(allG); rG = np.array(rim)
    ax.scatter(aG[:, 0], aG[:, 1], s=1.5, c="#dddddd", linewidths=0, rasterized=True, label="all Mystics")
    ax.scatter(rG[:, 0], rG[:, 1], s=3, c="#a8a8a8", linewidths=0, label="flipped Mystics")
    for k, col, mk, name in (("x", "#1f77b4", "o", "fork, branch x"), ("y", "#d95f02", "s", "fork, branch y"), ("xy", "k", "D", "fork, both")):
        m = [i for i in range(n) if B[i] == k]
        if m:
            ax.scatter(Z[m, 0], Z[m, 1], s=16, c=col, marker=mk, linewidths=0.3, edgecolors="white", label=name, zorder=3)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_visible(False)
    ax.set_title("Forks in the Mystic window $Z_0(G)$, by branch taken")
    ax.legend(loc="upper right", fontsize=7, handletextpad=0.3, borderaxespad=0.0, markerscale=1.2)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "ribbon_fork_window.png")); plt.close(fig)
    json.dump(out, open(os.path.join(HERE, "ribbon_fork_window.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        assert out["loo_nearest_neighbour_accuracy"] > out["majority_baseline"]
        print("ribbon_fork_window: tests passed (the window point predicts the branch better than the majority guess)")


if __name__ == "__main__":
    main()

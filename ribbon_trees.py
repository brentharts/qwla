#!/usr/bin/env python3
"""
The trees of the active-ribbon network: where chains begin and end, and which branch a ribbon takes at a fork.
(ROADMAP Stage 9w.)

ribbon_network.py links every Mystic of a patch to the next by the three links s, x, y. A Mystic has at most one link
in, so the network is a forest of directed trees. This script reports

 1  the role of a Mystic (root, middle, fork, end, isolated) against the label of its level-1 supertile;
 2  the trees: number of Mystics, depth, the highest level of supertile boundary they cross, and (level 5) whether their
    root and their ends lie at the boundary of the patch (chains cut by the boundary) or inside it (chains that begin
    and end in the tiling);
 3  with dumps from generic_phase.py: at each fork inside an active ribbon, which branch was removed by each phason
    step (x, y or both), and whether one fork takes different branches for different step directions.

Usage: python3 ribbon_trees.py [dump.json ...] [--tests]        (about 3 minutes; level 6 for the trees, level 5 for the boundary)
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recognisability_radius as RR  # noqa: E402
import ribbon_network as RN  # noqa: E402


def links(P, L):
    tree = cKDTree(P)
    succ, pred = defaultdict(dict), {}
    for u in range(len(P)):
        for k, (p, M) in RN.LINKS.items():
            for v in tree.query_ball_point(P[u] + L[u] @ p, 1e-3):
                if np.abs(L[u] @ M - L[v]).max() < 1e-6:
                    succ[u][k] = v; pred[v] = u
    return succ, pred


def forest(level, boundary=False):
    levels, leaves = RR.build(level)
    G = [i for i, (T, lab, b) in enumerate(leaves) if lab == "Gamma1"]
    P = np.array([leaves[i][0][:, 2] for i in G]); L = np.array([leaves[i][0][:, :2] for i in G])
    anc = {1: np.array([levels[0][leaves[i][2]][2] for i in G])}
    for k in range(2, level):
        anc[k] = np.array([levels[k - 1][a][2] for a in anc[k - 1]])
    lab1 = [levels[1][a][1] for a in anc[1]]
    succ, pred = links(P, L)
    n = len(G)
    role = []
    for u in range(n):
        hp, hs = u in pred, u in succ
        role.append("isolated" if not hp and not hs else "root" if not hp else "end" if not hs else ("fork" if len(succ[u]) == 2 else "middle"))
    bd = None
    if boundary:
        import shapely
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
        REF = np.array(RR.SP.get_spectre_points(1.0, 1.0))
        groups = defaultdict(list)
        for (T, lab, b) in leaves:
            groups[levels[1][levels[0][b][2]][2]].append(Polygon((T[:, :2] @ REF.T).T + T[:, 2]).buffer(1e-6))
        ext = unary_union([unary_union(v) for v in groups.values()]).exterior
        bd = np.asarray(shapely.distance(shapely.points(P), ext))
    roles = {r: dict(n=role.count(r), supertile_labels=dict(Counter(lab1[u] for u in range(n) if role[u] == r).most_common()),
                     near_boundary=(float(np.mean([bd[u] < 12 for u in range(n) if role[u] == r])) if bd is not None else None)) for r in ("root", "middle", "fork", "end", "isolated")}
    trees = Counter(); cut = defaultdict(list)
    for r in [u for u in range(n) if role[u] == "root"]:
        stack, nodes, md = [(r, 0)], [], 0
        while stack:
            x, dp = stack.pop(); nodes.append(x); md = max(md, dp)
            stack += [(y, dp + 1) for y in succ.get(x, {}).values()]
        span = max([kk for kk in range(1, level) if len(set(anc[kk][nodes])) > 1] or [0])
        trees[(span, len(nodes), md)] += 1
        if bd is not None:
            ends = [x for x in nodes if x not in succ]
            cut[span].append((bool(bd[r] < 12), float(np.mean(bd[ends] < 12))))
    return dict(level=level, mystics=n, units=[len(levels[k]) for k in sorted(levels)], roles=roles,
                trees=[dict(span_level=k[0], mystics=k[1], depth=k[2], count=v) for k, v in sorted(trees.items())],
                boundary={str(k): dict(trees=len(v), root_at_boundary=float(np.mean([a for a, _ in v])), ends_at_boundary=float(np.mean([b for _, b in v]))) for k, v in sorted(cut.items())})


def forks_in_dumps(files):
    import stack_trace as ST
    events, byfork = Counter(), defaultdict(set)
    for f in files:
        dump = json.load(open(f))[0]
        d = ST.prepare(f)
        P, L, lab, nb = d["P"], d["L"], d["lab"], d["nb"]
        sp, u = defaultdict(list), 0
        for a, pl in enumerate(dump["tiles"]):                       # the Spectres of each dumped key, in prepare()'s order
            for Q in pl:
                for _ in range(1 if len(Q) == 14 else 2):
                    sp[a].append(u); u += 1
        assert u == len(P)
        G = [i for i in range(len(P)) if lab[i] == "Gamma1"]
        succ, _ = links(P[G], L[G])
        succ = {G[a]: {k: G[b] for k, b in v.items()} for a, v in succ.items()}
        for st in dump["steps"]:
            R = {v for a in st["removed"] for v in sp[a]}
            theta = lambda v: any(w in R and lab[w] == "Theta" for w in nb.get(v, []))     # the B motif of an active ribbon
            for i, s in succ.items():
                if len(s) < 2 or i not in R:
                    continue
                case = "".join(k for k in "xy" if s[k] in R and theta(s[k]))
                if case:
                    events[case] += 1; byfork[(f, i)].add(case)
    pat = Counter("+".join(sorted(v)) for v in byfork.values())
    return dict(events=dict(events), forks=len(byfork), by_fork=dict(pat),
                one_branch_only=pat.get("x", 0) + pat.get("y", 0), both_in_one_step=sum(v for k, v in pat.items() if "xy" in k),
                switches_between_steps=sum(v for k, v in pat.items() if "x" in k.split("+") and "y" in k.split("+")))


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    out = dict(level6=forest(6), level5=forest(5, boundary=True))
    for key in ("level6", "level5"):
        r = out[key]
        print("level %d: %d Mystics; units per level %s" % (r["level"], r["mystics"], r["units"]))
        for name, v in r["roles"].items():
            print("   %-8s %6d  supertile labels %s%s" % (name, v["n"], v["supertile_labels"], "" if v["near_boundary"] is None else "  within 12 of the patch boundary: %.2f" % v["near_boundary"]))
        for t in r["trees"]:
            print("   trees crossing supertile boundaries up to level %d: %4d with %4d Mystics, depth %3d" % (t["span_level"], t["count"], t["mystics"], t["depth"]))
        for k, v in r["boundary"].items():
            print("   span level %s: %d trees; root at the patch boundary %.2f; ends at the patch boundary %.2f" % (k, v["trees"], v["root_at_boundary"], v["ends_at_boundary"]))
    if files:
        out["forks"] = forks_in_dumps(files)
        print("forks inside observed active ribbons:", out["forks"])
    json.dump(out, open(os.path.join(HERE, "ribbon_trees.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        for key in ("level6", "level5"):
            ro = out[key]["roles"]
            assert set(ro["fork"]["supertile_labels"]) == {"Gamma"} and ro["fork"]["n"] <= out[key]["units"][2]
            assert set(ro["middle"]["supertile_labels"]) <= {"Delta", "Sigma", "Pi"}
        print("ribbon_trees: tests passed (forks are the Mystics of Gamma supertiles; middles lie in Delta, Sigma or Pi)")


if __name__ == "__main__":
    main()

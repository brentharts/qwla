#!/usr/bin/env python3
"""
One network for all ribbons: the five links between Mystics.  (ROADMAP Stage 10c.)

Every ribbon, silent or active, passes from Mystic to Mystic. Between the first tiles of consecutive Mystics there
are five relative poses (offset in the frame of the first, turn of the frame):

    s      (6 + 2 sqrt3, 3 - sqrt3),          0     active, straight on
    x      (9/2 + sqrt3, 3 + sqrt3/2),      -60     active turn; the same pose joins the stacks b -> a and c -> a of a silent strand
    y      (6 + 3 sqrt3/2, 9/2 + sqrt3),    +60     active turn
    alpha  (3/2 - sqrt3, 3 + 5 sqrt3/2),    +60     silent: stacks a -> c and c -> c
    beta   (6 + 3 sqrt3/2, 9/2),            -60     silent: stacks c -> b

Applied to every Mystic of a substitution patch (no phason step) they give one directed graph. Reported: the links in
and out of a Mystic as a function of the label of its level-1 supertile; connectivity; degrees; the part that is left
when dead ends are pruned (the 2-core), where an infinite ribbon has to run. With dumps from generic_phase.py: the
share of flipped Mystics that have at least two flipped neighbours in this graph (ribbons are paths of the graph).

Usage: python3 ribbon_combined.py [--level 6] [dump.json ...] [--tests]
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

S3 = math.sqrt(3.0)


def rot(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


LINKS = {"s": (np.array([6 + 2 * S3, 3 - S3]), rot(0)), "x": (np.array([4.5 + S3, 3 + S3 / 2]), rot(-60)), "y": (np.array([6 + 1.5 * S3, 4.5 + S3]), rot(60)),
         "alpha": (np.array([1.5 - S3, 3 + 2.5 * S3]), rot(60)), "beta": (np.array([6 + 1.5 * S3, 4.5]), rot(-60))}


def graph(P, L):
    tree = cKDTree(P)
    out, inn = defaultdict(dict), defaultdict(dict)
    for u in range(len(P)):
        for k, (p, M) in LINKS.items():
            for v in tree.query_ball_point(P[u] + L[u] @ p, 1e-3):
                if np.abs(L[u] @ M - L[v]).max() < 1e-6:
                    out[u][k] = v; inn[v][k] = u
    return out, inn


def two_core(n, out):
    adj = defaultdict(set)
    for u in out:
        for v in out[u].values():
            adj[u].add(v); adj[v].add(u)
    deg = {u: len(adj[u]) for u in range(n)}
    alive = set(range(n))
    stack = [u for u in alive if deg[u] <= 1]
    while stack:
        u = stack.pop()
        if u not in alive:
            continue
        alive.discard(u)
        for v in adj[u]:
            if v in alive:
                deg[v] -= 1
                if deg[v] <= 1:
                    stack.append(v)
    return alive, adj, Counter(sum(1 for v in adj[u] if v in alive) for u in alive)


def components(nodes, adj):
    seen, sizes = set(), []
    for u in nodes:
        if u in seen:
            continue
        st, k = [u], 0
        seen.add(u)
        while st:
            x = st.pop(); k += 1
            for y in adj[x]:
                if y in nodes and y not in seen:
                    seen.add(y); st.append(y)
        sizes.append(k)
    return sorted(sizes, reverse=True)


def patch(level):
    levels, leaves = RR.build(level)
    G = [i for i, (T, lab, b) in enumerate(leaves) if lab == "Gamma1"]
    P = np.array([leaves[i][0][:, 2] for i in G]); L = np.array([leaves[i][0][:, :2] for i in G])
    lab1 = [levels[1][levels[0][leaves[i][2]][2]][1] for i in G]
    out, inn = graph(P, L)
    n = len(G)
    gram = defaultdict(Counter)
    for u in range(n):
        gram[lab1[u]][("+".join(sorted(inn[u])) or "-", "+".join(sorted(out[u])) or "-")] += 1
    edges = Counter((lab1[u], k, lab1[v]) for u in out for k, v in out[u].items())
    core, adj, coredeg = two_core(n, out)
    deg = Counter(len(adj[u]) for u in range(n))
    return dict(level=level, mystics=n, links=sum(len(v) for v in out.values()),
                grammar={k: {"in %s | out %s" % kk: c for kk, c in v.most_common()} for k, v in sorted(gram.items())},
                edges={"%s -%s-> %s" % k: c for k, c in edges.most_common()},
                components=components(set(range(n)), adj)[:3], degrees={str(k): v for k, v in sorted(deg.items())},
                core=len(core), core_share=len(core) / n, core_components=components(core, adj)[:3], core_degrees={str(k): v for k, v in sorted(coredeg.items())},
                core_labels=dict(Counter(lab1[u] for u in core).most_common()))


def observed(files):
    import stack_trace as ST
    tot = Counter()
    for f in files:
        dump = json.load(open(f))[0]
        d = ST.prepare(f)
        P, L, lab = d["P"], d["L"], d["lab"]
        sp, u = defaultdict(list), 0
        for a, pl in enumerate(dump["tiles"]):
            for Q in pl:
                for _ in range(1 if len(Q) == 14 else 2):
                    sp[a].append(u); u += 1
        G = [i for i in range(len(P)) if lab[i] == "Gamma1"]
        out, inn = graph(P[G], L[G])
        nbr = defaultdict(set)
        for a in out:
            for b in out[a].values():
                nbr[G[a]].add(G[b]); nbr[G[b]].add(G[a])
        for st in dump["steps"]:
            R = {v for a in st["removed"] for v in sp[a]}
            for u in R:
                if lab[u] == "Gamma1" and np.linalg.norm(P[u]) < d["R"] - 25:
                    tot[min(3, sum(1 for v in nbr[u] if v in R))] += 1
    n = sum(tot.values())
    return dict(flipped_mystics=n, flipped_neighbours={str(k): v for k, v in sorted(tot.items())}, share_with_two_or_more=(tot[2] + tot[3]) / max(1, n))


def main():
    level = int(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else 6
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    out = dict(patches=[patch(lv) for lv in range(4, level + 1)])
    for p in out["patches"]:
        print("level %d: %d Mystics, %d links; components %s; degrees %s; 2-core %d (%.3f), components %s, degrees %s" % (
            p["level"], p["mystics"], p["links"], p["components"], p["degrees"], p["core"], p["core_share"], p["core_components"], p["core_degrees"]))
    p = out["patches"][-1]
    print("links in and out of a Mystic, by the label of its supertile (level %d):" % p["level"])
    for k, v in p["grammar"].items():
        print("   %-7s %s" % (k, v))
    print("labels in the 2-core:", p["core_labels"])
    if files:
        out["observed"] = observed(files)
        print("observed:", out["observed"])
    json.dump(out, open(os.path.join(HERE, "ribbon_combined.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        assert all(len(q["components"]) == 1 for q in out["patches"])                       # one connected network
        g4, g6 = out["patches"][0]["grammar"], out["patches"][-1]["grammar"]
        assert all(set(g4[k]) <= set(g6[k]) for k in g4)                                    # the same grammar at every level
        print("ribbon_combined: tests passed (one connected network at every level; the same link grammar)")


if __name__ == "__main__":
    main()

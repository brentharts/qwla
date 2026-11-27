#!/usr/bin/env python3
"""
The network of possible active ribbons in a Spectre tiling, and the law of their words.  (ROADMAP Stage 9v.)

In an active ribbon (ribbon_words.py) the first Mystic tile of each motif is followed by the first Mystic tile of the
next motif at one of three poses, in the frame of the first:

    s   straight on      offset (6 + 2 sqrt3, 3 - sqrt3),          same orientation       (to an A motif, or after a B)
    x   one turn         offset (9/2 + sqrt3, 3 + sqrt3/2),        turned by -60 degrees  (to a B motif)
    y   the other turn   offset (6 + 3 sqrt3/2, 9/2 + sqrt3),      turned by +60 degrees  (to a B motif)

This script applies these three links to EVERY Mystic of a substitution patch, with no phason step: the result is the
network of all chains a ribbon could follow. Reported: how many links leave and enter a Mystic, where the network
forks, how long the chains are, the words along the longest chains, and (given dumps) whether the words of the ribbons
actually observed at generic phases occur in them.

Usage: python3 ribbon_network.py [--level 6] [dump.json ...] [--tests]
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
ROT = lambda deg: np.array([[math.cos(math.radians(deg)), -math.sin(math.radians(deg))], [math.sin(math.radians(deg)), math.cos(math.radians(deg))]])
LINKS = {"s": (np.array([6 + 2 * S3, 3 - S3]), ROT(0)), "x": (np.array([4.5 + S3, 3 + S3 / 2]), ROT(-60)), "y": (np.array([6 + 1.5 * S3, 4.5 + S3]), ROT(60))}


def network(level):
    levels, leaves = RR.build(level)
    G = [i for i, (T, lab, b) in enumerate(leaves) if lab == "Gamma1"]
    P = np.array([leaves[i][0][:, 2] for i in G]); L = np.array([leaves[i][0][:, :2] for i in G])
    tree = cKDTree(P)
    succ, pred = defaultdict(dict), defaultdict(dict)
    for u in range(len(G)):
        for k, (p, M) in LINKS.items():
            for v in tree.query_ball_point(P[u] + L[u] @ p, 1e-3):
                if np.abs(L[u] @ M - L[v]).max() < 1e-6:
                    succ[u][k] = v; pred[v][k] = u
    return len(G), succ, pred, [len(levels[k]) for k in sorted(levels)]


def chains(n, pred):
    up = {v: (u, k) for v in pred for k, u in pred[v].items()}
    depth = {}
    for v0 in range(n):
        path, v = [], v0
        while v in up and v not in depth:
            path.append(v); v = up[v][0]
        d = depth.setdefault(v, 0)
        for x in reversed(path):
            d += 1; depth[x] = d
    D = np.array([depth.get(v, 0) for v in range(n)])
    words, used = [], set()
    for v in np.argsort(-D):
        if v in used or D[v] == 0:
            continue
        w, x = [], v
        while x in up:
            w.append(up[x][1]); used.add(x); x = up[x][0]
        words.append("".join(reversed(w)))
    words.sort(key=len, reverse=True)
    return D, words


def main():
    level = int(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else 6
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    out = dict(levels=[])
    longest = {}
    for lv in range(3, level + 1):
        n, succ, pred, units = network(lv)
        D, words = chains(n, pred)
        so = Counter("".join(sorted(succ[u])) if u in succ else "-" for u in range(n))
        po = Counter(len(pred[u]) if u in pred else 0 for u in range(n))
        longest[lv] = words[0]
        rec = dict(level=lv, mystics=n, successors={k: v for k, v in sorted(so.items())}, predecessors={str(k): v for k, v in sorted(po.items())},
                   level2_supertiles=units[2] if len(units) > 2 else None, longest_chain=len(words[0]), chains_at_least_half=sum(len(w) >= len(words[0]) / 2 for w in words))
        out["levels"].append(rec)
        print("level %d: %d Mystics; links out %s; links in %s; level-2 supertiles %s; longest chain %d links" % (lv, n, rec["successors"], rec["predecessors"], rec["level2_supertiles"], rec["longest_chain"]))
    ls = [longest[lv] for lv in sorted(longest)]
    out["longest_lengths"] = [len(w) for w in ls]
    out["length_ratios"] = [len(b) / len(a) for a, b in zip(ls, ls[1:])]
    w = ls[-1]
    cnt = Counter(w)
    out["longest_word"] = w
    out["letters"] = dict(cnt); out["straight_per_turn"] = cnt["s"] / (cnt["x"] + cnt["y"])
    print("longest chains: lengths %s, ratios %s (lambda = %.4f)" % (out["longest_lengths"], ["%.3f" % r for r in out["length_ratios"]], RR.LAM))
    print("longest word at level %d (%d letters): %s" % (level, len(w), w))
    print("letters %s; s per turn %.4f; is each shorter longest word a prefix of the next? %s" % (dict(cnt), out["straight_per_turn"], [b.startswith(a) for a, b in zip(ls, ls[1:])]))
    D, words = chains(*(lambda t: (t[0], t[2]))(network(level)))
    top = words[:60]
    comp = []
    for k in range(1, 13):
        fac = set()
        for x in top:
            for i in range(len(x) - k + 1):
                fac.add(x[i:i + k])
        comp.append(len(fac))
    out["factor_complexity"] = comp
    print("number of distinct factors of length 1..12 in the 60 longest words:", comp)
    if files:
        import ribbon_words as RW
        import stack_trace as ST
        import tile_stacks as TS
        big = "|".join(words)
        hits = []
        for f in files:
            d = ST.prepare(f)
            for g in d["ribs"]:
                if TS.stacks(d["P"], d["L"], g):
                    continue
                seq, covered, labelled, s = RW.active_word(d, g)
                t = "".join("s" if (a2 - a1) % 360 == 0 else ("x" if (a2 - a1) % 360 == 60 else "y") for (k1, a1, h1), (k2, a2, h2) in zip(seq, seq[1:]))
                alt = t.translate(str.maketrans("xy", "yx"))
                core, acore = t[1:-1], alt[1:-1]
                hits.append(dict(tiles=len(g), letters=len(t), as_is=bool(t and t in big), swapped=bool(alt and alt in big),
                                 core_as_is=bool(core and core in big), core_swapped=bool(acore and acore in big)))
                print("  observed active ribbon, %3d tiles, %2d links: word found in the network %s (x,y swapped: %s); without its end links: %s (%s)" % (
                    len(g), len(t), hits[-1]["as_is"], hits[-1]["swapped"], hits[-1]["core_as_is"], hits[-1]["core_swapped"]))
        out["observed"] = hits
    json.dump(out, open(os.path.join(HERE, "ribbon_network.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        for rec in out["levels"]:
            assert set(rec["predecessors"]) <= {"0", "1"}                       # never two links into a Mystic
            assert set(rec["successors"]) <= {"-", "s", "xy"}                  # straight on, or a fork into both turns
            assert rec["level2_supertiles"] is None or rec["successors"].get("xy", 0) == rec["level2_supertiles"]
        print("ribbon_network: tests passed (one link in at most; out: none, straight, or a fork; one fork per level-2 supertile)")


if __name__ == "__main__":
    main()

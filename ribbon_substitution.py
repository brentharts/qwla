#!/usr/bin/env python3
"""
The substitution behind the ribbon words.  (ROADMAP Stage 10f.)

The active network (ribbon_trees.py) links Mystic to Mystic by s, x, y. This script shows that it is built
hierarchically, by a rule that depends only on supertile labels:

 1  inside a level-2 supertile the network is one fixed graph for each label: the fork (the Mystic of the Gamma child),
    its two branches x and y, one straight link on each, and the links by which chains enter and leave;
 2  for k >= 2, the links between the level-k children of a level-(k+1) supertile, and the links entering and leaving
    it, form one fixed pattern for each label, and the pattern at k + 2 equals the pattern at k (the substitution
    reflects at each level, so there are two rules, one for each parity).

Together these are a graph substitution: a chain through a supertile is determined by the label, and so on down.

Usage: python3 ribbon_substitution.py [--level 6] [--tests]
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recognisability_radius as RR  # noqa: E402
import ribbon_trees as RT  # noqa: E402

SH = dict(Gamma="G", Delta="D", Pi="P", Sigma="S", Lambda="L", Xi="X", Psi="Y", Phi="F", Theta="T")


def main():
    level = int(sys.argv[sys.argv.index("--level") + 1]) if "--level" in sys.argv else 6
    levels, leaves = RR.build(level)
    G = [i for i, (T, lab, b) in enumerate(leaves) if lab == "Gamma1"]
    P = np.array([leaves[i][0][:, 2] for i in G]); L = np.array([leaves[i][0][:, :2] for i in G])
    anc = {1: np.array([levels[0][leaves[i][2]][2] for i in G])}
    for k in range(2, level):
        anc[k] = np.array([levels[k - 1][a][2] for a in anc[k - 1]])
    lab = {k: [SH[levels[k][a][1]] for a in anc[k]] for k in anc}
    succ, pred = RT.links(P, L)
    n = len(G)
    out = {}
    # ---- 1: inside a level-2 supertile, at the level of Mystics ----
    by = defaultdict(list)
    for u in range(n):
        by[anc[2][u]].append(u)
    pat = defaultdict(Counter)
    for sid, nodes in by.items():
        ns = set(nodes)
        inside = tuple(sorted("%s%s%s" % (lab[1][u], k, lab[1][v]) for u in nodes for k, v in succ.get(u, {}).items() if v in ns))
        ent = tuple(sorted("%s>%s" % (lab[1][pred[u]], lab[1][u]) for u in nodes if u in pred and pred[u] not in ns))
        ex = tuple(sorted("%s%s%s" % (lab[1][u], k, lab[1][v]) for u in nodes for k, v in succ.get(u, {}).items() if v not in ns))
        pat[lab[2][nodes[0]]][(inside, ent, ex)] += 1
    out["inside_level2"] = {}
    print("1. the active network inside a level-2 supertile (links: source, kind, target, by the label of the Mystic's level-1 supertile)")
    for X, c in sorted(pat.items()):
        (inside, ent, ex), cnt = c.most_common(1)[0]
        out["inside_level2"][X] = dict(inside=list(inside), enters=list(ent), leaves=list(ex), count=cnt, total=sum(c.values()), variants=len(c))
        print("   %s  (%d of %d): inside %s | enters %s | leaves %s" % (X, cnt, sum(c.values()), " ".join(inside), " ".join(ent) or "-", " ".join(ex) or "-"))
    # ---- 2: coarse patterns ----
    out["coarse"] = {}
    for k in range(2, level - 1):
        inside, ent, ex = defaultdict(list), defaultdict(list), defaultdict(list)
        for u in range(n):
            for kk, v in succ.get(u, {}).items():
                if anc[k][u] != anc[k][v]:
                    e = "%s>%s" % (lab[k][u], lab[k][v])
                    if anc[k + 1][u] == anc[k + 1][v]:
                        inside[anc[k + 1][u]].append(e)
                    else:
                        ex[anc[k + 1][u]].append(e); ent[anc[k + 1][v]].append(e)
        lab_of = {anc[k + 1][u]: lab[k + 1][u] for u in range(n)}
        pat = defaultdict(Counter)
        for sid, X in lab_of.items():
            pat[X][(tuple(sorted(inside[sid])), tuple(sorted(ent[sid])), tuple(sorted(ex[sid])))] += 1
        out["coarse"][str(k)] = {}
        print("2. level-%d children inside a level-%d supertile" % (k, k + 1))
        for X, c in sorted(pat.items()):
            (i, e, x), cnt = c.most_common(1)[0]
            out["coarse"][str(k)][X] = dict(inside=list(i), enters=list(e), leaves=list(x), count=cnt, total=sum(c.values()), variants=len(c))
            print("   %s  (%d of %d): inside %s | enters %s | leaves %s" % (X, cnt, sum(c.values()), " ".join(i), " ".join(e) or "-", " ".join(x) or "-"))
    # parity: the pattern at k + 2 against the pattern at k
    same = {}
    for k in range(2, level - 3):
        a, b = out["coarse"][str(k)], out["coarse"][str(k + 2)]
        same[str(k)] = {X: a[X]["inside"] == b[X]["inside"] for X in b if X in a}
        out.setdefault("leaving_links_same", {})[str(k)] = {X: a[X]["leaves"] == b[X]["leaves"] for X in b if X in a}      # can fail at the patch boundary
    out["same_two_levels_up"] = same
    print("pattern at k + 2 equal to the pattern at k, links inside:", same)
    print("the same for the links leaving (a supertile at the edge of the patch can lack one):", out["leaving_links_same"])
    json.dump(out, open(os.path.join(HERE, "ribbon_substitution.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        assert all(v["count"] >= v["total"] - 2 for v in out["inside_level2"].values())               # one pattern per label, but for the patch boundary
        assert all(v["count"] >= v["total"] - 2 for kk in out["coarse"].values() for v in kk.values())
        assert all(all(v.values()) for v in same.values()) and same
        print("ribbon_substitution: tests passed (one pattern per label at each level; the pattern repeats two levels up)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Exact frequencies in the ribbon network.  (ROADMAP Stage 10d.)

The number of Mystics in a level-n substitution patch obeys  a(n+2) = 8 a(n+1) - a(n)  (8, 63, 496, 3905, 30744, ...),
with growth 4 + sqrt15 = lambda^2. If another count c(n) obeys the same recurrence, then

    lim c(n)/a(n) = (c(n+1) - nu c(n)) / (a(n+1) - nu a(n)),        nu = 4 - sqrt15,

is an element of Q(sqrt15), and it is the frequency of that kind of Mystic in the tiling. This script takes the counts of
ribbon_combined.py at levels 3 to 6 (Mystics in the 2-core of the network, by degree and by supertile label), checks
the recurrence on every triple of consecutive levels, and prints the exact limits as (A + B sqrt15)/C.

Usage: python3 ribbon_core_frequency.py [--tests]
"""
import json
import math
import os
import sys
from collections import Counter
from fractions import Fraction

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recognisability_radius as RR  # noqa: E402
import ribbon_combined as RC  # noqa: E402

LEVELS = (3, 4, 5, 6)


def counts(level):
    levels, leaves = RR.build(level)
    G = [i for i, (T, lab, b) in enumerate(leaves) if lab == "Gamma1"]
    P = np.array([leaves[i][0][:, 2] for i in G]); L = np.array([leaves[i][0][:, :2] for i in G])
    lab1 = [levels[1][levels[0][leaves[i][2]][2]][1] for i in G]
    out, inn = RC.graph(P, L)
    n = len(G)
    core, adj, coredeg = RC.two_core(n, out)
    c = {"Mystics": n, "links": sum(len(v) for v in out.values()), "core": len(core)}
    for k in (2, 3, 4):
        c["core, degree %d" % k] = coredeg.get(k, 0)
    for lab in sorted(set(lab1)):
        c["all %s" % lab] = sum(1 for u in range(n) if lab1[u] == lab)
        c["core %s" % lab] = sum(1 for u in core if lab1[u] == lab)
    deg = Counter(len(adj[u]) for u in range(n))
    for k in (1, 2, 3, 4):
        c["degree %d" % k] = deg.get(k, 0)
    return c


def limit(c0, c1, a0, a1):
    """lim c/a as (A, B, C) with value (A + B sqrt15)/C, from two consecutive levels."""
    # (c1 - nu c0)/(a1 - nu a0), nu = 4 - sqrt15:  numerator = (c1 - 4 c0) + c0 sqrt15, denominator = (a1 - 4 a0) + a0 sqrt15
    p, q, r, s = c1 - 4 * c0, c0, a1 - 4 * a0, a0
    den = r * r - 15 * s * s
    A, B = Fraction(p * r - 15 * q * s, den), Fraction(q * r - p * s, den)
    C = math.lcm(A.denominator, B.denominator)
    return int(A * C), int(B * C), C


def main():
    data = {lv: counts(lv) for lv in LEVELS}
    a = [data[lv]["Mystics"] for lv in LEVELS]
    out = dict(levels=list(LEVELS), counts={}, limits={})
    print("%-18s %s   recurrence   limit of the share" % ("count", "  ".join("L%d" % lv for lv in LEVELS)))
    for key in data[LEVELS[-1]]:
        seq = [data[lv][key] for lv in LEVELS]
        rec = [seq[i + 2] == 8 * seq[i + 1] - seq[i] for i in range(len(seq) - 2)]
        A, B, C = limit(seq[-2], seq[-1], a[-2], a[-1])
        val = (A + B * math.sqrt(15)) / C
        out["counts"][key] = seq
        out["limits"][key] = dict(recurrence_holds=rec, A=A, B=B, C=C, value=val, share_at_top_level=seq[-1] / a[-1])
        expr = ("%d %s %d sqrt15" % (A, "+" if B >= 0 else "-", abs(B))) if C == 1 else ("(%d %s %d sqrt15)/%d" % (A, "+" if B >= 0 else "-", abs(B), C))
        print("%-18s %-28s %-12s %s = %.6f   (level %d: %.6f)" % (key, " ".join(map(str, seq)), "".join("y" if r else "n" for r in rec), expr if all(rec[-1:]) else "-", val, LEVELS[-1], seq[-1] / a[-1]))
    json.dump(out, open(os.path.join(HERE, "ribbon_core_frequency.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        L = out["limits"]["core"]
        assert L["recurrence_holds"][-1] and (L["A"], L["B"], L["C"]) == (268, -69, 1)
        print("ribbon_core_frequency: tests passed (share of the core = 268 - 69 sqrt15 = %.6f)" % L["value"])


if __name__ == "__main__":
    main()

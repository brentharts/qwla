#!/usr/bin/env python3
"""
Ribbons as words: the active ribbons, and both kinds side by side.  (ROADMAP Stage 9u.)

Silent ribbons are chains of stacks with a turn rule (stack_trace.py, Stage 9t). Active ribbons have no stacks. Read
along the chain, their tile labels spell two motifs,

    A = Phi Sigma Xi Mystic      (5 tiles; Phi and Sigma may come in either order in the chain's tile order)
    B = Theta Sigma Mystic       (4 tiles)

and nothing else. This script parses every stack-free ribbon into its word, measures the turn of the first Mystic tile's
orientation from one motif to the next, and tests what decides the turn: the label of the level-1 supertile that holds
the Mystic of a B motif (its Theta and Sigma tiles are always in a Gamma supertile).

Usage: python3 ribbon_words.py dump.json [...] [--tests]      (dumps from generic_phase.py DUMP=...; uses stack_trace's cache)
"""
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tile_stacks as TS  # noqa: E402
import stack_trace as ST  # noqa: E402

SH = dict(Gamma1="G", Gamma2="g", Delta="D", Pi="P", Sigma="S", Lambda="L", Xi="X", Psi="Y", Phi="F", Theta="T", Gamma="G")
MOTIF = re.compile(r"(FSXGg|SFXGg|TSGg|STGg)")


def chain_order(d, g):
    nb, cent = d["nb"], d["cent"]
    gs = set(g)
    c = cent[g]; X = c - c.mean(0)
    w_, v_ = np.linalg.eigh(X.T @ X)
    start = g[int(np.argmin(X @ v_[:, 1]))]
    dist = {start: 0}; q = [start]
    while q:
        x = q.pop(0)
        for y in nb.get(x, []):
            if y in gs and y not in dist:
                dist[y] = dist[x] + 1; q.append(y)
    return sorted(dist, key=lambda u: (dist[u], u))


def active_word(d, g):
    """Motifs of a stack-free ribbon, read in the direction in which a Mystic's first tile precedes its second."""
    lab, par, L = d["lab"], d["par"], d["L"]
    order = chain_order(d, g)
    s = "".join(SH.get(lab[u], "?") for u in order)
    if s.count("gG") > s.count("Gg"):
        order = order[::-1]; s = s[::-1]
    seq, covered = [], 0
    for m in MOTIF.finditer(s):
        tiles = order[m.start():m.end()]
        gi = order[m.start() + m.group().index("G")]
        ang = round(math.degrees(math.atan2(L[gi][1, 0], L[gi][0, 0]))) % 360
        kind = "A" if "X" in m.group() else "B"
        host = None
        if kind == "B":                                           # supertile of the two Mystic tiles
            hs = {par[u] for u in tiles if lab[u] in ("Gamma1", "Gamma2")}
            host = hs.pop() if len(hs) == 1 else "mixed"
        seq.append((kind, ang, host)); covered += len(tiles)
    labelled = sum(c != "?" for c in s)
    return seq, covered, labelled, s


def main():
    files = [a for a in sys.argv[1:] if a.endswith(".json")]
    rule, by_host, rows = defaultdict(Counter), defaultdict(Counter), []
    letters_active, letters_silent = [], []
    for f in files:
        d = ST.prepare(f)
        for g in d["ribs"]:
            st = TS.stacks(d["P"], d["L"], g)
            share = sum(len(s) for s, _ in st) / len(g)
            if share >= 0.7:                                      # a silent strand: its motifs, from the turn string
                items, tn = ST.walk(d, g, st)
                t = "".join(tn)
                mot = re.findall(r"RRLL|RLL|RRL", t)
                letters_silent.append("".join({"RRLL": "s", "RLL": "l", "RRL": "r"}[m] for m in mot))
                rows.append(dict(file=os.path.basename(f), kind="silent", tiles=len(g), turns=t, letters=letters_silent[-1]))
                continue
            if st:
                continue                                          # a mixture of the two kinds
            seq, covered, labelled, s = active_word(d, g)
            word, turns, letters = "".join(k for k, _, _ in seq), [], []
            for (k1, a1, h1), (k2, a2, h2) in zip(seq, seq[1:]):
                t = {0: "0", 60: "L", -60: "R"}.get((a2 - a1 + 180) % 360 - 180, "?")
                turns.append(t); rule[(k1, k2)][t] += 1
                if k2 == "B":
                    by_host[h2][t] += 1
            for k, a, h in seq:
                letters.append("s" if k == "A" else {"Sigma": "l", "Delta": "r"}.get(h, "?"))
            letters_active.append("".join(letters))
            heading = np.cumsum([{"L": 1, "R": -1}.get(t, 0) for t in turns]) if turns else np.array([0])
            rows.append(dict(file=os.path.basename(f), kind="active", tiles=len(g), labelled=labelled, in_motifs=covered, word=word, turns="".join(turns),
                             letters=letters_active[-1], A=word.count("A"), B=word.count("B"), heading_range=[int(heading.min()), int(heading.max())]))
            print("active ribbon %3d tiles: %d of %d labelled tiles in motifs; word %s\n      turns %s   heading stays in %s (units of 60 degrees)" % (
                len(g), covered, labelled, word, " " + "".join(turns), rows[-1]["heading_range"]))
    act = [r for r in rows if r["kind"] == "active"]
    nA, nB = sum(r["A"] for r in act), sum(r["B"] for r in act)
    summ = dict(active=len(act), silent=sum(r["kind"] == "silent" for r in rows), motifs_A=nA, motifs_B=nB, A_per_B=nA / max(1, nB),
                coverage=[sum(r["in_motifs"] for r in act), sum(r["labelled"] for r in act)],
                turn_by_motifs={"%s>%s" % k: dict(v) for k, v in sorted(rule.items())},
                turn_at_B_by_supertile={str(k): dict(v) for k, v in sorted(by_host.items(), key=str)},
                letters_active=letters_active, letters_silent=letters_silent)
    print("turn between motifs:", summ["turn_by_motifs"])
    print("turn at a B motif, by the supertile of its Mystic:", summ["turn_at_B_by_supertile"])
    print("A per B: %.3f (%d, %d); tiles in motifs: %d of %d labelled" % (summ["A_per_B"], nA, nB, *summ["coverage"]))
    print("letters (s straight, l left, r right)  active:", letters_active)
    print("                                       silent:", letters_silent)
    json.dump(dict(summary=summ, ribbons=rows), open(os.path.join(HERE, "ribbon_words.json"), "w"), indent=1)
    if "--tests" in sys.argv:
        assert set(rule[("A", "A")]) <= {"0"} and set(rule[("B", "A")]) <= {"0"}
        good = by_host["Sigma"]["L"] + by_host["Delta"]["R"]; tot = sum(sum(v.values()) for v in by_host.values())
        assert good >= 0.95 * tot
        print("ribbon_words: tests passed (A never turns; the supertile of a B motif gives its turn in %d of %d cases)" % (good, tot))


if __name__ == "__main__":
    main()

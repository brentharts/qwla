#!/usr/bin/env python3
"""
Stage 8 of ROADMAP_QUANTUM_GRAVITY.md: does a local tile move change the cone data of a closed fold?

The wada-coudene flip data (region-level flips of the planar tiling) are not used here: they carry no fold, and the
engine's placement files are not in this workspace.  What the fold code can measure is the nearest thing: a local
*patch move* on a closed Spectre fold.  Start from a disc patch of N tiles that folds into a closed ball (exact DP),
and apply one of

  add     one tile from the frontier of the patch                     (N -> N+1)
  remove  one tile of the patch                                        (N -> N-1)
  swap    remove one tile, add one frontier tile                       (N -> N)

keeping the result only if it is again a closed fold.  For each pair (before, after) record the cone multisets, the
number of positive cone points n, and the Thurston moduli dimension max(0, n-3).  Ledger = 720 degrees in both.

Honest scope: this is a sample of moves on random patches, not the flip relation of the tiling.  A planar flip that
is strictly interior leaves the patch boundary alone, hence (the cone points of a fold sit on boundary corners) the
cone data; a move at the boundary may change them.  The measurement below is of boundary moves.

Run:  BALLGEN_DIR=/path/to/ballgen python3 cone_moves.py [draws_per_size] [seed]      (tests: --tests-only)
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter

from cone_ledger import _ballgen, moduli_dim, check_ledger, TOTAL


def closed_cones(pool, tiles, can_close, fold_with_holes):
    """Sorted deficits of the cone points of the fold of this tile set, or None if it is not a closed disc fold."""
    g = pool.graph(list(tiles))
    if g is None:
        return None
    A = [int(a) for a in g.loops[0][1]]
    if not can_close(A):
        return None
    r = fold_with_holes(g, [])
    if r is None or r.holes:
        return None
    return sorted(c.deficit for c in r.cones)


def positive(ds):
    return [d for d in ds if d > 0]


def neighbours(pool, tiles):
    s = set(tiles)
    fr = set()
    for t in s:
        fr |= pool.nbrs[t]
    return sorted(fr - s)


def moves(pool, tiles, rng, per_kind=6):
    """Random move targets: ('add'|'remove'|'swap', new tile list)."""
    s = list(tiles)
    out = []
    fr = neighbours(pool, s)
    for _ in range(per_kind):
        if fr:
            out.append(("add", s + [rng.choice(fr)]))
        if len(s) > 4:
            r = rng.choice(s)
            out.append(("remove", [t for t in s if t != r]))
            if fr:
                out.append(("swap", [t for t in s if t != r] + [rng.choice(fr)]))
    return out


def survey(per_n=40, seed=5, Ns=range(5, 20), levels=3):
    _ballgen()
    from spectre.patch import SpectrePatch
    from spectre.search import PatchPool
    from spectre.dp import can_close, fold_with_holes
    pool = PatchPool(SpectrePatch(levels=levels))
    rng = random.Random(seed)
    rows = []
    tried = Counter()
    for N in Ns:
        for _ in range(per_n):
            t0 = pool.random_subset(N, rng, rng.choice([0.0, 0.3, 0.7]))
            d0 = closed_cones(pool, t0, can_close, fold_with_holes)
            if d0 is None:
                continue
            for kind, t1 in moves(pool, t0, rng):
                tried[kind] += 1
                d1 = closed_cones(pool, t1, can_close, fold_with_holes)
                if d1 is None:
                    continue
                p0, p1 = positive(d0), positive(d1)
                rows.append({"kind": kind, "N0": N, "N1": len(t1), "before": p0, "after": p1,
                             "ledger0": sum(d0), "ledger1": sum(d1)})
    return rows, dict(tried)


def summarise(rows, tried):
    by = {}
    for kind in ("add", "remove", "swap"):
        rs = [r for r in rows if r["kind"] == kind]
        dn = Counter(len(r["after"]) - len(r["before"]) for r in rs)
        same = sum(1 for r in rs if r["before"] == r["after"])
        by[kind] = {
            "tried": tried.get(kind, 0), "closed_after": len(rs),
            "same_cone_multiset": same,
            "change_in_cone_count": dict(sorted(dn.items())),
            "change_in_moduli_dim": dict(sorted(Counter(moduli_dim(r["after"]) - moduli_dim(r["before"]) for r in rs).items())),
        }
    allr = rows
    return {
        "pairs": len(allr),
        "ledger_720_both": all(r["ledger0"] == TOTAL and r["ledger1"] == TOTAL for r in allr),
        "cone_count_changed": sum(1 for r in allr if len(r["after"]) != len(r["before"])),
        "multiset_changed": sum(1 for r in allr if r["before"] != r["after"]),
        "by_kind": by,
    }


# ---- tests ------------------------------------------------------------------------------------
def test_positive_filter():
    assert positive([0, 90, -30, 120]) == [90, 120]


def test_moves_have_the_right_sizes():
    class P:
        nbrs = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}, 4: set(), 5: set()}
    rng = random.Random(1)
    out = moves(P, [0, 1, 2, 3, 4], rng, per_kind=2)
    for kind, t in out:
        assert len(t) == {"add": 6, "remove": 4, "swap": 5}[kind] or kind == "add"
    assert {k for k, _ in out} <= {"add", "remove", "swap"}


def test_summary_counts():
    rows = [{"kind": "swap", "N0": 6, "N1": 6, "before": [90] * 4 + [120] * 3, "after": [90] * 4 + [120] * 3, "ledger0": 720, "ledger1": 720},
            {"kind": "add", "N0": 6, "N1": 7, "before": [90] * 4 + [120] * 3, "after": [180, 180, 120, 120, 120], "ledger0": 720, "ledger1": 720}]
    s = summarise(rows, {"swap": 3, "add": 2})
    assert s["pairs"] == 2 and s["cone_count_changed"] == 1 and s["multiset_changed"] == 1 and s["ledger_720_both"]
    assert s["by_kind"]["add"]["change_in_cone_count"] == {-2: 1}
    assert s["by_kind"]["add"]["change_in_moduli_dim"] == {-2: 1}


def test_ledger_helper_agrees():
    assert check_ledger([90] * 4 + [120] * 3)


def run_tests():
    fs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fs:
        f()
        print("ok", f.__name__)
    print("%d checks pass" % len(fs))


def main():
    per_n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    rows, tried = survey(per_n, seed)
    s = summarise(rows, tried)
    json.dump({"summary": s, "rows": rows}, open("cone_moves.json", "w"))
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    if "--tests-only" in sys.argv:
        run_tests()
    else:
        run_tests()
        main()

#!/usr/bin/env python3
"""
Stage 7 of ROADMAP_QUANTUM_GRAVITY.md: the cone-point ledger.

A closed surface made of flat tiles is flat except at its vertices; a vertex where the tile corners add up to
`a` degrees is a *cone point* with deficit `360 - a`.  The discrete Gauss-Bonnet (Descartes) theorem says the
deficits of a closed sphere add up to 720 degrees (4 pi).  In 2+1 gravity this is the whole content of
"point particles on a closed universe": each cone point is a particle, its deficit is its mass, and the
total mass is fixed by topology (Deser-Jackiw-'t Hooft).  This module computes that ledger for the three
substrates we have, with no new physics assumed:

  * the regular tetrahedron (the flat model of the Trionda):          4 cone points x 180 deg
  * the Hat Trick ball (flat hats on the 92-tile geodesic sphere):    12 cone points x 60 deg (pentagonal holes)
  * closed folds of random patches of Spectre tiles:                  a multiset of deficits in multiples of 30 deg

and the dimension of the moduli space of flat cone metrics (Thurston, "Shapes of polyhedra", 1998): n cone
points of positive curvature on a sphere give a complex-hyperbolic cone manifold of complex dimension n - 3.

Labels used here as in the roadmap: **Exact** (proved or exact finite computation), **Counted** (one sample),
**Hypothesis**.  Nothing below is a derivation of quantum gravity; the 2+1 reading is a dictionary.

The Spectre folding code and the Hat Trick code are in `brentharts/ballgen`; set BALLGEN_DIR to a clone
(default: a sibling directory called `ballgen`).

    python3 cone_ledger.py --tests-only
    python3 cone_ledger.py                       # tetrahedron, Hat Trick, Spectre survey
    python3 cone_ledger.py survey 300 5          # draws per size, seed
"""
import json
import math
import os
import random
import sys
from collections import Counter
from fractions import Fraction

FULL = 360
TOTAL = 720                      # sum of the deficits of a closed sphere, in degrees (4 pi)


def _ballgen():
    d = os.environ.get("BALLGEN_DIR")
    here = os.path.dirname(os.path.abspath(__file__))
    for c in ([d] if d else []) + [os.path.join(here, "..", "ballgen"), os.path.join(here, "ballgen")]:
        if c and os.path.exists(os.path.join(c, "the_hat_trick_ball.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return os.path.abspath(c)
    raise SystemExit("set BALLGEN_DIR to a clone of brentharts/ballgen (needs the_hat_trick_ball.py and spectre/)")


# ---------------------------------------------------------------------------------------------
# generic bookkeeping
# ---------------------------------------------------------------------------------------------
def deficit(angle_sum):
    """Angle deficit (degrees) at a vertex whose tile corners add up to angle_sum."""
    return FULL - angle_sum


def cone_angle(d):
    return FULL - d


def moduli_dim(deficits):
    """Complex dimension of the moduli space of flat metrics on the sphere with these positive deficits
    (Thurston 1998): n - 3 for n >= 3 cone points; 0 for n <= 3 (the metric is rigid up to scale; n = 2 is a
    football, n = 1 does not exist).  Deficits of zero are not singular points.  **Exact** (a theorem)."""
    n = sum(1 for d in deficits if d > 0)
    return max(0, n - 3)


def check_ledger(deficits):
    return sum(deficits) == TOTAL


# ---------------------------------------------------------------------------------------------
# the Trionda: the tetrahedron is the pillowcase
# ---------------------------------------------------------------------------------------------
def tetrahedron():
    """Flat model of the Trionda: three equilateral-triangle corners (60 deg) at each of 4 vertices.  **Exact.**"""
    d = [deficit(3 * 60)] * 4
    return {"deficits": d, "cone_angles": [cone_angle(x) for x in d], "ledger": sum(d), "moduli_dim": moduli_dim(d),
            "pillowcase": all(cone_angle(x) == 180 for x in d)}


# ---------------------------------------------------------------------------------------------
# the Hat Trick
# ---------------------------------------------------------------------------------------------
def hat_trick():
    """Flat model of the Hat Trick: the kite angles (60/90/120 deg at vertex/midpoint/centroid corners) add up to 360
    everywhere except at the corners of the 12 pentagonal holes.  A hole is a 10-gon whose corners hold 240 deg
    (centroids) and 180 deg (edge midpoints) of tile angle; its curvature is  K = 360 + 180 n - sum(theta)  with
    n = 10, and it is the deficit of the cone point that caps it.  **Exact** (computed from the actual complex)."""
    _ballgen()
    import the_hat_trick_ball as H
    from collections import defaultdict
    ball = H.HatTrick(0)
    tot = defaultdict(float)
    for k in ball.S.kites:
        for cid in ball.S.corners(k):
            tot[cid] += H.KITE_ANGLE[cid[0]]
    holes = []
    for p in ball.pent:
        loop = ball.pentagon_loop(p)
        theta = [int(round(tot[c])) for c in loop]
        holes.append(360 + 180 * len(loop) - sum(theta))
    off_hole = {c for c, a in tot.items() if round(a) != 360}
    in_holes = {c for p in ball.pent for c in ball.pentagon_loop(p)}
    return {"deficits": holes, "cone_angles": [cone_angle(x) for x in holes], "ledger": sum(holes),
            "moduli_dim": moduli_dim(holes), "only_hole_corners_are_not_flat": off_hole == in_holes,
            "hats": len(ball.hats), "pentagons": len(ball.pent)}


# ---------------------------------------------------------------------------------------------
# Spectre closed folds
# ---------------------------------------------------------------------------------------------
def spectre_survey(per_n=60, seed=5, Ns=range(4, 22), levels=3):
    """Random disc patches of N Spectre tiles that fold into a closed ball (exact DP); for each, the deficits of
    its cone points.  **Counted**: statements are about these random patches."""
    _ballgen()
    from spectre.patch import SpectrePatch
    from spectre.search import PatchPool
    from spectre.dp import can_close, fold_with_holes
    pool = PatchPool(SpectrePatch(levels=levels))
    rng = random.Random(seed)
    rows = []
    for N in Ns:
        disc = 0
        for _ in range(per_n):
            g = pool.graph(pool.random_subset(N, rng, rng.choice([0.0, 0.3, 0.7])))
            if g is None:
                continue
            disc += 1
            A = [int(a) for a in g.loops[0][1]]
            if not can_close(A):
                continue
            r = fold_with_holes(g, [])
            if r is None:
                continue
            ds = sorted(c.deficit for c in r.cones)
            rows.append({"N": N, "deficits": ds, "ledger": r.ledger, "holes": len(r.holes)})
    return rows


def summarise(rows):
    """Distribution of the number of cone points, the deficits that occur, the ledger, and the moduli dimension."""
    pos = [[d for d in r["deficits"] if d > 0] for r in rows]
    return {
        "balls": len(rows),
        "ledger_all_720": all(r["ledger"] == TOTAL and check_ledger(r["deficits"]) for r in rows),
        "no_holes": all(r["holes"] == 0 for r in rows),
        "deficits_multiples_of_30": all(d % 30 == 0 for r in rows for d in r["deficits"]),
        "deficit_values": dict(sorted(Counter(d for r in rows for d in r["deficits"]).items())),
        "negative_deficit_seen": any(d < 0 for r in rows for d in r["deficits"]),
        "n_cone_points": dict(sorted(Counter(len(p) for p in pos).items())),
        "moduli_dim": dict(sorted(Counter(moduli_dim(p) for p in pos).items())),
        "multisets": [[list(k), v] for k, v in Counter(tuple(p) for p in pos).most_common(12)],
        "pillowcase_folds": sum(1 for p in pos if p == [180] * 4),
        "by_size": {str(N): sum(1 for r in rows if r["N"] == N) for N in sorted({r["N"] for r in rows})},
    }


# ---------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------
def test_tetrahedron_is_the_pillowcase():
    t = tetrahedron()
    assert t["ledger"] == TOTAL and t["pillowcase"] and t["moduli_dim"] == 1 and t["cone_angles"] == [180] * 4


def test_hat_trick_has_twelve_cones_of_sixty():
    h = hat_trick()
    assert h["deficits"] == [60] * 12 and h["ledger"] == TOTAL and h["cone_angles"] == [300] * 12
    assert h["only_hole_corners_are_not_flat"] and h["hats"] == 60 and h["pentagons"] == 12
    assert h["moduli_dim"] == 9


def test_moduli_dimension_rule():
    assert moduli_dim([240, 240, 240]) == 0                 # three cones: rigid
    assert moduli_dim([180] * 4) == 1                       # four cones: the pillowcase family, one modulus
    assert moduli_dim([60] * 12) == 9
    assert moduli_dim([360, 360]) == 0                      # a football
    assert moduli_dim([90, 90, 0, 90, 90, 180, 180]) == moduli_dim([90, 90, 90, 90, 180, 180]) == 3  # zeros are smooth


def test_ledger_helpers():
    assert deficit(300) == 60 and cone_angle(60) == 300 and check_ledger([180] * 4) and not check_ledger([180] * 3)


def test_spectre_closed_folds_obey_gauss_bonnet():
    rows = spectre_survey(per_n=25, seed=5, Ns=range(4, 11))
    assert len(rows) >= 20, len(rows)
    s = summarise(rows)
    assert s["ledger_all_720"] and s["no_holes"] and s["deficits_multiples_of_30"] and not s["negative_deficit_seen"], s


def run_tests():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} checks passed")


def main(argv):
    if "--tests-only" in argv:
        run_tests()
        return
    out = {"tetrahedron": tetrahedron(), "hat_trick": hat_trick()}
    for k in ("tetrahedron", "hat_trick"):
        v = out[k]
        print(f"{k}: deficits {Counter(v['deficits'])}, cone angle {sorted(set(v['cone_angles']))}, ledger {v['ledger']}, "
              f"moduli dim {v['moduli_dim']}")
    args = [a for a in argv if a != "survey"]
    per_n = int(args[0]) if args else 60
    seed = int(args[1]) if len(args) > 1 else 5
    rows = spectre_survey(per_n=per_n, seed=seed)
    out["spectre"] = summarise(rows)
    print(json.dumps(out["spectre"], indent=1))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cone_ledger.json"), "w"), indent=1, default=int)


if __name__ == "__main__":
    main(sys.argv[1:])

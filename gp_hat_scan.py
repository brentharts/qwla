#!/usr/bin/env python3
"""
Stage 10 of ROADMAP_QUANTUM_GRAVITY.md (first part): does the hat tile the sphere beyond GP(f, 0)?

The Hat Trick lives on GP(3,0), the 92-tile Goldberg-Coxeter sphere; the earlier scan (hexasphere repo,
`examples/hat_trick_frequencies.py`) found hat tilings with 12 pentagon holes for no other GP(f,0), f = 5, 7, 9.
Here every Goldberg-Coxeter class GP(m,n), including the chiral ones (n > 0), is built combinatorially from the
Eisenstein lattice, the 3-kite split of every triangle is made, the 12 degree-5 vertices are holes, and the
exact covers by hats (Knuth's Algorithm X, from `hexasphere.hattrick`) are counted for several symmetry groups.

    T = m^2 + m n + n^2      vertices 10 T + 2, triangles 20 T, kites 60 (T - 1), hats 7.5 (T - 1)

so T must be odd.  GP(m,n) and GP(n,m) are mirror images, so with the hat's hand fixed both are scanned.
**Exact** for what is exhausted; a search that stops at its node limit is reported as not exhausted.
Nothing here proves non-existence for larger T or for other symmetry classes.

Needs `brentharts/hexasphere` (set HEXASPHERE_DIR to a clone; default: a sibling directory called `hexasphere`).

    python3 gp_hat_scan.py --tests-only
    python3 gp_hat_scan.py                       # scan, writes gp_hat_scan.json
"""
import json
import math
import os
import sys
import time

import numpy as np


def _hexasphere():
    here = os.path.dirname(os.path.abspath(__file__))
    d = os.environ.get("HEXASPHERE_DIR")
    for c in ([d] if d else []) + [os.path.join(here, "..", "hexasphere"), os.path.join(here, "hexasphere")]:
        if c and os.path.exists(os.path.join(c, "src", "hexasphere", "hattrick", "ball.py")):
            p = os.path.join(c, "src")
            if p not in sys.path:
                sys.path.insert(0, p)
            return os.path.abspath(c)
    raise SystemExit("set HEXASPHERE_DIR to a clone of brentharts/hexasphere")


W6 = complex(0.5, math.sqrt(3) / 2)                  # e^{i pi/3}


def goldberg_T(m, n):
    return m * m + m * n + n * n


def gp_sphere(m, n):
    """Icosahedral triangulation GP(m, n) of the sphere, counter-clockwise outward faces.

    Every icosahedron face (A, B, C) is the lattice triangle 0, P1 = m + n w, P2 = P1 w (w = e^{i pi/3}) in the
    Eisenstein lattice Z + Z w; the lattice points of the closed triangle map to the sphere by barycentric
    coordinates and shared points are merged.  The faces are the convex hull of the points on the sphere: every
    4-point rhombus of the near-regular lattice has a unique Delaunay diagonal, so this is the geodesic triangulation
    (checked by counts, degrees 5 and 6, and an exact manifold test).  (m, 0) is the ordinary frequency-m subdivision; n > 0 is chiral and GP(n, m) is its mirror image."""
    _hexasphere()
    from hexasphere.hattrick.kites import icosahedron
    V0, F0 = icosahedron()
    P1 = m + n * W6
    P2 = P1 * W6
    # lattice points (i, j) = i + j w inside the closed triangle (0, P1, P2); barycentric coordinates by solving
    M = np.array([[P1.real, P2.real], [P1.imag, P2.imag]])
    Minv = np.linalg.inv(M)
    R = m + n + 2
    inside = {}
    for i in range(-R, 2 * R):
        for j in range(-R, 2 * R):
            z = i + j * W6
            v, w_ = Minv @ np.array([z.real, z.imag])
            u = 1 - v - w_
            if min(u, v, w_) > -1e-9:
                inside[(i, j)] = (u, v, w_)
    verts, index = [], {}

    def vid(p):
        k = tuple(np.round(p, 6))
        if k not in index:
            index[k] = len(verts)
            verts.append(p)
        return index[k]

    for f in F0:
        A, B, C = V0[list(f)]
        for (u, v, w_) in inside.values():
            p = A * u + B * v + C * w_
            vid(p / np.linalg.norm(p))
    V = np.array(verts)
    from hexasphere.hattrick.kites import triangulate_points
    F = [tuple(int(x) for x in f) for f in triangulate_points(V)]
    return V, F


def check_sphere(V, F, T):
    """Counts and degrees of a Goldberg-Coxeter triangulation; every edge in two faces."""
    deg = np.zeros(len(V), int)
    edges = {}
    for f in F:
        for a in range(3):
            u, v = f[a], f[(a + 1) % 3]
            deg[u] += 1
            edges[(u, v)] = edges.get((u, v), 0) + 1
    ok_edges = all(c == 1 and (v, u) in edges for (u, v), c in edges.items())
    ctr = np.array([V[list(f)].sum(0) for f in F])
    nrm = np.array([np.cross(V[f[1]] - V[f[0]], V[f[2]] - V[f[0]]) for f in F])
    outward = bool(np.all(np.einsum("ij,ij->i", ctr, nrm) > 0))
    return {"vertices": len(V) == 10 * T + 2, "faces": len(F) == 20 * T, "degrees": set(deg) <= {5, 6} and int((deg == 5).sum()) == 12,
            "manifold": ok_edges, "outward": outward}


def scan_one(m, n, sym, both_hands=False, max_nodes=300000, max_tilings=1000):
    """Number of exact hat covers of GP(m,n) invariant under the symmetry group ``sym`` and whether the search ended."""
    _hexasphere()
    from hexasphere.hattrick import HatBall
    from hexasphere.hattrick.kites import rotation_group
    V, F = gp_sphere(m, n)
    t = time.time()
    try:
        b = HatBall(V, F, rotation_group(), symmetry=sym, both_hands=both_hands, max_tilings=max_tilings, max_nodes=max_nodes)
        out = {"tilings": len(b.tilings), "exhausted": b.search["exhausted"], "nodes": b.search["nodes"], "placements": len(b.placements)}
    except ValueError as e:
        s = str(e)
        if "no hat tiling" not in s:
            raise
        out = {"tilings": 0, "exhausted": "'exhausted': True" in s}
    out["seconds"] = round(time.time() - t, 1)
    return out


def classes(Tmax=40):
    """(m, n) with n >= 0, m >= n... both hands of each chiral pair, odd T only, T <= Tmax, sorted by T."""
    out = []
    for m in range(0, 8):
        for n in range(0, 8):
            T = goldberg_T(m, n)
            if 0 < T <= Tmax and T % 2 == 1 and T > 1 and math.gcd(m, n) >= 1:
                out.append((T, m, n))
    return [(m, n, T) for T, m, n in sorted(out)]


SYMS = ["I", "T", "C5", "C3", "C2", "C1"]


def scan(Tmax=40, syms=SYMS, verbose=True):
    rows = []
    for m, n, T in classes(Tmax):
        row = {"m": m, "n": n, "T": T, "hats": 7.5 * (T - 1), "chiral": m != 0 and n != 0 and m != n}
        for s in syms:
            row[s] = scan_one(m, n, s)
        row["C1_both_hands"] = scan_one(m, n, "C1", both_hands=True)
        rows.append(row)
        if verbose:
            print((m, n), {k: (v["tilings"], v["exhausted"]) for k, v in row.items() if isinstance(v, dict)}, flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
def test_goldberg_spheres_are_well_formed():
    for m, n in [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1), (1, 2), (3, 1), (2, 2), (3, 2)]:
        V, F = gp_sphere(m, n)
        c = check_sphere(V, F, goldberg_T(m, n))
        assert all(c.values()), ((m, n), c)


def test_gp30_is_the_hat_trick_grid():
    V, F = gp_sphere(3, 0)
    assert len(V) == 92 and len(F) == 180


def test_chiral_pairs_are_mirror_images():
    V1, F1 = gp_sphere(2, 1)
    V2, F2 = gp_sphere(1, 2)
    from scipy.spatial import cKDTree
    # a reflection maps one vertex set onto the other (some reflection of the icosahedron's frame)
    best = min(max(cKDTree(V2).query(V1 @ g.T)[0]) for g in _improper())
    assert best < 1e-6, best


def _improper():
    _hexasphere()
    from hexasphere.hattrick.kites import rotation_group
    return [-g for g in rotation_group()]


def test_scan_reproduces_the_hat_trick_and_the_known_negatives():
    assert scan_one(3, 0, "I")["tilings"] == 2
    r5 = scan_one(5, 0, "C1")
    assert r5["tilings"] == 0 and r5["exhausted"]


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
    Tmax = int(argv[0]) if argv else 40
    rows = scan(Tmax)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gp_hat_scan.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)


if __name__ == "__main__":
    main(sys.argv[1:])

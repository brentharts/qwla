#!/usr/bin/env python3
"""
q-deformation of the 12-puncture space of the Hat Trick ball (ROADMAP Stage 9c).

Classical (ring_bits.py): one spin-1/2 per cone point, the 132 singlets, and P V P = 72.5 - (1/3) sum_<ij> S_i.S_j.
With e_ij = 2 x (singlet projector of i, j) this is S_i.S_j = 1/4 - e_ij/2 (e/d is the singlet projector), and the e_ij generate a Temperley-Lieb
algebra with loop value d = 2. Asselmeyer-Maluga and Krol (arXiv:1211.3012, sections 3.3-3.4) read quantisation as the
Kauffman-bracket skein deformation of this structure (Turaev; Bullock-Przytycki), K_t(D^2, 2n) = TL_n. Here that
deformation is carried out for our 12 punctures:

  space     span of the 132 non-crossing pairings of 12 points (the order is a Hamiltonian cycle of the icosahedron),
            with the form G_ab = d^(loops of a over b), d = -A^2 - A^-2;
  edges     for the edges between neighbours in the order, e is a Temperley-Lieb generator; for the others the strand
            is first braided next to its partner, over the strands between for an edge inside the cycle and under for
            an edge outside (sigma = A + A^-1 e), which is the arc drawn on the sphere along that edge;
  level k   d = 2 cos(pi/(k+2)); the physical space is the quotient by the radical of G (SU(2)_k conformal blocks).

Output: dimension of the physical space and the spectrum of  H = sum_<ij> (1/4 - e_ij/d)  (= sum S_i.S_j at d = 2)
for a list of levels and the classical limit. Tests: classical spectrum equals the Heisenberg one; the closing edge
gives the same operator braided over or under (sphere relation); a second Hamiltonian cycle gives the same spectrum;
H is self-adjoint for G; dimensions equal the Verlinde count. Usage: python3 qdeform_punctures.py [--tests]
"""
import itertools
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
N = 12
PHI = (1 + 5 ** 0.5) / 2


def icosa():
    V = np.array([p for s1 in (1, -1) for s2 in (1, -1) for p in ((0, s1, s2 * PHI), (s1, s2 * PHI, 0), (s2 * PHI, 0, s1))], float)
    V /= np.linalg.norm(V[0])
    D = V @ V.T
    E = [(i, j) for i in range(N) for j in range(i + 1, N) if 0.3 < D[i, j] < 0.9]
    F = [(i, j, k) for i, j, k in itertools.combinations(range(N), 3) if (i, j) in E and (j, k) in E and (i, k) in E]
    assert len(E) == 30 and len(F) == 20
    return V, E, F


def hamiltonian_cycles(E, limit=2, seed=0):
    adj = {i: set() for i in range(N)}
    for i, j in E:
        adj[i].add(j)
        adj[j].add(i)
    rng = np.random.default_rng(seed)
    out = []

    def go(path):
        if len(out) >= limit:
            return
        if len(path) == N:
            if path[0] in adj[path[-1]]:
                out.append(list(path))
            return
        nb = sorted(adj[path[-1]] - set(path))
        rng.shuffle(nb)
        for v in nb:
            go(path + [v])
    go([0])
    return out


def sides(cycle, E, F):
    """For each edge not on the cycle: 0 or 1, the side of the cycle it lies on (the 20 faces cut along the cycle)."""
    cyc = {frozenset((cycle[i], cycle[(i + 1) % N])) for i in range(N)}
    comp = {f: None for f in F}
    edge_faces = {}
    for f in F:
        for a, b in itertools.combinations(f, 2):
            edge_faces.setdefault(frozenset((a, b)), []).append(f)
    c = 0
    for f in F:
        if comp[f] is None:
            stack = [f]
            comp[f] = c
            while stack:
                g = stack.pop()
                for a, b in itertools.combinations(g, 2):
                    e = frozenset((a, b))
                    if e in cyc:
                        continue
                    for h in edge_faces[e]:
                        if comp[h] is None:
                            comp[h] = c
                            stack.append(h)
            c += 1
    assert c == 2
    return {e: comp[edge_faces[e][0]] for e in edge_faces if e not in cyc}, cyc


def matchings(n):
    """Non-crossing perfect matchings of 0..n-1 as partner tuples."""
    def rec(pts):
        if not pts:
            yield {}
            return
        a = pts[0]
        for idx in range(1, len(pts), 2):
            b = pts[idx]
            for m1 in rec(pts[1:idx]):
                for m2 in rec(pts[idx + 1:]):
                    m = {a: b, b: a}
                    m.update(m1)
                    m.update(m2)
                    yield m
    return [tuple(m[i] for i in range(n)) for m in rec(list(range(n)))]


BASIS = matchings(N)
INDEX = {m: i for i, m in enumerate(BASIS)}
DIM = len(BASIS)


def loops(a, b):
    """Number of closed loops when pairing a is glued to pairing b."""
    seen, n = set(), 0
    for s in range(N):
        if s in seen:
            continue
        n += 1
        x = s
        while True:
            seen.add(x)
            x = a[x]
            seen.add(x)
            x = b[x]
            if x == s:
                break
    return n


LOOPS = np.array([[loops(a, b) for b in BASIS] for a in BASIS])


def tl_generator(i, d):
    """e_i on positions i, i+1 in the pairing basis (column = input)."""
    M = np.zeros((DIM, DIM), dtype=complex)
    for c, m in enumerate(BASIS):
        if m[i] == i + 1:
            M[c, c] = d
        else:
            p, q = m[i], m[i + 1]
            new = list(m)
            new[i], new[i + 1] = i + 1, i
            new[p], new[q] = q, p
            M[INDEX[tuple(new)], c] = 1.0
    return M


class Model:
    def __init__(self, A, cycle, E, F, over_side=None):
        self.A = A
        self.cycle = list(cycle)
        self.V_ = None
        self.over_side = over_side
        d = -A ** 2 - A ** -2
        assert abs(d.imag) < 1e-12
        self.d = float(d.real)
        self.e = [tl_generator(i, self.d) for i in range(N - 1)]
        I = np.eye(DIM)
        self.sig = [A * I + self.e[i] / A for i in range(N - 1)]
        self.sinv = [I / A + A * self.e[i] for i in range(N - 1)]
        self.pos = {v: k for k, v in enumerate(cycle)}
        self.side, self.cyc = sides(cycle, E, F)
        self.E = E
        self.G = self.d ** LOOPS.astype(float)

    def arc(self, i, j, over):
        """e for the pair at positions i < j, strand j braided over (or under) the strands between."""
        s = self.sig if over else self.sinv
        t = self.sinv if over else self.sig
        W = np.eye(DIM, dtype=complex)
        Wi = np.eye(DIM, dtype=complex)
        for k in range(j - 1, i, -1):          # sigma_{j-1} acts first, sigma_{i+1} last
            W = s[k] @ W
            Wi = Wi @ t[k]
        return Wi @ self.e[i] @ W

    def edge_op(self, a, b):
        i, j = sorted((self.pos[a], self.pos[b]))
        f = frozenset((a, b))
        if j == i + 1:
            return self.e[i]
        if f in self.cyc:                       # the closing edge (positions 0 and N-1)
            return self.arc(i, j, True)
        return self.arc(i, j, self.side[f] == (0 if self.over_side is None else self.over_side))

    def hamiltonian(self):
        H = np.zeros((DIM, DIM), dtype=complex)
        for a, b in self.E:
            H += 0.25 * np.eye(DIM) - self.edge_op(a, b) / self.d
        return H

    def physical(self, H, tol=1e-8):
        """(dimension, negative directions of G, eigenvalues of H on the quotient, residual of G H = H^* G)."""
        G = self.G
        w, U = np.linalg.eigh(G)
        keep = np.abs(w) > tol * np.abs(w).max()
        Q = U[:, keep]
        Hp = np.linalg.solve(Q.T @ G @ Q, Q.T @ G @ H @ Q)
        ev = np.linalg.eigvals(Hp)
        ev = ev[np.argsort(ev.real)]
        return int(keep.sum()), int((w[keep] < 0).sum()), ev, float(np.abs(G @ H - H.conj().T @ G).max())


def left_side(V, cycle, E, F):
    """Label (0 or 1) of the side of the cycle that lies to the LEFT of the cycle's direction, seen from outside."""
    side, cyc = sides(cycle, E, F)
    a, b = cycle[0], cycle[1]
    for f in F:
        if a in f and b in f:
            c = [x for x in f if x not in (a, b)][0]
            left = np.cross(V[b] - V[a], V[c] - V[a]) @ (V[a] + V[b] + V[c]) > 0      # a, b, c counter-clockwise from outside
            # the face's side label: any non-cycle edge of the faces reachable from it; use the component of this face
            comp = None
            for e2 in (frozenset((a, c)), frozenset((b, c))):
                if e2 in side:
                    comp = side[e2]
            if comp is None:
                continue
            return comp if left else 1 - comp
    raise RuntimeError("no face found")


def A_of_level(k):
    return 1j * np.exp(1j * math.pi / (2 * (k + 2)))


def verlinde(k):
    T = np.zeros((k + 1, k + 1))
    for i in range(k):
        T[i, i + 1] = T[i + 1, i] = 1
    return int(round(np.linalg.matrix_power(T, N)[0, 0]))


def run(levels=(1, 2, 3, 4, 5, 6, 8, 10, 12, 20, 50, 200), verbose=True):
    V, E, F = icosa()
    cyc1, cyc2 = hamiltonian_cycles(E, 2)
    rows = []
    for k in list(levels) + [None]:
        A = 1j if k is None else A_of_level(k)
        m = Model(A, cyc1, E, F)
        dim, neg, ev, herm = m.physical(m.hamiltonian())
        second = float(ev[np.argmax(ev.real > ev[0].real + 1e-7)].real) if dim > 1 else None
        deg = int((ev.real < ev[0].real + 1e-7).sum())
        rows.append(dict(level=k if k is not None else "classical", d=m.d, dim=dim, negative_norm=neg, E_min=float(ev[0].real),
                         deg_min=deg, E_second=second, E_max=float(ev[-1].real), max_imag=float(np.abs(ev.imag).max()), self_adjoint_residual=herm))
        if verbose:
            r = rows[-1]
            print("k=%-9s d=%.4f dim=%3d  E_min=%9.5f (x%d)  next=%s  E_max=%8.5f  |Im|<%.0e  resid=%.0e" % (
                r["level"], r["d"], r["dim"], r["E_min"], deg, "%9.5f" % second if second is not None else "    -    ", r["E_max"], r["max_imag"] + 1e-99, herm))
    return rows


def level2_majorana(cycle_index=0, over="left", conj=False, reverse=False):
    """Level 2: Gamma_ij = 2 e_ij/d - 1 behave as i s_ij g_i g_j for 12 Majorana operators, one per puncture.
    Returns the checks as a dict (all exact to rounding)."""
    V, E, F = icosa()
    cyc1 = hamiltonian_cycles(E, 2)[cycle_index]
    if reverse:
        cyc1 = cyc1[::-1]
    ls = left_side(V, cyc1, E, F)
    A2 = A_of_level(2)
    m = Model(1 / A2 if conj else A2, cyc1, E, F, over_side=ls if over == "left" else 1 - ls)
    G = m.G
    w, U = np.linalg.eigh(G)
    Q = U[:, np.abs(w) > 1e-8 * np.abs(w).max()]
    red = lambda X: np.linalg.solve(Q.T @ G @ Q, Q.T @ G @ X @ Q)
    I = np.eye(Q.shape[1])
    Gam = {e: 2 * red(m.edge_op(*e)) / m.d - I for e in E}
    g = lambda a, b: Gam[(min(a, b), max(a, b))]
    sq = max(np.abs(x @ x - I).max() for x in Gam.values())
    ac = max(np.abs(Gam[e] @ Gam[f] + Gam[f] @ Gam[e]).max() for e, f in itertools.combinations(E, 2) if set(e) & set(f))
    cm = max(np.abs(Gam[e] @ Gam[f] - Gam[f] @ Gam[e]).max() for e, f in itertools.combinations(E, 2) if not set(e) & set(f))
    flux = {}
    for i, j, k in F:
        if np.cross(V[j] - V[i], V[k] - V[i]) @ (V[i] + V[j] + V[k]) < 0:
            j, k = k, j
        P = g(i, j) @ g(j, k) @ g(k, i)
        c = np.trace(P) / len(I)
        assert np.abs(P - c * I).max() < 1e-9
        flux[(i, j, k)] = c
    adj = {i: [] for i in range(N)}
    for i, j in E:
        adj[i].append(j)
        adj[j].append(i)
    s, seen, stack = {}, {0}, [0]
    while stack:                                         # gauge: +1 on a spanning tree
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                s[(u, v)], s[(v, u)] = 1, -1
                stack.append(v)
    changed = True
    while changed:                                       # the rest from the face fluxes: s_ij s_jk s_ki = flux / (-i)
        changed = False
        for (i, j, k), f in flux.items():
            es = [(i, j), (j, k), (k, i)]
            unk = [e for e in es if e not in s]
            if len(unk) == 1:
                val = int(round((f / (-1j)).real)) * int(np.prod([s[e] for e in es if e in s]))
                a, b = unk[0]
                s[(a, b)], s[(b, a)] = val, -val
                changed = True
    a = np.zeros((N, N))
    for (u, v), x in s.items():
        a[u, v] = x
    ok = all(int(round((f / (-1j)).real)) == s[(i, j)] * s[(j, k)] * s[(k, i)] for (i, j, k), f in flux.items())
    ev = np.sort(np.linalg.eigvals(red(m.hamiltonian())).real)
    pred = np.sort(np.concatenate([np.full(math.comb(6, mm), -7.5 + 5 ** 0.5 / 2 * (6 - 2 * mm)) for mm in (0, 2, 4, 6)]))
    left, match = set(range(N)), []
    def pm(left):
        if not left:
            return []
        u = min(left)
        for v in adj[u]:
            if v in left:
                r = pm(left - {u, v})
                if r is not None:
                    return [(u, v)] + r
        return None
    match = pm(frozenset(range(N)))
    P = I.copy()
    for u, v in match:
        P = P @ g(u, v)
    return dict(dim=len(I), square=float(sq), anticommute=float(ac), commute=float(cm), fluxes=sorted({complex(np.round(c, 6)) for c in flux.values()}, key=str),
                signs_consistent=bool(ok), a_squared_plus_5=float(np.abs(a @ a + 5 * np.eye(N)).max()), spectrum_error=float(np.abs(ev - pred).max()),
                matching=match, matching_product=complex(np.round(np.trace(P) / len(I), 6)), matching_product_error=float(np.abs(P - np.trace(P) / len(I) * I).max()))


def run_tests():
    V, E, F = icosa()
    cyc1, cyc2 = hamiltonian_cycles(E, 2)
    assert DIM == 132 and cyc1 != cyc2
    m = Model(1j, cyc1, E, F)
    dim, neg, ev, herm = m.physical(m.hamiltonian())
    assert dim == 132 and neg == 0 and abs(ev[0].real + 6.187889964) < 1e-6 and abs(ev[-1].real - 3.527476863) < 1e-6
    print("1. classical point (d = 2): 132 states, sum S.S from %.6f to %.6f, as in ring_bits.py" % (ev[0].real, ev[-1].real))
    for k in (3, 7):
        m = Model(A_of_level(k), cyc1, E, F)
        a, b = m.arc(0, N - 1, True), m.arc(0, N - 1, False)
        assert np.abs(m.G @ (a - b)).max() < 1e-8 * np.abs(m.G @ a).max()
        e0, e1 = m.arc(2, 7, True), m.arc(2, 7, False)
        assert np.abs(m.G @ (e0 - e1)).max() > 1e-3
        assert np.abs(m.G @ (a @ a - m.d * a)).max() < 1e-7 * np.abs(m.G @ a).max()
    print("2. sphere relation: the closing edge braided over or under is the same operator; a chord is not; e^2 = d e")
    for k in (2, 4, 9):
        s = []
        for c in (cyc1, cyc2):
            m = Model(A_of_level(k), c, E, F)
            dim, neg, ev, herm = m.physical(m.hamiltonian())
            assert neg == 0 and herm < 1e-7 and np.abs(ev.imag).max() < 1e-7, (k, neg, herm, np.abs(ev.imag).max())
            s.append(np.sort(ev.real))
        assert s[0].shape == s[1].shape and np.abs(s[0] - s[1]).max() < 1e-6
    print("3. two Hamiltonian cycles give the same spectrum (k = 2, 4, 9); H is self-adjoint for G; G has no negative direction")
    dims = {k: Model(A_of_level(k), cyc1, E, F).physical(np.zeros((DIM, DIM)))[0] for k in (1, 2, 3, 4, 5, 6)}
    assert all(dims[k] == verlinde(k) for k in dims), (dims, {k: verlinde(k) for k in dims})
    print("4. dimension at level k equals the Verlinde count (closed walks of 12 steps on the A_(k+1) diagram):", dims)
    r = level2_majorana()
    assert r["dim"] == 32 and max(r["square"], r["anticommute"], r["commute"], r["a_squared_plus_5"], r["spectrum_error"], r["matching_product_error"]) < 1e-9
    assert r["signs_consistent"] and len(r["fluxes"]) == 1
    print("5. level 2: Gamma = 2e/d - 1 are Majorana bilinears (square 1, anticommute iff they share a puncture); every face has flux",
          r["fluxes"][0], "; the signed adjacency a has a^2 = -5; spectrum = -7.5 + (sqrt 5 / 2)(6 - 2m), m even")
    fx = {}
    for ci in (0, 1):
        for rev in (False, True):
            for over in ("left", "right"):
                for conj in (False, True):
                    fx[(ci, rev, over, conj)] = level2_majorana(ci, over, conj, rev)["fluxes"]
    assert all(len(v) == 1 for v in fx.values())
    val = {k: v[0] for k, v in fx.items()}
    base = val[(0, False, "left", False)]
    assert all(val[(ci, rev, "left", False)] == base for ci in (0, 1) for rev in (False, True))
    assert all(val[(ci, rev, "right", False)] == -base and val[(ci, rev, "left", True)] == -base and val[(ci, rev, "right", True)] == base
               for ci in (0, 1) for rev in (False, True))
    print("6. with 'over = the side to the left of the cycle, seen from outside' the face flux is", base, "for both cycles and both directions;",
          "it changes sign under over <-> under and under A -> 1/A: the sign is the orientation convention, nothing else")
    print("all tests passed")


if __name__ == "__main__":
    run_tests()
    if "--tests" not in sys.argv:
        rows = run()
        json.dump(rows, open(os.path.join(HERE, "qdeform_punctures.json"), "w"), indent=1)
        print("-> qdeform_punctures.json")

#!/usr/bin/env python3
"""
Stage 9 of ROADMAP_QUANTUM_GRAVITY.md, first part: quantise the sphere on which the Trionda lives.

The Trionda model of `brentharts/ballgen` (paper/trionda.tex) is the flow of v = p x grad f on the unit sphere
with f = xyz, an area-preserving flow, i.e. the Hamiltonian flow of f on the phase space S^2 (area form dA, total 4 pi).
Quantising that phase space with spin j (2j + 1 states, hbar_eff = 1/j) turns f into an operator.  This module checks,
with exact finite matrices:

  * H3 = sym(Jx Jy Jz) is invariant under the 12 rotations of the tetrahedral group T and under no larger rotation
    group; H6 = sum_a (a . J)^6 over the six 5-fold axes is invariant under the 60 rotations of I.  (The first
    invariant multipoles are l = 3 for T and l = 6 for I, the same numbers as the seam-harmonic selection rules.)
  * Ehrenfest: the expectation <J>(t)/j of a coherent state under H3 follows the classical Trionda flow, with an
    error that falls as j grows.  So the quantum model has the paper's flow as its classical limit.
  * Counting: a region of the sphere of area A holds about (2j + 1) A / (4 pi) states; the four Trionda panels hold
    (2j + 1)/4 each, and the number of T-invariant states is about (2j + 1)/12.

**Exact** for the matrix identities; **Counted** for the Ehrenfest convergence.  Not a derivation of anything about
gravity: this is a dictionary between an area-preserving flow and a spin system (area <-> number of states).

    python3 spin_sphere.py --tests-only
    python3 spin_sphere.py
"""
import itertools
import math
import sys

import numpy as np
from scipy.linalg import expm

# ---------------------------------------------------------------------------------------------
# spin operators
# ---------------------------------------------------------------------------------------------
def spin_ops(j):
    """Jx, Jy, Jz for spin j (integer or half-integer), as dense complex matrices, basis m = j, j-1, ..., -j."""
    d = int(round(2 * j + 1))
    m = j - np.arange(d)
    Jz = np.diag(m).astype(complex)
    Jp = np.zeros((d, d), complex)
    for k in range(1, d):
        Jp[k - 1, k] = math.sqrt(j * (j + 1) - m[k] * (m[k] + 1))
    Jm = Jp.conj().T
    return (Jp + Jm) / 2, (Jp - Jm) / (2j), Jz


def n_dot_J(J, n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    return n[0] * J[0] + n[1] * J[1] + n[2] * J[2]


def rotation(J, axis, angle):
    """Unitary of the rotation by `angle` about `axis` (right-handed)."""
    return expm(-1j * angle * n_dot_J(J, axis))


def H3(J):
    """Symmetrised Jx Jy Jz (average over the 6 orderings): the quantisation of j^3 x y z."""
    X, Y, Z = J
    return sum(a @ b @ c for a, b, c in itertools.permutations([X, Y, Z])) / 6


PHI = (1 + 5 ** 0.5) / 2
ICOSA_AXES = [(0, 1, PHI), (0, 1, -PHI), (1, PHI, 0), (1, -PHI, 0), (PHI, 0, 1), (-PHI, 0, 1)]   # the six 5-fold axes


def H6(J):
    """Sum over the six five-fold axes of (a . J)^6: the quantisation of an I-invariant sextic."""
    out = 0
    for a in ICOSA_AXES:
        A = n_dot_J(J, a)
        out = out + np.linalg.matrix_power(A, 6)
    return out


def tetra_rotations(J):
    """Generators of T: the half-turns about x, y, z and the third-turn about (1,1,1)."""
    return [rotation(J, (1, 0, 0), math.pi), rotation(J, (0, 1, 0), math.pi), rotation(J, (0, 0, 1), math.pi),
            rotation(J, (1, 1, 1), 2 * math.pi / 3)]


def icosa_rotations(J):
    """Generators of I (in the frame of ICOSA_AXES): a fifth-turn about a 5-fold axis, and a half-turn about the
    coordinate axes (which are 2-fold axes of this icosahedron)."""
    return [rotation(J, ICOSA_AXES[0], 2 * math.pi / 5), rotation(J, (1, 0, 0), math.pi), rotation(J, (0, 1, 0), math.pi),
            rotation(J, (0, 0, 1), math.pi)]


def comm_norm(A, B):
    return float(np.linalg.norm(A @ B - B @ A))


def invariant_count(j, generators_fn, order, element_fn):
    """Dimension of the subspace of spin j invariant under the group: (1/|G|) sum chi_j(g) over group elements
    (character chi_j(theta) = sin((2j+1) theta/2)/sin(theta/2)).  element_fn() yields the rotation angles of the
    elements."""
    tot = 0.0
    for th in element_fn():
        tot += 2 * j + 1 if abs(th) < 1e-12 else math.sin((2 * j + 1) * th / 2) / math.sin(th / 2)
    return tot / order


def tetra_angles():
    return [0.0] + [math.pi] * 3 + [2 * math.pi / 3] * 8             # 1 identity, 3 half-turns, 8 third-turns


def icosa_angles():
    return [0.0] + [math.pi] * 15 + [2 * math.pi / 3] * 20 + [2 * math.pi / 5] * 12 + [4 * math.pi / 5] * 12


# ---------------------------------------------------------------------------------------------
# classical flow
# ---------------------------------------------------------------------------------------------
def classical_flow(n0, tmax, steps=4000):
    """Heisenberg flow of j^3 xyz on the unit sphere, in time units of j^2: dn/dt = grad(xyz) x n  (RK4)."""
    def f(n):
        g = np.array([n[1] * n[2], n[0] * n[2], n[0] * n[1]])
        return np.cross(g, n)
    n, dt, out = np.array(n0, float), tmax / steps, []
    for _ in range(steps):
        out.append(n.copy())
        k1 = f(n); k2 = f(n + dt / 2 * k1); k3 = f(n + dt / 2 * k2); k4 = f(n + dt * k3)
        n = n + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        n /= np.linalg.norm(n)
    out.append(n.copy())
    return np.array(out), dt


def coherent_state(j, n):
    """Spin coherent state pointing along n (highest weight of n . J)."""
    J = spin_ops(j)
    w, v = np.linalg.eigh(n_dot_J(J, n))
    return v[:, -1]


def ehrenfest_error(j, n0=(0.3, 0.5, 0.8124038404635961), tmax=1.2):
    """Maximum over t <= tmax (in units of j^-2) of |<J>(t)/j - n_classical(t)|."""
    n0 = np.asarray(n0, float) / np.linalg.norm(n0)
    J = spin_ops(j)
    H = H3(J)
    w, V = np.linalg.eigh(H)
    psi0 = coherent_state(j, n0)
    c = V.conj().T @ psi0
    ts = np.linspace(0, tmax, 13)
    cl, dt = classical_flow(n0, tmax, steps=2400)
    err = 0.0
    for t in ts:
        psi = V @ (np.exp(-1j * w * t / j ** 2) * c)            # quantum time = classical time / j^2
        e = np.array([np.real(psi.conj() @ A @ psi) for A in J]) / j
        k = int(round(t / tmax * (len(cl) - 1)))
        err = max(err, float(np.linalg.norm(e - cl[k])))
    return err


# ---------------------------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------------------------
def test_h3_is_tetrahedrally_invariant_and_nothing_bigger():
    for j in (3, 5, 7.5):
        J = spin_ops(j)
        H = H3(J)
        for g in tetra_rotations(J):
            assert comm_norm(H, g) < 1e-9 * (1 + np.linalg.norm(H)), (j, comm_norm(H, g))
        # a fourth-turn about z (in the octahedral group, not in T) does not commute
        assert comm_norm(H, rotation(J, (0, 0, 1), math.pi / 2)) > 1e-3 * np.linalg.norm(H)
        # and a generic rotation does not
        assert comm_norm(H, rotation(J, (1, 2, 3), 0.7)) > 1e-3 * np.linalg.norm(H)


def test_h6_is_icosahedrally_invariant():
    for j in (6, 8):
        J = spin_ops(j)
        H = H6(J)
        for g in icosa_rotations(J):
            assert comm_norm(H, g) < 1e-8 * (1 + np.linalg.norm(H)), (j, comm_norm(H, g))
        assert comm_norm(H, rotation(J, (1, 2, 3), 0.7)) > 1e-3 * np.linalg.norm(H)


def test_h3_has_no_icosahedral_invariance():
    J = spin_ops(6)
    assert comm_norm(H3(J), icosa_rotations(J)[0]) > 1e-3


def test_ehrenfest_error_falls_with_j():
    e = [ehrenfest_error(j) for j in (10, 20, 40)]
    assert e[0] > e[1] > e[2], e
    assert e[2] < 0.5 * e[0], e


def test_state_counting():
    for j in (3, 6, 15, 30):
        nT = invariant_count(j, None, 12, tetra_angles)
        nI = invariant_count(j, None, 60, icosa_angles)
        assert abs(nT - round(nT)) < 1e-9 and abs(nI - round(nI)) < 1e-9
        assert abs(nT - (2 * j + 1) / 12) <= 1.5, (j, nT)
        assert abs(nI - (2 * j + 1) / 60) <= 1.5, (j, nI)


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
    print("Ehrenfest error (classical time <= 1.2, quantum time t/j^2):", {j: round(ehrenfest_error(j), 4) for j in (10, 20, 40, 80)})
    for j in (6, 15, 30, 60):
        print(f"j={j}: states {2*j+1}, per Trionda panel {(2*j+1)/4:g}, T-invariant {round(invariant_count(j, None, 12, tetra_angles))}, "
              f"I-invariant {round(invariant_count(j, None, 60, icosa_angles))}")


if __name__ == "__main__":
    main(sys.argv[1:])

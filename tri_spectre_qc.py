"""
Three coupled quantum computers over exact Spectre substitution tiers, with the pair-qubit
Hamiltonian, inter-tier bridges and geometry feedback.

Model (all terms are definitions of this toy, see ROADMAP_QUANTUM_GRAVITY.md, Stage 2):

  H_tier = -J sum_p X_p + kappa sum_{p<q} W_pq Z_p Z_q ,   kappa = (a h)^2 ,

with one qubit p per bump-and-dent pair, W the dipole coupling of the two pairs from the exact
tile centroids (`pair_couplings`), Z_p = <tau^z> the bump bit (+1: tile A is the bump), and X_p
the tunnelling between the two positions. Evolution is first-order Trotter with the functional
`State`. Geometry -> quantum: the centroids set W. Quantum -> geometry: <Z_p> displaces the flat
vertex P8 of both tiles of pair p (`set_pair_bit`), changing their areas by +-a h <Z_p>.

Note on the earlier version: it put every qubit in a Hadamard state, after which <Z> = 0 for
every later RZ/CZ, so the feedback could never act, and it projected the joint state onto
invented marginal amplitudes. Here one joint `State` is kept and read directly.
"""
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from bitarray import frozenbitarray as bitarray
from functional import seq

from quantumspectremodel import (
    A0,
    SubstitutionPatch,
    assign_qubits_to_patch,
    build_substitution_tier,
    pair_couplings,
    pair_qubit_tiles,
    set_pair_bit,
)
from state import State


def _tensor_product_states(states: List[State]) -> State:
    """Joint `State` over the concatenated registers [s_0, s_1, ...]."""
    total = sum(s.n_qubits for s in states)
    joint = State(n_qubits=total)
    dicts = [dict(s.state.to_list()) for s in states]
    pairs = []
    for idx in range(1 << total):
        bits = format(idx, f"0{total}b")
        amp, pos = 1 + 0j, 0
        for s, d in zip(states, dicts):
            amp *= d.get(bitarray(bits[pos: pos + s.n_qubits]), 0.0)
            pos += s.n_qubits
            if amp == 0j:
                break
        pairs.append((bitarray(bits), amp))
    joint.state = seq(pairs)
    return joint


def trotter_step(state: State, W: np.ndarray, offset: int, J: float, kappa: float, dt: float) -> None:
    """One first-order step of exp(-i dt H) for H = -J sum X + kappa sum W_pq Z_p Z_q on qubits offset+p."""
    n = W.shape[0]
    for p in range(n):
        for q in range(p + 1, n):
            if W[p, q] != 0.0:
                a, b = offset + p, offset + q
                state.cx(a, b)
                state.rz(b, 2.0 * dt * kappa * W[p, q])   # exp(-i dt kappa W Z Z)
                state.cx(a, b)
    for p in range(n):
        j = offset + p
        state.h(j)
        state.rz(j, -2.0 * dt * J)                          # exp(+i dt J X)
        state.h(j)


def exact_expectation_z(W: np.ndarray, J: float, kappa: float, t: float) -> np.ndarray:
    """<Z_p>(t) from the all-up state under H, by dense diagonalisation (n <= 12)."""
    n = W.shape[0]
    X = np.array([[0, 1], [1, 0]], complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    I = np.eye(2, dtype=complex)

    def op(single, p):
        mats = [I] * n
        mats[p] = single
        out = mats[0]
        for m in mats[1:]:
            out = np.kron(out, m)
        return out

    H = sum(-J * op(X, p) for p in range(n))
    for p in range(n):
        for q in range(p + 1, n):
            H = H + kappa * W[p, q] * op(Z, p) @ op(Z, q)
    w, v = np.linalg.eigh(H)
    psi0 = np.zeros(2 ** n, complex)
    psi0[0] = 1.0   # |00..0>, every <Z> = +1
    psi = v @ (np.exp(-1j * w * t) * (v.conj().T @ psi0))
    return np.array([float(np.real(psi.conj() @ op(Z, p) @ psi)) for p in range(n)])


@dataclass
class SpectreQuantumComputer:
    """One tier: an exact patch, its pair qubits and the couplings between them."""

    name: str
    patch: SubstitutionPatch
    offset: int = 0

    def __post_init__(self):
        self.W = pair_couplings(self.patch)

    @property
    def n_qubits(self) -> int:
        return self.W.shape[0]

    def sync_geometry(self, state: State, h: float) -> None:
        """Quantum -> geometry: <Z_p> moves P8 of both tiles of pair p."""
        for q, (A, B) in pair_qubit_tiles(self.patch).items():
            set_pair_bit(A, B, state.expectation_z(self.offset + q), h)

    def pair_area_sums(self) -> List[float]:
        return [A.area() + B.area() for A, B in pair_qubit_tiles(self.patch).values()]


class TriSpectreNetwork:
    """
    Three tiers (default generations 2, 3, 4 = 71, 559, 4401 tiles; the 9-tile patch has no pair)
    sharing one joint `State`, coupled by CX/CZ bridges between anchor pairs.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        curve_strength: float = 0.5,
        qubits_per_tier: Tuple[int, int, int] = (4, 4, 4),
        generations: Tuple[int, int, int] = (2, 3, 4),
        J: float = 1.0,
        h: float = 0.3,
    ):
        self.J, self.h = J, h
        self.kappa = (1.0 * h) ** 2          # (a h)^2 with a = 1
        self.computers: List[SpectreQuantumComputer] = []
        offset = 0
        for gen, budget in zip(generations, qubits_per_tier):
            patch = build_substitution_tier(gen, a=a, b=b, curve_strength=curve_strength)
            assign_qubits_to_patch(patch, mode="pairs", max_qubits=budget)
            qc = SpectreQuantumComputer(f"QC-{patch.n_tiles} ({patch.n_tiles} tiles)", patch, offset)
            self.computers.append(qc)
            offset += qc.n_qubits
        self.n_qubits = offset
        self.state = State(n_qubits=self.n_qubits, n_bits=self.n_qubits)   # |00..0>: every bump on tile A
        self.patches = [qc.patch for qc in self.computers]

    def run_local_dynamics(self, dt: float = 0.1, steps: int = 1) -> None:
        for _ in range(steps):
            for qc in self.computers:
                trotter_step(self.state, qc.W, qc.offset, self.J, self.kappa, dt)

    def entangle_across_computers(self, links: Optional[List[Tuple[int, int, int]]] = None) -> None:
        """CX/CZ bridges across anchor qubits (i_lo, i_mid, i_hi): a coupling chosen for this toy."""
        if links is None:
            links = [(0, 0, 0), (1, 1, 1)]
        lo, mid, hi = self.computers
        for i0, i1, i2 in links:
            g0 = lo.offset + i0 % lo.n_qubits
            g1 = mid.offset + i1 % mid.n_qubits
            g2 = hi.offset + i2 % hi.n_qubits
            self.state.cx(g0, g1)
            self.state.cx(g1, g2)
            self.state.cz(g0, g2)

    def sync_geometry(self) -> None:
        for qc in self.computers:
            qc.sync_geometry(self.state, self.h)

    def step_coupled_cycle(self, dt: float = 0.3, steps: int = 3, bridges: bool = True) -> None:
        """Local pair-qubit dynamics, then (optionally) the cross-tier bridges, then geometry feedback."""
        self.run_local_dynamics(dt=dt / steps, steps=steps)
        if bridges:
            self.entangle_across_computers()
        self.sync_geometry()

    def summary(self) -> List[Dict[str, object]]:
        rows = []
        for qc in self.computers:
            zs = [self.state.expectation_z(qc.offset + q) for q in range(qc.n_qubits)]
            areas = [t.area() for A, B in pair_qubit_tiles(qc.patch).values() for t in (A, B)]
            rows.append({
                "computer": qc.name,
                "tiles": qc.patch.n_tiles,
                "pairs": qc.patch.n_pairs,
                "qubits": qc.n_qubits,
                "mean_z": round(float(np.mean(zs)), 4),
                "area_min": round(min(areas), 4),
                "area_max": round(max(areas), 4),
                "pair_area_sum_err": round(max(abs(s - 2 * A0) for s in qc.pair_area_sums()), 12),
            })
        return rows


# ---- checks -----------------------------------------------------------------------------

def test_trotter_matches_exact_evolution():
    patch = build_substitution_tier(2)
    assign_qubits_to_patch(patch, max_qubits=4)
    W = pair_couplings(patch)
    J, kappa, t, steps = 1.0, 1.0, 0.6, 60
    st = State(n_qubits=4)
    for _ in range(steps):
        trotter_step(st, W, 0, J, kappa, t / steps)
    got = np.array([st.expectation_z(p) for p in range(4)])
    ref = exact_expectation_z(W, J, kappa, t)
    assert np.allclose(got, ref, atol=2e-2), (got, ref)
    assert np.abs(ref - 1.0).max() > 0.05, "the check is trivial if <Z> never moves"


def test_feedback_changes_area_and_keeps_the_ledger():
    net = TriSpectreNetwork(qubits_per_tier=(3, 3, 3))
    for _ in range(2):
        net.step_coupled_cycle(dt=0.6, steps=3)
    rows = net.summary()
    assert all(r["mean_z"] < 0.999 for r in rows), rows
    assert all(r["area_max"] > A0 + 1e-3 and r["area_min"] < A0 - 1e-3 for r in rows), rows
    assert all(r["pair_area_sum_err"] < 1e-9 for r in rows), rows


def test_all_tiles_outside_pairs_are_untouched():
    net = TriSpectreNetwork(qubits_per_tier=(3, 3, 3))
    net.step_coupled_cycle(dt=0.6, steps=3)
    for qc in net.computers:
        in_pair = {t for p in qc.patch.pair_to_qubit for t in qc.patch.pairs[p]}
        for i, t in enumerate(qc.patch.tiles):
            if i not in in_pair:
                assert abs(t.area() - A0) < 1e-9


def run_tests(verbose: bool = True) -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            if verbose:
                print(f"  {name:<62} PASS")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  {name:<62} FAIL: {exc!r}")
    print(f"{len(tests)} test(s), {failed} failed")
    return failed


if __name__ == "__main__":
    if "--tests-only" in sys.argv:
        sys.exit(1 if run_tests() else 0)
    net = TriSpectreNetwork(qubits_per_tier=(4, 4, 4))
    print("Before coupled cycle:")
    for r in net.summary():
        print(" ", r)
    net.step_coupled_cycle(dt=0.6, steps=3)
    print("\nAfter coupled cycle:")
    for r in net.summary():
        print(" ", r)

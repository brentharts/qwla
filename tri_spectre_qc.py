"""
Step 3: Three coupled quantum computers over the 9-, 71-, and 559-tile
Spectre substitution hierarchy, with inter-tier entanglement and geometry feedback.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from bitarray import frozenbitarray as bitarray
from functional import seq

from quantumspectremodel import (
    SubstitutionPatch,
    assign_qubits_to_patch,
    build_substitution_tier,
)
from state import State, flip


def _tensor_product_states(states: List[State]) -> State:
    """
    Form a joint `State` over concatenated qubit registers [s_0, s_1, ...].
    Local index for state `m` at offset `off` is `off + j`.
    """
    total_qubits = sum(s.n_qubits for s in states)
    joint = State(n_qubits=total_qubits)

    # materialize each state's basis dictionary
    dicts = [dict(s.state.to_list()) for s in states]
    pairs = []

    # Iterate over all basis combinations across the sub-registers
    # For modest total_qubits (e.g., <= 14), functional seq over range(2**total_qubits) stays pure
    for idx in range(1 << total_qubits):
        bits = format(idx, f"0{total_qubits}b")
        amp = 1+0j
        pos = 0
        for s, d in zip(states, dicts):
            sub_key = bitarray(bits[pos : pos + s.n_qubits])
            amp *= d.get(sub_key, 0.0)
            pos += s.n_qubits
            if amp == 0j:
                break
        pairs.append((bitarray(bits), amp))

    joint.state = seq(pairs)
    return joint


def _project_marginal_state(joint: State, offset: int, width: int) -> State:
    """
    Compute the diagonal/marginal state amplitudes over slice [offset : offset + width]
    from a joint `State` by summing probabilities on matching sub-bitstrings and
    preserving relative phase from the dominant/coherent branch.
    """
    sub = State(n_qubits=width)
    # Group amplitudes by sub-bitstring
    grouped: Dict[bitarray, complex] = {}
    prob_acc: Dict[bitarray, float] = {}

    for b, a in joint.state.to_list():
        key = bitarray(b.to01()[offset : offset + width])
        prob_acc[key] = prob_acc.get(key, 0.0) + (abs(a) ** 2)
        grouped[key] = grouped.get(key, 0j) + a

    # Normalize each sub-basis amplitude to match its marginal probability magnitude
    new_pairs = []
    for i in range(1 << width):
        key = bitarray(format(i, f"0{width}b"))
        p = prob_acc.get(key, 0.0)
        amp_sum = grouped.get(key, 0j)
        if p <= 1e-18:
            new_pairs.append((key, 0j))
        elif abs(amp_sum) > 1e-18:
            new_pairs.append((key, (amp_sum / abs(amp_sum)) * math.sqrt(p)))
        else:
            new_pairs.append((key, complex(math.sqrt(p), 0.0)))

    sub.state = seq(new_pairs)
    return sub


@dataclass
class SpectreQuantumComputer:
    """A quantum computer backed by a Spectre substitution patch and functional State."""

    name: str
    patch: SubstitutionPatch
    state: State = field(init=False)

    def __post_init__(self):
        n_q = max(1, self.patch.n_qubits)
        self.state = State(n_qubits=n_q, n_bits=n_q)

    def qubit_for_vertex(self, tile_idx: int, vertex_idx: int) -> Optional[int]:
        return self.patch.tiles[tile_idx].vertices[vertex_idx].data.get("qubit")

    def apply_geometry_circuit(self, twist_scale: float = 1.0) -> "SpectreQuantumComputer":
        """
        Apply Hadamards to mapped vertex qubits and rotate around Z using
        each vertex's current geometric twist.
        """
        visited_qubits = set()
        for tile in self.patch.tiles:
            for v in tile.vertices:
                q = v.data.get("qubit")
                if q is not None and q not in visited_qubits:
                    visited_qubits.add(q)
                    self.state.h(q)
                    angle = twist_scale * float(v.twist)
                    if abs(angle) > 1e-15:
                        self.state.rz(q, angle)

        # Entangle consecutive mapped qubits within this tier's ring
        ordered_q = sorted(visited_qubits)
        for idx in range(len(ordered_q) - 1):
            self.state.cz(ordered_q[idx], ordered_q[idx + 1])
        return self

    def sync_geometry_from_state(
        self, twist_gain: float = 0.25, modulate_curve_strength: bool = False
    ) -> None:
        """
        Map <Z_q> expectation values back onto vertex twists (and optionally
        tile curve_strength) so geometry responds to the quantum state.
        """
        q_to_exp: Dict[int, float] = {}
        for q in range(self.state.n_qubits):
            q_to_exp[q] = self.state.expectation_z(q)

        for tile in self.patch.tiles:
            tile_z_vals = []
            for v in tile.vertices:
                q = v.data.get("qubit")
                if q is not None and q in q_to_exp:
                    z_val = q_to_exp[q]
                    v.twist = twist_gain * z_val
                    tile_z_vals.append(z_val)
            if modulate_curve_strength and tile_z_vals:
                mean_z = sum(tile_z_vals) / len(tile_z_vals)
                tile.curve_strength = max(-0.9, min(1.5, 0.5 + 0.2 * mean_z))


class TriSpectreNetwork:
    """
    Three coupled quantum computers built from 9-, 71-, and 559-tile Spectre patches.
    """

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        curve_strength: float = 0.5,
        qubits_per_tier: Tuple[int, int, int] = (4, 4, 4),
    ):
        self.patches: List[SubstitutionPatch] = []
        for gen, q_budget in zip((1, 2, 3), qubits_per_tier):
            p = build_substitution_tier(
                target_generation=gen,
                a=a,
                b=b,
                curve_strength=curve_strength,
                start_kind="Single",
            )
            assign_qubits_to_patch(p, mode="boundary_and_hubs", max_qubits=q_budget)
            self.patches.append(p)

        self.qc_9 = SpectreQuantumComputer("QC-9 (9 tiles)", self.patches[0])
        self.qc_71 = SpectreQuantumComputer("QC-71 (71 tiles)", self.patches[1])
        self.qc_559 = SpectreQuantumComputer("QC-559 (559 tiles)", self.patches[2])
        self.computers = (self.qc_9, self.qc_71, self.qc_559)

    def run_local_circuits(self, twist_scale: float = 1.0) -> None:
        for qc in self.computers:
            qc.apply_geometry_circuit(twist_scale=twist_scale)

    def entangle_across_computers(self, links: Optional[List[Tuple[int, int, int]]] = None) -> State:
        """
        Couple `qc_9`, `qc_71`, and `qc_559` in a joint state and apply inter-computer
        CNOT/CZ bridges across `(q_in_9, q_in_71, q_in_559)` anchor triples, then
        update each computer's local state from the entangled joint state.
        """
        if links is None:
            links = [(0, 0, 0), (1, 1, 1)]

        off_9 = 0
        off_71 = self.qc_9.state.n_qubits
        off_559 = off_71 + self.qc_71.state.n_qubits

        joint = _tensor_product_states([self.qc_9.state, self.qc_71.state, self.qc_559.state])

        for i9, i71, i559 in links:
            g9 = off_9 + (i9 % self.qc_9.state.n_qubits)
            g71 = off_71 + (i71 % self.qc_71.state.n_qubits)
            g559 = off_559 + (i559 % self.qc_559.state.n_qubits)

            # Chain entanglement across the 9-, 71-, and 559-tile computers
            joint.cx(g9, g71)
            joint.cx(g71, g559)
            joint.cz(g9, g559)

        self.qc_9.state = _project_marginal_state(joint, off_9, self.qc_9.state.n_qubits)
        self.qc_71.state = _project_marginal_state(joint, off_71, self.qc_71.state.n_qubits)
        self.qc_559.state = _project_marginal_state(joint, off_559, self.qc_559.state.n_qubits)
        return joint

    def step_coupled_cycle(self, twist_gain: float = 0.25) -> None:
        """One full cycle: local geometry circuits -> cross-computer entanglement -> geometry feedback."""
        self.run_local_circuits(twist_scale=1.0)
        self.entangle_across_computers()
        for qc in self.computers:
            qc.sync_geometry_from_state(twist_gain=twist_gain, modulate_curve_strength=True)

    def summary(self) -> List[Dict[str, object]]:
        rows = []
        for qc in self.computers:
            sample_tile = qc.patch.tiles[0]
            rows.append(
                {
                    "computer": qc.name,
                    "tiles": qc.patch.n_tiles,
                    "qubits": qc.state.n_qubits,
                    "exp_z0": round(qc.state.expectation_z(0), 4),
                    "tile0_area": round(sample_tile.area(), 4),
                    "tile0_k": round(sample_tile.curve_strength, 4),
                }
            )
        return rows


if __name__ == "__main__":
    net = TriSpectreNetwork(qubits_per_tier=(4, 4, 4))
    print("Before coupled cycle:")
    for r in net.summary():
        print(" ", r)

    net.step_coupled_cycle(twist_gain=0.3)

    print("\nAfter coupled cycle:")
    for r in net.summary():
        print(" ", r)

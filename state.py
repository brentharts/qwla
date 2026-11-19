"""
Extended functional quantum state simulator using frozenbitarray and pyfunctional.
Supports single-qubit, parameterized phase/rotation, controlled, and Toffoli gates
without dense matrix algebra.
"""
import math, random
from cmath import exp
from math import pi, sqrt
from typing import Optional
from bitarray import bitarray as mut_bitarray
from bitarray import frozenbitarray as bitarray
from functional import seq


def _check_qubit(n_qubits: int, j: int, name: str = "j") -> None:
    if not isinstance(j, int) or isinstance(j, bool) or not (0 <= j < n_qubits):
        raise IndexError(f"{name}={j} out of range for {n_qubits} qubits")


def _check_cbit(n_bits: int, cbit: int) -> None:
    if not isinstance(cbit, int) or isinstance(cbit, bool) or not (0 <= cbit < n_bits):
        raise IndexError(f"cbit={cbit} out of range for {n_bits} classical bits")


def set_bit(x: bitarray, i: int, v: int) -> bitarray:
    """Set the i-th bit of bitarray x to value v (0 or 1)."""
    if not (0 <= i < len(x)):
        raise IndexError(f"bit index {i} out of range for length {len(x)}")
    new = mut_bitarray(x)
    new[i] = int(bool(v))
    return bitarray(new)


def flip(x: bitarray, i: int) -> bitarray:
    """Flip (negate) the i-th bit of bitarray x."""
    if not (0 <= i < len(x)):
        raise IndexError(f"bit index {i} out of range for length {len(x)}")
    mask = bitarray("0" * i + "1" + "0" * (len(x) - i - 1))
    return x ^ mask


class State:
    """
    Quantum state using a functional sequence of (bitarray, amplitude) pairs.
    """

    def __init__(
        self,
        n_qubits: int,
        n_bits: Optional[int] = None,
        *,
        m_bits: Optional[int] = None,
    ):

        """
        Initialize a quantum state with n_qubits qubits and n_bits classical bits.

        Args:
            n_qubits: Number of qubits in the system
            n_bits: Number of classical bits for measurement results (default: 0)
            m_bits: Keyword-only alias for n_bits; if both are supplied, they must match

        The state starts in 0...0 (ground state).
        """

        if m_bits is not None:
            if n_bits is not None and n_bits != m_bits:
                raise ValueError("n_bits and m_bits must match when both are supplied")
            n_bits = m_bits
        if n_bits is None:
            n_bits = 0
        if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits <= 0:
            raise ValueError("n_qubits must be a positive integer")
        if not isinstance(n_bits, int) or isinstance(n_bits, bool) or n_bits < 0:
            raise ValueError("n_bits must be a non-negative integer")

        self.n_qubits = n_qubits
        self.n_bits = n_bits
        self.state = seq(
            [
                (bitarray(format(i, f"0{n_qubits}b")), complex(1.0 if i == 0 else 0.0))
                for i in range(2**n_qubits)
            ]
        )
        self.cbits = [0] * n_bits

    def x(self, j: int):
        """Apply Pauli-X (NOT) gate to qubit j. This gate flips the basis states where qubit j is present."""
        _check_qubit(self.n_qubits, j, "j")
        self.state = self.state.smap(lambda b, a: (flip(b, j), a))
        return self

    def z(self, j: int):
        """Apply Pauli-Z gate to qubit j."""
        _check_qubit(self.n_qubits, j, "j")
        self.state = self.state.smap(lambda b, a: (b, -a if b[j] else a))
        return self

    def s(self, j: int):
        """Apply S (pi/2 phase) gate to qubit j."""
        _check_qubit(self.n_qubits, j, "j")
        self.state = self.state.smap(lambda b, a: (b, (1j ** int(b[j])) * a))
        return self

    def t(self, j: int):
        """Apply T (pi/4 phase) gate to qubit j."""
        _check_qubit(self.n_qubits, j, "j")
        phase = exp(1j * pi / 4)
        self.state = self.state.smap(lambda b, a: (b, (phase ** int(b[j])) * a))
        return self

    def phase(self, j: int, theta: float):
        """Apply phase shift diag(1, exp(1j * theta)) to qubit j."""
        _check_qubit(self.n_qubits, j, "j")
        if not math.isfinite(theta):
            raise ValueError("theta must be finite")
        p = exp(1j * theta)
        self.state = self.state.smap(lambda b, a: (b, p * a if b[j] else a))
        return self

    def rz(self, j: int, theta: float):
        """Apply rotation around Z by angle theta: exp(-i*theta/2)|0><0| + exp(+i*theta/2)|1><1|."""
        _check_qubit(self.n_qubits, j, "j")
        if not math.isfinite(theta):
            raise ValueError("theta must be finite")
        p0 = exp(-0.5j * theta)
        p1 = exp(0.5j * theta)
        self.state = self.state.smap(lambda b, a: (b, (p1 if b[j] else p0) * a))
        return self

    def h(self, j: int):
        """Apply Hadamard gate to qubit j."""
        _check_qubit(self.n_qubits, j, "j")
        norm = 1.0 / sqrt(2.0)
        self.state = (
            self.state.smap(
                lambda b, a: [
                    (set_bit(b, j, 0), a * norm),
                    (set_bit(b, j, 1), a * norm * (-1.0 if b[j] else 1.0)),
                ]
            )
            .flatten()
            .reduce_by_key(lambda x, y: x + y)
        )
        return self

    def cx(self, j: int, k: int):
        """Apply controlled-X (CNOT) with control qubit j and target qubit k."""
        _check_qubit(self.n_qubits, j, "j")
        _check_qubit(self.n_qubits, k, "k")
        if j == k:
            raise ValueError("control and target qubits must be distinct")
        self.state = self.state.smap(lambda b, a: (flip(b, k) if b[j] else b, a))
        return self

    def cz(self, j: int, k: int):
        """Apply controlled-Z between qubits j and k."""
        _check_qubit(self.n_qubits, j, "j")
        _check_qubit(self.n_qubits, k, "k")
        if j == k:
            raise ValueError("control and target qubits must be distinct")
        self.state = self.state.smap(lambda b, a: (b, -a if (b[j] and b[k]) else a))
        return self

    def ccx(self, c1: int, c2: int, t: int):
        """Apply Toffoli (CCNOT) with controls c1, c2 and target t."""
        _check_qubit(self.n_qubits, c1, "c1")
        _check_qubit(self.n_qubits, c2, "c2")
        _check_qubit(self.n_qubits, t, "t")
        if len({c1, c2, t}) != 3:
            raise ValueError("control and target qubits must be distinct")
        self.state = self.state.smap(
            lambda b, a: (flip(b, t) if (b[c1] and b[c2]) else b, a)
        )
        return self

    def expectation_z(self, j: int) -> float:
        """Return <Z_j> = P(|0>_j) - P(|1>_j) without modifying the state."""
        _check_qubit(self.n_qubits, j, "j")
        prob_0 = (
            self.state.filter(lambda s: not s[0][j])
            .smap(lambda _, a: abs(a) ** 2)
            .sum()
            .real
        )
        return float(2.0 * prob_0 - 1.0)

    def measure(self, j: int, cbit: Optional[int] = None) -> "State":
        """
        Measure qubit j in the computational basis and collapse the state safely.
        Args:
            j: Index of qubit to measure
            cbit: Optional classical bit to store the measurement result
        Returns:
            The state after measurement (collapsed)
        """
        _check_qubit(self.n_qubits, j, "j")
        if cbit is not None:
            _check_cbit(self.n_bits, cbit)

        prob_0 = float(
            self.state.filter(lambda s: not s[0][j])
            .smap(lambda _, a: abs(a) ** 2)
            .sum()
            .real
        )
        prob_0 = min(1.0, max(0.0, prob_0))

        if prob_0 <= 1e-15:
            measurement = 1
        elif prob_0 >= 1.0 - 1e-15:
            measurement = 0
        else:
            measurement = int(random.random() >= prob_0)

        if cbit is not None:
            self.cbits[cbit] = measurement

        if measurement == 0:
            denom = sqrt(prob_0)
            self.state = self.state.smap(
                lambda b, a: (b, (a / denom) if not b[j] else complex(0.0, 0.0))
            )
        else:
            prob_1 = max(1e-15, 1.0 - prob_0)
            denom = sqrt(prob_1)
            self.state = self.state.smap(
                lambda b, a: (b, (a / denom) if b[j] else complex(0.0, 0.0))
            )
        return self

    def __str__(self) -> str:
        """
        Return a string representation of the quantum state.
        Format: Each bitstring with its corresponding amplitude.
        """
        ordered = self.state.sorted(key=lambda x: x[0].to01())
        result = "Quantum state:\n" + "\n".join(
            [f"{b.to01()}: {a:.2f}" for b, a in ordered]
        )
        if self.n_bits > 0:
            result += f"\n\nClassical register: {self.cbits}"
        return result

"""
Exact Spectre substitution patches as `spectremodel.Tile` objects, with bump-and-dent
pairs and one qubit per pair.

What changed from the first version (see ROADMAP_QUANTUM_GRAVITY.md, Stages 0 and 1):

* The patches are the exact edge-matched Spectre tilings of `spectre_exact.py`
  (integer arithmetic on Z[zeta_12]), not tiles placed by sector offsets.
* A Mystic is two tiles. The old code counted it as one object, which is why it
  reported 8, 63, 496 "tiles" for the sequence 9, 71, 559.
* Qubits sit on bump-and-dent pairs (two tiles whose flat 14th vertices meet alone),
  because on a Spectre tiling a tile can carry the bit only together with a partner.
  The 9-tile patch has no pair, so the default tiers are generations 2, 3, 4
  (71, 559, 4401 tiles; 6, 82, 804 pairs).
* The bit acts on the geometry exactly: displacing the flat vertex P8 by +-h along
  the normal of the chord P7-P9 changes the area by +-a*h, and the two tiles of a
  pair move oppositely, so the area of the pair is conserved (the "area ledger").

Run `python3 quantumspectremodel.py --tests-only` for the checks.
"""

import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import spectre_exact as se
from spectremodel import Tile

FLAT_EXACT = 10   # index of the flat vertex in the exact outline
FLAT_MODEL = 8    # the same vertex, P8, in spectremodel.Tile
SHIFT = FLAT_EXACT - FLAT_MODEL  # model vertex i is exact vertex (i + 2) % 14

_SPECIES = {"Single": "Delta", "Mystic": "Gamma"}
_KIND = {"p": "Single", "g1": "Mystic", "g2": "Mystic"}


@dataclass
class SubstitutionPatch:
    """The tiles of a Spectre patch at substitution generation k, with pairs and qubits."""

    generation: int
    tiles: List[Tile]
    species: str = "Delta"
    pairs: List[Tuple[int, int]] = field(default_factory=list)       # (tile A, tile B), A < B
    pair_order: List[int] = field(default_factory=list)              # indices into pairs, centre first
    node_to_vertices: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    vertex_to_node: Dict[Tuple[int, int], int] = field(default_factory=dict)
    node_to_qubit: Dict[int, int] = field(default_factory=dict)
    pair_to_qubit: Dict[int, int] = field(default_factory=dict)     # pair index -> qubit
    b: float = 1.0                                                   # second edge length of Tile(1, b); 1 = Spectre

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def n_qubits(self) -> int:
        return len(self.node_to_qubit) + len(self.pair_to_qubit)


def _xy(v) -> Tuple[float, float]:
    z = se.to_c(v)
    return (z.real, z.imag)


def _centroid(pts: np.ndarray) -> np.ndarray:
    x, y = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (3.0 * cr.sum())


def build_substitution_tier(
    target_generation: int,
    a: float = 1.0,
    b: float = 1.0,
    curve_strength: float = 0.5,
    start_kind: str = "Single",
) -> SubstitutionPatch:
    """
    Exact patch at generation k. `start_kind` "Single" gives 1, 9, 71, 559, 4401 tiles
    and "Mystic" gives 2, 8, 62, 488, 3842. The tiles are placed by the exact transforms
    at a = b = 1 (the Spectre); b != 1 then moves every vertex to its position in the
    Tile(1, b) family by `set_patch_b` (the tiling is the same for every b, only the scale
    of the two planes of the lift changes). The scale a is fixed at 1.
    """
    if target_generation < 0:
        raise ValueError("target_generation must be >= 0")
    if start_kind not in _SPECIES:
        raise ValueError(f"Unknown tile kind: {start_kind!r}")
    if not math.isclose(a, 1.0):
        raise ValueError("exact patches fix the scale a = 1 (b may vary, see set_patch_b)")
    if not b > 0.0:
        raise ValueError("b must be positive")

    species = _SPECIES[start_kind]
    leaves = se.build(target_generation)[target_generation][0][species]
    tiles: List[Tile] = []
    for idx, (T, kind) in enumerate(leaves):
        outline = se.outline(T)
        t = Tile(a=1.0, b=1.0, curve_strength=curve_strength)
        keys = []
        for i, v in enumerate(t.vertices):
            ev = outline[(i + SHIFT) % 14]
            v.position = _xy(ev)
            keys.append(tuple(int(x) for x in ev))
        t.data.update(
            kind=_KIND[kind], leaf=kind, generation=target_generation,
            tile_index=idx, exact_vertices=keys,
        )
        t.data["p8_base"] = np.array(t.vertices[FLAT_MODEL].position, dtype=float)
        tiles.append(t)

    patch = SubstitutionPatch(generation=target_generation, tiles=tiles, species=species)
    cluster_shared_vertices(patch)
    find_pairs(patch)
    if not math.isclose(b, 1.0):
        set_patch_b(patch, b)
    return patch


def cluster_shared_vertices(patch: SubstitutionPatch) -> None:
    """Group the vertices of all tiles by their exact Z^4 coordinates (no tolerance)."""
    patch.node_to_vertices.clear()
    patch.vertex_to_node.clear()
    ids: Dict[Tuple[int, ...], int] = {}
    for t_idx, tile in enumerate(patch.tiles):
        for v_idx, key in enumerate(tile.data["exact_vertices"]):
            nid = ids.setdefault(key, len(ids))
            patch.node_to_vertices.setdefault(nid, []).append((t_idx, v_idx))
            patch.vertex_to_node[(t_idx, v_idx)] = nid
            tile.vertices[v_idx].data["node_id"] = nid


def tile_centroid(tile: Tile) -> np.ndarray:
    """Centroid of the straight outline (the Bezier edges are point symmetric about their chords)."""
    return _centroid(np.array([np.asarray(v.position, float) for v in tile.vertices]))


# ---- the Spectre -> Hat family: the same tiling at every b -------------------------------
#
# A vertex is p + q w + z (r + s w) in Z[zeta_12]. Moving along the family Tile(1, b) keeps the
# integer vector and rescales the plane z (r + s w) against p + q w:
#
#     x_b(p, q, r, s) = (p + q w) + b z (r + s w),     w = e^{i pi / 3},  z = e^{i pi / 6}.
#
# Checked: at b = 1 it is the Spectre; every tile has edge lengths (b, b, a, a, b, b, a, a, a, a, b,
# b, a, a) (a = 1; Tile(1, b)) or the swapped pattern (Tile(b, 1)), and the swapped class is the
# odd-orientation (Mystic) minority, 8, 63 and 496 tiles at generations 2, 3 and 4; tile areas are
# exactly A(1, b) and A(b, 1); every flat pair consists of two Tile(1, b) tiles, so a pair has area
# 2 A(1, b); and the dipole couplings agree with the independent patch code of `toymodel_v6`
# (spectre-monotile-py, `t4_patch`) to 1e-14 at b = 1.2, 1.5 and sqrt(3).

def tile_area_ab(a: float, b: float) -> float:
    """Area of Tile(a, b): 2 sqrt3 a^2 + 3 a b + sqrt3 b^2 (independent of the curve strength)."""
    r3 = math.sqrt(3.0)
    return 2.0 * r3 * a * a + 3.0 * a * b + r3 * b * b


def exact_position(key: Sequence[int], b: float = 1.0) -> Tuple[float, float]:
    """Position at b of the lattice point `key` = (p, q, r, s) of Z[zeta_12]."""
    p, q, r, s = (int(x) for x in key)
    z = (p + q * se.W) + b * se.Z * (r + s * se.W)
    return (z.real, z.imag)


def tile_polygon_at_b(tile: Tile, b: float) -> np.ndarray:
    """The 14 straight-outline vertices (model order) of `tile` at b, from its exact vertices."""
    return np.array([exact_position(k, b) for k in tile.data["exact_vertices"]])


def set_patch_b(patch: SubstitutionPatch, b: float) -> None:
    """
    Move every tile of `patch` to Tile(1, b) geometry. Resets any bit displacement (P8 returns to
    the flat position of the new geometry) and updates `tile.data["p8_base"]` and `patch.b`.
    The curve handles keep their relative construction, so areas stay A(1, b) and A(b, 1).
    """
    if not b > 0.0:
        raise ValueError("b must be positive")
    for tile in patch.tiles:
        for v, key in zip(tile.vertices, tile.data["exact_vertices"]):
            v.position = exact_position(key, b)
        tile.data["p8_base"] = np.array(tile.vertices[FLAT_MODEL].position, dtype=float)
    patch.b = float(b)


def find_pairs(patch: SubstitutionPatch) -> List[Tuple[int, int]]:
    """
    Bump-and-dent pairs: two tiles whose flat vertices P8 coincide and meet no other
    vertex. Fills `patch.pairs` and `patch.pair_order` (nearest the patch centre first).
    """
    at: Dict[Tuple[int, ...], List[Tuple[int, int]]] = defaultdict(list)
    for t_idx, tile in enumerate(patch.tiles):
        for v_idx, key in enumerate(tile.data["exact_vertices"]):
            at[key].append((t_idx, v_idx))
    pairs = []
    for key, lst in at.items():
        if len(lst) == 2 and all(v == FLAT_MODEL for _, v in lst):
            i, j = sorted(t for t, _ in lst)
            pairs.append((i, j))
    pairs.sort()
    patch.pairs = pairs
    if pairs:
        cents = np.array([tile_centroid(t) for t in patch.tiles])
        mids = np.array([0.5 * (cents[i] + cents[j]) for i, j in pairs])
        order = np.argsort(np.linalg.norm(mids - cents.mean(axis=0), axis=1), kind="stable")
        patch.pair_order = [int(k) for k in order]
    else:
        patch.pair_order = []
    return pairs


def assign_qubits_to_patch(
    patch: SubstitutionPatch,
    mode: str = "pairs",
    max_qubits: Optional[int] = 10,
) -> Dict[int, int]:
    """
    Write `vertex.data["qubit"]` (or remove it) on every Vertex of the patch.

    Modes:
      - "pairs" (default): one qubit per bump-and-dent pair, the pairs nearest the patch
        centre first. Both tiles of a pair get the same qubit on their flat vertex P8 and
        `tile.data["pair_qubit"]`, `tile.data["pair_role"]` ("A" is the bump when <Z> = +1).
      - "nodes": one qubit per shared vertex of the tiling, the most shared first (the old
        behaviour; physically unmotivated, kept for comparison).
    Returns the map from pair index (mode "pairs") or node id (mode "nodes") to qubit.
    """
    for tile in patch.tiles:
        for v in tile.vertices:
            v.data.pop("qubit", None)
        tile.data.pop("pair_qubit", None)
        tile.data.pop("pair_role", None)
    patch.node_to_qubit = {}
    patch.pair_to_qubit = {}

    if mode == "pairs":
        chosen = patch.pair_order if max_qubits is None else patch.pair_order[:max_qubits]
        for q, p in enumerate(chosen):
            patch.pair_to_qubit[p] = q
            for role, t_idx in zip("AB", patch.pairs[p]):
                tile = patch.tiles[t_idx]
                tile.vertices[FLAT_MODEL].data["qubit"] = q
                tile.data["pair_qubit"] = q
                tile.data["pair_role"] = role
        return dict(patch.pair_to_qubit)

    if mode == "nodes":
        nodes = sorted(patch.node_to_vertices, key=lambda n: (-len(patch.node_to_vertices[n]), n))
        nodes = nodes if max_qubits is None else nodes[:max_qubits]
        patch.node_to_qubit = {nid: q for q, nid in enumerate(nodes)}
        for nid, q in patch.node_to_qubit.items():
            for t_idx, v_idx in patch.node_to_vertices[nid]:
                patch.tiles[t_idx].vertices[v_idx].data["qubit"] = q
        return dict(patch.node_to_qubit)
    raise ValueError(f"Unknown qubit assignment mode: {mode!r}")


def pair_qubit_tiles(patch: SubstitutionPatch) -> Dict[int, Tuple[Tile, Tile]]:
    """qubit -> (tile A, tile B) for the pair qubits of `patch`."""
    return {q: tuple(patch.tiles[t] for t in patch.pairs[p]) for p, q in patch.pair_to_qubit.items()}


def _dipole_matrix(cents: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    n = len(cents)
    W = np.zeros((n, n))
    inv = lambda X, Y: 1.0 / np.linalg.norm(X - Y)
    for p in range(n):
        for q in range(p + 1, n):
            Ap, Bp = cents[p]
            Aq, Bq = cents[q]
            W[p, q] = W[q, p] = inv(Ap, Aq) - inv(Ap, Bq) - inv(Bp, Aq) + inv(Bp, Bq)
    return W


def pair_couplings(patch: SubstitutionPatch) -> np.ndarray:
    """
    Dipole coupling matrix W between the pair qubits (indexed by qubit). The bit is read
    as a charge +-1 at the centroid of each tile, with opposite charges in a pair, so
    W_pq = 1/r_AA' - 1/r_AB' - 1/r_BA' + 1/r_BB' (the Coulomb energy of two dipoles,
    in units of (a*h)^2). It has both signs and falls as r^-3. Uses the current tile geometry
    (the patch's b; displacements of P8 move the centroids slightly, so call it before bits are set).
    """
    return _dipole_matrix({q: (tile_centroid(A), tile_centroid(B)) for q, (A, B) in pair_qubit_tiles(patch).items()})


def pair_couplings_at_b(patch: SubstitutionPatch, b: float) -> np.ndarray:
    """
    The same matrix at b, from the exact vertices, without touching the patch: independent of the
    patch's current b and of any bit displacement. The qubit pairs are those of `assign_qubits_to_patch`.
    """
    cents = {}
    for p, q in patch.pair_to_qubit.items():
        i, j = patch.pairs[p]
        cents[q] = (_centroid(tile_polygon_at_b(patch.tiles[i], b)), _centroid(tile_polygon_at_b(patch.tiles[j], b)))
    return _dipole_matrix(cents)


# ---- the bit acts on the geometry -------------------------------------------------------

def bump_direction(tile: Tile) -> np.ndarray:
    """Unit normal of the chord P7-P9 pointing out of the tile (the convex side of P8)."""
    p7 = np.asarray(tile.vertices[7].position, float)
    p9 = np.asarray(tile.vertices[9].position, float)
    chord = p9 - p7
    n = np.array([-chord[1], chord[0]])
    n /= np.linalg.norm(n)
    if np.dot(n, np.asarray(tile.data["p8_base"], float) - tile_centroid(tile)) < 0:
        n = -n
    return n


def set_pair_bit(tile_a: Tile, tile_b: Tile, z: float, h: float) -> None:
    """
    Set the expectation z = <tau^z> in [-1, 1] of a pair on the geometry: P8 of tile A moves
    out by z*h and P8 of tile B moves in by z*h (tile B's outward normal reversed). The area of
    A becomes A0 + a*h*z and that of B A0 - a*h*z, so the sum is conserved.
    """
    if not -1.0 - 1e-12 <= z <= 1.0 + 1e-12:
        raise ValueError("z must lie in [-1, 1]")
    for tile, sign in ((tile_a, +1.0), (tile_b, -1.0)):
        base = np.asarray(tile.data["p8_base"], float)
        tile.vertices[FLAT_MODEL].position = base + sign * z * h * bump_direction(tile)


def outline_is_simple(tile: Tile, samples: int = 40) -> bool:
    """True when the sampled Bezier outline of `tile` does not cross itself (all segment pairs, vectorised)."""
    out = tile.outline(samples=samples)
    pts = np.asarray(np.column_stack(out) if isinstance(out, tuple) else out, float)
    if np.linalg.norm(pts[0] - pts[-1]) < 1e-9:
        pts = pts[:-1]
    n = len(pts)
    P, Q = pts, np.roll(pts, -1, axis=0)

    def orient(a, b, c):   # a, b: (n, 2) segment ends; c: (m, 2) points -> (n, m)
        return ((b[:, None, 0] - a[:, None, 0]) * (c[None, :, 1] - a[:, None, 1])
                - (b[:, None, 1] - a[:, None, 1]) * (c[None, :, 0] - a[:, None, 0]))

    d1, d2 = orient(P, Q, P), orient(P, Q, Q)        # ends of segment j against segment i
    cross = (d1 * d2 < 0) & ((d1 * d2).T < 0)         # strict crossing needs both directions to straddle
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    neighbours = (gap <= 1) | (gap == n - 1)          # segments sharing an end point are not a crossing
    return not bool(np.any(cross & ~neighbours))


def bit_outlines_are_simple(patch: SubstitutionPatch, h: float, max_pairs: Optional[int] = None) -> bool:
    """
    True when both values of the bit (z = +1 and z = -1, displacement h) leave every checked pair
    tile with a simple outline. Only the qubit pairs are checked (the first `max_pairs` if given);
    each pair is returned to z = 0 afterwards. The reflex value is the one that fails first: with
    curve_strength 0.5 it is valid at h = sqrt(3) and crosses by h = 2, with curve_strength 0.8 it
    crosses already at sqrt(3) (the adjacent-edge crossing of MATH_GAPS.md O1 in spectre-monotile-py).
    """
    tiles = list(pair_qubit_tiles(patch).values())
    if max_pairs is not None:
        tiles = tiles[:max_pairs]
    ok = True
    for A, B in tiles:
        for z in (1.0, -1.0):
            set_pair_bit(A, B, z, h)
            ok = ok and outline_is_simple(A) and outline_is_simple(B)
        set_pair_bit(A, B, 0.0, h)
    return ok


def build_three_tier_hierarchy(
    a: float = 1.0,
    b: float = 1.0,
    curve_strength: float = 0.5,
    qubits_per_tier: Tuple[int, int, int] = (4, 4, 4),
    generations: Tuple[int, int, int] = (2, 3, 4),
    mode: str = "pairs",
) -> Tuple[SubstitutionPatch, SubstitutionPatch, SubstitutionPatch]:
    """Three exact tiers (default generations 2, 3, 4: 71, 559, 4401 tiles) with qubits attached."""
    tiers = []
    for gen, budget in zip(generations, qubits_per_tier):
        patch = build_substitution_tier(gen, a=a, b=b, curve_strength=curve_strength)
        assign_qubits_to_patch(patch, mode=mode, max_qubits=budget)
        tiers.append(patch)
    return tiers[0], tiers[1], tiers[2]


# ---- checks -----------------------------------------------------------------------------

A0 = 3.0 * math.sqrt(3.0) + 3.0   # area of the Spectre, a = b = 1


def test_tile_counts_are_9_71_559():
    singles = [len(build_substitution_tier(k).tiles) for k in range(0, 5)]
    mystics = [len(build_substitution_tier(k, start_kind="Mystic").tiles) for k in range(0, 5)]
    assert singles == [1, 9, 71, 559, 4401], singles
    assert mystics == [2, 8, 62, 488, 3842], mystics


def test_tiles_are_the_exact_spectre():
    patch = build_substitution_tier(2)
    for t in patch.tiles:
        assert abs(t.area() - A0) < 1e-9
        pts = np.array([np.asarray(v.position, float) for v in t.vertices])
        edges = np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)
        assert np.allclose(edges, 1.0), edges


def test_patch_is_edge_matched_with_no_overlap():
    frac = {}
    for k in (2, 3):
        patch = build_substitution_tier(k)
        seen = defaultdict(int)
        for t in patch.tiles:
            keys = t.data["exact_vertices"]
            for e in range(14):
                seen[(keys[e], keys[(e + 1) % 14])] += 1
        assert max(seen.values()) == 1, "an edge is used twice in the same direction (overlap)"
        matched = sum(1 for (u, v) in seen if (v, u) in seen)
        frac[k] = matched / len(seen)
    # unmatched edges are the patch boundary, whose share falls as the patch grows
    assert frac[3] > frac[2] > 0.8 and frac[3] > 0.9, frac


def test_pair_counts():
    counts = [len(build_substitution_tier(k).pairs) for k in range(0, 5)]
    assert counts == [0, 0, 6, 82, 804], counts


def test_pair_qubit_assignment():
    patch = build_substitution_tier(3)
    assign_qubits_to_patch(patch, max_qubits=5)
    assert patch.n_qubits == 5
    for q, (A, B) in pair_qubit_tiles(patch).items():
        assert A.vertices[FLAT_MODEL].data["qubit"] == q == B.vertices[FLAT_MODEL].data["qubit"]
        assert {A.data["pair_role"], B.data["pair_role"]} == {"A", "B"}
    W = pair_couplings(patch)
    assert W.shape == (5, 5) and np.allclose(W, W.T) and np.allclose(np.diag(W), 0.0)
    assert (W > 0).any() and (W < 0).any(), "dipole couplings have both signs"


def test_bit_changes_area_by_a_h_and_conserves_the_pair():
    patch = build_substitution_tier(2)
    assign_qubits_to_patch(patch, max_qubits=6)
    A, B = next(iter(pair_qubit_tiles(patch).values()))
    h = 0.3
    for z in (1.0, 0.4, -0.7, -1.0):
        set_pair_bit(A, B, z, h)
        assert abs(A.area() - (A0 + h * z)) < 1e-9, (z, A.area())
        assert abs(B.area() - (A0 - h * z)) < 1e-9
        assert abs(A.area() + B.area() - 2 * A0) < 1e-9
    set_pair_bit(A, B, 0.0, h)
    assert abs(A.area() - A0) < 1e-9


def test_bit_outlines_stay_simple_up_to_the_flow_value_of_h():
    # the flow of pair_hamiltonian.flow_parameters has h = sqrt(3) at the Spectre end
    patch = build_substitution_tier(2, curve_strength=0.5)
    assign_qubits_to_patch(patch, max_qubits=6)
    assert bit_outlines_are_simple(patch, 0.3)
    assert bit_outlines_are_simple(patch, math.sqrt(3.0))
    assert not bit_outlines_are_simple(patch, 2.3), "the check must be able to fail"
    steep = build_substitution_tier(2, curve_strength=0.8)
    assign_qubits_to_patch(steep, max_qubits=6)
    assert not bit_outlines_are_simple(steep, math.sqrt(3.0))
    A, B = next(iter(pair_qubit_tiles(patch).values()))
    assert abs(A.area() - A0) < 1e-9, "the check must leave the pair at z = 0"


def test_patch_at_b_is_the_tile_family_with_the_odd_class_swapped():
    r3 = math.sqrt(3.0)
    pattern_a = np.array([1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0], bool)       # True = a b-edge of Tile(1, b)
    for gen, minority in ((2, 8), (3, 63)):                                      # documented odd-orientation counts
        patch = build_substitution_tier(gen)
        base = [np.array([v.position for v in t.vertices]) for t in patch.tiles]
        b = 1.5
        set_patch_b(patch, b)
        swapped = 0
        for t in patch.tiles:
            P = tile_polygon_at_b(t, b)
            L = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
            if np.allclose(L, np.where(pattern_a, b, 1.0)):
                assert abs(t.area() - tile_area_ab(1.0, b)) < 1e-9
            else:
                assert np.allclose(L, np.where(pattern_a, 1.0, b)), L
                assert abs(t.area() - tile_area_ab(b, 1.0)) < 1e-9
                swapped += 1
            assert outline_is_simple(t)
        assert swapped == minority, (gen, swapped)
        assert patch.b == b
        set_patch_b(patch, 1.0)
        assert all(np.allclose(np.array([v.position for v in t.vertices]), P0) for t, P0 in zip(patch.tiles, base))
    assert abs(tile_area_ab(1.0, 1.0) - A0) < 1e-12 and abs(tile_area_ab(1.0, r3) - (2 * r3 + 3 * r3 + 3 * r3)) < 1e-12


def test_building_at_b_equals_moving_to_b_and_pairs_do_not_depend_on_b():
    one = build_substitution_tier(3)
    at = build_substitution_tier(3, b=1.5)
    assert one.pairs == at.pairs and at.b == 1.5
    set_patch_b(one, 1.5)
    assert all(np.allclose([v.position for v in t.vertices], [v.position for v in u.vertices]) for t, u in zip(one.tiles, at.tiles))
    sums = {round(at.tiles[i].area() + at.tiles[j].area(), 9) for i, j in at.pairs}
    assert sums == {round(2 * tile_area_ab(1.0, 1.5), 9)}, "every pair is two Tile(1, b) tiles: the ledger baseline"
    for bad in ({"b": 0.0}, {"b": -1.0}, {"a": 2.0}):
        try:
            build_substitution_tier(2, **bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be refused")


def test_couplings_at_b_match_toymodel_v6():
    # Reference values from spectre-monotile-py, research/toymodel_v6/t6_patch.py (independent patch
    # code, tiles placed by rigid motions): the 14 central pairs of the generation-3 patch.
    ref = {1.2: (0.058838762326, 0.905111729954, -0.175252192217),
           1.5: (0.060142482672, 0.934352767593, -0.182908148696),
           math.sqrt(3.0): (0.060508607724, 0.948242260986, -0.186243729110)}
    patch = build_substitution_tier(3)
    assign_qubits_to_patch(patch, max_qubits=14)
    iu = np.triu_indices(14, 1)
    for b, (wmax, wabs, wsum) in ref.items():
        W = pair_couplings_at_b(patch, b)
        assert abs(np.abs(W).max() - wmax) < 1e-10 and abs(np.abs(W[iu]).sum() - wabs) < 1e-10 and abs(W[iu].sum() - wsum) < 1e-10, b
        moved = build_substitution_tier(3, b=b)
        assign_qubits_to_patch(moved, max_qubits=14)
        assert np.allclose(pair_couplings(moved), W, atol=1e-13), "geometry at b and the exact-vertex route agree"
    assert np.allclose(pair_couplings_at_b(patch, 1.0), pair_couplings(patch), atol=1e-13)


def test_bit_at_b_changes_area_by_h_and_keeps_the_pair():
    b, h = 1.5, 0.3
    patch = build_substitution_tier(2, b=b)
    assign_qubits_to_patch(patch, max_qubits=6)
    A, B = next(iter(pair_qubit_tiles(patch).values()))
    base = tile_area_ab(1.0, b)
    for z in (1.0, -0.6, 0.0):
        set_pair_bit(A, B, z, h)
        assert abs(A.area() - (base + h * z)) < 1e-9 and abs(B.area() - (base - h * z)) < 1e-9


def test_swapped_class_is_the_mystic_with_the_closed_form_density_and_areas():
    # Cross-check with `pyCICY.theories.spectre` (brentharts/CICY), which derives these in exact Q(sqrt15), Q(sqrt3):
    # the Mystic fraction of all tiles is (5 - sqrt15)/10 and A(1, sqrt3) = 8 sqrt3, A(sqrt3, 1) = 10 sqrt3.
    r3, r15 = math.sqrt(3.0), math.sqrt(15.0)
    for gen, total, swapped in ((2, 71, 8), (3, 559, 63), (4, 4401, 496)):
        assert abs(swapped / total - (5 - r15) / 10) < 4e-5 / 10 ** (gen - 2) + 1e-7, (gen, swapped / total)
    assert abs(swapped / total - (5 - r15) / 10) < 1e-8
    assert abs(tile_area_ab(1.0, r3) - 8 * r3) < 1e-12 and abs(tile_area_ab(r3, 1.0) - 10 * r3) < 1e-12
    assert abs(tile_area_ab(r3, 1.0) - tile_area_ab(1.0, r3) - r3 * (r3 ** 2 - 1.0)) < 1e-12      # mystic asymmetry sqrt3 (b^2 - a^2)


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
    p_lo, p_mid, p_hi = build_three_tier_hierarchy()
    for p in (p_lo, p_mid, p_hi):
        print(f"k={p.generation}: tiles={p.n_tiles}, pairs={p.n_pairs}, "
              f"topological_nodes={len(p.node_to_vertices)}, mapped_qubits={p.n_qubits}")

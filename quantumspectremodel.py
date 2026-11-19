"""
Step 2: Substitution hierarchy (9, 71, 559 tiles) and vertex-to-qubit mapping
built on top of spectremodel.Tile.
"""

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple
from spectremodel import N_EDGES, Tile, edge_vectors


def _rot2d(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=float)


@dataclass
class SubstitutionPatch:
    """Collection of Spectre tiles at a given substitution generation k."""

    generation: int
    tiles: List[Tile]
    node_to_vertices: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)
    vertex_to_node: Dict[Tuple[int, int], int] = field(default_factory=dict)
    node_to_qubit: Dict[int, int] = field(default_factory=dict)

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def n_qubits(self) -> int:
        return len(self.node_to_qubit)


def _apply_rigid_transform(
    tile: Tile, offset: np.ndarray, angle_rad: float, generation: int, kind: str, parent_id: int
) -> Tile:
    """Return a copy of `tile` rotated and translated in 2D with metadata stored in `.data`."""
    out = tile.copy()
    for v in out.vertices:
        v.position = _rot2d(np.asarray(v.position, dtype=float), angle_rad) + offset
    out.data["kind"] = kind
    out.data["generation"] = generation
    out.data["parent_id"] = parent_id
    return out


def expand_single_tile(
    base_tile: Tile,
    kind: str,
    generation: int,
    parent_id: int = 0,
    step_scale: Optional[float] = None,
) -> List[Tile]:
    """
    Expand one tile according to the Spectre substitution rules:
      - Single -> 1 Mystic + 7 Singles (9 tiles)
      - Mystic -> 1 Mystic + 6 Singles (8 tiles)

    Places child tiles around local edge/sector offsets so spatial adjacency
    and shared vertex clustering can be inspected on the patch.
    """
    if kind not in ("Single", "Mystic"):
        raise ValueError(f"Unknown tile kind: {kind!r}")

    child_specs: List[str] = ["Mystic"] + (["Single"] * (7 if kind == "Single" else 6))
    ev = edge_vectors(base_tile.a, base_tile.b)
    # Characteristic span of the tile used to space child offsets cleanly
    span = step_scale if step_scale is not None else float(np.linalg.norm(ev[:4].sum(axis=0)) + 1e-6)

    children: List[Tile] = []
    for idx, child_kind in enumerate(child_specs):
        sector_angle = (idx * math.pi) / 3.0
        direction = np.array([math.cos(sector_angle), math.sin(sector_angle)], dtype=float)
        # Mystic sits at the anchor; Singles fan out around 60-degree sectors
        radius = 0.0 if idx == 0 else span * (0.85 + 0.15 * (idx % 2))
        offset = radius * direction
        child = _apply_rigid_transform(
            base_tile,
            offset=offset,
            angle_rad=sector_angle if idx > 0 else 0.0,
            generation=generation,
            kind=child_kind,
            parent_id=parent_id,
        )
        child.data["tile_index"] = idx
        children.append(child)
    return children


def build_substitution_tier(
    target_generation: int,
    a: float = 1.0,
    b: float = 1.0,
    curve_strength: float = 0.5,
    start_kind: str = "Single",
) -> SubstitutionPatch:
    """
    Build the substitution patch at generation k starting from 1 tile:
      - k=0 -> 1 tile
      - k=1 -> 9 tiles (from Single) or 8 tiles (from Mystic)
      - k=2 -> 71 tiles (from Single)
      - k=3 -> 559 tiles (from Single)
    """
    if target_generation < 0:
        raise ValueError("target_generation must be >= 0")

    seed = Tile(a=a, b=b, curve_strength=curve_strength)
    seed.data["kind"] = start_kind
    seed.data["generation"] = 0
    seed.data["tile_index"] = 0

    current_tiles = [seed]
    for gen in range(1, target_generation + 1):
        next_tiles: List[Tile] = []
        for parent_idx, ptile in enumerate(current_tiles):
            pkind = ptile.data.get("kind", "Single")
            expanded = expand_single_tile(
                ptile,
                kind=pkind,
                generation=gen,
                parent_id=parent_idx,
            )
            next_tiles.extend(expanded)
        for t_idx, t in enumerate(next_tiles):
            t.data["tile_index"] = t_idx
        current_tiles = next_tiles

    patch = SubstitutionPatch(generation=target_generation, tiles=current_tiles)
    cluster_shared_vertices(patch)
    return patch


def cluster_shared_vertices(patch: SubstitutionPatch, tol: float = 1e-6) -> None:
    """
    Group vertices across all tiles in `patch` whose 2D positions lie within `tol`
    into shared topological node IDs.
    """
    patch.node_to_vertices.clear()
    patch.vertex_to_node.clear()

    # Quantize coordinates into grid buckets for fast neighbor lookup
    inv = 1.0 / tol
    buckets: Dict[Tuple[int, int], List[int]] = {}
    node_centers: List[np.ndarray] = []

    for t_idx, tile in enumerate(patch.tiles):
        for v_idx, v in enumerate(tile.vertices):
            pos = np.asarray(v.position, dtype=float)
            bx, by = int(round(pos[0] * inv)), int(round(pos[1] * inv))

            matched_node: Optional[int] = None
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for cand_node in buckets.get((bx + dx, by + dy), ()):
                        if np.linalg.norm(pos - node_centers[cand_node]) <= tol:
                            matched_node = cand_node
                            break
                    if matched_node is not None:
                        break
                if matched_node is not None:
                    break

            if matched_node is None:
                matched_node = len(node_centers)
                node_centers.append(pos.copy())
                patch.node_to_vertices[matched_node] = []
                buckets.setdefault((bx, by), []).append(matched_node)

            key = (t_idx, v_idx)
            patch.vertex_to_node[key] = matched_node
            patch.node_to_vertices[matched_node].append(key)
            v.data["node_id"] = matched_node


def assign_qubits_to_patch(
    patch: SubstitutionPatch,
    mode: str = "boundary_and_hubs",
    max_qubits: Optional[int] = 10,
) -> Dict[int, int]:
    """
    Assign local qubit indices (0 .. n_qubits - 1) to selected topological nodes
    in `patch`, writing `vertex.data["qubit"]` (or `None` if unmapped) on every
    `Vertex` in the patch.

    Modes:
      - "unique_nodes": map every shared node in the patch (warning: large for k >= 2).
      - "boundary_and_hubs": prioritize shared/boundary nodes and tile-0 anchor
        vertices up to `max_qubits` so `State` simulations remain tractable.
    """
    for tile in patch.tiles:
        for v in tile.vertices:
            v.data.pop("qubit", None)

    all_nodes = sorted(patch.node_to_vertices.keys())
    if mode == "unique_nodes":
        selected_nodes = all_nodes if max_qubits is None else all_nodes[:max_qubits]
    elif mode == "boundary_and_hubs":
        # Rank nodes by how many tile vertices share them, then by tile/vertex order
        shared_first = sorted(
            all_nodes,
            key=lambda nid: (-len(patch.node_to_vertices[nid]), nid),
        )
        selected_nodes = shared_first if max_qubits is None else shared_first[:max_qubits]
    else:
        raise ValueError(f"Unknown qubit assignment mode: {mode!r}")

    node_to_q = {nid: q_idx for q_idx, nid in enumerate(selected_nodes)}
    patch.node_to_qubit = node_to_q

    for nid, q_idx in node_to_q.items():
        for t_idx, v_idx in patch.node_to_vertices[nid]:
            patch.tiles[t_idx].vertices[v_idx].data["qubit"] = q_idx

    return node_to_q


def build_three_tier_hierarchy(
    a: float = 1.0,
    b: float = 1.0,
    curve_strength: float = 0.5,
    qubits_per_tier: Tuple[int, int, int] = (6, 8, 10),
    mode: str = "boundary_and_hubs",
) -> Tuple[SubstitutionPatch, SubstitutionPatch, SubstitutionPatch]:
    """
    Construct the three Spectre substitution tiers with 9, 71, and 559 tiles
    (generations k = 1, 2, 3 starting from a Single tile) and attach qubit
    mappings to their vertices.
    """
    tiers = []
    for gen, q_budget in zip((1, 2, 3), qubits_per_tier):
        patch = build_substitution_tier(
            target_generation=gen,
            a=a,
            b=b,
            curve_strength=curve_strength,
            start_kind="Single",
        )
        assign_qubits_to_patch(patch, mode=mode, max_qubits=q_budget)
        tiers.append(patch)
    return tiers[0], tiers[1], tiers[2]


if __name__ == "__main__":
    p9, p71, p559 = build_three_tier_hierarchy(qubits_per_tier=(6, 7, 8))
    for p in (p9, p71, p559):
        print(
            f"k={p.generation}: tiles={p.n_tiles}, topological_nodes={len(p.node_to_vertices)}, "
            f"mapped_qubits={p.n_qubits}"
        )

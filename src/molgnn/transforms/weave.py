"""Sparse 2D pair inputs for the 2016 Weave architecture.

The original Weave model keeps a representation for ordered atom pairs in
addition to atom representations.  This transform derives that pair view from
one canonical, unbatched molecular graph without materialising a dense
``[N, N, F]`` tensor.  The built-in profile is the common W2N2 setting: all
self-pairs and all atom pairs at graph distance at most two.

``weave_pair_attr`` has the project-local 22-wide profile:

* canonical 14-wide bond features for covalent pairs (zeros otherwise),
* one same-ring indicator, and
* seven exact shortest-path-distance indicators for distances 1 through 7.

Only the first two distance channels can be active in this W2N2 transform;
the full seven-wide block preserves a stable Weave pair-feature contract.
"""

from __future__ import annotations

from collections import Counter

import torch
from rdkit import Chem
from torch import Tensor

from ..data import MolecularData
from ..featurizer import CANONICAL_FEATURE_SCHEMA_V1, atom_features, bond_features
from .base import TransformError

WEAVE_MAX_PAIR_DISTANCE = 2
WEAVE_SAME_RING_DIM = 1
WEAVE_GRAPH_DISTANCE_DIM = 7
WEAVE_PAIR_FEATURE_DIM = (
    CANONICAL_FEATURE_SCHEMA_V1.bond_dim
    + WEAVE_SAME_RING_DIM
    + WEAVE_GRAPH_DISTANCE_DIM
)


def add_weave_inputs(data: MolecularData) -> MolecularData:
    """Clone one canonical graph and attach sparse W2N2 Weave pair tensors.

    The source graph must be an unbatched canonical-2D sample whose SMILES,
    atom count, directed covalent connectivity, and 14-wide bond features are
    consistent with one another.  The returned clone has:

    * ``weave_pair_index``: ordered ``[2, Q]`` atom pairs, including every
      self-pair and both directions of every non-self pair within two bonds;
    * ``weave_pair_attr``: ``[Q, 22]`` features aligned with those pairs.

    This transform is deliberately independent of targets, coordinates, and
    generic batching.  It must run before PyG batches the samples.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; Weave inputs must be derived before PyG batching"
        )

    smiles = getattr(data, "smiles", None)
    if not isinstance(smiles, str) or not smiles:
        raise TransformError(f"sample {sample} is missing source SMILES metadata")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TransformError(f"sample {sample} has invalid source SMILES")

    x = _tensor(data, "x")
    edge_index = _tensor(data, "edge_index")
    edge_attr = _tensor(data, "edge_attr")
    _validate_canonical_graph(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        mol=mol,
        sample=sample,
    )

    pair_index, pair_attr = _weave_pairs(
        mol=mol,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=x.shape[0],
    )
    if pair_index.shape[0] != 2 or pair_attr.shape != (
        pair_index.shape[1],
        WEAVE_PAIR_FEATURE_DIM,
    ):
        raise AssertionError("Weave pair feature construction produced an invalid shape")

    transformed = data.clone()
    transformed.weave_pair_index = pair_index
    transformed.weave_pair_attr = pair_attr
    return transformed


def _validate_canonical_graph(
    *,
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
    mol: Chem.Mol,
    sample: int | str,
) -> None:
    """Validate the canonical 2D profile and its agreement with SMILES."""

    node_count = mol.GetNumAtoms()
    if (
        x.ndim != 2
        or x.shape != (node_count, CANONICAL_FEATURE_SCHEMA_V1.atom_dim)
        or x.dtype != torch.float32
        or not torch.isfinite(x).all()
    ):
        raise TransformError(
            f"sample {sample} must provide finite canonical float32 x with shape "
            f"[N, {CANONICAL_FEATURE_SCHEMA_V1.atom_dim}] matching source SMILES"
        )
    if node_count < 1:
        raise TransformError(f"sample {sample} must provide at least one atom")
    expected_x = torch.stack(
        [atom_features(atom) for atom in mol.GetAtoms()], dim=0
    ).to(device=x.device)
    if not torch.equal(x, expected_x):
        raise TransformError(
            f"sample {sample} x does not match canonical source-atom features"
        )
    if (
        edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
        or edge_index.device != x.device
    ):
        raise TransformError(
            f"sample {sample} edge_index must be [2, E] torch.long on the node device"
        )
    edge_count = edge_index.shape[1]
    if (
        edge_attr.shape
        != (edge_count, CANONICAL_FEATURE_SCHEMA_V1.bond_dim)
        or edge_attr.dtype != torch.float32
        or edge_attr.device != x.device
        or not torch.isfinite(edge_attr).all()
    ):
        raise TransformError(
            f"sample {sample} must provide finite canonical float32 edge_attr "
            f"with shape [E, {CANONICAL_FEATURE_SCHEMA_V1.bond_dim}]"
        )
    if edge_count and (edge_index.min() < 0 or edge_index.max() >= node_count):
        raise TransformError(f"sample {sample} edge_index contains an invalid node")
    if edge_count and bool((edge_index[0] == edge_index[1]).any()):
        raise TransformError(f"sample {sample} edge_index must not contain self-loops")

    expected_edges = [
        direction
        for bond in mol.GetBonds()
        for direction in (
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()),
        )
    ]
    actual_edges = [tuple(pair) for pair in edge_index.t().detach().cpu().tolist()]
    if Counter(actual_edges) != Counter(expected_edges):
        raise TransformError(
            f"sample {sample} edge_index does not match source SMILES connectivity"
        )

    for row, (source, target) in enumerate(actual_edges):
        bond = mol.GetBondBetweenAtoms(source, target)
        if bond is None:
            raise TransformError(
                f"sample {sample} has no source bond for edge {source}->{target}"
            )
        expected_attr = bond_features(bond).to(device=x.device)
        if not torch.equal(edge_attr[row], expected_attr):
            raise TransformError(
                f"sample {sample} edge_attr does not match canonical source-bond features"
            )


def _weave_pairs(
    *,
    mol: Chem.Mol,
    edge_index: Tensor,
    edge_attr: Tensor,
    num_nodes: int,
) -> tuple[Tensor, Tensor]:
    """Return deterministic ordered W2N2 pairs and aligned 22-wide features."""

    # The validated graph has exactly one directed edge per covalent bond
    # direction.  Keep its actual feature rows so pair attributes remain tied
    # to the project canonical schema rather than a second featurizer copy.
    actual_edges = [tuple(pair) for pair in edge_index.t().detach().cpu().tolist()]
    bond_rows = {pair: row for row, pair in enumerate(actual_edges)}
    neighbours: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target in actual_edges:
        neighbours[source].add(target)

    ring_pairs = _same_ring_pairs(mol)
    pair_records: list[tuple[int, int, int]] = []
    for source in range(num_nodes):
        distances = _bounded_distances(
            source=source,
            neighbours=neighbours,
            maximum_distance=WEAVE_MAX_PAIR_DISTANCE,
        )
        for target in sorted(distances):
            pair_records.append((source, target, distances[target]))

    device = edge_index.device
    pair_index = torch.tensor(
        [(source, target) for source, target, _distance in pair_records],
        dtype=torch.long,
        device=device,
    ).t().contiguous()
    pair_attr = torch.zeros(
        (len(pair_records), WEAVE_PAIR_FEATURE_DIM),
        dtype=torch.float32,
        device=device,
    )
    bond_dim = CANONICAL_FEATURE_SCHEMA_V1.bond_dim
    distance_offset = bond_dim + WEAVE_SAME_RING_DIM
    for row, (source, target, distance) in enumerate(pair_records):
        bond_row = bond_rows.get((source, target))
        if bond_row is not None:
            pair_attr[row, :bond_dim] = edge_attr[bond_row]
        if (source, target) in ring_pairs:
            pair_attr[row, bond_dim] = 1.0
        if distance:
            # All emitted non-self pairs have distance <= 2, but retain the
            # seven-channel public feature schema used by Weave profiles.
            pair_attr[row, distance_offset + distance - 1] = 1.0
    return pair_index, pair_attr


def _bounded_distances(
    *, source: int, neighbours: list[set[int]], maximum_distance: int
) -> dict[int, int]:
    """Find graph distances no larger than ``maximum_distance`` from one atom."""

    distances = {source: 0}
    frontier = [source]
    for distance in range(1, maximum_distance + 1):
        next_frontier: list[int] = []
        for node in frontier:
            for neighbour in sorted(neighbours[node]):
                if neighbour not in distances:
                    distances[neighbour] = distance
                    next_frontier.append(neighbour)
        if not next_frontier:
            break
        frontier = next_frontier
    return distances


def _same_ring_pairs(mol: Chem.Mol) -> set[tuple[int, int]]:
    """Return ordered atom pairs that occur together in one RDKit SSSR ring."""

    pairs: set[tuple[int, int]] = set()
    for ring in Chem.GetSymmSSSR(mol):
        atom_indices = tuple(int(atom_index) for atom_index in ring)
        for source in atom_indices:
            pairs.update((source, target) for target in atom_indices)
    return pairs


def _tensor(data: MolecularData, name: str) -> Tensor:
    value = getattr(data, name, None)
    if not isinstance(value, Tensor):
        raise TransformError(f"sample {_sample_id(data)} is missing tensor field '{name}'")
    return value


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.reshape(-1)[0].item())
    return "<unknown>"


__all__ = [
    "WEAVE_GRAPH_DISTANCE_DIM",
    "WEAVE_MAX_PAIR_DISTANCE",
    "WEAVE_PAIR_FEATURE_DIM",
    "WEAVE_SAME_RING_DIM",
    "add_weave_inputs",
]

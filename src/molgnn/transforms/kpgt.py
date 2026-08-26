"""Exact KPGT line-graph, path, and knowledge inputs derived from SMILES.

Everything in this module is model-specific preparation for the KPGT
architecture (PAPER: KDD 2022; OFFICIAL CODE revision
``47dc1646c70b2138a157de481d24a1ac35d174cd``).  It deliberately avoids the
shared featurizer because the canonical schema lacks radical-electron,
mass, and chiral-center fields required by the official 137/14 layout.

Contract replicated from ``OFFICIAL CODE src/data/featurizer.py``
(``smiles_to_graph_tune``) with the fixed downstream settings
(``max_length=5``, ``n_virtual_nodes=2``, ``add_self_loop=True``):

- canonical RDKit renumbering via ``CanonicalRankAtoms``;
- one line-node per undirected bond (sorted atom pair), plus one node per
  unbonded atom (indicator ``-1``) so zero-bond molecules still run;
- directed attention edges between line-nodes sharing an atom, along every
  shortest molecular path of 4..6 atoms expressed as line-node sequences,
- two knowledge nodes (fingerprint indicator ``1``, descriptor indicator
  ``2``) connected to every earlier node in both directions;
- a self-loop on every node.

The transform never reads or creates coordinates: KPGT is pure 2-D topology.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from rdkit import Chem
from torch import Tensor

from ..data import MolecularData
from ..models.kpgt_2022.constants import (
    D_EDGE_FEATS,
    D_NODE_FEATS,
    DESCRIPTOR_DIM,
    N_ATOM_TYPES,
    N_BOND_TYPES,
    VIRTUAL_ATOM_FEATURE_PLACEHOLDER,
    VIRTUAL_ATOM_INDICATOR,
    VIRTUAL_BOND_FEATURE_PLACEHOLDER,
    VIRTUAL_PATH_INDICATOR,
    KPGTVocab,
)
from ..models.kpgt_2022.descriptors import compute_rdkit2d_normalized
from .base import TransformError

KPGT_PATH_LENGTH = 5
KPGT_VIRTUAL_NODES = 2
FINGERPRINT_BITS = 512


# --- Exact official atom/bond featurizers (dgllife 0.2.8 semantics) ---------

_ATOMIC_NUMBERS = list(range(1, 101))
_DEGREES = list(range(11))
_RADICAL_ELECTRONS = list(range(5))
_HYBRIDIZATIONS = (
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
)
_TOTAL_H = list(range(5))
_BOND_TYPES = (
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
)
_BOND_STEREO = (
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
)


def _one_hot_unknown(value, allowable_set) -> list[bool]:
    """dgllife ``one_hot_encoding`` with ``encode_unknown=True``.

    dgllife appends one ``None`` bucket and marks it when the value is
    outside the allowed set; in-set values leave that bucket False.
    """

    extended = (*allowable_set, None)
    return [value == item for item in extended]


def _atom_feature_vector(atom: Chem.Atom) -> list[float]:
    atomic_number_block = _one_hot_unknown(atom.GetAtomicNum(), _ATOMIC_NUMBERS)
    degree_block = _one_hot_unknown(atom.GetDegree(), _DEGREES)
    formal_charge_block = [float(atom.GetFormalCharge())]
    radical_block = _one_hot_unknown(atom.GetNumRadicalElectrons(), _RADICAL_ELECTRONS)
    hybridization_block = _one_hot_unknown(atom.GetHybridization(), _HYBRIDIZATIONS)
    aromatic_block = [bool(atom.GetIsAromatic())]
    total_h_block = _one_hot_unknown(atom.GetTotalNumHs(), _TOTAL_H)
    chiral_center_block = [bool(atom.HasProp("_ChiralityPossible"))]
    if not atom.HasProp("_CIPCode"):
        chirality_block = [False, False]
    else:
        cip_code = atom.GetProp("_CIPCode")
        chirality_block = [cip_code == "R", cip_code == "S"]
    mass_block = [atom.GetMass() * 0.01]

    values = [
        *atomic_number_block,
        *degree_block,
        *formal_charge_block,
        *radical_block,
        *hybridization_block,
        *aromatic_block,
        *total_h_block,
        *chiral_center_block,
        *chirality_block,
        *mass_block,
    ]
    if len(values) != D_NODE_FEATS:
        raise AssertionError("KPGT atom feature width must be 137")
    return values


def _bond_feature_vector(bond: Chem.Bond) -> list[float]:
    type_block = _one_hot_unknown(bond.GetBondType(), _BOND_TYPES)
    conjugated_block = [bool(bond.GetIsConjugated())]
    ring_block = [bool(bond.IsInRing())]
    stereo_block = _one_hot_unknown(bond.GetStereo(), _BOND_STEREO)
    values = [*type_block, *conjugated_block, *ring_block, *stereo_block]
    if len(values) != D_EDGE_FEATS:
        raise AssertionError("KPGT bond feature width must be 14")
    return values


def kpgt_atom_type_index(feature: list[float] | Tensor) -> int:
    """Atomic-number block bucket used by the vocabulary lookup."""

    block = [float(value) for value in list(feature)[:N_ATOM_TYPES]]
    return block.index(1.0)


def kpgt_bond_type_index(feature: list[float] | Tensor) -> int:
    """Bond-type block bucket used by the vocabulary lookup."""

    block = [float(value) for value in list(feature)[:N_BOND_TYPES]]
    return block.index(1.0)


def compute_kpgt_fingerprint(mol: Chem.Mol) -> list[float]:
    """Exact 512-bit ``RDKFingerprint(minPath=1, maxPath=7)``."""

    vector = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=FINGERPRINT_BITS)
    return [float(bit) for bit in vector]


# --- Line graph / path construction -----------------------------------------


def build_kpgt_line_graph(smiles: str) -> dict[str, object]:
    """Replicate ``smiles_to_graph_tune`` for the fixed downstream contract."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TransformError(f"invalid SMILES for KPGT inputs: {smiles!r}")
    new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
    mol = Chem.rdmolops.RenumberAtoms(mol, new_order)

    n_atoms = mol.GetNumAtoms()
    vocab = _shared_vocab()
    atom_features = [_atom_feature_vector(atom) for atom in mol.GetAtoms()]

    virtual_placeholder_atom = [float(VIRTUAL_ATOM_FEATURE_PLACEHOLDER)] * D_NODE_FEATS
    placeholder_bond = [float(VIRTUAL_BOND_FEATURE_PLACEHOLDER)] * D_EDGE_FEATS

    begin_end: list[list[list[float]]] = []
    bond_attrs: list[list[float]] = []
    triplet_labels: list[int] = []
    indicators: list[int] = []
    pair_to_triplet: list[list[float]] = [
        [float("nan")] * n_atoms for _ in range(n_atoms)
    ]

    bonded_atoms: set[int] = set()
    for triplet_id, bond in enumerate(mol.GetBonds()):
        first, second = sorted((bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()))
        begin_end.append([atom_features[first], atom_features[second]])
        bond_feature = _bond_feature_vector(bond)
        bond_attrs.append(bond_feature)
        bonded_atoms.add(first)
        bonded_atoms.add(second)
        triplet_labels.append(
            vocab.index(
                kpgt_atom_type_index(atom_features[first]),
                kpgt_atom_type_index(atom_features[second]),
                kpgt_bond_type_index(bond_feature),
            )
        )
        indicators.append(0)
        pair_to_triplet[first][second] = triplet_id
        pair_to_triplet[second][first] = triplet_id

    for atom_id in range(n_atoms):
        if atom_id in bonded_atoms:
            continue
        begin_end.append([atom_features[atom_id], list(virtual_placeholder_atom)])
        bond_attrs.append(list(placeholder_bond))
        triplet_labels.append(
            vocab.index(kpgt_atom_type_index(atom_features[atom_id]), 999, 999)
        )
        indicators.append(VIRTUAL_ATOM_INDICATOR)

    edges: list[tuple[int, int]] = []
    paths: list[list[int]] = []
    virtual_path_flags: list[int] = []
    self_loop_flags: list[int] = []

    def add_edge(source: int, target: int, path: list[int], *, virtual: int, self_loop: int) -> None:
        edges.append((source, target))
        paths.append(path)
        virtual_path_flags.append(virtual)
        self_loop_flags.append(self_loop)

    # Line-graph edges: permutations of the triplets sharing each atom.
    for atom_id in range(n_atoms):
        node_ids = [
            int(value)
            for value in pair_to_triplet[atom_id]
            if not math.isnan(value)
        ]
        if len(node_ids) < 2:
            continue
        for source in node_ids:
            for target in node_ids:
                if source == target:
                    continue
                path = [int(source)] + [VIRTUAL_PATH_INDICATOR] * (KPGT_PATH_LENGTH - 2) + [int(target)]
                add_edge(int(source), int(target), path, virtual=0, self_loop=0)

    # Molecular shortest paths of 4..6 atoms expressed over shared line-nodes.
    adjacency: list[list[int]] = [[] for _ in range(n_atoms)]
    for bond in mol.GetBonds():
        begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[begin].append(end)
        adjacency[end].append(begin)
    for neighbors in adjacency:
        neighbors.sort()

    cutoff = KPGT_PATH_LENGTH + 1
    for source in range(n_atoms):
        discovery = _bfs_paths(adjacency, source, cutoff)
        for target, atom_path in discovery:
            path_length = len(atom_path)
            if not 3 < path_length <= cutoff:
                continue
            triplet_ids = [
                int(pair_to_triplet[atom_path[step]][atom_path[step + 1]])
                for step in range(path_length - 1)
            ]
            start_triplet = triplet_ids[0]
            end_triplet = triplet_ids[-1]
            middle = triplet_ids[1:-1]
            padding = KPGT_PATH_LENGTH - len(middle) - 2
            triplet_path = [start_triplet, *middle, *([VIRTUAL_PATH_INDICATOR] * padding), end_triplet]
            add_edge(start_triplet, end_triplet, triplet_path, virtual=0, self_loop=0)

    # Two knowledge nodes connected to all previously created nodes.
    for knowledge_index in range(KPGT_VIRTUAL_NODES):
        knowledge_id = len(begin_end)
        for existing in range(knowledge_id):
            forward = [knowledge_id] + [VIRTUAL_PATH_INDICATOR] * (KPGT_PATH_LENGTH - 2) + [existing]
            backward = [existing] + [VIRTUAL_PATH_INDICATOR] * (KPGT_PATH_LENGTH - 2) + [knowledge_id]
            add_edge(knowledge_id, existing, forward, virtual=knowledge_index + 1, self_loop=0)
            add_edge(existing, knowledge_id, backward, virtual=knowledge_index + 1, self_loop=0)
        begin_end.append(
            [list(virtual_placeholder_atom), list(virtual_placeholder_atom)]
        )
        bond_attrs.append(list(placeholder_bond))
        triplet_labels.append(vocab.index(999, 999, 999))
        indicators.append(knowledge_index + 1)

    # Self-loops on every node.
    total_nodes = len(begin_end)
    for node in range(total_nodes):
        loop_path = [node] + [VIRTUAL_PATH_INDICATOR] * (KPGT_PATH_LENGTH - 2) + [node]
        add_edge(node, node, loop_path, virtual=0, self_loop=1)

    fingerprint = compute_kpgt_fingerprint(mol)
    descriptor = compute_rdkit2d_normalized(mol)

    return {
        "begin_end": begin_end,
        "bond_attrs": bond_attrs,
        "labels": triplet_labels,
        "indicators": indicators,
        "edges": edges,
        "paths": paths,
        "virtual_path": virtual_path_flags,
        "self_loop": self_loop_flags,
        "fingerprint": fingerprint,
        "descriptor": descriptor,
    }


def add_kpgt_inputs(data: MolecularData) -> MolecularData:
    """Attach exact KPGT topology, knowledge, and path tensors to one sample."""

    smiles = getattr(data, "smiles", None)
    if not isinstance(smiles, str) or not smiles:
        sample = _sample_id(data)
        raise TransformError(f"sample {sample} requires a non-empty 'smiles' for KPGT inputs")
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {_sample_id(data)} is already batched; KPGT inputs must be derived before batching"
        )

    graph = build_kpgt_line_graph(smiles)
    transformed = data.clone()
    transformed.kpgt_begin_end = torch.tensor(
        graph["begin_end"], dtype=torch.float32
    ).reshape(len(graph["indicators"]), 2, D_NODE_FEATS)
    transformed.kpgt_bond_attr = torch.tensor(
        graph["bond_attrs"], dtype=torch.float32
    ).reshape(len(graph["indicators"]), D_EDGE_FEATS)
    transformed.kpgt_node_indicator = torch.tensor(graph["indicators"], dtype=torch.long)
    transformed.kpgt_triplet_label = torch.tensor(graph["labels"], dtype=torch.long)
    edge_index = (
        torch.tensor(graph["edges"], dtype=torch.long).t().contiguous()
        if graph["edges"]
        else torch.empty((2, 0), dtype=torch.long)
    )
    transformed.kpgt_attention_edge_index = edge_index
    transformed.kpgt_path_index = torch.tensor(
        graph["paths"], dtype=torch.long
    ).reshape(edge_index.shape[1], KPGT_PATH_LENGTH)
    transformed.kpgt_virtual_path = torch.tensor(graph["virtual_path"], dtype=torch.bool)
    transformed.kpgt_self_loop = torch.tensor(graph["self_loop"], dtype=torch.bool)
    transformed.kpgt_fingerprint = torch.tensor(
        graph["fingerprint"], dtype=torch.float32
    ).reshape(1, FINGERPRINT_BITS)
    transformed.kpgt_descriptor = torch.tensor(
        graph["descriptor"], dtype=torch.float32
    ).reshape(1, DESCRIPTOR_DIM)
    real_nodes = int((transformed.kpgt_node_indicator <= 0).sum())
    transformed.kpgt_token_count = torch.tensor([real_nodes], dtype=torch.long)
    return transformed


def _bfs_paths(
    adjacency: list[list[int]], source: int, cutoff: int
) -> list[tuple[int, list[int]]]:
    """NetworkX-equivalent BFS paths in discovery order up to ``cutoff`` nodes."""

    order: list[tuple[int, list[int]]] = [(source, [source])]
    visited = {source}
    queue: deque[tuple[int, list[int]]] = deque([(source, [source])])
    while queue:
        current, path = queue.popleft()
        if len(path) >= cutoff:
            continue
        for neighbor in adjacency[current]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            extended = [*path, neighbor]
            order.append((neighbor, extended))
            queue.append((neighbor, extended))
    return order


_VOCAB: KPGTVocab | None = None


def _shared_vocab() -> KPGTVocab:
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = KPGTVocab(N_ATOM_TYPES, N_BOND_TYPES)
    return _VOCAB


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = [
    "FINGERPRINT_BITS",
    "KPGT_PATH_LENGTH",
    "KPGT_VIRTUAL_NODES",
    "add_kpgt_inputs",
    "build_kpgt_line_graph",
    "compute_kpgt_fingerprint",
]

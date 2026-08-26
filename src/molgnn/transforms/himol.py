"""HiMol's deterministic atom--motif--graph topology view.

The transform owns all BRICS/ring preprocessing and leaves the canonical
``MolecularData`` graph untouched.  The model view is built from SMILES, as in
the author implementation, so native QM9 coordinates and explicit-H SDF nodes
are deliberately not consumed by this 2-D architecture.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from torch import Tensor

from ..data import MolecularData
from .base import TransformError

ATOM_TYPE_COUNT = 121
DEGREE_COUNT = 11
BOND_TYPE_COUNT = 7
BOND_META_COUNT = 3
GRAPH_TOKEN_ID = 119
MOTIF_TOKEN_ID = 120
SELF_LOOP_EDGE_ID = 4
MOTIF_GRAPH_EDGE_ID = 5
ATOM_MOTIF_EDGE_ID = 6

_BOND_TYPE_TO_ID = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


class HiMolData(MolecularData):
    """Molecular sample with a separately indexed hierarchical graph."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        node_attr = getattr(self, "himol_node_attr", None)
        num_hierarchy_nodes = (
            int(node_attr.shape[0]) if isinstance(node_attr, Tensor) else 0
        )
        if key in {
            "himol_edge_index",
            "himol_atom_node_index",
            "himol_graph_node_index",
        }:
            return num_hierarchy_nodes
        if key == "himol_bond_index":
            atom_target = getattr(self, "himol_atom_target", None)
            return int(atom_target.shape[0]) if isinstance(atom_target, Tensor) else 0
        if key == "himol_batch":
            return 1
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in {"himol_edge_index", "himol_bond_index"}:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def decompose_himol_motifs(mol: Chem.Mol) -> tuple[tuple[int, ...], ...]:
    """Return deterministic BRICS fragments with multi-ring refinement.

    This is the chemical intent of ``data_utils.motif_decomp`` without its
    set-order dependence.  A multi-ring BRICS fragment becomes its contained
    minimum rings plus a non-empty non-ring residual.
    """

    num_atoms = mol.GetNumAtoms()
    if num_atoms < 1:
        return ()
    if num_atoms == 1:
        return ((0,),)

    broken = {
        frozenset((int(pair[0][0]), int(pair[0][1])))
        for pair in BRICS.FindBRICSBonds(mol)
    }
    adjacency: list[set[int]] = [set() for _ in range(num_atoms)]
    for bond in mol.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        if frozenset((source, target)) in broken:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)

    components: list[tuple[int, ...]] = []
    remaining = set(range(num_atoms))
    while remaining:
        root = min(remaining)
        stack = [root]
        component: set[int] = set()
        while stack:
            atom = stack.pop()
            if atom in component:
                continue
            component.add(atom)
            remaining.discard(atom)
            stack.extend(sorted(adjacency[atom] - component, reverse=True))
        if len(component) < num_atoms:
            components.append(tuple(sorted(component)))

    rings = tuple(
        tuple(sorted(int(atom) for atom in ring)) for ring in Chem.GetSymmSSSR(mol)
    )
    motifs: list[tuple[int, ...]] = []
    for component in components:
        component_set = set(component)
        contained = [ring for ring in rings if set(ring).issubset(component_set)]
        if len(contained) <= 1:
            motifs.append(component)
            continue
        motifs.extend(contained)
        ring_atoms = set().union(*(set(ring) for ring in contained))
        residual = tuple(sorted(component_set - ring_atoms))
        if residual:
            motifs.append(residual)

    return tuple(sorted(set(motifs), key=lambda motif: (motif[0], len(motif), motif)))


def add_himol_inputs(data: MolecularData) -> HiMolData:
    """Attach HiMol's augmented hierarchy and pretraining targets."""

    sample = _sample_name(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; HiMol inputs must be built before batching"
        )
    smiles = getattr(data, "smiles", None)
    if not isinstance(smiles, str) or not smiles:
        raise TransformError(f"sample {sample} requires a non-empty 'smiles' for HiMol")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() < 1:
        raise TransformError(f"sample {sample} has invalid or empty SMILES {smiles!r}")

    atom_rows: list[list[int]] = []
    for atom in mol.GetAtoms():
        atomic_number = int(atom.GetAtomicNum())
        degree = int(atom.GetDegree())
        if not 1 <= atomic_number <= 118:
            raise TransformError(
                f"sample {sample} atomic number {atomic_number} is outside HiMol's vocabulary"
            )
        if not 0 <= degree < DEGREE_COUNT:
            raise TransformError(
                f"sample {sample} atom degree {degree} is outside HiMol's vocabulary"
            )
        atom_rows.append([atomic_number - 1, degree])

    chemical_edges: list[tuple[int, int]] = []
    chemical_attrs: list[tuple[int, int]] = []
    physical_bonds: list[tuple[int, int]] = []
    physical_bond_types: list[int] = []
    for bond in mol.GetBonds():
        bond_type = _BOND_TYPE_TO_ID.get(bond.GetBondType())
        if bond_type is None:
            raise TransformError(
                f"sample {sample} bond type {bond.GetBondType()!r} is unsupported by HiMol"
            )
        source, target = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        in_ring = 2 if bond.IsInRing() else 1
        chemical_edges.extend(((source, target), (target, source)))
        chemical_attrs.extend(((bond_type, in_ring), (bond_type, in_ring)))
        physical_bonds.append((source, target))
        physical_bond_types.append(bond_type)

    motifs = decompose_himol_motifs(mol)
    num_atoms = len(atom_rows)
    num_motifs = len(motifs)
    graph_node = num_atoms + num_motifs
    node_rows = atom_rows + [[MOTIF_TOKEN_ID, 0]] * num_motifs + [[GRAPH_TOKEN_ID, 0]]

    hierarchy_edges: list[tuple[int, int]] = []
    hierarchy_attrs: list[tuple[int, int]] = []
    if motifs:
        for motif_offset, motif in enumerate(motifs):
            motif_node = num_atoms + motif_offset
            for atom in motif:
                hierarchy_edges.extend(((atom, motif_node), (motif_node, atom)))
                hierarchy_attrs.extend(((ATOM_MOTIF_EDGE_ID, 0),) * 2)
            hierarchy_edges.extend(((motif_node, graph_node), (graph_node, motif_node)))
            hierarchy_attrs.extend(((MOTIF_GRAPH_EDGE_ID, 0),) * 2)
    else:
        for atom in range(num_atoms):
            hierarchy_edges.extend(((atom, graph_node), (graph_node, atom)))
            hierarchy_attrs.extend(((MOTIF_GRAPH_EDGE_ID, 0),) * 2)

    all_edges = chemical_edges + hierarchy_edges
    all_attrs = chemical_attrs + hierarchy_attrs
    transformed = HiMolData(**data.clone()._store)
    transformed.himol_node_attr = torch.tensor(node_rows, dtype=torch.long)
    transformed.himol_edge_index = _edge_tensor(all_edges)
    transformed.himol_edge_attr = _row_tensor(all_attrs, width=2)
    transformed.himol_batch = torch.zeros(len(node_rows), dtype=torch.long)
    transformed.himol_atom_node_index = torch.arange(num_atoms, dtype=torch.long)
    transformed.himol_graph_node_index = torch.tensor([graph_node], dtype=torch.long)
    transformed.himol_atom_target = torch.tensor(
        [row[0] for row in atom_rows], dtype=torch.long
    )
    transformed.himol_bond_index = _edge_tensor(physical_bonds)
    transformed.himol_bond_target = torch.tensor(physical_bond_types, dtype=torch.long)
    transformed.himol_num_atoms = torch.tensor([num_atoms], dtype=torch.long)
    transformed.himol_num_bonds = torch.tensor([len(physical_bonds)], dtype=torch.long)
    return transformed


def _edge_tensor(edges: Iterable[tuple[int, int]]) -> Tensor:
    rows = list(edges)
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long).t().contiguous()


def _row_tensor(rows: Iterable[tuple[int, int]], *, width: int) -> Tensor:
    values = list(rows)
    if not values:
        return torch.empty((0, width), dtype=torch.long)
    return torch.tensor(values, dtype=torch.long)


def _sample_name(data: MolecularData) -> str:
    sample_id = getattr(data, "sample_id", None)
    if isinstance(sample_id, Tensor) and sample_id.numel() == 1:
        return str(int(sample_id.item()))
    return "<unknown>"


__all__ = ["HiMolData", "add_himol_inputs", "decompose_himol_motifs"]

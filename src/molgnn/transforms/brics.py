"""BRICS-pruned molecular graph view required by HiGNN."""

from __future__ import annotations

from collections import Counter

import torch
from rdkit import Chem
from rdkit.Chem import BRICS
from torch import Tensor

from ..data import MolecularData
from .base import TransformError


def add_brics_fragments(data: MolecularData) -> MolecularData:
    """Clone a canonical graph and attach its BRICS fragment view."""

    smiles = getattr(data, "smiles", None)
    if not isinstance(smiles, str) or not smiles:
        raise TransformError(f"sample {_sample_id(data)} is missing source SMILES metadata")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TransformError(f"sample {_sample_id(data)} has invalid source SMILES")

    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    if not isinstance(x, Tensor) or x.ndim != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid x")
    if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid edge_index")
    if edge_index.dtype != torch.long:
        raise TransformError(f"sample {_sample_id(data)} edge_index must have dtype torch.long")
    if not isinstance(edge_attr, Tensor) or edge_attr.ndim != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid edge_attr")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise TransformError(f"sample {_sample_id(data)} has mismatched edge feature count")
    if mol.GetNumAtoms() != x.shape[0]:
        raise TransformError(f"sample {_sample_id(data)} source SMILES atom count does not match x")

    expected_edges = [
        direction
        for bond in mol.GetBonds()
        for direction in (
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            (bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()),
        )
    ]
    actual_edges = [tuple(pair) for pair in edge_index.t().tolist()]
    if Counter(actual_edges) != Counter(expected_edges):
        raise TransformError(
            f"sample {_sample_id(data)} edge_index does not match source SMILES connectivity"
        )

    edge_positions = {edge: position for position, edge in enumerate(actual_edges)}
    for source, target in expected_edges[::2]:
        forward = edge_positions[(source, target)]
        reverse = edge_positions[(target, source)]
        if not torch.equal(edge_attr[forward], edge_attr[reverse]):
            raise TransformError(
                f"sample {_sample_id(data)} has mismatched features for edge {source}<->{target}"
            )

    cut_bonds = {
        frozenset((source, target)) for (source, target), _labels in BRICS.FindBRICSBonds(mol)
    }
    keep = torch.tensor(
        [frozenset(edge) not in cut_bonds for edge in actual_edges],
        dtype=torch.bool,
        device=edge_index.device,
    )
    atom_to_fragment = _connected_components(mol, cut_bonds, device=x.device)

    transformed = data.clone()
    transformed.brics_edge_index = edge_index[:, keep].clone()
    transformed.brics_edge_attr = edge_attr[keep.to(edge_attr.device)].clone()
    transformed.atom_to_fragment = atom_to_fragment
    return transformed


def _connected_components(
    mol: Chem.Mol,
    cut_bonds: set[frozenset[int]],
    *,
    device: torch.device,
) -> Tensor:
    adjacency: list[list[int]] = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        if frozenset((source, target)) in cut_bonds:
            continue
        adjacency[source].append(target)
        adjacency[target].append(source)

    assignments = [-1] * mol.GetNumAtoms()
    fragment = 0
    for root in range(mol.GetNumAtoms()):
        if assignments[root] >= 0:
            continue
        assignments[root] = fragment
        stack = [root]
        while stack:
            atom = stack.pop()
            for neighbor in adjacency[atom]:
                if assignments[neighbor] < 0:
                    assignments[neighbor] = fragment
                    stack.append(neighbor)
        fragment += 1
    return torch.tensor(assignments, dtype=torch.long, device=device)


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_brics_fragments"]

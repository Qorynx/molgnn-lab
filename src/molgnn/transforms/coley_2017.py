"""Paper-specific attributes for Coley et al.'s 2017 graph embedding."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import torch
from rdkit import Chem
from rdkit.Chem import EState, rdMolDescriptors, rdPartialCharges
from torch import Tensor

from ..data import MolecularData
from .base import TransformError

COLEY_2017_ATOM_DIM = 32
COLEY_2017_BOND_DIM = 8

_ATOMIC_NUMBERS = (5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999)
_HEAVY_NEIGHBORS = (0, 1, 2, 3, 4, 5)
_HYDROGEN_COUNTS = (0, 1, 2, 3, 4)
_BOND_ORDERS = (1.0, 1.5, 2.0, 3.0)


def add_coley_2017_features(data: MolecularData) -> MolecularData:
    """Return a clone carrying the exact 32/8 paper-specific feature tensors."""

    smiles = getattr(data, "smiles", None)
    if not isinstance(smiles, str) or not smiles:
        raise TransformError(f"sample {_sample_id(data)} is missing source SMILES metadata")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TransformError(f"sample {_sample_id(data)} has invalid source SMILES")

    canonical_x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    if not isinstance(canonical_x, Tensor) or canonical_x.ndim != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid x")
    if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise TransformError(f"sample {_sample_id(data)} has invalid edge_index")
    if edge_index.dtype != torch.long:
        raise TransformError(f"sample {_sample_id(data)} edge_index must have dtype torch.long")
    if mol.GetNumAtoms() != canonical_x.shape[0]:
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

    atom_tensor = _atom_feature_tensor(mol, device=canonical_x.device)
    edge_rows: list[list[float]] = []
    for source, target in actual_edges:
        bond = mol.GetBondBetweenAtoms(source, target)
        if bond is None:
            raise TransformError(
                f"sample {_sample_id(data)} has no source bond for edge {source}->{target}"
            )
        edge_rows.append(_bond_features(bond))
    edge_tensor = (
        torch.tensor(edge_rows, dtype=torch.float32, device=edge_index.device)
        if edge_rows
        else torch.empty((0, COLEY_2017_BOND_DIM), dtype=torch.float32, device=edge_index.device)
    )

    if atom_tensor.shape != (mol.GetNumAtoms(), COLEY_2017_ATOM_DIM):
        raise AssertionError("Coley atom feature dimension mismatch")
    if edge_tensor.shape != (edge_index.shape[1], COLEY_2017_BOND_DIM):
        raise AssertionError("Coley bond feature dimension mismatch")
    if not torch.isfinite(atom_tensor).all() or not torch.isfinite(edge_tensor).all():
        raise TransformError(f"sample {_sample_id(data)} produced non-finite Coley features")

    transformed = data.clone()
    transformed.mge_x = atom_tensor
    transformed.mge_edge_attr = edge_tensor
    return transformed


def _atom_feature_tensor(mol: Chem.Mol, *, device: torch.device) -> Tensor:
    crippen = list(rdMolDescriptors._CalcCrippenContribs(mol))
    tpsa = list(rdMolDescriptors._CalcTPSAContribs(mol))
    labute_asa = list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])
    estate = list(EState.EStateIndices(mol))
    rdPartialCharges.ComputeGasteigerCharges(mol)

    rows: list[list[float]] = []
    for index, atom in enumerate(mol.GetAtoms()):
        charge = _finite_charge(atom.GetProp("_GasteigerCharge"))
        hydrogen_charge = _finite_charge(atom.GetProp("_GasteigerHCharge"))
        rows.append(
            [
                *_one_hot_last(atom.GetAtomicNum(), _ATOMIC_NUMBERS),
                *_one_hot_last(len(atom.GetNeighbors()), _HEAVY_NEIGHBORS),
                *_one_hot_last(atom.GetTotalNumHs(), _HYDROGEN_COUNTS),
                float(atom.GetFormalCharge()),
                float(atom.IsInRing()),
                float(atom.GetIsAromatic()),
                float(crippen[index][0]),
                float(crippen[index][1]),
                float(tpsa[index]),
                float(labute_asa[index]),
                float(estate[index]),
                charge,
                hydrogen_charge,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _bond_features(bond: Chem.Bond) -> list[float]:
    return [
        *_one_hot_last(bond.GetBondTypeAsDouble(), _BOND_ORDERS),
        float(bond.GetIsAromatic()),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        1.0,
    ]


def _one_hot_last(value: int | float, choices: Sequence[int | float]) -> list[float]:
    index = choices.index(value) if value in choices else len(choices) - 1
    return [float(position == index) for position in range(len(choices))]


def _finite_charge(value: str) -> float:
    charge = float(value)
    return charge if math.isfinite(charge) else 0.0


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = [
    "COLEY_2017_ATOM_DIM",
    "COLEY_2017_BOND_DIM",
    "add_coley_2017_features",
]

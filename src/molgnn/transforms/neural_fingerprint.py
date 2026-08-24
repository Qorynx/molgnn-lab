"""Model-specific feature transform for Neural Fingerprint (Duvenaud et al. 2015)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import torch
from rdkit import Chem
from torch import Tensor

from ..data import MolecularData
from .base import TransformError

NEURAL_FP_ATOM_DIM = 62
NEURAL_FP_BOND_DIM = 6

_ELEMENTS: tuple[str, ...] = (
    "C",
    "N",
    "O",
    "S",
    "F",
    "Si",
    "P",
    "Cl",
    "Br",
    "Mg",
    "Na",
    "Ca",
    "Fe",
    "As",
    "Al",
    "I",
    "B",
    "V",
    "K",
    "Tl",
    "Yb",
    "Sb",
    "Sn",
    "Ag",
    "Pd",
    "Co",
    "Se",
    "Ti",
    "Zn",
    "H",
    "Li",
    "Ge",
    "Cu",
    "Au",
    "Ni",
    "Cd",
    "In",
    "Mn",
    "Zr",
    "Cr",
    "Pt",
    "Hg",
    "Pb",
    "Unknown",
)
_DEGREES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
_ATTACHED_HYDROGENS: tuple[int, ...] = (0, 1, 2, 3, 4)
_IMPLICIT_VALENCES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)


def add_neural_fingerprint_inputs(data: MolecularData) -> MolecularData:
    """Derive 62-D atom and 6-D bond features for Neural Fingerprint.

    Paper and official repository (HIPS/neural-fingerprint) use implicit-hydrogen
    2D topology with exact degree-specific one-hot encodings.
    """
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
        else torch.empty((0, NEURAL_FP_BOND_DIM), dtype=torch.float32, device=edge_index.device)
    )

    if atom_tensor.shape != (mol.GetNumAtoms(), NEURAL_FP_ATOM_DIM):
        raise AssertionError("Neural Fingerprint atom feature dimension mismatch")
    if edge_tensor.shape != (edge_index.shape[1], NEURAL_FP_BOND_DIM):
        raise AssertionError("Neural Fingerprint bond feature dimension mismatch")
    if not torch.isfinite(atom_tensor).all() or not torch.isfinite(edge_tensor).all():
        raise TransformError(f"sample {_sample_id(data)} produced non-finite features")

    transformed = data.clone()
    transformed.neural_fp_x = atom_tensor
    transformed.neural_fp_edge_attr = edge_tensor
    return transformed


def _atom_feature_tensor(mol: Chem.Mol, *, device: torch.device) -> Tensor:
    rows: list[list[float]] = []
    for atom in mol.GetAtoms():
        rows.append(
            [
                *_one_of_k_encoding_unk(atom.GetSymbol(), _ELEMENTS),
                *_one_of_k_encoding(atom.GetDegree(), _DEGREES),
                *_one_of_k_encoding_unk(atom.GetTotalNumHs(), _ATTACHED_HYDROGENS),
                *_one_of_k_encoding_unk(atom.GetImplicitValence(), _IMPLICIT_VALENCES),
                float(atom.GetIsAromatic()),
            ]
        )
    return torch.tensor(rows, dtype=torch.float32, device=device)


def _bond_features(bond: Chem.Bond) -> list[float]:
    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.rdchem.BondType.SINGLE),
        float(bond_type == Chem.rdchem.BondType.DOUBLE),
        float(bond_type == Chem.rdchem.BondType.TRIPLE),
        float(bond_type == Chem.rdchem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def _one_of_k_encoding(value: int | str, allowable_set: Sequence[int | str]) -> list[float]:
    """Exact one-hot encoding; if not found, returns all-zero."""
    return [float(value == item) for item in allowable_set]


def _one_of_k_encoding_unk(value: int | str, allowable_set: Sequence[int | str]) -> list[float]:
    """Maps inputs not in the allowable set to the last element."""
    target = value if value in allowable_set else allowable_set[-1]
    return [float(target == item) for item in allowable_set]


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = [
    "NEURAL_FP_ATOM_DIM",
    "NEURAL_FP_BOND_DIM",
    "add_neural_fingerprint_inputs",
]

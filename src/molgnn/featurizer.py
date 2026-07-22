"""Canonical RDKit-to-PyG molecular featurization."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, BondType, ChiralType, HybridizationType
from torch import Tensor

from .data import MolecularData

_CategoryT = TypeVar("_CategoryT")


@dataclass(frozen=True)
class FeatureSchema:
    """Versioned dimensions of the canonical atom and bond feature layout."""

    version: str
    atom_dim: int
    bond_dim: int


# All vocabularies contain known values only. ``one_hot_with_unknown`` adds
# one final bucket for values that are outside a vocabulary.
ATOMIC_NUMBER_VOCAB: tuple[int, ...] = tuple(range(1, 119))
DEGREE_VOCAB: tuple[int, ...] = tuple(range(6))
FORMAL_CHARGE_VOCAB: tuple[int, ...] = (-2, -1, 0, 1, 2)
HYBRIDIZATION_VOCAB: tuple[HybridizationType, ...] = (
    HybridizationType.S,
    HybridizationType.SP,
    HybridizationType.SP2,
    HybridizationType.SP2D,
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
)
TOTAL_HYDROGEN_VOCAB: tuple[int, ...] = tuple(range(5))
CHIRAL_TAG_VOCAB: tuple[ChiralType, ...] = (
    ChiralType.CHI_UNSPECIFIED,
    ChiralType.CHI_TETRAHEDRAL_CW,
    ChiralType.CHI_TETRAHEDRAL_CCW,
    ChiralType.CHI_OTHER,
)
BOND_TYPE_VOCAB: tuple[BondType, ...] = (
    BondType.SINGLE,
    BondType.DOUBLE,
    BondType.TRIPLE,
    BondType.AROMATIC,
)
BOND_STEREO_VOCAB: tuple[BondStereo, ...] = (
    BondStereo.STEREONONE,
    BondStereo.STEREOANY,
    BondStereo.STEREOZ,
    BondStereo.STEREOE,
    BondStereo.STEREOCIS,
    BondStereo.STEREOTRANS,
)

ATOM_FEATURE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("atomic_number", len(ATOMIC_NUMBER_VOCAB) + 1),
    ("degree", len(DEGREE_VOCAB) + 1),
    ("formal_charge", len(FORMAL_CHARGE_VOCAB) + 1),
    ("hybridization", len(HYBRIDIZATION_VOCAB) + 1),
    ("aromatic", 1),
    ("in_ring", 1),
    ("total_hydrogens", len(TOTAL_HYDROGEN_VOCAB) + 1),
    ("chiral_tag", len(CHIRAL_TAG_VOCAB) + 1),
)
BOND_FEATURE_BLOCKS: tuple[tuple[str, int], ...] = (
    ("bond_type", len(BOND_TYPE_VOCAB) + 1),
    ("conjugated", 1),
    ("in_ring", 1),
    ("stereo", len(BOND_STEREO_VOCAB) + 1),
)
ATOM_FEATURE_DIM = sum(width for _, width in ATOM_FEATURE_BLOCKS)
BOND_FEATURE_DIM = sum(width for _, width in BOND_FEATURE_BLOCKS)

CANONICAL_FEATURE_SCHEMA_V1 = FeatureSchema(
    version="canonical_2d_v1",
    atom_dim=ATOM_FEATURE_DIM,
    bond_dim=BOND_FEATURE_DIM,
)


def one_hot_with_unknown(value: _CategoryT, vocabulary: Sequence[_CategoryT]) -> tuple[float, ...]:
    """Encode ``value`` with a deterministic final bucket for unknown values."""

    index = vocabulary.index(value) if value in vocabulary else len(vocabulary)
    return tuple(float(position == index) for position in range(len(vocabulary) + 1))


def atom_features(atom: Chem.Atom) -> Tensor:
    """Return one canonical float32 feature vector for one RDKit atom."""

    values: list[float] = [
        *one_hot_with_unknown(atom.GetAtomicNum(), ATOMIC_NUMBER_VOCAB),
        *one_hot_with_unknown(atom.GetDegree(), DEGREE_VOCAB),
        *one_hot_with_unknown(atom.GetFormalCharge(), FORMAL_CHARGE_VOCAB),
        *one_hot_with_unknown(atom.GetHybridization(), HYBRIDIZATION_VOCAB),
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        *one_hot_with_unknown(atom.GetTotalNumHs(), TOTAL_HYDROGEN_VOCAB),
        *one_hot_with_unknown(atom.GetChiralTag(), CHIRAL_TAG_VOCAB),
    ]
    features = torch.tensor(values, dtype=torch.float32)
    if features.shape != (CANONICAL_FEATURE_SCHEMA_V1.atom_dim,):
        raise AssertionError("atom feature dimension does not match the canonical schema")
    return features


def bond_features(bond: Chem.Bond) -> Tensor:
    """Return one canonical float32 feature vector for one RDKit bond."""

    values: list[float] = [
        *one_hot_with_unknown(bond.GetBondType(), BOND_TYPE_VOCAB),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        *one_hot_with_unknown(bond.GetStereo(), BOND_STEREO_VOCAB),
    ]
    features = torch.tensor(values, dtype=torch.float32)
    if features.shape != (CANONICAL_FEATURE_SCHEMA_V1.bond_dim,):
        raise AssertionError("bond feature dimension does not match the canonical schema")
    return features


def _target_tensor(values: Sequence[float] | Tensor, *, name: str) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[0] != 1:
        raise ValueError(f"{name} must have shape [T] or [1, T]")
    return tensor.contiguous()


def _mask_tensor(values: Sequence[bool] | Tensor, *, num_targets: int) -> Tensor:
    tensor = torch.as_tensor(values, dtype=torch.bool)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape != (1, num_targets):
        raise ValueError("target_mask must have shape [T] or [1, T]")
    return tensor.contiguous()


def featurize_mol(
    mol: Chem.Mol,
    *,
    targets: Sequence[float] | Tensor,
    target_mask: Sequence[bool] | Tensor,
    sample_id: int,
    smiles: str | None = None,
) -> MolecularData:
    """Convert a sanitized RDKit molecule into one canonical PyG sample."""

    if mol is None:
        raise ValueError("mol must be a valid RDKit molecule")

    atom_vectors = [atom_features(atom) for atom in mol.GetAtoms()]
    if atom_vectors:
        x = torch.stack(atom_vectors, dim=0)
    else:
        x = torch.empty((0, CANONICAL_FEATURE_SCHEMA_V1.atom_dim), dtype=torch.float32)

    edge_pairs: list[tuple[int, int]] = []
    edge_vectors: list[Tensor] = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        edge_feature = bond_features(bond)
        edge_pairs.extend(((begin, end), (end, begin)))
        edge_vectors.extend((edge_feature, edge_feature.clone()))

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_vectors, dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, CANONICAL_FEATURE_SCHEMA_V1.bond_dim), dtype=torch.float32)

    y = _target_tensor(targets, name="targets")
    y_mask = _mask_tensor(target_mask, num_targets=y.shape[1])
    if y_mask.any() and not torch.isfinite(y[y_mask]).all():
        raise ValueError("observed targets must be finite")
    y = torch.where(y_mask, y, torch.zeros_like(y))

    if x.shape[1] != CANONICAL_FEATURE_SCHEMA_V1.atom_dim:
        raise AssertionError("atom matrix dimension does not match the canonical schema")
    if edge_attr.shape[1] != CANONICAL_FEATURE_SCHEMA_V1.bond_dim:
        raise AssertionError("bond matrix dimension does not match the canonical schema")

    data = MolecularData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        y_mask=y_mask,
        sample_id=torch.tensor([sample_id], dtype=torch.long),
    )
    if smiles is not None:
        data.smiles = smiles
    return data


def featurize_smiles(
    smiles: str,
    *,
    targets: Sequence[float] | Tensor,
    target_mask: Sequence[bool] | Tensor,
    sample_id: int,
) -> MolecularData:
    """Parse a SMILES with RDKit's default sanitization, then featurize it."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    return featurize_mol(
        mol,
        targets=targets,
        target_mask=target_mask,
        sample_id=sample_id,
        smiles=smiles,
    )


__all__ = [
    "ATOMIC_NUMBER_VOCAB",
    "ATOM_FEATURE_BLOCKS",
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_BLOCKS",
    "BOND_FEATURE_DIM",
    "BOND_STEREO_VOCAB",
    "BOND_TYPE_VOCAB",
    "CANONICAL_FEATURE_SCHEMA_V1",
    "CHIRAL_TAG_VOCAB",
    "DEGREE_VOCAB",
    "FORMAL_CHARGE_VOCAB",
    "HYBRIDIZATION_VOCAB",
    "TOTAL_HYDROGEN_VOCAB",
    "FeatureSchema",
    "atom_features",
    "bond_features",
    "featurize_mol",
    "featurize_smiles",
    "one_hot_with_unknown",
]

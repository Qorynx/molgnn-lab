"""Pinned legacy feature tables and geometry constants for ChemRL-GEM."""

from __future__ import annotations

import math

import torch
from rdkit.Chem import rdchem

ATOM_FEATURE_NAMES = (
    "atomic_num",
    "formal_charge",
    "degree",
    "chiral_tag",
    "total_numHs",
    "is_aromatic",
    "hybridization",
)
BOND_FEATURE_NAMES = ("bond_dir", "bond_type", "is_in_ring")

# These are the legacy CompoundKit cardinalities before its +5 embedding
# padding.  They are intentionally not inferred from the installed RDKit.
ATOM_VOCAB_SIZES = (119, 17, 12, 4, 10, 2, 8)
BOND_VOCAB_SIZES = (7, 22, 2)
ATOM_EMBED_SIZES = tuple(size + 5 for size in ATOM_VOCAB_SIZES)
BOND_EMBED_SIZES = tuple(size + 5 for size in BOND_VOCAB_SIZES)

DEFAULT_EMBED_DIM = 32
DEFAULT_LAYER_NUM = 8
DEFAULT_DROPOUT = 0.5
DEFAULT_ATOMIC_DISTANCE_BINS = 30
ATOMIC_DISTANCE_CUTOFF = 20.0
ATOM_MASK_RATIO = 0.15
FINGERPRINT_SIZE = 494
CONTEXT_VOCAB_SIZE = 2400

BOND_LENGTH_CENTERS = torch.arange(0.0, 2.0, 0.1, dtype=torch.float32)
BOND_ANGLE_CENTERS = torch.arange(0.0, math.pi, 0.1, dtype=torch.float32)
RBF_GAMMA = 10.0

# Explicit maps preserve the old RDKit enum ordering used by CompoundKit.
_CHIRAL = {
    rdchem.ChiralType.CHI_UNSPECIFIED: 0,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
    rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
    rdchem.ChiralType.CHI_OTHER: 3,
}
_HYBRIDIZATION = {
    rdchem.HybridizationType.UNSPECIFIED: 0,
    rdchem.HybridizationType.S: 1,
    rdchem.HybridizationType.SP: 2,
    rdchem.HybridizationType.SP2: 3,
    rdchem.HybridizationType.SP3: 4,
    rdchem.HybridizationType.SP2D: 5,
    rdchem.HybridizationType.SP3D: 6,
    rdchem.HybridizationType.SP3D2: 7,
}
_BOND_DIR = {
    rdchem.BondDir.NONE: 0,
    rdchem.BondDir.BEGINWEDGE: 1,
    rdchem.BondDir.BEGINDASH: 2,
    rdchem.BondDir.ENDDOWNRIGHT: 3,
    rdchem.BondDir.ENDUPRIGHT: 4,
    rdchem.BondDir.EITHERDOUBLE: 5,
    rdchem.BondDir.UNKNOWN: 6,
}
_BOND_TYPE = {
    rdchem.BondType.UNSPECIFIED: 0,
    rdchem.BondType.SINGLE: 1,
    rdchem.BondType.DOUBLE: 2,
    rdchem.BondType.TRIPLE: 3,
    rdchem.BondType.QUADRUPLE: 4,
    rdchem.BondType.QUINTUPLE: 5,
    rdchem.BondType.HEXTUPLE: 6,
    rdchem.BondType.ONEANDAHALF: 7,
    rdchem.BondType.TWOANDAHALF: 8,
    rdchem.BondType.THREEANDAHALF: 9,
    rdchem.BondType.FOURANDAHALF: 10,
    rdchem.BondType.FIVEANDAHALF: 11,
    rdchem.BondType.AROMATIC: 12,
    rdchem.BondType.IONIC: 13,
    rdchem.BondType.HYDROGEN: 14,
    rdchem.BondType.THREECENTER: 15,
    rdchem.BondType.DATIVEONE: 16,
    rdchem.BondType.DATIVE: 17,
    rdchem.BondType.DATIVEL: 18,
    rdchem.BondType.DATIVER: 19,
    rdchem.BondType.OTHER: 20,
    rdchem.BondType.ZERO: 21,
}
_BOND_STEREO = {
    rdchem.BondStereo.STEREONONE: 0,
    rdchem.BondStereo.STEREOANY: 1,
    rdchem.BondStereo.STEREOZ: 2,
    rdchem.BondStereo.STEREOE: 3,
    rdchem.BondStereo.STEREOCIS: 4,
    rdchem.BondStereo.STEREOTRANS: 5,
}


def legacy_category_id(value: object, *, name: str) -> int:
    """Return CompoundKit's ``safe_index(value) + 1`` ID."""

    if name == "atomic_num":
        index = int(value) - 1 if isinstance(value, int) and 1 <= value <= 118 else 118
    elif name == "formal_charge":
        index = int(value) + 5 if isinstance(value, int) and -5 <= value <= 11 else 16
    elif name == "degree":
        index = int(value) if isinstance(value, int) and 0 <= value <= 10 else 11
    elif name == "chiral_tag":
        index = _CHIRAL.get(value, 3)
    elif name == "total_numHs":
        index = int(value) if isinstance(value, int) and 0 <= value <= 8 else 9
    elif name == "is_aromatic":
        index = int(value) if int(value) in (0, 1) else 1
    elif name == "hybridization":
        index = _HYBRIDIZATION.get(value, 7)
    elif name == "bond_dir":
        index = _BOND_DIR.get(value, 6)
    elif name == "bond_type":
        index = _BOND_TYPE.get(value, 21)
    elif name == "is_in_ring":
        index = int(value) if int(value) in (0, 1) else 1
    else:
        raise KeyError(name)
    return index + 1


def self_loop_id(name: str) -> int:
    """Return the legacy self-loop ID (feature-size + 2)."""

    try:
        return BOND_VOCAB_SIZES[BOND_FEATURE_NAMES.index(name)] + 2
    except ValueError as exc:
        raise KeyError(name) from exc


__all__ = [
    "ATOM_EMBED_SIZES",
    "ATOM_FEATURE_NAMES",
    "ATOM_MASK_RATIO",
    "ATOM_VOCAB_SIZES",
    "ATOMIC_DISTANCE_CUTOFF",
    "BOND_ANGLE_CENTERS",
    "BOND_EMBED_SIZES",
    "BOND_FEATURE_NAMES",
    "BOND_LENGTH_CENTERS",
    "BOND_VOCAB_SIZES",
    "CONTEXT_VOCAB_SIZE",
    "DEFAULT_ATOMIC_DISTANCE_BINS",
    "DEFAULT_DROPOUT",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_LAYER_NUM",
    "FINGERPRINT_SIZE",
    "RBF_GAMMA",
    "legacy_category_id",
    "self_loop_id",
]


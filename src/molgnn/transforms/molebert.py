"""Discrete 2-D inputs used by Mole-BERT.

The shared featurizer intentionally exposes continuous one-hot blocks.  The
Mole-BERT encoder is checkpoint-compatible with the author implementation and
therefore receives its own small integer view derived here.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondDir, BondType
from torch import Tensor

from ..data import MolecularData
from .base import TransformError

_ATOM_BLOCKS = (119, 7, 6, 8, 1, 1, 6, 5)
_CHIRALITY_OFFSET = sum(_ATOM_BLOCKS[:-1])
_BOND_TYPE_VALUES = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC)
_BOND_DIR_VALUES = (BondDir.NONE, BondDir.ENDUPRIGHT, BondDir.ENDDOWNRIGHT)


def _argmax_block(values: Tensor, start: int, width: int, name: str) -> Tensor:
    block = values[:, start : start + width]
    if block.ndim != 2 or block.shape[1] != width:
        raise TransformError(f"canonical {name} block is missing or has the wrong width")
    indices = block.argmax(dim=1)
    if not torch.allclose(block.sum(dim=1), torch.ones(block.shape[0], device=block.device)):
        raise TransformError(f"canonical {name} block is not one-hot")
    return indices.to(torch.long)


def _canonical_atom_attributes(data: MolecularData) -> Tensor:
    if not isinstance(data.x, Tensor) or data.x.ndim != 2 or data.x.shape[1] < sum(_ATOM_BLOCKS):
        raise TransformError("Mole-BERT requires the canonical atom feature schema")
    atomic_index = _argmax_block(data.x, 0, _ATOM_BLOCKS[0], "atomic_number")
    chirality = _argmax_block(data.x, _CHIRALITY_OFFSET, _ATOM_BLOCKS[-1], "chirality")
    # The final canonical unknown bucket has no paper-defined counterpart.
    # Treating it as unspecified is deterministic and keeps the 4-tag contract.
    chirality = torch.where(chirality == 4, torch.zeros_like(chirality), chirality)
    if torch.any(atomic_index >= 118):
        raise TransformError("Mole-BERT only supports atomic numbers 1..118")
    return torch.stack((atomic_index, chirality), dim=1)


def _canonical_bond_types(data: MolecularData) -> Tensor:
    if not isinstance(data.edge_attr, Tensor) or data.edge_attr.ndim != 2 or data.edge_attr.shape[1] < 5:
        raise TransformError("Mole-BERT requires the canonical bond feature schema")
    return _argmax_block(data.edge_attr, 0, 5, "bond_type")


def _molecule_from_data(data: MolecularData, atom_attr: Tensor, bond_type: Tensor) -> Chem.Mol:
    """Reconstruct a light RDKit graph for SMILES atom-order alignment."""

    mol = Chem.RWMol()
    for atomic_index in atom_attr[:, 0].tolist():
        mol.AddAtom(Chem.Atom(int(atomic_index) + 1))
    seen: set[tuple[int, int]] = set()
    bond_types = _BOND_TYPE_VALUES + (BondType.UNSPECIFIED,)
    for edge_id, (source, target) in enumerate(data.edge_index.t().tolist()):
        pair = (min(source, target), max(source, target))
        if pair in seen:
            continue
        seen.add(pair)
        if bond_type[edge_id].item() >= len(bond_types) - 1:
            raise TransformError("unknown canonical bond type")
        mol.AddBond(pair[0], pair[1], bond_types[int(bond_type[edge_id])])
    return mol.GetMol()


def _parsed_direction_map(smiles: str, data: MolecularData, atom_attr: Tensor, bond_type: Tensor) -> dict[tuple[int, int], int]:
    """Return direction IDs in canonical atom order, including explicit-H QM9 graphs."""

    parsed = Chem.MolFromSmiles(smiles)
    candidates = [parsed]
    if parsed is not None:
        candidates.append(Chem.AddHs(parsed))
    if parsed is not None and parsed.GetNumAtoms() == atom_attr.shape[0]:
        parsed_numbers = [atom.GetAtomicNum() - 1 for atom in parsed.GetAtoms()]
        if parsed_numbers == atom_attr[:, 0].tolist():
            pair_types: dict[tuple[int, int], int] = {}
            for edge_id, (source, target) in enumerate(data.edge_index.t().tolist()):
                pair_types.setdefault((min(source, target), max(source, target)), int(bond_type[edge_id]))
            result: dict[tuple[int, int], int] = {}
            for bond in parsed.GetBonds():
                pair = (min(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()), max(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                if pair not in pair_types or bond.GetBondType() not in _BOND_TYPE_VALUES or _BOND_TYPE_VALUES.index(bond.GetBondType()) != pair_types[pair]:
                    break
                direction = _BOND_DIR_VALUES.index(bond.GetBondDir()) if bond.GetBondDir() in _BOND_DIR_VALUES else None
                if direction is None:
                    raise TransformError(f"unsupported RDKit bond direction {bond.GetBondDir()!r}")
                result[pair] = direction
            else:
                return result
    canonical = _molecule_from_data(data, atom_attr, bond_type)
    for query in candidates:
        if query is None or query.GetNumAtoms() != canonical.GetNumAtoms():
            continue
        match = canonical.GetSubstructMatch(query)
        if not match:
            continue
        result: dict[tuple[int, int], int] = {}
        for bond in query.GetBonds():
            u = match[bond.GetBeginAtomIdx()]
            v = match[bond.GetEndAtomIdx()]
            direction = _BOND_DIR_VALUES.index(bond.GetBondDir()) if bond.GetBondDir() in _BOND_DIR_VALUES else None
            if direction is None:
                raise TransformError(f"unsupported RDKit bond direction {bond.GetBondDir()!r}")
            result[(min(u, v), max(u, v))] = direction
        return result
    # QM9 molecules generally have no directional SMILES bonds.  If the
    # supplied SMILES cannot represent the explicit-H graph, retain the safe
    # NONE value rather than inventing a stereo direction.
    return {}


def add_molebert_inputs(data: MolecularData) -> MolecularData:
    """Attach Mole-BERT's integer atom and bond attributes to one sample."""

    if not isinstance(data, MolecularData):
        raise TransformError("Mole-BERT transform requires MolecularData")
    atom_attr = _canonical_atom_attributes(data)
    bond_type = _canonical_bond_types(data)
    if data.edge_index.ndim != 2 or data.edge_index.shape[0] != 2:
        raise TransformError("data.edge_index must have shape [2, E]")
    if bond_type.shape[0] != data.edge_index.shape[1]:
        raise TransformError("canonical bond attributes and edge_index are misaligned")

    direction_by_pair: dict[tuple[int, int], int] = {}
    smiles = getattr(data, "smiles", None)
    if isinstance(smiles, str) and smiles:
        direction_by_pair = _parsed_direction_map(smiles, data, atom_attr, bond_type)
    directions = []
    for edge_id, (source, target) in enumerate(data.edge_index.t().tolist()):
        directions.append(direction_by_pair.get((min(source, target), max(source, target)), 0))
    bond_attr = torch.stack((bond_type, torch.tensor(directions, dtype=torch.long, device=bond_type.device)), dim=1)
    data.molebert_atom_attr = atom_attr.contiguous()
    data.molebert_bond_attr = bond_attr.contiguous()
    return data


__all__ = ["add_molebert_inputs"]

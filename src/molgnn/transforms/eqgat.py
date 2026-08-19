"""EQGAT's coordinate-derived radius graph and SMILES conformer proxy."""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor

from ..data import MolecularData
from ..models.eqgat_2022.constants import EQGAT_CUTOFF, EQGAT_MAX_NEIGHBORS
from .base import TransformError


def add_eqgat_inputs(data: MolecularData) -> MolecularData:
    """Attach EQGAT's capped directed radius graph to one unbatched sample.

    Native atomic numbers and coordinates are retained unchanged.  A sample
    with neither field may use a deterministic ETKDG/MMFF heavy-atom conformer
    derived from its source SMILES; this proxy is model-local and explicitly
    marked, never introduced by the shared 2-D featurizer.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; EQGAT inputs must be derived before batching"
        )

    atomic_number = getattr(data, "atomic_number", None)
    pos = getattr(data, "pos", None)
    if atomic_number is None and pos is None:
        atomic_number, pos = _smiles_proxy(data, sample=sample)
        is_proxy = True
    else:
        _validate_native_inputs(atomic_number, pos, sample=sample)
        assert isinstance(atomic_number, Tensor)
        assert isinstance(pos, Tensor)
        is_proxy = False

    transformed = data.clone()
    transformed.atomic_number = atomic_number
    transformed.pos = pos
    transformed.eqgat_edge_index = _radius_edge_index(pos)
    transformed.eqgat_geometry_is_proxy = torch.tensor(
        [is_proxy], dtype=torch.bool, device=pos.device
    )
    return transformed


def _smiles_proxy(data: MolecularData, *, sample: int | str) -> tuple[Tensor, Tensor]:
    smiles = getattr(data, "smiles", None)
    x = getattr(data, "x", None)
    if not isinstance(smiles, str) or not smiles:
        raise TransformError(
            f"sample {sample} requires atomic_number and pos, or source SMILES for a conformer proxy"
        )
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1:
        raise TransformError(f"sample {sample} has invalid x for an EQGAT conformer proxy")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise TransformError(f"sample {sample} has invalid source SMILES")
    if mol.GetNumAtoms() != x.shape[0]:
        raise TransformError(f"sample {sample} source SMILES atom count does not match x")
    atomic_number = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long,
        device=x.device,
    )
    return atomic_number, _embed_heavy_atom_conformer(mol, sample=sample, device=x.device)


def _embed_heavy_atom_conformer(
    mol: Chem.Mol, *, sample: int | str, device: torch.device
) -> Tensor:
    """Construct a deterministic proxy conformer without changing atom IDs."""

    if mol.GetNumAtoms() == 1:
        return torch.zeros((1, 3), dtype=torch.float32, device=device)
    mol_h = Chem.AddHs(mol)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 0x5C4E
    if AllChem.EmbedMolecule(mol_h, parameters) != 0:
        raise TransformError(f"sample {sample} could not embed an EQGAT conformer proxy")
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
    except RuntimeError:
        # An embedded geometry is still usable when a force field lacks a
        # parameter for an unusual atom or charge state.
        pass

    conformer = mol_h.GetConformer()
    positions = [
        (
            float(conformer.GetAtomPosition(index).x),
            float(conformer.GetAtomPosition(index).y),
            float(conformer.GetAtomPosition(index).z),
        )
        for index in range(mol.GetNumAtoms())
    ]
    pos = torch.tensor(positions, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(pos).all()):
        raise TransformError(f"sample {sample} conformer proxy contains non-finite coordinates")
    return pos


def _validate_native_inputs(
    atomic_number: object, pos: object, *, sample: int | str
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(f"sample {sample} requires atomic_number for native EQGAT geometry")
    if (
        atomic_number.ndim != 1
        or atomic_number.numel() < 1
        or atomic_number.dtype != torch.long
        or bool((atomic_number <= 0).any())
    ):
        raise TransformError(
            f"sample {sample} atomic_number must be a non-empty positive long tensor with shape [N]"
        )
    if not isinstance(pos, Tensor):
        raise TransformError(f"sample {sample} requires pos for native EQGAT geometry")
    if (
        pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the atomic_number device"
        )


def _radius_edge_index(pos: Tensor) -> Tensor:
    """Return nearest-first directed edges ``j -> i`` in EQGAT's radius.

    The official ATOM3D transform limits each target to 32 neighbors.  We
    compute distances in small target chunks so topology construction does not
    retain an ``N x N x 3`` displacement tensor, then use source-ID order as a
    deterministic tie break for equal distances.
    """

    node_count = pos.shape[0]
    source_ids = torch.arange(node_count, dtype=torch.long, device=pos.device)
    source_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    chunk_size = 256
    for start in range(0, node_count, chunk_size):
        stop = min(start + chunk_size, node_count)
        distances = torch.cdist(pos[start:stop], pos, p=2)
        for local_target in range(stop - start):
            target = start + local_target
            keep = (source_ids != target) & (distances[local_target] < EQGAT_CUTOFF)
            candidates = source_ids[keep]
            if candidates.numel() == 0:
                continue
            candidate_distances = distances[local_target, keep]
            order = torch.argsort(candidate_distances, stable=True)
            selected = candidates[order[:EQGAT_MAX_NEIGHBORS]]
            source_parts.append(selected)
            target_parts.append(torch.full_like(selected, target))
    if not source_parts:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    return torch.stack((torch.cat(source_parts), torch.cat(target_parts)), dim=0)


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_eqgat_inputs"]

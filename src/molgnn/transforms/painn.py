"""PaiNN's coordinate-derived radius graph and SMILES conformer proxy."""

from __future__ import annotations

import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch import Tensor

from ..data import MolecularData
from ..models.painn_2021.constants import PAINN_CUTOFF
from .base import TransformError


def add_painn_inputs(data: MolecularData) -> MolecularData:
    """Attach PaiNN's directed reciprocal radius graph to one sample.

    Native ``atomic_number`` and ``pos`` are preserved.  SMILES-only records
    receive a deterministic RDKit conformer proxy local to this transform.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; PaiNN inputs must be derived before batching"
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
    transformed.painn_edge_index = _radius_edge_index(pos)
    transformed.painn_geometry_is_proxy = torch.tensor(
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
        raise TransformError(f"sample {sample} has invalid x for a PaiNN conformer proxy")

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise TransformError(f"sample {sample} has invalid source SMILES")
    if molecule.GetNumAtoms() != x.shape[0]:
        raise TransformError(f"sample {sample} source SMILES atom count does not match x")
    atomic_number = torch.tensor(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        dtype=torch.long,
        device=x.device,
    )
    return atomic_number, _embed_heavy_atom_conformer(
        molecule, sample=sample, device=x.device
    )


def _embed_heavy_atom_conformer(
    molecule: Chem.Mol, *, sample: int | str, device: torch.device
) -> Tensor:
    if molecule.GetNumAtoms() == 1:
        return torch.zeros((1, 3), dtype=torch.float32, device=device)

    molecule_h = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = 0x5C4E
    if AllChem.EmbedMolecule(molecule_h, parameters) != 0:
        raise TransformError(f"sample {sample} could not embed a PaiNN conformer proxy")
    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule_h):
            AllChem.MMFFOptimizeMolecule(molecule_h, maxIters=200)
        else:
            AllChem.UFFOptimizeMolecule(molecule_h, maxIters=200)
    except RuntimeError:
        # Keep a valid embedded structure when a force field lacks parameters.
        pass

    conformer = molecule_h.GetConformer()
    positions = [
        (
            float(conformer.GetAtomPosition(index).x),
            float(conformer.GetAtomPosition(index).y),
            float(conformer.GetAtomPosition(index).z),
        )
        for index in range(molecule.GetNumAtoms())
    ]
    result = torch.tensor(positions, dtype=torch.float32, device=device)
    if not bool(torch.isfinite(result).all()):
        raise TransformError(f"sample {sample} conformer proxy contains non-finite coordinates")
    return result


def _validate_native_inputs(
    atomic_number: object, pos: object, *, sample: int | str
) -> None:
    if not isinstance(atomic_number, Tensor):
        raise TransformError(f"sample {sample} requires atomic_number for native PaiNN geometry")
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
        raise TransformError(f"sample {sample} requires pos for native PaiNN geometry")
    if (
        pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the atomic_number device"
        )
    if _has_coincident_distinct_atoms(pos):
        raise TransformError(f"sample {sample} contains coincident distinct atoms")


def _radius_edge_index(pos: Tensor) -> Tensor:
    """Return deterministic source-major directed edges within the cutoff."""

    node_count = pos.shape[0]
    node_ids = torch.arange(node_count, dtype=torch.long, device=pos.device)
    source_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    chunk_size = 256
    for start in range(0, node_count, chunk_size):
        stop = min(start + chunk_size, node_count)
        distances = torch.cdist(pos[start:stop], pos, p=2)
        for local_target in range(stop - start):
            target = start + local_target
            keep = (node_ids != target) & (distances[local_target] < PAINN_CUTOFF)
            sources = node_ids[keep]
            if sources.numel():
                source_parts.append(sources)
                target_parts.append(torch.full_like(sources, target))
    if not source_parts:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    source = torch.cat(source_parts)
    target = torch.cat(target_parts)
    # The chunk loop is target-major; normalize to source-major for reproducible
    # transforms across chunk sizes and to match the other radius transforms.
    order = torch.argsort(source * node_count + target, stable=True)
    return torch.stack((source[order], target[order]), dim=0)


def _has_coincident_distinct_atoms(pos: Tensor) -> bool:
    node_count = pos.shape[0]
    if node_count < 2:
        return False
    node_ids = torch.arange(node_count, dtype=torch.long, device=pos.device)
    chunk_size = 256
    for start in range(0, node_count, chunk_size):
        stop = min(start + chunk_size, node_count)
        distances = torch.cdist(pos[start:stop], pos, p=2)
        for local_target in range(stop - start):
            target = start + local_target
            if bool(((node_ids != target) & (distances[local_target] <= 1.0e-8)).any()):
                return True
    return False


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_painn_inputs"]

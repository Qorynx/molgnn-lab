"""ChemRL-GEM atom-bond and bond-angle graph construction."""

from __future__ import annotations

import torch
from rdkit import Chem
from torch import Tensor

from ..data import MolecularData
from ..models.chemrl_gem_2022.constants import (
    BOND_FEATURE_NAMES,
    self_loop_id,
)
from .base import TransformError, geometry_is_proxy, with_shared_geometry


def add_chemrl_gem_inputs(data: MolecularData) -> MolecularData:
    """Attach GEM's directed atom graph, line graph, lengths and angles.

    The transform preserves canonical ``edge_index`` and builds the source
    graph (including self-loops) in a namespaced field.  Geometry is supplied
    only by the shared provider; no conformer is generated here.
    """

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(f"sample {sample} is already batched; GEM inputs are per-sample")
    data = with_shared_geometry(data)
    atomic_number = getattr(data, "atomic_number", None)
    pos = getattr(data, "pos", None)
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    _validate_inputs(atomic_number, pos, x, edge_index, edge_attr, sample=sample)
    assert isinstance(atomic_number, Tensor)
    assert isinstance(pos, Tensor)
    assert isinstance(x, Tensor)
    assert isinstance(edge_index, Tensor)
    assert isinstance(edge_attr, Tensor)

    atom_attr = _atom_attributes(x, atomic_number)
    bond_attr = _bond_attributes(edge_attr)
    node_count = atomic_number.shape[0]
    loops = torch.arange(node_count, dtype=torch.long, device=edge_index.device)
    loop_index = torch.stack((loops, loops), dim=0)
    gem_edge_index = torch.cat((edge_index, loop_index), dim=1)
    loop_attr = torch.tensor(
        [[self_loop_id(name) for name in BOND_FEATURE_NAMES]] * node_count,
        dtype=torch.long,
        device=edge_index.device,
    )
    gem_bond_attr = torch.cat((bond_attr, loop_attr), dim=0)
    source, target = gem_edge_index
    gem_bond_length = torch.linalg.vector_norm(pos[target] - pos[source], dim=-1)
    angle_edge_index, angles = _bond_angle_graph(gem_edge_index, pos)

    transformed = data.clone()
    transformed.chemrl_gem_atom_attr = atom_attr
    transformed.chemrl_gem_edge_index = gem_edge_index
    transformed.chemrl_gem_bond_attr = gem_bond_attr
    transformed.chemrl_gem_bond_length = gem_bond_length.to(torch.float32)
    transformed.chemrl_gem_angle_edge_index = angle_edge_index
    transformed.chemrl_gem_bond_angle = angles
    transformed.chemrl_gem_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )
    return transformed


def _validate_inputs(
    atomic_number: object,
    pos: object,
    x: object,
    edge_index: object,
    edge_attr: object,
    *,
    sample: int | str,
) -> None:
    if not isinstance(atomic_number, Tensor) or atomic_number.ndim != 1 or atomic_number.numel() < 1 or atomic_number.dtype != torch.long:
        raise TransformError(f"sample {sample} atomic_number must be a non-empty long vector")
    if not isinstance(pos, Tensor) or pos.shape != (atomic_number.shape[0], 3) or pos.dtype != torch.float32 or not bool(torch.isfinite(pos).all()):
        raise TransformError(f"sample {sample} pos must have shape [N, 3] and finite float32 values")
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] != atomic_number.shape[0] or x.dtype != torch.float32 or x.shape[1] < 153:
        raise TransformError(f"sample {sample} requires the canonical 153-column atom features")
    if not isinstance(edge_index, Tensor) or edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
        raise TransformError(f"sample {sample} edge_index must have shape [2, E]")
    if not isinstance(edge_attr, Tensor) or edge_attr.shape != (edge_index.shape[1], 14) or edge_attr.dtype != torch.float32:
        raise TransformError(f"sample {sample} requires canonical 14-column bond features")
    if any(value.device != atomic_number.device for value in (pos, x, edge_index, edge_attr)):
        raise TransformError(f"sample {sample} GEM tensors must share one device")


def _argmax_block(values: Tensor, start: int, width: int) -> Tensor:
    return values[:, start : start + width].argmax(dim=1)


def _atom_attributes(x: Tensor, atomic_number: Tensor) -> Tensor:
    degree = _argmax_block(x, 119, 7)
    formal = _argmax_block(x, 126, 6)
    hybrid = _argmax_block(x, 132, 8)
    hydrogens = _argmax_block(x, 142, 6)
    chiral = _argmax_block(x, 148, 5)
    # Canonical formal-charge values are [-2, -1, 0, 1, 2, unknown], while
    # legacy CompoundKit starts at -5.  Preserve the source table index.
    formal_source = torch.where(formal < 5, formal + 3, torch.full_like(formal, 16))
    hybrid_source = torch.tensor((1, 2, 3, 5, 4, 6, 7, 7), dtype=torch.long, device=x.device)
    hybrid = hybrid_source[hybrid]
    chiral = chiral.clamp_max(3)
    atom = torch.stack(
        (
            torch.where(
                (atomic_number >= 1) & (atomic_number <= 118),
                atomic_number,
                torch.full_like(atomic_number, 119),
            ),
            formal_source + 1,
            torch.where(degree <= 5, degree, torch.full_like(degree, 11)) + 1,
            chiral + 1,
            torch.where(hydrogens < 5, hydrogens, torch.full_like(hydrogens, 9)) + 1,
            x[:, 140].round().clamp(0, 1).to(torch.long) + 1,
            hybrid + 1,
        ),
        dim=1,
    )
    return atom.to(torch.long).contiguous()


def _bond_attributes(edge_attr: Tensor) -> Tensor:
    bond_type = _argmax_block(edge_attr, 0, 5)
    source_type = torch.tensor((1, 2, 3, 12, 21), dtype=torch.long, device=edge_attr.device)
    bond_type = source_type[bond_type] + 1
    bond_dir = torch.ones_like(bond_type)
    in_ring = edge_attr[:, 6].round().clamp(0, 1).to(torch.long) + 1
    return torch.stack((bond_dir, bond_type, in_ring), dim=1).contiguous()


def _bond_angle_graph(edge_index: Tensor, pos: Tensor) -> tuple[Tensor, Tensor]:
    """Build source ``HT`` line graph and its directed-vector angles."""

    source, target = edge_index
    line_sources: list[int] = []
    line_targets: list[int] = []
    angles: list[Tensor] = []
    for target_edge in range(edge_index.shape[1]):
        junction = int(source[target_edge])
        incoming = torch.nonzero(target == junction, as_tuple=False).flatten()
        for source_edge_tensor in incoming:
            source_edge = int(source_edge_tensor)
            if source_edge == target_edge:
                continue
            line_sources.append(source_edge)
            line_targets.append(target_edge)
            first = pos[target[source_edge]] - pos[source[source_edge]]
            second = pos[target[target_edge]] - pos[source[target_edge]]
            first_norm = torch.linalg.vector_norm(first)
            second_norm = torch.linalg.vector_norm(second)
            if float(first_norm) == 0.0 or float(second_norm) == 0.0:
                angles.append(pos.new_zeros(()))
            else:
                cosine = (first / (first_norm + 1.0e-5) * second / (second_norm + 1.0e-5)).sum()
                angles.append(torch.arccos(cosine.clamp(-1.0, 1.0)))
    if not line_sources:
        return (
            torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
            pos.new_empty((0,), dtype=torch.float32),
        )
    return (
        torch.tensor((line_sources, line_targets), dtype=torch.long, device=edge_index.device),
        torch.stack(angles).to(torch.float32),
    )


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["add_chemrl_gem_inputs"]

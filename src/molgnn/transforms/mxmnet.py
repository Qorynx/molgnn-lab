"""MXMNet's deterministic local/global multiplex graph transform."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..data import MolecularData
from ..models.mxmnet_2020.constants import MXMNET_GLOBAL_CUTOFF
from .base import TransformError, geometry_is_proxy, with_shared_geometry


class MXMNetData(MolecularData):
    """Molecular sample whose angle tables reference local directed-edge IDs."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in {"mxmnet_two_hop_edge_index", "mxmnet_one_hop_edge_index"}:
            local_edge_index = getattr(self, "mxmnet_local_edge_index", None)
            if not isinstance(local_edge_index, Tensor):
                raise ValueError(
                    "MXMNetData requires mxmnet_local_edge_index for batching"
                )
            return int(local_edge_index.shape[1])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        if key in {"mxmnet_two_hop_edge_index", "mxmnet_one_hop_edge_index"}:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def add_mxmnet_inputs(data: MolecularData) -> MXMNetData:
    """Attach MXMNet topology while preserving canonical graph fields."""

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; MXMNet inputs must be built before batching"
        )

    atomic_number = getattr(data, "atomic_number", None)
    pos = getattr(data, "pos", None)
    if not isinstance(atomic_number, Tensor) or not isinstance(pos, Tensor):
        data = with_shared_geometry(data)
        atomic_number = data.atomic_number
        pos = data.pos
    edge_index = getattr(data, "edge_index", None)
    _validate_inputs(atomic_number, pos, edge_index, sample=sample)
    assert isinstance(atomic_number, Tensor)
    assert isinstance(pos, Tensor)
    assert isinstance(edge_index, Tensor)

    local_edge_index = _bidirected_local_graph(
        edge_index, num_nodes=atomic_number.shape[0]
    )
    global_edge_index = _radius_graph(pos, cutoff=MXMNET_GLOBAL_CUTOFF, sample=sample)
    two_hop, one_hop = _local_edge_pairs(
        local_edge_index, num_nodes=atomic_number.shape[0]
    )

    transformed = MXMNetData(**data.clone()._store)
    transformed.atomic_number = atomic_number
    transformed.pos = pos
    transformed.mxmnet_local_edge_index = local_edge_index
    transformed.mxmnet_global_edge_index = global_edge_index
    transformed.mxmnet_two_hop_edge_index = two_hop
    transformed.mxmnet_one_hop_edge_index = one_hop
    transformed.mxmnet_geometry_is_proxy = torch.tensor(
        [geometry_is_proxy(data)], dtype=torch.bool, device=pos.device
    )
    return transformed


def _validate_inputs(
    atomic_number: object,
    pos: object,
    edge_index: object,
    *,
    sample: int | str,
) -> None:
    if (
        not isinstance(atomic_number, Tensor)
        or atomic_number.ndim != 1
        or atomic_number.numel() < 1
        or atomic_number.dtype != torch.long
        or bool((atomic_number <= 0).any())
    ):
        raise TransformError(
            f"sample {sample} atomic_number must be non-empty positive long [N]"
        )
    if (
        not isinstance(pos, Tensor)
        or pos.shape != (atomic_number.shape[0], 3)
        or pos.dtype != torch.float32
        or pos.device != atomic_number.device
        or not bool(torch.isfinite(pos).all())
    ):
        raise TransformError(
            f"sample {sample} pos must be finite float32 [N, 3] on the atom device"
        )
    if (
        not isinstance(edge_index, Tensor)
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
        or edge_index.dtype != torch.long
        or edge_index.device != atomic_number.device
    ):
        raise TransformError(
            f"sample {sample} edge_index must have shape [2, E] long on the atom device"
        )
    if edge_index.numel() and (
        edge_index.min() < 0 or edge_index.max() >= atomic_number.shape[0]
    ):
        raise TransformError(f"sample {sample} edge_index contains an invalid atom")


def _bidirected_local_graph(edge_index: Tensor, *, num_nodes: int) -> Tensor:
    """Return sorted, deduplicated directed covalent edges."""

    if edge_index.shape[1] == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    source, target = edge_index
    non_self = source != target
    low = torch.minimum(source[non_self], target[non_self])
    high = torch.maximum(source[non_self], target[non_self])
    if low.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
    undirected = torch.unique(low * num_nodes + high, sorted=True)
    low = torch.div(undirected, num_nodes, rounding_mode="floor")
    high = undirected.remainder(num_nodes)
    directed = torch.cat((low * num_nodes + high, high * num_nodes + low))
    directed = torch.sort(directed).values
    return torch.stack(
        (
            torch.div(directed, num_nodes, rounding_mode="floor"),
            directed.remainder(num_nodes),
        ),
        dim=0,
    ).contiguous()


def _radius_graph(
    pos: Tensor, *, cutoff: float, sample: int | str, chunk_size: int = 256
) -> Tensor:
    """Return source-major directed radius edges without quadratic storage."""

    node_count = pos.shape[0]
    targets = torch.arange(node_count, dtype=torch.long, device=pos.device)
    source_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    for start in range(0, node_count, chunk_size):
        stop = min(start + chunk_size, node_count)
        sources = torch.arange(start, stop, dtype=torch.long, device=pos.device)
        distances = torch.cdist(pos[start:stop], pos, p=2)
        non_self = sources[:, None] != targets[None, :]
        if bool(((distances <= 1.0e-8) & non_self).any()):
            raise TransformError(f"sample {sample} contains coincident distinct atoms")
        pairs = torch.nonzero(non_self & (distances <= cutoff), as_tuple=False)
        if pairs.numel():
            source_parts.append(sources[pairs[:, 0]])
            target_parts.append(targets[pairs[:, 1]])
    if not source_parts:
        return torch.empty((2, 0), dtype=torch.long, device=pos.device)
    return torch.stack((torch.cat(source_parts), torch.cat(target_parts)), dim=0)


def _local_edge_pairs(edge_index: Tensor, *, num_nodes: int) -> tuple[Tensor, Tensor]:
    """Build paper-correct two-hop and one-hop directed-angle tables."""

    edge_count = edge_index.shape[1]
    if edge_count == 0:
        empty = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        return empty, empty.clone()
    source, target = edge_index
    incoming_by_target = [
        torch.nonzero(target == node, as_tuple=False).flatten()
        for node in range(num_nodes)
    ]
    two_incoming: list[Tensor] = []
    two_base: list[Tensor] = []
    one_sibling: list[Tensor] = []
    one_base: list[Tensor] = []
    for base_edge in range(edge_count):
        j = int(source[base_edge].item())
        i = int(target[base_edge].item())

        incoming = incoming_by_target[j]
        incoming = incoming[source[incoming] != i]
        if incoming.numel():
            two_incoming.append(incoming)
            two_base.append(torch.full_like(incoming, base_edge))

        siblings = incoming_by_target[i]
        siblings = siblings[source[siblings] != j]
        if siblings.numel():
            one_sibling.append(siblings)
            one_base.append(torch.full_like(siblings, base_edge))

    two_hop = _pair_tensor(two_incoming, two_base, edge_index.device)
    one_hop = _pair_tensor(one_sibling, one_base, edge_index.device)
    return two_hop, one_hop


def _pair_tensor(
    first: list[Tensor], second: list[Tensor], device: torch.device
) -> Tensor:
    if not first:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.stack((torch.cat(first), torch.cat(second)), dim=0).contiguous()


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = ["MXMNetData", "add_mxmnet_inputs"]

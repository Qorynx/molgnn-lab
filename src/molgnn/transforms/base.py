"""Minimal graph-transform contract used by model registry metadata."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor

from ..data import MolecularData
from ..geometry import GeometryError, ensure_sample_geometry


class GraphTransform(Protocol):
    """Callable that derives a model-specific view from canonical graph data."""

    def __call__(self, data: MolecularData) -> MolecularData: ...


class TransformError(ValueError):
    """Raised when a graph transform cannot be resolved or applied."""


def geometry_is_proxy(data: MolecularData) -> bool:
    """Return the shared geometry provenance marker for one sample."""

    marker = getattr(data, "geometry_is_proxy", None)
    return (
        bool(marker.flatten()[0].item())
        if isinstance(marker, Tensor) and marker.numel()
        else False
    )


def with_shared_geometry(data: MolecularData) -> MolecularData:
    """Route direct transform calls through the shared geometry provider."""

    try:
        transformed, _ = ensure_sample_geometry(data)
    except GeometryError as exc:
        raise TransformError(str(exc)) from exc
    return transformed


__all__ = [
    "GraphTransform",
    "TransformError",
    "geometry_is_proxy",
    "with_shared_geometry",
]

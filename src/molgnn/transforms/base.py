"""Minimal graph-transform contract used by model registry metadata."""

from __future__ import annotations

from typing import Protocol

from ..data import MolecularData


class GraphTransform(Protocol):
    """Callable that derives a model-specific view from canonical graph data."""

    def __call__(self, data: MolecularData) -> MolecularData: ...


class TransformError(ValueError):
    """Raised when a graph transform cannot be resolved or applied."""


__all__ = ["GraphTransform", "TransformError"]

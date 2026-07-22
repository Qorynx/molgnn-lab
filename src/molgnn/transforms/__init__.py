"""Small explicit registry for model-selected graph transforms."""

from __future__ import annotations

from .base import GraphTransform, TransformError
from .brics import add_brics_fragments
from .coley_2017 import add_coley_2017_features
from .directed_edges import add_reverse_edge_index

_TRANSFORMS: dict[str, GraphTransform] = {}


def register_graph_transform(name: str, transform: GraphTransform) -> None:
    """Register one named graph transform."""

    clean_name = _name(name)
    if clean_name in _TRANSFORMS:
        raise TransformError(f"graph transform '{clean_name}' is already registered")
    if not callable(transform):
        raise TransformError("graph transform must be callable")
    _TRANSFORMS[clean_name] = transform


def register_builtin_transforms() -> None:
    """Register built-in transforms exactly once."""

    if "directed_edges" not in _TRANSFORMS:
        register_graph_transform("directed_edges", add_reverse_edge_index)
    if "coley_2017_features" not in _TRANSFORMS:
        register_graph_transform("coley_2017_features", add_coley_2017_features)
    if "brics_fragments" not in _TRANSFORMS:
        register_graph_transform("brics_fragments", add_brics_fragments)


def get_graph_transform(name: str | None) -> GraphTransform | None:
    """Resolve a transform name; ``None`` means the canonical graph unchanged."""

    if name is None:
        return None
    clean_name = _name(name)
    try:
        return _TRANSFORMS[clean_name]
    except KeyError as exc:
        available = ", ".join(sorted(_TRANSFORMS)) or "<none>"
        raise TransformError(
            f"unknown graph transform '{clean_name}'. Available transforms: {available}"
        ) from exc


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransformError("graph transform name must be a non-empty string")
    return value.strip()


__all__ = [
    "GraphTransform",
    "TransformError",
    "add_brics_fragments",
    "add_coley_2017_features",
    "add_reverse_edge_index",
    "get_graph_transform",
    "register_builtin_transforms",
    "register_graph_transform",
]

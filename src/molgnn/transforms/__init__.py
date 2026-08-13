"""Small explicit registry for model-selected graph transforms."""

from __future__ import annotations

from .ampnn import add_ampnn_edge_types
from .base import GraphTransform, TransformError
from .brics import add_brics_fragments
from .coley_2017 import add_coley_2017_features
from .dimenet import add_dimenet_inputs
from .directed_edges import add_reverse_edge_index
from .egnn import add_egnn_inputs
from .fragnet import add_fragnet_inputs
from .himnet import add_himnet_inputs
from .mpnn import add_mpnn_3d_distance_bins_inputs, add_mpnn_edge_types
from .potentialnet import add_potentialnet_inputs
from .weave import add_weave_inputs

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
    if "ampnn_edge_types" not in _TRANSFORMS:
        register_graph_transform("ampnn_edge_types", add_ampnn_edge_types)
    if "coley_2017_features" not in _TRANSFORMS:
        register_graph_transform("coley_2017_features", add_coley_2017_features)
    if "mpnn_edge_types" not in _TRANSFORMS:
        register_graph_transform("mpnn_edge_types", add_mpnn_edge_types)
    if "mpnn_3d_distance_bins" not in _TRANSFORMS:
        register_graph_transform(
            "mpnn_3d_distance_bins", add_mpnn_3d_distance_bins_inputs
        )
    if "dimenet_inputs" not in _TRANSFORMS:
        register_graph_transform("dimenet_inputs", add_dimenet_inputs)
    if "brics_fragments" not in _TRANSFORMS:
        register_graph_transform("brics_fragments", add_brics_fragments)
    if "himnet_inputs" not in _TRANSFORMS:
        register_graph_transform("himnet_inputs", add_himnet_inputs)
    if "fragnet_inputs" not in _TRANSFORMS:
        register_graph_transform("fragnet_inputs", add_fragnet_inputs)
    if "potentialnet_inputs" not in _TRANSFORMS:
        register_graph_transform("potentialnet_inputs", add_potentialnet_inputs)
    if "weave_inputs" not in _TRANSFORMS:
        register_graph_transform("weave_inputs", add_weave_inputs)
    if "egnn_inputs" not in _TRANSFORMS:
        register_graph_transform("egnn_inputs", add_egnn_inputs)


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
    "add_ampnn_edge_types",
    "add_brics_fragments",
    "add_coley_2017_features",
    "add_dimenet_inputs",
    "add_egnn_inputs",
    "add_fragnet_inputs",
    "add_himnet_inputs",
    "add_mpnn_3d_distance_bins_inputs",
    "add_mpnn_edge_types",
    "add_potentialnet_inputs",
    "add_reverse_edge_index",
    "add_weave_inputs",
    "get_graph_transform",
    "register_builtin_transforms",
    "register_graph_transform",
]

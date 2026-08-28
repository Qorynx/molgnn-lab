"""Small explicit registry for model-selected graph transforms."""

from __future__ import annotations

from .ampnn import add_ampnn_edge_types
from .base import GraphTransform, TransformError
from .brics import add_brics_fragments
from .chemrl_gem import add_chemrl_gem_inputs
from .coley_2017 import add_coley_2017_features
from .dgt import add_dgt_inputs
from .dimenet import add_dimenet_inputs
from .directed_edges import add_reverse_edge_index
from .egnn import add_egnn_inputs
from .eqgat import add_eqgat_inputs
from .equiformer import add_equiformer_inputs
from .fragnet import add_fragnet_inputs
from .gemnet import add_gemnet_q_inputs, add_gemnet_t_inputs
from .gpspp import add_gpspp_inputs
from .graphmvp import add_graphmvp_inputs
from .graphormer import add_graphormer_inputs
from .grover import add_grover_inputs
from .himnet import add_himnet_inputs
from .himol import HiMolData, add_himol_inputs
from .hmgnn import add_hmgnn_inputs
from .kpgt import add_kpgt_inputs
from .mat import add_mat_inputs
from .mgcn import add_mgcn_inputs
from .molebert import add_molebert_inputs
from .mpnn import add_mpnn_3d_distance_bins_inputs, add_mpnn_edge_types
from .mxmnet import MXMNetData, add_mxmnet_inputs
from .neural_fingerprint import add_neural_fingerprint_inputs
from .painn import add_painn_inputs
from .potentialnet import add_potentialnet_inputs
from .pretrain_gnns import add_pretrain_gnns_inputs
from .schnet import add_schnet_inputs
from .spherenet import add_spherenet_inputs
from .three_d_infomax import add_three_d_infomax_inputs
from .transformer_m import add_transformer_m_inputs
from .visnet import add_visnet_inputs
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
    if "dgt_inputs" not in _TRANSFORMS:
        register_graph_transform("dgt_inputs", add_dgt_inputs)
    if "gemnet_t_inputs" not in _TRANSFORMS:
        register_graph_transform("gemnet_t_inputs", add_gemnet_t_inputs)
    if "gemnet_q_inputs" not in _TRANSFORMS:
        register_graph_transform("gemnet_q_inputs", add_gemnet_q_inputs)
    if "schnet_inputs" not in _TRANSFORMS:
        register_graph_transform("schnet_inputs", add_schnet_inputs)
    if "painn_inputs" not in _TRANSFORMS:
        register_graph_transform("painn_inputs", add_painn_inputs)
    if "visnet_inputs" not in _TRANSFORMS:
        register_graph_transform("visnet_inputs", add_visnet_inputs)
    if "eqgat_inputs" not in _TRANSFORMS:
        register_graph_transform("eqgat_inputs", add_eqgat_inputs)
    if "equiformer_inputs" not in _TRANSFORMS:
        register_graph_transform("equiformer_inputs", add_equiformer_inputs)
    if "hmgnn_inputs" not in _TRANSFORMS:
        register_graph_transform("hmgnn_inputs", add_hmgnn_inputs)
    if "gpspp_inputs" not in _TRANSFORMS:
        register_graph_transform("gpspp_inputs", add_gpspp_inputs)
    if "graphormer_inputs" not in _TRANSFORMS:
        register_graph_transform("graphormer_inputs", add_graphormer_inputs)
    if "grover_inputs" not in _TRANSFORMS:
        register_graph_transform("grover_inputs", add_grover_inputs)
    if "brics_fragments" not in _TRANSFORMS:
        register_graph_transform("brics_fragments", add_brics_fragments)
    if "himnet_inputs" not in _TRANSFORMS:
        register_graph_transform("himnet_inputs", add_himnet_inputs)
    if "himol_inputs" not in _TRANSFORMS:
        register_graph_transform("himol_inputs", add_himol_inputs)
    if "fragnet_inputs" not in _TRANSFORMS:
        register_graph_transform("fragnet_inputs", add_fragnet_inputs)
    if "potentialnet_inputs" not in _TRANSFORMS:
        register_graph_transform("potentialnet_inputs", add_potentialnet_inputs)
    if "transformer_m_inputs" not in _TRANSFORMS:
        register_graph_transform("transformer_m_inputs", add_transformer_m_inputs)
    if "weave_inputs" not in _TRANSFORMS:
        register_graph_transform("weave_inputs", add_weave_inputs)
    if "egnn_inputs" not in _TRANSFORMS:
        register_graph_transform("egnn_inputs", add_egnn_inputs)
    if "mat_inputs" not in _TRANSFORMS:
        register_graph_transform("mat_inputs", add_mat_inputs)
    if "mgcn_inputs" not in _TRANSFORMS:
        register_graph_transform("mgcn_inputs", add_mgcn_inputs)
    if "mxmnet_inputs" not in _TRANSFORMS:
        register_graph_transform("mxmnet_inputs", add_mxmnet_inputs)
    if "molebert_inputs" not in _TRANSFORMS:
        register_graph_transform("molebert_inputs", add_molebert_inputs)
    if "graphmvp_inputs" not in _TRANSFORMS:
        register_graph_transform("graphmvp_inputs", add_graphmvp_inputs)
    if "chemrl_gem_inputs" not in _TRANSFORMS:
        register_graph_transform("chemrl_gem_inputs", add_chemrl_gem_inputs)
    if "neural_fingerprint_inputs" not in _TRANSFORMS:
        register_graph_transform(
            "neural_fingerprint_inputs", add_neural_fingerprint_inputs
        )
    if "kpgt_inputs" not in _TRANSFORMS:
        register_graph_transform("kpgt_inputs", add_kpgt_inputs)
    if "three_d_infomax_inputs" not in _TRANSFORMS:
        register_graph_transform(
            "three_d_infomax_inputs", add_three_d_infomax_inputs
        )
    if "spherenet_inputs" not in _TRANSFORMS:
        register_graph_transform("spherenet_inputs", add_spherenet_inputs)
    if "pretrain_gnns_inputs" not in _TRANSFORMS:
        register_graph_transform("pretrain_gnns_inputs", add_pretrain_gnns_inputs)
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
    "HiMolData",
    "MXMNetData",
    "TransformError",
    "add_ampnn_edge_types",
    "add_brics_fragments",
    "add_chemrl_gem_inputs",
    "add_coley_2017_features",
    "add_dgt_inputs",
    "add_dimenet_inputs",
    "add_egnn_inputs",
    "add_eqgat_inputs",
    "add_equiformer_inputs",
    "add_fragnet_inputs",
    "add_gemnet_q_inputs",
    "add_gemnet_t_inputs",
    "add_gpspp_inputs",
    "add_graphmvp_inputs",
    "add_graphormer_inputs",
    "add_grover_inputs",
    "add_himnet_inputs",
    "add_himol_inputs",
    "add_hmgnn_inputs",
    "add_kpgt_inputs",
    "add_mat_inputs",
    "add_mgcn_inputs",
    "add_molebert_inputs",
    "add_mpnn_3d_distance_bins_inputs",
    "add_mpnn_edge_types",
    "add_mxmnet_inputs",
    "add_neural_fingerprint_inputs",
    "add_painn_inputs",
    "add_potentialnet_inputs",
    "add_pretrain_gnns_inputs",
    "add_reverse_edge_index",
    "add_schnet_inputs",
    "add_spherenet_inputs",
    "add_three_d_infomax_inputs",
    "add_transformer_m_inputs",
    "add_visnet_inputs",
    "add_weave_inputs",
    "get_graph_transform",
    "register_builtin_transforms",
    "register_graph_transform",
]

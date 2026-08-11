"""Built-in runtime registration, separate from architecture imports."""

from __future__ import annotations

from ..registry import available_models, register_model
from .attentivefp_2020 import AttentiveFP
from .dmpnn_2024 import DMPNN
from .gcn_baseline import GCNBaseline
from .hignn_2023 import HiGNN
from .himnet_2026 import HimNet
from .molecular_graph_embedding_2017 import MolecularGraphEmbedding
from .mpnn_2017 import MPNN, MPNNDistanceBins3D
from .potentialnet_2018 import PotentialNet
from .trimnet_2020 import TrimNet2020


def register_builtin_models() -> None:
    """Register built-ins exactly once for the shared experiment runner."""

    if "gcn_baseline" not in available_models():
        register_model(
            "gcn_baseline",
            required_batch_fields=GCNBaseline.required_batch_fields,
            prediction_reducer_name="identity",
            benchmark_order=10,
        )(GCNBaseline)
    if "attentivefp" not in available_models():
        register_model(
            "attentivefp",
            required_batch_fields=AttentiveFP.required_batch_fields,
            prediction_reducer_name="identity",
            benchmark_order=20,
        )(AttentiveFP)
    if "dmpnn" not in available_models():
        register_model(
            "dmpnn",
            required_batch_fields=DMPNN.required_batch_fields,
            graph_transform_name="directed_edges",
            prediction_reducer_name="identity",
            benchmark_order=30,
        )(DMPNN)
    if "molecular_graph_embedding" not in available_models():
        register_model(
            "molecular_graph_embedding",
            required_batch_fields=MolecularGraphEmbedding.required_batch_fields,
            graph_transform_name="coley_2017_features",
            prediction_reducer_name="identity",
            benchmark_order=50,
        )(MolecularGraphEmbedding)
    if "hignn" not in available_models():
        register_model(
            "hignn",
            required_batch_fields=HiGNN.required_batch_fields,
            graph_transform_name="brics_fragments",
            prediction_reducer_name="identity",
            benchmark_order=40,
        )(HiGNN)
    if "mpnn" not in available_models():
        register_model(
            "mpnn",
            required_batch_fields=MPNN.required_batch_fields,
            graph_transform_name="mpnn_edge_types",
            prediction_reducer_name="identity",
            benchmark_order=45,
        )(MPNN)
    if "mpnn_3d_distance_bins" not in available_models():
        register_model(
            "mpnn_3d_distance_bins",
            required_batch_fields=MPNNDistanceBins3D.required_batch_fields,
            graph_transform_name="mpnn_3d_distance_bins",
            transform_output_fields=(
                "mpnn_3d_edge_index",
                "mpnn_3d_edge_type",
            ),
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=46,
        )(MPNNDistanceBins3D)
    if "trimnet_2020" not in available_models():
        register_model(
            "trimnet_2020",
            required_batch_fields=TrimNet2020.required_batch_fields,
            prediction_reducer_name="identity",
            benchmark_order=60,
        )(TrimNet2020)
    if "himnet" not in available_models():
        register_model(
            "himnet",
            required_batch_fields=HimNet.required_batch_fields,
            graph_transform_name="himnet_inputs",
            prediction_reducer_name="identity",
            benchmark_order=70,
        )(HimNet)
    if "potentialnet" not in available_models():
        register_model(
            "potentialnet",
            required_batch_fields=PotentialNet.required_batch_fields,
            optional_batch_fields=(
                "potentialnet_stage2_edge_index",
                "potentialnet_stage2_edge_type",
                "potentialnet_use_spatial",
            ),
            graph_transform_name="potentialnet_inputs",
            transform_output_fields=(
                "ligand_mask",
                "potentialnet_bond_edge_index",
                "potentialnet_bond_edge_type",
                "potentialnet_stage2_edge_index",
                "potentialnet_stage2_edge_type",
                "potentialnet_use_spatial",
            ),
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=80,
        )(PotentialNet)


__all__ = ["register_builtin_models"]

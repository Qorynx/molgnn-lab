"""Built-in runtime registration, separate from architecture imports."""

from __future__ import annotations

from ..registry import available_models, register_model
from .ampnn_emnn_2020.ampnn import AMPNN
from .ampnn_emnn_2020.emnn import EMNN
from .attentivefp_2020 import AttentiveFP
from .dimenet_2020 import DimeNet2020
from .dmpnn_2024 import DMPNN
from .fragnet_2026 import FragNet
from .gcn_baseline import GCNBaseline
from .hignn_2023 import HiGNN
from .himnet_2026 import HimNet
from .molecular_graph_embedding_2017 import MolecularGraphEmbedding
from .mpnn_2017 import MPNN, MPNNDistanceBins3D
from .mvgnn_2020 import MVGNNcross
from .potentialnet_2018 import PotentialNet
from .resgat_2024 import ResGAT
from .trimnet_2020 import TrimNet2020
from .weave_2016 import Weave


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
    if "ampnn" not in available_models():
        register_model(
            "ampnn",
            required_batch_fields=AMPNN.required_batch_fields,
            graph_transform_name="ampnn_edge_types",
            transform_output_fields=("ampnn_edge_type",),
            prediction_reducer_name="identity",
            benchmark_order=25,
        )(AMPNN)
    if "dmpnn" not in available_models():
        register_model(
            "dmpnn",
            required_batch_fields=DMPNN.required_batch_fields,
            graph_transform_name="directed_edges",
            prediction_reducer_name="identity",
            benchmark_order=30,
        )(DMPNN)
    if "dimenet" not in available_models():
        register_model(
            "dimenet",
            required_batch_fields=DimeNet2020.required_batch_fields,
            graph_transform_name="dimenet_inputs",
            transform_output_fields=(
                "dimenet_edge_index",
                "dimenet_triplet_edge_index",
            ),
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=32,
        )(DimeNet2020)
    if "emnn" not in available_models():
        register_model(
            "emnn",
            required_batch_fields=EMNN.required_batch_fields,
            graph_transform_name="directed_edges",
            prediction_reducer_name="identity",
            benchmark_order=35,
        )(EMNN)
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
    if "weave" not in available_models():
        register_model(
            "weave",
            required_batch_fields=Weave.required_batch_fields,
            graph_transform_name="weave_inputs",
            transform_output_fields=("weave_pair_index", "weave_pair_attr"),
            prediction_reducer_name="identity",
            benchmark_order=65,
        )(Weave)
    if "himnet" not in available_models():
        register_model(
            "himnet",
            required_batch_fields=HimNet.required_batch_fields,
            graph_transform_name="himnet_inputs",
            prediction_reducer_name="identity",
            benchmark_order=70,
        )(HimNet)
    if "fragnet" not in available_models():
        register_model(
            "fragnet",
            required_batch_fields=FragNet.required_batch_fields,
            graph_transform_name="fragnet_inputs",
            transform_output_fields=(
                "frag_index",
                "x_frags",
                "atom_to_fragment",
                "frag_batch",
                "edge_index_bonds_graph",
                "edge_attr_bonds",
                "frag_connection_features",
                "edge_index_fbonds",
                "edge_attr_fbonds",
            ),
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=80,
        )(FragNet)
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
    if "resgat" not in available_models():
        register_model(
            "resgat",
            required_batch_fields=ResGAT.required_batch_fields,
            prediction_reducer_name="identity",
            benchmark_order=75,
        )(ResGAT)
    if "mvgnn_cross" not in available_models():
        register_model(
            "mvgnn_cross",
            required_batch_fields=MVGNNcross.required_batch_fields,
            prediction_reducer_name="identity",
            benchmark_order=76,
        )(MVGNNcross)


__all__ = ["register_builtin_models"]

"""Built-in runtime registration, separate from architecture imports."""

from __future__ import annotations

from ..registry import available_models, register_model
from .ampnn_emnn_2020.ampnn import AMPNN
from .ampnn_emnn_2020.emnn import EMNN
from .attentivefp_2020 import AttentiveFP
from .dimenet_2020 import DimeNet2020
from .dmpnn_2024 import DMPNN
from .egnn_2021 import EGNN
from .eqgat_2022 import EQGAT
from .equiformer_2023 import Equiformer
from .ewaldmp_2023 import EwaldMP
from .fragnet_2026 import FragNet
from .gcn_baseline import GCNBaseline
from .gemnet_2021 import GemNetQ, GemNetT
from .gpspp_2023 import GPSPlusPlus
from .graphormer_2021 import Graphormer
from .grover_2021 import GROVER
from .hignn_2023 import HiGNN
from .himnet_2026 import HimNet
from .hmgnn_2020 import HMGNN
from .mat_2020 import MAT
from .molclr_2022.model import MolCLRGCN, MolCLRGIN
from .molebert_2023 import MoleBERT
from .molecular_graph_embedding_2017 import MolecularGraphEmbedding
from .mpnn_2017 import MPNN, MPNNDistanceBins3D
from .mvgnn_2020 import MVGNNcross
from .painn_2021 import PaiNN
from .potentialnet_2018 import PotentialNet
from .resgat_2024 import ResGAT
from .schnet_2017 import SchNet
from .transformer_m_2023 import TransformerM
from .trimnet_2020 import TrimNet2020
from .visnet_2023 import ViSNet
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
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=32,
        )(DimeNet2020)
    if "gemnet_t" not in available_models():
        register_model(
            "gemnet_t",
            required_batch_fields=GemNetT.required_batch_fields,
            graph_transform_name="gemnet_t_inputs",
            transform_output_fields=(
                "atomic_number",
                "pos",
                "gemnet_edge_index",
                "gemnet_reverse_edge_index",
                "gemnet_triplet_edge_index",
            ),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=40,
        )(GemNetT)
    if "gemnet_q" not in available_models():
        register_model(
            "gemnet_q",
            required_batch_fields=GemNetQ.required_batch_fields,
            graph_transform_name="gemnet_q_inputs",
            transform_output_fields=(
                "atomic_number",
                "pos",
                "gemnet_edge_index",
                "gemnet_reverse_edge_index",
                "gemnet_triplet_edge_index",
                "gemnet_interaction_edge_index",
                "gemnet_quadruplet_edge_index",
                "gemnet_quadruplet_interaction_index",
            ),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=41,
        )(GemNetQ)
    if "schnet" not in available_models():
        register_model(
            "schnet",
            required_batch_fields=SchNet.required_batch_fields,
            graph_transform_name="schnet_inputs",
            transform_output_fields=(
                "atomic_number",
                "pos",
                "schnet_edge_index",
            ),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=33,
        )(SchNet)
    if "eqgat" not in available_models():
        register_model(
            "eqgat",
            required_batch_fields=EQGAT.required_batch_fields,
            graph_transform_name="eqgat_inputs",
            transform_output_fields=(
                "atomic_number",
                "pos",
                "eqgat_edge_index",
            ),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=34,
        )(EQGAT)
    if "hmgnn" not in available_models():
        register_model(
            "hmgnn",
            required_batch_fields=HMGNN.required_batch_fields,
            graph_transform_name="hmgnn_inputs",
            transform_output_fields=(
                "atomic_number",
                "pos",
                "hmgnn_atom_edge_index",
                "hmgnn_body_atom_index",
                "hmgnn_body_edge_index",
            ),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=35,
        )(HMGNN)
    if "equiformer" not in available_models():
        register_model(
            "equiformer",
            required_batch_fields=Equiformer.required_batch_fields,
            graph_transform_name="equiformer_inputs",
            transform_output_fields=(
                "atomic_number",
                "pos",
                "equiformer_edge_index",
            ),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=36,
        )(Equiformer)
    if "painn" not in available_models():
        register_model(
            "painn",
            required_batch_fields=PaiNN.required_batch_fields,
            graph_transform_name="painn_inputs",
            transform_output_fields=(
                "atomic_number",
                "pos",
                "painn_edge_index",
            ),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=37,
        )(PaiNN)
    if "visnet" not in available_models():
        register_model(
            "visnet",
            required_batch_fields=ViSNet.required_batch_fields,
            graph_transform_name="visnet_inputs",
            transform_output_fields=(
                "atomic_number",
                "pos",
                "visnet_edge_index",
            ),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=38,
        )(ViSNet)
    if "ewaldmp" not in available_models():
        register_model(
            "ewaldmp",
            required_batch_fields=EwaldMP.required_batch_fields,
            graph_transform_name="painn_inputs",
            transform_output_fields=(
                "atomic_number",
                "pos",
                "painn_edge_index",
            ),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=39,
        )(EwaldMP)
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
            geometry_requirement="required",
            geometry_role="pure_3d",
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
    if "gpspp" not in available_models():
        register_model(
            "gpspp",
            required_batch_fields=GPSPlusPlus.required_batch_fields,
            graph_transform_name="gpspp_inputs",
            transform_output_fields=("gpspp_pair_index", "gpspp_spd"),
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=67,
        )(GPSPlusPlus)
    if "transformer_m" not in available_models():
        register_model(
            "transformer_m",
            required_batch_fields=TransformerM.required_batch_fields,
            graph_transform_name="transformer_m_inputs",
            transform_output_fields=(
                "transformer_m_x",
                "transformer_m_in_degree",
                "transformer_m_out_degree",
                "transformer_m_pair_index",
                "transformer_m_spatial_pos",
                "transformer_m_path_index",
                "transformer_m_path_step",
                "transformer_m_path_type",
            ),
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=68,
        )(TransformerM)
    if "graphormer" not in available_models():
        register_model(
            "graphormer",
            required_batch_fields=Graphormer.required_batch_fields,
            graph_transform_name="graphormer_inputs",
            transform_output_fields=(
                "graphormer_x",
                "graphormer_in_degree",
                "graphormer_out_degree",
                "graphormer_pair_index",
                "graphormer_spatial_pos",
                "graphormer_path_index",
                "graphormer_path_step",
                "graphormer_path_edge_type",
            ),
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=69,
        )(Graphormer)
    if "grover" not in available_models():
        register_model(
            "grover",
            required_batch_fields=GROVER.required_batch_fields,
            graph_transform_name="grover_inputs",
            transform_output_fields=(
                "grover_f_atoms",
                "grover_f_bonds",
                "grover_reverse_bond",
            ),
            prediction_reducer_name="identity",
            benchmark_enabled=False,
            benchmark_order=69,
        )(GROVER)
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
            geometry_requirement="required",
            geometry_role="hybrid",
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
            geometry_requirement="optional",
            geometry_role="hybrid",
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
    if "mat" not in available_models():
        register_model(
            "mat",
            required_batch_fields=MAT.required_batch_fields,
            graph_transform_name="mat_inputs",
            geometry_requirement="required",
            geometry_role="hybrid",
            prediction_reducer_name="identity",
            benchmark_order=78,
        )(MAT)
    if "egnn" not in available_models():
        register_model(
            "egnn",
            required_batch_fields=EGNN.required_batch_fields,
            graph_transform_name="egnn_inputs",
            transform_output_fields=("pos",),
            geometry_requirement="required",
            geometry_role="pure_3d",
            prediction_reducer_name="identity",
            benchmark_order=77,
        )(EGNN)
    if "molclr_gin" not in available_models():
        register_model(
            "molclr_gin",
            default_parameters=dict(
                emb_dim=300, feat_dim=256, num_layer=5, drop_ratio=0.0, pool="mean"
            ),
            required_batch_fields=MolCLRGIN.required_batch_fields,
            optional_batch_fields=(),
            graph_transform_name=None,
            transform_output_fields=(),
            prediction_reducer_name="identity",
            benchmark_enabled=True,
            benchmark_order=79,
        )(MolCLRGIN)
    if "molclr_gcn" not in available_models():
        register_model(
            "molclr_gcn",
            default_parameters=dict(
                emb_dim=300, feat_dim=256, num_layer=5, drop_ratio=0.0, pool="mean"
            ),
            required_batch_fields=MolCLRGCN.required_batch_fields,
            optional_batch_fields=(),
            graph_transform_name=None,
            transform_output_fields=(),
            prediction_reducer_name="identity",
            benchmark_enabled=True,
            benchmark_order=81,
        )(MolCLRGCN)
    if "molebert" not in available_models():
        register_model(
            "molebert",
            default_parameters=dict(
                hidden_dim=300,
                num_layers=5,
                jk="last",
                dropout=0.0,
                pooling="mean",
                pretrained_checkpoint=None,
            ),
            required_batch_fields=MoleBERT.required_batch_fields,
            graph_transform_name="molebert_inputs",
            transform_output_fields=("molebert_atom_attr", "molebert_bond_attr"),
            prediction_reducer_name="identity",
            benchmark_enabled=True,
            benchmark_order=83,
        )(MoleBERT)


__all__ = ["register_builtin_models"]

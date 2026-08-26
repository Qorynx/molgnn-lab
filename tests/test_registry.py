"""Phase 3 model registry contracts."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from molgnn.models.registration import register_builtin_models
from molgnn.registry import (
    BuildContext,
    RegistryError,
    available_models,
    benchmark_models,
    build_model,
    get_model_spec,
    register_model,
    resolve_benchmark_models,
    resolve_model_parameters,
    validate_required_batch_fields,
)


def test_builtin_registration_and_context_injection() -> None:
    register_builtin_models()
    model = build_model(
        "gcn_baseline",
        {"hidden_dim": 8, "num_layers": 1},
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    assert isinstance(model, nn.Module)
    assert "gcn_baseline" in available_models()
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    attentive = build_model(
        "attentivefp",
        {"hidden_dim": 8, "num_atom_layers": 1, "num_molecule_layers": 1},
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    spec = get_model_spec("attentivefp")
    assert isinstance(attentive, nn.Module)
    assert spec.required_batch_fields == ("x", "edge_index", "edge_attr", "batch")
    assert spec.graph_transform_name is None
    assert spec.prediction_reducer_name == "identity"
    ampnn = build_model(
        "ampnn",
        {},
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    ampnn_spec = get_model_spec("ampnn")
    assert isinstance(ampnn, nn.Module)
    assert ampnn_spec.required_batch_fields == (
        "x",
        "edge_index",
        "ampnn_edge_type",
        "batch",
    )
    assert ampnn_spec.graph_transform_name == "ampnn_edge_types"
    assert ampnn_spec.transform_output_fields == ("ampnn_edge_type",)
    assert ampnn_spec.prediction_reducer_name == "identity"
    assert ampnn_spec.benchmark_enabled is True
    assert ampnn_spec.benchmark_order == 25
    dmpnn = build_model(
        "dmpnn",
        {"hidden_dim": 8, "depth": 2, "ffn_hidden_dim": 8},
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    dmpnn_spec = get_model_spec("dmpnn")
    assert isinstance(dmpnn, nn.Module)
    assert dmpnn_spec.graph_transform_name == "directed_edges"
    assert dmpnn_spec.prediction_reducer_name == "identity"
    assert "reverse_edge_index" in dmpnn_spec.required_batch_fields
    dimenet = build_model(
        "dimenet",
        {
            "hidden_dim": 8,
            "num_blocks": 1,
            "num_bilinear": 2,
            "num_spherical": 2,
            "num_radial": 2,
            "num_before_skip": 1,
            "num_after_skip": 1,
            "num_dense_output": 1,
        },
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    dimenet_spec = get_model_spec("dimenet")
    assert isinstance(dimenet, nn.Module)
    assert dimenet_spec.required_batch_fields == (
        "atomic_number",
        "pos",
        "dimenet_edge_index",
        "dimenet_triplet_edge_index",
        "batch",
    )
    assert dimenet_spec.graph_transform_name == "dimenet_inputs"
    assert dimenet_spec.transform_output_fields == (
        "dimenet_edge_index",
        "dimenet_triplet_edge_index",
    )
    assert dimenet_spec.prediction_reducer_name == "identity"
    assert dimenet_spec.benchmark_enabled is False
    assert dimenet_spec.benchmark_order == 32
    dimenet_pp = build_model(
        "dimenet_pp",
        {
            "hidden_dim": 8,
            "interaction_dim": 4,
            "basis_dim": 2,
            "output_dim": 8,
            "num_blocks": 1,
            "num_spherical": 2,
            "num_radial": 2,
            "num_before_skip": 1,
            "num_after_skip": 1,
            "num_dense_output": 1,
        },
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    dimenet_pp_spec = get_model_spec("dimenet_pp")
    assert isinstance(dimenet_pp, nn.Module)
    assert dimenet_pp_spec.required_batch_fields == (
        "atomic_number",
        "pos",
        "dimenet_edge_index",
        "dimenet_triplet_edge_index",
        "batch",
    )
    assert dimenet_pp_spec.graph_transform_name == "dimenet_inputs"
    assert dimenet_pp_spec.transform_output_fields == (
        "dimenet_edge_index",
        "dimenet_triplet_edge_index",
    )
    assert dimenet_pp_spec.prediction_reducer_name == "identity"
    assert dimenet_pp_spec.geometry_requirement == "required"
    assert dimenet_pp_spec.geometry_role == "pure_3d"
    assert dimenet_pp_spec.benchmark_enabled is False
    assert dimenet_pp_spec.benchmark_order == 42
    dgt = build_model(
        "dgt",
        {
            "dim_h": 16,
            "num_heads": 4,
            "num_layers": 2,
            "dropout": 0.0,
            "attn_dropout": 0.0,
            "head_layers": 1,
            "spd_max_length": 4,
            "rwse_steps": 4,
        },
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    dgt_spec = get_model_spec("dgt")
    assert isinstance(dgt, nn.Module)
    assert dgt_spec.graph_transform_name == "dgt_inputs"
    assert dgt_spec.transform_output_fields == (
        "dgt_e2e_edge_index",
        "dgt_e2e_node_index",
        "dgt_e_batch",
        "dgt_spd_index",
        "dgt_spd_lengths",
        "dgt_e2e_spd_index",
        "dgt_e2e_spd_lengths",
        "dgt_rwse",
        "dgt_e2e_rwse",
    )
    assert dgt_spec.prediction_reducer_name == "identity"
    assert dgt_spec.geometry_requirement == "none"
    assert dgt_spec.geometry_role == "none"
    assert dgt_spec.benchmark_enabled is False
    assert dgt_spec.benchmark_order == 43
    gpspp = build_model(
        "gpspp",
        {
            "node_dim": 8,
            "edge_dim": 4,
            "global_dim": 4,
            "depth": 1,
            "num_heads": 2,
            "max_spd": 4,
            "decoder_hidden_dim": 8,
        },
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    gpspp_spec = get_model_spec("gpspp")
    assert isinstance(gpspp, nn.Module)
    assert gpspp_spec.required_batch_fields == (
        "x",
        "edge_index",
        "edge_attr",
        "gpspp_pair_index",
        "gpspp_spd",
        "batch",
    )
    assert gpspp_spec.graph_transform_name == "gpspp_inputs"
    assert gpspp_spec.transform_output_fields == (
        "gpspp_pair_index",
        "gpspp_spd",
    )
    assert gpspp_spec.prediction_reducer_name == "identity"
    assert gpspp_spec.benchmark_enabled is False
    assert gpspp_spec.benchmark_order == 67
    emnn = build_model(
        "emnn",
        {},
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    emnn_spec = get_model_spec("emnn")
    assert isinstance(emnn, nn.Module)
    assert emnn_spec.required_batch_fields == (
        "x",
        "edge_index",
        "edge_attr",
        "reverse_edge_index",
        "batch",
    )
    assert emnn_spec.graph_transform_name == "directed_edges"
    assert emnn_spec.prediction_reducer_name == "identity"
    assert emnn_spec.benchmark_enabled is True
    assert emnn_spec.benchmark_order == 35
    mpnn = build_model(
        "mpnn",
        {
            "hidden_dim": 160,
            "num_message_passing_steps": 1,
            "readout_hidden_dim": 4,
            "readout_num_hidden_layers": 1,
        },
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    mpnn_spec = get_model_spec("mpnn")
    assert isinstance(mpnn, nn.Module)
    assert mpnn_spec.required_batch_fields == (
        "x",
        "edge_index",
        "mpnn_edge_type",
        "batch",
    )
    assert mpnn_spec.graph_transform_name == "mpnn_edge_types"
    assert mpnn_spec.prediction_reducer_name == "identity"
    mpnn_3d = build_model(
        "mpnn_3d_distance_bins",
        {
            "hidden_dim": 160,
            "num_message_passing_steps": 1,
            "readout_hidden_dim": 4,
            "readout_num_hidden_layers": 1,
        },
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    mpnn_3d_spec = get_model_spec("mpnn_3d_distance_bins")
    assert isinstance(mpnn_3d, nn.Module)
    assert mpnn_3d_spec.required_batch_fields == (
        "x",
        "mpnn_3d_edge_index",
        "mpnn_3d_edge_type",
        "batch",
    )
    assert mpnn_3d_spec.graph_transform_name == "mpnn_3d_distance_bins"
    assert mpnn_3d_spec.transform_output_fields == (
        "mpnn_3d_edge_index",
        "mpnn_3d_edge_type",
    )
    assert mpnn_3d_spec.prediction_reducer_name == "identity"
    assert mpnn_3d_spec.benchmark_enabled is False
    assert mpnn_3d_spec.benchmark_order == 46
    molecular_graph_embedding = build_model(
        "molecular_graph_embedding",
        {
            "depth": 1,
            "message_dim": 8,
            "fingerprint_dim": 16,
            "predictor_hidden_dim": 8,
        },
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    molecular_graph_embedding_spec = get_model_spec("molecular_graph_embedding")
    assert isinstance(molecular_graph_embedding, nn.Module)
    assert molecular_graph_embedding_spec.graph_transform_name == "coley_2017_features"
    assert molecular_graph_embedding_spec.prediction_reducer_name == "identity"
    assert "mge_x" in molecular_graph_embedding_spec.required_batch_fields
    hignn = build_model(
        "hignn",
        {"hidden_dim": 8, "num_layers": 1, "num_slices": 2, "feature_reduction": 2},
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    hignn_spec = get_model_spec("hignn")
    assert isinstance(hignn, nn.Module)
    assert hignn_spec.graph_transform_name == "brics_fragments"
    assert hignn_spec.prediction_reducer_name == "identity"
    assert "atom_to_fragment" in hignn_spec.required_batch_fields
    trimnet = build_model(
        "trimnet_2020",
        {"hidden_dim": 8, "depth": 1, "heads": 2, "num_timesteps": 1},
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    trimnet_spec = get_model_spec("trimnet_2020")
    assert isinstance(trimnet, nn.Module)
    assert trimnet_spec.required_batch_fields == (
        "x",
        "edge_index",
        "edge_attr",
        "batch",
    )
    assert trimnet_spec.graph_transform_name is None
    assert trimnet_spec.prediction_reducer_name == "identity"
    weave = build_model(
        "weave",
        {
            "hidden_dim": 8,
            "num_weave_modules": 1,
            "graph_feature_dim": 8,
            "predictor_hidden_dims": (8,),
            "dropout": 0.0,
        },
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    weave_spec = get_model_spec("weave")
    assert isinstance(weave, nn.Module)
    assert weave_spec.required_batch_fields == (
        "x",
        "weave_pair_index",
        "weave_pair_attr",
        "batch",
    )
    assert weave_spec.graph_transform_name == "weave_inputs"
    assert weave_spec.transform_output_fields == (
        "weave_pair_index",
        "weave_pair_attr",
    )
    assert weave_spec.prediction_reducer_name == "identity"
    assert weave_spec.benchmark_enabled is True
    assert weave_spec.benchmark_order == 65
    himnet = build_model(
        "himnet",
        {
            "hidden_dim": 8,
            "depth": 1,
            "interaction_heads": 2,
            "fusion_heads": 2,
            "dropout": 0.0,
        },
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    himnet_spec = get_model_spec("himnet")
    assert isinstance(himnet, nn.Module)
    assert himnet_spec.graph_transform_name == "himnet_inputs"
    assert himnet_spec.prediction_reducer_name == "identity"
    assert "himnet_fp" in himnet_spec.required_batch_fields
    fragnet_spec = get_model_spec("fragnet")
    assert fragnet_spec.graph_transform_name == "fragnet_inputs"
    assert fragnet_spec.transform_output_fields == (
        "frag_index",
        "x_frags",
        "atom_to_fragment",
        "frag_batch",
        "edge_index_bonds_graph",
        "edge_attr_bonds",
        "frag_connection_features",
        "edge_index_fbonds",
        "edge_attr_fbonds",
    )
    assert fragnet_spec.benchmark_enabled is False
    potentialnet = build_model(
        "potentialnet",
        {
            "bond_hidden_dim": 48,
            "spatial_hidden_dim": 48,
            "gather_dim": 48,
            "num_bond_steps": 1,
            "num_spatial_steps": 1,
            "readout_hidden_dims": (8,),
        },
        BuildContext(atom_dim=44, bond_dim=5, num_targets=2),
    )
    potentialnet_spec = get_model_spec("potentialnet")
    assert isinstance(potentialnet, nn.Module)
    assert potentialnet_spec.graph_transform_name == "potentialnet_inputs"
    assert potentialnet_spec.optional_batch_fields == (
        "potentialnet_stage2_edge_index",
        "potentialnet_stage2_edge_type",
        "potentialnet_use_spatial",
    )
    assert potentialnet_spec.transform_output_fields == (
        "ligand_mask",
        "potentialnet_bond_edge_index",
        "potentialnet_bond_edge_type",
        "potentialnet_stage2_edge_index",
        "potentialnet_stage2_edge_type",
        "potentialnet_use_spatial",
    )
    assert potentialnet_spec.benchmark_enabled is False


def test_unknown_model_lists_available_models() -> None:
    register_builtin_models()
    with pytest.raises(
        RegistryError,
match="Available models: 3d_infomax, ampnn, attentivefp, chemrl_gem, dgt, dimenet, dimenet_pp, dmpnn, egnn, emnn, eqgat, equiformer, ewaldmp, fragnet, "
            "gcn_baseline, gemnet_q, gemnet_t, gpspp, graphmvp, graphormer, grover, hignn, himnet, himol, hmgnn, kpgt, mat, mgcn, molclr_gcn, "
            "molclr_gin, molebert, molecular_graph_embedding, mpnn, mpnn_3d_distance_bins, "
            "mvgnn_cross, neural_fingerprint, painn, potentialnet, pretrain_gnns, resgat, schnet, spherenet, transformer_m, trimnet_2020, visnet, weave",
    ):
        build_model("missing", {}, BuildContext(1, 1, 1))


def test_pretrain_gnns_registration_contract_and_factory() -> None:
    register_builtin_models()
    spec = get_model_spec("pretrain_gnns")
    assert spec.graph_transform_name == "pretrain_gnns_inputs"
    assert spec.transform_output_fields == (
        "pretrain_gnns_atom_attr",
        "pretrain_gnns_bond_attr",
    )
    assert spec.geometry_requirement == "none"
    assert spec.geometry_role == "none"
    assert spec.benchmark_enabled is False
    model = build_model(
        "pretrain_gnns",
        {"num_layer": 2, "emb_dim": 8},
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    assert isinstance(model, nn.Module)


def test_himol_registration_contract_and_factory() -> None:
    from molgnn.models.himol_2023 import HiMol

    register_builtin_models()
    spec = get_model_spec("himol")
    assert spec.required_batch_fields == HiMol.required_batch_fields
    assert spec.graph_transform_name == "himol_inputs"
    assert spec.transform_output_fields == HiMol.required_batch_fields
    assert spec.geometry_requirement == "none"
    assert spec.geometry_role == "topology_2d"
    assert spec.benchmark_enabled is True
    model = build_model(
        "himol",
        {"num_layer": 2, "emb_dim": 8, "drop_ratio": 0.0},
        BuildContext(atom_dim=153, bond_dim=14, num_targets=2),
    )
    assert isinstance(model, nn.Module)


def test_unknown_model_parameter_is_rejected() -> None:
    register_builtin_models()
    with pytest.raises(RegistryError, match="unknown parameter"):
        build_model("gcn_baseline", {"not_a_parameter": 1}, BuildContext(1, 1, 1))


def test_benchmark_selection_uses_default_order_and_preserves_explicit_order() -> None:
    register_builtin_models()

    assert tuple(spec.name for spec in benchmark_models()) == (
        "gcn_baseline",
        "neural_fingerprint",
        "attentivefp",
        "ampnn",
        "dmpnn",
        "emnn",
        "hignn",
        "mpnn",
        "molecular_graph_embedding",
        "trimnet_2020",
        "weave",
        "himnet",
        "resgat",
        "mvgnn_cross",
        "egnn",
        "mat",
        "molclr_gin",
        "molclr_gcn",
        "molebert",
        "graphmvp",
        "chemrl_gem",
        "himol",
    )
    assert tuple(
        spec.name for spec in resolve_benchmark_models(("dmpnn", "gcn_baseline"))
    ) == ("dmpnn", "gcn_baseline")
    with pytest.raises(RegistryError, match="must not contain duplicates"):
        resolve_benchmark_models(("dmpnn", "dmpnn"))
    with pytest.raises(RegistryError, match="unknown model 'missing'"):
        resolve_benchmark_models(("missing",))


def test_model_parameters_materialize_defaults_and_reject_context_overrides() -> None:
    register_builtin_models()
    spec = get_model_spec("gcn_baseline")
    context = BuildContext(atom_dim=153, bond_dim=14, num_targets=2)

    resolved = resolve_model_parameters(spec, {"hidden_dim": 8}, context)

    assert resolved == {"hidden_dim": 8, "num_layers": 3, "dropout": 0.0}
    assert not {"atom_dim", "bond_dim", "num_targets"} & set(resolved)
    with pytest.raises(RegistryError, match="managed by BuildContext"):
        resolve_model_parameters(spec, {"atom_dim": 153}, context)


def test_registered_default_parameters_support_required_factory_arguments() -> None:
    class TinyWithRequiredWidth(nn.Module):
        def __init__(self, width: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(width))

    register_model(
        "registry_test_defaults",
        default_parameters={"width": 3},
        benchmark_enabled=False,
        benchmark_order=999,
    )(TinyWithRequiredWidth)
    spec = get_model_spec("registry_test_defaults")

    assert resolve_model_parameters(spec, {}, BuildContext(1, 1, 1)) == {"width": 3}
    assert isinstance(
        build_model("registry_test_defaults", {}, BuildContext(1, 1, 1)), nn.Module
    )


def test_required_batch_fields_fail_before_training() -> None:
    register_builtin_models()
    incomplete = SimpleNamespace(
        x=torch.zeros((1, 153)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 14)),
        batch=torch.zeros(1, dtype=torch.long),
    )
    with pytest.raises(RegistryError, match="reverse_edge_index"):
        validate_required_batch_fields(incomplete, get_model_spec("dmpnn"))


def test_duplicate_registration_is_rejected() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    register_model("registry_test_tiny")(Tiny)
    with pytest.raises(RegistryError, match="already registered"):
        register_model("registry_test_tiny")(Tiny)


def test_optional_batch_fields_cannot_overlap_required_fields() -> None:
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    with pytest.raises(RegistryError, match="must not duplicate"):
        register_model(
            "registry_test_optional_overlap",
            required_batch_fields=("x",),
            optional_batch_fields=("x",),
        )(Tiny)

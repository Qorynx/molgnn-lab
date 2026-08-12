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
        match="Available models: attentivefp, dmpnn, gcn_baseline, hignn, "
        "himnet, molecular_graph_embedding, mpnn, mpnn_3d_distance_bins, "
        "potentialnet, trimnet_2020",
    ):
        build_model("missing", {}, BuildContext(1, 1, 1))


def test_unknown_model_parameter_is_rejected() -> None:
    register_builtin_models()
    with pytest.raises(RegistryError, match="unknown parameter"):
        build_model("gcn_baseline", {"not_a_parameter": 1}, BuildContext(1, 1, 1))


def test_benchmark_selection_uses_default_order_and_preserves_explicit_order() -> None:
    register_builtin_models()

    assert tuple(spec.name for spec in benchmark_models()) == (
        "gcn_baseline",
        "attentivefp",
        "dmpnn",
        "hignn",
        "mpnn",
        "molecular_graph_embedding",
        "trimnet_2020",
        "himnet",
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

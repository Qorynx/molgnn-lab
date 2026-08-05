"""Durable regression checks for architecture invariants and runtime contracts."""

import torch
from torch_geometric.loader import DataLoader

from molgnn.featurizer import featurize_smiles
from molgnn.models.hignn_2023 import HiGNN
from molgnn.models.himnet_2026 import HimNet
from molgnn.models.registration import register_builtin_models
from molgnn.registry import available_models, get_model_spec
from molgnn.transforms import add_brics_fragments, add_himnet_inputs


def _hignn_batch():
    samples = [
        add_brics_fragments(
            featurize_smiles(smiles, targets=[0.0], target_mask=[True], sample_id=index)
        )
        for index, smiles in enumerate(("CC(=O)NCC", "C"))
    ]
    return next(iter(DataLoader(samples, batch_size=len(samples))))


def test_builtin_models_expose_runtime_input_contracts() -> None:
    register_builtin_models()

    expected = {
        "gcn_baseline",
        "attentivefp",
        "dmpnn",
        "hignn",
        "himnet",
        "molecular_graph_embedding",
        "trimnet_2020",
    }
    assert expected <= set(available_models())
    for name in expected:
        spec = get_model_spec(name)
        assert spec.required_batch_fields
        assert spec.required_batch_fields == spec.factory.required_batch_fields
        assert not hasattr(spec, "architecture_card")


def test_hignn_cross_gat_shares_one_projection_and_keeps_bipartite_fusion() -> None:
    batch = _hignn_batch()
    model = HiGNN(
        atom_dim=153,
        bond_dim=14,
        hidden_dim=8,
        num_layers=1,
        num_slices=2,
        feature_reduction=2,
        dropout=0.0,
    ).eval()
    fusion = model.fragment_to_molecule

    # HiGNN Eq. (12) applies W6 to both the fragment and molecule inputs.
    # The tuple-channel GAT constructor would instead create lin_src/lin_dst.
    assert fusion.in_channels == model.hidden_dim
    assert fusion.lin is not None
    assert fusion.lin_src is None
    assert fusion.lin_dst is None

    calls: list[tuple[torch.Size, torch.Size, tuple[int, int]]] = []

    def capture(_module, inputs, kwargs) -> None:
        source, target = inputs[0]
        calls.append((source.shape, target.shape, kwargs["size"]))

    handle = fusion.register_forward_pre_hook(capture, with_kwargs=True)
    model(batch)
    handle.remove()

    assert len(calls) == 1
    source_shape, target_shape, size = calls[0]
    num_fragments = int(batch.atom_to_fragment.max()) + 1
    assert source_shape == (num_fragments, model.hidden_dim)
    assert target_shape == (batch.num_graphs, model.hidden_dim)
    assert size == (num_fragments, batch.num_graphs)


def test_himnet_exposes_source_backed_hierarchy_and_fusion_invariants() -> None:
    samples = [
        add_himnet_inputs(
            featurize_smiles(smiles, targets=[0.0], target_mask=[True], sample_id=index)
        )
        for index, smiles in enumerate(("CC(=O)NCC", "C"))
    ]
    batch = next(iter(DataLoader(samples, batch_size=2)))
    model = HimNet(
        atom_dim=153,
        bond_dim=14,
        hidden_dim=8,
        depth=2,
        dropout=0.0,
        interaction_heads=2,
        fusion_heads=2,
    ).eval()

    assert model.directed_encoder.W_alpha.out_features == 1
    assert model.directed_encoder.edge_attention.out_proj.__class__.__name__ == "Identity"
    assert model.interaction_encoder.cross_attention.num_heads == 2
    assert model.feature_fusion.num_heads == 2
    assert model(batch).shape == (2, 1)

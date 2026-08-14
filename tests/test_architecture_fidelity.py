"""Durable regression checks for architecture invariants and runtime contracts."""

import torch
from torch_geometric.loader import DataLoader

from molgnn.featurizer import featurize_smiles
from molgnn.models.dimenet_2020 import DimeNet2020
from molgnn.models.hignn_2023 import HiGNN
from molgnn.models.himnet_2026 import HimNet
from molgnn.models.mpnn_2017 import MPNN, MPNNDistanceBins3D
from molgnn.models.potentialnet_2018 import PotentialNet
from molgnn.models.weave_2016 import Weave
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
        "ampnn",
        "gcn_baseline",
        "attentivefp",
        "dimenet",
        "dmpnn",
        "emnn",
        "gpspp",
        "hignn",
        "himnet",
        "molecular_graph_embedding",
        "mpnn",
        "mpnn_3d_distance_bins",
        "mvgnn_cross",
        "potentialnet",
        "trimnet_2020",
        "weave",
        "egnn",
        "mat",
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
    assert (
        model.directed_encoder.edge_attention.out_proj.__class__.__name__ == "Identity"
    )
    assert model.interaction_encoder.cross_attention.num_heads == 2
    assert model.feature_fusion.num_heads == 2
    assert model(batch).shape == (2, 1)


def test_mpnn_exposes_tied_typed_message_and_custom_gru_topology() -> None:
    model = MPNN(
        atom_dim=3,
        hidden_dim=4,
        num_message_passing_steps=2,
        readout_hidden_dim=4,
        readout_num_hidden_layers=1,
    )

    assert model.message_function.incoming_weights.shape == (4, 4, 4)
    assert model.message_function.outgoing_weights.shape == (4, 4, 4)
    assert model.message_function.message_bias.shape == (8,)
    assert model.update_function.message_update.bias is None
    assert model.update_function.state_candidate.bias is None
    assert model.graph_readout.gate_network.hidden_layers[0].in_features == 7
    assert model.graph_readout.value_network.hidden_layers[0].in_features == 7


def test_mpnn_distance_bin_variant_keeps_the_tied_mpnn_topology() -> None:
    model = MPNNDistanceBins3D(
        atom_dim=3,
        hidden_dim=4,
        num_message_passing_steps=2,
        readout_hidden_dim=4,
        readout_num_hidden_layers=1,
    )

    assert model.num_edge_types == 14
    assert model.message_function.incoming_weights.shape == (14, 4, 4)
    assert model.message_function.outgoing_weights.shape == (14, 4, 4)


def test_mpnn_preserves_legacy_directional_messages_and_gru_update() -> None:
    model = MPNN(
        atom_dim=1,
        hidden_dim=1,
        num_edge_types=2,
        num_message_passing_steps=1,
        readout_hidden_dim=1,
        readout_num_hidden_layers=1,
    )
    message = model.message_function
    update = model.update_function
    with torch.no_grad():
        message.incoming_weights.copy_(torch.tensor([[[2.0]], [[-1.0]]]))
        message.outgoing_weights.copy_(torch.tensor([[[0.5]], [[3.0]]]))
        message.message_bias.copy_(torch.tensor([0.1, 0.2]))
        update.message_update.weight.copy_(torch.tensor([[0.2, -0.1]]))
        update.state_update.weight.copy_(torch.tensor([[0.3]]))
        update.message_reset.weight.copy_(torch.tensor([[-0.4, 0.1]]))
        update.state_reset.weight.copy_(torch.tensor([[0.2]]))
        update.message_candidate.weight.copy_(torch.tensor([[0.5, 0.6]]))
        update.state_candidate.weight.copy_(torch.tensor([[-0.7]]))

    hidden = torch.tensor([[1.0], [4.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_type = torch.tensor([0, 1])
    messages = message(hidden, edge_index, edge_type)

    assert torch.allclose(
        messages,
        torch.tensor([[-3.9, 2.2], [2.1, 3.2]]),
        atol=1e-6,
    )
    gate = torch.sigmoid(
        messages @ update.message_update.weight.T
        + hidden @ update.state_update.weight.T
    )
    reset = torch.sigmoid(
        messages @ update.message_reset.weight.T + hidden @ update.state_reset.weight.T
    )
    candidate = torch.tanh(
        messages @ update.message_candidate.weight.T
        + (reset * hidden) @ update.state_candidate.weight.T
    )
    expected = (1 - gate) * hidden + gate * candidate

    assert torch.allclose(update(hidden, messages), expected, atol=1e-6)


def test_weave_exposes_coupled_atom_pair_modules_and_fixed_histogram_readout() -> None:
    model = Weave(
        atom_dim=3,
        pair_dim=2,
        hidden_dim=4,
        num_weave_modules=2,
        graph_feature_dim=5,
        predictor_hidden_dims=(6,),
        dropout=0.0,
    )
    first = model.weave_modules[0]

    assert len(model.weave_modules) == 2
    assert first.atom_to_atom.in_features == 3
    assert first.pair_to_atom.in_features == 2
    assert first.atom_to_pair.in_features == 6
    assert first.update_atom.in_features == 8
    assert first.update_pair.in_features == 8
    assert model.histogram_readout.gaussian_bins == 11
    assert model.histogram_readout.output_dim == 55


def test_potentialnet_keeps_two_tied_stages_and_ligand_only_readout() -> None:
    model = PotentialNet(
        atom_dim=44,
        bond_hidden_dim=48,
        spatial_hidden_dim=48,
        gather_dim=48,
        num_bond_steps=2,
        num_spatial_steps=3,
        readout_hidden_dims=(16,),
    )

    assert len(model.stage1.message_network.networks) == 5
    assert len(model.stage2.message_network.networks) == 9
    assert model.stage1.num_steps == 2
    assert model.stage2.num_steps == 3
    assert model.stage1.gate.gate_network.in_features == 44 + 48
    assert model.stage2.gate.gate_network.in_features == 48 + 48
    assert model.readout.input_dim == 48


def test_dimenet_keeps_directed_edge_states_and_independent_block_stacks() -> None:
    model = DimeNet2020(
        hidden_dim=8,
        num_blocks=2,
        num_bilinear=3,
        num_spherical=2,
        num_radial=2,
        num_before_skip=1,
        num_after_skip=1,
        num_dense_output=1,
        num_targets=1,
    )

    assert model.required_batch_fields == (
        "atomic_number",
        "pos",
        "dimenet_edge_index",
        "dimenet_triplet_edge_index",
        "batch",
    )
    assert model.cutoff == 5.0
    assert model.envelope_p == 6
    assert len(model.interaction_blocks) == 2
    assert len(model.output_blocks) == 3
    assert model.interaction_blocks[0] is not model.interaction_blocks[1]
    assert model.interaction_blocks[0].bilinear.shape == (8, 3, 8)
    assert model.output_blocks[0].output_projection.bias is None

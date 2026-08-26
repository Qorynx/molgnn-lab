"""Durable regression checks for architecture invariants and runtime contracts."""

import math
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData
from torch_geometric.loader import DataLoader

from molgnn.data import MolecularData
from molgnn.featurizer import featurize_smiles
from molgnn.models.dimenet_2020 import DimeNet2020
from molgnn.models.chemrl_gem_2022 import ChemRLGEMEncoder
from molgnn.models.chemrl_gem_2022.checkpoint import load_chemrl_gem_encoder
from molgnn.models.hignn_2023 import HiGNN
from molgnn.models.himnet_2026 import HimNet
from molgnn.models.kpgt_2022 import KPGT, KPGTVocab, LiGhTEncoder
from molgnn.models.three_d_infomax_2022 import (
    Net3D,
    NTXentMultiplePositives,
    ThreeDInfomax,
    multi_positive_infomax_loss,
)
from molgnn.models.three_d_infomax_2022.checkpoint import (
    CheckpointError as ThreeDInfomaxCheckpointError,
    convert_official_checkpoint,
    load_pretrained_encoder as load_three_d_infomax_encoder,
)
from molgnn.models.three_d_infomax_2022.layers import PNALayer
from molgnn.models.kpgt_2022.checkpoint import (
    CheckpointError,
    infer_checkpoint_profile,
    load_pretrained_backbone,
)
from molgnn.models.kpgt_2022.descriptors import RDKIT_2D_DESCRIPTOR_NAMES
from molgnn.models.kpgt_2022.layers import destination_softmax
from molgnn.models.kpgt_2022.pretraining import (
    KPGTPretrainer,
    compute_pretraining_losses,
    corrupt_pretraining_batch,
    disturb_descriptor,
    disturb_fingerprint,
    mask_line_nodes,
    resume_pretraining_checkpoint,
    save_pretraining_checkpoint,
    train_pretraining_epoch,
)
from molgnn.models.mpnn_2017 import MPNN, MPNNDistanceBins3D
from molgnn.models.potentialnet_2018 import PotentialNet
from molgnn.models.registration import register_builtin_models
from molgnn.models.weave_2016 import Weave
from molgnn.registry import available_models, get_model_spec
from molgnn.transforms import add_brics_fragments, add_himnet_inputs, add_kpgt_inputs


def test_chemrl_gem_encoder_has_official_tensor_schema() -> None:
    encoder = ChemRLGEMEncoder()
    state = encoder.state_dict()
    assert len(state) == 164
    assert state["init_atom_embedding.embed_list.0.weight"].shape == (124, 32)
    assert state["init_atom_embedding.embed_list.1.weight"].shape == (22, 32)
    assert state["init_bond_embedding.embed_list.1.weight"].shape == (27, 32)
    assert state["bond_angle_float_rbf_list.0.linear_list.0.weight"].shape == (32, 32)


def test_chemrl_gem_official_checkpoints_load_strictly() -> None:
    root = Path(__file__).resolve().parents[1]
    checkpoint_root = root / "pretrained" / "chemrl_gem_2022" / "pretrain_models-chemrl_gem"
    for filename in ("class.pdparams", "regr.pdparams"):
        encoder = ChemRLGEMEncoder()
        info = load_chemrl_gem_encoder(encoder, checkpoint_root / filename)
        assert info["tensor_count"] == 164


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
        "kpgt",
        "molecular_graph_embedding",
        "mpnn",
        "mpnn_3d_distance_bins",
        "mvgnn_cross",
        "potentialnet",
        "trimnet_2020",
        "weave",
        "egnn",
        "mat",
        "molclr_gin",
        "molclr_gcn",
        "molebert",
        "graphmvp",
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


# --- KPGT (KDD 2022) ---------------------------------------------------------


def test_kpgt_registers_official_runtime_contract() -> None:
    register_builtin_models()
    spec = get_model_spec("kpgt")

    assert spec.geometry_requirement == "none"
    assert spec.geometry_role == "none"
    assert spec.graph_transform_name == "kpgt_inputs"
    assert spec.prediction_reducer_name == "identity"
    defaults = dict(spec.default_parameters)
    assert defaults["d_g_feats"] == 768
    assert defaults["n_mol_layers"] == 12
    assert defaults["n_heads"] == 12
    assert defaults["d_hpath_ratio"] == 12
    assert defaults["path_length"] == 5
    assert defaults["attn_drop"] == 0.1
    assert defaults["feat_drop"] == 0.1
    assert defaults["predictor_hidden_dim"] == 256
    # Scratch initialization stays the default.
    assert defaults["pretrained_checkpoint"] is None


def test_kpgt_vocab_matches_official_layout() -> None:
    vocab = KPGTVocab()
    assert vocab.vocab_size == 25857

    def official_id(atom_1: int, bond: int, atom_2: int) -> int:
        start = sum(5 * (101 - k) for k in range(min(atom_1, atom_2)))
        low, high = sorted((atom_1, atom_2))
        return start + bond * (101 - low) + (high - low)

    assert vocab.index(6, 6, 0) == official_id(6, 0, 6)
    for atom_1 in (0, 5, 50, 100):
        for bond in range(5):
            for offset in (0, 7):
                atom_2 = min(100, atom_1 + offset)
                assert vocab.index(atom_1, atom_2, bond) == official_id(
                    atom_1, bond, atom_2
                )
    assert vocab.index(6, 999, 999) == 25755 + 6
    assert vocab.index(999, 999, 999) == 25856
    assert vocab.index(200, 200, 3) == 25857


def _kpgt_sample(smiles: str, sample_id: int = 0):
    return add_kpgt_inputs(
        featurize_smiles(smiles, targets=[0.0], target_mask=[True], sample_id=sample_id)
    )


def test_kpgt_features_follow_official_layout_and_knowledge() -> None:
    sample = _kpgt_sample("CCO")
    indicators = sample.kpgt_node_indicator.tolist()
    # One line-node per bond plus two knowledge nodes; ethanol has 2 bonds.
    assert indicators == [0, 0, 1, 2]
    assert sample.kpgt_begin_end.shape == (4, 2, 137)
    assert sample.kpgt_bond_attr.shape == (4, 14)

    begin_end = sample.kpgt_begin_end
    knowledge = sample.kpgt_triplet_label[-2:].tolist()
    assert knowledge == [25856, 25856]
    carbon_block = begin_end[0, 0, :101]
    assert int(carbon_block.argmax()) == 5 and float(carbon_block.sum()) == 1.0

    # Canonical renumbering may permute atoms; locate elements by their blocks.
    atomic_blocks = begin_end[:, :, :101]
    masses = begin_end[:, :, -1]
    oxygen_slots = atomic_blocks.argmax(dim=-1) == 7
    assert bool(oxygen_slots.any())
    assert torch.allclose(
        masses[oxygen_slots],
        torch.full_like(masses[oxygen_slots], 15.999 * 0.01),
        atol=1e-4,
    )
    carbon_slots = atomic_blocks.argmax(dim=-1) == 5
    assert torch.allclose(
        masses[carbon_slots],
        torch.full_like(masses[carbon_slots], 12.011 * 0.01),
        atol=1e-4,
    )
    chirality = begin_end[0, 0, -3:-1].tolist()
    assert chirality == [False, False]

    bond_attr = sample.kpgt_bond_attr[0]
    assert int(bond_attr[:5].argmax()) == 0
    assert bond_attr.shape[0] == 14

    fingerprint = sample.kpgt_fingerprint
    assert fingerprint.shape == (1, 512)
    assert bool(((fingerprint == 0) | (fingerprint == 1)).all())
    descriptor = sample.kpgt_descriptor
    assert descriptor.shape == (1, 200)
    assert len(RDKIT_2D_DESCRIPTOR_NAMES) == 200
    assert bool(torch.isfinite(descriptor).all())
    assert float(descriptor.min()) >= 0.0 and float(descriptor.max()) <= 1.0


def test_kpgt_isolated_atom_node_uses_virtual_slots() -> None:
    sample = _kpgt_sample("C")
    indicators = sample.kpgt_node_indicator.tolist()
    assert indicators == [-1, 1, 2]
    begin_end = sample.kpgt_begin_end
    isolated = begin_end[0]
    assert float((isolated[0][:101] == 0).sum()) == 100
    assert int(isolated[0][:101].argmax()) == 5
    # Second atom slot is the official placeholder vector of -1 values.
    assert torch.equal(isolated[1], torch.full((137,), -1.0))
    assert torch.equal(sample.kpgt_bond_attr[0], torch.full((14,), -1.0))


def test_kpgt_distance_bias_keeps_official_overrides_and_symmetry() -> None:
    encoder = LiGhTEncoder(d_g_feats=16, d_hpath_ratio=4, n_mol_layers=1, n_heads=4)
    encoder.eval()
    sentinel = -1_000_000
    path_index = torch.tensor(
        [
            [0, sentinel, sentinel, sentinel, 1],
            [2, sentinel, sentinel, sentinel, 0],
            [3, sentinel, sentinel, sentinel, 0],
            [4, sentinel, sentinel, sentinel, 0],
        ]
    )
    virtual_path = torch.tensor([False, True, True, False])
    self_loop = torch.tensor([False, False, False, True])

    with torch.no_grad():
        feats = encoder.featurize_path(path_index, virtual_path, self_loop)
    length_two = encoder.path_len_emb(torch.tensor(2))
    virtual = encoder.virtual_path_emb.weight[0]
    loop = encoder.self_loop_emb.weight[0]
    # Line-graph edge keeps the generic path-length bias.
    assert torch.allclose(feats[0], length_two)
    # OFFICIAL CODE stores vp as a BoolTensor, so both the fingerprint and the
    # descriptor edges collapse to the special virtual embedding (the audit
    # plan's asymmetry note does not match either source revision).
    assert torch.allclose(feats[1], virtual)
    assert torch.allclose(feats[2], virtual)
    # Self-loops win over every other override.
    assert torch.allclose(feats[3], loop)


def test_kpgt_attention_messages_flow_to_destinations() -> None:
    torch.manual_seed(7)
    layer = LiGhTEncoder(d_g_feats=16, d_hpath_ratio=4, n_mol_layers=1, n_heads=4)
    layer.eval()

    scores = torch.tensor([[math.log(1.0)], [math.log(3.0)]])
    weights = destination_softmax(scores, torch.tensor([0, 0]))
    expected = torch.softmax(scores.flatten(), dim=0)
    assert torch.allclose(weights.flatten(), expected)

    # Replicate the official message computation with zeroed queries/keys so
    # attention is uniform per destination; values then carry source states.
    triplet_h = torch.zeros(3, 16)
    triplet_h[0, 7] = 1.0
    triplet_h[1, 7] = 3.0
    edge_index = torch.tensor([[0, 1], [2, 2]])
    block = layer.mol_T_layers[0]
    with torch.no_grad():
        qkv = (
            block.qkv(block.attention_norm(triplet_h))
            .reshape(-1, 3, 4, 4)
            .permute(1, 0, 2, 3)
        )
        value = qkv[2]
        query = torch.zeros_like(qkv[0])
        key = torch.zeros_like(qkv[1])
        attention = (query[edge_index[0]] * key[edge_index[1]]).sum(dim=-1)
        softmax = destination_softmax(attention, edge_index[1])
        assert torch.allclose(softmax, torch.full((2, 4), 0.5))

        messages = value[edge_index[0]].view(-1, 4, 4)
        messages = (messages * softmax.unsqueeze(-1)).view(-1, 16)
        aggregated = torch.zeros_like(triplet_h)
        aggregated.index_add_(0, edge_index[1], messages)

    # Node 2 receives the mean of both sources; sources stay untouched.
    expected_second = ((value[0] + value[1]) / 2.0).view(-1)
    assert torch.allclose(aggregated[2], expected_second)
    assert torch.allclose(aggregated[0], torch.zeros(16))
    assert torch.allclose(aggregated[1], torch.zeros(16))


def _tiny_pretrainer() -> KPGTPretrainer:
    return KPGTPretrainer(
        d_g_feats=16,
        d_hpath_ratio=4,
        n_mol_layers=1,
        path_length=5,
        n_heads=4,
        attn_drop=0.0,
        feat_drop=0.0,
        predictor_hidden_dim=8,
    )


def test_kpgt_corruption_matches_official_semantics() -> None:
    generator = torch.Generator().manual_seed(22)
    indicators = torch.tensor([0, 0, 0, -1, 0, 1, 2])
    labels = torch.tensor([10, 11, 11, 42, 13, 99, 99])
    mask, sl_labels, _, _, _ = mask_line_nodes(
        indicators, labels, candi_rate=0.6, generator=generator
    )

    knowledge_mask = mask[[5, 6]]
    assert knowledge_mask.tolist() == [0, 0]
    valid_positions = {0, 1, 2, 3, 4}
    selected = {int(index) for index in torch.nonzero(mask >= 1).flatten()}
    assert selected <= valid_positions
    expected_candidates = int(len(valid_positions) * 0.6)
    assert len(selected) >= 1
    assert len(selected) <= expected_candidates
    assert torch.equal(sl_labels, labels[mask >= 1])
    assert set(mask[mask >= 1].tolist()) <= {1, 2, 3}

    fingerprint = torch.zeros(2, 512)
    disturbed = disturb_fingerprint(fingerprint, 0.5, generator)
    flipped = int((disturbed != fingerprint).sum())
    assert flipped == int(2 * 512 * 0.5)
    assert bool(((disturbed == 0) | (disturbed == 1)).all())

    descriptor = torch.zeros(2, 200)
    changed = disturb_descriptor(descriptor, 0.5, generator)
    changed_positions = changed != descriptor
    assert int(changed_positions.sum()) == int(2 * 200 * 0.5)
    assert float(changed[changed_positions].min()) >= 0.0
    assert float(changed[changed_positions].max()) <= 1.0


def test_kpgt_random_replacement_copies_a_different_token_node() -> None:
    node_count = 100
    indicators = torch.zeros(node_count, dtype=torch.long)
    labels = torch.arange(node_count, dtype=torch.long) + 1000
    begin_end = torch.zeros(node_count, 2, 137)
    bond_attr = torch.zeros(node_count, 14)
    begin_end[:, 0, 0] = torch.arange(node_count)
    bond_attr[:, 0] = torch.arange(node_count)

    mask, _, replaced_begin_end, replaced_bond_attr, _ = mask_line_nodes(
        indicators,
        labels,
        generator=torch.Generator().manual_seed(7),
        begin_end=begin_end,
        bond_attr=bond_attr,
    )

    replace_ids = torch.nonzero(mask == 2, as_tuple=False).flatten()
    assert len(replace_ids) > 0
    for replace_id in replace_ids.tolist():
        source_id = int(replaced_begin_end[replace_id, 0, 0].item())
        assert labels[source_id] != labels[replace_id]
        assert replaced_bond_attr[replace_id, 0] == source_id


def test_kpgt_pretrainer_consumes_disturbed_knowledge_inputs() -> None:
    batch = Batch.from_data_list([_kpgt_sample("CCO")])
    pretrainer = _tiny_pretrainer().eval()
    fields = {
        name: getattr(batch, name) for name in pretrainer.backbone.required_batch_fields
    }
    mask = torch.zeros_like(fields["kpgt_node_indicator"])

    with torch.no_grad():
        zero_outputs = pretrainer(
            fields,
            torch.zeros_like(fields["kpgt_fingerprint"]),
            torch.zeros_like(fields["kpgt_descriptor"]),
            mask,
        )
        one_outputs = pretrainer(
            fields,
            torch.ones_like(fields["kpgt_fingerprint"]),
            torch.ones_like(fields["kpgt_descriptor"]),
            mask,
        )

    assert any(
        not torch.equal(zero, one)
        for zero, one in zip(zero_outputs, one_outputs)
        if zero.numel() > 0
    )


def test_kpgt_pretraining_losses_and_tiny_epoch_resume(tmp_path) -> None:
    samples = [_kpgt_sample(value, index) for index, value in enumerate(("CC", "C", "CCN"))]
    batch = Batch.from_data_list(samples)
    pretrainer = _tiny_pretrainer()
    corrupted = corrupt_pretraining_batch(batch, generator=torch.Generator().manual_seed(5))

    fields = {
        name: getattr(batch, name) for name in pretrainer.backbone.required_batch_fields
    }
    fields["kpgt_begin_end"] = corrupted["begin_end"]
    fields["kpgt_bond_attr"] = corrupted["bond_attr"]
    fields["kpgt_node_indicator"] = corrupted["indicators"]
    sl_predictions, fp_predictions, md_predictions = pretrainer(
        fields, corrupted["fingerprint"], corrupted["descriptor"], corrupted["mask"]
    )
    losses = compute_pretraining_losses(
        sl_predictions,
        fp_predictions,
        md_predictions,
        corrupted["sl_labels"],
        corrupted["target_fingerprint"],
        corrupted["target_descriptor"],
    )
    manual = (
        torch.nn.functional.cross_entropy(sl_predictions, corrupted["sl_labels"])
        + torch.nn.functional.binary_cross_entropy_with_logits(
            fp_predictions, corrupted["target_fingerprint"]
        )
        + torch.nn.functional.mse_loss(md_predictions, corrupted["target_descriptor"])
    ) / 3
    assert torch.allclose(losses["loss"], manual)

    optimizer = torch.optim.Adam(pretrainer.parameters(), lr=1e-3)
    stats = train_pretraining_epoch(
        pretrainer, [batch], optimizer, device="cpu", generator=torch.Generator().manual_seed(1)
    )
    assert all(0 <= value < float("inf") and not math.isnan(value) for value in stats.values())

    checkpoint_path = save_pretraining_checkpoint(
        tmp_path / "pretrain.pth", pretrainer, optimizer=optimizer, step=9
    )
    restored = _tiny_pretrainer()
    info = resume_pretraining_checkpoint(checkpoint_path, restored)
    assert info["step"] == 9
    for original, copy in zip(
        pretrainer.state_dict().values(), restored.state_dict().values()
    ):
        assert torch.equal(original, copy)


def test_kpgt_checkpoint_loader_validates_profile_shapes_and_heads() -> None:
    pretrainer = _tiny_pretrainer()
    official_state = {
        key: value
        for key, value in pretrainer.backbone.state_dict().items()
        if not key.startswith(("node_predictor.", "fp_predictor.", "md_predictor.", "predictor."))
    }
    profile = infer_checkpoint_profile(official_state)
    assert profile["d_g_feats"] == 16
    assert profile["d_hpath_ratio"] == 4
    assert profile["path_length"] == 5
    assert profile["d_fp_feats"] == 512
    assert profile["d_md_feats"] == 200

    downstream = KPGT(
        num_targets=2,
        d_g_feats=16,
        d_hpath_ratio=4,
        n_mol_layers=1,
        path_length=5,
        n_heads=4,
        attn_drop=0.0,
        feat_drop=0.0,
        predictor_hidden_dim=8,
    )
    checkpoint_path = Path(__file__).parent / "_kpgt_base_test_fixture.pth"
    try:
        torch.save(
            {"model_state_dict": {f"module.{k}": v for k, v in official_state.items()}},
            checkpoint_path,
        )
        info = load_pretrained_backbone(downstream, checkpoint_path)
        assert info["loaded_tensors"] == len(official_state)
        assert info["skipped_head_tensors"] == 0
        for key, value in downstream.model.state_dict().items():
            assert torch.equal(value, pretrainer.backbone.model.state_dict()[key])

        mismatched = KPGT(
            num_targets=1,
            d_g_feats=32,
            d_hpath_ratio=4,
            n_mol_layers=1,
            path_length=5,
            n_heads=4,
            attn_drop=0.0,
            feat_drop=0.0,
            predictor_hidden_dim=8,
        )
        with pytest.raises(CheckpointError, match="profile does not match"):
            load_pretrained_backbone(mismatched, checkpoint_path)
        with pytest.raises(CheckpointError, match="does not exist"):
            load_pretrained_backbone(downstream, checkpoint_path.with_suffix(".missing"))
    finally:
        checkpoint_path.unlink(missing_ok=True)


# --- 3D Infomax (ICML 2022) --------------------------------------------------


def test_three_d_infomax_registers_topology_only_contract() -> None:
    register_builtin_models()
    spec = get_model_spec("3d_infomax")

    assert spec.geometry_requirement == "none"
    assert spec.geometry_role == "topology_2d"
    assert spec.graph_transform_name == "three_d_infomax_inputs"
    assert spec.prediction_reducer_name == "identity"
    defaults = dict(spec.default_parameters)
    assert defaults["aggregators"] == ("mean", "max", "min", "std")
    assert defaults["scalers"] == ("identity", "amplification", "attenuation")
    assert defaults["readout_aggregators"] == ("min", "max", "mean")
    assert defaults["pretrained_checkpoint"] is None


def test_pna_layer_matches_dgl_degree_bucket_reference() -> None:
    """Independent DGL degree-bucket reference with mixed in-degrees."""

    torch.manual_seed(9)
    hidden = 4
    layer = PNALayer(
        in_dim=hidden,
        out_dim=hidden,
        in_dim_edges=hidden,
        aggregators=["mean", "max", "min", "std"],
        scalers=["identity", "amplification", "attenuation"],
        pretrans_layers=1,
        posttrans_layers=1,
        mid_batch_norm=False,
        last_batch_norm=False,
        residual=False,
    )
    layer.eval()
    x = torch.randn(4, hidden)
    edge_attr = torch.randn(3, hidden)
    edge_index = torch.tensor([[0, 1, 2], [2, 2, 3]])  # degrees: 0, 0, 2, 1

    with torch.no_grad():
        messages = layer.pretrans(
            torch.cat([x[edge_index[0]], x[edge_index[1]], edge_attr], dim=-1)
        )
        node_two = messages[:2]
        mean_two = node_two.mean(dim=0)
        var_two = torch.relu(node_two.square().mean(dim=0) - mean_two.square())
        aggregate = torch.zeros((4, 4 * hidden))
        aggregate[2] = torch.cat(
            (mean_two, node_two.max(dim=0).values, node_two.min(dim=0).values, torch.sqrt(var_two + 1e-5))
        )
        aggregate[3] = torch.cat(
            (messages[2], messages[2], messages[2], torch.full((hidden,), math.sqrt(1e-5)))
        )
        degree = torch.tensor([0.0, 0.0, 2.0, 1.0]).unsqueeze(-1)
        log_degree = torch.log1p(degree)
        attenuation = torch.where(
            log_degree > 0,
            aggregate / log_degree.clamp_min(torch.finfo(aggregate.dtype).tiny),
            torch.zeros_like(aggregate),
        )
        expected_nodes_block = torch.cat((aggregate, aggregate * log_degree, attenuation), dim=-1)
        # OFFICIAL CODE concatenates [h_in, aggregated].
        expected_input = torch.cat([x, expected_nodes_block], dim=-1)
        expected_nodes = layer.posttrans(expected_input)

        actual = layer(x, edge_index, edge_attr)

    assert torch.allclose(actual, expected_nodes, atol=1e-6)


def test_pna_segments_use_actual_degree_and_zero_only_empty_nodes() -> None:
    from molgnn.models.three_d_infomax_2022.layers import (
        _segment_max,
        _segment_min,
        _segment_statistics,
    )

    messages = torch.tensor([[2.0, -2.0], [4.0, -4.0], [6.0, -6.0]])
    dst = torch.tensor([0, 1, 1])  # degrees 1, 2, 0
    mean, var, degree = _segment_statistics(messages, dst, num_nodes=3)
    assert torch.equal(degree, torch.tensor([1, 2, 0]))
    assert torch.equal(mean, torch.tensor([[2.0, -2.0], [5.0, -5.0], [0.0, 0.0]]))
    assert torch.equal(var, torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]))
    assert torch.equal(
        _segment_min(messages, dst, 3),
        torch.tensor([[2.0, -2.0], [4.0, -6.0], [0.0, 0.0]]),
    )
    assert torch.equal(
        _segment_max(messages, dst, 3),
        torch.tensor([[2.0, -2.0], [6.0, -4.0], [0.0, 0.0]]),
    )


def test_three_d_infomax_readout_std_uses_mean_of_squares() -> None:
    from molgnn.models.three_d_infomax_2022.model import _graph_segment_statistics

    statistics = _graph_segment_statistics(
        torch.tensor([[1.0], [3.0]]),
        torch.tensor([0, 0]),
        num_graphs=1,
    )
    assert torch.allclose(statistics["std"], torch.tensor([[math.sqrt(1.0 + 1e-5)]]))


def test_three_d_infomax_pretrainer_defaults_to_official_pna_profile() -> None:
    from molgnn.models.three_d_infomax_2022.pretraining import ThreeDInfomaxPretrainer

    pretrainer = ThreeDInfomaxPretrainer()
    assert len(pretrainer.pna.node_gnn.mp_layers) == 7
    first_layer = pretrainer.pna.node_gnn.mp_layers[0]
    assert first_layer.aggregator_names == ["mean", "max", "min", "std"]
    assert first_layer.scaler_names == ["identity", "amplification", "attenuation"]
    assert len(first_layer.pretrans.fully_connected) == 2
    assert first_layer.pretrans.fully_connected[0].batch_norm.momentum == pytest.approx(0.93)


def test_net3d_invariants_and_chemistry_free_inputs() -> None:
    torch.manual_seed(4)
    net3d = Net3D(hidden_dim=20, target_dim=256)
    net3d.eval()

    positions = torch.randn(6, 3)
    arange = torch.arange(6)
    sources = torch.repeat_interleave(arange, 5)
    destinations = torch.cat(
        [torch.cat([arange[:index], arange[index + 1 :]]) for index in range(6)]
    )
    edges = torch.stack((sources, destinations), dim=0)
    node_batch = torch.zeros(6, dtype=torch.long)

    with torch.no_grad():
        reference = net3d(positions, edges, node_batch, 1)
        rotation = torch.tensor(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        reflection = torch.diag(torch.tensor([1.0, 1.0, -1.0]))
        moved = net3d(positions + 7.0, edges, node_batch, 1)
        rotated = net3d(positions @ rotation.T, edges, node_batch, 1)
        reflected = net3d(positions @ reflection.T, edges, node_batch, 1)
        permutation = torch.tensor([2, 0, 5, 1, 4, 3])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(6)
        permuted = net3d(
            positions[permutation],
            inverse[edges],
            node_batch[permutation],
            1,
        )

    for name, output in (
        ("translation", moved),
        ("rotation", rotated),
        ("reflection", reflected),
    ):
        assert torch.allclose(reference, output, atol=1e-5), name
    assert (reference - permuted).abs().max() < 1e-4

    with pytest.raises(ValueError, match="must not consume atom identity"):
        Net3D(use_node_features=True)


def test_infomax_loss_matches_official_denominator_on_fixed_tensors() -> None:
    generator = torch.Generator().manual_seed(21)
    batch_size, conformers, metric_dim = 4, 3, 8
    z1 = torch.randn(batch_size, metric_dim, dtype=torch.float64, generator=generator)
    z2 = torch.randn(batch_size * conformers, metric_dim, dtype=torch.float64, generator=generator)
    owner = torch.arange(batch_size).repeat_interleave(conformers)
    tau = 0.1

    official = NTXentMultiplePositives(tau=tau)(z1, z2)
    stable = multi_positive_infomax_loss(z1, z2, owner, tau=tau)
    assert torch.allclose(official, stable, atol=1e-10)

    # Independent hand-derived reference of the source formula.
    z2_viewed = z2.view(batch_size, conformers, metric_dim)
    sims = torch.einsum("ik,juk->iju", z1, z2_viewed)
    sims = sims / torch.einsum("i,ju->iju", z1.norm(dim=1), z2_viewed.norm(dim=2))
    exp_matrix = torch.exp(sims / tau).sum(dim=2)
    pos = torch.diagonal(exp_matrix)
    reference = (-torch.log(pos / (exp_matrix.sum(dim=1) - pos))).mean()
    assert torch.allclose(official, reference, atol=1e-12)

    single = NTXentMultiplePositives(tau=tau)(z1, z2_viewed[:, 0, :])
    stable_single = multi_positive_infomax_loss(
        z1,
        z2.view(batch_size, conformers, metric_dim)[:, 0, :],
        torch.arange(batch_size),
        tau=tau,
    )
    assert torch.allclose(single, stable_single, atol=1e-12)

    with pytest.raises(ValueError, match="at least two molecules"):
        multi_positive_infomax_loss(z1[:1], z2[:conformers], owner[:conformers], tau=tau)


def test_three_d_infomax_checkpoint_roundtrip_and_strict_loading(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    original = root / "pretrained" / "three_d_infomax_2022" / "best_checkpoint_35epochs.pt"
    converted = root / "pretrained" / "three_d_infomax_2022" / "pna_encoder_qmugs.pt"
    if not original.is_file() or not converted.is_file():
        pytest.skip("pinned 3D Infomax artifacts are not present")

    encoder_state = convert_official_checkpoint(original)
    assert len(encoder_state) == 159
    assert "atom_encoder.atom_embedding_list.0.weight" in encoder_state
    assert not any(key.startswith("output.") for key in encoder_state)

    model = ThreeDInfomax(
        num_targets=1,
        hidden_dim=200,
        propagation_depth=7,
        pretrans_layers=2,
        posttrans_layers=1,
        mid_batch_norm=True,
        last_batch_norm=True,
        batch_norm_momentum=0.93,
        readout_hidden_dim=200,
    )
    info = load_three_d_infomax_encoder(model, converted)
    assert info["loaded_tensors"] == 159
    state = model.node_gnn.state_dict()
    assert all(torch.equal(state[key], value) for key, value in encoder_state.items())

    mismatched = ThreeDInfomax(
        num_targets=1,
        hidden_dim=64,
        propagation_depth=2,
        mid_batch_norm=False,
        last_batch_norm=False,
        readout_batchnorm=False,
    )
    with pytest.raises(ThreeDInfomaxCheckpointError):
        load_three_d_infomax_encoder(mismatched, converted)

    # Encoder-only export/import: a scratch state saved in the converted
    # artifact format must reload strictly.
    scratch = ThreeDInfomax(num_targets=2, hidden_dim=16, propagation_depth=1, readout_batchnorm=False)
    exported_path = tmp_path / "scratch_encoder.pt"
    torch.save(
        {
            "format_version": 1,
            "source_sha256": "scratch",
            "scope": "three_d_infomax_pna_encoder",
            "encoder_state": scratch.node_gnn.state_dict(),
        },
        exported_path,
    )
    fresh = ThreeDInfomax(num_targets=1, hidden_dim=16, propagation_depth=1, readout_batchnorm=False)
    load_three_d_infomax_encoder(fresh, exported_path)
    assert torch.equal(
        fresh.node_gnn.mp_layers[0].pretrans.fully_connected[0].linear.weight,
        scratch.node_gnn.mp_layers[0].pretrans.fully_connected[0].linear.weight,
    )

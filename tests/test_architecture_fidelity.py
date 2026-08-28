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
from molgnn.models.chemrl_gem_2022 import ChemRLGEMEncoder
from molgnn.models.chemrl_gem_2022.checkpoint import load_chemrl_gem_encoder
from molgnn.models.dimenet_2020 import DimeNet2020
from molgnn.models.dimenet_pp_2020 import DimeNetPlusPlus2020
from molgnn.models.hignn_2023 import HiGNN
from molgnn.models.himnet_2026 import HimNet
from molgnn.models.kpgt_2022 import KPGT, KPGTVocab, LiGhTEncoder
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
from molgnn.models.three_d_infomax_2022 import (
    Net3D,
    NTXentMultiplePositives,
    ThreeDInfomax,
    multi_positive_infomax_loss,
)
from molgnn.models.three_d_infomax_2022.checkpoint import (
    CheckpointError as ThreeDInfomaxCheckpointError,
)
from molgnn.models.three_d_infomax_2022.checkpoint import (
    convert_official_checkpoint,
)
from molgnn.models.three_d_infomax_2022.checkpoint import (
    load_pretrained_encoder as load_three_d_infomax_encoder,
)
from molgnn.models.three_d_infomax_2022.layers import PNALayer
from molgnn.models.weave_2016 import Weave
from molgnn.registry import available_models, get_model_spec
from molgnn.transforms import (
    add_brics_fragments,
    add_dgt_inputs,
    add_himnet_inputs,
    add_kpgt_inputs,
)
from molgnn.transforms.dgt import pairwise_random_walk_landing_probs


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
        "dgt",
        "dimenet",
        "dimenet_pp",
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
        "pvd_torchmd_et",
        "trimnet_2020",
        "weave",
        "egnn",
        "mat",
        "mgcn",
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


def test_dimenet_pp_keeps_directed_edge_states_and_hadamard_blocks() -> None:
    model = DimeNetPlusPlus2020(
        hidden_dim=8,
        interaction_dim=4,
        basis_dim=2,
        output_dim=8,
        num_blocks=2,
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
    assert not hasattr(model.interaction_blocks[0], "bilinear")
    assert model.interaction_blocks[0].down_projection.weight.shape == (4, 8)
    assert model.interaction_blocks[0].up_projection.weight.shape == (8, 4)
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


# --- DGT / DimeNet++ / SphereNet post-review fixes (2026-08) ------------------


def test_dgt_pairwise_rwse_matches_powers_of_transition_matrix() -> None:
    """All-pairs RWSE must equal P^k[i, j], not diagonal-only endpoints."""

    # Asymmetric one-way path 0 -> 1 -> 2: P is upper-triangular and its
    # off-diagonal entries are exactly the quantities the old diagonal-only
    # representation could never express.
    edges = torch.tensor([[0, 1], [1, 2]])
    flat = pairwise_random_walk_landing_probs(edges, num_nodes=3, steps=3)
    blocks = flat.reshape(3, 3, 3)  # [i, j, k]
    transition = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
    )
    assert torch.allclose(blocks[:, :, 0], transition)
    two_step = transition @ transition
    assert torch.allclose(blocks[:, :, 1], two_step)
    # Off-diagonal element (0, 2) at k=2 exists only in the pairwise view.
    assert abs(blocks[0, 2, 1].item() - 1.0) < 1e-6


def test_dgt_pairwise_rwse_batching_keeps_graphs_separate() -> None:
    samples = [
        add_dgt_inputs(
            featurize_smiles(value, targets=[0.0], target_mask=[True], sample_id=index)
        )
        for index, value in enumerate(("CCO", "C", "c1ccccc1"))
    ]
    batch = Batch.from_data_list(list[BaseData](samples))
    counts = torch.bincount(batch.batch)
    starts = torch.cumsum(counts.square(), dim=0) - counts.square()
    for index, sample in enumerate(samples):
        num_atoms = int(sample.x.shape[0])
        start = int(starts[index])
        block = batch.dgt_rwse[start : start + num_atoms**2]
        assert torch.allclose(
            block.reshape(num_atoms, num_atoms, -1),
            sample.dgt_rwse.reshape(num_atoms, num_atoms, -1),
        )


def test_dgt_transform_preserves_isolated_highest_index_node() -> None:
    """Disconnected SMILES must retain dense SPDE/RWSE dimensions."""

    sample = add_dgt_inputs(
        featurize_smiles("CC.O", targets=[0.0], target_mask=[True], sample_id=7)
    )
    assert sample.x.shape[0] == 3
    assert sample.dgt_rwse.shape == (9, 16)
    assert sample.dgt_spd_index.shape[0] == 2
    # Atom 2 is isolated; its random-walk rows and columns remain present.
    rwse = sample.dgt_rwse.reshape(3, 3, 16)
    assert torch.allclose(rwse[2], torch.zeros_like(rwse[2]))
    assert torch.allclose(rwse[:, 2], torch.zeros_like(rwse[:, 2]))


def _dgt_tiny_model():
    from molgnn.models.dgt_2026 import DGT2026

    return DGT2026(
        atom_dim=153,
        bond_dim=14,
        dim_h=8,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        attn_dropout=0.0,
    )


def test_dgt_train_mode_survives_single_atom_and_single_bond_streams() -> None:
    model = _dgt_tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def step(samples: list[MolecularData]) -> None:
        batch = Batch.from_data_list(list[BaseData](samples))
        prediction = model(batch)
        assert prediction.shape == (len(samples), 1)
        assert bool(torch.isfinite(prediction).all())
        loss = prediction.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients
        assert all(bool(torch.isfinite(g).all()) for g in gradients)
        optimizer.step()

    # Single atom: atom stream has one row; bond stream is empty.
    step([
        add_dgt_inputs(
            featurize_smiles("C", targets=[0.0], target_mask=[True], sample_id=0)
        )
    ])
    # CO: one undirected bond -> bond stream has exactly one row.
    step([
        add_dgt_inputs(
            featurize_smiles("CO", targets=[0.0], target_mask=[True], sample_id=1)
        )
    ])


def test_dgt_attention_dropout_uses_one_shared_connection_mask() -> None:

    from molgnn.models.dgt_2026.layers import DGTAttention

    torch.manual_seed(123)
    attention = DGTAttention(dim_h=4, num_heads=2, attn_dropout=0.5)
    attention.train()
    batch_size, num_nodes = 1, 3
    h = torch.randn(batch_size, num_nodes, 4)
    e_att = torch.randn(batch_size, num_nodes, num_nodes, 4)
    e_val = torch.randn(batch_size, num_nodes, num_nodes, 4)
    mask = torch.ones(batch_size, num_nodes, num_nodes, dtype=torch.bool)

    torch.manual_seed(999)
    got = attention(h, e_att, e_val, mask)

    # Manual reference replicating OFFICIAL CODE dgt_layer.py semantics:
    # dropout applies to the mask itself and broadcasts over head/feature
    # channels, so one connection (i, j) shares a single dropout outcome.
    query = attention.Q(h).view(batch_size, num_nodes, 2, 2)
    key = attention.K(h).view(batch_size, num_nodes, 2, 2)
    values = attention.V(h)
    scaling = float(2) ** -0.5
    scores = torch.einsum("bihk,bjhk->bijh", query, key * scaling).unsqueeze(-1)
    mask_view = mask.view(batch_size, num_nodes, num_nodes, 1, 1)
    scores = scores - 1e24 * (~mask_view)
    scores = scores + e_att.view(batch_size, num_nodes, num_nodes, 2, 2)
    scores = scores.reshape(batch_size, num_nodes, num_nodes, 4)
    scores = torch.softmax(scores, dim=2)
    # Re-seed so the manual replay draws the exact dropout mask consumed
    # inside the module call above.
    torch.manual_seed(999)
    connection = attention.attn_dropout(mask_view.float()).squeeze(-1)
    manual_scores = scores * connection
    manual = torch.einsum("bijk,bjk->bik", manual_scores, values)
    manual = manual + (manual_scores * e_val).sum(2)
    manual = attention.out_proj(manual)
    assert torch.allclose(got, manual, atol=1e-6)


def test_dimenet_pp_rejects_cutoff_mismatched_with_transform() -> None:
    from molgnn.models.dimenet_2020.constants import DIMENET_CUTOFF
    from molgnn.models.dimenet_pp_2020 import DimeNetPlusPlus2020

    kwargs = {
        "hidden_dim": 8,
        "interaction_dim": 8,
        "basis_dim": 4,
        "output_dim": 8,
        "num_blocks": 1,
    }
    with pytest.raises(ValueError, match="DIMENET_CUTOFF"):
        DimeNetPlusPlus2020(cutoff=DIMENET_CUTOFF + 1.5, **kwargs)
    with pytest.raises(ValueError, match="DIMENET_CUTOFF"):
        DimeNetPlusPlus2020(cutoff=4.0, **kwargs)
    model = DimeNetPlusPlus2020(cutoff=DIMENET_CUTOFF, **kwargs)
    assert model.cutoff == pytest.approx(DIMENET_CUTOFF)


def test_spherenet_rejects_cutoff_mismatched_with_transform() -> None:
    from molgnn.models.spherenet_2022 import SphereNet2022
    from molgnn.models.spherenet_2022.constants import SPHERENET_CUTOFF

    kwargs = {
        "hidden_dim": 8,
        "interaction_dim": 8,
        "output_dim": 8,
        "basis_dim_distance": 4,
        "basis_dim_angle": 4,
        "basis_dim_torsion": 4,
        "num_blocks": 1,
        "num_spherical": 2,
        "num_radial": 4,
    }
    with pytest.raises(ValueError, match="SPHERENET_CUTOFF"):
        SphereNet2022(cutoff=SPHERENET_CUTOFF + 1.0, **kwargs)
    with pytest.raises(ValueError, match="SPHERENET_CUTOFF"):
        SphereNet2022(cutoff=4.5, **kwargs)
    model = SphereNet2022(cutoff=SPHERENET_CUTOFF, **kwargs)
    assert model.cutoff == pytest.approx(SPHERENET_CUTOFF)


# --- HiMol (Communications Chemistry 2023) ---------------------------------


def test_himol_motif_decomposition_is_deterministic_and_refines_multi_ring_fragments() -> None:
    from rdkit import Chem

    from molgnn.transforms.himol import decompose_himol_motifs

    biphenyl = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")
    assert decompose_himol_motifs(biphenyl) == (
        (0, 1, 2, 3, 4, 5),
        (6, 7, 8, 9, 10, 11),
    )
    fused_with_linker = Chem.MolFromSmiles("c1ccc2ccccc2c1CC(=O)NCC")
    first = decompose_himol_motifs(fused_with_linker)
    second = decompose_himol_motifs(fused_with_linker)
    assert first == second
    assert (0, 1, 2, 3, 8, 9) in first
    assert (3, 4, 5, 6, 7, 8) in first
    assert (10, 11, 12) in first


# --- MGCN (AAAI 2019) -------------------------------------------------------


def test_mgcn_registers_pure_3d_contract() -> None:
    from molgnn.models.mgcn_2019 import MGCN

    register_builtin_models()
    spec = get_model_spec("mgcn")
    assert spec.required_batch_fields == MGCN.required_batch_fields == (
        "atomic_number",
        "pos",
        "mgcn_edge_index",
        "batch",
    )
    assert spec.graph_transform_name == "mgcn_inputs"
    assert spec.transform_output_fields == (
        "atomic_number",
        "pos",
        "mgcn_edge_index",
    )
    assert spec.geometry_requirement == "required"
    assert spec.geometry_role == "pure_3d"
    assert spec.prediction_reducer_name == "identity"
    assert spec.benchmark_enabled is False


def test_mgcn_pair_embedding_is_symmetric_and_collision_free() -> None:
    from molgnn.models.mgcn_2019.layers import PairEmbedding

    embedding = PairEmbedding(hidden_dim=8, max_atomic_number=118)
    pairs = torch.tensor(
        [(6, 6), (6, 8), (8, 6), (1, 1), (1, 6), (6, 1), (8, 8), (7, 7)]
    )
    source = pairs[:, 0]
    target = pairs[:, 1]
    ids = PairEmbedding.pair_id(source, target)
    # (6, 8) and (8, 6) share one id; no collisions within the range.
    assert ids[1].item() == ids[2].item()
    assert ids[4].item() == ids[5].item()
    assert ids[1].item() != ids[3].item()
    assert torch.unique(ids).numel() == 6

    # Forward embeds directed pairs symmetrically.
    assert torch.allclose(embedding(source, target), embedding(target, source))


def test_mgcn_gaussian_rbf_peaks_at_centers() -> None:
    from molgnn.models.mgcn_2019.layers import GaussianRBF

    rbf = GaussianRBF(num_rbf=5, low=0.0, high=5.0, beta=1.0)
    centers = rbf.centers
    assert torch.allclose(centers, torch.tensor([0.0, 1.25, 2.5, 3.75, 5.0]))
    values = rbf(centers)
    assert torch.allclose(torch.diag(values), torch.ones(5))
    assert bool(torch.isfinite(values).all())
    assert bool((values <= 1.0).all())


def test_mgcn_interaction_layers_are_unshared_and_concat_T_plus_1_levels() -> None:
    from molgnn.models.mgcn_2019 import MGCN

    model = MGCN(num_targets=2, hidden_dim=8, num_layers=2, readout_hidden_dim=8)
    assert len(model.interactions) == 2
    assert model.interactions[0] is not model.interactions[1]
    # Readout consumes all T+1 concatenated levels.
    assert model.readout[0].in_features == 3 * 8
    assert model.readout[0].out_features == 8
    assert model.readout[2].out_features == 2


def test_mgcn_interaction_matches_paper_equations_on_tiny_graph() -> None:
    """Manual reference for paper Eqs. (5)--(6) with eta=0.8 and the
    old-edge/new-edge update timing: the message must consume the level-l
    edge state, not the freshly produced level l+1 state."""
    from molgnn.models.mgcn_2019.layers import InteractionLayer

    torch.manual_seed(0)
    layer = InteractionLayer(hidden_dim=1, rbf_dim=1, eta=0.8)
    # Pin every weight to make the reference arithmetic exact.
    with torch.no_grad():
        layer.edge_update.weight.copy_(torch.tensor([[0.5]]))
        layer.edge_update.bias.copy_(torch.tensor([-0.2]))
        layer.atom_proj.weight.copy_(torch.tensor([[0.3]]))
        layer.atom_proj.bias.copy_(torch.tensor([0.1]))
        layer.dist_proj.weight.copy_(torch.tensor([[-0.4]]))
        layer.dist_proj.bias.copy_(torch.tensor([0.2]))
        layer.edge_proj.weight.copy_(torch.tensor([[0.6]]))
        layer.edge_proj.bias.copy_(torch.tensor([0.05]))
        layer.message_linear.weight.copy_(torch.tensor([[0.7]]))
        layer.message_linear.bias.copy_(torch.tensor([0.15]))

    atom = torch.tensor([[2.0], [3.0]])
    edge = torch.tensor([[1.0], [4.0]])
    rbf = torch.tensor([[0.5], [1.5]])
    # Two-atom complete graph: (0->1) and (1->0).
    edge_index = torch.tensor([[0, 1], [1, 0]])

    atom_next, edge_next = layer(atom, edge, rbf, edge_index)

    # Eq. (5): edge_next = eta * edge + (1 - eta) * W_ue(atom_s * atom_t)
    expected_edge_01 = 0.8 * 1.0 + 0.2 * (0.5 * 2.0 * 3.0 - 0.2)
    expected_edge_10 = 0.8 * 4.0 + 0.2 * (0.5 * 2.0 * 3.0 - 0.2)
    assert torch.allclose(
        edge_next, torch.tensor([[expected_edge_01], [expected_edge_10]]), atol=1e-6
    )

    # Eq. (6) uses the OLD edge (level l): for edge (0->1), source=0, target=1.
    def reference_message(source_atom: float, dist: float, old_edge: float) -> float:
        atom_feat = 0.3 * source_atom + 0.1
        dist_feat = -0.4 * dist + 0.2
        edge_feat = 0.6 * old_edge + 0.05
        return math.tanh(0.7 * (atom_feat * dist_feat + edge_feat) + 0.15)

    message_01 = reference_message(2.0, 0.5, 1.0)
    message_10 = reference_message(3.0, 1.5, 4.0)
    # Eq. (4): atom_next aggregates messages into the target.
    assert torch.allclose(
        atom_next, torch.tensor([[message_10], [message_01]]), atol=1e-6
    )


# --- Pre-training via denoising / official TorchMD-ET ----------------------


def _pvd_sample(atomic_number: torch.Tensor, pos: torch.Tensor, sample_id: int):
    from molgnn.transforms import add_pvd_inputs

    node_count = atomic_number.shape[0]
    canonical = MolecularData(
        x=torch.zeros((node_count, 1), dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        y=torch.tensor([float(sample_id)]),
        y_mask=torch.tensor([True]),
        sample_id=torch.tensor([sample_id]),
        smiles="",
        atomic_number=atomic_number,
        pos=pos,
    )
    return add_pvd_inputs(canonical)


def test_pvd_radius_graph_keeps_source_self_loops_cap_and_batching() -> None:
    from molgnn.models.pvd_2023.geometry import build_pvd_radius_graph

    dense_pos = torch.zeros((40, 3), dtype=torch.float32)
    dense_pos[:, 0] = torch.arange(40) * 1.0e-3
    dense_edges = build_pvd_radius_graph(dense_pos, max_num_neighbors=32)
    _, target = dense_edges
    assert torch.equal(torch.bincount(target, minlength=40), torch.full((40,), 32))
    assert torch.equal(
        torch.sort(dense_edges[0, dense_edges[0] == dense_edges[1]]).values,
        torch.arange(40),
    )

    samples = [
        _pvd_sample(
            torch.tensor([6, 8, 1]),
            torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.1, 0.0]]),
            0,
        ),
        _pvd_sample(
            torch.tensor([7, 1]),
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0]]),
            1,
        ),
    ]
    batch = Batch.from_data_list(list[BaseData](samples))
    source, target = batch.pvd_edge_index
    assert torch.equal(batch.batch[source], batch.batch[target])
    self_nodes = source[source == target]
    assert torch.equal(torch.sort(self_nodes).values, torch.arange(5))


def test_pvd_torchmd_et_matches_official_source_golden_and_symmetries() -> None:
    from molgnn.models.pvd_2023 import PVDTorchMDET

    torch.manual_seed(4)
    sample = _pvd_sample(
        torch.tensor([6, 8, 1, 7]),
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.2, 0.1],
                [-0.4, 1.2, 0.3],
                [0.3, -0.5, 1.4],
            ]
        ),
        0,
    )
    batch = Batch.from_data_list(list[BaseData]([sample]))
    model = PVDTorchMDET(
        atom_dim=1,
        bond_dim=1,
        num_targets=2,
        hidden_channels=8,
        num_layers=2,
        num_rbf=6,
        num_heads=2,
    ).eval()
    with torch.no_grad():
        prediction = model(batch)
        noise_prediction = model.predict_noise(batch)
    # Generated once against the supplied official source with identical
    # state_dict and a shim only for its removed torch_cluster dependency.
    assert torch.allclose(
        prediction,
        torch.tensor([[0.2532393932, -0.4312753081]]),
        atol=1.0e-6,
    )
    assert torch.allclose(
        noise_prediction.flatten()[:8],
        torch.tensor(
            [
                -0.8378018737,
                1.1172088385,
                -0.0757732987,
                2.4714024067,
                -0.5568763018,
                2.3193545341,
                -0.0745793954,
                0.4436298013,
            ]
        ),
        atol=1.0e-6,
    )

    rotation = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    rotated = batch.clone()
    rotated.pos = batch.pos @ rotation.T
    translated = batch.clone()
    translated.pos = batch.pos + torch.tensor([3.0, -2.0, 4.0])
    with torch.no_grad():
        rotated_scalar = model(rotated)
        translated_scalar = model(translated)
        rotated_vector = model.predict_noise(rotated)
        translated_vector = model.predict_noise(translated)
    # The source whitening norm uses a tiny anisotropic SVD regularizer, so
    # rotation parity is approximate at ~1e-3 while translation is numerical.
    assert torch.allclose(prediction, translated_scalar, atol=1.0e-6)
    assert torch.allclose(noise_prediction, translated_vector, atol=1.0e-6)
    assert torch.allclose(prediction, rotated_scalar, atol=3.0e-3)
    assert torch.allclose(
        noise_prediction @ rotation.T, rotated_vector, atol=4.0e-3
    )


def test_pvd_denoising_rebuilds_graph_centers_noise_and_optimizes() -> None:
    from molgnn.models.pvd_2023 import PVDPretrainer, PVDTorchMDET
    from molgnn.models.pvd_2023.denoising import corrupt_pvd_batch

    crossing = _pvd_sample(
        torch.tensor([6, 8]),
        torch.tensor([[0.0, 0.0, 0.0], [4.9, 0.0, 0.0]]),
        0,
    )
    crossing_batch = Batch.from_data_list(list[BaseData]([crossing]))
    model = PVDTorchMDET(
        atom_dim=1,
        bond_dim=1,
        num_targets=1,
        hidden_channels=8,
        num_layers=2,
        num_rbf=6,
        num_heads=2,
    )
    original_pos = crossing_batch.pos.clone()
    original_edges = crossing_batch.pvd_edge_index.clone()
    noisy_batch, _ = corrupt_pvd_batch(
        crossing_batch,
        model,
        noise=torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]),
    )
    assert original_edges.shape[1] == 4
    assert noisy_batch.pvd_edge_index.shape[1] == 2
    assert torch.equal(crossing_batch.pos, original_pos)
    assert torch.equal(crossing_batch.pvd_edge_index, original_edges)

    training_sample = _pvd_sample(
        torch.tensor([6, 8, 1, 7]),
        torch.tensor(
            [[0.0, 0.0, 0.0], [1.1, 0.2, 0.1], [-0.4, 1.2, 0.3], [0.3, -0.5, 1.4]]
        ),
        0,
    )
    training_batch = Batch.from_data_list(list[BaseData]([training_sample]))
    pretrainer = PVDPretrainer(model, centering="paper_centered")
    optimizer = torch.optim.Adam(pretrainer.parameters(), lr=1.0e-3)
    optimizer.zero_grad(set_to_none=True)
    losses = pretrainer.compute_loss(
        training_batch, generator=torch.Generator().manual_seed(17)
    )
    assert torch.allclose(losses.noise_target.mean(dim=0), torch.zeros(3), atol=1.0e-7)
    assert losses.noise_prediction.shape == (4, 3)
    assert torch.isfinite(losses.total)
    losses.total.backward()
    assert model.encoder.embedding.weight.grad is not None
    assert model.noise_head.output_network[0].vec1_proj.weight.grad is not None
    optimizer.step()


def test_pvd_official_weight_only_checkpoint_loads_strictly() -> None:
    from molgnn.models.pvd_2023 import PVDPretrainer, PVDTorchMDET
    from molgnn.models.pvd_2023.checkpoint import DEFAULT_CHECKPOINT, load_pvd_pretrainer

    model = PVDTorchMDET(atom_dim=1, bond_dim=1, num_targets=1)
    pretrainer = PVDPretrainer(model)
    info = load_pvd_pretrainer(pretrainer, DEFAULT_CHECKPOINT)
    assert info["tensor_count"] == 147
    assert info["global_step"] == 400000
    assert torch.allclose(
        pretrainer.position_normalizer.std,
        torch.full((3,), 0.04),
        atol=2.0e-6,
    )


def test_pvd_native_qm9_schema_supports_supervised_and_denoising_steps() -> None:
    from molgnn.models.pvd_2023 import PVDPretrainer, PVDTorchMDET

    samples = [
        _pvd_sample(
            torch.tensor([6, 1, 1, 1, 1]),
            torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.6, 0.6, 0.6],
                    [-0.6, -0.6, 0.6],
                    [-0.6, 0.6, -0.6],
                    [0.6, -0.6, -0.6],
                ]
            ),
            0,
        ),
        _pvd_sample(
            torch.tensor([8, 1, 1]),
            torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.5, 0.0], [-0.8, 0.5, 0.0]]),
            1,
        ),
    ]
    for index, sample in enumerate(samples):
        sample.y = torch.arange(12, dtype=torch.float32).reshape(1, 12) + index
        sample.y_mask = torch.ones((1, 12), dtype=torch.bool)
    batch = Batch.from_data_list(list[BaseData](samples))
    model = PVDTorchMDET(
        atom_dim=153,
        bond_dim=14,
        num_targets=12,
        hidden_channels=8,
        num_layers=2,
        num_rbf=6,
        num_heads=2,
    )
    prediction = model(batch)
    assert prediction.shape == (2, 12)
    supervised_loss = torch.nn.functional.mse_loss(prediction, batch.y)
    supervised_loss.backward()
    assert model.property_head.output_network[1].update_net[2].weight.grad is not None

    model.zero_grad(set_to_none=True)
    denoising = PVDPretrainer(model).compute_loss(
        batch, generator=torch.Generator().manual_seed(23)
    )
    denoising.total.backward()
    assert denoising.noise_prediction.shape == (8, 3)
    assert model.noise_head.output_network[1].update_net[2].weight.grad is not None

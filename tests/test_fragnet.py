"""Focused contract checks for the coordinate-backed FragNet transform."""

import pytest
import torch
from torch_geometric.data import Batch

from molgnn.featurizer import featurize_smiles
from molgnn.models.fragnet_2026 import FragNet
from molgnn.transforms import TransformError, add_fragnet_inputs


def _sample(smiles: str, sample_id: int):
    data = featurize_smiles(
        smiles,
        targets=[0.0],
        target_mask=[True],
        sample_id=sample_id,
    )
    node_ids = torch.arange(data.x.shape[0], dtype=torch.float32)
    data.pos = torch.stack(
        (node_ids, node_ids.remainder(2), node_ids.remainder(3)), dim=-1
    )
    return data


def test_fragnet_requires_supplied_coordinates() -> None:
    data = featurize_smiles(
        "CCO", targets=[0.0], target_mask=[True], sample_id=0
    )

    with pytest.raises(TransformError, match="requires finite float32 pos"):
        add_fragnet_inputs(data)


def test_fragnet_batches_fragment_indices_after_a_single_fragment_sample() -> None:
    single_fragment = add_fragnet_inputs(_sample("CC", 0))
    fragmented = add_fragnet_inputs(_sample("CCOC(=O)NCC", 1))
    batch = Batch.from_data_list([single_fragment, fragmented])
    model = FragNet(
        atom_dim=153,
        bond_dim=14,
        num_targets=1,
        emb_dim=8,
        num_layers=1,
        num_heads=2,
        drop_ratio=0.0,
        head_hidden_dims=(8,),
    ).eval()

    assert single_fragment.x_frags.shape[0] == 1
    assert single_fragment.frag_index.numel() == 0
    assert fragmented.frag_index.numel()
    assert batch.frag_index.min().item() >= single_fragment.x_frags.shape[0]
    assert torch.equal(
        batch.frag_batch[batch.frag_index[0]],
        batch.frag_batch[batch.frag_index[1]],
    )
    assert model(batch).shape == (2, 1)

    malformed = batch.clone()
    malformed.frag_index = batch.frag_index.clone()
    malformed.frag_index[1, 0] = 0
    with pytest.raises(ValueError, match="frag_index must not connect different graphs"):
        model(malformed)

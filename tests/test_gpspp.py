"""End-to-end invariants for the 2D GPS++ hybrid graph architecture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from molgnn.featurizer import featurize_smiles
from molgnn.models.gpspp_2023 import GPSPlusPlus
from molgnn.transforms import add_gpspp_inputs
from molgnn.cli import main


def _sample(smiles: str, sample_id: int):
    return add_gpspp_inputs(
        featurize_smiles(
            smiles,
            targets=[0.0, 0.0],
            target_mask=[True, True],
            sample_id=sample_id,
        )
    )


def _model(num_targets: int = 2) -> GPSPlusPlus:
    torch.manual_seed(17)
    return GPSPlusPlus(
        atom_dim=153,
        bond_dim=14,
        node_dim=8,
        edge_dim=4,
        global_dim=4,
        depth=2,
        num_heads=2,
        max_spd=4,
        encoder_dropout=0.0,
        node_dropout=0.0,
        edge_dropout=0.0,
        global_dropout=0.0,
        attention_dropout=0.0,
        ffn_dropout=0.0,
        max_stochastic_depth=0.0,
        decoder_hidden_dim=8,
        num_targets=num_targets,
    ).eval()


def test_forward_backward_multitarget_and_state_dict_round_trip() -> None:
    batch = Batch.from_data_list([_sample("CCO", 0), _sample("C", 1)])
    model = _model()

    prediction = model(batch)
    prediction.square().sum().backward()
    restored = _model()
    restored.load_state_dict(model.state_dict())

    assert prediction.shape == (2, 2)
    assert torch.isfinite(prediction).all()
    assert model.global_embedding.grad is not None
    assert torch.isfinite(model.global_embedding.grad).all()
    assert torch.allclose(restored(batch), prediction, atol=1e-6)


def test_atom_relabeling_edge_order_and_companion_batch_preserve_prediction() -> None:
    model = _model()
    original = _sample("CCO", 0)
    expected = model(Batch.from_data_list([original]))

    raw = featurize_smiles(
        "CCO", targets=[0.0, 0.0], target_mask=[True, True], sample_id=1
    )
    order = torch.tensor([2, 0, 1], dtype=torch.long)
    old_to_new = torch.empty_like(order)
    old_to_new[order] = torch.arange(order.numel(), dtype=torch.long)
    relabeled = raw.clone()
    relabeled.x = raw.x[order]
    relabeled.edge_index = old_to_new[raw.edge_index]
    relabeled.edge_attr = raw.edge_attr.clone()
    relabeled = add_gpspp_inputs(relabeled)

    reverse_edges = original.clone()
    edge_order = torch.arange(
        original.edge_index.shape[1] - 1, -1, -1, dtype=torch.long
    )
    reverse_edges.edge_index = original.edge_index[:, edge_order]
    reverse_edges.edge_attr = original.edge_attr[edge_order]
    reverse_edges = add_gpspp_inputs(reverse_edges)

    companion = _sample("N", 2)
    relabeled_prediction = model(Batch.from_data_list([relabeled]))
    reordered_prediction = model(Batch.from_data_list([reverse_edges]))
    mixed_prediction = model(Batch.from_data_list([original, companion]))

    assert torch.allclose(relabeled_prediction, expected, atol=1e-6)
    assert torch.allclose(reordered_prediction, expected, atol=1e-6)
    assert torch.allclose(mixed_prediction[:1], expected, atol=1e-6)


def test_model_rejects_cross_graph_edges_and_malformed_all_pair_contract() -> None:
    batch = Batch.from_data_list([_sample("CC", 0), _sample("CO", 1)])
    model = _model()

    cross_edge = batch.clone()
    cross_edge.edge_index = batch.edge_index.clone()
    cross_edge.edge_index[1, 0] = 2
    with pytest.raises(ValueError, match="different graphs"):
        model(cross_edge)

    cross_pair = batch.clone()
    cross_pair.gpspp_pair_index = batch.gpspp_pair_index.clone()
    cross_pair.gpspp_pair_index[1, 0] = 2
    with pytest.raises(
        ValueError, match="gpspp_pair_index must not connect different graphs"
    ):
        model(cross_pair)

    bad_self_distance = batch.clone()
    bad_self_distance.gpspp_spd = batch.gpspp_spd.clone()
    bad_self_distance.gpspp_spd[0] = 1
    with pytest.raises(ValueError, match="zero for every self pair"):
        model(bad_self_distance)


def test_public_import_remains_independent_from_registry_and_transforms() -> None:
    script = """
import sys
from molgnn.models import GPSPlusPlus
assert GPSPlusPlus.__name__ == 'GPSPlusPlus'
assert 'molgnn.registry' not in sys.modules
assert 'molgnn.transforms' not in sys.modules
"""

    subprocess.run([sys.executable, "-c", script], check=True)


def test_cli_runner_uses_registered_transform_and_common_artifacts(
    tmp_path: Path,
) -> None:
    """The ordinary CSV/SMILES workflow must select GPS++ without a special case."""

    project_root = Path.cwd()
    fixture = project_root / "tests" / "fixtures" / "tiny_regression.csv"
    config = tmp_path / "gpspp.yaml"
    config.write_text(
        "\n".join(
            (
                f"extends: {(project_root / 'configs' / 'base.yaml').as_posix()}",
                "experiment:",
                "  name: gpspp_smoke",
                "  seed: 17",
                f"  output_dir: {tmp_path.as_posix()}",
                "data:",
                f"  path: {fixture.as_posix()}",
                "  split_ratios: [0.7, 0.2, 0.1]",
                "model:",
                "  name: gpspp",
                "  parameters: {node_dim: 8, edge_dim: 4, global_dim: 4, depth: 1, "
                "num_heads: 2, max_spd: 4, encoder_dropout: 0.0, node_dropout: 0.0, "
                "edge_dropout: 0.0, global_dropout: 0.0, attention_dropout: 0.0, "
                "ffn_dropout: 0.0, max_stochastic_depth: 0.0, decoder_hidden_dim: 8}",
                "training:",
                "  epochs: 1",
                "  batch_size: 4",
                "  patience: 1",
                "  device: cpu",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config)]) == 0

    run_dir = tmp_path / "gpspp_smoke" / "seed_017"
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert (run_dir / "best.ckpt").is_file()
    assert (run_dir / "test_predictions.csv").is_file()

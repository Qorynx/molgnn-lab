"""Sparse topology invariants for the planned AMPNN and EMNN models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch
from torch import Tensor, nn
from torch_geometric.data import Batch, Data
from torch_geometric.data.data import BaseData
from torch_geometric.utils import scatter

from molgnn.data import MolecularData
from molgnn.cli import main
from molgnn.featurizer import featurize_smiles
from molgnn.models.ampnn_emnn_2020.ampnn import AMPNN
from molgnn.models.ampnn_emnn_2020 import EMNNData
from molgnn.models.ampnn_emnn_2020.emnn import EMNN
from molgnn.transforms.ampnn import add_ampnn_edge_types
from molgnn.transforms.base import TransformError


def test_ampnn_selects_relation_networks_and_aggregates_by_target() -> None:
    model = _ampnn(
        atom_dim=1,
        message_dim=1,
        num_edge_types=2,
        num_message_passing_steps=1,
        message_hidden_dims=(),
        attention_hidden_dims=(),
    ).eval()
    _set_linear_weight(model.message_networks[0], [[1.0]])
    _set_linear_weight(model.message_networks[1], [[2.0]])
    _set_linear_weight(model.attention_networks[0], [[0.0]])
    _set_linear_weight(model.attention_networks[1], [[0.0]])

    hidden = torch.tensor([[3.0], [5.0], [0.0]])
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    edge_type = torch.tensor([0, 1], dtype=torch.long)

    # Equal energies make the two relation-specific messages average to 6.5.
    actual = model._aggregate_messages(hidden, edge_index, edge_type)
    assert torch.allclose(actual, torch.tensor([[0.0], [0.0], [6.5]]), atol=1e-6)


def test_ampnn_uses_raw_h0_and_one_tied_bias_free_gru() -> None:
    batch = _ampnn_batch()
    model = _ampnn(num_message_passing_steps=2).eval()
    states: list[Tensor] = []

    def capture(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        states.append(inputs[1].detach().clone())

    handle = model.gru.register_forward_pre_hook(capture)
    prediction = model(batch)
    handle.remove()

    assert prediction.shape == (2, 1)
    assert len(states) == 2
    assert torch.equal(states[0], batch.x)
    assert model.gru.bias_ih is None
    assert model.gru.bias_hh is None
    assert len(model.message_networks) == model.num_edge_types
    assert len(model.attention_networks) == model.num_edge_types


def test_ampnn_transform_and_forward_support_canonical_and_edgeless_samples() -> None:
    canonical = featurize_smiles(
        "C=CC#N", targets=[0.0], target_mask=[True], sample_id=4
    )
    transformed = add_ampnn_edge_types(canonical)
    assert transformed.ampnn_edge_type.dtype == torch.long
    assert torch.equal(
        transformed.ampnn_edge_type,
        canonical.edge_attr[:, :4].argmax(dim=-1),
    )

    edgeless = Batch.from_data_list(
        cast(
            list[BaseData],
            [
                Data(
                    x=torch.tensor([[0.2, -0.1, 0.7]]),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    ampnn_edge_type=torch.empty(0, dtype=torch.long),
                )
            ],
        )
    )
    prediction = _ampnn(num_targets=2)(edgeless)
    assert prediction.shape == (1, 2)
    assert torch.isfinite(prediction).all()

    canonical.edge_attr[:, :5] = 0
    canonical.edge_attr[:, 4] = 1
    with pytest.raises(TransformError, match="unsupported bond type"):
        add_ampnn_edge_types(canonical)


def test_ampnn_is_edge_order_invariant_and_rejects_cross_graph_edges() -> None:
    torch.manual_seed(17)
    batch = _ampnn_batch()
    model = _ampnn(dropout=0.3).eval()
    expected = model(batch)

    samples = batch.to_data_list()
    permuted = []
    for sample in samples:
        edge_index = cast(Tensor, sample.edge_index)
        edge_type = cast(Tensor, sample.ampnn_edge_type)
        order = torch.arange(edge_index.shape[1] - 1, -1, -1)
        permuted.append(
            Data(
                x=sample.x,
                edge_index=edge_index[:, order],
                ampnn_edge_type=edge_type[order],
            )
        )
    actual = model(Batch.from_data_list(cast(list[BaseData], permuted)))
    assert torch.allclose(actual, expected, atol=1e-6)

    invalid = batch.clone()
    invalid.edge_index = invalid.edge_index.clone()
    invalid.edge_index[1, 0] = 3
    with pytest.raises(ValueError, match="different graphs"):
        model(invalid)


def test_emnn_nonbacktracking_attention_keeps_static_edge_item() -> None:
    model = _emnn(
        atom_dim=1,
        bond_dim=1,
        edge_hidden_dim=1,
        edge_embedding_hidden_dims=(),
        message_hidden_dims=(),
        attention_hidden_dims=(),
    ).eval()
    _set_linear_weight(model.message_network, [[1.0]])
    _set_linear_weight(model.attention_network, [[0.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    reverse = torch.tensor([1, 0, 3, 2], dtype=torch.long)
    static_edges = torch.tensor([[10.0], [20.0], [30.0], [40.0]])
    memories = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

    actual = model._edge_messages(static_edges, memories, edge_index, reverse)

    # For 1 -> 2, the only allowed predecessor is 0 -> 1. The reverse
    # 2 -> 1 is excluded; static e'_12 remains in the attention set.
    expected = torch.tensor([[10.0], [12.0], [15.5], [40.0]])
    assert torch.allclose(actual, expected, atol=1e-6)


def test_emnn_nonbacktracking_attention_is_stable_after_excluding_unique_maximum() -> (
    None
):
    model = _emnn(
        atom_dim=1,
        bond_dim=1,
        edge_hidden_dim=1,
        edge_embedding_hidden_dims=(),
        message_hidden_dims=(),
        attention_hidden_dims=(),
    ).eval()
    _set_linear_weight(model.message_network, [[1.0]])
    _set_linear_weight(model.attention_network, [[-1000.0]])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    reverse = torch.tensor([1, 0, 3, 2], dtype=torch.long)
    static_edges = torch.ones((4, 1), requires_grad=True)
    memories = torch.tensor([[1.0], [0.0], [0.0], [-1.0]], requires_grad=True)

    actual = model._edge_messages(static_edges, memories, edge_index, reverse)

    # For 1 -> 2, its reverse 2 -> 1 is the unique +1000 logit but is
    # excluded.  The valid predecessor 0 -> 1 and static item both have
    # -1000 logits, so their exact attention output is 1 rather than NaN.
    assert torch.isfinite(actual).all()
    assert torch.allclose(actual, torch.tensor([[1.0], [-1.0], [1.0], [1.0]]))
    actual.sum().backward()
    assert static_edges.grad is not None and torch.isfinite(static_edges.grad).all()
    assert memories.grad is not None and torch.isfinite(memories.grad).all()


def test_emnn_uses_previous_edge_memory_and_aggregates_to_source_nodes() -> None:
    batch = _emnn_batch()
    model = _emnn(num_message_passing_steps=2).eval()
    gru_inputs: list[tuple[Tensor, Tensor]] = []
    final_edge_memories: list[Tensor] = []
    readout_inputs: list[tuple[Tensor, Tensor]] = []

    def capture_gru(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        gru_inputs.append((inputs[0].detach().clone(), inputs[1].detach().clone()))

    def capture_gru_output(
        _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor
    ) -> None:
        final_edge_memories.append(output.detach().clone())

    def capture_readout(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        readout_inputs.append((inputs[0].detach().clone(), inputs[1].detach().clone()))

    handles = (
        model.gru.register_forward_pre_hook(capture_gru),
        model.gru.register_forward_hook(capture_gru_output),
        model.graph_readout.register_forward_pre_hook(capture_readout),
    )
    prediction = model(batch)
    for handle in handles:
        handle.remove()

    assert prediction.shape == (2, 1)
    assert len(gru_inputs) == model.num_message_passing_steps
    assert torch.equal(gru_inputs[0][1], torch.zeros_like(gru_inputs[0][1]))
    assert not torch.equal(gru_inputs[1][1], torch.zeros_like(gru_inputs[1][1]))
    assert model.gru.bias_ih is None
    assert model.gru.bias_hh is None
    assert len(final_edge_memories) == model.num_message_passing_steps
    expected_nodes = scatter(
        final_edge_memories[-1],
        batch.edge_index[0],
        dim=0,
        dim_size=batch.x.shape[0],
        reduce="sum",
    )
    assert torch.allclose(readout_inputs[-1][0], expected_nodes, atol=1e-6)
    assert torch.allclose(readout_inputs[-1][1], expected_nodes, atol=1e-6)


def test_emnn_rejects_bad_reverse_pairs_and_handles_edgeless_graphs() -> None:
    batch = _emnn_batch()
    model = _emnn()
    invalid = batch.clone()
    invalid.edge_attr = invalid.edge_attr.clone()
    invalid.edge_attr[1, 0] = 5.0
    with pytest.raises(ValueError, match="match along reverse"):
        model(invalid)

    edgeless = Batch.from_data_list(
        cast(
            list[BaseData],
            [
                MolecularData(
                    x=torch.tensor([[0.2, -0.1, 0.7]]),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 2)),
                    reverse_edge_index=torch.empty(0, dtype=torch.long),
                )
            ],
        )
    )
    prediction = _emnn(num_targets=2)(edgeless)
    assert prediction.shape == (1, 2)
    assert torch.isfinite(prediction).all()


def test_emnn_data_batches_reverse_edges_by_directed_edge_count() -> None:
    x = torch.randn(3, 3)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    edge_attr = torch.tensor([[1.0, -0.2], [1.0, -0.2], [0.4, 0.8], [0.4, 0.8]])
    reverse = torch.tensor([1, 0, 3, 2])
    samples = [
        EMNNData(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            reverse_edge_index=reverse,
        )
        for _ in range(2)
    ]
    batch = Batch.from_data_list(samples)

    assert batch.reverse_edge_index.tolist() == [1, 0, 3, 2, 5, 4, 7, 6]
    output = _emnn()(batch)
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("model_name", ("ampnn", "emnn"))
def test_models_are_atom_permutation_invariant_and_batch_isolated(
    model_name: str,
) -> None:
    model, batch = (
        (_ampnn().eval(), _ampnn_batch())
        if model_name == "ampnn"
        else (_emnn().eval(), _emnn_batch())
    )
    expected = model(batch)
    samples = batch.to_data_list()

    # Renumber nodes within each graph while preserving the directed edge
    # records (including EMNN's edge-local reverse map).  A molecular graph
    # representation must not depend on arbitrary RDKit/PyG atom numbering.
    relabelled: list[BaseData] = []
    for sample in samples:
        relabelled_sample = sample.clone()
        num_nodes = sample.x.shape[0]
        new_to_old = torch.arange(num_nodes - 1, -1, -1)
        old_to_new = torch.empty_like(new_to_old)
        old_to_new[new_to_old] = torch.arange(num_nodes)
        relabelled_sample.x = sample.x[new_to_old]
        relabelled_sample.edge_index = old_to_new[sample.edge_index]
        relabelled.append(relabelled_sample)

    actual = model(Batch.from_data_list(relabelled))
    assert torch.allclose(actual, expected, atol=1e-6)

    first_graph_only = model(Batch.from_data_list([samples[0]]))
    assert torch.allclose(first_graph_only[0], expected[0], atol=1e-6)


@pytest.mark.parametrize("model_name", ("ampnn", "emnn"))
def test_models_keep_their_2d_topology_when_pos_is_present(model_name: str) -> None:
    model, batch = (
        (_ampnn().eval(), _ampnn_batch())
        if model_name == "ampnn"
        else (_emnn().eval(), _emnn_batch())
    )
    expected = model(batch)
    coordinate_carrying = batch.clone()
    coordinate_carrying.pos = torch.randn(batch.x.shape[0], 3)

    assert torch.allclose(model(coordinate_carrying), expected, atol=1e-6)


@pytest.mark.parametrize(
    ("model_name", "parameters"),
    [
        (
            "ampnn",
            "{message_dim: 8, num_message_passing_steps: 1, "
            "message_hidden_dims: [8], attention_hidden_dims: [8], gather_dim: 8, "
            "gather_gate_hidden_dims: [8], gather_value_hidden_dims: [8], "
            "predictor_hidden_dims: [8], dropout: 0.0}",
        ),
        (
            "emnn",
            "{edge_hidden_dim: 8, num_message_passing_steps: 1, "
            "edge_embedding_hidden_dims: [8], message_hidden_dims: [8], "
            "attention_hidden_dims: [8], gather_dim: 8, "
            "gather_gate_hidden_dims: [8], gather_value_hidden_dims: [8], "
            "predictor_hidden_dims: [8], dropout: 0.0}",
        ),
    ],
)
def test_cli_runs_shared_csv_workflow(
    tmp_path: Path, model_name: str, parameters: str
) -> None:
    project_root = Path.cwd()
    fixture = project_root / "tests" / "fixtures" / "tiny_regression.csv"
    config = tmp_path / f"{model_name}.yaml"
    config.write_text(
        "\n".join(
            [
                "extends: " + str(project_root / "configs" / "base.yaml"),
                "experiment:",
                f"  name: {model_name}_smoke",
                "  seed: 17",
                f"  output_dir: {tmp_path.as_posix()}",
                "data:",
                f"  path: {fixture.as_posix()}",
                "  split_ratios: [0.7, 0.2, 0.1]",
                "model:",
                f"  name: {model_name}",
                f"  parameters: {parameters}",
                "training:",
                "  epochs: 1",
                "  batch_size: 4",
                "  patience: 1",
                "  device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(["train", "--config", str(config)]) == 0
    run_dir = tmp_path / f"{model_name}_smoke" / "seed_017"
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))[
        "status"
    ] == ("completed")
    assert (run_dir / "best.ckpt").is_file()
    assert (run_dir / "test_predictions.csv").is_file()


def _ampnn(**overrides: object) -> AMPNN:
    parameters: dict[str, object] = {
        "atom_dim": 3,
        "message_dim": 4,
        "num_message_passing_steps": 2,
        "message_hidden_dims": (4,),
        "attention_hidden_dims": (4,),
        "gather_dim": 4,
        "gather_gate_hidden_dims": (4,),
        "gather_value_hidden_dims": (4,),
        "predictor_hidden_dims": (4,),
        "dropout": 0.0,
        "num_targets": 1,
    }
    parameters.update(overrides)
    return AMPNN(**parameters)  # type: ignore[arg-type]


def _ampnn_batch() -> Batch:
    samples = [
        Data(
            x=torch.tensor([[0.1, 0.2, -0.1], [0.5, -0.3, 0.4], [-0.2, 0.6, 0.3]]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            ampnn_edge_type=torch.tensor([0, 0, 1, 1]),
        ),
        Data(
            x=torch.tensor([[0.3, -0.2, 0.8], [-0.4, 0.1, 0.2]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            ampnn_edge_type=torch.tensor([3, 3]),
        ),
    ]
    return Batch.from_data_list(cast(list[BaseData], samples))


def _emnn(**overrides: object) -> EMNN:
    parameters: dict[str, object] = {
        "atom_dim": 3,
        "bond_dim": 2,
        "edge_hidden_dim": 4,
        "num_message_passing_steps": 2,
        "edge_embedding_hidden_dims": (4,),
        "message_hidden_dims": (4,),
        "attention_hidden_dims": (4,),
        "gather_dim": 4,
        "gather_gate_hidden_dims": (4,),
        "gather_value_hidden_dims": (4,),
        "predictor_hidden_dims": (4,),
        "dropout": 0.0,
        "num_targets": 1,
    }
    parameters.update(overrides)
    return EMNN(**parameters)  # type: ignore[arg-type]


def _emnn_batch() -> Batch:
    samples = [
        MolecularData(
            x=torch.tensor([[0.1, 0.2, -0.1], [0.5, -0.3, 0.4], [-0.2, 0.6, 0.3]]),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
            edge_attr=torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]),
            reverse_edge_index=torch.tensor([1, 0, 3, 2]),
        ),
        MolecularData(
            x=torch.tensor([[0.3, -0.2, 0.8], [-0.4, 0.1, 0.2]]),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_attr=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
            reverse_edge_index=torch.tensor([1, 0]),
        ),
    ]
    return Batch.from_data_list(cast(list[BaseData], samples))


def _set_linear_weight(network: nn.Module, values: list[list[float]]) -> None:
    linear = next(layer for layer in network.modules() if isinstance(layer, nn.Linear))
    with torch.no_grad():
        linear.weight.copy_(torch.tensor(values, dtype=linear.weight.dtype))

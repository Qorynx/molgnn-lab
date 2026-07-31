"""Public CLI checks for runtime model-contract inspection."""

import json

from molgnn.cli import main
from molgnn.registry import available_models, clear_registry


def test_describe_model_text_lazily_registers_builtins(capsys) -> None:
    clear_registry()

    exit_code = main(["describe-model", "--model", "hignn"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert "Model: hignn" in captured.out
    assert "Required batch fields: x, edge_index, edge_attr" in captured.out
    assert "Graph transform: brics_fragments" in captured.out
    assert "Prediction reducer: identity" in captured.out
    assert "Benchmark enabled: true" in captured.out
    assert "Benchmark order: 40" in captured.out
    assert "architecture card" not in captured.out.lower()
    assert "paper" not in captured.out.lower()
    assert "source" not in captured.out.lower()
    assert "hignn" in available_models()


def test_describe_model_json_exposes_only_the_runtime_contract(capsys) -> None:
    exit_code = main(["describe-model", "--model", "dmpnn", "--format", "json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert captured.err == ""
    assert payload["name"] == "dmpnn"
    assert payload["required_batch_fields"] == [
        "x",
        "edge_index",
        "edge_attr",
        "reverse_edge_index",
        "batch",
    ]
    assert payload["graph_transform_name"] == "directed_edges"
    assert payload["prediction_reducer_name"] == "identity"
    assert payload["benchmark_enabled"] is True
    assert payload["benchmark_order"] == 30
    assert set(payload) == {
        "name",
        "required_batch_fields",
        "graph_transform_name",
        "prediction_reducer_name",
        "benchmark_enabled",
        "benchmark_order",
    }


def test_describe_model_reports_unknown_names_without_a_traceback(capsys) -> None:
    exit_code = main(["describe-model", "--model", "missing_model"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Model description error: unknown model 'missing_model'" in captured.err
    assert "Available models:" in captured.err
    assert "Traceback" not in captured.err

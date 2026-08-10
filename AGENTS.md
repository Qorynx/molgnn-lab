# AGENTS.md

Benchmark framework for 2D molecular GNNs. Single package `molgnn` (src layout), CLI entry `molgnn.cli:main`. Docs of record: `README.md` (run the README commands as written — paths like `../data/dataset.csv` are relative to `configs/`).

## Environment

- Python must be `>=3.11,<3.12` (pyproject.toml). The machine's default `python` is 3.13 — create a 3.11 venv before install/run.
- `requirements.txt` is just `.`; real deps live in `pyproject.toml`. Install the package (`pip install -e .` in a 3.11 env) or tests/CLI cannot `import molgnn`.
- There is **no CI, no lint/typecheck/test config** in the repo. No ruff/mypy/pytest/tox/make config files exist; nothing to run for "verification" except the tracked tests.
- Datasets (`data/`, `datasets/`, `*.csv`) are gitignored and never shipped. `configs/` tracks only `example.yaml` (`configs/base.yaml` is private/absent).

## Adding a model (the main task in this repo)

Four touchpoints, all required:

1. `src/molgnn/models/<name>[_<year>]/` — simple ones are flat files (e.g. `gcn_baseline.py`), complex ones are packages (`hignn_2023/model.py` + `data.py` + `layers.py`). Subclass `BaseMolecularModel` from `models/base.py`; `forward(batch) -> Tensor` of shape `[batch_size, num_targets]`. Declare `required_batch_fields` as a class attribute.
2. `src/molgnn/models/__init__.py` — add the symbol to the `_EXPORTS` dict. This file uses lazy PEP-562 `__getattr__` on purpose (keeps CLI import lightweight); keep it lazy, do not import architecture modules eagerly.
3. `src/molgnn/models/registration.py` — register in `register_builtin_models()` with `required_batch_fields`, `graph_transform_name` (if a transform is needed), `prediction_reducer_name`, `benchmark_order`.
4. If the model needs a helper transform: add it under `src/molgnn/transforms/` and register it in `transforms/__init__.py::register_builtin_transforms()` (it must be callable `MolecularData -> MolecularData`).

Registry rules that bite (enforced in `registry.py` via `inspect.signature`, not docs):
- `atom_dim`, `bond_dim`, `num_targets` are injected from `BuildContext` — your constructor must accept these exact names, and they **cannot** be set/overridden via YAML `model.parameters` (raises `RegistryError`). All other constructor params must have defaults or be YAML-provided; unknown params raise `RegistryError`.
- Every model's `forward` must accept a PyG `Batch` (matching `required_batch_fields`) and reject cross-graph edges with a `ValueError` — a preflight run in `runner.py:_preflight_model` checks output shape and this is enforced by tracked tests. See `models/contracts.py::validate_batched_molecular_graph` and reuse it.

## Graph conventions (non-obvious, must preserve)

- Canonical runner represents each bond as **two reverse edges, no self-loops**. Models whose equations exclude supplied self-loops should request `forbid_self_loops=True`.
- `edge_index`/`batch` always describe the real graph batch; extra fields differ per architecture. Verify a model's contract with `molgnn describe-model --model <name>` (text or `--format json`) — treat that as source of truth, not assumed schemas.
- D-MPNN's `reverse_edge_index` must be an involution (`reverse_edge_index[reverse_edge_index]` returns the original edges).

## Tests

- Only four files are tracked (gitignore keeps the rest private): `test_architecture_fidelity.py`, `test_model_input_contracts.py`, `test_describe_model_cli.py`, `test_cli_hooks.py`. Don't add tracked tests without asking.
- `pytest` is **not** a dependency — install it yourself. Run from repo root after installing the package.
- `test_cli_hooks.py` depends on private fixtures `configs/base.yaml` and `tests/fixtures/tiny_regression.csv`, which are gitignored and absent in this checkout — that test will fail locally until those exist.
- New-model verification: `molgnn validate-config --config configs/example.yaml`, `molgnn describe-model --model <name>`, and run `test_model_input_contracts.py` + `test_architecture_fidelity.py`.

## CLI / flow

- `molgnn train` (single model, all seeds, writes summary) → `runner.run_experiments`; `molgnn benchmark` (model-outer, seed-inner) → `runner.run_benchmark`. Model registration is lazy — runners call `register_builtin_models()` at run time, not at import.
- `train` supports local Python hooks `--featurizer path.py:callable` / `--training-strategy path.py:callable` (see README); hooks must return `MolecularData` / `StrategyResult` respectively.

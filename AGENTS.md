# AGENTS.md

Compact guide for OpenCode sessions working in `molgnn-lab`. The README
is the comprehensive source of truth (Vietnamese, kept as-is for project
identity); this file only collects facts that an agent would otherwise
have to dig for.

## What this repo is

Architecture-only framework for 2-D molecular and 3-D complex GNNs.
Models live under `src/molgnn/models/<name>[_YYYY]/`, are registered in
`src/molgnn/models/registration.py::register_builtin_models`, and are
selected by `model.name` in a YAML config. Runtime CLI is `molgnn`
(see `[project.scripts]` in `pyproject.toml`).

## Required environment

- Python **3.11** only — `pyproject.toml` pins `requires-python = ">=3.11,<3.12"`. The expected interpreter is `C:\Users\nhata\AppData\Local\Programs\Python\Python311\python.exe`. Do not use 3.10 or 3.12+.
- `.venv` already exists at the repo root with the lab's pinned torch 2.12 / PyG 2.8. Activate via `.venv\Scripts\Activate.ps1` (Windows) and use `.venv\Scripts\python.exe` for all calls.
- CUDA is optional; the framework runs on CPU. BACE smoke runs are CPU-only.

## Install and verify

```bash
.venv\Scripts\python.exe -m pip install -e .   # already done in this checkout
.venv\Scripts\python.exe -m pip install pytest  # only if `pytest` is missing
.venv\Scripts\python.exe -m pytest -q
```

`requirements.txt` is intentionally trivial (`.` only) — runtime deps
live in `pyproject.toml`. Do not hand-edit `requirements.txt`.

## Branch + port workflow

The project is set up for incremental model ports. Each new model
lands on its own branch named `feat/add-<NAME>` (matching the directory
name), cut from `origin/main`. The standard prompts for this are
served by OpenCode (see `.opencode/PROMPT_add_model.md`).

Adding a model means:
1. New directory under `src/molgnn/models/<name>[_YYYY]/` with
   `__init__.py`, `model.py`, and (if >150 LoC) `layers.py`.
2. New transform under `src/molgnn/transforms/<name>.py` if the model
   needs derived views (BRICS fragments, 3-D conformers, line graphs,
   all-pairs dist-bin views, etc.).
3. Registry wiring in `src/molgnn/models/registration.py`,
   `_EXPORTS` / `__all__` in `src/molgnn/models/__init__.py`, and
   `register_builtin_transforms` + `__all__` in
   `src/molgnn/transforms/__init__.py`.
4. Hard-coded expected lists in `tests/test_registry.py` and
   `tests/test_architecture_fidelity.py` need the new model name.

`atom_dim` / `bond_dim` / `num_targets` are injected by `BuildContext`
at build time — declare them as plain constructor args (no default),
validate them, and store them on `self`. The registry raises
`RegistryError` if a YAML parameter shadows them.

## Architecture conventions (the easy ones to get wrong)

- **Canonical featurizer**: 153-dim atom features, 14-dim bond features
  (`molgnn.featurizer.CANONICAL_FEATURE_SCHEMA_V1`). Do **not** implement
  a custom featurizer inside a model — every port ports a different
  custom featurizer; the lab's contract is to use the canonical one.
- **Canonical graph layout**: every bond stored as two directed edges
  (canonical `edge_index` in `molgnn.featurizer.featurize_mol`). No
  self-loops in the input. `validate_batched_molecular_graph` rejects
  cross-graph edges. **Bipartite** models (HiGNN fragment-to-mol) must
  skip that check.
- **Subclass `BaseMolecularModel`**, implement `forward(batch) -> Tensor`
  of shape `[num_graphs, num_targets]`. Raw logits for
  `binary_classification` (no Sigmoid — paired with `bce_with_logits`).
- **Stored `__inc__` and `__cat_dim__`**: handled by
  `molgnn.data.MolecularData`. Per-model extras live in
  `__inc__` (e.g. `frag_index`, `edge_index_bonds_graph`).
- **Module-level constants** in `layers.py` (e.g.
  `_DISTANCE_KERNELS = ("softmax", "exp")`); no dicts / registries
  inside `__init__`.
- **Class-name registration** uses the model registry's `_EXPORTS`
  lazy dict — never `import` a model at module top level.

## Tracked tests vs the test directory

**All** tests under `tests/` are gitignored *except* the ones explicitly
un-ignored in `.gitignore`. The un-ignored (public) set is the contract:

- `tests/test_architecture_fidelity.py`
- `tests/test_model_input_contracts.py`
- `tests/test_describe_model_cli.py`
- `tests/test_cli_hooks.py`
- `tests/test_mpnn_3d.py`
- `tests/test_potentialnet.py`
- `tests/test_registry.py`
- `tests/test_fragnet.py`
- `tests/test_weave.py`
- `tests/test_ampnn_emnn*.py`
- `tests/test_dimenet*.py`
- `tests/test_gpspp*.py`

When adding a model, expand the hard-coded expected lists in
`tests/test_registry.py::test_unknown_model_lists_available_models` and
`tests/test_benchmark_selection_uses_default_order...` and the
`expected` set in `tests/test_architecture_fidelity.py::test_builtin_models_expose_runtime_input_contracts`.

## Smoke test pattern (mandatory after a model port)

```bash
.venv\Scripts\python.exe tmp\test_<name>_smoke.py
.venv\Scripts\python.exe -m pytest tests/test_describe_model_cli.py tests/test_model_input_contracts.py tests/test_architecture_fidelity.py tests/test_registry.py -q
```

The smoke script should call `add_<name>_inputs` (if applicable),
featurize three SMILES, batch them, run the model, and assert the
output shape is `(3, 1)`. The `describe-model` round-trip is part of
the smoke — if the registry entry is wrong, the smoke fails on it.

## Audit / verification helpers

- `molgnn describe-model --model <name>` — prints the runtime contract
  (required batch fields, optional transform, prediction reducer,
  benchmark metadata). Read this first when debugging a model-port
  failure; the source of truth is the registry entry, not the README.
- `molgnn validate-config --config <yaml>` — checks a config without
  starting a run. Does not import any `--featurizer` /
  `--training-strategy` hook.

## Things NOT to do

- Do **not** commit or push on the human's behalf — the porting
  workflow explicitly forbids it. Stage and commit only when the
  human runs the `git commit` / `git push` themselves.
- Do **not** touch `tmp/`, `paper_pdf/`, `data_test/`, or `runs/`. They
  are gitignored and used as scratch / dataset / output space; writes
  there vanish on the next clone.
- Do **not** commit pretrained weights downloaded into `tmp/`. They are
  scratch material for that single port.
- Do **not** override `__cat_dim__` for edge-index-style fields — the
  PyG default is correct. Only override `__inc__` for fields whose
  per-graph increment isn't 1.
- Do **not** apply Sigmoid on classification output.
- Do **not** add `np.random.seed(...)` / `torch.manual_seed(...)` to
  model `__init__`. Construction must not depend on global state.

## Documented caveats during model ports

These are easy to get wrong even after reading the paper:

- **3-D-coordinate models** (EGNN, MAT, DimeNet, MPNN-3D) need
  per-sample `pos: float32 [N, 3]`. The lab's bundled transform
  synthesises conformers with RDKit `ETKDG` + `UFFOptimizeMolecule` /
  `MMFFOptimizeMolecule`. **This is not what the paper does** — the
  papers train on pre-computed (QM9 DFT, MD17) geometries. Treat
  numbers from BACE / BBBP as a *lower bound* on the architecture's
  true capability and document this prominently in the
  module-level docstring of the model.
- **Bipartite / line-graph / dummy-node** models must bypass
  `validate_batched_molecular_graph`. HiGNN (fragment↔mol),
  DimeNet (edge↔triplet), MAT (dummy node) and similar all need
  either `forbid_self_loops=False` plus extra `__inc__` entries or
  the upstream's specialised batching.
- **BACE on EGNN / MAT** is a 2-D dataset; the model synthesises
  3-D conformers per molecule. Performance numbers are dominated by
  this gap, not by model fidelity. Numerical verification against
  the paper's equations is the right correctness check.
- The `featurize_smiles` / `data.py` module is treated as the shared
  contract. Do not introduce a parallel featurization path; instead
  layer model-specific fields via the `graph_transform_name` hook.
- `molgnn` is a long-running CLI. When iterating, prefer the
  single-seed `experiment.seeds: [42]` form over multi-seed sweeps.

## Useful file locations

- `src/molgnn/featurizer.py` — canonical 153 / 14 schema, `featurize_smiles`.
- `src/molgnn/data.py` — `MolecularData`, `validate_molecular_data`, the
  shared sample contract.
- `src/molgnn/registry.py` — `BuildContext`, `register_model`,
  `available_models`, `benchmark_models`, `resolve_benchmark_models`.
- `src/molgnn/models/registration.py` — single source of truth for
  built-in model registration (including `benchmark_order`,
  `graph_transform_name`, `transform_output_fields`).
- `src/molgnn/transforms/` — one file per helper, all exported via
  `register_builtin_transforms`.
- `src/molgnn/cli.py` — `molgnn train / describe-model / validate-config
  / benchmark` entrypoints and the two runtime hooks
  (`--featurizer`, `--training-strategy`).
- `configs/example.yaml`, `configs/ampnn_smoke.yaml`,
  `configs/emnn_smoke.yaml` — the only un-ignored configs; everything
  else under `configs/` is gitignored.

# molgnn-lab

`molgnn-lab` is a small model zoo for training and comparing graph neural
networks on molecular data. Experiments are configured with YAML files and run
from the `molgnn` command-line interface.

## Requirements

- Python 3.11
- Windows or Linux
- A GPU is optional

## Installation

From the project directory, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Or on Linux:

```bash
source .venv/bin/activate
```

Install the project and check that the CLI is available:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
molgnn --version
```

PyTorch is installed as a project dependency. If you need a specific CUDA
build, install the matching PyTorch package before installing `molgnn-lab`.

## Prepare a dataset

The project does not download datasets automatically. For a basic regression
experiment, create `data/dataset.csv` with a SMILES column and a target column:

```csv
smiles,target
CCO,0.42
CC(=O)O,0.81
```

Binary classification targets should be `0` or `1`. If your column names or
file location are different, update the `data` section in the YAML config.

## Run an experiment

The included example uses a GCN for regression. First, check its config:

```bash
molgnn validate-config --config configs/example.yaml
```

Then start training:

```bash
molgnn train --config configs/example.yaml
```

Results are written to `runs/example_gcn_regression/`. The folder contains the
resolved config, checkpoints, training history, metrics, and test predictions.
For each seed, `run_results.json` stores the dataset split and one nested
loss/metrics object per completed epoch. Test history and predictions remain
separate CSV artifacts. After every configured seed finishes,
`aggregate_metrics.json` reports final test metrics from each best-validation
checkpoint as mean and sample standard deviation (`ddof=1`). A single completed
run has `std: null` because its run-to-run variation cannot be estimated.

To run another experiment, copy an existing file from `configs/` and change the
dataset, model, and training settings. The main fields are:

```yaml
data:
  path: ../data/dataset.csv
  smiles_column: smiles
  target_columns: [target]

model:
  name: gcn_baseline

training:
  epochs: 20
  device: auto

task:
  type: regression
```

Paths in a config file are resolved relative to that config's directory.

## Models and input requirements

Models can require different graph fields, especially models that use 3D
coordinates or fragment graphs. Check a model before using it:

```bash
molgnn describe-model --model dmpnn
molgnn describe-model --model dmpnn --format json
```

For a quick check, the repository also includes lightweight configs such as
`configs/gcn_smoke.yaml`, `configs/ampnn_smoke.yaml`, and
`configs/emnn_smoke.yaml`.

## Benchmark multiple models

Use a config with a `models` list, then run:

```bash
molgnn benchmark --config configs/esol_all_models_3seeds_50epochs.yaml
```

Each model and seed gets its own output directory. Models with special data
requirements are skipped by the default benchmark unless they are selected
explicitly. The benchmark root contains one nested `aggregate_metrics.json`
covering every selected model. It is marked `partial` and records failures when
not all configured model/seed runs complete.

## Common commands

```bash
# Validate a config
molgnn validate-config --config configs/example.yaml

# Train one model
molgnn train --config configs/example.yaml

# Inspect a model's expected inputs
molgnn describe-model --model hignn

# Show CLI help
molgnn --help
```

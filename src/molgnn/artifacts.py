"""Small atomic writer for experiment artifacts."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import cast

import torch
import yaml


class ArtifactError(RuntimeError):
    """Raised when an artifact cannot be written consistently."""


RUN_RESULTS_SCHEMA_VERSION = 1
AGGREGATE_METRICS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunPaths:
    """Canonical paths for one experiment and seed."""

    experiment_dir: Path
    seed_dir: Path
    experiment_config: Path
    run_results_json: Path
    summary_csv: Path
    aggregate_metrics_json: Path
    config_yaml: Path
    status_json: Path
    test_history_csv: Path
    best_checkpoint: Path
    last_checkpoint: Path
    test_predictions_csv: Path

    @classmethod
    def create(
        cls,
        root: str | Path,
        experiment_name: str,
        seed: int,
        *,
        initialize_status: bool = True,
    ) -> RunPaths:
        root_path = Path(root).expanduser().resolve()
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ArtifactError("experiment_name must be non-empty")
        if (
            "/" in experiment_name
            or "\\" in experiment_name
            or ".." in Path(experiment_name).parts
        ):
            raise ArtifactError("experiment_name must not contain path traversal")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ArtifactError("seed must be an integer")
        experiment_dir = root_path / experiment_name
        seed_dir = experiment_dir / f"seed_{seed:03d}"
        experiment_dir.mkdir(parents=True, exist_ok=True)
        seed_dir.mkdir(parents=True, exist_ok=True)
        paths = cls(
            experiment_dir=experiment_dir,
            seed_dir=seed_dir,
            experiment_config=experiment_dir / "experiment_config.yaml",
            run_results_json=seed_dir / "run_results.json",
            summary_csv=experiment_dir / "summary.csv",
            aggregate_metrics_json=experiment_dir / "aggregate_metrics.json",
            config_yaml=seed_dir / "config.yaml",
            status_json=seed_dir / "status.json",
            test_history_csv=seed_dir / "test_history.csv",
            best_checkpoint=seed_dir / "best.ckpt",
            last_checkpoint=seed_dir / "last.ckpt",
            test_predictions_csv=seed_dir / "test_predictions.csv",
        )
        if initialize_status:
            for artifact in (
                paths.config_yaml,
                paths.run_results_json,
                paths.status_json,
                paths.test_history_csv,
                paths.best_checkpoint,
                paths.last_checkpoint,
                paths.test_predictions_csv,
                seed_dir / "split.csv",
                seed_dir / "loss_history.csv",
                seed_dir / "metrics_history.csv",
            ):
                artifact.unlink(missing_ok=True)
            paths.set_status("pending")
        return paths

    @property
    def run_dir(self) -> Path:
        """Alias for the seed-specific directory."""
        return self.seed_dir

    @property
    def best_ckpt(self) -> Path:
        """Alias for :attr:`best_checkpoint`."""
        return self.best_checkpoint

    def write_experiment_config(self, value: object) -> Path:
        _write_yaml_atomic(self.experiment_config, value)
        return self.experiment_config

    def write_config(self, value: object) -> Path:
        _write_yaml_atomic(self.config_yaml, value)
        return self.config_yaml

    def write_split_rows(self, rows: Sequence[Mapping[str, object]]) -> Path:
        payload = _read_run_results(self.run_results_json)
        payload["split"] = _normalize_split_rows(rows)
        _write_json_atomic(self.run_results_json, payload)
        return self.run_results_json

    def write_summary(self, rows: Sequence[Mapping[str, object]]) -> Path:
        """Replace the experiment summary with one row per configured seed."""
        if not rows:
            raise ArtifactError("cannot write an empty experiment summary")
        preferred = ("seed", "split_seed", "status", "error_type", "error_message")
        keys = set().union(*(row.keys() for row in rows))
        header = tuple(item for item in preferred if item in keys) + tuple(
            sorted(keys - set(preferred))
        )
        normalized = [{key: row.get(key, "") for key in header} for row in rows]
        _write_csv_atomic(self.summary_csv, normalized, preferred_header=header)
        return self.summary_csv

    def write_aggregate_metrics(self, value: Mapping[str, object]) -> Path:
        """Write the multi-seed aggregate metrics atomically."""
        _write_json_atomic(self.aggregate_metrics_json, value)
        return self.aggregate_metrics_json

    def append_epoch_history(
        self,
        row: Mapping[str, object] | object,
        *,
        learning_rate: float | None = None,
        epoch_seconds: float | None = None,
    ) -> Path:
        loss = _loss_row(row)
        metrics = _metrics_row(
            row,
            learning_rate=learning_rate,
            epoch_seconds=epoch_seconds,
        )
        loss_epoch = loss.pop("epoch")
        metrics_epoch = metrics.pop("epoch")
        if loss_epoch != metrics_epoch:
            raise ArtifactError("loss and metrics history must use the same epoch")
        payload = _read_run_results(self.run_results_json)
        training = cast(dict[str, object], payload["training"])
        epochs = cast(list[object], training["epochs"])
        epochs.append(
            {
                "epoch": loss_epoch,
                "loss": loss,
                "metrics": metrics,
            }
        )
        _write_json_atomic(self.run_results_json, payload)
        return self.run_results_json

    def write_test_history(
        self, row: Mapping[str, object] | object, **extra: object
    ) -> Path:
        values = _mapping_row(row)
        values.update(extra)
        _append_csv(self.test_history_csv, [values])
        return self.test_history_csv

    def write_test_predictions(
        self,
        result: object,
    ) -> Path:
        predictions = getattr(result, "predictions", None)
        targets = getattr(result, "targets", None)
        mask = getattr(result, "mask", None)
        smiles = getattr(result, "smiles", None)
        if not (
            isinstance(predictions, torch.Tensor)
            and isinstance(targets, torch.Tensor)
            and isinstance(mask, torch.Tensor)
            and isinstance(smiles, tuple)
            and all(isinstance(value, str) for value in smiles)
        ):
            raise ArtifactError(
                "test prediction result must include predictions, targets, mask, and SMILES"
            )
        predictions = cast(torch.Tensor, predictions)
        targets = cast(torch.Tensor, targets)
        mask = cast(torch.Tensor, mask)
        smiles = cast(tuple[str, ...], smiles)
        if predictions.shape != targets.shape or mask.shape != targets.shape:
            raise ArtifactError("test prediction tensors must have identical shapes")
        if len(smiles) != targets.shape[0]:
            raise ArtifactError("SMILES count must match the test sample count")
        rows: list[dict[str, object]] = []
        for sample_index in range(targets.shape[0]):
            target_values = [
                float(targets[sample_index, target_index].item())
                if bool(mask[sample_index, target_index].item())
                else None
                for target_index in range(targets.shape[1])
            ]
            prediction_values = [
                float(predictions[sample_index, target_index].item())
                for target_index in range(predictions.shape[1])
            ]
            if targets.shape[1] == 1:
                target_value: object = (
                    "" if target_values[0] is None else target_values[0]
                )
                prediction_value: object = prediction_values[0]
            else:
                target_value = json.dumps(target_values, separators=(",", ":"))
                prediction_value = json.dumps(prediction_values, separators=(",", ":"))
            rows.append(
                {
                    "smiles": smiles[sample_index],
                    "target": target_value,
                    "prediction": prediction_value,
                }
            )
        _write_csv_atomic(
            self.test_predictions_csv,
            rows,
            preferred_header=("smiles", "target", "prediction"),
        )
        return self.test_predictions_csv

    def set_status(self, status: str, *, error: BaseException | None = None) -> Path:
        if status not in {"pending", "running", "completed", "failed"}:
            raise ArtifactError("status must be pending, running, completed, or failed")
        payload: dict[str, object] = {"status": status}
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error_message"] = str(error)[:500]
        _write_json_atomic(self.status_json, payload)
        return self.status_json

    def mark_running(self) -> Path:
        return self.set_status("running")

    def mark_completed(self) -> Path:
        return self.set_status("completed")

    def mark_failed(self, error: BaseException) -> Path:
        return self.set_status("failed", error=error)


def _empty_run_results() -> dict[str, object]:
    return {
        "schema_version": RUN_RESULTS_SCHEMA_VERSION,
        "split": [],
        "training": {"epochs": []},
    }


def _read_run_results(path: Path) -> dict[str, object]:
    if not path.exists():
        return _empty_run_results()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"failed to read run results '{path}': {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactError("run results must contain a JSON object")
    if payload.get("schema_version") != RUN_RESULTS_SCHEMA_VERSION:
        raise ArtifactError("run results schema version is unsupported")
    split = payload.get("split")
    training = payload.get("training")
    if not isinstance(split, list) or not isinstance(training, dict):
        raise ArtifactError("run results structure is invalid")
    if not isinstance(training.get("epochs"), list):
        raise ArtifactError("run results training.epochs must be a list")
    return payload


def _normalize_split_rows(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    if not rows:
        raise ArtifactError("cannot write an empty run split")
    header = ("dataset_index", "sample_id", "split")
    required_keys = set(header)
    normalized: list[dict[str, object]] = []
    for row in rows:
        if set(row.keys()) != required_keys:
            raise ArtifactError("run split row keys do not match required fields")
        normalized.append({key: _to_builtin(row[key]) for key in header})
    return normalized


def _write_yaml_atomic(path: Path, value: object) -> None:
    _atomic_text_write(
        path, yaml.safe_dump(_to_builtin(value), sort_keys=False, allow_unicode=True)
    )


def _write_json_atomic(path: Path, value: object) -> None:
    _atomic_text_write(
        path, json.dumps(_to_builtin(value), indent=2, sort_keys=True) + "\n"
    )


def write_json_atomic(path: str | Path, value: object) -> Path:
    """Write a standalone JSON artifact with the shared atomic writer."""
    destination = Path(path)
    _write_json_atomic(destination, value)
    return destination


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ArtifactError(f"failed to write artifact '{path}': {exc}") from exc


def _append_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    preferred_header: Sequence[str] | None = None,
) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    row_keys = tuple(rows[0].keys())
    header = tuple(preferred_header) if preferred_header is not None else row_keys
    if set(row_keys) != set(header):
        raise ArtifactError(
            f"CSV row keys do not match required header for '{path.name}'"
        )
    exists = path.is_file() and path.stat().st_size > 0
    if exists:
        with path.open("r", encoding="utf-8", newline="") as stream:
            existing_header = tuple(next(csv.reader(stream), ()))
        if existing_header != header:
            raise ArtifactError(f"CSV header mismatch for '{path.name}'")
    try:
        with path.open("a", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(header), extrasaction="raise"
            )
            if not exists:
                writer.writeheader()
            writer.writerows(
                {key: _to_builtin(row.get(key, "")) for key in header} for row in rows
            )
    except OSError as exc:
        raise ArtifactError(f"failed to append CSV '{path}': {exc}") from exc


def _write_csv_atomic(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    *,
    preferred_header: Sequence[str] | None = None,
) -> None:
    if not rows:
        raise ArtifactError(f"cannot write empty CSV artifact '{path.name}'")
    row_keys = tuple(rows[0].keys())
    header = tuple(preferred_header) if preferred_header is not None else row_keys
    if set(row_keys) != set(header):
        raise ArtifactError(
            f"CSV row keys do not match required header for '{path.name}'"
        )
    for row in rows:
        if set(row.keys()) != set(header):
            raise ArtifactError(
                f"CSV row keys do not match required header for '{path.name}'"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            writer = csv.DictWriter(
                stream, fieldnames=list(header), extrasaction="raise"
            )
            writer.writeheader()
            writer.writerows(
                {key: _to_builtin(row.get(key, "")) for key in header} for row in rows
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ArtifactError(f"failed to write CSV artifact '{path}': {exc}") from exc


def _mapping_row(value: Mapping[str, object] | object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if is_dataclass(value):
        return {str(key): item for key, item in asdict(value).items()}  # type: ignore[arg-type]
    raise ArtifactError("artifact history row must be a mapping or dataclass")


def _loss_row(value: Mapping[str, object] | object) -> dict[str, object]:
    dataclass_row = is_dataclass(value)
    row = _mapping_row(value)
    if "epoch_number" in row:
        row["epoch"] = row.pop("epoch_number")
    else:
        epoch_value = row.get("epoch")
        if dataclass_row and isinstance(epoch_value, int):
            row["epoch"] = epoch_value + 1
    return {
        "epoch": row.get("epoch", ""),
        "train_optimization_loss": row.get("train_optimization_loss", ""),
        "train_eval_loss": row.get("train_eval_loss", ""),
        "val_loss": row.get("val_loss", ""),
    }


def _metrics_row(
    value: Mapping[str, object] | object,
    *,
    learning_rate: float | None,
    epoch_seconds: float | None,
) -> dict[str, object]:
    dataclass_row = is_dataclass(value)
    row = _mapping_row(value)
    nested_metrics = row.pop("metrics", None)
    if isinstance(nested_metrics, Mapping):
        row.update({str(key): item for key, item in nested_metrics.items()})
    if "epoch_number" in row:
        row["epoch"] = row.pop("epoch_number")
    else:
        epoch_value = row.get("epoch")
        if dataclass_row and isinstance(epoch_value, int):
            row["epoch"] = epoch_value + 1
    output: dict[str, object] = {"epoch": row.get("epoch", "")}
    output["learning_rate"] = "" if learning_rate is None else learning_rate
    output["epoch_seconds"] = "" if epoch_seconds is None else epoch_seconds
    loss_keys = {"train_optimization_loss", "train_eval_loss", "train_loss", "val_loss"}
    output.update(
        {
            key: item
            for key, item in row.items()
            if key.startswith(("train_", "val_")) and key not in loss_keys
        }
    )
    return output


def _to_builtin(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return _to_builtin(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


__all__ = [
    "AGGREGATE_METRICS_SCHEMA_VERSION",
    "RUN_RESULTS_SCHEMA_VERSION",
    "ArtifactError",
    "RunPaths",
    "write_json_atomic",
]

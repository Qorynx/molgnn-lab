"""Explicit dataset-source loading at the shared runtime boundary."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from .config import DataConfig, TaskConfig
from .dataset import CsvMoleculeDataset, MolecularDataset, SampleFeaturizer
from .featurizer import FeatureSchema


class DatasetSourceError(ValueError):
    """Raised when a configured dataset source cannot be resolved."""


@dataclass(frozen=True)
class DatasetLoadSummary:
    """Source-neutral counts and source-specific artifact summary fields.

    ``skipped_rows`` is the internal, source-neutral count used by the runner.
    ``artifact_fields`` keeps the serialized CSV metadata stable: the existing
    source continues to expose ``skipped_invalid_smiles`` rather than acquiring
    a new public artifact key.
    """

    source_rows: int
    valid_rows: int
    skipped_rows: int
    num_targets: int
    feature_schema_version: str
    artifact_fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("source_rows", "valid_rows", "skipped_rows", "num_targets"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise DatasetSourceError(
                    f"dataset summary {name} must be a non-negative integer"
                )
        if self.num_targets < 1:
            raise DatasetSourceError("dataset summary num_targets must be positive")
        if self.valid_rows + self.skipped_rows != self.source_rows:
            raise DatasetSourceError(
                "dataset summary source_rows must equal valid_rows plus skipped_rows"
            )
        if (
            not isinstance(self.feature_schema_version, str)
            or not self.feature_schema_version.strip()
        ):
            raise DatasetSourceError(
                "dataset summary feature_schema_version must be a non-empty string"
            )
        if not isinstance(self.artifact_fields, Mapping):
            raise DatasetSourceError("dataset summary artifact_fields must be a mapping")
        forbidden = {"source_rows", "valid_rows"} & set(self.artifact_fields)
        if forbidden:
            names = ", ".join(sorted(forbidden))
            raise DatasetSourceError(
                f"dataset summary artifact_fields cannot override: {names}"
            )
        object.__setattr__(
            self,
            "artifact_fields",
            MappingProxyType(dict(self.artifact_fields)),
        )

    def to_artifact_dict(self) -> dict[str, object]:
        """Return the stable summary representation saved with each run."""

        return {
            "source_rows": self.source_rows,
            "valid_rows": self.valid_rows,
            **self.artifact_fields,
        }


@dataclass(frozen=True)
class DatasetSourceResult:
    """Generic result returned by a dataset source loader.

    The indexable sample collection remains deliberately small and is consumed
    by the shared splitting, batching, task-scaling, and transform lifecycle.
    Source provenance is carried alongside it so the runner does not assume
    that every source is one CSV file with the CSV dataset's attributes.
    """

    dataset: MolecularDataset
    feature_schema: FeatureSchema
    summary: DatasetLoadSummary
    fingerprint: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            dataset_size = len(self.dataset)
        except TypeError as exc:
            raise DatasetSourceError(
                "dataset source result dataset must support len()"
            ) from exc
        if (
            isinstance(dataset_size, bool)
            or not isinstance(dataset_size, int)
            or dataset_size < 1
        ):
            raise DatasetSourceError("dataset source result dataset must be non-empty")
        if not isinstance(self.feature_schema, FeatureSchema):
            raise DatasetSourceError(
                "dataset source result feature_schema must be a FeatureSchema"
            )
        if not isinstance(self.summary, DatasetLoadSummary):
            raise DatasetSourceError(
                "dataset source result summary must be a DatasetLoadSummary"
            )
        if self.summary.valid_rows != dataset_size:
            raise DatasetSourceError(
                "dataset source result summary valid_rows must equal len(dataset)"
            )
        if self.summary.feature_schema_version != self.feature_schema.version:
            raise DatasetSourceError(
                "dataset source result summary feature_schema_version must match "
                "feature_schema"
            )
        if not isinstance(self.fingerprint, str) or not self.fingerprint.strip():
            raise DatasetSourceError(
                "dataset source result fingerprint must be a non-empty string"
            )
        if not isinstance(self.metadata, Mapping):
            raise DatasetSourceError("dataset source result metadata must be a mapping")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class DatasetLoader(Protocol):
    """Callable shape reserved for future dataset-source adapters."""

    def __call__(
        self,
        data_config: DataConfig,
        task_config: TaskConfig,
        *,
        featurizer: SampleFeaturizer | None,
        feature_schema_version: str | None,
    ) -> DatasetSourceResult: ...


_SOURCES: dict[str, DatasetLoader] = {}


def register_dataset_source(name: str, loader: DatasetLoader) -> None:
    """Register one named dataset loader exactly once."""

    clean_name = _name(name)
    if clean_name in _SOURCES:
        raise DatasetSourceError(f"dataset source '{clean_name}' is already registered")
    if not callable(loader):
        raise DatasetSourceError("dataset source loader must be callable")
    _SOURCES[clean_name] = loader


def register_builtin_dataset_sources() -> None:
    """Register built-in sources exactly once."""

    if "csv_smiles" not in _SOURCES:
        _SOURCES["csv_smiles"] = _load_csv_smiles
    if "pdbbind_complex" not in _SOURCES:
        _SOURCES["pdbbind_complex"] = _load_pdbbind_complex


def available_dataset_sources() -> tuple[str, ...]:
    """Return source names in deterministic order."""

    return tuple(sorted(_SOURCES))


def load_dataset(
    data_config: DataConfig,
    task_config: TaskConfig,
    *,
    featurizer: SampleFeaturizer | None = None,
    feature_schema_version: str | None = None,
) -> DatasetSourceResult:
    """Build a generic source result through the configured dataset loader."""

    if not isinstance(data_config, DataConfig):
        raise DatasetSourceError("data_config must be a DataConfig")
    if not isinstance(task_config, TaskConfig):
        raise DatasetSourceError("task_config must be a TaskConfig")
    if featurizer is not None and not callable(featurizer):
        raise DatasetSourceError("featurizer must be callable or None")
    if feature_schema_version is not None and (
        not isinstance(feature_schema_version, str) or not feature_schema_version.strip()
    ):
        raise DatasetSourceError("feature_schema_version must be a non-empty string or None")

    register_builtin_dataset_sources()
    source_name = _name(data_config.source)
    try:
        loader = _SOURCES[source_name]
    except KeyError as exc:
        available = ", ".join(available_dataset_sources()) or "<none>"
        raise DatasetSourceError(
            f"unknown dataset source '{source_name}'. Available sources: {available}"
        ) from exc
    return loader(
        data_config,
        task_config,
        featurizer=featurizer,
        feature_schema_version=feature_schema_version,
    )


def _load_csv_smiles(
    data_config: DataConfig,
    task_config: TaskConfig,
    *,
    featurizer: SampleFeaturizer | None,
    feature_schema_version: str | None,
) -> DatasetSourceResult:
    """Build the existing CSV/SMILES dataset and preserve its public API."""

    dataset = CsvMoleculeDataset(
        data_config.path,
        smiles_column=data_config.smiles_column,
        target_columns=data_config.target_columns,
        task_type=task_config.type,
        invalid_smiles=data_config.invalid_smiles,
        id_column=data_config.id_column,
        split_column=data_config.split_column,
        featurizer=featurizer,
        feature_schema_version=feature_schema_version,
    )
    return DatasetSourceResult(
        dataset=dataset,
        feature_schema=dataset.feature_schema,
        summary=DatasetLoadSummary(
            source_rows=dataset.summary.source_rows,
            valid_rows=dataset.summary.valid_rows,
            skipped_rows=dataset.summary.skipped_invalid_smiles,
            num_targets=dataset.summary.num_targets,
            feature_schema_version=dataset.summary.feature_schema_version,
            artifact_fields={
                "skipped_invalid_smiles": dataset.summary.skipped_invalid_smiles,
            },
        ),
        fingerprint=_file_fingerprint(dataset.path),
    )


def _load_pdbbind_complex(
    data_config: DataConfig,
    task_config: TaskConfig,
    *,
    featurizer: SampleFeaturizer | None,
    feature_schema_version: str | None,
) -> DatasetSourceResult:
    """Load one explicit ligand--pocket manifest without a SMILES hook ABI.

    The existing featurizer hook receives only a SMILES string, targets, and a
    row index.  It cannot safely construct a protein--ligand complex, so this
    source rejects it rather than silently dropping structural information.
    """

    if featurizer is not None or feature_schema_version is not None:
        raise DatasetSourceError(
            "pdbbind_complex does not support the SMILES-only featurizer hook"
        )
    ligand_path_column = data_config.ligand_path_column
    protein_path_column = data_config.protein_path_column
    if ligand_path_column is None or protein_path_column is None:
        raise DatasetSourceError(
            "pdbbind_complex requires ligand_path_column and protein_path_column"
        )
    from .datasets.pdbbind import PDBBindComplexDataset

    dataset = PDBBindComplexDataset(
        data_config.path,
        ligand_path_column=ligand_path_column,
        protein_path_column=protein_path_column,
        target_columns=data_config.target_columns,
        task_type=task_config.type,
        id_column=data_config.id_column,
        split_column=data_config.split_column,
        invalid_complex=data_config.invalid_smiles,
        strip_hydrogens=data_config.strip_hydrogens,
    )
    return DatasetSourceResult(
        dataset=dataset,
        feature_schema=dataset.feature_schema,
        summary=DatasetLoadSummary(
            source_rows=dataset.summary.source_rows,
            valid_rows=dataset.summary.valid_rows,
            skipped_rows=dataset.summary.skipped_invalid_complexes,
            num_targets=dataset.summary.num_targets,
            feature_schema_version=dataset.summary.feature_schema_version,
            artifact_fields={
                "skipped_invalid_complexes": dataset.summary.skipped_invalid_complexes,
            },
        ),
        fingerprint=dataset.fingerprint(),
        metadata={
            "structure_format": "local_pdbbind_manifest",
            "ligand_path_column": ligand_path_column,
            "protein_path_column": protein_path_column,
            "strip_hydrogens": data_config.strip_hydrogens,
        },
    )


def _file_fingerprint(path: Path) -> str:
    """Return a stable SHA-256 fingerprint for one existing source file."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetSourceError("dataset source name must be a non-empty string")
    return value.strip()


__all__ = [
    "DatasetLoadSummary",
    "DatasetLoader",
    "DatasetSourceError",
    "DatasetSourceResult",
    "available_dataset_sources",
    "load_dataset",
    "register_builtin_dataset_sources",
    "register_dataset_source",
]

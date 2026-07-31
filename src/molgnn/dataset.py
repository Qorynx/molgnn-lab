"""CSV-backed molecular dataset and target parsing utilities."""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, cast, overload

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader

from .data import MolecularData, validate_molecular_data
from .featurizer import (
    CANONICAL_FEATURE_SCHEMA_V1,
    FeatureSchema,
    featurize_smiles,
)

if TYPE_CHECKING:
    from .splits import SplitIndices
    from .transforms import GraphTransform

logger = logging.getLogger(__name__)

TaskType = Literal["regression", "binary_classification"]
InvalidSmilesPolicy = Literal["error", "skip"]
SampleFeaturizer = Callable[..., MolecularData]


class DatasetError(ValueError):
    """Raised when a CSV cannot satisfy the molecular dataset contract."""


@dataclass(frozen=True)
class DatasetSummary:
    """Counts and schema metadata captured while loading a CSV dataset."""

    source_rows: int
    valid_rows: int
    skipped_invalid_smiles: int
    num_targets: int
    feature_schema_version: str


@dataclass(frozen=True)
class DataLoaders:
    """The four canonical loaders consumed by the shared training loop."""

    train: PyGDataLoader
    train_eval: PyGDataLoader
    validation: PyGDataLoader
    test: PyGDataLoader


@dataclass(frozen=True)
class PreparedDataset:
    """Immutable sample view prepared once for one model architecture."""

    samples: tuple[MolecularData, ...]

    @overload
    def __getitem__(self, index: int) -> MolecularData: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[MolecularData, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> MolecularData | tuple[MolecularData, ...]:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


class MolecularDataset(Protocol):
    """Indexable dataset contract needed by the shared DataLoader builder."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> MolecularData: ...


class CsvMoleculeDataset(Dataset[MolecularData]):
    """Eager in-memory dataset of canonical or caller-featurized samples.

    ``sample_id`` is always the zero-based source CSV row position.  This
    remains stable when invalid rows are skipped and deliberately does not use
    the optional ``id_column`` as a tensor identifier.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        smiles_column: str = "smiles",
        target_columns: Sequence[str] = ("target",),
        task_type: TaskType = "regression",
        invalid_smiles: InvalidSmilesPolicy = "error",
        id_column: str | None = None,
        split_column: str | None = None,
        schema: FeatureSchema | None = None,
        featurizer: SampleFeaturizer | None = None,
        feature_schema_version: str | None = None,
    ) -> None:
        self.path = Path(path).expanduser().resolve()
        self.smiles_column = _require_column_name(smiles_column, "smiles_column")
        self.target_names = _normalise_target_columns(target_columns)
        self.task_type = _validate_task_type(task_type)
        self.invalid_smiles = _validate_invalid_smiles_policy(invalid_smiles)
        self.id_column = _optional_column_name(id_column, "id_column")
        self.split_column = _optional_column_name(split_column, "split_column")
        if schema is not None and not isinstance(schema, FeatureSchema):
            raise DatasetError("schema must be a FeatureSchema or None")
        if featurizer is not None and not callable(featurizer):
            raise DatasetError("featurizer must be callable or None")
        if feature_schema_version is not None and (
            not isinstance(feature_schema_version, str)
            or not feature_schema_version.strip()
        ):
            raise DatasetError(
                "feature_schema_version must be a non-empty string or None"
            )

        dataframe = _read_and_validate_csv(
            self.path,
            smiles_column=self.smiles_column,
            target_columns=self.target_names,
            id_column=self.id_column,
            split_column=self.split_column,
        )
        samples: list[MolecularData] = []
        sample_ids: list[int] = []
        smiles_values: list[str] = []
        split_labels: list[str | None] = []
        skipped_invalid_smiles = 0

        for source_row_id, (_, row) in enumerate(dataframe.iterrows()):
            raw_smiles = row[self.smiles_column]
            smiles = "" if _is_missing(raw_smiles) else str(raw_smiles).strip()
            molecule = _parse_smiles(smiles)
            if molecule is None:
                message = (
                    f"Invalid SMILES at source row {source_row_id} "
                    f"({self.smiles_column}={smiles!r}) in '{self.path}'"
                )
                if self.invalid_smiles == "error":
                    raise DatasetError(message)
                skipped_invalid_smiles += 1
                continue

            targets, target_mask = _parse_targets(
                row,
                self.target_names,
                task_type=self.task_type,
                source_row_id=source_row_id,
            )
            if featurizer is None:
                data = featurize_smiles(
                    smiles,
                    targets=targets,
                    target_mask=target_mask,
                    sample_id=source_row_id,
                )
            else:
                try:
                    data = featurizer(
                        smiles,
                        targets=targets,
                        target_mask=target_mask,
                        sample_id=source_row_id,
                    )
                except Exception as exc:
                    raise DatasetError(
                        f"Custom featurizer failed at source row {source_row_id}: {exc}"
                    ) from exc
                if not isinstance(data, MolecularData):
                    raise DatasetError(
                        "Custom featurizer must return molgnn.data.MolecularData"
                    )
            data.smiles = smiles
            samples.append(data)
            sample_ids.append(source_row_id)
            smiles_values.append(smiles)
            if self.split_column is None:
                split_labels.append(None)
            else:
                raw_split = row[self.split_column]
                split_labels.append(
                    None if _is_missing(raw_split) else str(raw_split).strip()
                )

        if schema is None:
            schema = (
                CANONICAL_FEATURE_SCHEMA_V1
                if featurizer is None
                else _infer_feature_schema(
                    samples,
                    version=feature_schema_version or "custom_features",
                )
            )
        self.feature_schema = schema
        for data, source_row_id in zip(samples, sample_ids, strict=True):
            try:
                validate_molecular_data(
                    data,
                    schema=self.feature_schema,
                    num_targets=len(self.target_names),
                )
            except (TypeError, ValueError) as exc:
                raise DatasetError(
                    f"Invalid molecular sample at source row {source_row_id}: {exc}"
                ) from exc

        self._samples = samples
        self._sample_ids = tuple(sample_ids)
        self._smiles = tuple(smiles_values)
        self._split_labels = (
            tuple(split_labels) if self.split_column is not None else None
        )
        self.summary = DatasetSummary(
            source_rows=len(dataframe),
            valid_rows=len(samples),
            skipped_invalid_smiles=skipped_invalid_smiles,
            num_targets=len(self.target_names),
            feature_schema_version=self.feature_schema.version,
        )
        if skipped_invalid_smiles:
            logger.warning(
                "Skipped %d invalid SMILES row(s) from %s",
                skipped_invalid_smiles,
                self.path,
            )

    @overload
    def __getitem__(self, index: int) -> MolecularData: ...

    @overload
    def __getitem__(self, index: slice) -> list[MolecularData]: ...

    def __getitem__(self, index: int | slice) -> MolecularData | list[MolecularData]:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def sample_ids(self) -> tuple[int, ...]:
        """Map dataset indices to original zero-based CSV row positions."""
        return self._sample_ids

    @property
    def index_to_sample_id(self) -> tuple[int, ...]:
        """Alias for :attr:`sample_ids` used by split/artifact code."""
        return self._sample_ids

    def sample_id_for_index(self, index: int) -> int:
        """Return the source row id for one dataset index."""
        return self._sample_ids[index]

    @property
    def smiles(self) -> tuple[str, ...]:
        """Canonical input SMILES in dataset index order."""
        return self._smiles

    @property
    def split_labels(self) -> tuple[str | None, ...] | None:
        """Optional predefined split labels in dataset index order."""
        return self._split_labels


def prepare_model_samples(
    dataset: MolecularDataset,
    graph_transform: GraphTransform | None = None,
) -> PreparedDataset:
    """Materialize one reusable canonical or model-transformed sample view."""
    canonical_samples = tuple(dataset[index] for index in range(len(dataset)))
    if graph_transform is None:
        return PreparedDataset(canonical_samples)
    return PreparedDataset(
        tuple(graph_transform(sample) for sample in canonical_samples)
    )


def build_dataloaders(
    prepared: MolecularDataset,
    splits: SplitIndices,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    graph_transform: GraphTransform | None = None,
) -> DataLoaders:
    """Build seeded loaders from a reusable prepared sample view.

    Passing a canonical dataset plus ``graph_transform`` remains supported as a
    temporary compatibility bridge. New orchestration code should call
    :func:`prepare_model_samples` once, then reuse the returned view per seed.
    """
    from .splits import validate_split_indices

    validate_split_indices(splits, len(prepared))
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise DatasetError("batch_size must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise DatasetError("seed must be an integer")
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise DatasetError("num_workers must be a non-negative integer")

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    prepared_view = (
        prepared
        if isinstance(prepared, PreparedDataset) and graph_transform is None
        else prepare_model_samples(prepared, graph_transform)
    )

    def samples(indices: tuple[int, ...]) -> list[MolecularData]:
        return [prepared_view[index] for index in indices]

    train_samples = samples(splits.train)
    return DataLoaders(
        train=PyGDataLoader(
            train_samples,
            batch_size=batch_size,
            shuffle=True,
            generator=train_generator,
            num_workers=num_workers,
            drop_last=False,
        ),
        train_eval=PyGDataLoader(
            samples(splits.train),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
        validation=PyGDataLoader(
            samples(splits.validation),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
        test=PyGDataLoader(
            samples(splits.test),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        ),
    )


def _require_column_name(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_column_name(value: str | None, field: str) -> str | None:
    if value is None:
        return None
    return _require_column_name(value, field)


def _normalise_target_columns(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise DatasetError("target_columns must be a non-empty sequence of strings")
    try:
        result = tuple(_require_column_name(item, "target_columns") for item in value)
    except TypeError as exc:
        raise DatasetError(
            "target_columns must be a non-empty sequence of strings"
        ) from exc
    if not result:
        raise DatasetError("target_columns must not be empty")
    if len(set(result)) != len(result):
        raise DatasetError("target_columns must not contain duplicates")
    return result


def _validate_task_type(value: str) -> TaskType:
    if value not in {"regression", "binary_classification"}:
        raise DatasetError("task_type must be regression or binary_classification")
    return cast(TaskType, value)


def _validate_invalid_smiles_policy(value: str) -> InvalidSmilesPolicy:
    if value not in {"error", "skip"}:
        raise DatasetError("invalid_smiles must be error or skip")
    return cast(InvalidSmilesPolicy, value)


def _read_and_validate_csv(
    path: Path,
    *,
    smiles_column: str,
    target_columns: Sequence[str],
    id_column: str | None,
    split_column: str | None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV dataset does not exist: '{path}'")
    if not path.is_file():
        raise DatasetError(f"CSV dataset path is not a file: '{path}'")
    try:
        dataframe = pd.read_csv(path)
    except (
        OSError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as exc:
        raise DatasetError(f"Cannot read CSV dataset '{path}': {exc}") from exc

    required = {smiles_column, *target_columns}
    if id_column is not None:
        required.add(id_column)
    if split_column is not None:
        required.add(split_column)
    missing = sorted(required - set(dataframe.columns))
    if missing:
        names = ", ".join(missing)
        raise DatasetError(f"Missing required CSV column(s) in '{path}': {names}")
    return dataframe


def _is_missing(value: object) -> bool:
    result = pd.isna(value)
    return bool(result) if isinstance(result, bool) else False


def _parse_smiles(smiles: str):
    if not smiles:
        return None
    try:
        from rdkit import Chem

        return Chem.MolFromSmiles(smiles)
    except (TypeError, ValueError):
        return None


def _parse_targets(
    row: pd.Series,
    target_columns: Sequence[str],
    *,
    task_type: TaskType,
    source_row_id: int,
) -> tuple[list[float], list[bool]]:
    targets: list[float] = []
    target_mask: list[bool] = []
    for column in target_columns:
        raw_value = row[column]
        if _is_missing(raw_value) or (
            isinstance(raw_value, str) and not raw_value.strip()
        ):
            targets.append(0.0)
            target_mask.append(False)
            continue
        try:
            if isinstance(raw_value, bool):
                raise ValueError
            parsed = float(str(raw_value).strip())
        except (TypeError, ValueError) as exc:
            raise DatasetError(
                f"Invalid target at source row {source_row_id}, column '{column}': {raw_value!r}"
            ) from exc
        if not math.isfinite(parsed):
            raise DatasetError(
                f"Invalid non-finite target at source row {source_row_id}, column '{column}'"
            )
        if task_type == "binary_classification" and parsed not in {0.0, 1.0}:
            raise DatasetError(
                f"Classification target at source row {source_row_id}, column '{column}' "
                f"must be 0 or 1, got {raw_value!r}"
            )
        targets.append(parsed)
        target_mask.append(True)
    return targets, target_mask


def _infer_feature_schema(
    samples: Sequence[MolecularData], *, version: str
) -> FeatureSchema:
    """Infer a stable feature width contract from custom featurizer output."""

    if not samples:
        raise DatasetError("Custom featurizer produced no valid molecular samples")
    first = samples[0]
    x = getattr(first, "x", None)
    edge_attr = getattr(first, "edge_attr", None)
    if (
        not isinstance(x, torch.Tensor)
        or x.ndim != 2
        or x.shape[1] < 1
        or not isinstance(edge_attr, torch.Tensor)
        or edge_attr.ndim != 2
        or edge_attr.shape[1] < 1
    ):
        raise DatasetError(
            "Custom featurizer output must provide x and edge_attr matrices with "
            "positive feature widths"
        )
    return FeatureSchema(
        version=version,
        atom_dim=int(x.shape[1]),
        bond_dim=int(edge_attr.shape[1]),
    )


__all__ = [
    "CsvMoleculeDataset",
    "DataLoaders",
    "DatasetError",
    "DatasetSummary",
    "MolecularDataset",
    "PreparedDataset",
    "SampleFeaturizer",
    "build_dataloaders",
    "prepare_model_samples",
]

"""Train/validation/test split implementations."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .config import DataConfig
from .dataset import MolecularDataset


class SplitError(ValueError):
    """Raised when a split cannot satisfy the dataset split contract."""


class SplitDataset(MolecularDataset, Protocol):
    """Dataset metadata required by the shared split and split-artifact path."""

    @property
    def sample_ids(self) -> tuple[int, ...]: ...

    @property
    def smiles(self) -> tuple[str, ...]: ...

    @property
    def split_labels(self) -> tuple[str | None, ...] | None: ...

    def sample_id_for_index(self, index: int) -> int: ...


@dataclass(frozen=True)
class SplitIndices:
    """Dataset-index assignments for the three canonical partitions."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


def random_split(
    dataset: MolecularDataset,
    split_ratios: Sequence[float],
    seed: int,
) -> SplitIndices:
    """Create a seeded random split over valid dataset indices."""
    sizes = _split_sizes(len(dataset), split_ratios)
    indices = list(range(len(dataset)))
    random.Random(_validate_seed(seed)).shuffle(indices)
    split = SplitIndices(
        train=tuple(indices[: sizes[0]]),
        validation=tuple(indices[sizes[0] : sizes[0] + sizes[1]]),
        test=tuple(indices[sizes[0] + sizes[1] :]),
    )
    validate_split_indices(split, dataset)
    return split


def scaffold_split(
    dataset: SplitDataset,
    split_ratios: Sequence[float],
    seed: int | None = None,
) -> SplitIndices:
    """Create Chemprop's seeded Bemis-Murcko scaffold split.

    This follows Chemprop's ``SCAFFOLD_BALANCED`` implementation by delegating
    to astartes' molecular scaffold sampler. Molecules sharing a scaffold stay
    together, while scaffold clusters small enough for the holdout partitions
    are shuffled reproducibly with ``seed``. Acyclic molecules share the empty
    Bemis-Murcko scaffold and therefore remain in one partition.
    """
    ratio_values = tuple(split_ratios)
    _split_sizes(len(dataset), ratio_values)
    effective_seed = 0 if seed is None else _validate_seed(seed)
    train_ratio, validation_ratio, test_ratio = (float(value) for value in ratio_values)

    from astartes import train_val_test_split
    from rdkit import Chem

    molecules: list[object] = []
    for smiles in dataset.smiles:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise SplitError(f"Cannot compute scaffold for valid SMILES {smiles!r}")
        # Chemprop removes atom-map numbers before deriving scaffolds.
        for atom in molecule.GetAtoms():
            atom.SetAtomMapNum(0)
        molecules.append(molecule)

    try:
        result = train_val_test_split(
            np.asarray(molecules, dtype=object),
            sampler="scaffold",
            train_size=train_ratio,
            val_size=validation_ratio,
            test_size=test_ratio,
            return_indices=True,
            random_state=effective_seed,
        )
    except Exception as exc:
        raise SplitError(f"Chemprop scaffold split failed: {exc}") from exc

    train_indices, validation_indices, test_indices = result[-3:]
    split = SplitIndices(
        train=tuple(int(index) for index in train_indices),
        validation=tuple(int(index) for index in validation_indices),
        test=tuple(int(index) for index in test_indices),
    )
    validate_split_indices(split, dataset)
    return split


def predefined_split(
    dataset: SplitDataset,
    assignments: Sequence[str] | Mapping[int, str] | None = None,
) -> SplitIndices:
    """Build a split from explicit train/validation/test labels.

    When ``assignments`` is omitted, labels are read from the dataset's
    configured ``split_column``. Mapping keys may be dataset indices or stable
    source sample ids; sequence values are always in dataset-index order.
    """
    labels = _resolve_assignments(dataset, assignments)
    partitions: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for index, raw_label in enumerate(labels):
        label = _normalise_split_label(raw_label)
        partitions[label].append(index)

    split = SplitIndices(
        train=tuple(partitions["train"]),
        validation=tuple(partitions["validation"]),
        test=tuple(partitions["test"]),
    )
    validate_split_indices(split, dataset)
    return split


def make_split(dataset: SplitDataset, config: DataConfig, seed: int) -> SplitIndices:
    """Dispatch to the split strategy declared by :class:`DataConfig`."""
    if config.split == "random":
        return random_split(dataset, config.split_ratios, seed)
    if config.split == "scaffold":
        return scaffold_split(dataset, config.split_ratios, seed)
    if config.split == "predefined":
        return predefined_split(dataset)
    raise SplitError(f"Unsupported split strategy: {config.split!r}")


def validate_split_indices(split: SplitIndices, dataset: MolecularDataset | int) -> None:
    """Validate bounds, coverage, uniqueness and non-empty partitions."""
    if not isinstance(split, SplitIndices):
        raise SplitError("split must be a SplitIndices instance")
    dataset_size = len(dataset) if not isinstance(dataset, int) else dataset
    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int) or dataset_size < 1:
        raise SplitError("dataset size must be a positive integer")

    seen: set[int] = set()
    for name, values in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        if not values:
            raise SplitError(f"split '{name}' must not be empty")
        local: set[int] = set()
        for index in values:
            if isinstance(index, bool) or not isinstance(index, int):
                raise SplitError(f"split '{name}' contains a non-integer index: {index!r}")
            if index < 0 or index >= dataset_size:
                raise SplitError(
                    f"split '{name}' contains index {index} outside [0, {dataset_size})"
                )
            if index in local:
                raise SplitError(f"split '{name}' contains duplicate index {index}")
            if index in seen:
                raise SplitError(f"split index {index} appears in more than one partition")
            local.add(index)
            seen.add(index)

    expected = set(range(dataset_size))
    if seen != expected:
        missing = sorted(expected - seen)
        raise SplitError(f"split does not cover all dataset indices; missing {missing}")


def split_rows(split: SplitIndices, dataset: SplitDataset) -> list[dict[str, int | str]]:
    """Return stable per-sample split records for artifact writers."""
    validate_split_indices(split, dataset)
    labels: dict[int, str] = {}
    for name, indices in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        labels.update({index: name for index in indices})
    return [
        {
            "dataset_index": index,
            "sample_id": dataset.sample_id_for_index(index),
            "split": labels[index],
        }
        for index in range(len(dataset))
    ]


def _resolve_assignments(
    dataset: SplitDataset,
    assignments: Sequence[str] | Mapping[int, str] | None,
) -> tuple[str, ...]:
    if assignments is None:
        if dataset.split_labels is None:
            raise SplitError("predefined split requires assignments or a dataset split_column")
        if any(label is None for label in dataset.split_labels):
            raise SplitError("predefined split contains missing split labels")
        return tuple(label for label in dataset.split_labels if label is not None)

    if isinstance(assignments, Mapping):
        values: list[str] = []
        for index, sample_id in enumerate(dataset.sample_ids):
            if index in assignments:
                values.append(assignments[index])
            elif sample_id in assignments:
                values.append(assignments[sample_id])
            else:
                raise SplitError(f"predefined assignments missing dataset index {index}")
        return tuple(values)

    if isinstance(assignments, (str, bytes)):
        raise SplitError("predefined assignments must have one label per dataset sample")
    try:
        assignment_count = len(assignments)
    except TypeError as exc:
        raise SplitError("predefined assignments must have one label per dataset sample") from exc
    if assignment_count != len(dataset):
        raise SplitError("predefined assignments must have one label per dataset sample")
    return tuple(assignments)


def _normalise_split_label(value: object) -> str:
    if not isinstance(value, str):
        raise SplitError(f"invalid predefined split label: {value!r}")
    normalised = value.strip().lower()
    aliases = {"val": "validation", "valid": "validation"}
    normalised = aliases.get(normalised, normalised)
    if normalised not in {"train", "validation", "test"}:
        raise SplitError(
            f"invalid predefined split label {value!r}; expected train, validation, or test"
        )
    return normalised


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitError("seed must be an integer")
    return seed


def _split_sizes(dataset_size: int, split_ratios: Sequence[float]) -> tuple[int, int, int]:
    if dataset_size < 3:
        raise SplitError("at least 3 valid samples are required for a three-way split")
    if isinstance(split_ratios, (str, bytes)):
        raise SplitError("split_ratios must contain exactly three values")
    try:
        ratio_values = tuple(split_ratios)
    except TypeError as exc:
        raise SplitError("split_ratios must contain exactly three values") from exc
    if len(ratio_values) != 3:
        raise SplitError("split_ratios must contain exactly three values")
    try:
        ratios = tuple(
            float(value) if not isinstance(value, bool) else float("nan") for value in ratio_values
        )
    except (TypeError, ValueError) as exc:
        raise SplitError("split_ratios must contain three numeric values") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
        raise SplitError("split_ratios must contain three positive finite values")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise SplitError("split_ratios must sum to 1.0")

    raw = [dataset_size * ratio for ratio in ratios]
    sizes = [math.floor(value) for value in raw]
    remainder = dataset_size - sum(sizes)
    order = sorted(range(3), key=lambda index: (-(raw[index] - sizes[index]), index))
    for index in order[:remainder]:
        sizes[index] += 1

    for empty_index, size in enumerate(sizes):
        if size != 0:
            continue
        donors = [index for index, donor_size in enumerate(sizes) if donor_size > 1]
        if not donors:
            raise SplitError("split ratios produce an empty partition; choose different ratios")
        donor = max(donors, key=lambda index: (sizes[index], -index))
        sizes[donor] -= 1
        sizes[empty_index] = 1
    return tuple(sizes)  # type: ignore[return-value]


__all__ = [
    "SplitError",
    "SplitDataset",
    "SplitIndices",
    "make_split",
    "predefined_split",
    "random_split",
    "scaffold_split",
    "split_rows",
    "validate_split_indices",
]

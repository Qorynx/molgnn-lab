"""Native-coordinate QM9 adapter preserving the shared molecular schema."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import overload

import torch
from rdkit import Chem
from torch.utils.data import Dataset
from torch_geometric.datasets import QM9 as PyGQM9

from ..data import MolecularData, validate_molecular_data
from ..dataset import DatasetError
from ..featurizer import CANONICAL_FEATURE_SCHEMA_V1, featurize_mol

QM9_TARGETS = (
    "mu",
    "alpha",
    "homo",
    "lumo",
    "gap",
    "r2",
    "zpve",
    "u0",
    "u",
    "h",
    "g",
    "cv",
    "u0_atom",
    "u_atom",
    "h_atom",
    "g_atom",
    "a",
    "b",
    "c",
)


class QM9DatasetError(DatasetError):
    """Raised when native QM9 records cannot satisfy the shared contract."""


class QM9MolecularDataset(Dataset[MolecularData]):
    """Lazy canonical view over PyG QM9 with exact SDF coordinates."""

    def __init__(self, root: str | Path, *, target_columns: Sequence[str]) -> None:
        self.path = Path(root).expanduser().resolve()
        self.target_names = _target_columns(target_columns)
        self._target_indices = tuple(
            QM9_TARGETS.index(name.lower()) for name in self.target_names
        )
        self.feature_schema = CANONICAL_FEATURE_SCHEMA_V1
        self._source = PyGQM9(str(self.path))
        if len(self._source) < 1:
            raise QM9DatasetError("QM9 source is empty")
        raw_sdf = Path(self._source.raw_paths[0]).resolve()
        if not raw_sdf.is_file():
            raise QM9DatasetError(f"QM9 raw SDF is unavailable: '{raw_sdf}'")
        self._supplier = Chem.SDMolSupplier(
            str(raw_sdf), removeHs=False, sanitize=False
        )
        self._sample_ids = tuple(
            int(self._source.get(index).idx) for index in range(len(self._source))
        )
        self._smiles = tuple(
            str(self._source.get(index).smiles) for index in range(len(self._source))
        )

    @overload
    def __getitem__(self, index: int) -> MolecularData: ...

    @overload
    def __getitem__(self, index: slice) -> list[MolecularData]: ...

    def __getitem__(self, index: int | slice) -> MolecularData | list[MolecularData]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        record = self._source.get(index)
        source_index = int(record.idx)
        mol = self._supplier[source_index]
        if mol is None:
            raise QM9DatasetError(f"QM9 SDF molecule {source_index} could not be read")
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            raise QM9DatasetError(
                f"QM9 SDF molecule {source_index} could not be sanitized: {exc}"
            ) from exc
        target_indices = torch.tensor(self._target_indices, dtype=torch.long)
        targets = record.y.reshape(-1)[target_indices].to(torch.float32)
        data = featurize_mol(
            mol,
            targets=targets,
            target_mask=torch.ones(len(self.target_names), dtype=torch.bool),
            sample_id=source_index,
            smiles=str(record.smiles),
        )
        atomic_number = record.z.to(dtype=torch.long, device="cpu")
        pos = record.pos.to(dtype=torch.float32, device="cpu")
        if atomic_number.shape[0] != data.x.shape[0] or not torch.equal(
            atomic_number,
            torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
        ):
            raise QM9DatasetError(
                f"QM9 SDF/PyG atom order mismatch at source molecule {source_index}"
            )
        data.atomic_number = atomic_number
        data.pos = pos
        validate_molecular_data(data, self.feature_schema, len(self.target_names))
        return data

    def __len__(self) -> int:
        return len(self._source)

    @property
    def sample_ids(self) -> tuple[int, ...]:
        return self._sample_ids

    @property
    def index_to_sample_id(self) -> tuple[int, ...]:
        return self._sample_ids

    def sample_id_for_index(self, index: int) -> int:
        return self._sample_ids[index]

    @property
    def smiles(self) -> tuple[str, ...]:
        return self._smiles

    @property
    def split_labels(self) -> None:
        return None

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        processed_path = Path(self._source.processed_paths[0])
        with processed_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _target_columns(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise QM9DatasetError("QM9 target_columns must be a sequence")
    result = tuple(str(value).strip() for value in values)
    if not result or any(not value for value in result):
        raise QM9DatasetError("QM9 target_columns must contain non-empty names")
    unknown = [value for value in result if value.lower() not in QM9_TARGETS]
    if unknown:
        raise QM9DatasetError(
            "unknown QM9 target(s): "
            + ", ".join(unknown)
            + "; available: "
            + ", ".join(QM9_TARGETS)
        )
    if len({value.lower() for value in result}) != len(result):
        raise QM9DatasetError("QM9 target_columns must not contain duplicates")
    return result


__all__ = ["QM9_TARGETS", "QM9DatasetError", "QM9MolecularDataset"]

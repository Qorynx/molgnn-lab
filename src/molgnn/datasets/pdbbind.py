"""Local PDBBind-style complex loading for the full PotentialNet input path.

This adapter intentionally consumes an explicit CSV manifest rather than
downloading or assuming a particular PDBBind directory layout.  One manifest
row becomes one combined ligand--pocket ``MolecularData`` sample.  Its
five-channel atom/bond feature profile matches the DGL-LifeSci PotentialNet
reference and is labelled as such; it is not presented as a paper-mandated
feature schema.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from torch import Tensor
from torch.utils.data import Dataset

from ..data import MolecularData, validate_molecular_data
from ..dataset import DatasetError, TaskType
from ..featurizer import FeatureSchema


_DGL_ATOM_TYPES = (
    "H",
    "N",
    "O",
    "C",
    "P",
    "S",
    "F",
    "Br",
    "Cl",
    "I",
    "Fe",
    "Zn",
    "Mg",
    "Na",
    "Mn",
    "Ca",
    "Co",
    "Ni",
    "Se",
    "Cu",
    "Cd",
    "Hg",
    "K",
)
_DGL_TOTAL_DEGREES = tuple(range(5))
_DGL_FORMAL_CHARGES = (-1, 0, 1)
_DGL_IMPLICIT_VALENCES = tuple(range(4))
_DGL_EXPLICIT_VALENCES = tuple(range(8))
_BOND_TYPES = (BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC)

POTENTIALNET_DGL_LIFESCI_FEATURE_SCHEMA_V1 = FeatureSchema(
    version="potentialnet_dgl_lifesci_44_v1",
    atom_dim=(
        len(_DGL_ATOM_TYPES)
        + len(_DGL_TOTAL_DEGREES)
        + len(_DGL_FORMAL_CHARGES)
        + 1
        + len(_DGL_IMPLICIT_VALENCES)
        + len(_DGL_EXPLICIT_VALENCES)
    ),
    bond_dim=len(_BOND_TYPES) + 1,
)


class PDBBindDatasetError(DatasetError):
    """Raised when an explicit complex manifest cannot be materialized."""


InvalidComplexPolicy = Literal["error", "skip"]


@dataclass(frozen=True)
class PDBBindDatasetSummary:
    """Loading counts retained by the PDBBind-style source adapter."""

    source_rows: int
    valid_rows: int
    skipped_invalid_complexes: int
    num_targets: int
    feature_schema_version: str


class PDBBindComplexDataset(Dataset[MolecularData]):
    """Eager local complex dataset with stable source-row sample identifiers.

    ``ligand_path_column`` may point to ``.sdf``, ``.mol``, or ``.mol2``
    files. ``protein_path_column`` points to the already selected pocket in
    PDB format.  Relative paths resolve against the manifest's directory.
    Explicit hydrogens are consistently removed from both graph and coordinate
    tensors when ``strip_hydrogens`` is enabled.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        ligand_path_column: str = "ligand_path",
        protein_path_column: str = "protein_path",
        target_columns: Sequence[str] = ("target",),
        task_type: TaskType = "regression",
        id_column: str | None = "complex_id",
        split_column: str | None = None,
        invalid_complex: InvalidComplexPolicy = "error",
        strip_hydrogens: bool = True,
    ) -> None:
        self.path = Path(manifest_path).expanduser().resolve()
        self.ligand_path_column = _column_name(ligand_path_column, "ligand_path_column")
        self.protein_path_column = _column_name(protein_path_column, "protein_path_column")
        self.target_names = _target_columns(target_columns)
        self.task_type = _task_type(task_type)
        self.id_column = _optional_column_name(id_column, "id_column")
        self.split_column = _optional_column_name(split_column, "split_column")
        self.invalid_complex = _invalid_complex_policy(invalid_complex)
        if not isinstance(strip_hydrogens, bool):
            raise PDBBindDatasetError("strip_hydrogens must be a boolean")
        self.strip_hydrogens = strip_hydrogens

        dataframe = _read_manifest(
            self.path,
            ligand_path_column=self.ligand_path_column,
            protein_path_column=self.protein_path_column,
            target_columns=self.target_names,
            id_column=self.id_column,
            split_column=self.split_column,
        )
        samples: list[MolecularData] = []
        sample_ids: list[int] = []
        smiles_values: list[str] = []
        split_labels: list[str | None] = []
        complex_ids: list[str] = []
        structural_files: set[Path] = {self.path}
        skipped = 0

        for source_row_id, (_, row) in enumerate(dataframe.iterrows()):
            try:
                ligand_path = _manifest_file(
                    row[self.ligand_path_column], self.path.parent, "ligand"
                )
                protein_path = _manifest_file(
                    row[self.protein_path_column], self.path.parent, "protein"
                )
                targets, target_mask = _targets(
                    row,
                    self.target_names,
                    task_type=self.task_type,
                    source_row_id=source_row_id,
                )
                ligand = _load_ligand(ligand_path)
                protein = _load_protein(protein_path)
                complex_id = _complex_id(row, self.id_column, source_row_id)
                data, ligand_smiles = _complex_data(
                    ligand,
                    protein,
                    targets=targets,
                    target_mask=target_mask,
                    sample_id=source_row_id,
                    complex_id=complex_id,
                    strip_hydrogens=self.strip_hydrogens,
                )
            except Exception as exc:
                message = (
                    f"Invalid complex at source row {source_row_id} in '{self.path}': {exc}"
                )
                if self.invalid_complex == "error":
                    raise PDBBindDatasetError(message) from exc
                skipped += 1
                continue

            try:
                validate_molecular_data(
                    data,
                    schema=POTENTIALNET_DGL_LIFESCI_FEATURE_SCHEMA_V1,
                    num_targets=len(self.target_names),
                )
            except (TypeError, ValueError) as exc:
                raise PDBBindDatasetError(
                    f"Invalid constructed complex at source row {source_row_id}: {exc}"
                ) from exc
            samples.append(data)
            sample_ids.append(source_row_id)
            smiles_values.append(ligand_smiles)
            split_labels.append(
                None
                if self.split_column is None or _missing(row[self.split_column])
                else str(row[self.split_column]).strip()
            )
            complex_ids.append(complex_id)
            structural_files.update((ligand_path, protein_path))

        if not samples:
            raise PDBBindDatasetError(f"No valid complexes found in '{self.path}'")

        self.feature_schema = POTENTIALNET_DGL_LIFESCI_FEATURE_SCHEMA_V1
        self._samples = tuple(samples)
        self._sample_ids = tuple(sample_ids)
        self._smiles = tuple(smiles_values)
        self._complex_ids = tuple(complex_ids)
        self._split_labels = (
            tuple(split_labels) if self.split_column is not None else None
        )
        self._referenced_files = tuple(sorted(structural_files))
        self.summary = PDBBindDatasetSummary(
            source_rows=len(dataframe),
            valid_rows=len(samples),
            skipped_invalid_complexes=skipped,
            num_targets=len(self.target_names),
            feature_schema_version=self.feature_schema.version,
        )

    def __getitem__(self, index: int) -> MolecularData:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

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
        """Canonical ligand SMILES used by the existing split/artifact path."""

        return self._smiles

    @property
    def complex_ids(self) -> tuple[str, ...]:
        return self._complex_ids

    @property
    def split_labels(self) -> tuple[str | None, ...] | None:
        return self._split_labels

    @property
    def referenced_files(self) -> tuple[Path, ...]:
        return self._referenced_files

    def fingerprint(self) -> str:
        """Hash the manifest and every structural file actually consumed."""

        return fingerprint_files(self._referenced_files)


def fingerprint_files(paths: Iterable[Path]) -> str:
    """Return a content-sensitive deterministic fingerprint for local files."""

    digest = hashlib.sha256()
    resolved = sorted({Path(path).expanduser().resolve() for path in paths})
    for path in resolved:
        if not path.is_file():
            raise PDBBindDatasetError(f"Referenced structural file is missing: '{path}'")
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(
    path: Path,
    *,
    ligand_path_column: str,
    protein_path_column: str,
    target_columns: Sequence[str],
    id_column: str | None,
    split_column: str | None,
) -> pd.DataFrame:
    if not path.is_file():
        raise PDBBindDatasetError(f"Complex manifest is not a file: '{path}'")
    try:
        dataframe = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise PDBBindDatasetError(f"Cannot read complex manifest '{path}': {exc}") from exc
    required = {ligand_path_column, protein_path_column, *target_columns}
    if id_column is not None:
        required.add(id_column)
    if split_column is not None:
        required.add(split_column)
    missing = sorted(required - set(dataframe.columns))
    if missing:
        names = ", ".join(missing)
        raise PDBBindDatasetError(f"Missing required manifest column(s) in '{path}': {names}")
    return dataframe


def _load_ligand(path: Path) -> Chem.Mol:
    suffix = path.suffix.lower()
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        molecule = next((item for item in supplier if item is not None), None)
    elif suffix == ".mol2":
        molecule = Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    else:
        molecule = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=False)
    if molecule is None:
        raise PDBBindDatasetError(f"Cannot parse ligand structure '{path}'")
    return molecule


def _load_protein(path: Path) -> Chem.Mol:
    if path.suffix.lower() not in {".pdb", ".ent"}:
        raise PDBBindDatasetError(
            f"Protein/pocket structure must be PDB-format, got '{path.suffix}'"
        )
    molecule = Chem.MolFromPDBFile(
        str(path), sanitize=False, removeHs=False, proximityBonding=True
    )
    if molecule is None:
        raise PDBBindDatasetError(f"Cannot parse protein/pocket structure '{path}'")
    try:
        molecule.UpdatePropertyCache(strict=False)
    except RuntimeError:
        # Atom-level helpers below independently fall back for unavailable
        # valence metadata in imperfect PDB files.
        pass
    return molecule


def _complex_data(
    ligand: Chem.Mol,
    protein: Chem.Mol,
    *,
    targets: Sequence[float],
    target_mask: Sequence[bool],
    sample_id: int,
    complex_id: str,
    strip_hydrogens: bool,
) -> tuple[MolecularData, str]:
    ligand_indices = _atom_indices(ligand, strip_hydrogens)
    protein_indices = _atom_indices(protein, strip_hydrogens)
    if not ligand_indices:
        raise PDBBindDatasetError("ligand has no retained atoms")
    if not protein_indices:
        raise PDBBindDatasetError("protein/pocket has no retained atoms")

    ligand_x, ligand_pos = _node_tensors(ligand, ligand_indices)
    protein_x, protein_pos = _node_tensors(protein, protein_indices)
    ligand_edge_index, ligand_edge_attr = _bond_tensors(ligand, ligand_indices, 0)
    protein_edge_index, protein_edge_attr = _bond_tensors(
        protein, protein_indices, len(ligand_indices)
    )
    x = torch.cat((ligand_x, protein_x), dim=0)
    pos = torch.cat((ligand_pos, protein_pos), dim=0)
    edge_index = torch.cat((ligand_edge_index, protein_edge_index), dim=1)
    edge_attr = torch.cat((ligand_edge_attr, protein_edge_attr), dim=0)
    ligand_mask = torch.zeros(x.shape[0], dtype=torch.bool)
    ligand_mask[: len(ligand_indices)] = True
    y = torch.as_tensor(targets, dtype=torch.float32).reshape(1, -1)
    y_mask = torch.as_tensor(target_mask, dtype=torch.bool).reshape(1, -1)
    smiles = _ligand_smiles(ligand)
    data = MolecularData(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        y_mask=y_mask,
        sample_id=torch.tensor([sample_id], dtype=torch.long),
        pos=pos,
        ligand_mask=ligand_mask,
    )
    data.smiles = smiles
    data.complex_id = complex_id
    return data, smiles


def _atom_indices(molecule: Chem.Mol, strip_hydrogens: bool) -> tuple[int, ...]:
    return tuple(
        index
        for index, atom in enumerate(molecule.GetAtoms())
        if not strip_hydrogens or atom.GetAtomicNum() != 1
    )


def _node_tensors(molecule: Chem.Mol, indices: Sequence[int]) -> tuple[Tensor, Tensor]:
    if not molecule.GetNumConformers():
        raise PDBBindDatasetError("structure has no 3D conformer")
    conformer = molecule.GetConformer()
    features: list[Tensor] = []
    positions: list[tuple[float, float, float]] = []
    for index in indices:
        atom = molecule.GetAtomWithIdx(index)
        features.append(_atom_features(atom))
        point = conformer.GetAtomPosition(index)
        positions.append((float(point.x), float(point.y), float(point.z)))
    return torch.stack(features), torch.tensor(positions, dtype=torch.float32)


def _bond_tensors(
    molecule: Chem.Mol, indices: Sequence[int], offset: int
) -> tuple[Tensor, Tensor]:
    retained = {atom_index: position + offset for position, atom_index in enumerate(indices)}
    pairs: list[tuple[int, int]] = []
    features: list[Tensor] = []
    for bond in molecule.GetBonds():
        source = retained.get(bond.GetBeginAtomIdx())
        target = retained.get(bond.GetEndAtomIdx())
        if source is None or target is None:
            continue
        feature = _bond_features(bond)
        pairs.extend(((source, target), (target, source)))
        features.extend((feature, feature.clone()))
    if not pairs:
        return (
            torch.empty((2, 0), dtype=torch.long),
            torch.empty(
                (0, POTENTIALNET_DGL_LIFESCI_FEATURE_SCHEMA_V1.bond_dim),
                dtype=torch.float32,
            ),
        )
    return torch.tensor(pairs, dtype=torch.long).t().contiguous(), torch.stack(features)


def _atom_features(atom: Chem.Atom) -> Tensor:
    values = [
        *_one_hot(_safe_atom_value(atom, "GetSymbol", ""), _DGL_ATOM_TYPES),
        *_one_hot(_safe_atom_value(atom, "GetTotalDegree", -1), _DGL_TOTAL_DEGREES),
        *_one_hot(_safe_atom_value(atom, "GetFormalCharge", -99), _DGL_FORMAL_CHARGES),
        float(bool(_safe_atom_value(atom, "GetIsAromatic", False))),
        *_one_hot(
            _safe_atom_value(atom, "GetImplicitValence", -1), _DGL_IMPLICIT_VALENCES
        ),
        *_one_hot(
            _safe_atom_value(atom, "GetExplicitValence", -1), _DGL_EXPLICIT_VALENCES
        ),
    ]
    feature = torch.tensor(values, dtype=torch.float32)
    if feature.shape != (POTENTIALNET_DGL_LIFESCI_FEATURE_SCHEMA_V1.atom_dim,):
        raise AssertionError("PotentialNet atom feature dimension mismatch")
    return feature


def _bond_features(bond: Chem.Bond) -> Tensor:
    values = [
        *_one_hot(bond.GetBondType(), _BOND_TYPES),
        float(bond.IsInRing()),
    ]
    return torch.tensor(values, dtype=torch.float32)


def _one_hot(value: object, vocabulary: Sequence[object]) -> tuple[float, ...]:
    return tuple(float(value == candidate) for candidate in vocabulary)


def _safe_atom_value(atom: Chem.Atom, method: str, fallback: object) -> object:
    try:
        return cast(object, getattr(atom, method)())
    except RuntimeError:
        return fallback


def _ligand_smiles(molecule: Chem.Mol) -> str:
    try:
        value = Chem.MolToSmiles(molecule, canonical=True)
    except (RuntimeError, ValueError) as exc:
        raise PDBBindDatasetError("could not derive canonical ligand SMILES") from exc
    if not value:
        raise PDBBindDatasetError("ligand canonical SMILES is empty")
    return value


def _targets(
    row: pd.Series,
    target_columns: Sequence[str],
    *,
    task_type: TaskType,
    source_row_id: int,
) -> tuple[list[float], list[bool]]:
    values: list[float] = []
    mask: list[bool] = []
    for column in target_columns:
        raw = row[column]
        if _missing(raw) or (isinstance(raw, str) and not raw.strip()):
            values.append(0.0)
            mask.append(False)
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise PDBBindDatasetError(
                f"target '{column}' at source row {source_row_id} is not numeric"
            ) from exc
        if not math.isfinite(value):
            raise PDBBindDatasetError(
                f"target '{column}' at source row {source_row_id} must be finite"
            )
        if task_type == "binary_classification" and value not in {0.0, 1.0}:
            raise PDBBindDatasetError(
                f"binary target '{column}' at source row {source_row_id} must be 0 or 1"
            )
        values.append(value)
        mask.append(True)
    return values, mask


def _manifest_file(value: object, directory: Path, kind: str) -> Path:
    if _missing(value) or not str(value).strip():
        raise PDBBindDatasetError(f"{kind} path is empty")
    path = Path(str(value).strip())
    path = path if path.is_absolute() else directory / path
    path = path.expanduser().resolve()
    if not path.is_file():
        raise PDBBindDatasetError(f"{kind} structure file does not exist: '{path}'")
    return path


def _complex_id(row: pd.Series, column: str | None, source_row_id: int) -> str:
    if column is None or _missing(row[column]) or not str(row[column]).strip():
        return str(source_row_id)
    return str(row[column]).strip()


def _column_name(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PDBBindDatasetError(f"{field} must be a non-empty string")
    return value.strip()


def _optional_column_name(value: str | None, field: str) -> str | None:
    return None if value is None else _column_name(value, field)


def _target_columns(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise PDBBindDatasetError("target_columns must be a non-empty sequence")
    columns = tuple(_column_name(item, "target_columns") for item in value)
    if not columns or len(set(columns)) != len(columns):
        raise PDBBindDatasetError("target_columns must be non-empty and unique")
    return columns


def _task_type(value: str) -> TaskType:
    if value not in {"regression", "binary_classification"}:
        raise PDBBindDatasetError("task_type must be regression or binary_classification")
    return cast(TaskType, value)


def _invalid_complex_policy(value: str) -> InvalidComplexPolicy:
    if value not in {"error", "skip"}:
        raise PDBBindDatasetError("invalid_complex must be error or skip")
    return cast(InvalidComplexPolicy, value)


def _missing(value: object) -> bool:
    result = pd.isna(value)
    return bool(result) if isinstance(result, bool) else False


__all__ = [
    "PDBBindComplexDataset",
    "PDBBindDatasetError",
    "PDBBindDatasetSummary",
    "POTENTIALNET_DGL_LIFESCI_FEATURE_SCHEMA_V1",
    "fingerprint_files",
]

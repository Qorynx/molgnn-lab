"""Shared deterministic geometry enrichment for coordinate-backed models.

The canonical featurizer remains strictly 2-D.  This module is the single
runtime boundary that supplies coordinates to model-specific graph transforms.
SMILES-only samples receive one deterministic ETKDGv3 conformer; native
dataset geometry is validated and preserved exactly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from torch import Tensor

from .data import MolecularData
from .dataset import MolecularDataset

ETKDG_V3_SEED = 0x5C4E
ETKDG_MAX_OPTIMIZATION_STEPS = 200


class GeometryError(ValueError):
    """Raised when a sample cannot satisfy the shared geometry contract."""


@dataclass(frozen=True)
class GeometryRecord:
    """Reproducibility metadata for one geometry-enriched sample."""

    sample_id: int | str
    smiles: str
    source: Literal["native", "etkdg_v3"]
    optimizer: str
    coordinate_sha256: str


@dataclass(frozen=True)
class GeometryDataset:
    """Geometry-enriched samples with source dataset metadata forwarding."""

    samples: tuple[MolecularData, ...]
    source: MolecularDataset

    def __getitem__(self, index: int) -> MolecularData:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    def __getattr__(self, name: str) -> object:
        return getattr(self.source, name)


@dataclass(frozen=True)
class GeometryDatasetResult:
    """Materialized shared geometry and its compact benchmark provenance."""

    dataset: GeometryDataset
    records: tuple[GeometryRecord, ...]

    def metadata(self) -> dict[str, object]:
        digest = hashlib.sha256()
        for record in self.records:
            digest.update(str(record.sample_id).encode("utf-8"))
            digest.update(record.coordinate_sha256.encode("ascii"))
        native_count = sum(record.source == "native" for record in self.records)
        optimizer_counts: dict[str, int] = {}
        for record in self.records:
            optimizer_counts[record.optimizer] = (
                optimizer_counts.get(record.optimizer, 0) + 1
            )
        return {
            "geometry": {
                "policy": "native_or_etkdg_v3",
                "generator": "ETKDGv3",
                "seed": ETKDG_V3_SEED,
                "optimizer_policy": "MMFF94_or_UFF_or_raw_ETKDG",
                "failure_policy": "error",
                "rdkit_version": rdBase.rdkitVersion,
                "sample_count": len(self.records),
                "native_count": native_count,
                "generated_count": len(self.records) - native_count,
                "optimizer_counts": optimizer_counts,
                "coordinate_manifest_sha256": digest.hexdigest(),
                "cache": "process_memory_and_materialized_dataset",
            }
        }


def prepare_geometry_dataset(dataset: MolecularDataset) -> GeometryDatasetResult:
    """Attach one shared geometry view to every sample in dataset order."""

    samples: list[MolecularData] = []
    records: list[GeometryRecord] = []
    for index in range(len(dataset)):
        sample, record = ensure_sample_geometry(dataset[index])
        samples.append(sample)
        records.append(record)
    return GeometryDatasetResult(
        GeometryDataset(tuple(samples), dataset), tuple(records)
    )


def ensure_sample_geometry(data: MolecularData) -> tuple[MolecularData, GeometryRecord]:
    """Preserve native geometry or deterministically derive it from SMILES."""

    sample = _sample_id(data)
    x = getattr(data, "x", None)
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1:
        raise GeometryError(f"sample {sample} has invalid x for geometry enrichment")
    if x.dtype != torch.float32 or not bool(torch.isfinite(x).all()):
        raise GeometryError(f"sample {sample} x must be finite float32")

    raw_smiles = getattr(data, "smiles", None)
    smiles = raw_smiles if isinstance(raw_smiles, str) else ""
    existing_atomic_number = getattr(data, "atomic_number", None)
    existing_pos = getattr(data, "pos", None)
    if existing_pos is not None:
        _validate_pos(existing_pos, x.shape[0], x.device, sample=sample)
        _validate_atomic_number(
            existing_atomic_number, x.shape[0], x.device, sample=sample
        )
        assert isinstance(existing_atomic_number, Tensor)
        atomic_number = existing_atomic_number
        pos = existing_pos
        source: Literal["native", "etkdg_v3"] = "native"
        optimizer = "native"
    else:
        if not smiles:
            raise GeometryError(f"sample {sample} is missing source SMILES metadata")
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            raise GeometryError(f"sample {sample} has invalid source SMILES")
        if molecule.GetNumAtoms() != x.shape[0]:
            raise GeometryError(
                f"sample {sample} source SMILES atom count does not match x"
            )
        atomic_number = torch.tensor(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
            dtype=torch.long,
            device=x.device,
        )
        if existing_atomic_number is not None:
            _validate_atomic_number(
                existing_atomic_number, x.shape[0], x.device, sample=sample
            )
            assert isinstance(existing_atomic_number, Tensor)
            if not torch.equal(existing_atomic_number, atomic_number):
                raise GeometryError(
                    f"sample {sample} atomic_number does not match source SMILES atom order"
                )
        coordinates, numbers, optimizer = _cached_etkdg_geometry(smiles)
        if numbers != tuple(int(value) for value in atomic_number.tolist()):
            raise GeometryError(
                f"sample {sample} ETKDG atom order does not match canonical features"
            )
        pos = torch.tensor(coordinates, dtype=torch.float32, device=x.device)
        source = "etkdg_v3"

    transformed = data.clone()
    transformed.atomic_number = atomic_number
    transformed.pos = pos
    transformed.geometry_is_proxy = torch.tensor(
        [source == "etkdg_v3"], dtype=torch.bool, device=x.device
    )
    return transformed, GeometryRecord(
        sample_id=sample,
        smiles=smiles,
        source=source,
        optimizer=optimizer,
        coordinate_sha256=_coordinate_hash(pos),
    )


@lru_cache(maxsize=16384)
def _cached_etkdg_geometry(
    source_smiles: str,
) -> tuple[tuple[tuple[float, float, float], ...], tuple[int, ...], str]:
    molecule = Chem.MolFromSmiles(source_smiles)
    if molecule is None or molecule.GetNumAtoms() < 1:
        raise GeometryError(f"cannot build ETKDG geometry for {source_smiles!r}")
    atomic_numbers = tuple(atom.GetAtomicNum() for atom in molecule.GetAtoms())
    if molecule.GetNumAtoms() == 1:
        return (((0.0, 0.0, 0.0),), atomic_numbers, "single_atom")

    molecule_h = Chem.AddHs(molecule)
    parameters = AllChem.ETKDGv3()
    parameters.randomSeed = ETKDG_V3_SEED
    if AllChem.EmbedMolecule(molecule_h, parameters) != 0:
        raise GeometryError(
            f"ETKDGv3 embedding failed for source SMILES {source_smiles!r}"
        )

    optimizer = "raw_etkdg_v3"
    try:
        if AllChem.MMFFHasAllMoleculeParams(molecule_h):
            status = AllChem.MMFFOptimizeMolecule(
                molecule_h, maxIters=ETKDG_MAX_OPTIMIZATION_STEPS
            )
            optimizer = "mmff94" if status == 0 else "mmff94_not_converged"
        elif AllChem.UFFHasAllMoleculeParams(molecule_h):
            status = AllChem.UFFOptimizeMolecule(
                molecule_h, maxIters=ETKDG_MAX_OPTIMIZATION_STEPS
            )
            optimizer = "uff" if status == 0 else "uff_not_converged"
    except (RuntimeError, ValueError):
        optimizer = "raw_etkdg_v3"

    conformer = molecule_h.GetConformer()
    coordinates = tuple(
        (
            float(conformer.GetAtomPosition(index).x),
            float(conformer.GetAtomPosition(index).y),
            float(conformer.GetAtomPosition(index).z),
        )
        for index in range(molecule.GetNumAtoms())
    )
    pos = torch.tensor(coordinates, dtype=torch.float32)
    _validate_pos(
        pos, molecule.GetNumAtoms(), torch.device("cpu"), sample=source_smiles
    )
    return coordinates, atomic_numbers, optimizer


def _validate_pos(
    pos: object, node_count: int, device: torch.device, *, sample: int | str
) -> None:
    if (
        not isinstance(pos, Tensor)
        or pos.shape != (node_count, 3)
        or pos.dtype != torch.float32
        or pos.device != device
        or not bool(torch.isfinite(pos).all())
    ):
        raise GeometryError(
            f"sample {sample} pos must have shape [N, 3] finite float32 on the node device"
        )


def _validate_atomic_number(
    atomic_number: object,
    node_count: int,
    device: torch.device,
    *,
    sample: int | str,
) -> None:
    if (
        not isinstance(atomic_number, Tensor)
        or atomic_number.shape != (node_count,)
        or atomic_number.dtype != torch.long
        or atomic_number.device != device
        or bool((atomic_number <= 0).any())
    ):
        raise GeometryError(
            f"sample {sample} atomic_number must have shape [N] positive long on the node device"
        )


def _coordinate_hash(pos: Tensor) -> str:
    values = pos.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.flatten()[0].item())
    return "<unknown>"


__all__ = [
    "ETKDG_V3_SEED",
    "GeometryDataset",
    "GeometryDatasetResult",
    "GeometryError",
    "GeometryRecord",
    "ensure_sample_geometry",
    "prepare_geometry_dataset",
]

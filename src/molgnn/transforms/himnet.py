"""Derived unified hierarchy and fingerprints required by HimNet."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rdkit import DataStructs, RDConfig
from rdkit.Chem import ChemicalFeatures, MACCSkeys, rdFingerprintGenerator
from torch import Tensor

from ..data import MolecularData
from ..models.himnet_2026.data import HimNetData
from .base import TransformError
from .brics import _BRICSPartition, _resolve_brics_partition

ATOM_PAIR_DIM = 2048
MACCS_DIM = 167
MORGAN_BITS_DIM = 2048
MORGAN_COUNTS_DIM = 2048
PHARMACOPHORE_DIM = 27
HIMNET_FINGERPRINT_DIM = (
    ATOM_PAIR_DIM
    + MACCS_DIM
    + MORGAN_BITS_DIM
    + MORGAN_COUNTS_DIM
    + PHARMACOPHORE_DIM
)

_ATOM_PAIR_GENERATOR = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=ATOM_PAIR_DIM)
_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=MORGAN_BITS_DIM,
)
_FEATURE_FACTORY = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
)
_PHARMACOPHORE_TYPES = tuple(
    definition.split(".", maxsplit=1)[1]
    for definition in _FEATURE_FACTORY.GetFeatureDefs()
)


def add_himnet_inputs(data: MolecularData) -> HimNetData:
    """Clone one canonical graph and attach the complete HimNet input view.

    Atom features and canonical bond features are reused exactly as supplied.
    RDKit is used only here to derive BRICS hierarchy fields and source-style
    fingerprint views before the shared runner batches samples.
    """

    partition = _resolve_brics_partition(data)
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    edge_attr = getattr(data, "edge_attr", None)
    assert isinstance(x, Tensor)
    assert isinstance(edge_index, Tensor)
    assert isinstance(edge_attr, Tensor)
    if x.device != edge_index.device or x.device != edge_attr.device:
        raise TransformError("canonical HimNet inputs must share one device")

    (
        hierarchy_x,
        hierarchy_edge_index,
        hierarchy_edge_attr,
        hierarchy_node_type,
    ) = _hierarchy_tensors(partition, x, edge_index, edge_attr)
    hierarchy_reverse = _reverse_edge_index(hierarchy_edge_index)
    cloned = data.clone()
    transformed = HimNetData(**cloned.to_dict())
    transformed.himnet_x = hierarchy_x
    transformed.himnet_edge_index = hierarchy_edge_index
    transformed.himnet_edge_attr = hierarchy_edge_attr
    transformed.himnet_reverse_edge_index = hierarchy_reverse
    transformed.himnet_node_batch = torch.zeros(
        hierarchy_x.shape[0], dtype=torch.long, device=hierarchy_x.device
    )
    transformed.himnet_node_type = hierarchy_node_type
    transformed.himnet_fp = _fingerprint_tensor(partition, device=hierarchy_x.device)
    return transformed


def _hierarchy_tensors(
    partition: _BRICSPartition,
    x: Tensor,
    edge_index: Tensor,
    edge_attr: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build the corrected paired hierarchy topology for one molecule."""

    atom_count = x.shape[0]
    bond_dim = edge_attr.shape[1]
    atom_to_fragment = partition.atom_to_fragment.to(device=x.device)
    # The upstream fallback uses direct atom--global links when no BRICS cut
    # exists.  Retaining that behavior prevents a whole-molecule pseudo-motif.
    motif_count = int(atom_to_fragment.max().item()) + 1 if partition.boundary_bonds else 0
    if motif_count:
        motif_x = x.new_zeros((motif_count, x.shape[1]))
        motif_x.index_add_(0, atom_to_fragment, x)
    else:
        motif_x = x.new_empty((0, x.shape[1]))
    global_x = x.sum(dim=0, keepdim=True)
    hierarchy_x = torch.cat((x, motif_x, global_x), dim=0)
    hierarchy_node_type = torch.cat(
        (
            torch.zeros(atom_count, dtype=torch.long, device=x.device),
            torch.ones(motif_count, dtype=torch.long, device=x.device),
            torch.full((1,), 2, dtype=torch.long, device=x.device),
        )
    )

    canonical_edges = [tuple(edge) for edge in edge_index.detach().cpu().t().tolist()]
    edge_features = {
        edge: edge_attr[position]
        for position, edge in enumerate(canonical_edges)
    }
    pairs: list[tuple[int, int]] = list(canonical_edges)
    features: list[Tensor] = [edge_attr[position] for position in range(edge_attr.shape[0])]
    zero_relation = edge_attr.new_zeros((bond_dim,))
    global_index = atom_count + motif_count

    if motif_count:
        for atom_a, atom_b in partition.boundary_bonds:
            motif_a = atom_count + int(atom_to_fragment[atom_a].item())
            motif_b = atom_count + int(atom_to_fragment[atom_b].item())
            if motif_a == motif_b:
                raise TransformError("BRICS boundary atoms must belong to different motifs")
            pairs.extend(((motif_a, motif_b), (motif_b, motif_a)))
            features.extend((edge_features[(atom_a, atom_b)], edge_features[(atom_b, atom_a)]))
        for atom_index in range(atom_count):
            motif_index = atom_count + int(atom_to_fragment[atom_index].item())
            pairs.extend(((atom_index, motif_index), (motif_index, atom_index)))
            features.extend((zero_relation, zero_relation))
        for motif_index in range(atom_count, atom_count + motif_count):
            pairs.extend(((motif_index, global_index), (global_index, motif_index)))
            features.extend((zero_relation, zero_relation))
    else:
        for atom_index in range(atom_count):
            pairs.extend(((atom_index, global_index), (global_index, atom_index)))
            features.extend((zero_relation, zero_relation))

    hierarchy_edge_index = torch.tensor(
        pairs,
        dtype=torch.long,
        device=edge_index.device,
    ).t().contiguous()
    hierarchy_edge_attr = torch.stack(features, dim=0).to(device=edge_attr.device)
    return hierarchy_x, hierarchy_edge_index, hierarchy_edge_attr, hierarchy_node_type


def _reverse_edge_index(edge_index: Tensor) -> Tensor:
    """Pair every hierarchy edge with its exact opposite orientation."""

    pairs = [tuple(edge) for edge in edge_index.detach().cpu().t().tolist()]
    reverse = [-1] * len(pairs)
    unmatched: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for position, (source, target) in enumerate(pairs):
        candidates = unmatched[(target, source)]
        if candidates:
            counterpart = candidates.pop()
            reverse[position] = counterpart
            reverse[counterpart] = position
        else:
            unmatched[(source, target)].append(position)
    if any(position < 0 for position in reverse):
        raise TransformError("HimNet hierarchy construction produced an unpaired edge")
    return torch.tensor(reverse, dtype=torch.long, device=edge_index.device)


def _fingerprint_tensor(partition: _BRICSPartition, *, device: torch.device) -> Tensor:
    """Return the five fixed-width source-style fingerprint views as float32."""

    try:
        atom_pairs = _as_numpy(
            _ATOM_PAIR_GENERATOR.GetCountFingerprint(partition.mol), ATOM_PAIR_DIM
        )
        maccs = _as_numpy(MACCSkeys.GenMACCSKeys(partition.mol), MACCS_DIM)
        morgan_bits = _as_numpy(
            _MORGAN_GENERATOR.GetFingerprint(partition.mol), MORGAN_BITS_DIM
        )
        morgan_counts = _as_numpy(
            _MORGAN_GENERATOR.GetCountFingerprint(partition.mol), MORGAN_COUNTS_DIM
        )
        feature_types = {feature.GetType() for feature in _FEATURE_FACTORY.GetFeaturesForMol(partition.mol)}
        pharmacophore = np.asarray(
            [float(feature_type in feature_types) for feature_type in _PHARMACOPHORE_TYPES],
            dtype=np.float32,
        )
        values = np.concatenate(
            (atom_pairs, maccs, morgan_bits, morgan_counts, pharmacophore), axis=0
        )
    except Exception:
        # This is the source implementation's defensive all-zero fallback for
        # an otherwise valid molecule whose optional fingerprint calculation
        # fails in a particular RDKit build.
        values = np.zeros(HIMNET_FINGERPRINT_DIM, dtype=np.float32)
    if values.shape != (HIMNET_FINGERPRINT_DIM,):
        raise TransformError("HimNet fingerprint generator returned an unexpected width")
    return torch.from_numpy(values).to(device=device, dtype=torch.float32).unsqueeze(0)


def _as_numpy(fingerprint: object, width: int) -> np.ndarray:
    values = np.zeros(width, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values


__all__ = ["HIMNET_FINGERPRINT_DIM", "add_himnet_inputs"]

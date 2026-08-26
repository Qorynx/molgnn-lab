"""Canonical molecular graph data contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch_geometric.data import Data

if TYPE_CHECKING:
    from .featurizer import FeatureSchema


class MolecularData(Data):
    """PyG sample produced by the canonical molecular featurizer.

    Validation is provided by :func:`validate_molecular_data`; this class
    intentionally keeps the shared data container thin so model code can
    consume standard PyG fields.
    """

    x: Tensor  # pyright: ignore[reportIncompatibleMethodOverride]
    edge_index: Tensor  # pyright: ignore[reportIncompatibleMethodOverride]
    edge_attr: Tensor  # pyright: ignore[reportIncompatibleMethodOverride]
    chemrl_gem_atom_attr: Tensor
    chemrl_gem_edge_index: Tensor
    chemrl_gem_bond_attr: Tensor
    chemrl_gem_bond_length: Tensor
    chemrl_gem_angle_edge_index: Tensor
    chemrl_gem_bond_angle: Tensor
    chemrl_gem_geometry_is_proxy: Tensor
    neural_fp_x: Tensor
    neural_fp_edge_attr: Tensor
    molebert_atom_attr: Tensor
    molebert_bond_attr: Tensor
    graphmvp_simple_atom_attr: Tensor
    graphmvp_simple_bond_attr: Tensor
    graphmvp_ogb_atom_attr: Tensor
    graphmvp_ogb_bond_attr: Tensor
    y: Tensor  # pyright: ignore[reportIncompatibleMethodOverride]
    y_mask: Tensor
    sample_id: Tensor
    smiles: str
    reverse_edge_index: Tensor
    ampnn_edge_type: Tensor
    mpnn_edge_type: Tensor
    mpnn_3d_edge_index: Tensor
    mpnn_3d_edge_type: Tensor
    atomic_number: Tensor
    dimenet_edge_index: Tensor
    dimenet_triplet_edge_index: Tensor
    schnet_edge_index: Tensor
    schnet_geometry_is_proxy: Tensor
    painn_edge_index: Tensor
    painn_geometry_is_proxy: Tensor
    gemnet_edge_index: Tensor
    gemnet_reverse_edge_index: Tensor
    gemnet_triplet_edge_index: Tensor
    gemnet_interaction_edge_index: Tensor
    gemnet_quadruplet_edge_index: Tensor
    gemnet_quadruplet_interaction_index: Tensor
    gemnet_geometry_is_proxy: Tensor
    visnet_edge_index: Tensor
    visnet_geometry_is_proxy: Tensor
    eqgat_edge_index: Tensor
    eqgat_geometry_is_proxy: Tensor
    equiformer_edge_index: Tensor
    equiformer_geometry_is_proxy: Tensor
    hmgnn_atom_edge_index: Tensor
    hmgnn_body_atom_index: Tensor
    hmgnn_body_edge_index: Tensor
    hmgnn_geometry_is_proxy: Tensor
    gpspp_pair_index: Tensor
    gpspp_spd: Tensor
    weave_pair_index: Tensor
    weave_pair_attr: Tensor
    pos: Tensor
    ligand_mask: Tensor
    potentialnet_bond_edge_index: Tensor
    potentialnet_bond_edge_type: Tensor
    potentialnet_stage2_edge_index: Tensor
    potentialnet_stage2_edge_type: Tensor
    potentialnet_use_spatial: Tensor
    brics_edge_index: Tensor
    brics_edge_attr: Tensor
    atom_to_fragment: Tensor
    frag_index: Tensor
    x_frags: Tensor
    frag_batch: Tensor
    edge_index_bonds_graph: Tensor
    edge_attr_bonds: Tensor
    frag_connection_features: Tensor
    edge_index_fbonds: Tensor
    edge_attr_fbonds: Tensor

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Offset model-specific local indices when PyG batches graphs."""

        if key == "reverse_edge_index":
            return self.edge_index.shape[1]
        if key == "chemrl_gem_angle_edge_index":
            gem_edge_index = getattr(self, "chemrl_gem_edge_index", None)
            if not isinstance(gem_edge_index, Tensor):
                raise ValueError(
                    "MolecularData requires chemrl_gem_edge_index to batch "
                    "chemrl_gem_angle_edge_index"
                )
            return gem_edge_index.shape[1]
        if key == "dimenet_triplet_edge_index":
            dimenet_edge_index = getattr(self, "dimenet_edge_index", None)
            if not isinstance(dimenet_edge_index, Tensor):
                raise ValueError(
                    "MolecularData requires dimenet_edge_index to batch "
                    "dimenet_triplet_edge_index"
                )
            return dimenet_edge_index.shape[1]
        if key in {"gemnet_reverse_edge_index", "gemnet_triplet_edge_index"}:
            gemnet_edge_index = getattr(self, "gemnet_edge_index", None)
            if not isinstance(gemnet_edge_index, Tensor):
                raise ValueError(f"MolecularData requires gemnet_edge_index to batch {key}")
            return gemnet_edge_index.shape[1]
        if key == "gemnet_quadruplet_edge_index":
            gemnet_edge_index = getattr(self, "gemnet_edge_index", None)
            if not isinstance(gemnet_edge_index, Tensor):
                raise ValueError(
                    "MolecularData requires gemnet_edge_index to batch gemnet_quadruplet_edge_index"
                )
            return gemnet_edge_index.shape[1]
        if key == "gemnet_quadruplet_interaction_index":
            interaction_edge_index = getattr(self, "gemnet_interaction_edge_index", None)
            if not isinstance(interaction_edge_index, Tensor):
                raise ValueError(
                    "MolecularData requires gemnet_interaction_edge_index to batch "
                    "gemnet_quadruplet_interaction_index"
                )
            return interaction_edge_index.shape[1]
        if key == "hmgnn_body_edge_index":
            body_atom_index = getattr(self, "hmgnn_body_atom_index", None)
            if not isinstance(body_atom_index, Tensor):
                raise ValueError(
                    "MolecularData requires hmgnn_body_atom_index to batch "
                    "hmgnn_body_edge_index"
                )
            return body_atom_index.shape[1]
        if key == "atom_to_fragment":
            return int(value.max().item()) + 1 if isinstance(value, Tensor) and value.numel() else 0
        if key == "frag_index":
            fragment_features = getattr(self, "x_frags", None)
            return (
                int(fragment_features.shape[0])
                if isinstance(fragment_features, Tensor)
                else 0
            )
        if key == "edge_index_bonds_graph":
            return self.edge_index.shape[1] if isinstance(self.edge_index, Tensor) else 0
        if key == "edge_index_fbonds":
            # Both axes are connection indices (graph where connections are
            # nodes and edges connect sharing connections).
            connection_features = getattr(self, "frag_connection_features", None)
            if isinstance(connection_features, Tensor):
                return int(connection_features.shape[0])
            return 0
        if key in {"kpgt_attention_edge_index", "kpgt_path_index"}:
            indicator = getattr(self, "kpgt_node_indicator", None)
            if not isinstance(indicator, Tensor):
                raise ValueError(
                    "MolecularData requires kpgt_node_indicator to batch "
                    f"{key} over KPGT line-nodes"
                )
            return int(indicator.shape[0])
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        """Concatenate one-dimensional model metadata along axis zero."""

        if key in {
            "reverse_edge_index",
            "atom_to_fragment",
            "gemnet_reverse_edge_index",
            "gemnet_quadruplet_interaction_index",
        }:
            return 0
        if key == "kpgt_path_index":
            # [A, path_length] rows are per-edge path records.
            return 0
        if key in {
            "dimenet_triplet_edge_index",
            "gemnet_triplet_edge_index",
            "gemnet_quadruplet_edge_index",
            "chemrl_gem_angle_edge_index",
        }:
            return 1
        return super().__cat_dim__(key, value, *args, **kwargs)


def validate_molecular_data(
    data: MolecularData,
    schema: FeatureSchema,
    num_targets: int,
) -> None:
    """Validate one canonical graph sample at the data/model boundary.

    The validator deliberately checks only the shared molecular contract. It
    does not know about model-specific fragment fields or task-specific label
    semantics beyond the target shape and observed-value mask.
    """

    if isinstance(num_targets, bool) or not isinstance(num_targets, int) or num_targets < 1:
        raise ValueError("num_targets must be a positive integer")

    fields = ("x", "edge_index", "edge_attr", "y", "y_mask", "sample_id")
    values: dict[str, Tensor] = {}
    for field_name in fields:
        value = getattr(data, field_name, None)
        if value is None:
            raise ValueError(f"MolecularData is missing required field '{field_name}'")
        if not isinstance(value, Tensor):
            raise TypeError(f"MolecularData field '{field_name}' must be a torch.Tensor")
        values[field_name] = value

    x = values["x"]
    edge_index = values["edge_index"]
    edge_attr = values["edge_attr"]
    y = values["y"]
    y_mask = values["y_mask"]
    sample_id = values["sample_id"]

    if x.ndim != 2 or x.shape[1] != schema.atom_dim:
        raise ValueError(f"data.x must have shape [N, {schema.atom_dim}]")
    if x.dtype != torch.float32:
        raise TypeError("data.x must have dtype torch.float32")
    if not torch.isfinite(x).all():
        raise ValueError("data.x must contain only finite values")

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("data.edge_index must have shape [2, E]")
    if edge_index.dtype != torch.int64:
        raise TypeError("data.edge_index must have dtype torch.int64")

    if edge_attr.ndim != 2 or edge_attr.shape[1] != schema.bond_dim:
        raise ValueError(f"data.edge_attr must have shape [E, {schema.bond_dim}]")
    if edge_attr.dtype != torch.float32:
        raise TypeError("data.edge_attr must have dtype torch.float32")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError("data.edge_attr row count must equal data.edge_index edge count")
    if not torch.isfinite(edge_attr).all():
        raise ValueError("data.edge_attr must contain only finite values")

    if edge_index.shape[1] > 0:
        num_nodes = x.shape[0]
        if num_nodes < 1:
            raise ValueError("data.edge_index cannot contain edges when data.x has no nodes")
        if edge_index.min() < 0 or edge_index.max() >= num_nodes:
            raise ValueError("data.edge_index contains an index outside [0, num_nodes)")

    reverse_edge_index = getattr(data, "reverse_edge_index", None)
    if reverse_edge_index is not None:
        if not isinstance(reverse_edge_index, Tensor):
            raise TypeError("data.reverse_edge_index must be a torch.Tensor")
        edge_count = edge_index.shape[1]
        if reverse_edge_index.shape != (edge_count,):
            raise ValueError("data.reverse_edge_index must have shape [E]")
        if reverse_edge_index.dtype != torch.int64:
            raise TypeError("data.reverse_edge_index must have dtype torch.int64")
        if edge_count and (reverse_edge_index.min() < 0 or reverse_edge_index.max() >= edge_count):
            raise ValueError("data.reverse_edge_index contains an index outside [0, E)")
        if not torch.equal(
            reverse_edge_index[reverse_edge_index],
            torch.arange(edge_count, device=reverse_edge_index.device),
        ):
            raise ValueError("data.reverse_edge_index must be an involution")
        if edge_count and not torch.equal(edge_index[:, reverse_edge_index], edge_index.flip(0)):
            raise ValueError("data.reverse_edge_index must map each edge to its reverse")

    expected_target_shape = (1, num_targets)
    if y.shape != expected_target_shape:
        raise ValueError(f"data.y must have shape [1, {num_targets}]")
    if y.dtype != torch.float32:
        raise TypeError("data.y must have dtype torch.float32")

    if y_mask.shape != expected_target_shape:
        raise ValueError(f"data.y_mask must have shape [1, {num_targets}]")
    if y_mask.dtype != torch.bool:
        raise TypeError("data.y_mask must have dtype torch.bool")
    if y_mask.any() and not torch.isfinite(y[y_mask]).all():
        raise ValueError("observed data.y values must be finite")

    if sample_id.shape != (1,):
        raise ValueError("data.sample_id must have shape [1]")
    if sample_id.dtype != torch.int64:
        raise TypeError("data.sample_id must have dtype torch.int64")


__all__ = ["MolecularData", "validate_molecular_data"]

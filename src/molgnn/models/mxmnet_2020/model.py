"""Modern PyG implementation of the MXMNet molecular architecture."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .basis import MXMNetRadialBasis, MXMNetSphericalBasis
from .constants import (
    MXMNET_ENVELOPE_EXPONENT,
    MXMNET_GLOBAL_CUTOFF,
    MXMNET_HIDDEN_DIM,
    MXMNET_LOCAL_CUTOFF,
    MXMNET_MAX_ATOMIC_NUMBER,
    MXMNET_NUM_LAYERS,
    MXMNET_NUM_RADIAL,
    MXMNET_NUM_SPHERICAL,
    MXMNET_SPHERICAL_RADIAL,
)
from .layers import DenseMLP, GlobalMessagePassing, LocalMessagePassing


class MXMNet2020(BaseMolecularModel):
    """Invariant molecular mechanics-driven multiplex graph network."""

    required_batch_fields = (
        "atomic_number",
        "pos",
        "mxmnet_local_edge_index",
        "mxmnet_global_edge_index",
        "mxmnet_two_hop_edge_index",
        "mxmnet_one_hop_edge_index",
        "batch",
    )

    def __init__(
        self,
        num_targets: int,
        hidden_dim: int = MXMNET_HIDDEN_DIM,
        num_layers: int = MXMNET_NUM_LAYERS,
        num_radial: int = MXMNET_NUM_RADIAL,
        num_spherical: int = MXMNET_NUM_SPHERICAL,
        spherical_num_radial: int = MXMNET_SPHERICAL_RADIAL,
        envelope_exponent: int = MXMNET_ENVELOPE_EXPONENT,
        max_atomic_number: int = MXMNET_MAX_ATOMIC_NUMBER,
    ) -> None:
        super().__init__()
        for value, name in (
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (num_layers, "num_layers"),
            (num_radial, "num_radial"),
            (num_spherical, "num_spherical"),
            (spherical_num_radial, "spherical_num_radial"),
            (envelope_exponent, "envelope_exponent"),
            (max_atomic_number, "max_atomic_number"),
        ):
            _positive_int(value, name)

        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.spherical_num_radial = spherical_num_radial
        self.envelope_exponent = envelope_exponent
        self.max_atomic_number = max_atomic_number
        self.local_cutoff = MXMNET_LOCAL_CUTOFF
        self.global_cutoff = MXMNET_GLOBAL_CUTOFF

        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.local_radial_basis = MXMNetRadialBasis(
            num_radial, self.local_cutoff, envelope_exponent
        )
        self.global_radial_basis = MXMNetRadialBasis(
            num_radial, self.global_cutoff, envelope_exponent
        )
        self.spherical_basis = MXMNetSphericalBasis(
            num_spherical,
            spherical_num_radial,
            self.local_cutoff,
            envelope_exponent,
        )
        self.local_radial_projection = DenseMLP((num_radial, hidden_dim))
        self.global_radial_projection = DenseMLP((num_radial, hidden_dim))
        spherical_width = num_spherical * spherical_num_radial
        self.two_hop_spherical_projection = DenseMLP((spherical_width, hidden_dim))
        self.one_hop_spherical_projection = DenseMLP((spherical_width, hidden_dim))
        self.global_layers = nn.ModuleList(
            GlobalMessagePassing(hidden_dim) for _ in range(num_layers)
        )
        self.local_layers = nn.ModuleList(
            LocalMessagePassing(hidden_dim, num_targets) for _ in range(num_layers)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.atom_embedding.weight, -math.sqrt(3.0), math.sqrt(3.0))
        self.local_radial_basis.reset_parameters()
        self.global_radial_basis.reset_parameters()

    def forward(self, batch: Batch) -> Tensor:
        """Return raw graph predictions with shape ``[B, num_targets]``."""

        (
            atomic_number,
            pos,
            local_edge_index,
            global_edge_index,
            two_hop_edge_index,
            one_hop_edge_index,
            graph_batch,
            num_graphs,
        ) = self._batch_tensors(batch)

        local_source, local_target = local_edge_index
        global_source, global_target = global_edge_index
        local_distances = torch.linalg.vector_norm(
            pos[local_target] - pos[local_source], dim=-1
        )
        global_distances = torch.linalg.vector_norm(
            pos[global_target] - pos[global_source], dim=-1
        )
        self._validate_distances(local_distances, global_distances)

        local_radial = self.local_radial_projection(
            self.local_radial_basis(local_distances)
        )
        global_radial = self.global_radial_projection(
            self.global_radial_basis(global_distances)
        )

        incoming_edge, two_hop_base = two_hop_edge_index
        two_hop_cosines = self._two_hop_cosines(
            pos,
            local_edge_index,
            local_distances,
            incoming_edge,
            two_hop_base,
        )
        two_hop_spherical = self.two_hop_spherical_projection(
            self.spherical_basis.forward_from_cosine(
                local_distances[incoming_edge], two_hop_cosines
            )
        )

        sibling_edge, one_hop_base = one_hop_edge_index
        one_hop_cosines = self._one_hop_cosines(
            pos,
            local_edge_index,
            local_distances,
            sibling_edge,
            one_hop_base,
        )
        one_hop_spherical = self.one_hop_spherical_projection(
            self.spherical_basis.forward_from_cosine(
                local_distances[sibling_edge], one_hop_cosines
            )
        )

        atom = self.atom_embedding(atomic_number)
        atom_outputs = atom.new_zeros((atom.shape[0], self.num_targets))
        for global_layer, local_layer in zip(
            self.global_layers, self.local_layers, strict=True
        ):
            atom = global_layer(atom, global_radial, global_edge_index)
            atom, contribution = local_layer(
                atom,
                local_radial,
                two_hop_spherical,
                one_hop_spherical,
                two_hop_edge_index,
                one_hop_edge_index,
                local_edge_index,
            )
            atom_outputs = atom_outputs + contribution

        return scatter(
            atom_outputs,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        values = tuple(
            getattr(batch, name, None) for name in self.required_batch_fields
        )
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(
                f"batch must provide {', '.join(self.required_batch_fields)} tensors"
            )
        (
            atomic_number,
            pos,
            local_edge_index,
            global_edge_index,
            two_hop_edge_index,
            one_hop_edge_index,
            graph_batch,
        ) = values
        assert isinstance(atomic_number, Tensor)
        assert isinstance(pos, Tensor)
        assert isinstance(local_edge_index, Tensor)
        assert isinstance(global_edge_index, Tensor)
        assert isinstance(two_hop_edge_index, Tensor)
        assert isinstance(one_hop_edge_index, Tensor)
        assert isinstance(graph_batch, Tensor)

        node_count = atomic_number.shape[0] if atomic_number.ndim == 1 else -1
        if (
            atomic_number.ndim != 1
            or node_count < 1
            or atomic_number.dtype != torch.long
            or atomic_number.min() < 1
            or atomic_number.max() > self.max_atomic_number
        ):
            raise ValueError(
                "batch.atomic_number must be positive long [N] within the configured vocabulary"
            )
        if (
            pos.shape != (node_count, 3)
            or pos.dtype != torch.float32
            or not bool(torch.isfinite(pos).all())
        ):
            raise ValueError("batch.pos must have shape [N, 3] finite torch.float32")
        if any(value.device != pos.device for value in values):
            raise ValueError("all MXMNet batch tensors must share the position device")

        num_graphs = validate_batched_molecular_graph(
            local_edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="mxmnet_local_edge_index",
            forbid_self_loops=True,
        )
        global_graphs = validate_batched_molecular_graph(
            global_edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="mxmnet_global_edge_index",
            forbid_self_loops=True,
        )
        if global_graphs != num_graphs:
            raise ValueError("MXMNet local/global graph counts do not match")
        self._validate_reciprocal_edges(local_edge_index, node_count, "local")
        self._validate_reciprocal_edges(global_edge_index, node_count, "global")
        self._validate_two_hop(local_edge_index, two_hop_edge_index)
        self._validate_one_hop(local_edge_index, one_hop_edge_index)
        return (
            atomic_number,
            pos,
            local_edge_index,
            global_edge_index,
            two_hop_edge_index,
            one_hop_edge_index,
            graph_batch,
            num_graphs,
        )

    @staticmethod
    def _validate_reciprocal_edges(
        edge_index: Tensor, node_count: int, label: str
    ) -> None:
        if edge_index.numel() == 0:
            return
        source, target = edge_index
        encoded = source * node_count + target
        if torch.unique(encoded).numel() != encoded.numel():
            raise ValueError(f"MXMNet {label} graph must not contain duplicate edges")
        if not bool(torch.isin(target * node_count + source, encoded).all()):
            raise ValueError(f"MXMNet {label} graph must be bidirected")

    @staticmethod
    def _validate_two_hop(edge_index: Tensor, pair_index: Tensor) -> None:
        _validate_edge_pair_table(pair_index, edge_index.shape[1], "two-hop")
        if pair_index.numel() == 0:
            return
        incoming, base = pair_index
        source, target = edge_index
        if not torch.equal(target[incoming], source[base]):
            raise ValueError("MXMNet two-hop table must encode k -> j -> i")
        if bool((source[incoming] == target[base]).any()):
            raise ValueError("MXMNet two-hop table must exclude immediate reversal")

    @staticmethod
    def _validate_one_hop(edge_index: Tensor, pair_index: Tensor) -> None:
        _validate_edge_pair_table(pair_index, edge_index.shape[1], "one-hop")
        if pair_index.numel() == 0:
            return
        sibling, base = pair_index
        source, target = edge_index
        if not torch.equal(target[sibling], target[base]):
            raise ValueError("MXMNet one-hop edges must share their target atom")
        if bool((source[sibling] == source[base]).any()):
            raise ValueError("MXMNet one-hop table must exclude immediate reversal")

    def _validate_distances(
        self, local_distances: Tensor, global_distances: Tensor
    ) -> None:
        if bool((local_distances <= 0).any()) or bool((global_distances <= 0).any()):
            raise ValueError("MXMNet graph edges must join distinct coordinates")
        if bool((global_distances > self.global_cutoff + 1.0e-6).any()):
            raise ValueError("MXMNet global graph contains an edge beyond its cutoff")

    @staticmethod
    def _two_hop_cosines(
        pos: Tensor,
        edge_index: Tensor,
        distances: Tensor,
        incoming: Tensor,
        base: Tensor,
    ) -> Tensor:
        source, target = edge_index
        first = pos[source[base]] - pos[target[base]]
        second = pos[source[incoming]] - pos[target[incoming]]
        cosine = (first * second).sum(dim=-1) / (distances[base] * distances[incoming])
        return cosine.clamp(min=-1.0, max=1.0)

    @staticmethod
    def _one_hop_cosines(
        pos: Tensor,
        edge_index: Tensor,
        distances: Tensor,
        sibling: Tensor,
        base: Tensor,
    ) -> Tensor:
        source, target = edge_index
        first = pos[target[base]] - pos[source[base]]
        second = pos[source[sibling]] - pos[target[sibling]]
        cosine = (first * second).sum(dim=-1) / (distances[base] * distances[sibling])
        return cosine.clamp(min=-1.0, max=1.0)


def _validate_edge_pair_table(pair_index: Tensor, edge_count: int, label: str) -> None:
    if (
        not isinstance(pair_index, Tensor)
        or pair_index.ndim != 2
        or pair_index.shape[0] != 2
        or pair_index.dtype != torch.long
    ):
        raise ValueError(f"MXMNet {label} table must have shape [2, Q] and be long")
    if pair_index.numel() and (pair_index.min() < 0 or pair_index.max() >= edge_count):
        raise ValueError(f"MXMNet {label} table contains an invalid local edge ID")
    if pair_index.shape[1]:
        encoded = pair_index[0] * edge_count + pair_index[1]
        if torch.unique(encoded).numel() != encoded.numel():
            raise ValueError(f"MXMNet {label} table must not contain duplicates")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


__all__ = ["MXMNet2020"]

"""Paper-oriented DimeNet-2020 over an explicit coordinate-backed contract."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .basis import RadialBesselBasis, SphericalBesselBasis
from .constants import DIMENET_CUTOFF, DIMENET_ENVELOPE_P, DIMENET_MAX_ATOMIC_NUMBER
from .layers import EmbeddingBlock, InteractionBlock, OutputBlock


class DimeNet2020(BaseMolecularModel):
    """Directional radius-graph message passing with paper-literal triplets.

    The model keeps hidden states on directed radius edges ``j -> i``.  Its
    architecture-specific transform derives that topology and its edge-ID
    triplets before PyG batching; distances, angles and basis values are
    recomputed here from ``pos`` so outputs retain coordinate gradients.

    The bundled profile intentionally fixes the cutoff at five Angstrom and
    uses the sixth-order envelope shared with ``dimenet_inputs``.  A structural
    source or a deliberately prepared library sample must supply the required
    atomic numbers and coordinates; the canonical 2D feature matrix is not
    interpreted as either input.
    """

    required_batch_fields = (
        "atomic_number",
        "pos",
        "dimenet_edge_index",
        "dimenet_triplet_edge_index",
        "batch",
    )

    def __init__(
        self,
        hidden_dim: int = 128,
        num_blocks: int = 6,
        num_bilinear: int = 8,
        num_spherical: int = 7,
        num_radial: int = 6,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        num_dense_output: int = 3,
        max_atomic_number: int = DIMENET_MAX_ATOMIC_NUMBER,
        output_initializer: str = "zeros",
        num_targets: int = 1,
    ) -> None:
        super().__init__()
        for value, name in (
            (hidden_dim, "hidden_dim"),
            (num_blocks, "num_blocks"),
            (num_bilinear, "num_bilinear"),
            (num_spherical, "num_spherical"),
            (num_radial, "num_radial"),
            (max_atomic_number, "max_atomic_number"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        _nonnegative_int(num_before_skip, "num_before_skip")
        _nonnegative_int(num_after_skip, "num_after_skip")
        _nonnegative_int(num_dense_output, "num_dense_output")
        if output_initializer not in {"zeros", "glorot_orthogonal"}:
            raise ValueError(
                "output_initializer must be 'zeros' or 'glorot_orthogonal'"
            )

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.num_bilinear = num_bilinear
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.envelope_p = DIMENET_ENVELOPE_P
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip
        self.num_dense_output = num_dense_output
        self.max_atomic_number = max_atomic_number
        self.output_initializer = output_initializer
        self.num_targets = num_targets
        self.cutoff = DIMENET_CUTOFF

        self.radial_basis = RadialBesselBasis(
            num_radial,
            cutoff=self.cutoff,
            envelope_p=self.envelope_p,
        )
        self.spherical_basis = SphericalBesselBasis(
            num_spherical,
            num_radial,
            cutoff=self.cutoff,
            envelope_p=self.envelope_p,
        )
        self.embedding_block = EmbeddingBlock(
            num_radial,
            hidden_dim,
            max_atomic_number,
        )
        self.interaction_blocks = nn.ModuleList(
            InteractionBlock(
                hidden_dim,
                num_bilinear,
                num_spherical,
                num_radial,
                num_before_skip=num_before_skip,
                num_after_skip=num_after_skip,
            )
            for _ in range(num_blocks)
        )
        self.output_blocks = nn.ModuleList(
            OutputBlock(
                num_radial,
                hidden_dim,
                num_targets,
                num_dense_output=num_dense_output,
                output_initializer=output_initializer,
            )
            for _ in range(num_blocks + 1)
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw target values/logits with shape ``[num_graphs, T]``."""

        (
            atomic_number,
            pos,
            edge_index,
            triplet_edge_index,
            graph_batch,
            num_graphs,
        ) = self._batch_tensors(batch)
        source, target = edge_index
        distances = torch.linalg.vector_norm(pos[target] - pos[source], dim=-1)
        self._validate_radius_distances(distances)
        rbf = self.radial_basis(distances)

        idx_kj, idx_ji = triplet_edge_index
        if idx_kj.numel():
            angle_cosines = self._triplet_angle_cosines(
                pos,
                edge_index,
                distances,
                idx_kj,
                idx_ji,
            )
            sbf = self.spherical_basis.forward_from_cosine(
                distances[idx_kj], angle_cosines
            )
        else:
            sbf = rbf.new_empty((0, self.num_spherical * self.num_radial))

        messages = self.embedding_block(atomic_number, rbf, source, target)
        atom_outputs = self.output_blocks[0](
            messages,
            rbf,
            target,
            atomic_number.shape[0],
        )
        for interaction_block, output_block in zip(
            self.interaction_blocks, self.output_blocks[1:], strict=True
        ):
            messages = interaction_block(messages, rbf, sbf, idx_kj, idx_ji)
            atom_outputs = atom_outputs + output_block(
                messages,
                rbf,
                target,
                atomic_number.shape[0],
            )
        return scatter(
            atom_outputs,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce="sum",
        )

    def _batch_tensors(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Fetch and validate DimeNet's prepared sparse radius-graph view."""

        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        atomic_number, pos, edge_index, triplet_edge_index, graph_batch = values
        assert isinstance(atomic_number, Tensor)
        assert isinstance(pos, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(triplet_edge_index, Tensor)
        assert isinstance(graph_batch, Tensor)

        node_count = atomic_number.shape[0] if atomic_number.ndim == 1 else -1
        if (
            atomic_number.ndim != 1
            or node_count < 1
            or atomic_number.dtype != torch.long
        ):
            raise ValueError(
                "batch.atomic_number must have shape [N] and dtype torch.long with N >= 1"
            )
        if atomic_number.min() < 1 or atomic_number.max() > self.max_atomic_number:
            raise ValueError(
                "batch.atomic_number contains a value outside the configured vocabulary"
            )
        if (
            pos.shape != (node_count, 3)
            or pos.dtype != torch.float32
            or not torch.isfinite(pos).all()
        ):
            raise ValueError("batch.pos must have shape [N, 3] finite torch.float32")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.dimenet_edge_index must have shape [2, E] and dtype torch.long"
            )
        if (
            triplet_edge_index.ndim != 2
            or triplet_edge_index.shape[0] != 2
            or triplet_edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.dimenet_triplet_edge_index must have shape [2, Q] and dtype torch.long"
            )
        if graph_batch.shape != (node_count,) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if graph_batch.numel() == 0 or graph_batch.min() < 0:
            raise ValueError("batch.batch must contain non-negative graph indices")
        if any(value.device != pos.device for value in values):
            raise ValueError("all DimeNet batch tensors must share the position device")

        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="dimenet_edge_index",
            forbid_self_loops=True,
        )
        self._validate_bidirected_radius_edges(edge_index, node_count)
        self._validate_triplet_topology(edge_index, triplet_edge_index)
        return (
            atomic_number,
            pos,
            edge_index,
            triplet_edge_index,
            graph_batch,
            num_graphs,
        )

    @staticmethod
    def _validate_bidirected_radius_edges(edge_index: Tensor, node_count: int) -> None:
        """Reject duplicate or one-way radius edges in a prepared model batch."""

        edge_count = edge_index.shape[1]
        if edge_count == 0:
            return
        source, target = edge_index
        encoded = source * node_count + target
        if torch.unique(encoded).numel() != edge_count:
            raise ValueError(
                "batch.dimenet_edge_index must not contain duplicate edges"
            )
        reverse_encoded = target * node_count + source
        if not bool(torch.isin(reverse_encoded, encoded).all()):
            raise ValueError(
                "batch.dimenet_edge_index must provide a reciprocal edge for every radius edge"
            )

    @staticmethod
    def _validate_triplet_topology(
        edge_index: Tensor, triplet_edge_index: Tensor
    ) -> None:
        """Ensure edge-ID triplets are legal non-backtracking directed paths."""

        edge_count = edge_index.shape[1]
        incoming, outgoing = triplet_edge_index
        triplet_count = incoming.shape[0]
        if triplet_count and (
            incoming.min() < 0
            or outgoing.min() < 0
            or incoming.max() >= edge_count
            or outgoing.max() >= edge_count
        ):
            raise ValueError(
                "batch.dimenet_triplet_edge_index contains an invalid directed-edge index"
            )
        if triplet_count == 0:
            return
        source, target = edge_index
        if not torch.equal(target[incoming], source[outgoing]):
            raise ValueError(
                "batch.dimenet_triplet_edge_index must encode k -> j -> i paths"
            )
        if bool((source[incoming] == target[outgoing]).any()):
            raise ValueError(
                "batch.dimenet_triplet_edge_index must exclude immediate backtracking"
            )
        encoded = incoming * edge_count + outgoing
        if torch.unique(encoded).numel() != triplet_count:
            raise ValueError(
                "batch.dimenet_triplet_edge_index must not contain duplicate triplets"
            )

    def _validate_radius_distances(self, distances: Tensor) -> None:
        """Keep prepared topology compatible with the fixed bundled cutoff."""

        if distances.numel() and (
            torch.any(distances <= 0) or torch.any(distances > self.cutoff)
        ):
            raise ValueError(
                "batch.dimenet_edge_index must contain nonzero radius edges within the fixed cutoff"
            )

    @staticmethod
    def _triplet_angle_cosines(
        pos: Tensor,
        edge_index: Tensor,
        edge_distances: Tensor,
        idx_kj: Tensor,
        idx_ji: Tensor,
    ) -> Tensor:
        """Return cosines of paper-literal interior angles at the middle atom.

        The spherical basis uses only real ``m=0`` harmonics, which are
        functions of this cosine.  Evaluating it directly is mathematically
        equivalent to forming ``atan2(||cross||, dot)`` and taking ``cos``,
        but avoids the undefined higher derivatives of a vector norm at a
        valid collinear triplet.
        """

        source, target = edge_index
        k = source[idx_kj]
        j = target[idx_kj]
        incoming_direction = pos[k] - pos[j]
        outgoing_direction = pos[target[idx_ji]] - pos[j]
        dot = (incoming_direction * outgoing_direction).sum(dim=-1)
        return dot / (edge_distances[idx_kj] * edge_distances[idx_ji])


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = ["DimeNet2020"]

"""Project-facing GemNet-T and GemNet-Q graph-level models."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.autograd import grad
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .basis import CircularBasis, RadialBasis, TensorBasis
from .constants import (
    GEMNET_CUTOFF,
    GEMNET_INTERACTION_CUTOFF,
    GEMNET_MAX_ATOMIC_NUMBER,
)
from .geometry import angle_cosine, quadruplet_geometry
from .layers import (
    BasisDownProjection,
    Dense,
    EdgeEmbedding,
    InteractionBlock,
    OutputBlock,
)


class GemNet(BaseMolecularModel):
    """Shared directed-edge core for the T and Q variants."""

    variant = ""

    def __init__(
        self,
        num_targets: int,
        *,
        num_spherical: int = 7,
        num_radial: int = 6,
        num_blocks: int = 4,
        atom_embedding_dim: int = 128,
        edge_dim: int = 128,
        triplet_dim: int = 64,
        quadruplet_dim: int = 32,
        rbf_dim: int = 16,
        cbf_dim: int = 16,
        sbf_dim: int = 32,
        bilinear_triplet_dim: int = 64,
        bilinear_quadruplet_dim: int = 32,
        num_before_skip: int = 1,
        num_after_skip: int = 1,
        num_concat: int = 1,
        num_atom: int = 2,
        readout: str = "sum",
        output_initializer: str = "he_orthogonal",
        max_atomic_number: int = GEMNET_MAX_ATOMIC_NUMBER,
        epsilon: float = 1.0e-8,
        scaling_factors: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__()
        if self.variant not in {"t", "q"}:
            raise ValueError("instantiate the fixed GemNetT or GemNetQ variant")
        for value, name in (
            (num_targets, "num_targets"),
            (num_spherical, "num_spherical"),
            (num_radial, "num_radial"),
            (num_blocks, "num_blocks"),
            (atom_embedding_dim, "atom_embedding_dim"),
            (edge_dim, "edge_dim"),
            (triplet_dim, "triplet_dim"),
            (quadruplet_dim, "quadruplet_dim"),
            (rbf_dim, "rbf_dim"),
            (cbf_dim, "cbf_dim"),
            (sbf_dim, "sbf_dim"),
            (bilinear_triplet_dim, "bilinear_triplet_dim"),
            (bilinear_quadruplet_dim, "bilinear_quadruplet_dim"),
            (max_atomic_number, "max_atomic_number"),
        ):
            _positive_int(value, name)
        for value, name in (
            (num_before_skip, "num_before_skip"),
            (num_after_skip, "num_after_skip"),
            (num_concat, "num_concat"),
            (num_atom, "num_atom"),
        ):
            _nonnegative_int(value, name)
        if readout not in {"sum", "mean"}:
            raise ValueError("readout must be either 'sum' or 'mean'")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if scaling_factors is not None:
            for name, value in scaling_factors.items():
                if not isinstance(name, str) or not name:
                    raise ValueError("scaling-factor names must be non-empty strings")
                if not isinstance(value, (int, float)) or value <= 0:
                    raise ValueError("scaling factors must be positive numbers")

        self.use_quadruplets = self.variant == "q"
        self.num_targets = num_targets
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.num_blocks = num_blocks
        self.max_atomic_number = max_atomic_number
        self.readout = readout
        self.epsilon = float(epsilon)
        self.cutoff = GEMNET_CUTOFF
        self.interaction_cutoff = GEMNET_INTERACTION_CUTOFF

        self.radial_basis = RadialBasis(num_radial, self.cutoff)
        self.circular_basis = CircularBasis(num_spherical, num_radial, self.cutoff)
        self.tensor_basis = (
            TensorBasis(num_spherical, num_radial, self.cutoff)
            if self.use_quadruplets
            else None
        )
        self.interaction_circular_basis = (
            CircularBasis(num_spherical, num_radial, self.interaction_cutoff)
            if self.use_quadruplets
            else None
        )

        self.rbf_triplet_down = Dense(num_radial, rbf_dim)
        self.rbf_atom_down = Dense(num_radial, rbf_dim)
        self.rbf_output_down = Dense(num_radial, rbf_dim)
        self.triplet_basis_down = BasisDownProjection(
            num_spherical, num_radial, cbf_dim
        )
        self.rbf_quadruplet_down = (
            Dense(num_radial, rbf_dim) if self.use_quadruplets else None
        )
        self.quadruplet_cbf_down = (
            Dense(num_spherical * num_radial, cbf_dim)
            if self.use_quadruplets
            else None
        )
        self.quadruplet_basis_down = (
            BasisDownProjection(num_spherical**2, num_radial, sbf_dim)
            if self.use_quadruplets
            else None
        )

        self.atom_embedding = nn.Embedding(max_atomic_number + 1, atom_embedding_dim)
        nn.init.uniform_(self.atom_embedding.weight, -3**0.5, 3**0.5)
        self.edge_embedding = EdgeEmbedding(atom_embedding_dim, num_radial, edge_dim)
        scales = dict(scaling_factors or {})
        self.interaction_blocks = nn.ModuleList(
            InteractionBlock(
                atom_embedding_dim,
                edge_dim,
                triplet_dim,
                quadruplet_dim,
                rbf_dim,
                cbf_dim,
                sbf_dim,
                bilinear_triplet_dim,
                bilinear_quadruplet_dim,
                num_before_skip,
                num_after_skip,
                num_concat,
                num_atom,
                use_quadruplets=self.use_quadruplets,
                scales=scales,
                prefix=f"block_{index}",
            )
            for index in range(num_blocks)
        )
        self.output_blocks = nn.ModuleList(
            OutputBlock(
                atom_embedding_dim,
                edge_dim,
                rbf_dim,
                num_targets,
                num_atom,
                scale=float(scales.get(f"output_{index}.sum", 1.0)),
                output_initializer=output_initializer,
            )
            for index in range(num_blocks + 1)
        )

    def forward(self, batch: Batch) -> Tensor:
        tensors = self._validate_batch(batch)
        atomic_number = tensors["atomic_number"]
        pos = tensors["pos"]
        edge_index = tensors["edge_index"]
        reverse_edge = tensors["reverse_edge"]
        triplet_index = tensors["triplet_index"]
        graph_batch = tensors["graph_batch"]
        num_graphs = int(tensors["num_graphs"])
        source, target = edge_index

        displacement = pos[source] - pos[target]
        distances = torch.linalg.vector_norm(displacement, dim=-1)
        rbf = self.radial_basis(distances)
        triplet_reduce, triplet_expand = triplet_index
        triplet_cosine = angle_cosine(
            displacement[triplet_reduce],
            displacement[triplet_expand],
            epsilon=self.epsilon,
        )
        triplet_radial = self.triplet_basis_down(
            self.circular_basis.radial_components(distances)
        )
        triplet_angular = self.circular_basis.angular_components(triplet_cosine)

        rbf_triplet = self.rbf_triplet_down(rbf)
        rbf_atom = self.rbf_atom_down(rbf)
        rbf_output = self.rbf_output_down(rbf)
        rbf_quadruplet: Tensor | None = None
        quadruplet_cbf: Tensor | None = None
        quadruplet_radial: Tensor | None = None
        quadruplet_angular: Tensor | None = None
        quadruplet_reduce: Tensor | None = None
        quadruplet_expand: Tensor | None = None

        if self.use_quadruplets:
            interaction_edge_index = tensors["interaction_edge_index"]
            quadruplet_edge_index = tensors["quadruplet_edge_index"]
            quadruplet_interaction_index = tensors["quadruplet_interaction_index"]
            interaction_source, interaction_target = interaction_edge_index
            interaction_distances = torch.linalg.vector_norm(
                pos[interaction_source] - pos[interaction_target], dim=-1
            )
            cosine_cab, cosine_abd, cosine_plane, sine_plane = quadruplet_geometry(
                pos,
                edge_index,
                interaction_edge_index,
                quadruplet_edge_index,
                quadruplet_interaction_index,
                epsilon=self.epsilon,
            )
            quadruplet_reduce, quadruplet_expand = quadruplet_edge_index
            assert self.interaction_circular_basis is not None
            assert self.tensor_basis is not None
            assert self.quadruplet_cbf_down is not None
            assert self.quadruplet_basis_down is not None
            assert self.rbf_quadruplet_down is not None
            quadruplet_cbf = self.quadruplet_cbf_down(
                self.interaction_circular_basis.flattened(
                    interaction_distances[quadruplet_interaction_index],
                    cosine_abd,
                )
            )
            quadruplet_radial = self.quadruplet_basis_down(
                self.tensor_basis.radial_components(distances)
            )
            quadruplet_angular = self.tensor_basis.angular_components(
                cosine_cab, cosine_plane, sine_plane
            )
            rbf_quadruplet = self.rbf_quadruplet_down(rbf)

        atoms = self.atom_embedding(atomic_number)
        messages = self.edge_embedding(atoms, rbf, source, target)
        atom_outputs = self.output_blocks[0](
            messages, rbf_output, target, atomic_number.shape[0]
        )
        for block, output in zip(
            self.interaction_blocks, self.output_blocks[1:], strict=True
        ):
            atoms, messages = block(
                atoms,
                messages,
                source,
                target,
                reverse_edge,
                rbf_triplet,
                triplet_radial,
                triplet_angular,
                triplet_reduce,
                triplet_expand,
                rbf_atom,
                rbf_quadruplet=rbf_quadruplet,
                quadruplet_cbf=quadruplet_cbf,
                quadruplet_radial=quadruplet_radial,
                quadruplet_angular=quadruplet_angular,
                quadruplet_reduce=quadruplet_reduce,
                quadruplet_expand=quadruplet_expand,
            )
            atom_outputs = atom_outputs + output(
                messages, rbf_output, target, atomic_number.shape[0]
            )
        return scatter(
            atom_outputs,
            graph_batch,
            dim=0,
            dim_size=num_graphs,
            reduce=self.readout,
        )

    def forward_with_forces(self, batch: Batch) -> tuple[Tensor, Tensor]:
        if not isinstance(getattr(batch, "pos", None), Tensor) or not batch.pos.requires_grad:
            raise ValueError("batch.pos must require gradients to calculate forces")
        prediction = self(batch)
        coordinate_gradient = grad(
            prediction.sum(),
            batch.pos,
            create_graph=self.training,
            retain_graph=True,
            allow_unused=True,
        )[0]
        forces = (
            torch.zeros_like(batch.pos)
            if coordinate_gradient is None
            else -coordinate_gradient
        )
        return prediction, forces

    def _validate_batch(self, batch: Batch) -> dict[str, Tensor | int]:
        common_names = (
            "atomic_number",
            "pos",
            "gemnet_edge_index",
            "gemnet_reverse_edge_index",
            "gemnet_triplet_edge_index",
            "batch",
        )
        common = tuple(getattr(batch, name, None) for name in common_names)
        if not all(isinstance(value, Tensor) for value in common):
            raise ValueError(f"batch must provide {', '.join(common_names)} tensors")
        atomic_number, pos, edge_index, reverse_edge, triplet_index, graph_batch = common
        assert isinstance(atomic_number, Tensor)
        assert isinstance(pos, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(reverse_edge, Tensor)
        assert isinstance(triplet_index, Tensor)
        assert isinstance(graph_batch, Tensor)
        node_count = atomic_number.shape[0] if atomic_number.ndim == 1 else -1
        if (
            atomic_number.ndim != 1
            or node_count < 1
            or atomic_number.dtype != torch.long
            or atomic_number.min() < 1
            or atomic_number.max() > self.max_atomic_number
        ):
            raise ValueError("batch.atomic_number contains an invalid GemNet vocabulary value")
        if pos.shape != (node_count, 3) or pos.dtype != torch.float32:
            raise ValueError("batch.pos must have shape [N, 3] and dtype torch.float32")
        if not bool(torch.isfinite(pos).all()):
            raise ValueError("batch.pos must contain only finite values")
        if any(value.device != pos.device for value in common):
            raise ValueError("all GemNet batch tensors must share the position device")

        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=node_count,
            device=pos.device,
            edge_field="gemnet_edge_index",
            forbid_self_loops=True,
        )
        edge_count = edge_index.shape[1]
        _validate_reverse(edge_index, reverse_edge)
        _validate_triplets(edge_index, triplet_index)
        _validate_cutoff(pos, edge_index, self.cutoff, "gemnet_edge_index", self.epsilon)
        result: dict[str, Tensor | int] = {
            "atomic_number": atomic_number,
            "pos": pos,
            "edge_index": edge_index,
            "reverse_edge": reverse_edge,
            "triplet_index": triplet_index,
            "graph_batch": graph_batch,
            "num_graphs": num_graphs,
        }

        if self.use_quadruplets:
            q_names = (
                "gemnet_interaction_edge_index",
                "gemnet_quadruplet_edge_index",
                "gemnet_quadruplet_interaction_index",
            )
            q_values = tuple(getattr(batch, name, None) for name in q_names)
            if not all(isinstance(value, Tensor) for value in q_values):
                raise ValueError(f"batch must provide {', '.join(q_names)} tensors")
            interaction_edge_index, quadruplet_edge_index, quadruplet_interaction_index = q_values
            assert isinstance(interaction_edge_index, Tensor)
            assert isinstance(quadruplet_edge_index, Tensor)
            assert isinstance(quadruplet_interaction_index, Tensor)
            if any(value.device != pos.device for value in q_values):
                raise ValueError("all GemNet-Q tensors must share the position device")
            validate_batched_molecular_graph(
                interaction_edge_index,
                graph_batch,
                num_nodes=node_count,
                device=pos.device,
                edge_field="gemnet_interaction_edge_index",
                forbid_self_loops=True,
            )
            _validate_cutoff(
                pos,
                interaction_edge_index,
                self.interaction_cutoff,
                "gemnet_interaction_edge_index",
                self.epsilon,
            )
            _validate_quadruplets(
                edge_index,
                interaction_edge_index,
                quadruplet_edge_index,
                quadruplet_interaction_index,
                edge_count,
            )
            result.update(
                interaction_edge_index=interaction_edge_index,
                quadruplet_edge_index=quadruplet_edge_index,
                quadruplet_interaction_index=quadruplet_interaction_index,
            )
        return result


class GemNetT(GemNet):
    variant = "t"
    required_batch_fields = (
        "atomic_number",
        "pos",
        "gemnet_edge_index",
        "gemnet_reverse_edge_index",
        "gemnet_triplet_edge_index",
        "batch",
    )

class GemNetQ(GemNet):
    variant = "q"
    required_batch_fields = (
        "atomic_number",
        "pos",
        "gemnet_edge_index",
        "gemnet_reverse_edge_index",
        "gemnet_triplet_edge_index",
        "gemnet_interaction_edge_index",
        "gemnet_quadruplet_edge_index",
        "gemnet_quadruplet_interaction_index",
        "batch",
    )

def _validate_reverse(edge_index: Tensor, reverse_edge: Tensor) -> None:
    edge_count = edge_index.shape[1]
    if reverse_edge.shape != (edge_count,) or reverse_edge.dtype != torch.long:
        raise ValueError("batch.gemnet_reverse_edge_index must have shape [E] and dtype long")
    if edge_count == 0:
        return
    if reverse_edge.min() < 0 or reverse_edge.max() >= edge_count:
        raise ValueError("batch.gemnet_reverse_edge_index contains an invalid edge ID")
    if not torch.equal(reverse_edge[reverse_edge], torch.arange(edge_count, device=reverse_edge.device)):
        raise ValueError("batch.gemnet_reverse_edge_index must be an involution")
    if not torch.equal(edge_index[:, reverse_edge], edge_index.flip(0)):
        raise ValueError("batch.gemnet_reverse_edge_index must map every edge to its reverse")


def _validate_triplets(edge_index: Tensor, triplet_index: Tensor) -> None:
    edge_count = edge_index.shape[1]
    if triplet_index.ndim != 2 or triplet_index.shape[0] != 2 or triplet_index.dtype != torch.long:
        raise ValueError("batch.gemnet_triplet_edge_index must have shape [2, T] and dtype long")
    reduce_edge, expand_edge = triplet_index
    if reduce_edge.numel() == 0:
        return
    if reduce_edge.min() < 0 or expand_edge.min() < 0 or reduce_edge.max() >= edge_count or expand_edge.max() >= edge_count:
        raise ValueError("batch.gemnet_triplet_edge_index contains an invalid edge ID")
    if bool((reduce_edge[1:] < reduce_edge[:-1]).any()):
        raise ValueError("GemNet triplets must be sorted by their reducing edge")
    source, target = edge_index
    if not torch.equal(target[reduce_edge], target[expand_edge]):
        raise ValueError("GemNet triplet edges must point towards the same atom")
    if bool((source[reduce_edge] == source[expand_edge]).any()):
        raise ValueError("GemNet triplets must contain distinct source atoms")
    encoded = reduce_edge * max(edge_count, 1) + expand_edge
    if torch.unique(encoded).numel() != encoded.numel():
        raise ValueError("GemNet triplets must not contain duplicates")


def _validate_quadruplets(
    edge_index: Tensor,
    interaction_edge_index: Tensor,
    quadruplet_edge_index: Tensor,
    quadruplet_interaction_index: Tensor,
    edge_count: int,
) -> None:
    interaction_count = interaction_edge_index.shape[1]
    if quadruplet_edge_index.ndim != 2 or quadruplet_edge_index.shape[0] != 2 or quadruplet_edge_index.dtype != torch.long:
        raise ValueError("batch.gemnet_quadruplet_edge_index must have shape [2, Q] and dtype long")
    if quadruplet_interaction_index.shape != (quadruplet_edge_index.shape[1],) or quadruplet_interaction_index.dtype != torch.long:
        raise ValueError("batch.gemnet_quadruplet_interaction_index must have shape [Q] and dtype long")
    ca_edge, db_edge = quadruplet_edge_index
    if ca_edge.numel() == 0:
        return
    if ca_edge.min() < 0 or db_edge.min() < 0 or ca_edge.max() >= edge_count or db_edge.max() >= edge_count:
        raise ValueError("GemNet quadruplets contain an invalid embedding-edge ID")
    if quadruplet_interaction_index.min() < 0 or quadruplet_interaction_index.max() >= interaction_count:
        raise ValueError("GemNet quadruplets contain an invalid interaction-edge ID")
    if bool((ca_edge[1:] < ca_edge[:-1]).any()):
        raise ValueError("GemNet quadruplets must be sorted by their reducing edge")
    edge_source, edge_target = edge_index
    interaction_source, interaction_target = interaction_edge_index
    c, a = edge_source[ca_edge], edge_target[ca_edge]
    d, b = edge_source[db_edge], edge_target[db_edge]
    if not torch.equal(a, interaction_target[quadruplet_interaction_index]) or not torch.equal(b, interaction_source[quadruplet_interaction_index]):
        raise ValueError("GemNet quadruplet endpoints disagree with the interaction edge")
    if bool(((a == b) | (a == c) | (a == d) | (b == c) | (b == d) | (c == d)).any()):
        raise ValueError("GemNet quadruplets must contain four distinct atoms")


def _validate_cutoff(
    pos: Tensor,
    edge_index: Tensor,
    cutoff: float,
    field: str,
    epsilon: float,
) -> None:
    if edge_index.shape[1] == 0:
        return
    source, target = edge_index
    encoded = source * pos.shape[0] + target
    if torch.unique(encoded).numel() != encoded.numel():
        raise ValueError(f"batch.{field} must not contain duplicate edges")
    distances = torch.linalg.vector_norm(pos[source] - pos[target], dim=-1)
    if bool((distances <= epsilon).any()) or bool((distances > cutoff + 1.0e-6).any()):
        raise ValueError(f"batch.{field} contains an invalid radius edge")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = ["GemNet", "GemNetQ", "GemNetT"]

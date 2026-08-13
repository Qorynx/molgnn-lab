"""Directed-edge neural blocks for the original DimeNet architecture."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


class EmbeddingBlock(nn.Module):
    """Construct initial messages for directed radius edges ``j -> i``.

    The radial projection is intentionally linear and bias-free before the
    concatenation.  This preserves the paper-corrected embedding profile and
    avoids the legacy source's extra bias and activation at this point.
    """

    def __init__(
        self,
        num_radial: int,
        hidden_dim: int,
        max_atomic_number: int,
    ) -> None:
        super().__init__()
        _positive_int(num_radial, "num_radial")
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(max_atomic_number, "max_atomic_number")
        self.num_radial = num_radial
        self.hidden_dim = hidden_dim
        self.max_atomic_number = max_atomic_number
        self.atom_embedding = nn.Embedding(max_atomic_number + 1, hidden_dim)
        self.radial_projection = nn.Linear(num_radial, hidden_dim, bias=False)
        self.message_projection = nn.Linear(3 * hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the atom table and dense projections independently."""

        nn.init.uniform_(self.atom_embedding.weight, -math.sqrt(3), math.sqrt(3))
        glorot_orthogonal_(self.radial_projection.weight)
        glorot_orthogonal_(self.message_projection.weight)
        nn.init.zeros_(self.message_projection.bias)

    def forward(
        self,
        atomic_number: Tensor,
        rbf: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Return one initial hidden message per ``source -> target`` edge."""

        _validate_atomic_number(atomic_number, self.max_atomic_number)
        edge_count = _validate_edge_endpoints(
            source, target, atomic_number.shape[0], atomic_number.device
        )
        _validate_features(rbf, edge_count, self.num_radial, "rbf", atomic_number)

        atom_embeddings = self.atom_embedding(atomic_number)
        radial_features = self.radial_projection(rbf)
        return self.activation(
            self.message_projection(
                torch.cat(
                    (
                        atom_embeddings[source],
                        atom_embeddings[target],
                        radial_features,
                    ),
                    dim=-1,
                )
            )
        )


class ResidualLayer(nn.Module):
    """The two-dense-layer SiLU residual unit used inside interactions."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot_orthogonal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        glorot_orthogonal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    def forward(self, values: Tensor) -> Tensor:
        _validate_hidden(values, self.hidden_dim, "values")
        return values + self.activation(
            self.linear2(self.activation(self.linear1(values)))
        )


class InteractionBlock(nn.Module):
    """One directional ``k -> j -> i`` interaction/update block.

    ``target_rbf`` remains indexed by all directed ``j -> i`` edges.  The
    block explicitly gathers it with ``idx_ji`` so radial modulation uses the
    target edge distance ``d_ji``.  The supplied spherical features already
    correspond to incoming ``d_kj``/angle pairs, keeping the two branches
    distinct by construction.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_bilinear: int,
        num_spherical: int,
        num_radial: int,
        *,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
    ) -> None:
        super().__init__()
        for value, name in (
            (hidden_dim, "hidden_dim"),
            (num_bilinear, "num_bilinear"),
            (num_spherical, "num_spherical"),
            (num_radial, "num_radial"),
        ):
            _positive_int(value, name)
        _nonnegative_int(num_before_skip, "num_before_skip")
        _nonnegative_int(num_after_skip, "num_after_skip")
        self.hidden_dim = hidden_dim
        self.num_bilinear = num_bilinear
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.num_before_skip = num_before_skip
        self.num_after_skip = num_after_skip

        self.radial_projection = nn.Linear(num_radial, hidden_dim, bias=False)
        self.spherical_projection = nn.Linear(
            num_spherical * num_radial, num_bilinear, bias=False
        )
        self.incoming_projection = nn.Linear(hidden_dim, hidden_dim)
        self.target_projection = nn.Linear(hidden_dim, hidden_dim)
        self.bilinear = nn.Parameter(torch.empty(hidden_dim, num_bilinear, hidden_dim))
        self.before_skip = nn.ModuleList(
            ResidualLayer(hidden_dim) for _ in range(num_before_skip)
        )
        self.final_projection = nn.Linear(hidden_dim, hidden_dim)
        self.after_skip = nn.ModuleList(
            ResidualLayer(hidden_dim) for _ in range(num_after_skip)
        )
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot_orthogonal_(self.radial_projection.weight)
        glorot_orthogonal_(self.spherical_projection.weight)
        glorot_orthogonal_(self.incoming_projection.weight)
        nn.init.zeros_(self.incoming_projection.bias)
        glorot_orthogonal_(self.target_projection.weight)
        nn.init.zeros_(self.target_projection.bias)
        nn.init.normal_(self.bilinear, mean=0.0, std=2 / self.hidden_dim)
        for layer in self.before_skip:
            layer.reset_parameters()
        glorot_orthogonal_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
        for layer in self.after_skip:
            layer.reset_parameters()

    def forward(
        self,
        messages: Tensor,
        target_rbf: Tensor,
        sbf: Tensor,
        idx_kj: Tensor,
        idx_ji: Tensor,
    ) -> Tensor:
        """Update all directed-edge messages from the supplied triplets."""

        _validate_hidden(messages, self.hidden_dim, "messages")
        edge_count = messages.shape[0]
        _validate_features(
            target_rbf, edge_count, self.num_radial, "target_rbf", messages
        )
        triplet_count = _validate_triplet_indices(
            idx_kj, idx_ji, edge_count, messages.device
        )
        _validate_features(
            sbf,
            triplet_count,
            self.num_spherical * self.num_radial,
            "sbf",
            messages,
        )

        target_messages = self.activation(self.target_projection(messages))
        incoming_messages = self.activation(self.incoming_projection(messages))
        target_filter = self.radial_projection(target_rbf[idx_ji])
        spherical_filter = self.spherical_projection(sbf)
        triplet_messages = torch.einsum(
            "qb,qf,fbh->qh",
            spherical_filter,
            incoming_messages[idx_kj] * target_filter,
            self.bilinear,
        )
        aggregated = scatter(
            triplet_messages,
            idx_ji,
            dim=0,
            dim_size=edge_count,
            reduce="sum",
        )

        hidden = target_messages + aggregated
        for layer in self.before_skip:
            hidden = layer(hidden)
        hidden = self.activation(self.final_projection(hidden)) + messages
        for layer in self.after_skip:
            hidden = layer(hidden)
        return hidden


class OutputBlock(nn.Module):
    """Convert directed messages into additive atom-wise target contributions."""

    def __init__(
        self,
        num_radial: int,
        hidden_dim: int,
        num_targets: int,
        *,
        num_dense_output: int = 3,
        output_initializer: str = "zeros",
    ) -> None:
        super().__init__()
        for value, name in (
            (num_radial, "num_radial"),
            (hidden_dim, "hidden_dim"),
            (num_targets, "num_targets"),
        ):
            _positive_int(value, name)
        _nonnegative_int(num_dense_output, "num_dense_output")
        if output_initializer not in {"zeros", "glorot_orthogonal"}:
            raise ValueError(
                "output_initializer must be 'zeros' or 'glorot_orthogonal'"
            )
        self.num_radial = num_radial
        self.hidden_dim = hidden_dim
        self.num_targets = num_targets
        self.num_dense_output = num_dense_output
        self.output_initializer = output_initializer
        self.radial_projection = nn.Linear(num_radial, hidden_dim, bias=False)
        self.dense_layers = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_dense_output)
        )
        self.output_projection = nn.Linear(hidden_dim, num_targets, bias=False)
        self.activation = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        glorot_orthogonal_(self.radial_projection.weight)
        for layer in self.dense_layers:
            glorot_orthogonal_(layer.weight)
            nn.init.zeros_(layer.bias)
        if self.output_initializer == "zeros":
            nn.init.zeros_(self.output_projection.weight)
        else:
            glorot_orthogonal_(self.output_projection.weight)

    def forward(
        self,
        messages: Tensor,
        rbf: Tensor,
        target: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Return ``[N, T]`` additive values after summing incoming edges."""

        _validate_hidden(messages, self.hidden_dim, "messages")
        edge_count = messages.shape[0]
        _validate_features(rbf, edge_count, self.num_radial, "rbf", messages)
        if (
            not isinstance(target, Tensor)
            or target.ndim != 1
            or target.dtype != torch.long
            or target.shape[0] != edge_count
        ):
            raise ValueError("target must have shape [E] and dtype torch.long")
        if target.device != messages.device:
            raise ValueError("target and messages must share a device")
        _positive_int(num_nodes, "num_nodes")
        if target.numel() and (target.min() < 0 or target.max() >= num_nodes):
            raise ValueError("target contains a node index outside [0, num_nodes)")

        values = self.radial_projection(rbf) * messages
        values = scatter(values, target, dim=0, dim_size=num_nodes, reduce="sum")
        for layer in self.dense_layers:
            values = self.activation(layer(values))
        return self.output_projection(values)


@torch.no_grad()
def glorot_orthogonal_(tensor: Tensor, scale: float = 2.0) -> None:
    """Initialize a dense tensor with DimeNet's scaled orthogonal profile."""

    if tensor.ndim < 2:
        raise ValueError("tensor must have at least two dimensions")
    nn.init.orthogonal_(tensor)
    variance = tensor.var(unbiased=False)
    if variance <= 0:
        variance = tensor.new_tensor(1.0)
    factor = torch.sqrt(
        tensor.new_tensor(scale) / ((tensor.shape[-2] + tensor.shape[-1]) * variance)
    )
    tensor.mul_(factor)


def _validate_atomic_number(atomic_number: Tensor, max_atomic_number: int) -> None:
    if (
        not isinstance(atomic_number, Tensor)
        or atomic_number.ndim != 1
        or atomic_number.dtype != torch.long
    ):
        raise ValueError("atomic_number must have shape [N] and dtype torch.long")
    if atomic_number.numel() < 1:
        raise ValueError("atomic_number must contain at least one atom")
    if atomic_number.min() < 1 or atomic_number.max() > max_atomic_number:
        raise ValueError(
            "atomic_number contains a value outside the configured vocabulary"
        )


def _validate_edge_endpoints(
    source: Tensor,
    target: Tensor,
    num_nodes: int,
    device: torch.device,
) -> int:
    if (
        not isinstance(source, Tensor)
        or not isinstance(target, Tensor)
        or source.ndim != 1
        or target.ndim != 1
        or source.dtype != torch.long
        or target.dtype != torch.long
        or source.shape != target.shape
    ):
        raise ValueError("source and target must be matching [E] torch.long tensors")
    if source.device != device or target.device != device:
        raise ValueError("source, target, and atomic_number must share a device")
    if source.numel() and (
        source.min() < 0
        or target.min() < 0
        or source.max() >= num_nodes
        or target.max() >= num_nodes
    ):
        raise ValueError("source or target contains an invalid node index")
    return source.shape[0]


def _validate_triplet_indices(
    idx_kj: Tensor,
    idx_ji: Tensor,
    edge_count: int,
    device: torch.device,
) -> int:
    if (
        not isinstance(idx_kj, Tensor)
        or not isinstance(idx_ji, Tensor)
        or idx_kj.ndim != 1
        or idx_ji.ndim != 1
        or idx_kj.dtype != torch.long
        or idx_ji.dtype != torch.long
        or idx_kj.shape != idx_ji.shape
    ):
        raise ValueError("idx_kj and idx_ji must be matching [Q] torch.long tensors")
    if idx_kj.device != device or idx_ji.device != device:
        raise ValueError("triplet indices and messages must share a device")
    if idx_kj.numel() and (
        idx_kj.min() < 0
        or idx_ji.min() < 0
        or idx_kj.max() >= edge_count
        or idx_ji.max() >= edge_count
    ):
        raise ValueError("triplet indices contain an invalid directed-edge index")
    return idx_kj.shape[0]


def _validate_features(
    values: Tensor,
    row_count: int,
    width: int,
    name: str,
    reference: Tensor,
) -> None:
    if (
        not isinstance(values, Tensor)
        or values.shape != (row_count, width)
        or not torch.is_floating_point(values)
    ):
        raise ValueError(
            f"{name} must have shape [{row_count}, {width}] and be a floating tensor"
        )
    if values.device != reference.device:
        raise ValueError(f"{name} and messages must share a device")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def _validate_hidden(values: Tensor, hidden_dim: int, name: str) -> None:
    if (
        not isinstance(values, Tensor)
        or values.ndim != 2
        or values.shape[1] != hidden_dim
        or not torch.is_floating_point(values)
    ):
        raise ValueError(f"{name} must have shape [E, {hidden_dim}] and be floating")
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


__all__ = [
    "EmbeddingBlock",
    "InteractionBlock",
    "OutputBlock",
    "ResidualLayer",
    "glorot_orthogonal_",
]

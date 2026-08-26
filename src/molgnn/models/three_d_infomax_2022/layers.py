"""PyTorch/PyG port of the official 3D Infomax PNA building blocks.

Provenance: ``OFFICIAL CODE`` HannesStark/3DInfomax revision
``5cd32629c690e119bcae8726acedefdb0aa037fc`` (files ``models/pna.py``,
``models/base_layers.py``, ``commons/mol_encoder.py``), itself derived from
the reference PNA implementation. Module/attribute names mirror the official
classes so pretrained checkpoints map without renaming.

Faithful quirks kept on purpose:

- ``FCLayer`` applies Linear -> activation -> dropout -> BatchNorm;
- linear weights use ``xavier_uniform_`` with gain ``1 / in_dim``;
- DGL invokes the reducer in same-in-degree buckets, so aggregation and the
  amplification/attenuation scalers use each node's actual in-degree;
  ``avg_d["log"]`` remains hard-coded to ``1.0`` by the official encoder;
- with a single scaler the scaled-concat branch is skipped entirely;
- nodes without incoming edges receive all-zero aggregate features.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

AGGREGATOR_EPS = 1e-5


class AtomEncoder(nn.Module):
    """Official OGB categorical atom encoder (one embedding table per column)."""

    def __init__(
        self,
        emb_dim: int,
        vocab_sizes: tuple[int, ...] = (119, 4, 12, 12, 10, 6, 6, 2, 2),
    ) -> None:
        super().__init__()
        self.atom_embedding_list = nn.ModuleList(
            [nn.Embedding(size, emb_dim) for size in vocab_sizes]
        )
        for embedding in self.atom_embedding_list:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x: Tensor) -> Tensor:
        embedding = x.new_zeros((x.shape[0], self.atom_embedding_list[0].embedding_dim))
        for column, table in enumerate(self.atom_embedding_list):
            embedding = embedding + table(x[:, column])
        return embedding


class BondEncoder(nn.Module):
    """Official OGB categorical bond encoder."""

    def __init__(self, emb_dim: int, vocab_sizes: tuple[int, ...] = (5, 6, 2)) -> None:
        super().__init__()
        self.bond_embedding_list = nn.ModuleList(
            [nn.Embedding(size, emb_dim) for size in vocab_sizes]
        )
        for embedding in self.bond_embedding_list:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, edge_attr: Tensor) -> Tensor:
        embedding = edge_attr.new_zeros(
            (edge_attr.shape[0], self.bond_embedding_list[0].embedding_dim)
        )
        for column, table in enumerate(self.bond_embedding_list):
            embedding = embedding + table(edge_attr[:, column])
        return embedding


def _activation(value: str | nn.Module) -> nn.Module | None:
    if callable(value):
        return value
    lookup = {
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "leakyrelu": nn.LeakyReLU,
        "softplus": nn.Softplus,
        "none": None,
    }
    key = value.lower()
    if key not in lookup:
        raise ValueError(f"unhandled activation function {value!r}")
    factory = lookup[key]
    return factory() if factory is not None else None


class FCLayer(nn.Module):
    """Linear -> activation -> dropout -> BatchNorm, official order."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str | nn.Module = "relu",
        dropout: float = 0.0,
        batch_norm: bool = False,
        batch_norm_momentum: float = 0.1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.batch_norm = (
            nn.BatchNorm1d(out_dim, momentum=batch_norm_momentum) if batch_norm else None
        )
        self.activation = _activation(activation)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.0 / in_dim)
        if self.linear.bias is not None:
            self.linear.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        h = self.linear(x)
        if self.activation is not None:
            h = self.activation(h)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.batch_norm is not None:
            h = self.batch_norm(h)
        return h


class MLP(nn.Module):
    """Official stacked FCLayers with separate mid/last settings."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: int,
        hidden_size: int | None = None,
        mid_activation: str | nn.Module = "relu",
        last_activation: str | nn.Module = "none",
        dropout: float = 0.0,
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        effective_hidden = in_dim if hidden_size is None else hidden_size
        self.fully_connected = nn.ModuleList()
        if layers <= 1:
            self.fully_connected.append(
                FCLayer(
                    in_dim,
                    out_dim,
                    activation=last_activation,
                    batch_norm=last_batch_norm,
                    dropout=dropout,
                    batch_norm_momentum=batch_norm_momentum,
                )
            )
        else:
            self.fully_connected.append(
                FCLayer(
                    in_dim,
                    effective_hidden,
                    activation=mid_activation,
                    batch_norm=mid_batch_norm,
                    dropout=dropout,
                    batch_norm_momentum=batch_norm_momentum,
                )
            )
            for _ in range(layers - 2):
                self.fully_connected.append(
                    FCLayer(
                        effective_hidden,
                        effective_hidden,
                        activation=mid_activation,
                        batch_norm=mid_batch_norm,
                        dropout=dropout,
                        batch_norm_momentum=batch_norm_momentum,
                    )
                )
            self.fully_connected.append(
                FCLayer(
                    effective_hidden,
                    out_dim,
                    activation=last_activation,
                    batch_norm=last_batch_norm,
                    dropout=dropout,
                    batch_norm_momentum=batch_norm_momentum,
                )
            )

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.fully_connected:
            x = layer(x)
        return x


def _segment_statistics(
    messages: Tensor, dst: Tensor, num_nodes: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Mean/variance and in-degree for DGL-equivalent degree buckets."""

    hidden = messages.shape[1]
    counts = torch.bincount(dst, minlength=num_nodes)
    total = messages.new_zeros((num_nodes, hidden))
    squared = messages.new_zeros((num_nodes, hidden))
    total.index_add_(0, dst, messages)
    squared.index_add_(0, dst, messages * messages)
    divisor = counts.clamp_min(1).to(messages.dtype).unsqueeze(-1)
    mean = total / divisor
    var = torch.relu(squared / divisor - mean * mean)
    return mean, var, counts


def _segment_min(messages: Tensor, dst: Tensor, num_nodes: int) -> Tensor:
    result = messages.new_full((num_nodes, messages.shape[1]), float("inf"))
    result.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(messages), messages, reduce="amin", include_self=True)
    has_messages = torch.bincount(dst, minlength=num_nodes) > 0
    return torch.where(has_messages.unsqueeze(-1), result, torch.zeros_like(result))


def _segment_max(messages: Tensor, dst: Tensor, num_nodes: int) -> Tensor:
    result = messages.new_full((num_nodes, messages.shape[1]), float("-inf"))
    result.scatter_reduce_(0, dst.unsqueeze(-1).expand_as(messages), messages, reduce="amax", include_self=True)
    has_messages = torch.bincount(dst, minlength=num_nodes) > 0
    return torch.where(has_messages.unsqueeze(-1), result, torch.zeros_like(result))


def scale_identity(h: Tensor, degree: Tensor, avg_d: dict[str, float]) -> Tensor:
    return h


def scale_amplification(h: Tensor, degree: Tensor, avg_d: dict[str, float]) -> Tensor:
    # log(D + 1) / d * h with the official hard-coded average d.
    log_degree = torch.log1p(degree.to(h.dtype)).unsqueeze(-1)
    return h * (log_degree / avg_d["log"])


def scale_attenuation(h: Tensor, degree: Tensor, avg_d: dict[str, float]) -> Tensor:
    # (log(D + 1))^-1 / d * h.
    log_degree = torch.log1p(degree.to(h.dtype)).unsqueeze(-1)
    safe_log_degree = log_degree.clamp_min(torch.finfo(h.dtype).tiny)
    factor = torch.where(
        log_degree > 0,
        avg_d["log"] / safe_log_degree,
        torch.zeros_like(log_degree),
    )
    return h * factor


PNA_SCALERS = {
    "identity": scale_identity,
    "amplification": scale_amplification,
    "attenuation": scale_attenuation,
}


class PNALayer(nn.Module):
    """One PNA propagation layer over directed edges ``src -> dst``."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        in_dim_edges: int,
        aggregators: list[str],
        scalers: list[str],
        activation: str | nn.Module = "relu",
        last_activation: str | nn.Module = "none",
        dropout: float = 0.0,
        residual: bool = True,
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        batch_norm_momentum: float = 0.1,
        avg_d: dict[str, float] | None = None,
        posttrans_layers: int = 2,
        pretrans_layers: int = 1,
    ) -> None:
        super().__init__()
        self.aggregator_names = list(aggregators)
        self.scaler_names = list(scalers)
        self.scalers = [PNA_SCALERS[name] for name in self.scaler_names]
        self.edge_features = in_dim_edges > 0
        self.avg_d = {"log": 1.0} if avg_d is None else dict(avg_d)
        self.residual = residual
        if in_dim != out_dim:
            self.residual = False

        self.pretrans = MLP(
            in_dim=(2 * in_dim + in_dim_edges) if self.edge_features else (2 * in_dim),
            hidden_size=in_dim,
            out_dim=in_dim,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            layers=pretrans_layers,
            mid_activation=activation,
            dropout=dropout,
            last_activation=last_activation,
            batch_norm_momentum=batch_norm_momentum,
        )
        self.posttrans = MLP(
            in_dim=(len(self.aggregator_names) * len(self.scalers) + 1) * in_dim,
            hidden_size=out_dim,
            out_dim=out_dim,
            layers=posttrans_layers,
            mid_activation=activation,
            last_activation=last_activation,
            dropout=dropout,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            batch_norm_momentum=batch_norm_momentum,
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor | None) -> Tensor:
        src, dst = edge_index[0], edge_index[1]
        num_nodes = x.shape[0]
        h_in = x

        parts = [x[src], x[dst]]
        if self.edge_features:
            if edge_attr is None:
                raise ValueError("layer expects edge features but none were provided")
            parts.append(edge_attr)
        messages = self.pretrans(torch.cat(parts, dim=-1))

        mean, var, degree = _segment_statistics(messages, dst, num_nodes)
        std = torch.sqrt(var + AGGREGATOR_EPS)
        std = torch.where(degree.unsqueeze(-1) > 0, std, torch.zeros_like(std))
        statistics = {
            "mean": mean,
            "std": std,
            "min": _segment_min(messages, dst, num_nodes),
            "max": _segment_max(messages, dst, num_nodes),
        }
        aggregated = torch.cat(
            [statistics[name] for name in self.aggregator_names], dim=-1
        )

        if len(self.scalers) > 1:
            scaled = torch.cat(
                [
                    scaler(aggregated, degree=degree, avg_d=self.avg_d)
                    for scaler in self.scalers
                ],
                dim=-1,
            )
        else:
            # Official quirk: a lone scaler bypasses scaling completely.
            scaled = aggregated
        h = self.posttrans(torch.cat([h_in, scaled], dim=-1))
        if self.residual:
            h = h + h_in
        return h


class PNAGNN(nn.Module):
    """Official ``PNAGNN``: categorical encoders followed by PNA layers."""

    def __init__(
        self,
        hidden_dim: int,
        aggregators: list[str],
        scalers: list[str],
        residual: bool = True,
        activation: str | nn.Module = "relu",
        last_activation: str | nn.Module = "none",
        mid_batch_norm: bool = False,
        last_batch_norm: bool = False,
        batch_norm_momentum: float = 0.1,
        propagation_depth: int = 5,
        dropout: float = 0.0,
        posttrans_layers: int = 1,
        pretrans_layers: int = 1,
        atom_vocab_sizes: tuple[int, ...] = (119, 4, 12, 12, 10, 6, 6, 2, 2),
        bond_vocab_sizes: tuple[int, ...] = (5, 6, 2),
        **kwargs: object,
    ) -> None:
        super().__init__()
        self.mp_layers = nn.ModuleList()
        for _ in range(propagation_depth):
            self.mp_layers.append(
                PNALayer(
                    in_dim=hidden_dim,
                    out_dim=int(hidden_dim),
                    in_dim_edges=hidden_dim,
                    aggregators=list(aggregators),
                    scalers=list(scalers),
                    residual=residual,
                    dropout=dropout,
                    activation=activation,
                    last_activation=last_activation,
                    mid_batch_norm=mid_batch_norm,
                    last_batch_norm=last_batch_norm,
                    avg_d={"log": 1.0},
                    posttrans_layers=posttrans_layers,
                    pretrans_layers=pretrans_layers,
                    batch_norm_momentum=batch_norm_momentum,
                )
            )
        self.atom_encoder = AtomEncoder(emb_dim=hidden_dim, vocab_sizes=atom_vocab_sizes)
        self.bond_encoder = BondEncoder(emb_dim=hidden_dim, vocab_sizes=bond_vocab_sizes)

    def forward(self, atom_attr: Tensor, edge_index: Tensor, bond_attr: Tensor | None) -> Tensor:
        x = self.atom_encoder(atom_attr)
        edge_h = self.bond_encoder(bond_attr) if bond_attr is not None else None
        for layer in self.mp_layers:
            x = layer(x, edge_index, edge_h)
        return x


__all__ = [
    "AGGREGATOR_EPS",
    "MLP",
    "PNAGNN",
    "PNA_SCALERS",
    "AtomEncoder",
    "BondEncoder",
    "FCLayer",
    "PNALayer",
    "scale_amplification",
    "scale_attenuation",
    "scale_identity",
]

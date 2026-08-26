"""Net3D: distance-only invariant complete-graph encoder for pretraining.

Provenance: ``OFFICIAL CODE`` revision ``5cd32629c690e119bcae8726acedefdb0aa037fc``
(``models/net3d.py``). Contract kept from the source:

- complete directed graph without self-loops is built by the model-owned
  collator; this module only consumes ``edge_index`` and distances;
- inputs are coordinates/distances plus ONE learned constant node vector —
  atomic numbers are deliberately never used so the 3-D branch cannot
  shortcut mutual information with chemical identity;
- Fourier distance encoding uses four powers of two and appends the raw
  distance (9 channels for the official profile);
- each layer updates edge states additively and gates messages through a
  learned sigmoid soft-edge network;
- checkpoint profile: hidden 20, one propagation layer, ``mean`` node
  reduction, min/max/mean readout, output 256.

Distances are invariant to translation/rotation/reflection and the readouts
are permutation invariant, matching the paper's design.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import MLP


def fourier_encode_dist(x: Tensor, num_encodings: int = 4, include_self: bool = True) -> Tensor:
    """Official Fourier encoding: sin/cos at scales ``2**k`` plus raw value."""

    original = x.unsqueeze(-1)
    scales = torch.pow(
        2.0, torch.arange(num_encodings, device=x.device, dtype=x.dtype)
    )
    scaled = original / scales
    encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1)
    if include_self:
        encoded = torch.cat([encoded, original], dim=-1)
    # The official helper ends with a bare squeeze(); with the source's
    # [E, 1] distance inputs this yields [E, 2k + 1].
    return encoded.squeeze()


class Net3DLayer(nn.Module):
    """Soft-edge gated propagation layer over the complete directed graph."""

    def __init__(
        self,
        edge_dim: int,
        hidden_dim: int,
        batch_norm: bool,
        batch_norm_momentum: float,
        dropout: float,
        mid_activation: str | nn.Module,
        reduce_func: str,
        message_net_layers: int,
        update_net_layers: int,
    ) -> None:
        super().__init__()
        if reduce_func not in {"sum", "mean"}:
            raise ValueError(f"reduce function not supported: {reduce_func!r}")
        self.reduce_name = reduce_func
        self.message_network = MLP(
            in_dim=hidden_dim * 2 + edge_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            layers=message_net_layers,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation=mid_activation,
        )
        self.update_network = MLP(
            in_dim=hidden_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            layers=update_net_layers,
            mid_activation=mid_activation,
            dropout=dropout,
            last_activation="none",
        )
        self.soft_edge_network = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: Tensor, edge_state: Tensor, edge_index: Tensor, num_nodes: int
    ) -> tuple[Tensor, Tensor]:
        src, dst = edge_index[0], edge_index[1]
        message_input = torch.cat([x[src], x[dst], edge_state], dim=-1)
        message = self.message_network(message_input)
        # Edge states accumulate updates across propagation layers.
        edge_state = edge_state + message
        gate = torch.sigmoid(self.soft_edge_network(message))
        gated = message * gate

        total = x.new_zeros((num_nodes, x.shape[1]))
        total.index_add_(0, dst, gated)
        if self.reduce_name == "mean":
            counts = torch.bincount(dst, minlength=num_nodes).unsqueeze(-1).to(x.dtype)
            aggregated = total / counts.clamp_min(1.0)
        else:
            aggregated = total

        h = x + self.update_network(aggregated + x)
        return h, edge_state


class Net3D(nn.Module):
    """Invariant complete-graph encoder producing one vector per conformer."""

    def __init__(
        self,
        *,
        hidden_dim: int = 20,
        target_dim: int = 256,
        fourier_encodings: int = 4,
        propagation_depth: int = 1,
        dropout: float = 0.0,
        batch_norm: bool = True,
        batch_norm_momentum: float = 0.93,
        readout_batchnorm: bool = True,
        readout_hidden_dim: int | None = None,
        readout_layers: int = 1,
        readout_aggregators: tuple[str, ...] = ("min", "max", "mean"),
        node_wise_output_layers: int = 0,
        message_net_layers: int = 1,
        update_net_layers: int = 1,
        reduce_func: str = "mean",
        activation: str = "SiLU",
        use_node_features: bool = False,
        atom_vocab_sizes: tuple[int, ...] = (119, 4, 12, 12, 10, 6, 6, 2, 2),
        **kwargs: object,
    ) -> None:
        super().__init__()
        _ = kwargs  # official configs pass unused keys such as hidden_edge_dim
        if use_node_features:
            # Supported only to mirror the source signature; the paper's 3D
            # branch must stay chemistry-free.
            raise ValueError(
                "Net3D must not consume atom identity; keep use_node_features=False"
            )
        self.fourier_encodings = fourier_encodings
        edge_in_dim = 1 if fourier_encodings == 0 else 2 * fourier_encodings + 1
        self.edge_input = MLP(
            in_dim=edge_in_dim,
            hidden_size=hidden_dim,
            out_dim=hidden_dim,
            mid_batch_norm=batch_norm,
            last_batch_norm=batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            layers=1,
            mid_activation=activation,
            dropout=dropout,
            last_activation=activation,
        )
        # One learned constant vector seeds every node.
        self.node_embedding = nn.Parameter(torch.empty((hidden_dim,)))
        nn.init.normal_(self.node_embedding)

        self.mp_layers = nn.ModuleList(
            [
                Net3DLayer(
                    edge_dim=hidden_dim,
                    hidden_dim=hidden_dim,
                    batch_norm=batch_norm,
                    batch_norm_momentum=batch_norm_momentum,
                    dropout=dropout,
                    mid_activation=activation,
                    reduce_func=reduce_func,
                    message_net_layers=message_net_layers,
                    update_net_layers=update_net_layers,
                )
                for _ in range(propagation_depth)
            ]
        )

        self.node_wise_output_layers = node_wise_output_layers
        if self.node_wise_output_layers > 0:
            self.node_wise_output_network = MLP(
                in_dim=hidden_dim,
                hidden_size=hidden_dim,
                out_dim=hidden_dim,
                mid_batch_norm=batch_norm,
                last_batch_norm=batch_norm,
                batch_norm_momentum=batch_norm_momentum,
                layers=node_wise_output_layers,
                mid_activation=activation,
                dropout=dropout,
                last_activation="none",
            )

        effective_readout_hidden = hidden_dim if readout_hidden_dim is None else readout_hidden_dim
        self.readout_aggregators = list(readout_aggregators)
        # NOTE: the official module passes only mid_batch_norm here even when
        # readout_batchnorm=True, so single-layer readouts carry no BatchNorm.
        self.output = MLP(
            in_dim=hidden_dim * len(self.readout_aggregators),
            hidden_size=effective_readout_hidden,
            out_dim=target_dim,
            mid_batch_norm=readout_batchnorm,
            batch_norm_momentum=batch_norm_momentum,
            layers=readout_layers,
        )

    def forward(
        self,
        positions: Tensor,
        edge_index: Tensor,
        conformer_node_batch: Tensor,
        num_conformers: int,
    ) -> Tensor:
        num_nodes = positions.shape[0]
        x = self.node_embedding.unsqueeze(0).expand(num_nodes, -1)
        distances = torch.norm(positions[edge_index[0]] - positions[edge_index[1]], p=2, dim=-1).unsqueeze(-1)
        edge_state = F.silu(self.edge_input(fourier_encode_dist(distances, self.fourier_encodings)))
        for layer in self.mp_layers:
            x, edge_state = layer(x, edge_state, edge_index, num_nodes)
        if self.node_wise_output_layers > 0:
            x = self.node_wise_output_network(x)

        statistics_total = x.new_zeros((num_conformers, x.shape[1]))
        statistics_total.index_add_(0, conformer_node_batch, x)
        counts = (
            torch.bincount(conformer_node_batch, minlength=num_conformers)
            .unsqueeze(-1)
            .to(x.dtype)
        )
        minimum = x.new_full((num_conformers, x.shape[1]), float("inf"))
        minimum.scatter_reduce_(
            0,
            conformer_node_batch.unsqueeze(-1).expand_as(x),
            x,
            reduce="amin",
            include_self=True,
        )
        maximum = x.new_full((num_conformers, x.shape[1]), float("-inf"))
        maximum.scatter_reduce_(
            0,
            conformer_node_batch.unsqueeze(-1).expand_as(x),
            x,
            reduce="amax",
            include_self=True,
        )
        statistics = {
            "mean": statistics_total / counts.clamp_min(1.0),
            "min": minimum,
            "max": maximum,
            "sum": statistics_total,
        }
        readout = torch.cat([statistics[name] for name in self.readout_aggregators], dim=-1)
        return self.output(readout)


__all__ = ["Net3D", "Net3DLayer", "fourier_encode_dist"]

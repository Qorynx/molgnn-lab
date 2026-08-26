"""3D Infomax downstream: PNA encoder, graph readout, prediction head.

Provenance: ``OFFICIAL CODE`` revision ``5cd32629c690e119bcae8726acedefdb0aa037fc``
(``models/pna.py``). The fine-tuned model of the paper keeps only the 2-D PNA
encoder; Net3D and coordinates are pretraining-only and are never read here.
The predictor returns raw ``[batch_size, num_targets]`` values (no sigmoid)
so shared masked multitask losses apply unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from ..base import BaseMolecularModel
from .layers import MLP, PNAGNN


def _graph_segment_statistics(
    node_features: Tensor, batch: Tensor, num_graphs: int
) -> dict[str, Tensor]:
    total = node_features.new_zeros((num_graphs, node_features.shape[1]))
    squared = node_features.new_zeros((num_graphs, node_features.shape[1]))
    total.index_add_(0, batch, node_features)
    squared.index_add_(0, batch, node_features.square())
    counts = torch.bincount(batch, minlength=num_graphs).unsqueeze(-1).to(node_features.dtype)
    minimum = node_features.new_full((num_graphs, node_features.shape[1]), float("inf"))
    minimum.scatter_reduce_(
        0,
        batch.unsqueeze(-1).expand_as(node_features),
        node_features,
        reduce="amin",
        include_self=True,
    )
    maximum = node_features.new_full((num_graphs, node_features.shape[1]), float("-inf"))
    maximum.scatter_reduce_(
        0,
        batch.unsqueeze(-1).expand_as(node_features),
        node_features,
        reduce="amax",
        include_self=True,
    )
    return {
        "mean": total / counts.clamp_min(1.0),
        "min": minimum,
        "max": maximum,
        "sum": total,
        "std": torch.sqrt(
            torch.relu(squared / counts.clamp_min(1.0) - (total / counts.clamp_min(1.0)).square())
            + 1e-5
        ),
    }


class PNAGraphReadout(nn.Module):
    """Official readout: per-graph aggregators followed by an MLP head."""

    def __init__(
        self,
        hidden_dim: int,
        target_dim: int,
        aggregators: list[str],
        *,
        readout_batchnorm: bool = True,
        readout_hidden_dim: int | None = None,
        readout_layers: int = 2,
        batch_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.aggregators = list(aggregators)
        effective_hidden = hidden_dim if readout_hidden_dim is None else readout_hidden_dim
        self.output = MLP(
            in_dim=hidden_dim * len(self.aggregators),
            hidden_size=effective_hidden,
            out_dim=target_dim,
            mid_batch_norm=readout_batchnorm,
            layers=readout_layers,
            batch_norm_momentum=batch_norm_momentum,
        )

    def forward(self, node_features: Tensor, batch: Tensor) -> Tensor:
        statistics = _graph_segment_statistics(node_features, batch, int(batch.max().item()) + 1)
        readout = torch.cat([statistics[name] for name in self.aggregators], dim=-1)
        return self.output(readout)


class ThreeDInfomax(BaseMolecularModel):
    """Downstream 3D Infomax predictor: pure-topology PNA over OGB features.

    Scratch initialization is the default; a converted encoder-only
    checkpoint loads only when an explicit ``pretrained_checkpoint`` path is
    provided. Coordinates (``pos``) are intentionally ignored so the model
    runs identically on MoleculeNet and native QM9 samples.
    """

    required_batch_fields = (
        "three_d_infomax_atom_attr",
        "three_d_infomax_bond_attr",
        "edge_index",
        "batch",
    )

    def __init__(
        self,
        *,
        atom_dim: int = 9,
        bond_dim: int = 3,
        num_targets: int = 1,
        hidden_dim: int = 70,
        propagation_depth: int = 4,
        aggregators: tuple[str, ...] = ("mean", "max", "min", "std"),
        scalers: tuple[str, ...] = ("identity", "amplification", "attenuation"),
        readout_aggregators: tuple[str, ...] = ("min", "max", "mean"),
        residual: bool = True,
        activation: str = "relu",
        last_activation: str = "none",
        mid_batch_norm: bool = True,
        last_batch_norm: bool = True,
        batch_norm_momentum: float = 0.1,
        dropout: float = 0.0,
        posttrans_layers: int = 1,
        pretrans_layers: int = 1,
        readout_batchnorm: bool = True,
        readout_hidden_dim: int | None = None,
        readout_layers: int = 2,
        pretrained_checkpoint: str | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim < 1 or propagation_depth < 1 or num_targets < 1:
            raise ValueError("hidden_dim, propagation_depth and num_targets must be >= 1")
        unknown_aggregators = set(aggregators) - {"mean", "max", "min", "std"}
        if unknown_aggregators:
            raise ValueError(f"unsupported aggregators: {sorted(unknown_aggregators)}")
        unknown_scalers = set(scalers) - {"identity", "amplification", "attenuation"}
        if unknown_scalers:
            raise ValueError(f"unsupported scalers: {sorted(unknown_scalers)}")
        _ = bond_dim  # accepted for BuildContext symmetry; bond width is fixed

        self.hidden_dim = hidden_dim
        self.num_targets = num_targets
        # Attribute name mirrors the official module for checkpoint loading.
        self.node_gnn = PNAGNN(
            hidden_dim=hidden_dim,
            aggregators=list(aggregators),
            scalers=list(scalers),
            residual=residual,
            activation=activation,
            last_activation=last_activation,
            mid_batch_norm=mid_batch_norm,
            last_batch_norm=last_batch_norm,
            batch_norm_momentum=batch_norm_momentum,
            propagation_depth=propagation_depth,
            dropout=dropout,
            posttrans_layers=posttrans_layers,
            pretrans_layers=pretrans_layers,
        )
        self.readout = PNAGraphReadout(
            hidden_dim=hidden_dim,
            target_dim=num_targets,
            aggregators=list(readout_aggregators),
            readout_batchnorm=readout_batchnorm,
            readout_hidden_dim=readout_hidden_dim,
            readout_layers=readout_layers,
            batch_norm_momentum=batch_norm_momentum,
        )

        if pretrained_checkpoint is not None:
            from .checkpoint import load_pretrained_encoder

            load_pretrained_encoder(self, pretrained_checkpoint)

    def encode_nodes(self, atom_attr: Tensor, edge_index: Tensor, bond_attr: Tensor) -> Tensor:
        return self.node_gnn(atom_attr, edge_index, bond_attr)

    def forward(self, batch: Batch) -> Tensor:
        atom_attr = getattr(batch, "three_d_infomax_atom_attr", None)
        bond_attr = getattr(batch, "three_d_infomax_bond_attr", None)
        edge_index = getattr(batch, "edge_index", None)
        graph_batch = getattr(batch, "batch", None)
        missing = [
            name
            for name, value in (
                ("three_d_infomax_atom_attr", atom_attr),
                ("three_d_infomax_bond_attr", bond_attr),
                ("edge_index", edge_index),
                ("batch", graph_batch),
            )
            if not isinstance(value, Tensor)
        ]
        if missing:
            raise ValueError(f"batch is missing tensor field(s): {', '.join(missing)}")
        node_features = self.encode_nodes(atom_attr, edge_index, bond_attr)
        return self.readout(node_features, graph_batch)


__all__ = ["PNAGraphReadout", "ThreeDInfomax"]

"""ChemRL-GEM self-supervised objectives and local masking utilities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from .constants import (
    ATOMIC_DISTANCE_CUTOFF,
    ATOM_MASK_RATIO,
    CONTEXT_VOCAB_SIZE,
    DEFAULT_ATOMIC_DISTANCE_BINS,
    DEFAULT_DROPOUT,
    DEFAULT_EMBED_DIM,
    DEFAULT_LAYER_NUM,
    FINGERPRINT_SIZE,
)
from .layers import PredictionMLP
from .model import GeoGNNEncoder


@dataclass(frozen=True)
class MaskedGEMView:
    batch: Batch
    masked_nodes: Tensor
    masked_edges: Tensor


def mask_chemrl_gem_batch(
    batch: Batch,
    rate: float = ATOM_MASK_RATIO,
    *,
    generator: torch.Generator | None = None,
) -> MaskedGEMView:
    """Mask selected atoms and their one-hop bond/angle neighborhoods."""

    if not 0 < float(rate) <= 1:
        raise ValueError("mask rate must be in (0, 1]")
    required = (
        "batch",
        "chemrl_gem_atom_attr",
        "chemrl_gem_edge_index",
        "chemrl_gem_bond_attr",
        "chemrl_gem_bond_length",
        "chemrl_gem_angle_edge_index",
        "chemrl_gem_bond_angle",
    )
    if not all(isinstance(getattr(batch, name, None), Tensor) for name in required):
        raise ValueError("batch is missing ChemRL-GEM fields for masking")
    masked = batch.clone()
    atom_attr = masked.chemrl_gem_atom_attr.clone()
    bond_attr = masked.chemrl_gem_bond_attr.clone()
    bond_length = masked.chemrl_gem_bond_length.clone()
    bond_angle = masked.chemrl_gem_bond_angle.clone()
    masked_nodes: list[int] = []
    masked_edges: list[int] = []
    graph_batch = batch.batch
    for graph_id in range(int(graph_batch.max().item()) + 1):
        nodes = torch.nonzero(graph_batch == graph_id, as_tuple=False).flatten()
        count = max(1, min(nodes.numel(), int(nodes.numel() * float(rate))))
        selected = nodes[torch.randperm(nodes.numel(), generator=generator, device=nodes.device)[:count]]
        masked_nodes.extend(selected.tolist())
        source, target = batch.chemrl_gem_edge_index
        edge_mask = (torch.isin(source, selected) | torch.isin(target, selected))
        selected_edges = torch.nonzero(edge_mask, as_tuple=False).flatten()
        masked_edges.extend(selected_edges.tolist())
        atom_attr[selected] = 0
        bond_attr[selected_edges] = 0
        bond_length[selected_edges] = 0
        if selected_edges.numel():
            line_source, line_target = batch.chemrl_gem_angle_edge_index
            angle_mask = torch.isin(line_source, selected_edges) | torch.isin(line_target, selected_edges)
            bond_angle[angle_mask] = 0
    masked.chemrl_gem_atom_attr = atom_attr
    masked.chemrl_gem_bond_attr = bond_attr
    masked.chemrl_gem_bond_length = bond_length
    masked.chemrl_gem_bond_angle = bond_angle
    return MaskedGEMView(
        masked,
        torch.tensor(masked_nodes, dtype=torch.long, device=atom_attr.device),
        torch.tensor(masked_edges, dtype=torch.long, device=bond_attr.device),
    )


def build_geometry_pretraining_targets(batch: Batch) -> dict[str, Tensor]:
    """Derive source-compatible Bar/Blr/Adc targets from a GEM batch."""

    edge_index = batch.chemrl_gem_edge_index
    lengths = batch.chemrl_gem_bond_length
    line_index = batch.chemrl_gem_angle_edge_index
    line_source, line_target = line_index
    first_edge = edge_index[:, line_source]
    second_edge = edge_index[:, line_target]
    triplets = torch.stack((first_edge[0], first_edge[1], second_edge[1]), dim=1)
    graph_batch = batch.batch
    pair_i: list[Tensor] = []
    pair_j: list[Tensor] = []
    distances: list[Tensor] = []
    for graph_id in range(int(graph_batch.max().item()) + 1):
        nodes = torch.nonzero(graph_batch == graph_id, as_tuple=False).flatten()
        pair_i.append(nodes.repeat_interleave(nodes.numel()))
        pair_j.append(nodes.repeat(nodes.numel()))
        distances.append(torch.linalg.vector_norm(batch.pos[pair_i[-1]] - batch.pos[pair_j[-1]], dim=-1))
    return {
        "Ba_node_i": triplets[:, 0],
        "Ba_node_j": triplets[:, 1],
        "Ba_node_k": triplets[:, 2],
        "Ba_bond_angle": batch.chemrl_gem_bond_angle,
        "Bl_node_i": edge_index[0],
        "Bl_node_j": edge_index[1],
        "Bl_bond_length": lengths,
        "Ad_node_i": torch.cat(pair_i) if pair_i else edge_index.new_empty((0,)),
        "Ad_node_j": torch.cat(pair_j) if pair_j else edge_index.new_empty((0,)),
        "Ad_atom_dist": torch.cat(distances) if distances else lengths.new_empty((0,)),
    }


class ChemRLGEMPretrainer(nn.Module):
    """Joint source-style Cm/Fg/Bar/Blr/Adc pretrainer.

    ``targets`` uses the source names documented in ``chemrl_gem.md``.  The
    geometry targets can be generated with :func:`build_geometry_pretraining_targets`;
    fingerprint/context labels remain dataset-provided because their exact
    legacy hash/SMARTS vocabulary is data preprocessing, not model topology.
    """

    def __init__(
        self,
        *,
        embed_dim: int = DEFAULT_EMBED_DIM,
        layer_num: int = DEFAULT_LAYER_NUM,
        dropout: float = DEFAULT_DROPOUT,
        hidden_size: int = 256,
        tasks: tuple[str, ...] = ("Cm", "Fg", "Bar", "Blr", "Adc"),
        cm_vocab: int = CONTEXT_VOCAB_SIZE,
        fg_size: int = FINGERPRINT_SIZE,
        adc_vocab: int = DEFAULT_ATOMIC_DISTANCE_BINS,
    ) -> None:
        super().__init__()
        allowed = {"Cm", "Fg", "Bar", "Blr", "Adc"}
        if not tasks or any(task not in allowed for task in tasks):
            raise ValueError("tasks must be a non-empty subset of Cm, Fg, Bar, Blr, Adc")
        self.pretrain_tasks = tuple(dict.fromkeys(tasks))
        self.encoder = GeoGNNEncoder(
            embed_dim=embed_dim,
            layer_num=layer_num,
            dropout_rate=dropout,
        )
        self.cm_vocab = int(cm_vocab)
        self.adc_vocab = int(adc_vocab)
        if "Cm" in self.pretrain_tasks:
            self.Cm_linear = nn.Linear(embed_dim, self.cm_vocab + 3)
        if "Fg" in self.pretrain_tasks:
            self.Fg_linear = nn.Linear(embed_dim, fg_size)
        if "Bar" in self.pretrain_tasks:
            self.Bar_mlp = PredictionMLP(embed_dim * 3, hidden_size, 1, dropout)
        if "Blr" in self.pretrain_tasks:
            self.Blr_mlp = PredictionMLP(embed_dim * 2, hidden_size, 1, dropout)
        if "Adc" in self.pretrain_tasks:
            self.Adc_mlp = PredictionMLP(embed_dim * 2, hidden_size, self.adc_vocab + 3, dropout)

    def forward(
        self,
        batch: Batch,
        targets: Mapping[str, Tensor],
        *,
        masked_batch: Batch | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, Tensor]:
        view = mask_chemrl_gem_batch(batch, generator=generator) if masked_batch is None else MaskedGEMView(masked_batch, torch.empty(0, dtype=torch.long, device=batch.batch.device), torch.empty(0, dtype=torch.long, device=batch.batch.device))
        clean_node, _, clean_graph = self._encode(batch)
        masked_node, _, masked_graph = self._encode(view.batch)
        losses: dict[str, Tensor] = {}
        if "Cm" in self.pretrain_tasks:
            node_i, context = _target_pair(targets, "Cm_node_i", "Cm_context_id", batch.batch.device)
            losses["Cm_loss"] = nn.functional.cross_entropy(self.Cm_linear(clean_node[node_i]), context)
            losses["Cm_loss"] = losses["Cm_loss"] + nn.functional.cross_entropy(self.Cm_linear(masked_node[node_i]), context)
        if "Fg" in self.pretrain_tasks:
            labels = _target(targets, "Fg_label", batch.batch.device)
            losses["Fg_loss"] = nn.functional.binary_cross_entropy_with_logits(self.Fg_linear(clean_graph), labels)
            losses["Fg_loss"] = losses["Fg_loss"] + nn.functional.binary_cross_entropy_with_logits(self.Fg_linear(masked_graph), labels)
        if "Bar" in self.pretrain_tasks:
            losses["Bar_loss"] = self._bar_loss(clean_node, masked_node, targets, batch.batch.device)
        if "Blr" in self.pretrain_tasks:
            losses["Blr_loss"] = self._blr_loss(clean_node, masked_node, targets, batch.batch.device)
        if "Adc" in self.pretrain_tasks:
            losses["Adc_loss"] = self._adc_loss(clean_node, masked_node, targets, batch.batch.device)
        if not losses:
            raise ValueError("no active ChemRL-GEM pretraining task")
        losses["loss"] = sum(losses.values())
        return losses

    def _encode(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor]:
        return self.encoder(
            batch.chemrl_gem_atom_attr,
            batch.chemrl_gem_edge_index,
            batch.chemrl_gem_bond_attr,
            batch.chemrl_gem_bond_length,
            batch.chemrl_gem_angle_edge_index,
            batch.chemrl_gem_bond_angle,
            batch.batch,
        )

    def _bar_loss(self, clean: Tensor, masked: Tensor, targets: Mapping[str, Tensor], device: torch.device) -> Tensor:
        i, j, k = (_target(targets, name, device) for name in ("Ba_node_i", "Ba_node_j", "Ba_node_k"))
        label = _target(targets, "Ba_bond_angle", device).reshape(-1, 1) / torch.pi
        inputs = torch.cat((clean[i], clean[j], clean[k]), dim=1)
        masked_inputs = torch.cat((masked[i], masked[j], masked[k]), dim=1)
        return nn.functional.smooth_l1_loss(self.Bar_mlp(inputs), label) + nn.functional.smooth_l1_loss(self.Bar_mlp(masked_inputs), label)

    def _blr_loss(self, clean: Tensor, masked: Tensor, targets: Mapping[str, Tensor], device: torch.device) -> Tensor:
        i, j = (_target(targets, name, device) for name in ("Bl_node_i", "Bl_node_j"))
        label = _target(targets, "Bl_bond_length", device).reshape(-1, 1)
        inputs = torch.cat((clean[i], clean[j]), dim=1)
        masked_inputs = torch.cat((masked[i], masked[j]), dim=1)
        return nn.functional.smooth_l1_loss(self.Blr_mlp(inputs), label) + nn.functional.smooth_l1_loss(self.Blr_mlp(masked_inputs), label)

    def _adc_loss(self, clean: Tensor, masked: Tensor, targets: Mapping[str, Tensor], device: torch.device) -> Tensor:
        i, j = (_target(targets, name, device) for name in ("Ad_node_i", "Ad_node_j"))
        distance = _target(targets, "Ad_atom_dist", device).reshape(-1).clamp(0.0, ATOMIC_DISTANCE_CUTOFF)
        labels = (distance / ATOMIC_DISTANCE_CUTOFF * self.adc_vocab).to(torch.long)
        inputs = torch.cat((clean[i], clean[j]), dim=1)
        masked_inputs = torch.cat((masked[i], masked[j]), dim=1)
        return nn.functional.cross_entropy(self.Adc_mlp(inputs), labels) + nn.functional.cross_entropy(self.Adc_mlp(masked_inputs), labels)


def _target(targets: Mapping[str, Tensor], name: str, device: torch.device) -> Tensor:
    value = targets.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"pretraining targets must provide {name}")
    return value.to(device)


def _target_pair(targets: Mapping[str, Tensor], first: str, second: str, device: torch.device) -> tuple[Tensor, Tensor]:
    return _target(targets, first, device).reshape(-1).long(), _target(targets, second, device).reshape(-1).long()


__all__ = [
    "ChemRLGEMPretrainer",
    "MaskedGEMView",
    "build_geometry_pretraining_targets",
    "mask_chemrl_gem_batch",
]


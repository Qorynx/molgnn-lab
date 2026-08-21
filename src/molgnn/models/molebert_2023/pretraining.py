"""MAM and TMCL self-supervised objectives for Mole-BERT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from .layers import MASK_ATOM, MASK_BOND
from .model import MoleBERTEncoder
from .tokenizer import MoleBERTTokenizer


@dataclass(frozen=True)
class MaskedView:
    batch: Batch
    masked_nodes: Tensor
    masked_edges: Tensor


def mask_batch(batch: Batch, rate: float, *, mask_edges: bool = False, generator: torch.Generator | None = None) -> MaskedView:
    """Clone a batch and mask a deterministic random subset per graph."""

    if not 0 < float(rate) <= 1:
        raise ValueError("mask rate must be in (0, 1]")
    masked = batch.clone()
    atom_attr = masked.molebert_atom_attr.clone()
    bond_attr = masked.molebert_bond_attr.clone()
    graph_ids = batch.batch
    masked_nodes: list[int] = []
    masked_edges: list[int] = []
    for graph_id in range(int(graph_ids.max().item()) + 1):
        nodes = torch.nonzero(graph_ids == graph_id, as_tuple=False).flatten()
        count = max(1, min(nodes.numel(), int(nodes.numel() * rate)))
        order = torch.randperm(nodes.numel(), generator=generator, device=nodes.device)[:count]
        selected_nodes = nodes[order]
        masked_nodes.extend(selected_nodes.tolist())
        atom_attr[selected_nodes, 0] = MASK_ATOM
        atom_attr[selected_nodes, 1] = 0
        if mask_edges and bond_attr.shape[0]:
            edge_mask = (batch.edge_index[0].unsqueeze(1) == selected_nodes).any(dim=1) | (batch.edge_index[1].unsqueeze(1) == selected_nodes).any(dim=1)
            selected_edges = torch.nonzero(edge_mask, as_tuple=False).flatten()
            masked_edges.extend(selected_edges.tolist())
            bond_attr[selected_edges, 0] = MASK_BOND
            bond_attr[selected_edges, 1] = 0
    masked.molebert_atom_attr = atom_attr
    masked.molebert_bond_attr = bond_attr
    return MaskedView(
        masked,
        torch.tensor(masked_nodes, dtype=torch.long, device=atom_attr.device),
        torch.tensor(masked_edges, dtype=torch.long, device=bond_attr.device),
    )


def contrastive_loss(first: Tensor, second: Tensor, temperature: float = 0.1) -> Tensor:
    first = nn.functional.normalize(first, dim=-1)
    second = nn.functional.normalize(second, dim=-1)
    logits = first @ second.t() / temperature
    labels = torch.arange(first.shape[0], dtype=torch.long, device=first.device)
    return nn.functional.cross_entropy(logits, labels)


def cosine_triplet_hinge(clean: Tensor, light: Tensor, heavy: Tensor) -> Tensor:
    clean = nn.functional.normalize(clean, dim=-1)
    light = nn.functional.normalize(light, dim=-1)
    heavy = nn.functional.normalize(heavy, dim=-1)
    return torch.relu((clean * heavy).sum(dim=-1) - (clean * light).sum(dim=-1)).mean()


class MoleBERTPretrainer(nn.Module):
    """Joint MAM/TMCL pretrainer; tokenizer parameters remain frozen."""

    def __init__(self, num_layers: int = 5, hidden_dim: int = 300, num_tokens: int = 512, mu: float = 0.1, mask_edge: bool = False) -> None:
        super().__init__()
        self.encoder = MoleBERTEncoder(num_layers=num_layers, hidden_dim=hidden_dim)
        self.projection_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.atom_head_light = nn.Linear(hidden_dim, num_tokens)
        self.atom_head_heavy = nn.Linear(hidden_dim, num_tokens)
        self.mask_edge = bool(mask_edge)
        self.mu = float(mu)

    def forward(self, batch: Batch, tokenizer: MoleBERTTokenizer, *, mask_rate_light: float = 0.15, mask_rate_heavy: float = 0.30, generator: torch.Generator | None = None) -> dict[str, Tensor]:
        with torch.no_grad():
            tokenizer.eval()
            labels = tokenizer.tokenize(batch)
        light = mask_batch(batch, mask_rate_light, mask_edges=self.mask_edge, generator=generator)
        heavy = mask_batch(batch, mask_rate_heavy, mask_edges=self.mask_edge, generator=generator)
        node_clean = self.encoder(batch.molebert_atom_attr, batch.edge_index, batch.molebert_bond_attr)
        node_light = self.encoder(light.batch.molebert_atom_attr, light.batch.edge_index, light.batch.molebert_bond_attr)
        node_heavy = self.encoder(heavy.batch.molebert_atom_attr, heavy.batch.edge_index, heavy.batch.molebert_bond_attr)
        graph_clean = global_mean_pool(node_clean, batch.batch)
        graph_light = self.projection_head(global_mean_pool(node_light, batch.batch))
        graph_heavy = self.projection_head(global_mean_pool(node_heavy, batch.batch))
        loss_mam = nn.functional.cross_entropy(self.atom_head_light(node_light[light.masked_nodes]), labels[light.masked_nodes])
        loss_mam = loss_mam + nn.functional.cross_entropy(self.atom_head_heavy(node_heavy[heavy.masked_nodes]), labels[heavy.masked_nodes])
        loss_con = contrastive_loss(graph_light, graph_heavy)
        loss_tri = cosine_triplet_hinge(self.projection_head(graph_clean.detach()), graph_light, graph_heavy)
        loss = loss_mam + loss_con + self.mu * loss_tri
        return {"loss": loss, "loss_mam": loss_mam, "loss_con": loss_con, "loss_tri": loss_tri}


__all__ = [
    "MaskedView",
    "MoleBERTPretrainer",
    "contrastive_loss",
    "cosine_triplet_hinge",
    "mask_batch",
]

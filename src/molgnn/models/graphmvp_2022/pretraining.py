"""GraphMVP Eq. 7 pretraining primitives.

This module is intentionally model-owned.  It does not alter the supervised
runner; callers can use ``GraphMVPPretrainer.compute_loss`` in a small custom
loop or a future dedicated pretraining entry point.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import subgraph

from ...data import MolecularData
from .model import GraphMVPEncoder, GraphMVPSchNetEncoder


@dataclass(frozen=True)
class GraphMVPPretrainingLoss:
    total: Tensor
    contrastive: Tensor
    generative: Tensor
    contrastive_accuracy: Tensor


class VariationalRepresentationReconstruction(nn.Module):
    """One directional VAE-like representation reconstruction head."""

    def __init__(self, emb_dim: int, beta: float = 1.0, detach_target: bool = True) -> None:
        super().__init__()
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.beta = float(beta)
        self.detach_target = bool(detach_target)
        self.fc_mu = nn.Linear(emb_dim, emb_dim)
        self.fc_var = nn.Linear(emb_dim, emb_dim)
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        if self.detach_target:
            target = target.detach()
        mu = self.fc_mu(source)
        log_var = self.fc_var(source).clamp(-30.0, 20.0)
        std = torch.exp(0.5 * log_var)
        latent = mu + torch.randn_like(std) * std
        prediction = self.decoder(latent)
        reconstruction = F.mse_loss(prediction, target)
        kl = torch.mean(
            -0.5 * torch.sum(1.0 + log_var - mu.square() - log_var.exp(), dim=1)
        )
        return reconstruction + self.beta * kl


def symmetric_infonce(x: Tensor, y: Tensor, *, temperature: float = 0.1, normalize: bool = True) -> tuple[Tensor, Tensor]:
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("contrastive views must have the same shape [B, F]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if normalize:
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    logits = x @ y.t() / temperature
    labels = torch.arange(x.shape[0], device=x.device)
    loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    accuracy = 0.5 * (
        (logits.argmax(dim=1) == labels).float().mean()
        + (logits.t().argmax(dim=1) == labels).float().mean()
    )
    return loss, accuracy


def symmetric_ebm_nce(
    x: Tensor,
    y: Tensor,
    *,
    temperature: float = 0.1,
    normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Symmetric EBM-NCE with in-batch negatives and no false self-negative."""

    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("contrastive views must have the same shape [B, F]")
    if normalize:
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
    scores = x @ y.t() / temperature
    diagonal = scores.diagonal()
    if scores.shape[0] == 1:
        zero = diagonal.sum() * 0.0
        return zero, torch.ones((), device=x.device)
    mask = ~torch.eye(scores.shape[0], dtype=torch.bool, device=x.device)
    negative = scores.masked_select(mask)
    positive_loss = F.binary_cross_entropy_with_logits(
        diagonal, torch.ones_like(diagonal)
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative, torch.zeros_like(negative)
    )
    loss = positive_loss + negative_loss
    accuracy = 0.5 * (
        (diagonal > 0).float().mean() + (negative < 0).float().mean()
    )
    return loss, accuracy


class GraphMVPPretrainer(nn.Module):
    """Joint 2-D GIN / 3-D SchNet pretrainer for GraphMVP Eq. 7."""

    def __init__(
        self,
        *,
        feature_profile: str = "simple",
        hidden_dim: int = 300,
        num_layers: int = 5,
        dropout: float = 0.0,
        alpha_1: float = 1.0,
        alpha_2: float = 1.0,
        contrastive: str = "ebm_nce",
        temperature: float = 0.1,
        normalize: bool = True,
        beta: float = 1.0,
        cutoff: float = 10.0,
    ) -> None:
        super().__init__()
        if contrastive not in {"ebm_nce", "infonce"}:
            raise ValueError("contrastive must be ebm_nce or infonce")
        self.feature_profile = feature_profile
        self.encoder_2d = GraphMVPEncoder(
            feature_profile=feature_profile,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.encoder_3d = GraphMVPSchNetEncoder(
            hidden_dim=hidden_dim,
            num_filters=128,
            num_interactions=6,
            num_gaussians=51,
            cutoff=cutoff,
            readout="mean",
        )
        self.reconstruct_2d_to_3d = VariationalRepresentationReconstruction(hidden_dim, beta)
        self.reconstruct_3d_to_2d = VariationalRepresentationReconstruction(hidden_dim, beta)
        self.alpha_1 = float(alpha_1)
        self.alpha_2 = float(alpha_2)
        self.contrastive = contrastive
        self.temperature = float(temperature)
        self.normalize = bool(normalize)

    def representations(self, batch: Batch) -> tuple[Tensor, Tensor]:
        graph_batch = getattr(batch, "batch", None)
        if not isinstance(graph_batch, Tensor):
            raise TypeError("pretraining batch must contain batch assignments")
        if self.feature_profile == "simple":
            node_2d = self.encoder_2d(
                batch.graphmvp_simple_atom_attr,
                batch.edge_index,
                batch.graphmvp_simple_bond_attr,
            )
        else:
            node_2d = self.encoder_2d(
                batch.graphmvp_ogb_atom_attr,
                batch.edge_index,
                batch.graphmvp_ogb_bond_attr,
            )
        h2d = global_mean_pool(node_2d, graph_batch)
        atomic_number = getattr(batch, "atomic_number", None)
        pos = getattr(batch, "pos", getattr(batch, "positions", None))
        if not isinstance(atomic_number, Tensor) or not isinstance(pos, Tensor):
            raise TypeError("GraphMVP pretraining requires atomic_number and pos")
        # Project geometry's 1-based nuclear charges to the source's 0-based
        # embedding indices.  Accept already-indexed source batches too.
        z = atomic_number.to(torch.long)
        if z.numel() and int(z.min()) >= 1:
            z = z - 1
        h3d = self.encoder_3d(z, pos.to(torch.float32), graph_batch)
        return h2d, h3d

    def compute_loss(self, batch: Batch) -> GraphMVPPretrainingLoss:
        h2d, h3d = self.representations(batch)
        if self.contrastive == "infonce":
            contrastive, accuracy = symmetric_infonce(
                h2d, h3d, temperature=self.temperature, normalize=self.normalize
            )
        else:
            contrastive, accuracy = symmetric_ebm_nce(
                h2d, h3d, temperature=self.temperature, normalize=self.normalize
            )
        generative = 0.5 * (
            self.reconstruct_2d_to_3d(h2d, h3d)
            + self.reconstruct_3d_to_2d(h3d, h2d)
        )
        total = self.alpha_1 * contrastive + self.alpha_2 * generative
        return GraphMVPPretrainingLoss(total, contrastive, generative, accuracy)

    def forward(self, batch: Batch) -> Tensor:
        return self.compute_loss(batch).total


def paired_connected_subgraph(
    data: MolecularData,
    *,
    mask_ratio: float = 0.15,
    generator: torch.Generator | None = None,
) -> tuple[MolecularData, MolecularData]:
    """Create paired topology/geometry views with one shared retained subset."""

    if not 0 <= mask_ratio < 1:
        raise ValueError("mask_ratio must be in [0, 1)")
    if not isinstance(data, MolecularData):
        raise TypeError("paired_connected_subgraph requires MolecularData")
    node_count = int(data.x.shape[0])
    if node_count < 1:
        raise ValueError("cannot mask an empty molecular graph")
    keep_count = max(1, min(node_count, round(node_count * (1.0 - mask_ratio))))
    edge_index = data.edge_index
    adjacency: list[list[int]] = [[] for _ in range(node_count)]
    for source, target in edge_index.t().tolist():
        adjacency[source].append(target)
    root = int(torch.randint(node_count, (), generator=generator).item())
    selected = [root]
    selected_set = {root}
    frontier = list(adjacency[root])
    while len(selected) < keep_count:
        candidates = [value for value in frontier if value not in selected_set]
        if not candidates:
            candidates = [value for value in range(node_count) if value not in selected_set]
        choice = candidates[int(torch.randint(len(candidates), (), generator=generator).item())]
        selected.append(choice)
        selected_set.add(choice)
        frontier.extend(adjacency[choice])
    subset = torch.tensor(sorted(selected), dtype=torch.long, device=edge_index.device)
    new_edge, _, edge_mask = subgraph(
        subset, edge_index, relabel_nodes=True, num_nodes=node_count, return_edge_mask=True
    )
    views = []
    for _ in range(2):
        view = data.clone()
        view.x = data.x[subset]
        view.edge_index = new_edge
        view.edge_attr = data.edge_attr[edge_mask]
        for field in ("graphmvp_simple_atom_attr", "graphmvp_ogb_atom_attr", "atomic_number", "pos"):
            value = getattr(data, field, None)
            if isinstance(value, Tensor) and value.shape[0] == node_count:
                setattr(view, field, value[subset])
        for field in ("graphmvp_simple_bond_attr", "graphmvp_ogb_bond_attr"):
            value = getattr(data, field, None)
            if isinstance(value, Tensor) and value.shape[0] == edge_index.shape[1]:
                setattr(view, field, value[edge_mask])
        views.append(view)
    return views[0], views[1]


__all__ = [
    "GraphMVPPretrainer",
    "GraphMVPPretrainingLoss",
    "VariationalRepresentationReconstruction",
    "paired_connected_subgraph",
    "symmetric_ebm_nce",
    "symmetric_infonce",
]

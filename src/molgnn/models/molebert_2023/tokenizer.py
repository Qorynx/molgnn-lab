"""Group VQ-VAE tokenizer used to create Mole-BERT atom targets."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from .layers import NUM_CHIRALITY_TAGS, MoleBERTGINConv
from .model import MoleBERTEncoder

DEFAULT_GROUP_BOUNDARIES = ((0, 128), (128, 256), (256, 384), (384, 512))


class GroupVectorQuantizer(nn.Module):
    """Nearest-code quantizer with disjoint element-group codebook slices."""

    def __init__(
        self,
        embedding_dim: int,
        num_tokens: int = 512,
        commitment_cost: float = 0.25,
        group_boundaries: Sequence[tuple[int, int]] = DEFAULT_GROUP_BOUNDARIES,
    ) -> None:
        super().__init__()
        if num_tokens < 1 or embedding_dim < 1:
            raise ValueError("embedding_dim and num_tokens must be positive")
        boundaries = tuple((int(start), int(end)) for start, end in group_boundaries)
        if not boundaries or boundaries[0][0] != 0 or boundaries[-1][1] != num_tokens:
            raise ValueError("group boundaries must cover the whole codebook")
        if any(start != previous_end or start < 0 or end <= start for (start, end), (_, previous_end) in zip(boundaries[1:], boundaries[:-1], strict=False)):
            raise ValueError("group boundaries must be contiguous")
        if not 0 <= float(commitment_cost):
            raise ValueError("commitment_cost must be non-negative")
        self.embedding_dim = embedding_dim
        self.num_tokens = num_tokens
        self.commitment_cost = float(commitment_cost)
        self.embeddings = nn.Embedding(num_tokens, embedding_dim)
        self.group_boundaries = boundaries

    def _group_for_atom(self, atomic_index: Tensor) -> Tensor:
        # Source-compatible indices: C=5, N=6, O=7; rare elements share the
        # fourth group.
        return torch.where(
            atomic_index == 5,
            torch.zeros_like(atomic_index),
            torch.where(
                atomic_index == 6,
                torch.ones_like(atomic_index),
                torch.where(atomic_index == 7, torch.full_like(atomic_index, 2), torch.full_like(atomic_index, 3)),
            ),
        )

    def code_indices(self, atom_attr: Tensor, embeddings: Tensor) -> Tensor:
        if atom_attr.ndim != 2 or atom_attr.shape[1] != 2 or embeddings.ndim != 2:
            raise ValueError("atom_attr must be [N,2] and embeddings must be [N,H]")
        if embeddings.shape[0] != atom_attr.shape[0] or embeddings.shape[1] != self.embedding_dim:
            raise ValueError("embedding shape does not match the quantizer")
        indices = torch.empty(atom_attr.shape[0], dtype=torch.long, device=embeddings.device)
        groups = self._group_for_atom(atom_attr[:, 0])
        for group_id, (start, end) in enumerate(self.group_boundaries):
            selected = groups == group_id
            if not selected.any():
                continue
            codebook = self.embeddings.weight[start:end]
            values = embeddings[selected]
            distances = (values.square().sum(dim=1, keepdim=True) + codebook.square().sum(dim=1) - 2 * values @ codebook.t())
            indices[selected] = distances.argmin(dim=1) + start
        return indices

    def forward(self, atom_attr: Tensor, embeddings: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        indices = self.code_indices(atom_attr, embeddings)
        quantized = self.embeddings(indices)
        codebook_loss = torch.nn.functional.mse_loss(quantized, embeddings.detach())
        commitment_loss = torch.nn.functional.mse_loss(embeddings, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        straight_through = embeddings + (quantized - embeddings).detach()
        return straight_through, indices, loss


class TokenizerDecoder(nn.Module):
    """Small GIN decoder reconstructing atom, chirality, and bond types."""

    def __init__(self, hidden_dim: int, bond_dim: int = 2) -> None:
        super().__init__()
        self.conv = MoleBERTGINConv(hidden_dim, hidden_dim)
        self.atom_head = nn.Linear(hidden_dim, 119)
        self.chiral_head = nn.Linear(hidden_dim, NUM_CHIRALITY_TAGS)
        self.bond_head = nn.Linear(hidden_dim, 4)
        self.bond_dim = bond_dim

    def forward(self, node_embeddings: Tensor, edge_index: Tensor, bond_attr: Tensor) -> dict[str, Tensor]:
        node_embeddings = self.conv(torch.relu(node_embeddings), edge_index, bond_attr)
        edge_embeddings = node_embeddings[edge_index[0]] + node_embeddings[edge_index[1]]
        return {
            "atom_logits": self.atom_head(node_embeddings),
            "chiral_logits": self.chiral_head(node_embeddings),
            "bond_logits": self.bond_head(edge_embeddings),
        }


class MoleBERTTokenizer(nn.Module):
    """Trainable context-aware atom tokenizer and frozen token lookup."""

    def __init__(self, num_layers: int = 5, hidden_dim: int = 300, num_tokens: int = 512) -> None:
        super().__init__()
        self.encoder = MoleBERTEncoder(num_layers=num_layers, hidden_dim=hidden_dim)
        self.quantizer = GroupVectorQuantizer(hidden_dim, num_tokens=num_tokens)
        self.decoder = TokenizerDecoder(hidden_dim)

    def forward(self, batch) -> dict[str, Tensor]:
        node_embeddings = self.encoder(batch.molebert_atom_attr, batch.edge_index, batch.molebert_bond_attr)
        quantized, indices, vq_loss = self.quantizer(batch.molebert_atom_attr, node_embeddings)
        decoded = self.decoder(quantized, batch.edge_index, batch.molebert_bond_attr)
        decoded["token_indices"] = indices
        decoded["vq_loss"] = vq_loss
        return decoded

    @torch.no_grad()
    def tokenize(self, batch) -> Tensor:
        self.eval()
        embeddings = self.encoder(batch.molebert_atom_attr, batch.edge_index, batch.molebert_bond_attr)
        return self.quantizer.code_indices(batch.molebert_atom_attr, embeddings)


def _scaled_cosine_loss(logits: Tensor, targets: Tensor, classes: int, gamma: float = 1.0) -> Tensor:
    probabilities = nn.functional.softmax(logits, dim=-1)
    one_hot = nn.functional.one_hot(targets, num_classes=classes).to(probabilities.dtype)
    cosine = nn.functional.cosine_similarity(probabilities, one_hot, dim=-1)
    return (1.0 - cosine).pow(gamma).mean()


def tokenizer_reconstruction_loss(output: dict[str, Tensor], batch, *, mode: str = "sce") -> Tensor:
    """Paper scaled-cosine reconstruction plus VQ loss.

    ``mode='cross_entropy'`` is retained only for source-comparison runs; the
    default follows Eq. (4) of the paper.
    """

    atom_attr = batch.molebert_atom_attr
    bond_attr = batch.molebert_bond_attr
    if mode == "sce":
        loss = _scaled_cosine_loss(output["atom_logits"], atom_attr[:, 0], 119)
        loss = loss + _scaled_cosine_loss(output["chiral_logits"], atom_attr[:, 1], NUM_CHIRALITY_TAGS)
    elif mode == "cross_entropy":
        loss = nn.functional.cross_entropy(output["atom_logits"], atom_attr[:, 0])
        loss = loss + nn.functional.cross_entropy(output["chiral_logits"], atom_attr[:, 1])
    else:
        raise ValueError("tokenizer reconstruction mode must be 'sce' or 'cross_entropy'")
    if bond_attr.shape[0]:
        loss = loss + (
            _scaled_cosine_loss(output["bond_logits"], bond_attr[:, 0], 4)
            if mode == "sce"
            else nn.functional.cross_entropy(output["bond_logits"], bond_attr[:, 0])
        )
    return loss + output["vq_loss"]


__all__ = [
    "DEFAULT_GROUP_BOUNDARIES",
    "GroupVectorQuantizer",
    "MoleBERTTokenizer",
    "tokenizer_reconstruction_loss",
]

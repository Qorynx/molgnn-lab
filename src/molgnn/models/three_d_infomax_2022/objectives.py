"""3D Infomax contrastive objectives.

Provenance: ``OFFICIAL CODE`` ``commons/losses.py``
``NTXentMultiplePositives`` at revision
``5cd32629c690e119bcae8726acedefdb0aa037fc`` and the paper's Equation (2).
The decisive property: the positive pair is EXCLUDED from the denominator,

    loss_i = -log( sum_pos exp(sim/tau) / sum_{negatives} exp(sim/tau) )

where negatives are all conformers of *other* molecules in the batch. Two
implementations are provided:

- :class:`NTXentMultiplePositives` is a faithful clone of the source tensor
  math for fixed conformer counts (used as parity reference);
- :func:`multi_positive_infomax_loss` reformulates it with owner masks and
  ``logsumexp`` so variable positive counts stay numerically stable while
  producing identical values whenever every molecule owns the same number of
  conformers.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class NTXentMultiplePositives(nn.Module):
    """Faithful port of the official loss (fixed C per molecule)."""

    def __init__(self, tau: float = 0.1, norm: bool = True) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError("tau must be positive")
        self.tau = tau
        self.norm = norm

    def forward(self, z1: Tensor, z2: Tensor) -> Tensor:
        batch_size, metric_dim = z1.size()
        if z2.shape[0] % batch_size != 0:
            raise ValueError("z2 rows must be batch_size * num_conformers")
        if batch_size < 2:
            raise ValueError("the contrastive batch requires at least two molecules")
        z2 = z2.view(batch_size, -1, metric_dim)  # [B, C, D]

        sim_matrix = torch.einsum("ik,juk->iju", z1, z2)
        if self.norm:
            z1_abs = z1.norm(dim=1)
            z2_abs = z2.norm(dim=2)
            sim_matrix = sim_matrix / torch.einsum("i,ju->iju", z1_abs, z2_abs)

        sim_matrix = torch.exp(sim_matrix / self.tau)
        sim_matrix = sim_matrix.sum(dim=2)  # [B, B]
        pos_sim = torch.diagonal(sim_matrix)
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        return -torch.log(loss).mean()


def multi_positive_infomax_loss(
    z2d: Tensor,
    z3d: Tensor,
    conformer_owner: Tensor,
    *,
    tau: float = 0.1,
) -> Tensor:
    """Owner-masked single/multi-positive objective with logsumexp stability.

    ``z3d`` holds one row per conformer; ``conformer_owner[t]`` maps row ``t``
    to its molecule. Positives of molecule ``i`` are exactly its owned rows,
    negatives are all remaining rows, matching the official denominator.
    With a constant conformer count this equals
    :class:`NTXentMultiplePositives`.
    """

    num_molecules = z2d.shape[0]
    if num_molecules < 2:
        raise ValueError("the contrastive batch requires at least two molecules")
    if z3d.shape[0] != conformer_owner.shape[0]:
        raise ValueError("z3d rows and conformer_owner entries must align")
    if int(conformer_owner.min().item()) < 0 or int(conformer_owner.max().item()) >= num_molecules:
        raise ValueError("conformer_owner references an unknown molecule")

    similarity = F.cosine_similarity(
        z2d.unsqueeze(1), z3d.unsqueeze(0), dim=-1
    ) / tau  # [B, T]
    owner = conformer_owner.to(z2d.device)
    molecule_index = torch.arange(num_molecules, device=z2d.device).unsqueeze(1)
    positive_mask = owner.unsqueeze(0) == molecule_index
    negative_mask = ~positive_mask

    if not bool(positive_mask.any()) or not bool(negative_mask.any()):
        raise ValueError("each molecule needs at least one positive and one negative")

    max_value = similarity.detach().max()
    exponentiated = torch.exp(similarity - max_value)
    positive_sum = (exponentiated * positive_mask).sum(dim=1)
    negative_sum = (exponentiated * negative_mask).sum(dim=1)
    # logsumexp form: log(sum_neg) - log(sum_pos), shifted back consistently.
    loss = (torch.log(negative_sum) - torch.log(positive_sum)).mean()
    return loss


__all__ = ["NTXentMultiplePositives", "multi_positive_infomax_loss"]

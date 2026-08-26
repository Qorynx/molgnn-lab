"""HiMol multi-level self-supervised pretraining (MSP)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch

from .layers import HiMolEncoder


@dataclass(frozen=True)
class HiMolPretrainingLoss:
    total: Tensor
    bond_link: Tensor
    atom_type: Tensor
    bond_type: Tensor
    atom_count: Tensor
    bond_count: Tensor
    weights: Tensor


class HiMolPretrainer(nn.Module):
    """Five-task MSP objective with an actually learnable softmax weight."""

    required_batch_fields = (
        "himol_node_attr",
        "himol_edge_index",
        "himol_edge_attr",
        "himol_batch",
        "himol_atom_node_index",
        "himol_graph_node_index",
        "himol_atom_target",
        "himol_bond_index",
        "himol_bond_target",
        "himol_num_atoms",
        "himol_num_bonds",
    )

    def __init__(
        self,
        *,
        num_layer: int = 5,
        emb_dim: int = 512,
        JK: str = "last",
        drop_ratio: float = 0.5,
        decoder_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not 0 <= float(decoder_dropout) < 1:
            raise ValueError("decoder_dropout must be in [0, 1)")
        self.gnn = HiMolEncoder(num_layer, emb_dim, JK, drop_ratio)
        width = self.gnn.output_dim
        self.bond_if_proj = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width)
        )
        self.bond_if_s = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 1)
        )
        self.feat_drop = nn.Dropout(decoder_dropout)
        self.bond_type_s = nn.Sequential(
            nn.Linear(2 * width, width), nn.ReLU(), nn.Linear(width, 4)
        )
        self.atom_type_s = nn.Sequential(
            nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 118)
        )
        count_width = max(1, width // 4)
        self.atom_num_s = nn.Sequential(
            nn.Linear(width, count_width), nn.Softplus(), nn.Linear(count_width, 1)
        )
        self.bond_num_s = nn.Sequential(
            nn.Linear(width, count_width), nn.Softplus(), nn.Linear(count_width, 1)
        )
        self.loss_logits = nn.Parameter(torch.zeros(5))

    def compute_loss(self, batch: Batch) -> HiMolPretrainingLoss:
        fields = _pretraining_fields(batch)
        node_representation = self.gnn(
            fields["node_attr"], fields["edge_index"], fields["edge_attr"]
        )
        atom_node_index = fields["atom_node_index"]
        graph_node_index = fields["graph_node_index"]
        atom_representation = node_representation[atom_node_index]
        graph_representation = node_representation[graph_node_index]

        atom_type = F.cross_entropy(
            self.atom_type_s(atom_representation), fields["atom_target"]
        )
        bond_index = fields["bond_index"]
        if bond_index.shape[1]:
            endpoints = torch.cat(
                (
                    atom_representation[bond_index[0]],
                    atom_representation[bond_index[1]],
                ),
                dim=1,
            )
            bond_type = F.cross_entropy(
                self.bond_type_s(endpoints), fields["bond_target"]
            )
        else:
            bond_type = atom_representation.sum() * 0.0

        atom_batch = fields["hierarchy_batch"][atom_node_index]
        bond_link = self._bond_link_loss(atom_representation, atom_batch, bond_index)
        atom_count = F.smooth_l1_loss(
            self.atom_num_s(graph_representation).squeeze(-1),
            fields["num_atoms"].to(graph_representation.dtype),
        )
        bond_count = F.smooth_l1_loss(
            self.bond_num_s(graph_representation).squeeze(-1),
            fields["num_bonds"].to(graph_representation.dtype),
        )
        losses = torch.stack((bond_link, atom_type, bond_type, atom_count, bond_count))
        weights = torch.softmax(self.loss_logits, dim=0)
        total = torch.sum(weights * losses)
        return HiMolPretrainingLoss(
            total, bond_link, atom_type, bond_type, atom_count, bond_count, weights
        )

    def _bond_link_loss(
        self, atom_representation: Tensor, atom_batch: Tensor, bond_index: Tensor
    ) -> Tensor:
        projected = self.feat_drop(self.bond_if_proj(atom_representation))
        graph_losses: list[Tensor] = []
        num_graphs = int(atom_batch.max().item()) + 1
        for graph_id in range(num_graphs):
            atom_ids = torch.nonzero(atom_batch == graph_id, as_tuple=False).flatten()
            local = projected[atom_ids]
            count = int(local.shape[0])
            left = local.repeat_interleave(count, dim=0)
            right = local.repeat((count, 1))
            logits = self.bond_if_s(torch.cat((left, right), dim=1)).reshape(
                count, count
            )
            labels = logits.new_zeros((count, count))
            if bond_index.shape[1]:
                offset = int(atom_ids[0])
                mask = (bond_index[0] >= offset) & (bond_index[0] < offset + count)
                local_bonds = bond_index[:, mask] - offset
                if local_bonds.numel():
                    labels[local_bonds[0], local_bonds[1]] = 1.0
                    labels[local_bonds[1], local_bonds[0]] = 1.0
            graph_losses.append(F.binary_cross_entropy_with_logits(logits, labels))
        return torch.stack(graph_losses).mean()

    def forward(self, batch: Batch) -> Tensor:
        return self.compute_loss(batch).total

    def save_encoder(self, path: str | Path) -> Path:
        """Export only the transferable HMGNN encoder state."""

        output = Path(path).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.gnn.state_dict(), output)
        return output


def _pretraining_fields(batch: Batch) -> dict[str, Tensor]:
    names = {
        "node_attr": "himol_node_attr",
        "edge_index": "himol_edge_index",
        "edge_attr": "himol_edge_attr",
        "hierarchy_batch": "himol_batch",
        "atom_node_index": "himol_atom_node_index",
        "graph_node_index": "himol_graph_node_index",
        "atom_target": "himol_atom_target",
        "bond_index": "himol_bond_index",
        "bond_target": "himol_bond_target",
        "num_atoms": "himol_num_atoms",
        "num_bonds": "himol_num_bonds",
    }
    result = {key: getattr(batch, field, None) for key, field in names.items()}
    missing = [
        names[key] for key, value in result.items() if not isinstance(value, Tensor)
    ]
    if missing:
        raise ValueError(
            f"pretraining batch is missing HiMol fields: {', '.join(missing)}"
        )
    typed = {key: value for key, value in result.items() if isinstance(value, Tensor)}
    if typed["bond_index"].ndim != 2 or typed["bond_index"].shape[0] != 2:
        raise ValueError("himol_bond_index must have shape [2, M]")
    if typed["bond_target"].shape != (typed["bond_index"].shape[1],):
        raise ValueError("himol_bond_target must align with physical bonds")
    if typed["atom_target"].shape != (typed["atom_node_index"].shape[0],):
        raise ValueError("himol_atom_target must align with atom nodes")
    if typed["num_atoms"].shape != typed["num_bonds"].shape:
        raise ValueError("HiMol count targets must have matching graph shapes")
    if typed["graph_node_index"].shape != typed["num_atoms"].shape:
        raise ValueError("HiMol count targets must align with graph nodes")
    return typed


__all__ = ["HiMolPretrainer", "HiMolPretrainingLoss"]

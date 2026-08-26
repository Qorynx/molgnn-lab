"""Attribute Masking for Pretrain-GNNs (model-owned pretraining stage).

Replicates ``OFFICIAL CODE chem/util.py::MaskAtom`` plus the training loop of
``chem/pretrain_masking.py``:

- per molecule ``sample_size = int(num_atoms * mask_rate + 1)`` distinct atoms
  are chosen (may exceed 15% on small molecules — official behavior);
- the mask label is copied from the atom row BEFORE corruption;
- masked atoms get ``(mask_token=119, chirality=0)``;
- optional bond masking corrupts both directed edges of every undirected bond
  touching a masked atom with ``(mask_bond_type=5, 0)``; the label set keeps
  one entry per unique undirected bond, independent of how the two directions
  are ordered in ``edge_index`` (the official ``[::2]`` pairing assumed
  adjacency, this implementation pairs explicitly);
- everything operates on clones; input samples never mutate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch

from .layers import MASK_ATOM_TOKEN, MASK_BOND_TYPE, NUM_REAL_BOND_TYPES


@dataclass(frozen=True)
class AttributeMaskBatch:
    """Batched masking state consumed by the Attribute Masking objective."""

    batch_2d: Batch
    masked_atom_indices: Tensor  # [M], batch-offset atom indices
    mask_atom_label: Tensor  # [M, 2] original categorical rows
    connected_edge_indices: Tensor  # [Eu] unique undirected bonds (batch offset)
    mask_edge_label: Tensor  # [Eu, 2]

    def to(self, device: torch.device | str) -> AttributeMaskBatch:
        return AttributeMaskBatch(
            batch_2d=self.batch_2d.to(device),
            masked_atom_indices=self.masked_atom_indices.to(device),
            mask_atom_label=self.mask_atom_label.to(device),
            connected_edge_indices=self.connected_edge_indices.to(device),
            mask_edge_label=self.mask_edge_label.to(device),
        )


def _sample_mask_indices(num_atoms: int, rate: float, generator: torch.Generator | None) -> list[int]:
    sample_size = int(num_atoms * rate + 1)  # official count quirk
    sample_size = min(sample_size, num_atoms)
    perm = torch.randperm(num_atoms, generator=generator)
    return perm[:sample_size].tolist()


def _undirected_pairs(edge_index: Tensor) -> dict[tuple[int, int], list[int]]:
    pairs: dict[tuple[int, int], list[int]] = {}
    for position, (u, v) in enumerate(edge_index.t().tolist()):
        key = (u, v) if u < v else (v, u)
        pairs.setdefault(key, []).append(position)
    return pairs


def build_attribute_mask_batch(
    samples: list,
    *,
    mask_rate: float = 0.15,
    mask_edge: bool = False,
    generator: torch.Generator | None = None,
) -> AttributeMaskBatch:
    """Create the masked clone of one list of transformed samples."""

    if not samples:
        raise ValueError("attribute masking requires at least one sample")
    cloned_samples = []
    masked_atom_blocks: list[Tensor] = []
    label_blocks: list[Tensor] = []
    edge_label_blocks: list[Tensor] = []
    connected_blocks: list[Tensor] = []

    node_offset = 0
    edge_offset = 0
    any_edges = False
    for sample in samples:
        atom_attr = sample.pretrain_gnns_atom_attr
        bond_attr = sample.pretrain_gnns_bond_attr
        num_atoms = int(atom_attr.shape[0])
        num_edges = int(bond_attr.shape[0])

        masked = _sample_mask_indices(num_atoms, mask_rate, generator)
        # Labels are captured before any corruption (official contract).
        labels = atom_attr[masked].clone()
        corrupted = atom_attr.clone()
        corrupted[masked, 0] = MASK_ATOM_TOKEN
        corrupted[masked, 1] = 0
        new_sample = sample.clone()
        new_sample.pretrain_gnns_atom_attr = corrupted
        cloned_samples.append(new_sample)

        masked_atom_blocks.append(
            torch.tensor([index + node_offset for index in masked], dtype=torch.long)
        )
        label_blocks.append(labels)

        connected: list[int] = []
        edge_labels: list[Tensor] = []
        if mask_edge and num_edges > 0:
            any_edges = True
            pairs = _undirected_pairs(sample.edge_index)
            masked_set = set(masked)
            seen: set[tuple[int, int]] = set()
            for (u, v), positions in pairs.items():
                if u not in masked_set and v not in masked_set:
                    continue
                key = (min(positions), max(positions))
                if key in seen:
                    continue
                seen.add(key)
                representative = min(positions)
                edge_labels.append(bond_attr[representative].view(1, -1).clone())
                # OFFICIAL CODE predicts from one representative direction
                # while corrupting both directions below.
                connected.append(representative + edge_offset)
                corrupted_bond = new_sample.pretrain_gnns_bond_attr.clone()
                for position in positions:
                    corrupted_bond[position, 0] = MASK_BOND_TYPE
                    corrupted_bond[position, 1] = 0
                new_sample.pretrain_gnns_bond_attr = corrupted_bond
        if edge_labels:
            edge_label_blocks.append(torch.cat(edge_labels, dim=0))
            connected_blocks.append(torch.tensor(connected, dtype=torch.long))

        node_offset += num_atoms
        edge_offset += num_edges

    if not any_edges or not edge_label_blocks:
        edge_label_tensor = torch.empty((0, 2), dtype=torch.long)
        connected_tensor = torch.empty(0, dtype=torch.long)
    else:
        edge_label_tensor = torch.cat(edge_label_blocks, dim=0)
        connected_tensor = torch.cat(connected_blocks, dim=0)

    return AttributeMaskBatch(
        batch_2d=Batch.from_data_list(cloned_samples),
        masked_atom_indices=torch.cat(masked_atom_blocks, dim=0),
        mask_atom_label=torch.cat(label_blocks, dim=0),
        connected_edge_indices=connected_tensor,
        mask_edge_label=edge_label_tensor,
    )


class AttributeMaskingObjective(nn.Module):
    """Official heads and losses for Attribute Masking.

    ``input_dim`` is the encoder's node-representation width and must match
    the ``JK`` mode (``(num_layer + 1) * emb_dim`` for ``concat``, ``emb_dim``
    otherwise). The atom head predicts 119 classes; the bond head predicts the
    four real bond classes. The mask token ids (119 for atoms, 5 for bonds)
    are corruption-only inputs and are never output classes.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        self.linear_pred_atoms = nn.Linear(input_dim, MASK_ATOM_TOKEN)
        self.linear_pred_bonds = nn.Linear(input_dim, NUM_REAL_BOND_TYPES)

    def forward(self, node_rep: Tensor, batch: AttributeMaskBatch) -> dict[str, Tensor]:
        pred_node = self.linear_pred_atoms(node_rep[batch.masked_atom_indices])
        loss = torch.nn.functional.cross_entropy(
            pred_node.double(), batch.mask_atom_label[:, 0]
        )
        outputs = {"loss": loss, "pred_node": pred_node}
        if batch.connected_edge_indices.numel():
            endpoints = batch.batch_2d.edge_index[:, batch.connected_edge_indices]
            edge_rep = node_rep[endpoints[0]] + node_rep[endpoints[1]]
            pred_edge = self.linear_pred_bonds(edge_rep)
            loss = loss + torch.nn.functional.cross_entropy(
                pred_edge.double(), batch.mask_edge_label[:, 0]
            )
            outputs["loss"] = loss
            outputs["pred_edge"] = pred_edge
        return outputs


__all__ = [
    "AttributeMaskBatch",
    "AttributeMaskingObjective",
    "build_attribute_mask_batch",
]

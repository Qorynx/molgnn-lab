"""Context Prediction for Pretrain-GNNs (model-owned pretraining stage).

Replicates ``OFFICIAL CODE`` ``chem/util.py::ExtractSubstructureContextPair``,
``chem/batch.py::BatchSubstructContext``, and the ``cbow`` training loop of
``chem/pretrain_contextpred.py``:

- substructure: nodes within ``k`` hops of a sampled root;
- context ring: ``l1 < distance <= l2`` (equivalent to the source's
  symmetric-difference of the nested hop sets because the sets are nested);
- anchors: overlap between substructure and context, re-indexed in context
  node ordering; the root is re-indexed in substructure ordering;
- molecules without a valid context/overlap are skipped by the collator
  exactly like the source ("If there is no context, just skip!!"); a batch
  with fewer than two usable molecules raises instead of producing NaN.
- negatives use the official cyclic shift of pooled contexts.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch


@dataclass(frozen=True)
class ContextPairBatch:
    """Batched substructure/context pair for one ContextPred step."""

    batch_substruct: Batch
    batch_context: Batch
    center_substruct_idx: Tensor  # [G] root rank within each substructure
    overlap_context_substruct_idx: Tensor  # [sum sizes] context-offset anchors
    batch_overlapped_context: Tensor  # [sum sizes] molecule index per anchor
    overlapped_context_size: Tensor  # [G]

    def to(self, device: torch.device | str) -> ContextPairBatch:
        return ContextPairBatch(
            batch_substruct=self.batch_substruct.to(device),
            batch_context=self.batch_context.to(device),
            center_substruct_idx=self.center_substruct_idx.to(device),
            overlap_context_substruct_idx=self.overlap_context_substruct_idx.to(device),
            batch_overlapped_context=self.batch_overlapped_context.to(device),
            overlapped_context_size=self.overlapped_context_size.to(device),
        )


def _hop_distances(edge_index: Tensor, num_nodes: int, root: int, cutoff: int) -> dict[int, int]:
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for u, v in edge_index.t().tolist():
        adjacency[u].append(v)
    distances = {root: 0}
    frontier = [root]
    for depth in range(1, cutoff + 1):
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor in sorted(adjacency[node]):
                if neighbor not in distances:
                    distances[neighbor] = depth
                    next_frontier.append(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return distances


def _reindex(nodes: list[int]) -> tuple[list[int], dict[int, int]]:
    ordered = sorted(nodes)
    mapping = {old: new for new, old in enumerate(ordered)}
    return ordered, mapping


def extract_substructure_context(
    atom_attr: Tensor,
    edge_index: Tensor,
    bond_attr: Tensor,
    *,
    k: int,
    l1: int,
    l2: int,
    root_idx: int,
) -> dict[str, Tensor] | None:
    """Source-equivalent extraction for one molecule at a fixed root.

    Returns ``None`` when either the context or the anchor overlap is empty
    (the collator then skips this molecule).
    """

    num_atoms = int(atom_attr.shape[0])
    distances = _hop_distances(edge_index, num_atoms, root_idx, max(l2, k))

    substruct_nodes = [node for node, distance in distances.items() if distance <= k]
    context_nodes = [node for node, distance in distances.items() if l1 < distance <= l2]
    overlap_nodes = set(substruct_nodes).intersection(context_nodes)
    if not context_nodes or not overlap_nodes:
        return None

    ordered_substruct, substruct_map = _reindex(substruct_nodes)
    ordered_context, context_map = _reindex(context_nodes)

    def gather_graph(nodes: list[int]) -> tuple[Tensor, Tensor, Tensor]:
        keep = torch.tensor(nodes, dtype=torch.long)
        sub_x = atom_attr[keep]
        mask = torch.isin(edge_index[0], keep) & torch.isin(edge_index[1], keep)
        sub_edges = edge_index[:, mask]
        sub_attr = bond_attr[mask]
        remap = torch.full((num_atoms,), -1, dtype=torch.long)
        remap[keep] = torch.arange(len(nodes))
        sub_edges = remap[sub_edges]
        return sub_x, sub_edges, sub_attr

    x_substruct, edge_index_substruct, edge_attr_substruct = gather_graph(ordered_substruct)
    x_context, edge_index_context, edge_attr_context = gather_graph(ordered_context)

    overlap_in_context = torch.tensor(
        [context_map[node] for node in sorted(overlap_nodes)], dtype=torch.long
    )
    return {
        "x_substruct": x_substruct,
        "edge_index_substruct": edge_index_substruct,
        "edge_attr_substruct": edge_attr_substruct,
        "center_substruct_idx": torch.tensor([substruct_map[root_idx]], dtype=torch.long),
        "x_context": x_context,
        "edge_index_context": edge_index_context,
        "edge_attr_context": edge_attr_context,
        "overlap_context_substruct_idx": overlap_in_context,
    }


def build_context_pair_batch(
    samples: list,
    *,
    k: int = 5,
    csize: int = 3,
    generator: torch.Generator | None = None,
) -> ContextPairBatch:
    """Sample roots and collate every molecule with a usable context."""

    if not samples:
        raise ValueError("context prediction requires at least one sample")
    l1 = k - 1
    l2 = l1 + csize

    substruct_blocks: list = []
    context_blocks: list = []
    centers: list[Tensor] = []
    overlaps: list[Tensor] = []
    owners: list[Tensor] = []
    sizes: list[int] = []
    cumsum_substruct = 0
    cumsum_context = 0

    for sample_index, sample in enumerate(samples):
        atom_attr = sample.pretrain_gnns_atom_attr
        num_atoms = int(atom_attr.shape[0])
        root_idx = int(torch.randint(num_atoms, (1,), generator=generator).item())
        extracted = extract_substructure_context(
            atom_attr,
            sample.edge_index,
            sample.pretrain_gnns_bond_attr,
            k=k,
            l1=l1,
            l2=l2,
            root_idx=root_idx,
        )
        if extracted is None:
            continue  # official behavior: skip graphs without a context
        substruct_data = sample.clone()
        substruct_data.x = extracted["x_substruct"]
        substruct_data.edge_index = extracted["edge_index_substruct"]
        substruct_data.edge_attr = extracted["edge_attr_substruct"]
        # The encoder consumes the model-local categorical view, so the
        # subgraph clone must carry the re-indexed subgraph rows, not the
        # full-molecule ones inherited from ``sample``.
        substruct_data.pretrain_gnns_atom_attr = extracted["x_substruct"]
        substruct_data.pretrain_gnns_bond_attr = extracted["edge_attr_substruct"]
        context_data = sample.clone()
        context_data.x = extracted["x_context"]
        context_data.edge_index = extracted["edge_index_context"]
        context_data.edge_attr = extracted["edge_attr_context"]
        context_data.pretrain_gnns_atom_attr = extracted["x_context"]
        context_data.pretrain_gnns_bond_attr = extracted["edge_attr_context"]
        substruct_blocks.append(substruct_data)
        context_blocks.append(context_data)

        # Official offsets: center into substructure ordering, anchors into
        # context ordering (BatchSubstructContext.cumsum).
        centers.append(extracted["center_substruct_idx"] + cumsum_substruct)
        overlaps.append(extracted["overlap_context_substruct_idx"] + cumsum_context)
        anchor_count = int(extracted["overlap_context_substruct_idx"].shape[0])
        owners.append(
            torch.full((anchor_count,), len(sizes), dtype=torch.long)
        )
        sizes.append(anchor_count)

        cumsum_substruct += int(extracted["x_substruct"].shape[0])
        cumsum_context += int(extracted["x_context"].shape[0])

    if len(substruct_blocks) < 2:
        raise ValueError(
            "ContextPred requires at least two molecules with a valid context; "
            f"got {len(substruct_blocks)}"
        )

    return ContextPairBatch(
        batch_substruct=Batch.from_data_list(substruct_blocks),
        batch_context=Batch.from_data_list(context_blocks),
        center_substruct_idx=torch.cat(centers, dim=0),
        overlap_context_substruct_idx=torch.cat(overlaps, dim=0),
        batch_overlapped_context=torch.cat(owners, dim=0),
        overlapped_context_size=torch.tensor(sizes, dtype=torch.long),
    )


def cycle_index(num: int, shift: int) -> Tensor:
    """Official cyclic shift used to build negative contexts."""

    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr


class ContextPredObjective(nn.Module):
    """CBOW scoring with the official cyclic-shift negatives."""

    def __init__(self, *, neg_samples: int = 1, context_pooling: str = "mean") -> None:
        super().__init__()
        if neg_samples < 1:
            raise ValueError("neg_samples must be >= 1")
        self.neg_samples = neg_samples
        self.context_pooling = context_pooling

    def forward(
        self,
        substruct_rep: Tensor,
        context_pair: ContextPairBatch,
        context_rep_nodes: Tensor,
    ) -> dict[str, Tensor]:
        anchors = context_rep_nodes[context_pair.overlap_context_substruct_idx]
        counts = (
            torch.bincount(
                context_pair.batch_overlapped_context,
                minlength=substruct_rep.shape[0],
            )
            .unsqueeze(-1)
            .to(anchors.dtype)
        )
        pooled = anchors.new_zeros((substruct_rep.shape[0], anchors.shape[1]))
        pooled.index_add_(0, context_pair.batch_overlapped_context, anchors)
        if self.context_pooling == "mean":
            context_rep = pooled / counts.clamp_min(1.0)
        elif self.context_pooling == "sum":
            context_rep = pooled
        else:
            raise ValueError(f"unsupported context pooling {self.context_pooling!r}")

        negatives = torch.cat(
            [
                context_rep[
                    cycle_index(len(context_rep), shift + 1).to(context_rep.device)
                ]
                for shift in range(self.neg_samples)
            ],
            dim=0,
        )
        pred_pos = (substruct_rep * context_rep).sum(dim=1)
        pred_neg = (substruct_rep.repeat((self.neg_samples, 1)) * negatives).sum(dim=1)

        criterion = nn.BCEWithLogitsLoss()
        loss_pos = criterion(pred_pos.double(), torch.ones_like(pred_pos))
        loss_neg = criterion(pred_neg.double(), torch.zeros_like(pred_neg))
        return {
            "loss": loss_pos + self.neg_samples * loss_neg,
            "loss_pos": loss_pos,
            "loss_neg": loss_neg,
            "pred_pos": pred_pos,
            "pred_neg": pred_neg,
        }


__all__ = [
    "ContextPairBatch",
    "ContextPredObjective",
    "build_context_pair_batch",
    "cycle_index",
    "extract_substructure_context",
]

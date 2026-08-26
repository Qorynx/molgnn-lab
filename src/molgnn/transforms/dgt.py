"""DGT (Dual Graph Transformer) model-local topology transform.

Liu et al., *Enhancing molecular property prediction of transformer models
with dual graph representation*, Nature Communications 2026.

The transform derives, from the shared 2-D canonical graph, everything the
DGT model needs beyond ``x`` / ``edge_index`` / ``edge_attr``:

- the line graph (bond graph) whose nodes are the undirected chemical bonds
  and whose edges connect bonds sharing an atom;
- shortest-path-distance (SPDE) sparse indices for both the atom graph and the
  bond graph;
- random-walk structural encodings (RWSE): flattened all-pairs landing
  probabilities ``[N*N, K]`` / ``[M*M, K]`` (``P^k[i, j]``, row-major per
  graph) for both graphs, matching the official ``rw_landing_all``.

Dense attention bias tensors are deliberately *not* stored here: they contain
learnable embeddings and are built inside ``DGTEmbedder.forward()``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from ..data import MolecularData
from .base import TransformError


class DGTData(MolecularData):
    """PyG sample with DGT line-graph local indices and batching offsets."""

    def __inc__(self, key: str, value, *args, **kwargs):
        if key == "dgt_e_batch":
            return 1
        if key == "dgt_e2e_edge_index":
            return int((self.edge_index[0] < self.edge_index[1]).sum())
        if key == "dgt_e2e_node_index":
            return int(self.x.size(0))
        if key in ("dgt_e2e_spd_index", "dgt_e2e_ring_index"):
            return int((self.edge_index[0] < self.edge_index[1]).sum())
        return super().__inc__(key, value, *args, **kwargs)


def add_dgt_inputs(data: MolecularData) -> DGTData:
    """Attach DGT's line-graph, SPDE and RWSE fields to one unbatched sample."""

    sample = _sample_id(data)
    if getattr(data, "batch", None) is not None:
        raise TransformError(
            f"sample {sample} is already batched; DGT inputs must be derived "
            "before PyG batching"
        )
    x = getattr(data, "x", None)
    edge_index = getattr(data, "edge_index", None)
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[0] < 1:
        raise TransformError(f"sample {sample} requires a non-empty atom feature matrix")
    if not isinstance(edge_index, Tensor) or edge_index.shape[0] != 2:
        raise TransformError(f"sample {sample} requires a [2, E] edge_index")

    num_atoms = int(x.shape[0])
    undirected_mask = edge_index[0] < edge_index[1]
    undirected = edge_index[:, undirected_mask]  # [2, M']
    num_bonds = int(undirected.shape[1])

    # --- line-graph topology ---
    e2e_edge_index, e2e_node_index = _line_graph_topology(edge_index)

    # --- atom-graph SPDE ---
    spd_index, spd_lengths = _shortest_paths(
        edge_index, num_nodes=num_atoms, spd_max_length=8
    )

    # --- bond-graph SPDE ---
    e2e_spd_index, e2e_spd_lengths = _shortest_paths(
        e2e_edge_index, num_nodes=num_bonds, spd_max_length=8
    )

    # --- RWSE (flattened all-pairs P^k[i, j], row-major per graph) ---
    rwse = pairwise_random_walk_landing_probs(
        edge_index, num_nodes=num_atoms, steps=16
    )
    e2e_rwse = pairwise_random_walk_landing_probs(
        e2e_edge_index, num_nodes=num_bonds, steps=16
    )

    e_batch = torch.zeros((num_bonds,), dtype=torch.long, device=x.device)

    transformed = DGTData(**data._store)
    transformed.dgt_e2e_edge_index = e2e_edge_index
    transformed.dgt_e2e_node_index = e2e_node_index
    transformed.dgt_e_batch = e_batch
    transformed.dgt_spd_index = spd_index
    transformed.dgt_spd_lengths = spd_lengths
    transformed.dgt_e2e_spd_index = e2e_spd_index
    transformed.dgt_e2e_spd_lengths = e2e_spd_lengths
    transformed.dgt_rwse = rwse
    transformed.dgt_e2e_rwse = e2e_rwse
    return transformed


def _line_graph_topology(edge_index: Tensor) -> tuple[Tensor, Tensor]:
    """Return bond-graph ``[2, L]`` edge index and ``[L]`` shared-atom index.

    Two undirected bonds are adjacent when they share an atom; both directions
    are recorded (bidirectional in PyG).  The shared atom id is recorded per
    line-graph edge.
    """

    undirected = edge_index[:, edge_index[0] < edge_index[1]]
    bond_list = undirected.transpose(0, 1).tolist()
    e2e_edges: list[tuple[int, int]] = []
    e2e_nodes: list[int] = []
    num_bonds = len(bond_list)
    for i in range(num_bonds):
        for j in range(i + 1, num_bonds):
            edge_i = bond_list[i]
            edge_j = bond_list[j]
            shared = [a for a in edge_i if a in edge_j]
            for atom in shared:
                e2e_edges.append((i, j))
                e2e_edges.append((j, i))
                e2e_nodes.append(atom)
                e2e_nodes.append(atom)

    if not e2e_edges:
        return (
            torch.empty((2, 0), dtype=torch.long, device=edge_index.device),
            torch.empty(0, dtype=torch.long, device=edge_index.device),
        )
    return (
        torch.tensor(e2e_edges, dtype=torch.long, device=edge_index.device)
        .t()
        .contiguous(),
        torch.tensor(e2e_nodes, dtype=torch.long, device=edge_index.device),
    )


def _shortest_paths(
    edge_index: Tensor, *, num_nodes: int, spd_max_length: int
) -> tuple[Tensor, Tensor]:
    """Compute shortest-path distances via adjacency matrix powers.

    Returns sparse ``(index, lengths)`` where lengths are integer distances in
    ``{1, ..., spd_max_length}``.  The diagonal is excluded (self-distance 0).
    Mirrors the official ``compute_shortest_paths`` algorithm without the dense
    batch, and is deterministic for a given graph.
    """

    if num_nodes == 0:
        empty_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)
        empty_lengths = torch.empty(0, dtype=torch.long, device=edge_index.device)
        return empty_index, empty_lengths
    if edge_index.numel() == 0:
        adjacency = edge_index.new_zeros((num_nodes, num_nodes))
    else:
        adjacency = to_dense_adj(
            edge_index, max_num_nodes=num_nodes
        ).squeeze(0).long()
    if adjacency.ndim != 2:
        adjacency = adjacency.unsqueeze(0)

    shortest = adjacency.clone()
    reach = adjacency.clone()
    for step in range(1, spd_max_length):
        reach = (adjacency @ reach > 0).long()
        shortest += ((shortest == 0) * reach * (step + 1)).long()
        if bool((shortest > 0).all()):
            break
    shortest = shortest.fill_diagonal_(0)
    return dense_to_sparse(shortest)


def pairwise_random_walk_landing_probs(
    edge_index: Tensor, *, num_nodes: int, steps: int
) -> Tensor:
    """Return flattened all-pairs RWSE ``[num_nodes * num_nodes, steps]``.

    With ``P = D^{-1} A`` the entry at flat row ``i * num_nodes + j`` and
    column ``k`` is exactly ``(P^k)[i, j]`` — the same quantity as
    ``rw_landing_all`` from the official ``get_rw_landing_probs``.  Rows use
    row-major ``(i, j)`` order so a per-graph block reshapes back to
    ``[N, N, K]`` without extra bookkeeping.
    """

    if num_nodes == 0:
        return torch.empty((0, steps), dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return torch.zeros(
            (num_nodes * num_nodes, steps),
            dtype=torch.float32,
            device=edge_index.device,
        )
    adjacency = to_dense_adj(
        edge_index, max_num_nodes=num_nodes
    ).squeeze(0).float()
    if adjacency.ndim != 2:
        adjacency = adjacency.unsqueeze(0)
    degree = adjacency.sum(dim=-1, keepdim=True)
    degree_inv = degree.pow(-1.0)
    degree_inv.masked_fill_(degree_inv == float("inf"), 0.0)
    transition = degree_inv * adjacency  # D^-1 A

    columns: list[Tensor] = []
    power = transition.clone().detach()
    for _ in range(steps):
        columns.append(power.reshape(num_nodes * num_nodes, 1))
        power = power @ transition
    return torch.cat(columns, dim=-1).float()


def _sample_id(data: MolecularData) -> int | str:
    value = getattr(data, "sample_id", None)
    if isinstance(value, Tensor) and value.numel():
        return int(value.reshape(-1)[0].item())
    return "<unknown>"


__all__ = ["DGTData", "add_dgt_inputs", "pairwise_random_walk_landing_probs"]

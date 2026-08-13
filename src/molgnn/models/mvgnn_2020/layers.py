"""Building blocks for the MV-GNNcross 2020 multi-view architecture.

Cross-dependent message passing on the molecular line graph (paper Section 3.5,
upstream ``MPNPlusEncoder`` in ``dglt/models/layers.py``). Two hidden states are
maintained simultaneously: a per-atom state and a per-bond state. Each hop, the
node aggregation reads the latest bond hidden states and the bond aggregation
reads the latest atom hidden states, so information flows across views at every
layer instead of only meeting at the readout.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter, to_dense_batch


# ---------------------------------------------------------------------------
# Line-graph utility
# ---------------------------------------------------------------------------


def compute_reverse_edge_index(edge_index: Tensor) -> Tensor:
    """Return the involution ``b2revb`` for a canonical PyG molecular graph.

    The canonical runner stores both directions of every bond without
    self-loops, so for each edge ``(src, dst)`` at position ``e`` the reverse
    direction ``(dst, src)`` exists at a unique position ``e'``. This helper
    materializes that permutation; it is the equivalent of the upstream
    ``b2revb`` mapping consumed by ``MPNPlusEncoder.one_hop``.

    Args:
        edge_index: ``[2, E]`` ``torch.long`` tensor with the canonical pair
            convention (both directions present, no self-loops).

    Returns:
        ``[E]`` ``torch.long`` tensor such that
        ``edge_index[:, reverse[e]] == edge_index[[1, 0], e]``.
    """

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edge_index.dtype != torch.long:
        raise ValueError("edge_index must be a torch.long tensor")

    # Guard against the (degenerate) bondless molecule so ``src.max()`` does
    # not raise; the returned empty tensor is consistent with scatter outputs.
    if edge_index.shape[1] == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    src = edge_index[0]
    dst = edge_index[1]
    # A dense ``[N_atoms, N_atoms]`` lookup table trades a small amount of
    # memory for an O(1) gather that mirrors the upstream ``b2revb`` semantics.
    # Memory is bounded by ``N_atoms^2`` per batch (a few hundred KB for typical
    # molecular-property-prediction batches).
    num_atoms = int(max(src.max().item(), dst.max().item())) + 1
    lookup = torch.full(
        (num_atoms, num_atoms), -1, dtype=torch.long, device=edge_index.device
    )
    forward_index = torch.arange(
        edge_index.shape[1], device=edge_index.device, dtype=torch.long
    )
    lookup[src, dst] = forward_index
    reverse = lookup[dst, src]
    if (reverse < 0).any():
        raise ValueError(
            "edge_index does not store both directions of every bond; "
            "the canonical molecular graph must be paired"
        )
    return reverse


# ---------------------------------------------------------------------------
# Self-attention readout
# ---------------------------------------------------------------------------


class SelfAttentionReadout(nn.Module):
    """Shared self-attentive graph readout (Equation 8 in the paper).

    Given per-node embeddings ``H \in R^{N \times d_out}`` and the per-node
    graph assignment, computes

        S = softmax(W_2 tanh(W_1 H^T))        (one row per attention head)
        xi = Flatten(S @ H)                   (R^{num_graphs \times r \cdot d_out})

    ``W_1`` and ``W_2`` are the **shared** parameters across the node-central
    and edge-central views, exactly as required by Section 3.3. ``S`` is
    computed per graph via :func:`torch_geometric.utils.to_dense_batch` so the
    softmax stays inside each molecule.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_hidden: int,
        attn_heads: int,
    ) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(attn_hidden, "attn_hidden")
        _positive_int(attn_heads, "attn_heads")
        self.hidden_dim = hidden_dim
        self.attn_hidden = attn_hidden
        self.attn_heads = attn_heads

        # Xavier init mirrors the upstream ``Attention.reset_parameters`` and
        # keeps the two views starting from the same statistical prior when
        # the parameters are shared across views (see ``MVGNNcross``).
        self.W_att1 = nn.Parameter(torch.empty(attn_hidden, hidden_dim))
        self.W_att2 = nn.Parameter(torch.empty(attn_heads, attn_hidden))
        nn.init.xavier_normal_(self.W_att1)
        nn.init.xavier_normal_(self.W_att2)

    def forward(
        self, h: Tensor, graph_batch: Tensor, num_graphs: int
    ) -> Tensor:
        """Return one ``[num_graphs, attn_heads * hidden_dim]`` tensor."""

        # Group nodes by graph and pad to ``[num_graphs, max_atoms, hidden]``.
        # ``mask`` is ``[num_graphs, max_atoms]`` with True for real atoms so
        # the softmax ignores the padded positions.
        h_dense, mask = to_dense_batch(h, graph_batch)
        # ``W_1 @ H^T`` -> ``[num_graphs, attn_hidden, max_atoms]`` after the
        # transpose that matches the upstream left-multiplied form.
        scores = torch.matmul(self.W_att1, h_dense.transpose(1, 2))
        scores = torch.tanh(scores)
        # ``W_2 @ scores`` -> ``[num_graphs, attn_heads, max_atoms]``.
        attn_logits = torch.matmul(self.W_att2, scores)
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1), -1.0e9)
        attn = torch.softmax(attn_logits, dim=-1)
        # ``S @ H`` -> ``[num_graphs, attn_heads, hidden_dim]``; flatten the
        # heads so the head MLP sees a fixed-size vector per graph.
        graph_embed = torch.einsum("brn,bnd->brd", attn, h_dense)
        return graph_embed.reshape(num_graphs, self.attn_heads * self.hidden_dim)


# ---------------------------------------------------------------------------
# Cross-dependent message passing hop
# ---------------------------------------------------------------------------


class MVGNNCrossHop(nn.Module):
    """One hop of cross-dependent message passing (paper Equation 9).

    Maintains two hidden states simultaneously: ``atom_h`` per node and
    ``bond_h`` per directed edge (the line graph). At every hop each state
    aggregates the *latest* hidden states of its neighbors in both views:

      * atom aggregation sums ``atom_h[neighbor]`` and ``bond_h[incoming]``;
      * bond aggregation (D-MPNN-style, excluding the reverse direction so the
        message does not flow back to the source node) sums both ``atom_h``
        and ``bond_h`` of the line-graph neighbors.

    The residual is applied with respect to the *input* projection (not the
    previous hop's output) so the gradients flow through the depth loop even
    when ``W_node`` and ``W_edge`` are shared across hops (matches upstream).

    The bond aggregation follows the upstream convention exactly: it sums
    only the bonds *incoming* to the source atom (``a2b`` in upstream
    parlance) and excludes the reverse bond. The atom aggregation sums the
    atom neighbors of the source atom and excludes the destination atom.
    """

    def __init__(
        self,
        hidden_dim: int,
        atom_dim: int,
        bond_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        _positive_int(atom_dim, "atom_dim")
        _positive_int(bond_dim, "bond_dim")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        self.hidden_dim = hidden_dim
        self.dropout_p = float(dropout)

        # Per-hop weights. ``attach_fea=True`` keeps the raw bond features in
        # the atom message and the raw atom features in the bond message; this
        # matches the upstream default (``self.attach_fea = not
        # args.no_attach_fea``).
        self.W_node = nn.Linear(
            2 * hidden_dim + bond_dim, hidden_dim, bias=True
        )
        self.W_edge = nn.Linear(
            2 * hidden_dim + atom_dim, hidden_dim, bias=True
        )
        self.act = nn.ReLU()
        self.dropout_layer = (
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

    def forward(
        self,
        atom_h: Tensor,
        bond_h: Tensor,
        atom_input: Tensor,
        bond_input: Tensor,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        reverse_edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return updated ``(atom_h, bond_h)`` tensors after one hop."""

        src, dst = edge_index[0], edge_index[1]
        num_atoms = atom_h.shape[0]

        # --- Aggregations on the *pre-hop* states ------------------------------
        # These are what the upstream feeds into the atom one_hop. Atom
        # neighbour sum uses ``scatter(atom_h[src], dst)`` which in the
        # canonical paired graph equals the upstream ``a2a`` aggregation.
        atom_nei_per_atom = scatter(
            atom_h[src], dst, dim=0, dim_size=num_atoms, reduce="sum"
        )
        # bond_nei[v] = sum of bond_h over bonds *incoming* to v (a2b in
        # upstream). This is the upstream's narrow line-graph convention.
        bond_nei_per_atom = scatter(
            bond_h, dst, dim=0, dim_size=num_atoms, reduce="sum"
        )
        # bond_e[v] = sum of raw bond features over bonds incoming to v.
        bond_e_per_atom = scatter(
            edge_attr, dst, dim=0, dim_size=num_atoms, reduce="sum"
        )
        # atom_nei_fea[v] = sum of raw atom features over neighbours of v; this
        # only depends on the constant ``x`` so it is safe to compute once.
        atom_nei_fea = scatter(
            x[src], dst, dim=0, dim_size=num_atoms, reduce="sum"
        )

        # --- Atom update (Eq. 9, top row) --------------------------------------
        # W_node consumes only the aggregated neighbour messages; ``atom_h``
        # enters via the residual from the input projection.
        node_input = torch.cat(
            (atom_nei_per_atom, bond_nei_per_atom, bond_e_per_atom),
            dim=-1,
        )
        atom_msg = self.W_node(node_input)
        atom_h_new = self.dropout_layer(self.act(atom_input + atom_msg))

        # --- Aggregations on the *new* atom state ------------------------------
        # The upstream bond one_hop receives ``attached_message = atom_message``
        # *after* the atom update, so the atom-neighbour sum and the
        # destination-atom subtraction must both use ``atom_h_new``.
        atom_nei_per_atom_new = scatter(
            atom_h_new[src], dst, dim=0, dim_size=num_atoms, reduce="sum"
        )

        # --- Bond update (Eq. 9, bottom row) -----------------------------------
        # Paper aggregation: ``AGGedge({h^(k-1)_vw, h^(k-1)_uv, h^(k-1)_u,
        # x_u | u in N_v \ w})``. Upstream realises this as
        # ``a_message[b2a] - rev_message`` where ``a_message = cat(
        # bond_nei_per_atom, atom_nei_per_atom_new, atom_nei_fea)`` and
        # ``rev_message = cat(bond_h[reverse], atom_h_new[w], x[w])``.
        bond_nei_excluded = bond_nei_per_atom[src] - bond_h[reverse_edge_index]
        atom_nei_excluded = atom_nei_per_atom_new[src] - atom_h_new[dst]
        atom_nei_fea_excluded = atom_nei_fea[src] - x[dst]

        # Component order matches the upstream ``a_message`` concatenation.
        edge_input = torch.cat(
            (bond_nei_excluded, atom_nei_excluded, atom_nei_fea_excluded),
            dim=-1,
        )
        bond_msg = self.W_edge(edge_input)
        bond_h_new = self.dropout_layer(self.act(bond_input + bond_msg))

        return atom_h_new, bond_h_new


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = [
    "MVGNNCrossHop",
    "SelfAttentionReadout",
    "compute_reverse_edge_index",
]
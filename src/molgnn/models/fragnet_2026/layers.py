"""FragNet 2026 four-graph GAT layer for molecular property prediction."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import add_self_loops, scatter, softmax


class FragNetLayerA(nn.Module):
    """Interleaved 4-graph attention layer from FragNet (Gihan et al. 2026).

    The layer performs four interacting multi-head attention passes on the
    coupled graphs produced by ``add_fragnet_inputs``:

    1. Bond (line) graph: nodes = directed edges of the atom graph;
       edges = pairs of bonds sharing an atom; edge features = ``cos(angle)``.
    2. Atom graph: standard atom graph with self-loops added internally;
       edge features = new bond-graph node features (+ zero self-loop attr).
    3. Fragment-bond graph: nodes = inter-fragment connections; edges
       connect pairs of connections that share a fragment; edge features =
       element-wise sum of the two connections' features.
    4. Fragment graph: nodes = BRICS fragments; edges = fragment-to-fragment
       (``frag_index``); edge features = new fragment-bond-graph node features.
    """

    def __init__(
        self,
        atom_in: int,
        atom_out: int,
        frag_in: int,
        frag_out: int,
        edge_in: int,
        edge_out: int,
        fedge_in: int,
        num_heads: int,
        bond_edge_in: int = 1,
        fbond_edge_in: int = 6,
    ) -> None:
        super().__init__()

        _positive_int(atom_in, "atom_in")
        _positive_int(atom_out, "atom_out")
        _positive_int(frag_in, "frag_in")
        _positive_int(frag_out, "frag_out")
        _positive_int(edge_in, "edge_in")
        _positive_int(edge_out, "edge_out")
        _positive_int(fedge_in, "fedge_in")
        _positive_int(num_heads, "num_heads")
        _positive_int(bond_edge_in, "bond_edge_in")
        _positive_int(fbond_edge_in, "fbond_edge_in")

        if atom_out % num_heads:
            raise ValueError("atom_out must be divisible by num_heads")
        if frag_out % num_heads:
            raise ValueError("frag_out must be divisible by num_heads")
        if edge_out % num_heads:
            raise ValueError("edge_out must be divisible by num_heads")
        if frag_out != atom_out:
            raise ValueError("frag_out must equal atom_out for the fragment-graph GAT")

        self.num_heads = num_heads
        self.atom_out = atom_out
        self.frag_out = frag_out
        self.edge_out = edge_out
        self.atom_out_per_head = atom_out // num_heads
        self.frag_out_per_head = frag_out // num_heads
        self.edge_out_per_head = edge_out // num_heads

        # Initial / inter-stage embeddings (used by the encoder's first layer and
        # by subsequent layers respectively).
        self.atom_embed = nn.Linear(atom_in, atom_out)
        self.frag_embed = nn.Linear(frag_in, frag_out)
        self.edge_embed = nn.Linear(edge_in, edge_out)

        # Per-head edge-feature embeddings for the line graph and fragment-bond
        # graph.
        self.edge_attr_bond_embed = nn.Linear(bond_edge_in, self.edge_out_per_head)
        self.edge_attr_fbond_embed = nn.Linear(fbond_edge_in, self.edge_out_per_head)

        # Multi-head node projections (full-dim output, reshaped to [N, H, d]).
        self.projection_b = nn.Linear(edge_in, edge_out)
        self.projection_fb = nn.Linear(fedge_in, edge_out)
        self.projection_a = nn.Linear(atom_in, atom_out)

        # Attention parameters: one per head.  Message layout = [target, edge,
        # source] for all four GATs.
        self.a_b = nn.Parameter(torch.empty(num_heads, 3 * self.edge_out_per_head))
        self.f_a_b = nn.Parameter(torch.empty(num_heads, 3 * self.edge_out_per_head))
        self.a = nn.Parameter(
            torch.empty(num_heads, 2 * self.atom_out_per_head + edge_out)
        )
        self.f = nn.Parameter(
            torch.empty(num_heads, 2 * self.atom_out_per_head + edge_out)
        )

        for tensor in (self.a_b, self.f_a_b, self.a, self.f):
            nn.init.xavier_uniform_(tensor, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(
        self,
        x_atoms: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        frag_index: Tensor,
        x_frags: Tensor,
        atom_to_frag_ids: Tensor,
        node_features_bond_graph: Tensor,
        edge_index_bonds_graph: Tensor,
        edge_attr_bonds: Tensor,
        node_features_fbond_graph: Tensor,
        edge_index_fbonds: Tensor,
        edge_attr_fbonds: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Run one round of the four interleaved GAT passes.

        Returns ``(x_atoms_new, x_frags_new, new_bond_features, new_fbond_features)``.
        """

        num_atoms = x_atoms.size(0)
        num_fragments = x_frags.size(0)
        num_bonds = node_features_bond_graph.size(0)
        num_connections = node_features_fbond_graph.size(0)

        # ----- 1. Bond (line) graph GAT. -----
        target_b, source_b = edge_index_bonds_graph[0], edge_index_bonds_graph[1]
        node_feats_b = self.projection_b(node_features_bond_graph).view(
            num_bonds, self.num_heads, self.edge_out_per_head
        )
        ea_bonds = self.edge_attr_bond_embed(edge_attr_bonds)  # [E_b, edge_out_per_head]
        ea_bonds = ea_bonds.unsqueeze(1).expand(-1, self.num_heads, -1)
        target_features_b = node_feats_b[target_b]
        source_features_b = node_feats_b[source_b]
        message_b = torch.cat([target_features_b, ea_bonds, source_features_b], dim=-1)
        attn_logits_b = (message_b * self.a_b).sum(dim=-1)
        attn_logits_b = self.leakyrelu(attn_logits_b)
        attn_probs_b = softmax(attn_logits_b, target_b, num_nodes=num_bonds)
        bond_messages = attn_probs_b.unsqueeze(-1) * source_features_b
        new_bond_features = scatter(
            bond_messages, target_b, dim=0, dim_size=num_bonds, reduce="sum"
        ).view(num_bonds, self.edge_out)

        # ----- 2. Atom-graph GAT (with self-loops). -----
        edge_index_with_self, _ = add_self_loops(edge_index, num_nodes=num_atoms)
        self_loop_attr = torch.zeros(
            num_atoms, self.edge_out, dtype=edge_attr.dtype, device=edge_attr.device
        )
        new_edge_attr = torch.cat([new_bond_features, self_loop_attr], dim=0)
        source_a, target_a = edge_index_with_self[0], edge_index_with_self[1]
        node_feats_a = self.projection_a(x_atoms).view(
            num_atoms, self.num_heads, self.atom_out_per_head
        )
        target_features_a = node_feats_a[target_a]
        source_features_a = node_feats_a[source_a]
        edge_attr_repeat = new_edge_attr.unsqueeze(1).expand(-1, self.num_heads, -1)
        message_a = torch.cat(
            [target_features_a, edge_attr_repeat, source_features_a], dim=-1
        )
        attn_logits_a = (message_a * self.a).sum(dim=-1)
        attn_logits_a = self.leakyrelu(attn_logits_a)
        attn_probs_a = softmax(attn_logits_a, target_a, num_nodes=num_atoms)
        atom_messages = attn_probs_a.unsqueeze(-1) * source_features_a
        x_atoms_new = scatter(
            atom_messages, target_a, dim=0, dim_size=num_atoms, reduce="sum"
        ).view(num_atoms, self.atom_out)

        # ----- 3. Pool atom -> fragment via atom_to_fragment assignment. -----
        x_frags_pooled = scatter(
            x_atoms_new, atom_to_frag_ids, dim=0, dim_size=num_fragments, reduce="sum"
        )

        # ----- 4. Fragment-bond (bipartite) graph GAT. -----
        # Transform builds edge_index_fbonds as [source=connection, target=fragment].
        source_fb = edge_index_fbonds[0]
        target_fb = edge_index_fbonds[1]
        node_feats_fb = self.projection_fb(node_features_fbond_graph).view(
            num_connections, self.num_heads, self.edge_out_per_head
        )
        ea_fbonds = self.edge_attr_fbond_embed(edge_attr_fbonds)
        ea_fbonds = ea_fbonds.unsqueeze(1).expand(-1, self.num_heads, -1)
        target_features_fb = node_feats_fb[target_fb]
        source_features_fb = node_feats_fb[source_fb]
        message_fb = torch.cat(
            [target_features_fb, ea_fbonds, source_features_fb], dim=-1
        )
        attn_logits_fb = (message_fb * self.f_a_b).sum(dim=-1)
        attn_logits_fb = self.leakyrelu(attn_logits_fb)
        attn_probs_fb = softmax(attn_logits_fb, target_fb, num_nodes=num_connections)
        fbond_messages = attn_probs_fb.unsqueeze(-1) * source_features_fb
        new_fbond_features = scatter(
            fbond_messages, target_fb, dim=0, dim_size=num_connections, reduce="sum"
        ).view(num_connections, self.edge_out)

        # ----- 5. Fragment-graph GAT (fragment-to-fragment edges). -----
        # frag_index has two directed edges per connection (Begin↔End);
        # new_fbond_features has one entry per connection, so repeat each
        # entry twice to align with the message-passing edges.
        if frag_index.shape[1] and num_connections:
            source_f, target_f = frag_index[0], frag_index[1]
            node_feats_f = x_frags_pooled.view(
                num_fragments, self.num_heads, self.frag_out_per_head
            )
            new_fbond_repeat = (
                new_fbond_features.repeat_interleave(2, dim=0)
                .unsqueeze(1)
                .expand(-1, self.num_heads, -1)
            )
            target_features_f = node_feats_f[target_f]
            source_features_f = node_feats_f[source_f]
            message_f = torch.cat(
                [target_features_f, new_fbond_repeat, source_features_f], dim=-1
            )
            attn_logits_f = (message_f * self.f).sum(dim=-1)
            attn_logits_f = self.leakyrelu(attn_logits_f)
            attn_probs_f = softmax(attn_logits_f, target_f, num_nodes=num_fragments)
            frag_messages = attn_probs_f.unsqueeze(-1) * source_features_f
            x_frags_new = scatter(
                frag_messages,
                target_f,
                dim=0,
                dim_size=num_fragments,
                reduce="sum",
            ).view(num_fragments, self.frag_out)
        else:
            x_frags_new = x_frags_pooled

        return x_atoms_new, x_frags_new, new_bond_features, new_fbond_features


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["FragNetLayerA"]

"""MV-GNNcross 2020 molecular property predictor.

Architecture-only port of the cross-dependent variant of Multi-View Graph
Neural Networks for Molecular Property Prediction (Ma et al., 2020;
arXiv:2005.13607). The model simultaneously maintains per-atom and per-bond
hidden states and lets each view read the other's hidden states at every
message-passing hop, then ensembles two MLP heads after a shared self-
attentive readout.

Disagreement loss (paper Section 3.4) is a training-time auxiliary that
stabilises the two-view training; it is not part of ``forward`` and so is not
implemented here — the shared experiment runner in this lab only consumes a
single per-graph prediction tensor.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import scatter

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import MVGNNCrossHop, SelfAttentionReadout, compute_reverse_edge_index


# Module-level constant — keeps the construction graph deterministic and easy to
# inspect from outside the class.
_VALID_ENSEMBLE_MODES = ("mean", "sum", "max")


class MVGNNcross(BaseMolecularModel):
    """Multi-View GNN with cross-dependent message passing (MV-GNNcross).

    Two parallel views are maintained at every layer: the node-central view
    tracks per-atom states and the edge-central view tracks per-bond states.
    At each hop the node aggregation includes the latest bond hidden states and
    the bond aggregation includes the latest atom hidden states, so the two
    information streams meet every layer (Section 3.5). After ``depth`` hops a
    shared self-attentive readout (Section 3.3) produces one graph-level
    embedding per view, two MLP heads produce the per-view predictions, and
    the final output is the ensemble of the two.

    The canonical 153/14 featuriser is used as-is. No additional graph
    transform is required.

    Notes on faithfulness:

    * The upstream's ``MPNPlusEncoder`` shares its ``W_node`` and ``W_edge``
      across hops; this model reuses a single :class:`MVGNNCrossHop` instance
      inside the ``depth - 1`` loop to reproduce that.
    * The upstream's bond aggregation sums only the bonds *incoming* to the
      source atom (``a2b`` in upstream parlance) and excludes the reverse
      bond. This is a narrower line-graph convention than the paper's
      Equation 9 suggests (``sum_{u in N_v \ w} h_vu`` over outgoing bonds),
      but it is what the upstream code actually implements.
    """

    required_batch_fields = ("x", "edge_index", "edge_attr", "batch")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        hidden_dim: int = 64,
        depth: int = 3,
        attn_hidden: int = 64,
        attn_heads: int = 1,
        dropout: float = 0.0,
        ensemble: str = "mean",
    ) -> None:
        super().__init__()

        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (depth, "depth"),
            (attn_hidden, "attn_hidden"),
            (attn_heads, "attn_heads"),
        ):
            _positive_int(value, name)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if ensemble not in _VALID_ENSEMBLE_MODES:
            raise ValueError(
                f"ensemble must be one of {_VALID_ENSEMBLE_MODES}; got {ensemble!r}"
            )

        # Per-task injected dimensions — never let YAML override these (the
        # registry raises ``RegistryError`` if a YAML parameter shadows them).
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.attn_hidden = attn_hidden
        self.attn_heads = attn_heads
        self.dropout_p = float(dropout)
        self.ensemble = ensemble

        # Input projections mirror the upstream's ``W_nin`` and ``W_ein``.
        self.W_atom_in = nn.Linear(atom_dim, hidden_dim)
        self.W_bond_in = nn.Linear(bond_dim, hidden_dim)
        # Output projections aggregate the final states back to per-atom tensors
        # by summing neighbours (same ``a2a``/``a2b`` aggregations used in the
        # hops) and concatenating with the raw atom features, matching the
        # upstream ``aggreate_to_atom_fea`` step in ``MPNPlusEncoder``.
        self.W_atom_out = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.W_bond_out = nn.Linear(atom_dim + hidden_dim, hidden_dim)
        self.act = nn.ReLU()
        # Single dropout module applied after every hop and after the output
        # projection, mirroring the upstream's single ``dropout_layer``.
        self.dropout_layer = (
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

        # ``depth - 1`` hops are evaluated on the same module so the W_node
        # and W_edge parameters are shared across hops (matches the upstream
        # ``for depth in range(self.depth - 1)`` loop in
        # ``MPNPlusEncoder.forward``).
        self.hop = MVGNNCrossHop(hidden_dim, atom_dim, bond_dim, self.dropout_p)

        # Shared self-attention readout — W_att1 and W_att2 are shared by
        # construction (single ``SelfAttentionReadout`` instance used twice).
        self.readout = SelfAttentionReadout(hidden_dim, attn_hidden, attn_heads)
        head_in = attn_heads * hidden_dim
        # Two parallel MLP heads mirror the paper's "two MLPs" producing the
        # per-view predictions that are then ensembled. Each head is a 2-layer
        # MLP with a ReLU after the first linear; the upstream applies no
        # additional dropout here and the final linear returns raw logits for
        # ``bce_with_logits``.
        self.head_node = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets),
        )
        self.head_edge = nn.Sequential(
            nn.Linear(head_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_targets),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or binary classification logits."""

        x, edge_index, edge_attr, graph_batch, num_graphs = self._validate_batch(
            batch
        )
        reverse_edge_index = compute_reverse_edge_index(edge_index)

        # Input projections; the activations seed the per-hop residual.
        atom_input = self.W_atom_in(x)
        bond_input = self.W_bond_in(edge_attr)
        atom_h = self.act(atom_input)
        bond_h = self.act(bond_input)

        for _ in range(max(self.depth - 1, 0)):
            atom_h, bond_h = self.hop(
                atom_h,
                bond_h,
                atom_input,
                bond_input,
                x,
                edge_index,
                edge_attr,
                reverse_edge_index,
            )

        # Output projection: aggregate neighbours per atom (same scatter pattern
        # as the hops) and concatenate the raw atom features, then apply
        # W_*out and (per the upstream ``aggreate_to_atom_fea``) dropout.
        # The bond view also ends in atom-space (upstream does the same so the
        # shared readout can consume both embeddings with identical shape).
        # Match the upstream ``a2b`` aggregation: bonds incoming to atom v,
        # i.e. ``scatter`` over ``edge_index[1]`` (the destination index).
        atom_out_nei = scatter(
            atom_h[edge_index[0]],
            edge_index[1],
            dim=0,
            dim_size=x.shape[0],
            reduce="sum",
        )
        atom_out = self.dropout_layer(
            self.act(self.W_atom_out(torch.cat((x, atom_out_nei), dim=-1)))
        )
        bond_out_nei = scatter(
            bond_h,
            edge_index[1],
            dim=0,
            dim_size=x.shape[0],
            reduce="sum",
        )
        bond_out = self.dropout_layer(
            self.act(self.W_bond_out(torch.cat((x, bond_out_nei), dim=-1)))
        )

        # Shared self-attentive readout produces per-view graph embeddings of
        # shape ``[num_graphs, attn_heads * hidden_dim]``.
        graph_atom = self.readout(atom_out, graph_batch, num_graphs)
        graph_edge = self.readout(bond_out, graph_batch, num_graphs)

        # Two MLP heads (one per view), then ensemble.
        pred_atom = self.head_node(graph_atom)
        pred_edge = self.head_edge(graph_edge)
        if self.ensemble == "mean":
            return 0.5 * (pred_atom + pred_edge)
        if self.ensemble == "sum":
            return pred_atom + pred_edge
        # ``max`` is provided for parity with the paper's "ensemble" wording;
        # predictions are logits so this is the elementwise max (rarely used).
        return torch.maximum(pred_atom, pred_edge)

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Validate tensors, fetch them, and return canonical inputs."""

        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(graph_batch, Tensor)

        if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != self.atom_dim:
            raise ValueError(
                f"batch.x must have shape [N, {self.atom_dim}] with N >= 1"
            )
        if x.dtype != torch.float32 or not torch.isfinite(x).all():
            raise ValueError("batch.x must contain finite torch.float32 values")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.long
        ):
            raise ValueError(
                "batch.edge_index must have shape [2, E] and dtype torch.long"
            )
        if edge_attr.shape != (edge_index.shape[1], self.bond_dim):
            raise ValueError(
                f"batch.edge_attr must have shape [E, {self.bond_dim}]"
            )
        if edge_attr.dtype != torch.float32 or not torch.isfinite(edge_attr).all():
            raise ValueError(
                "batch.edge_attr must contain finite torch.float32 values"
            )
        if edge_index.shape[1] and (
            edge_index.min().item() < 0 or edge_index.max().item() >= x.shape[0]
        ):
            raise ValueError("batch.edge_index contains an invalid node index")
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if any(
            value.device != x.device
            for value in (edge_index, edge_attr, graph_batch)
        ):
            raise ValueError(
                "all MVGNNcross batch tensors must be on the same device"
            )

        # The canonical graph layout stores both directions of each bond with
        # no self-loops, matching the upstream's ``a2b``/``b2a``/``b2revb``
        # conventions (Section 3.5 + the ``mol2graph`` construction in dglt).
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=True,
        )
        return x, edge_index, edge_attr, graph_batch, num_graphs


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["MVGNNcross"]
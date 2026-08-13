"""Building blocks for the EGNN 2021 architecture.

The ``EGCL`` (Equivariant Graph Convolutional Layer) is the core message-
passing layer of EGNN (paper Section 3, Equations 3-6).  Each layer maintains
two states per node — the invariant feature ``h`` and the equivariant
coordinate ``x`` — and fuses them so the message term can depend on the
relative geometry between atoms while the position update remains
equivariant to rotations, translations and reflections.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter


# Module-level constant — kept here rather than on the class so the supported
# coordinate aggregation modes are easy to enumerate from outside.
_COORDS_AGGREGATION_MODES = ("mean", "sum")


class EGCL(nn.Module):
    """Equivariant Graph Convolutional Layer (paper Section 3, Equations 3-6).

    The layer takes the current node features ``h``, the canonical edge index,
    the per-atom coordinates ``x``, and the edge attributes ``edge_attr``,
    then returns the updated ``(h, x, edge_attr)``.  Four sub-modules are
    owned by the layer:

    * ``edge_mlp`` consumes ``[h_i, h_j, ||x_i - x_j||^2, edge_attr]`` and emits
      the message embedding ``m_ij`` (Eq. 3).
    * ``node_mlp`` consumes ``[h_i, Σ_j m_ij]`` and emits the updated node
      feature ``h_i^{l+1}`` (Eq. 5 + 6).  A residual on ``h`` is applied when
      ``recurrent=True`` (the upstream default).
    * ``coord_mlp`` consumes ``m_ij`` and emits the per-edge scalar weight for
      the position update (Eq. 4).  The final ``Linear(hidden, 1)`` uses a
      Xavier-uniform init with ``gain=0.001`` to keep early coordinate drift
      bounded — the same stability trick the upstream ``gcl.py`` applies.
    """

    def __init__(
        self,
        hidden_dim: int,
        edges_in_d: int = 0,
        act_fn: nn.Module | None = None,
        recurrent: bool = True,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        coords_agg: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _positive_int(hidden_dim, "hidden_dim")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if coords_agg not in _COORDS_AGGREGATION_MODES:
            raise ValueError(
                f"coords_agg must be one of {_COORDS_AGGREGATION_MODES}; got {coords_agg!r}"
            )

        if act_fn is None:
            # Paper default — SiLU (a.k.a. Swish).  ReLU is the upstream's
            # QM9 alternative; we follow the paper text and the clean
            # reference (``egnn_clean``) which both use SiLU.
            act_fn = nn.SiLU()

        self.hidden_dim = hidden_dim
        self.recurrent = recurrent
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.coords_agg = coords_agg
        self.dropout_p = float(dropout)

        # The edge MLP fuses ``[h_i, h_j, radial, edge_attr]`` into a
        # ``hidden_dim`` message embedding.  ``radial`` is always one
        # scalar — the squared pairwise distance — matching the paper.
        edge_input_dim = 2 * hidden_dim + 1 + edges_in_d
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        )

        # Node MLP consumes ``[h_i, Σ m_ij]``.  We do not concat ``node_attr``
        # here; the upstream's QM9 variant passes the raw input ``h_0`` as a
        # separate node attribute but the canonical 153-dim features are
        # already used as the initial embedding by the outer ``EGNN`` model.
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Coordinate MLP emits the per-edge scalar weight in Eq. 4.  The
        # final ``Linear(hidden, 1)`` carries ``bias=False`` and a small
        # Xavier init — identical to the upstream ``gcl.py`` initialisation
        # which keeps early coordinate drift bounded.
        coord_layers: list[nn.Module] = [
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
        ]
        final = nn.Linear(hidden_dim, 1, bias=False)
        torch.nn.init.xavier_uniform_(final.weight, gain=0.001)
        coord_layers.append(final)
        if self.tanh:
            # Optional tanh bound on the coord weight; off by default
            # because the paper notes it can hurt accuracy.
            coord_layers.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_layers)

        # Optional edge attention — multiplicative gate on the message
        # embedding, matching the upstream's ``attention`` switch.
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),
            )

        self.dropout_layer = (
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        coord: Tensor,
        edge_attr: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Return updated ``(h, coord, edge_attr)`` after one EGCL hop.

        Args:
            h: ``[N, hidden_dim]`` invariant node features.
            edge_index: ``[2, E]`` ``torch.long`` (canonical paired layout).
            coord: ``[N, n]`` equivariant coordinates (paper uses ``n=3``).
            edge_attr: ``[E, edge_dim]`` or ``None``.  When ``None`` the
                edge MLP falls back to ``[h_i, h_j, radial]`` only.

        Returns:
            ``(h', coord', edge_attr')`` — ``edge_attr`` is forwarded
            unchanged so the caller can keep its pipeline intact.
        """

        src, dst = edge_index[0], edge_index[1]
        radial, coord_diff = self._coord2radial(coord, src, dst)

        # ---- Edge MLP (Eq. 3) -----------------------------------------------
        h_src = h[src]
        h_dst = h[dst]
        edge_input = [h_src, h_dst, radial]
        if edge_attr is not None:
            edge_input.append(edge_attr)
        m_ij = self.edge_mlp(torch.cat(edge_input, dim=-1))
        if self.attention:
            m_ij = m_ij * self.att_mlp(m_ij)

        # ---- Coordinate update (Eq. 4) -------------------------------------
        coord_weight = self.coord_mlp(m_ij)
        # Per-edge translation: ``(x_i - x_j) * coord_weight``.  Aggregated
        # by source atom so each atom receives the vector field induced by
        # all of its line-graph neighbours.  The clamp mirrors the upstream
        # ``gcl.py`` guard rail (always applied to the *full* per-edge
        # translation, not the bare weight) — only triggers in pathological
        # cases but prevents NaNs from exploding gradients.
        coord_msg = torch.clamp(
            coord_diff * coord_weight, min=-100.0, max=100.0
        )
        if self.coords_agg == "mean":
            coord_agg = scatter(
                coord_msg, src, dim=0, dim_size=coord.shape[0], reduce="mean"
            )
        else:
            coord_agg = scatter(
                coord_msg, src, dim=0, dim_size=coord.shape[0], reduce="sum"
            )
        coord_new = coord + coord_agg

        # ---- Node update (Eq. 5 + 6) ---------------------------------------
        # Sum aggregation over incoming messages — ``scatter_add`` with
        # ``src`` groups by source atom which matches the upstream
        # ``unsorted_segment_sum(..., row, ...)`` where ``row`` is the
        # source index of the directed edge.
        m_i = scatter(m_ij, src, dim=0, dim_size=h.shape[0], reduce="sum")
        node_input = torch.cat((h, m_i), dim=-1)
        node_msg = self.node_mlp(node_input)
        if self.recurrent:
            # Residual on the invariant features (matches the upstream
            # ``out = x + out`` line in ``node_model``).
            h_new = h + self.dropout_layer(node_msg)
        else:
            h_new = self.dropout_layer(node_msg)

        return h_new, coord_new, edge_attr

    def _coord2radial(
        self,
        coord: Tensor,
        src: Tensor,
        dst: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return ``(radial, coord_diff)`` per paper Eq. 3 + Eq. 4."""

        coord_diff = coord[src] - coord[dst]
        radial = (coord_diff * coord_diff).sum(dim=-1, keepdim=True)
        if self.normalize:
            # Optional normalisation — dividing by the inter-atomic distance
            # keeps the relative vectors unit-norm.  Off by default; the
            # paper notes it can help in some future works but they did not
            # use it.
            norm = radial.sqrt().detach() + 1.0e-8
            coord_diff = coord_diff / norm
        return radial, coord_diff


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["EGCL", "_COORDS_AGGREGATION_MODES"]
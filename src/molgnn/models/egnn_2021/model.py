"""EGNN 2021 molecular property predictor.

Architecture-only port of the Equivariant Graph Neural Network from
Garcia Satorras, Hoogeboom, Welling, ICML 2021 (arXiv:2102.09844).  The model
preserves E(n) equivariance on the coordinate stream (rotations,
translations and reflections in any dimension) while letting the invariant
feature stream exchange information through standard message passing.

The clean reference (``tmp/EGNN/egnn-main/models/egnn_clean/egnn_clean.py``)
is what this port mirrors.  The heavier QM9 training script
(``qm9/models.py``) wraps the same layer in masks and node-attribute
bookkeeping that is unnecessary for the canonical 153-dim featurisation
used in this lab.

.. warning::

   The paper trains and evaluates EGNN on **QM9**, which ships with
   DFT-optimised 3-D geometries (B3LYP/6-31G(2df,p)).  The lab's
   companion ``add_egnn_inputs`` transform, in contrast, synthesises a
   per-molecule ETKDG + MMFF conformer from SMILES on the fly.  That
   gap — synthetic vs. DFT-optimised geometry — is the dominant source
   of any performance regression you see on 2-D-only benchmarks
   (BACE, BBBP, …).  The model code itself is faithful (bit-exact match
   to paper Eqs 3-6); only the input geometry is the weak link.  To
   evaluate EGNN as the paper intended, swap in a dataset that ships with
   real 3-D coordinates and replace ``add_egnn_inputs`` with a transform
   that reads those conformers from disk.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import EGCL


class EGNN(BaseMolecularModel):
    """E(n) Equivariant Graph Neural Network (Satorras et al., 2021).

    Architecture (paper Section 3):

      1. ``embedding_in`` projects the canonical 153-dim atom features into
         a ``hidden_dim``-dim invariant embedding.
      2. ``n_layers`` :class:`EGCL` blocks alternately update the invariant
         features ``h`` (Eq. 5-6) and the equivariant coordinates ``x`` (Eq. 4)
         — the message embedding fuses both streams via the radial term
         ``||x_i - x_j||^2``.
      3. ``node_dec`` is a 2-layer MLP applied to the final per-atom features
         before pooling (matches the upstream ``EGNN.node_dec``).
      4. Sum-pooling over atoms inside each graph yields one graph-level
         embedding, which ``graph_dec`` then maps to ``num_targets``.

    The model outputs raw logits for ``binary_classification`` (no Sigmoid)
    and raw regression values otherwise — the lab's shared loss layer
    (``bce_with_logits`` / ``mse``) applies the appropriate activation.

    .. warning::

       EGNN's E(n)-equivariance is only meaningful when ``batch.pos`` is a
       real low-energy 3-D structure.  The companion ``egnn_inputs`` graph
       transform synthesises an ETKDG + MMFF conformer from SMILES —
       the paper instead trains on QM9, where ``pos`` is the DFT-optimised
       geometry.  Cross-benchmark numbers from this lab (e.g. BACE at
       hidden_dim=32, n_layers=3) should therefore be read as a lower
       bound, not the architecture's true capability.
    """

    required_batch_fields = ("x", "edge_index", "edge_attr", "pos", "batch")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        hidden_dim: int = 64,
        n_layers: int = 4,
        attention: bool = False,
        normalize: bool = False,
        tanh: bool = False,
        coords_agg: str = "mean",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (num_targets, "num_targets"),
            (hidden_dim, "hidden_dim"),
            (n_layers, "n_layers"),
        ):
            _positive_int(value, name)
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        # Per-task injected dimensions — never let YAML override these (the
        # registry raises ``RegistryError`` if a YAML parameter shadows them).
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.num_targets = num_targets
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.coords_agg = coords_agg
        self.dropout_p = float(dropout)

        # Input projection mirrors the upstream ``embedding_in`` linear.
        self.embedding_in = nn.Linear(atom_dim, hidden_dim)
        # Per-layer EGCL — each layer owns its own weights (no parameter
        # sharing across depth) to match the upstream ``EGNN`` wrapper.
        self.gcl_layers = nn.ModuleList(
            EGCL(
                hidden_dim=hidden_dim,
                edges_in_d=bond_dim,
                recurrent=True,
                attention=attention,
                normalize=normalize,
                tanh=tanh,
                coords_agg=coords_agg,
                dropout=self.dropout_p,
            )
            for _ in range(n_layers)
        )

        # Per-atom head MLP and per-graph head MLP — both 2-layer SiLU MLPs
        # with the intermediate width equal to ``hidden_dim``, matching the
        # upstream ``node_dec`` + ``graph_dec`` structure.  The graph head
        # emits ``num_targets`` outputs instead of a single QM9 target.
        self.node_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.graph_dec = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_targets),
        )
        # ``nn.Dropout(p=0)`` is mathematically a no-op so we instantiate it
        # unconditionally for symmetry with the head.
        self.head_dropout = (
            nn.Dropout(p=self.dropout_p) if self.dropout_p > 0 else nn.Identity()
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or binary classification logits."""

        x, edge_index, edge_attr, pos, graph_batch, num_graphs = self._validate_batch(
            batch
        )

        # Project canonical atom features into the invariant hidden space.
        h = self.embedding_in(x)
        coord = pos

        for gcl in self.gcl_layers:
            # ``edge_attr`` is forwarded unchanged by every EGCL layer (the
            # message lives inside ``gcl`` only) so we discard the returned
            # edge_attr handle with ``_``.
            h, coord, _ = gcl(h, edge_index, coord, edge_attr)

        # Per-atom MLP then sum pool.  The head Dropout is applied AFTER the
        # graph head to keep the invariant features regularised consistently.
        h = self.node_dec(h)
        graph_embed = global_add_pool(h, graph_batch, size=num_graphs)
        graph_embed = self.head_dropout(graph_embed)
        return self.graph_dec(graph_embed)

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Validate tensors, fetch them, and return canonical inputs."""

        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, pos, graph_batch = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(pos, Tensor)
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
        if pos.shape != (x.shape[0], 3):
            raise ValueError(
                f"batch.pos must have shape [N, 3] to match the canonical "
                f"[N, {self.atom_dim}] atom features"
            )
        if pos.dtype != torch.float32 or not torch.isfinite(pos).all():
            raise ValueError(
                "batch.pos must contain finite torch.float32 coordinates"
            )
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if any(
            value.device != x.device
            for value in (edge_index, edge_attr, pos, graph_batch)
        ):
            raise ValueError(
                "all EGNN batch tensors must be on the same device"
            )

        # The canonical graph layout stores both directions of each bond
        # with no self-loops, matching the upstream ``MPNEncoder`` assumption.
        # EGNN operates on a single homogeneous atom graph, so the standard
        # ``validate_batched_molecular_graph`` applies.
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=True,
        )
        return x, edge_index, edge_attr, pos, graph_batch, num_graphs


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["EGNN"]
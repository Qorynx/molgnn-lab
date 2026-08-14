"""MAT 2020 molecular property predictor.

Architecture-only port of the Molecule Attention Transformer from
Maziarka et al., arXiv:2002.08264v1 (Feb 2020).  The architecture adapts
Vaswani et al.'s Transformer encoder to chemical molecules by augmenting
self-attention with the molecular graph adjacency and inter-atomic
distance matrices (paper Equation 2):

    A^(i) = λ_a · ρ(Q K^T / √d_k) + λ_d · g(D) + λ_g · A

where ``ρ`` is softmax, ``g(D)`` is the upstream's default
``softmax(-D)`` distance kernel, and ``λ_a + λ_d + λ_g = 1``.

An artificial dummy node sits at row 0 of every input matrix (added by
the companion ``add_mat_inputs`` graph transform) so the model can
effectively down-weight irrelevant inputs by attending to it.  The dummy
gets ``pos = (1e6, 1e6, 1e6)`` so the distance kernel puts essentially
zero weight on the dummy in the attention kernel.

The clean reference is ``tmp/MAT/MAT-master/src/transformer.py``;
``tmp/MAT/MAT-master/src/featurization/data_utils.py`` documents the
featurisation and the dummy-node convention.

.. warning::

   MAT's N-body attention operates on per-graph dense matrices.  This
   port routes that through ``torch_geometric.utils.to_dense_batch`` /
   ``to_dense_adj`` rather than carrying custom ``__inc__`` plumbing on
   ``MolecularData`` — the lab's cleaner approach, even though the
   upstream keeps everything in numpy with hand-rolled padding.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, to_dense_batch

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import (
    EncoderLayer,
    LayerNorm,
    MoleculeMultiHeadAttention,
    PositionwiseFeedForward,
    _AGGREGATION_MODES,
)


def _clones(module: EncoderLayer, n: int) -> nn.ModuleList:
    """Produce ``N`` deep copies of ``module`` (matches the upstream ``clones`` helper)."""
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))


class Embeddings(nn.Module):
    """Project canonical 153-dim atom features into ``d_model`` and dropout.

    Mirrors the upstream ``transformer.Embeddings``: a single linear
    ``d_atom → d_model`` followed by dropout.  We use the canonical
    153-dim featurizer (per the lab's benchmark contract) instead of the
    paper's custom 26-dim featurization.
    """

    def __init__(self, d_model: int, d_atom: int, dropout: float) -> None:
        super().__init__()
        self.lut = nn.Linear(d_atom, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.lut(x))


class Generator(nn.Module):
    """Aggregate per-atom features into one prediction per graph.

    Matches the upstream ``transformer.Generator``: aggregate by ``mean``,
    ``sum``, or pull the dummy atom's own features (``dummy_node``); then
    project to ``n_output`` via a single linear (or a 2-layer MLP with
    leaky-relu and LayerNorm when ``n_generator_layers > 1``).
    """

    def __init__(
        self,
        d_model: int,
        aggregation_type: str = "mean",
        n_output: int = 1,
        n_generator_layers: int = 1,
        leaky_relu_slope: float = 0.01,
        dropout: float = 0.0,
        scale_norm: bool = False,
    ) -> None:
        super().__init__()
        if aggregation_type not in _AGGREGATION_MODES:
            raise ValueError(
                f"aggregation_type must be one of {_AGGREGATION_MODES}; "
                f"got {aggregation_type!r}"
            )
        if n_generator_layers < 1:
            raise ValueError("n_generator_layers must be >= 1")

        self.aggregation_type = aggregation_type

        if n_generator_layers == 1:
            self.proj = nn.Linear(d_model, n_output)
        else:
            layers: list[nn.Module] = []
            for _ in range(n_generator_layers - 1):
                layers.append(nn.Linear(d_model, d_model))
                layers.append(nn.LeakyReLU(leaky_relu_slope))
                if scale_norm:
                    layers.append(ScaleNorm(d_model))
                else:
                    layers.append(LayerNorm(d_model))
                layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(d_model, n_output))
            self.proj = nn.Sequential(*layers)

    def forward(
        self,
        x_dense: Tensor,
        mask: Tensor,
        num_graphs: int,
    ) -> Tensor:
        """Aggregate per-atom features into ``(num_graphs, n_output)``.

        ``x_dense`` carries the per-atom features INCLUDING the dummy atom at
        index 0; ``mask`` is ``(num_graphs, max_atoms)`` with ``True`` at
        real atoms and ``False`` at padding positions.
        """

        # Mask out padding atoms before aggregation so they contribute zero.
        x_masked = x_dense * mask.unsqueeze(-1).to(x_dense.dtype)
        if self.aggregation_type == "mean":
            pooled = x_masked.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1).to(
                x_dense.dtype
            )
        elif self.aggregation_type == "sum":
            pooled = x_masked.sum(dim=1)
        else:
            # ``dummy_node`` mode: each molecule's prediction reads its dummy
            # atom's own features (the model can use the dummy as a "summary"
            # channel — the paper notes this in Section 4.4).
            pooled = x_dense[:, 0]
        return self.proj(pooled)


class ScaleNorm(nn.Module):
    """Mirrors the upstream ``transformer.ScaleNorm`` for the deep generator."""

    def __init__(self, scale: float, eps: float = 1.0e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale) ** 0.5))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        norm = self.scale / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm


class MAT(BaseMolecularModel):
    """Molecule Attention Transformer (Maziarka et al., 2020).

    Pipeline (paper Figure 1):
      1. ``Embeddings`` projects the canonical 153-dim atom features
         (including the dummy atom at index 0 added by
         ``add_mat_inputs``) into ``d_model``.
      2. ``N`` pre-norm ``EncoderLayer`` blocks, each running the
         molecule-modified multi-head self-attention (Eq. 2) followed by
         a position-wise feed-forward.
      3. A final ``LayerNorm`` after the encoder stack.
      4. ``Generator`` aggregates the per-atom features into one
         graph-level prediction via mean / sum / dummy-readout.

    Adjacency and distance matrices are derived at runtime from
    ``edge_index`` (via :func:`torch_geometric.utils.to_dense_adj`) and
    ``pos`` (via :func:`torch_geometric.utils.to_dense_batch` + pairwise
    distances), so the model does not need any new
    ``MolecularData.__inc__`` plumbing.
    """

    required_batch_fields = ("x", "edge_index", "edge_attr", "pos", "batch")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        d_model: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        lambda_attention: float = 0.3,
        lambda_distance: float = 0.3,
        trainable_lambda: bool = False,
        distance_matrix_kernel: str = "softmax",
        aggregation_type: str = "mean",
        n_generator_layers: int = 1,
    ) -> None:
        super().__init__()

        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (num_targets, "num_targets"),
            (d_model, "d_model"),
            (num_layers, "num_layers"),
            (num_heads, "num_heads"),
            (n_generator_layers, "n_generator_layers"),
        ):
            _positive_int(value, name)
        if dim_feedforward is None:
            dim_feedforward = d_model
        _positive_int(dim_feedforward, "dim_feedforward")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

        # Per-task injected dimensions — never let YAML override these (the
        # registry raises ``RegistryError`` if a YAML parameter shadows them).
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.num_targets = num_targets
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout_p = float(dropout)
        self.lambda_attention = lambda_attention
        self.lambda_distance = lambda_distance
        self.trainable_lambda = trainable_lambda
        self.distance_matrix_kernel = distance_matrix_kernel
        self.aggregation_type = aggregation_type
        self.n_generator_layers = n_generator_layers

        self.embeddings = Embeddings(d_model, atom_dim, dropout)
        self.attn = MoleculeMultiHeadAttention(
            h=num_heads,
            d_model=d_model,
            dropout=dropout,
            lambda_attention=lambda_attention,
            lambda_distance=lambda_distance,
            trainable_lambda=trainable_lambda,
            distance_matrix_kernel=distance_matrix_kernel,
        )
        self.feed_forward = PositionwiseFeedForward(
            d_model=d_model,
            n_dense=2,
            dropout=dropout,
            leaky_relu_slope=0.0,
            output_activation="relu",
        )
        self.encoder = _MoleculeTransformerEncoder(
            EncoderLayer(d_model, self.attn, self.feed_forward, dropout),
            num_layers,
        )
        self.generator = Generator(
            d_model=d_model,
            aggregation_type=aggregation_type,
            n_output=num_targets,
            n_generator_layers=n_generator_layers,
            leaky_relu_slope=0.0,
            dropout=0.0,
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or binary classification logits."""

        x, edge_index, edge_attr, pos, graph_batch, num_graphs = self._validate_batch(
            batch
        )

        # Dense-batch the per-atom features and coordinates.  ``mask`` is
        # ``True`` at real positions (including the dummy atom at index 0 of
        # every graph).
        x_dense, mask = to_dense_batch(x, graph_batch)
        pos_dense, _ = to_dense_batch(pos, graph_batch)
        max_atoms = x_dense.size(1)

        # Pairwise distances per graph (set cross-padded entries to 1e6 so
        # the distance kernel ignores them, matching the upstream's
        # ``1e6``-padding convention).
        distances_dense = torch.cdist(pos_dense, pos_dense)
        distances_dense = distances_dense.masked_fill(
            ~(mask.unsqueeze(1) * mask.unsqueeze(2)), 1.0e6
        )

        # Sparse → dense adjacency with one self-loop per atom.  The
        # canonical layout stores ``edge_index`` without self-loops; we add
        # them here (``np.eye`` equivalent in the upstream).  Padding rows /
        # columns are zeroed by the mask product.
        adj_dense = to_dense_adj(edge_index, batch=graph_batch, max_num_nodes=max_atoms)
        adj_dense = adj_dense + torch.eye(
            max_atoms, device=adj_dense.device
        ).unsqueeze(0)
        adj_dense = adj_dense * mask.unsqueeze(1).to(adj_dense.dtype) * mask.unsqueeze(2).to(
            adj_dense.dtype
        )

        # Embed (linear + dropout applied to the last dim; works for both
        # sparse and dense shapes — we pass the dense tensor here so the
        # downstream attention layers see a (B, N, d_model) input).
        h = self.embeddings.lut(x_dense)
        h = self.embeddings.dropout(h)

        # Encoder: dense (B, max_N, d_model) → dense (B, max_N, d_model).
        h = self.encoder(h, adj_dense, distances_dense, mask)

        # Generator: dense → (num_graphs, n_output).
        return self.generator(h, mask, num_graphs)

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
        if pos.shape[0] != x.shape[0]:
            raise ValueError(
                f"batch.pos must have shape [{x.shape[0]}, 3] to match x; "
                f"got shape {tuple(pos.shape)}"
            )
        if pos.shape[1] != 3:
            raise ValueError("batch.pos must be a 3-D coordinate tensor")
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
                "all MAT batch tensors must be on the same device"
            )

        # The canonical graph layout stores both directions of each bond
        # with no self-loops, matching the upstream ``MPNEncoder`` assumption.
        # MAT operates on a single homogeneous atom graph, so the standard
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


class _MoleculeTransformerEncoder(nn.Module):
    """``N`` pre-norm ``EncoderLayer`` blocks + a final ``LayerNorm``.

    Mirrors the upstream ``transformer.Encoder``: ``N`` deep-copied
    ``EncoderLayer`` instances (each with its own parameters) followed by
    a final ``LayerNorm``.
    """

    def __init__(self, layer: EncoderLayer, n: int) -> None:
        super().__init__()
        _positive_int(n, "n")
        self.layers = _clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(
        self,
        x: Tensor,
        adj_matrix: Tensor,
        distances_matrix: Tensor,
        mask: Tensor,
    ) -> Tensor:
        for layer in self.layers:
            x = layer(x, adj_matrix, distances_matrix, mask)
        return self.norm(x)


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["MAT", "Embeddings", "Generator", "_AGGREGATION_MODES"]
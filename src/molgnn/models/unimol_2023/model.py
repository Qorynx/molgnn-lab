"""Uni-Mol (Zhou et al., ICLR 2023) supervised fine-tuning model.

3-D Transformer with additive pair bias from Gaussian RBF distances
(paper Eq. 1-2): a pre-LN encoder whose self-attention logits are biased by
``q_ij``, a per-head projection of Gaussian RBF kernels over pairwise
3-D distances.  A ``[CLS]`` atom (prepended by ``add_unimol_inputs``) is
read out at index 0 and fed to a dropout + Linear head producing raw logits.

Documented deviations from the original Uni-Mol (see the package-level
docstring in ``unimol_2023/__init__.py`` for the full list):

- **Pretraining is not ported.**  ``MaskLMHead``, ``DistanceHead``, and
  the SE(3)-equivariant coordinate head that train on 209 M PubChem 3-D
  conformers are intentionally absent.  What ships here is a
  randomly-initialised dense Transformer with pair bias — by far the
  largest deviation from the paper.
- **Atom-type vocabulary is collapsed.**  The upstream 30-entry atom
  embedding and 900-entry (30×30) pair-type embedding are replaced by
  a continuous ``nn.Linear(153, embed_dim)`` and a single-edge-type
  Gaussian.  See :class:`GaussianLayer` for the consequence.
- **The backbone consumes only ``x``, ``pos``, and ``batch``** — a
  dense Transformer that never reads ``edge_index`` / ``edge_attr``
  (those fields are still required so PyG batching works).
- **Head output convention** differs: ``bce_with_logits`` on
  ``[N, num_targets]`` (lab) vs. cross-entropy on ``[N, 2]`` with a
  tanh-pooler (upstream).  Same semantics, different loss contract.

.. warning::

   BACE / BBBP-class downstream numbers are a lower bound on Uni-Mol's
   true capability because we synthesise 3-D conformers via RDKit's
   ETKDG + MMFF, whereas the paper trained on pre-computed PubChem 3-D
   conformers.  Numerical verification against the paper's equations
   is the right correctness check.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import GaussianLayer, LinearHead, NonLinearHead, TransformerEncoderWithPair


class UniMol(BaseMolecularModel):
    """Uni-Mol dense 3-D Transformer with [CLS]-pooled supervised head.

    Pipeline:
      1. ``atom_proj`` projects the canonical 153-dim atom features
         (including the ``[CLS]`` atom at index 0 added by
         ``add_unimol_inputs``) into ``embed_dim``.
      2. ``gbf`` + ``gbf_proj`` turn pairwise 3-D distances into a per-head
         additive attention bias ``[bsz*num_heads, N, N]``.
      3. ``encoder`` runs ``num_layers`` pre-LN Transformer layers with the
         pair bias added to the attention logits.
      4. ``head`` maps the ``[CLS]`` representation to ``[num_graphs,
         num_targets]`` raw logits (no Sigmoid — paired with
         ``bce_with_logits``).
    """

    required_batch_fields = ("x", "edge_index", "edge_attr", "batch", "pos")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        embed_dim: int = 64,
        ffn_embed_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 2,
        num_gaussian_kernels: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        max_seq_len: int = 512,
        pooler_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (num_targets, "num_targets"),
            (embed_dim, "embed_dim"),
            (ffn_embed_dim, "ffn_embed_dim"),
            (num_layers, "num_layers"),
            (num_heads, "num_heads"),
            (num_gaussian_kernels, "num_gaussian_kernels"),
        ):
            _positive_int(value, name)
        if embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_gaussian_kernels < num_heads:
            raise ValueError("num_gaussian_kernels must be >= num_heads")
        for value, name in (
            (dropout, "dropout"),
            (attention_dropout, "attention_dropout"),
            (activation_dropout, "activation_dropout"),
            (pooler_dropout, "pooler_dropout"),
        ):
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        _positive_int(max_seq_len, "max_seq_len")

        # Per-task injected dimensions — never let YAML override these (the
        # registry raises ``RegistryError`` if a YAML parameter shadows them).
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.num_targets = num_targets
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_gaussian_kernels = num_gaussian_kernels
        self.dropout = float(dropout)
        self.attention_dropout = float(attention_dropout)
        self.activation_dropout = float(activation_dropout)
        self.max_seq_len = max_seq_len
        self.pooler_dropout = float(pooler_dropout)

        # Replaces the upstream ``nn.Embedding(30, embed_dim)`` token table:
        # the lab's canonical 153-dim atom features are continuous, so a
        # linear projection is the natural analogue.  The upstream does NOT
        # apply a T5-style ``embed_dim ** -0.5`` input scaling — the only
        # scaling lives inside the attention (``head_dim ** -0.5``), which
        # ``nn.MultiheadAttention`` already applies — so none is added here.
        self.atom_proj = nn.Linear(atom_dim, embed_dim)
        # Single edge type (no atom-type vocabulary in the lab).
        self.gbf = GaussianLayer(num_gaussian_kernels)
        self.gbf_proj = NonLinearHead(num_gaussian_kernels, num_heads)
        self.encoder = TransformerEncoderWithPair(
            embed_dim,
            ffn_embed_dim,
            num_layers,
            num_heads,
            dropout,
            attention_dropout,
            activation_dropout,
        )
        # ``head_dropout`` mirrors the upstream pooler-dropout attribute;
        # ``head`` applies it internally before the final Linear.
        self.head_dropout = nn.Dropout(pooler_dropout)
        self.head = LinearHead(embed_dim, num_targets, dropout_rate=pooler_dropout)

    def _get_dist_features(
        self, pos: Tensor, bsz: int, N: int, num_heads: int
    ) -> Tensor:
        """Turn dense 3-D coordinates into a per-head additive attention bias.

        ``pos`` is ``[bsz, N, 3]``; returns ``[bsz*num_heads, N, N]`` — the
        layout ``nn.MultiheadAttention`` expects for a 3-D float ``attn_mask``.
        """

        pos_flat = pos.reshape(bsz, N, 3)
        # Pairwise euclidean distances: [bsz, N, N].
        dist = torch.cdist(pos_flat, pos_flat)
        # Gaussian RBF over the collapsed single-edge-type axis: [bsz, N, N, K].
        edge_feat = self.gbf(dist.unsqueeze(-1))
        # Project to one value per attention head: [bsz, N, N, heads].
        edge_feat = self.gbf_proj(edge_feat)
        # Reshape to the attention-bias layout: [bsz*heads, N, N].
        return edge_feat.permute(0, 3, 1, 2).reshape(bsz * num_heads, N, N)

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or binary classification logits."""

        x, edge_index, edge_attr, graph_batch, pos, num_graphs = self._validate_batch(
            batch
        )

        # Dense-batch the per-atom features and coordinates.  ``mask`` is
        # ``True`` at real positions (including the [CLS] atom at index 0 of
        # every graph); padding rows are zero-filled.
        x_dense, mask = to_dense_batch(x, batch=graph_batch)
        pos_dense, _ = to_dense_batch(
            pos, batch=graph_batch, max_num_nodes=x_dense.shape[1]
        )
        bsz, N, _ = x_dense.shape
        if bsz != num_graphs:
            raise ValueError(
                f"dense batch size {bsz} does not match validated graph count {num_graphs}"
            )

        # Atom projection: [bsz, N, embed_dim].
        h = self.atom_proj(x_dense)
        # Pair bias from 3-D distances: [bsz*heads, N, N].
        attn_bias = self._get_dist_features(pos_dense, bsz, N, self.num_heads)
        # Encoder: [bsz, N, embed_dim].  The updated pair representation is
        # unused in the supervised fine-tuning path.
        h, _ = self.encoder(h, attn_bias)
        # [CLS] token at index 0 of every graph.
        cls_repr = h[:, 0, :]
        return self.head(cls_repr)

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
        """Validate tensors, fetch them, and return canonical inputs."""

        names = self.required_batch_fields
        values = tuple(getattr(batch, name, None) for name in names)
        if not all(isinstance(value, Tensor) for value in values):
            raise ValueError(f"batch must provide {', '.join(names)} tensors")
        x, edge_index, edge_attr, graph_batch, pos = values
        assert isinstance(x, Tensor)
        assert isinstance(edge_index, Tensor)
        assert isinstance(edge_attr, Tensor)
        assert isinstance(graph_batch, Tensor)
        assert isinstance(pos, Tensor)

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
            raise ValueError("all Uni-Mol batch tensors must be on the same device")

        # Uni-Mol is a dense Transformer: no message passing adds self-loops,
        # so ``forbid_self_loops=False`` is safe (the transform's two dummy
        # [CLS] self-loops are accepted).
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=False,
        )
        return x, edge_index, edge_attr, graph_batch, pos, num_graphs


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["UniMol"]
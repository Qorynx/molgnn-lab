"""MolCLR (Wang et al., 2022) supervised fine-tuning backbones.

Architecture-only ports of the GIN- and GCN-based encoders MolCLR uses for
its supervised fine-tuning experiments (Tables 1 and 2 of the paper).
Pretraining, the NT-Xent contrastive loss, and the graph augmentations
(atom masking, bond deletion, subgraph removal) are intentionally not
ported: per lab policy the models run on the canonical 153-dim atom /
14-dim bond featurization and are trained end-to-end with a supervised
objective only.

Deviations from the upstream ``ginet_finetune.py`` / ``gcn_finetune.py``
that we want future maintainers to see without re-reading the diff:

1. **Atom / bond embedding**. Upstream sums two ``nn.Embedding`` tables —
   ``nn.Embedding(num_atom_type=119)`` + ``nn.Embedding(num_chirality_tag=3)``
   for atoms, and ``nn.Embedding(num_bond_type=5)`` +
   ``nn.Embedding(num_bond_direction=3)`` for bonds. The lab's canonical
   continuous featurizer produces ``float32 [N, 153]`` atoms and
   ``float32 [E, 14]`` bonds, so we substitute a single ``nn.Linear`` for
   each side (``self.atom_proj`` here, ``edge_embedding`` inside the conv
   layer). The aggregation-side wiring (additive ``x_j + edge_attr``,
   Hu et al. modification) is unchanged.

2. **Self-loop edge rows**. Upstream adds self-loops and stamps an integer
   ``[4, 0]`` row onto the bond feature column. Our continuous bond
   representation has no "self-loop type" convention, so the added rows are
   zero-filled; ``Linear(14, emb_dim)(zeros) = zeros`` does not pollute the
   message with a fictitious bond signal.

3. **GCN symmetric normalization**. Upstream's ``GCNConv`` consumes the
   degree weight through the optional ``torch_sparse.matmul`` fast path
   in ``message_and_aggregate``; the pure-PyG path silently drops it
   (binds the result to ``__``). The lab omits ``torch_sparse`` and so we
   re-route the weight through ``message`` as
   ``edge_weight * (x_j + edge_attr)`` to keep the *intended* algorithm
   rather than the py-only artefact.

4. **Prediction head**. Upstream fine-tune: ``Linear(feat_dim, feat_dim//2)
   → Softplus → Dropout → Linear(feat_dim//2, out_dim)`` with
   ``out_dim=2`` for cross-entropy classification. Lab port: ``Linear
   (feat_dim, feat_dim) → ReLU → Dropout → Linear(feat_dim, num_targets)``
   with ``num_targets=1`` for binary classification paired with
   ``bce_with_logits``. The first projection keeps ``feat_dim`` wide
   instead of halving because the lab's BCE-with-logits head needs a
   single logit per (graph, target) slot, not a 2-way softmax.

5. **Output**.  ``forward`` returns raw ``[num_graphs, num_targets]``
   logits; no Sigmoid is applied (BCE-with-logits expects logits).
   Upstream's fine-tune returns the same shape for ``out_dim`` predictions
   and pairs with ``nn.CrossEntropyLoss`` / ``nn.L1Loss`` / ``nn.MSELoss``
   depending on task type.

6. **No pretraining artefacts**. ``molclr.py``'s ``utils/nt_xent.NTXentLoss``,
   the three augmentations (``dataset.py``, ``dataset_subgraph.py``,
   ``dataset_mix.py``), and the dual-encoder contrastive objective are
   not ported. The frame here is the *fine-tune* branch only.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from ..base import BaseMolecularModel
from ..contracts import validate_batched_molecular_graph
from .layers import GCNConv, GINEConv

_POOL_FUNCTIONS: dict[str, Callable[..., Tensor]] = {
    "mean": global_mean_pool,
    "add": global_add_pool,
    "max": global_max_pool,
}
_POOLS = ("mean", "add", "max")


class _MolCLRBase(BaseMolecularModel, ABC):
    """Shared wrapper around the MolCLR encoder + prediction head.

    The two public variants (``MolCLRGIN``, ``MolCLRGCN``) differ only in the
    message-passing layer they stack; everything else — atom projection, five
    conv blocks, inner-loop dropout schedule, pooling, and prediction head —
    is identical and lives here.
    """

    required_batch_fields = ("x", "edge_index", "edge_attr", "batch")

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        num_targets: int,
        emb_dim: int = 300,
        feat_dim: int = 256,
        num_layer: int = 5,
        drop_ratio: float = 0.0,
        pool: str = "mean",
    ) -> None:
        super().__init__()

        for value, name in (
            (atom_dim, "atom_dim"),
            (bond_dim, "bond_dim"),
            (num_targets, "num_targets"),
            (emb_dim, "emb_dim"),
            (feat_dim, "feat_dim"),
            (num_layer, "num_layer"),
        ):
            _positive_int(value, name)
        if pool not in _POOLS:
            raise ValueError(f"pool must be one of {'|'.join(_POOLS)}; got {pool!r}")
        if (
            isinstance(drop_ratio, bool)
            or not isinstance(drop_ratio, (float, int))
            or not 0 <= drop_ratio < 1
        ):
            raise ValueError("drop_ratio must be in [0, 1)")

        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.num_targets = num_targets
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.num_layer = num_layer
        self.drop_ratio = float(drop_ratio)

        # Upstream encodes atom types / bond types into small embedding
        # tables; the lab's canonical continuous features use linear
        # projections instead.
        self.atom_proj = nn.Linear(atom_dim, emb_dim)
        self.convs = nn.ModuleList(
            [self._conv_class(emb_dim, self.bond_dim) for _ in range(num_layer)]
        )
        self.batch_norms = nn.ModuleList(
            [nn.BatchNorm1d(emb_dim) for _ in range(num_layer)]
        )
        self.pool = _POOL_FUNCTIONS[pool]
        # Readout projection after pooling, then the supervised head.  Raw
        # logits come out of the head: binary classification pairs with
        # BCE-with-logits, so no Sigmoid is applied.
        self.feat_lin = nn.Linear(emb_dim, feat_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop_ratio),
            nn.Linear(feat_dim, num_targets),
        )

    def forward(self, batch: Batch) -> Tensor:
        """Return raw regression values or binary classification logits."""

        x, edge_index, edge_attr, graph_batch, num_graphs = self._validate_batch(
            batch
        )

        h = self.atom_proj(x)
        h = self._apply_message_passing(h, edge_index, edge_attr)
        h = self.pool(h, graph_batch)
        h = self.feat_lin(h)
        return self.pred_head(h)

    def _apply_message_passing(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        """Run the shared conv -> batch-norm -> dropout loop.

        Mirrors the upstream fine-tuning schedule: the last layer skips ReLU
        (dropout only), every earlier layer applies ReLU before dropout.
        """
        h = x
        for layer in range(self.num_layer):
            h = self.convs[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
        return h

    def _validate_batch(
        self, batch: Batch
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, int]:
        """Validate the homogeneous mol-graph batch tensors."""

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
            raise ValueError(f"batch.x must have shape [N, {self.atom_dim}] with N >= 1")
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
            raise ValueError("batch.edge_attr must contain finite torch.float32 values")
        if edge_index.shape[1] and (
            edge_index.min().item() < 0 or edge_index.max().item() >= x.shape[0]
        ):
            raise ValueError("batch.edge_index contains an invalid node index")
        if graph_batch.shape != (x.shape[0],) or graph_batch.dtype != torch.long:
            raise ValueError("batch.batch must have shape [N] and dtype torch.long")
        if any(value.device != x.device for value in values):
            raise ValueError("all MolCLR batch tensors must be on the same device")

        # The conv layers inject self-loops internally, so input self-loops are
        # neither expected nor forbidden here.
        num_graphs = validate_batched_molecular_graph(
            edge_index,
            graph_batch,
            num_nodes=x.shape[0],
            device=x.device,
            edge_field="edge_index",
            forbid_self_loops=False,
        )
        return x, edge_index, edge_attr, graph_batch, num_graphs


class MolCLRGIN(_MolCLRBase):
    """MolCLR's GIN backbone (Hu et al. GINEConv) for supervised fine-tuning."""

    _conv_class = GINEConv


class MolCLRGCN(_MolCLRBase):
    """MolCLR's GCN backbone (custom normalized GCNConv) for supervised fine-tuning."""

    _conv_class = GCNConv


def _positive_int(value: int, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")


__all__ = ["MolCLRGIN", "MolCLRGCN"]

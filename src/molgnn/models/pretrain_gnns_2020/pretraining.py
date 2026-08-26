"""Supervised graph-level pretraining stage for Pretrain-GNNs.

Replicates ``OFFICIAL CODE chem/pretrain_supervised.py``: the node-level
encoder is continued on ChEMBL's 1310 binary assays with labels in
``{-1, 0, +1}`` where ``0`` means missing. The loss is masked BCE-with-logits
divided by the number of valid labels. Only the encoder is exported after
pretraining; the 1310-task head stays behind.

Stages run strictly sequentially (node-level objective first, supervised
second); combining losses is deliberately not provided.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .context import ContextPredObjective
from .layers import MolecularGNN, jk_output_dim
from .masking import AttributeMaskingObjective


def supervised_pretraining_loss(
    predictions: Tensor, labels: Tensor
) -> dict[str, Tensor | None] | None:
    """Official masked multi-task BCE loss.

    ``labels`` uses ``-1/0/+1`` with ``0`` marking missing entries. Returns
    ``None`` when the batch carries no valid label at all so callers can skip
    the step explicitly instead of producing NaN.
    """

    if predictions.shape != labels.shape:
        raise ValueError("predictions and labels must share the [B, T] shape")
    # Official validity mask: is_valid = y ** 2 (nonzero for -1/+1).
    is_valid = labels.square().bool()
    valid_count = int(is_valid.sum().item())
    if valid_count == 0:
        return {"loss": None, "valid_labels": torch.zeros((), dtype=torch.long)}
    targets = (labels + 1) / 2
    loss_mat = F.binary_cross_entropy_with_logits(
        predictions.double(), targets.double(), reduction="none"
    )
    loss = torch.where(
        is_valid, loss_mat, torch.zeros_like(loss_mat)
    ).sum() / is_valid.sum()
    return {"loss": loss.to(predictions.dtype), "valid_labels": is_valid.sum()}


class SupervisedPretrainingHead(nn.Module):
    """Linear head over mean-pooled encoder states (official contract).

    ``input_dim`` must match the encoder's node-representation width
    (``(num_layer + 1) * emb_dim`` for ``JK=concat``, ``emb_dim`` otherwise).
    """

    def __init__(self, input_dim: int, num_tasks: int = 1310) -> None:
        super().__init__()
        if input_dim < 1 or num_tasks < 1:
            raise ValueError("input_dim and num_tasks must be positive")
        self.linear = nn.Linear(input_dim, num_tasks)

    def forward(self, pooled: Tensor) -> Tensor:
        return self.linear(pooled)


def sequential_stage_order() -> tuple[str, str]:
    """The official two-stage sequence (node level first)."""

    return ("node_level", "supervised")


def _as_list(value):
    return value if isinstance(value, Sequence) and not isinstance(value, str) else [value]


def _summarize(outputs: dict[str, object]) -> dict[str, object]:
    return {
        name: float(value.item())
        for name, value in outputs.items()
        if isinstance(value, Tensor) and value.dim() == 0
    }


class PretrainingLifecycle(nn.Module):
    """Runs the three official pretraining stages on prepared batches.

    Owns the main GIN encoder plus stage-local modules discarded after
    pretraining: attribute-masking linear heads, an auxiliary context GNN,
    and the supervised 1310-task head.  Only the main encoder is exported.

    Stages are sequential: a node-level objective (Attribute Masking or
    Context Prediction) must run before the supervised stage.
    """

    NODE_LEVEL_STAGES = ("masking", "contextpred")

    def __init__(
        self,
        *,
        num_layer: int = 5,
        emb_dim: int = 300,
        JK: str = "last",
        drop_ratio: float = 0.0,
        graph_pooling: str = "mean",
        context_layers: int = 3,
        neg_samples: int = 1,
        context_pooling: str = "mean",
        num_supervised_tasks: int = 1310,
        metadata: dict[str, object] | None = None,
    ) -> None:
        super().__init__()
        self.encoder = MolecularGNN(num_layer, emb_dim, JK=JK, drop_ratio=drop_ratio)
        encoder_dim = jk_output_dim(num_layer, emb_dim, JK)
        self.mask_objective = AttributeMaskingObjective(encoder_dim)
        self.context_gnn = MolecularGNN(
            context_layers, emb_dim, JK=JK, drop_ratio=drop_ratio
        )
        context_dim = jk_output_dim(context_layers, emb_dim, JK)
        self.context_projection = (
            nn.Identity()
            if context_dim == encoder_dim
            else nn.Linear(context_dim, encoder_dim)
        )
        self.context_objective = ContextPredObjective(
            neg_samples=neg_samples, context_pooling=context_pooling
        )
        self.supervised_head = SupervisedPretrainingHead(
            encoder_dim, num_tasks=num_supervised_tasks
        )
        if graph_pooling not in {"sum", "mean", "max"}:
            raise ValueError("graph_pooling must be sum, mean, or max")
        self.graph_pooling = graph_pooling
        self._stage: str | None = None
        self._metadata = dict(metadata or {})

    # --- public properties ---

    @property
    def stage(self) -> str | None:
        """Current stage: ``None``, ``"masking"``, ``"contextpred"``, or ``"supervised"``."""
        return self._stage

    @property
    def metadata(self) -> dict[str, object]:
        """Read-only copy of the lifecycle metadata."""
        return dict(self._metadata)

    # --- stage sequencing ---

    def _enter_node_level(self, objective: str) -> None:
        if self._stage is None:
            self._stage = objective
            return
        if self._stage != objective:
            raise ValueError(
                f"node-level objective {objective!r} cannot run after "
                f"stage {self._stage!r}"
            )

    def _enter_supervised(self) -> None:
        if self._stage is None:
            raise ValueError(
                "supervised stage requires a completed node-level stage "
                "(masking or contextpred)"
            )
        self._stage = "supervised"

    # --- graph pooling ---

    def _pool(self, node_features: Tensor, batch: Tensor) -> Tensor:
        num_graphs = int(batch.max().item()) + 1
        if self.graph_pooling == "sum":
            pooled = node_features.new_zeros(
                (num_graphs, node_features.shape[1])
            )
            return pooled.index_add_(0, batch, node_features)
        if self.graph_pooling == "max":
            pooled = node_features.new_full(
                (num_graphs, node_features.shape[1]), float("-inf")
            )
            pooled.scatter_reduce_(
                0,
                batch.unsqueeze(-1).expand_as(node_features),
                node_features,
                reduce="amax",
                include_self=True,
            )
            return pooled
        # mean
        pooled = node_features.new_zeros((num_graphs, node_features.shape[1]))
        pooled.index_add_(0, batch, node_features)
        counts = (
            torch.bincount(batch, minlength=num_graphs)
            .unsqueeze(-1)
            .to(node_features.dtype)
        )
        return pooled / counts.clamp_min(1.0)

    # --- step helpers ---

    @staticmethod
    def _zero_grads(optimizers) -> None:
        for opt in _as_list(optimizers):
            opt.zero_grad(set_to_none=True)

    @staticmethod
    def _step(optimizers, *, step: bool = True) -> None:
        if step:
            for opt in _as_list(optimizers):
                opt.step()

    # --- public stage steps ---

    def run_masking(
        self,
        mask_batch: object,
        optimizers: nn.Optimizer | Sequence[nn.Optimizer],
        *,
        step: bool = True,
    ) -> dict[str, object]:
        """One Attribute Masking optimizer step.

        With ``step=True`` the standard ``zero_grad -> backward -> step``
        cycle runs; with ``step=False`` gradients stay populated for
        inspection.  Returns a dict of scalar losses.  Raises if the stage is
        already past the node-level phase.
        """
        self._enter_node_level("masking")
        if step:
            self._zero_grads(optimizers)
        b = mask_batch.batch_2d  # type: ignore[attr-defined]
        node_rep = self.encoder(
            b.pretrain_gnns_atom_attr, b.edge_index, b.pretrain_gnns_bond_attr
        )
        outputs = self.mask_objective(node_rep, mask_batch)
        outputs["loss"].backward()
        self._step(optimizers, step=step)
        return _summarize(outputs)

    def run_contextpred(
        self,
        pair_batch: object,
        optimizers: nn.Optimizer | Sequence[nn.Optimizer],
        *,
        step: bool = True,
    ) -> dict[str, object]:
        """One Context Prediction optimizer step.

        Both the main encoder and the auxiliary context GNN receive gradients.
        Raises if the stage is already past the node-level phase.
        """
        self._enter_node_level("contextpred")
        if step:
            self._zero_grads(optimizers)
        sub = pair_batch.batch_substruct  # type: ignore[attr-defined]
        ctx = pair_batch.batch_context  # type: ignore[attr-defined]
        substruct_rep = self.encoder(
            sub.pretrain_gnns_atom_attr, sub.edge_index, sub.pretrain_gnns_bond_attr
        )[pair_batch.center_substruct_idx]  # type: ignore[attr-defined]
        context_node_rep = self.context_gnn(
            ctx.pretrain_gnns_atom_attr, ctx.edge_index, ctx.pretrain_gnns_bond_attr
        )
        context_node_rep = self.context_projection(context_node_rep)
        outputs = self.context_objective(substruct_rep, pair_batch, context_node_rep)
        outputs["loss"].backward()
        self._step(optimizers, step=step)
        return _summarize(outputs)

    def run_supervised(
        self,
        batch: object,
        labels: Tensor,
        optimizer: nn.Optimizer,
        *,
        step: bool = True,
    ) -> dict[str, object]:
        """One supervised 1310-task masked BCE optimizer step.

        Returns ``{"loss": None, "valid_labels": 0, "skipped": True}`` when
        the batch carries no valid label, so callers can skip the step
        explicitly.  Raises if no node-level stage has been run.
        """
        self._enter_supervised()
        if step:
            self._zero_grads(optimizer)
        node_rep = self.encoder(
            batch.pretrain_gnns_atom_attr, batch.edge_index, batch.pretrain_gnns_bond_attr  # type: ignore[attr-defined]
        )
        pooled = self._pool(node_rep, batch.batch)  # type: ignore[attr-defined]
        pred = self.supervised_head(pooled)
        result = supervised_pretraining_loss(pred, labels)
        if result is None or result["loss"] is None:
            return {"loss": None, "valid_labels": 0, "skipped": True}
        result["loss"].backward()
        self._step(optimizer, step=step)
        return {
            "loss": result["loss"].item(),
            "valid_labels": int(result["valid_labels"].item()),
            "skipped": False,
        }

    # --- checkpoint / export ---

    def load_pretrained(
        self,
        *,
        variant: str | None = None,
        checkpoint_path: str | None = None,
    ) -> dict[str, object]:
        """Load a pretrained checkpoint into the main encoder.

        Accepts a named ``variant`` or an explicit ``checkpoint_path``.
        Raises when both are provided or when loading fails.  The returned
        metadata (variant, checksum, chirality adaptation) is recorded in the
        lifecycle.
        """
        from .checkpoint import load_pretrained_encoder

        meta = load_pretrained_encoder(
            self.encoder, variant=variant, checkpoint_path=checkpoint_path
        )
        self._metadata.update(meta)
        return meta

    def export_encoder(self) -> dict[str, object]:
        """Export only the main encoder state dict and stage metadata.

        The returned dict is safe to save/load regardless of the parent
        lifecycle state.
        """
        return {
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.encoder.state_dict().items()
            },
            "metadata": {**self._metadata, "stage": self._stage},
        }


__all__ = [
    "PretrainingLifecycle",
    "SupervisedPretrainingHead",
    "sequential_stage_order",
    "supervised_pretraining_loss",
]

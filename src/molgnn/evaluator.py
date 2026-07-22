"""Model-agnostic evaluation and prediction collection."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .metrics import binary_classification_metrics, regression_metrics
from .tasks import (
    BinaryClassificationTaskAdapter,
    RegressionTaskAdapter,
    TaskAdapter,
    TaskError,
)


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot produce a valid result."""


@dataclass(frozen=True)
class EvaluationResult:
    """Aggregated evaluation values and optional ordered predictions."""

    loss: float
    metrics: dict[str, float]
    sample_count: int
    valid_target_count: int
    predictions: Tensor | None = None
    targets: Tensor | None = None
    mask: Tensor | None = None
    sample_ids: Tensor | None = None
    logits: Tensor | None = None
    predicted_labels: Tensor | None = None

    @property
    def probabilities(self) -> Tensor | None:
        """Alias for classification probabilities stored in ``predictions``."""
        return self.predictions


def evaluate(
    model: nn.Module,
    loader: Iterable[object],
    task_adapter: TaskAdapter,
    device: torch.device | str,
    *,
    return_predictions: bool = False,
    target_names: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate one loader while restoring the model's previous train mode."""
    if not isinstance(model, nn.Module):
        raise EvaluationError("model must be a torch.nn.Module")
    evaluation_device = torch.device(device)
    if not isinstance(task_adapter, (RegressionTaskAdapter, BinaryClassificationTaskAdapter)):
        raise EvaluationError("unsupported task adapter for evaluator")

    was_training = model.training
    raw_predictions: list[Tensor] = []
    target_batches: list[Tensor] = []
    mask_batches: list[Tensor] = []
    sample_id_batches: list[Tensor] = []
    loss_total = 0.0
    valid_target_count = 0
    sample_count = 0
    num_targets = _adapter_num_targets(task_adapter)

    try:
        model.eval()
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                batch_to = getattr(batch, "to", None)
                if not callable(batch_to):
                    raise EvaluationError(f"batch {batch_index} does not support .to(device)")
                device_batch = batch_to(evaluation_device)
                targets = getattr(device_batch, "y", None)
                mask = getattr(device_batch, "y_mask", None)
                sample_ids = getattr(device_batch, "sample_id", None)
                if not isinstance(targets, Tensor) or not isinstance(mask, Tensor):
                    raise EvaluationError(f"batch {batch_index} is missing y/y_mask tensors")
                if not isinstance(sample_ids, Tensor):
                    raise EvaluationError(f"batch {batch_index} is missing sample_id tensor")
                predictions = model(device_batch)
                if not isinstance(predictions, Tensor):
                    raise EvaluationError(f"model output at batch {batch_index} must be a Tensor")
                try:
                    batch_loss = task_adapter.loss(predictions, targets, mask)
                except TaskError as exc:
                    raise EvaluationError(
                        f"invalid task batch at batch {batch_index}: {exc}"
                    ) from exc
                batch_count = int(mask.sum().item())
                if batch_loss is not None:
                    if not torch.isfinite(batch_loss).item():
                        raise EvaluationError(f"non-finite evaluation loss at batch {batch_index}")
                    loss_total += float(batch_loss.item()) * batch_count
                    valid_target_count += batch_count
                sample_count += int(targets.shape[0])
                num_targets = int(targets.shape[-1])
                raw_predictions.append(predictions.detach().cpu())
                target_batches.append(targets.detach().cpu())
                mask_batches.append(mask.detach().cpu())
                sample_id_batches.append(sample_ids.detach().cpu().reshape(-1))
    finally:
        model.train(was_training)

    raw = _cat_or_empty(raw_predictions, (0, num_targets))
    targets = _cat_or_empty(target_batches, (0, num_targets))
    masks = _cat_or_empty(mask_batches, (0, num_targets), dtype=torch.bool)
    sample_ids = _cat_or_empty(sample_id_batches, (0,), dtype=torch.int64)
    if isinstance(task_adapter, RegressionTaskAdapter):
        prediction_values = task_adapter.inverse_predictions(raw)
        metrics = regression_metrics(prediction_values, targets, masks, target_names)
        predicted_labels = None
    else:
        prediction_values = torch.sigmoid(raw)
        metrics = binary_classification_metrics(raw, targets, masks, target_names)
        predicted_labels = (prediction_values >= 0.5).to(torch.float32)
    loss = loss_total / valid_target_count if valid_target_count else math.nan
    if not return_predictions:
        prediction_values = None
        targets = None
        masks = None
        sample_ids = None
        raw = None
        predicted_labels = None
    return EvaluationResult(
        loss=loss,
        metrics=metrics,
        sample_count=sample_count,
        valid_target_count=valid_target_count,
        predictions=prediction_values,
        targets=targets,
        mask=masks,
        sample_ids=sample_ids,
        logits=raw,
        predicted_labels=predicted_labels,
    )


def evaluate_model(*args: object, **kwargs: object) -> EvaluationResult:
    """Compatibility alias for :func:`evaluate`."""
    return evaluate(*args, **kwargs)  # type: ignore[arg-type]


def _adapter_num_targets(adapter: TaskAdapter) -> int:
    if isinstance(adapter, RegressionTaskAdapter) and adapter.scaler is not None:
        return adapter.scaler.mean.numel()
    return 0


def _cat_or_empty(
    values: list[Tensor],
    shape: tuple[int, ...],
    dtype: torch.dtype | None = None,
) -> Tensor:
    if values:
        return torch.cat(values, dim=0)
    return torch.empty(shape, dtype=dtype or torch.float32)


__all__ = ["EvaluationError", "EvaluationResult", "evaluate", "evaluate_model"]

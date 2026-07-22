"""Stateless masked regression and binary-classification metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    r2_score,
    roc_auc_score,
)
from torch import Tensor


class MetricError(ValueError):
    """Raised when metric inputs violate the task tensor contract."""


def regression_metrics(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
    target_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute masked RMSE/MAE/R2 per target and macro aggregates."""
    valid_mask = _validate_inputs(predictions, targets, mask)
    names = _target_names(predictions.shape[-1], target_names)
    values: dict[str, list[float]] = {metric: [] for metric in ("rmse", "mae", "r2")}
    result: dict[str, float] = {}

    for target_index, name in enumerate(names):
        observed = valid_mask[:, target_index]
        y_true = targets[:, target_index][observed].detach().cpu().numpy().astype(np.float64)
        y_pred = predictions[:, target_index][observed].detach().cpu().numpy().astype(np.float64)
        if y_true.size == 0:
            task_values = {metric: math.nan for metric in values}
        else:
            errors = y_pred - y_true
            task_values = {
                "rmse": float(np.sqrt(np.mean(np.square(errors)))),
                "mae": float(np.mean(np.abs(errors))),
                "r2": float(r2_score(y_true, y_pred)) if y_true.size >= 2 else math.nan,
            }
        for metric, value in task_values.items():
            values[metric].append(value)
            result[f"{metric}/{name}"] = value

    for metric, task_values in values.items():
        result[metric] = _nanmean(task_values)
    return result


def binary_classification_metrics(
    logits: Tensor,
    targets: Tensor,
    mask: Tensor | None = None,
    target_names: Sequence[str] | None = None,
) -> dict[str, float]:
    """Compute masked ROC-AUC, PRC-AUC, accuracy and balanced accuracy.

    The first argument is expected to contain raw model logits. Probabilities
    are obtained with sigmoid internally for AUC metrics and thresholded at
    ``0.5`` for accuracy metrics.
    """
    valid_mask = _validate_inputs(logits, targets, mask)
    names = _target_names(logits.shape[-1], target_names)
    metric_names = ("roc_auc", "prc_auc", "accuracy", "balanced_accuracy")
    values: dict[str, list[float]] = {metric: [] for metric in metric_names}
    result: dict[str, float] = {}
    probabilities = torch.sigmoid(logits)
    observed_labels = targets[valid_mask]
    if observed_labels.numel() and not torch.all((observed_labels == 0) | (observed_labels == 1)):
        raise MetricError("binary classification targets must be finite 0/1 values")

    for target_index, name in enumerate(names):
        observed = valid_mask[:, target_index]
        y_true = targets[:, target_index][observed].detach().cpu().numpy().astype(np.int64)
        y_probability = (
            probabilities[:, target_index][observed].detach().cpu().numpy().astype(np.float64)
        )
        if y_true.size == 0:
            task_values = {metric: math.nan for metric in metric_names}
        else:
            labels = (y_probability >= 0.5).astype(np.int64)
            has_both_classes = np.unique(y_true).size == 2
            balanced_accuracy = (
                float(np.mean(labels == y_true))
                if not has_both_classes
                else float(balanced_accuracy_score(y_true, labels))
            )
            task_values = {
                "roc_auc": float(roc_auc_score(y_true, y_probability))
                if has_both_classes
                else math.nan,
                "prc_auc": float(average_precision_score(y_true, y_probability))
                if has_both_classes
                else math.nan,
                "accuracy": float(accuracy_score(y_true, labels)),
                "balanced_accuracy": balanced_accuracy,
            }
        for metric, value in task_values.items():
            values[metric].append(value)
            result[f"{metric}/{name}"] = value

    for metric, task_values in values.items():
        result[metric] = _nanmean(task_values)
    return result


def _validate_inputs(predictions: Tensor, targets: Tensor, mask: Tensor | None) -> Tensor:
    if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
        raise MetricError("predictions and targets must be torch.Tensor values")
    if predictions.shape != targets.shape or predictions.ndim != 2:
        raise MetricError("predictions and targets must have identical shape [N, T]")
    if not torch.is_floating_point(predictions) or not torch.is_floating_point(targets):
        raise MetricError("predictions and targets must have floating dtypes")
    if mask is None:
        valid_mask = torch.ones_like(targets, dtype=torch.bool)
    else:
        if mask.shape != targets.shape or mask.dtype != torch.bool:
            raise MetricError("mask must have shape [N, T] and dtype torch.bool")
        valid_mask = mask
    observed_targets = targets[valid_mask]
    observed_predictions = predictions[valid_mask]
    if observed_targets.numel() and (
        not torch.isfinite(observed_targets).all() or not torch.isfinite(observed_predictions).all()
    ):
        raise MetricError("observed predictions and targets must be finite")
    return valid_mask


def _target_names(num_targets: int, target_names: Sequence[str] | None) -> tuple[str, ...]:
    if target_names is None:
        return tuple(f"task_{index}" for index in range(num_targets))
    if isinstance(target_names, (str, bytes)) or len(target_names) != num_targets:
        raise MetricError("target_names must contain one name per target")
    names = tuple(str(name).strip() for name in target_names)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise MetricError("target_names must be non-empty and unique")
    return names


def _nanmean(values: Sequence[float]) -> float:
    finite_values = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else math.nan


__all__ = [
    "MetricError",
    "binary_classification_metrics",
    "regression_metrics",
]

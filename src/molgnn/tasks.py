"""Target scaling primitives shared by regression training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import functional as F

from .config import TaskConfig
from .data import validate_molecular_data
from .dataset import MolecularDataset
from .featurizer import FeatureSchema


class TaskError(ValueError):
    """Raised when target scaling inputs violate the shared task contract."""


class TaskAdapter(Protocol):
    """Minimal loss boundary consumed by the shared trainer."""

    def loss(self, predictions: Tensor, targets: Tensor, mask: Tensor | None) -> Tensor | None:
        """Return a masked scalar loss, or ``None`` for an empty mask."""


@dataclass(frozen=True)
class RegressionTaskAdapter:
    """Masked MSE/MAE adapter with optional target standardization."""

    scaler: TargetScalerState | None = None
    loss_name: str = "mse"

    def __post_init__(self) -> None:
        if self.loss_name not in {"mse", "mae"}:
            raise TaskError("regression loss_name must be 'mse' or 'mae'")

    def loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None,
    ) -> Tensor | None:
        valid_mask = _validate_task_batch(predictions, targets, mask)
        if not valid_mask.any():
            return None
        scaled_targets = (
            transform_targets(targets, self.scaler, valid_mask)
            if self.scaler is not None
            else targets
        )
        error = predictions[valid_mask] - scaled_targets[valid_mask]
        return error.abs().mean() if self.loss_name == "mae" else error.square().mean()

    def inverse_predictions(self, predictions: Tensor) -> Tensor:
        """Map model outputs back to original units when scaling is enabled."""
        if self.scaler is None:
            return predictions
        return inverse_predictions(predictions, self.scaler)


@dataclass(frozen=True)
class BinaryClassificationTaskAdapter:
    """Masked binary cross-entropy adapter operating directly on logits."""

    def loss(
        self,
        predictions: Tensor,
        targets: Tensor,
        mask: Tensor | None,
    ) -> Tensor | None:
        valid_mask = _validate_task_batch(predictions, targets, mask)
        observed_targets = targets[valid_mask]
        if observed_targets.numel() == 0:
            return None
        if not torch.isfinite(observed_targets).all() or not torch.all(
            (observed_targets == 0) | (observed_targets == 1)
        ):
            raise TaskError("binary classification targets must be finite 0/1 values")
        return F.binary_cross_entropy_with_logits(
            predictions[valid_mask], observed_targets, reduction="mean"
        )


def build_task_adapter(
    config: TaskConfig,
    scaler: TargetScalerState | None = None,
) -> TaskAdapter:
    """Build the task adapter declared by a validated :class:`TaskConfig`."""
    if config.type == "regression":
        if config.loss not in {"mse", "mae"}:
            raise TaskError("regression task requires loss='mse' or loss='mae'")
        if config.target_scaling and scaler is None:
            raise TaskError("regression target scaling requires a TargetScalerState")
        return RegressionTaskAdapter(
            scaler if config.target_scaling else None, loss_name=config.loss
        )
    if config.type == "binary_classification":
        if config.loss != "bce_with_logits":
            raise TaskError("binary_classification requires loss='bce_with_logits'")
        if config.target_scaling:
            raise TaskError("binary_classification does not support target scaling")
        return BinaryClassificationTaskAdapter()
    raise TaskError(f"unsupported task type: {config.type!r}")


@dataclass(frozen=True)
class TargetScalerState:
    """Serializable per-target standardization parameters.

    ``mean`` and ``scale`` are one-dimensional float32 tensors.  Every scale
    is strictly positive; targets with fewer than two distinct observations
    use a neutral scale of ``1.0``.
    """

    mean: Tensor
    scale: Tensor

    def __post_init__(self) -> None:
        mean = _normalise_state_tensor(self.mean, "mean")
        scale = _normalise_state_tensor(self.scale, "scale")
        if mean.shape != scale.shape:
            raise TaskError("target scaler mean and scale must have the same shape")
        if (scale <= 0).any():
            raise TaskError("target scaler scale must be strictly positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def to_dict(self) -> dict[str, list[float]]:
        """Return a JSON/YAML-safe representation of the scaler state."""
        return {
            "mean": [float(value) for value in self.mean.tolist()],
            "scale": [float(value) for value in self.scale.tolist()],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TargetScalerState:
        """Restore a state produced by :meth:`to_dict`."""
        if "mean" not in value or "scale" not in value:
            raise TaskError("serialized target scaler state requires mean and scale")
        mean = value["mean"]
        scale = value["scale"]
        if isinstance(mean, (str, bytes)) or isinstance(scale, (str, bytes)):
            raise TaskError("serialized target scaler mean and scale must be sequences")
        try:
            return cls(
                mean=torch.tensor(list(mean), dtype=torch.float32),  # type: ignore[arg-type]
                scale=torch.tensor(list(scale), dtype=torch.float32),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise TaskError(
                "serialized target scaler mean and scale must be numeric sequences"
            ) from exc


def fit_target_scaler(
    dataset: MolecularDataset,
    train_indices: Sequence[int],
    *,
    feature_schema: FeatureSchema | None = None,
    num_targets: int | None = None,
) -> TargetScalerState:
    """Fit per-target population mean/std on observed training labels only.

    Existing CSV callers may omit ``feature_schema`` and ``num_targets``;
    they are read from the legacy dataset attributes. Dataset sources that
    carry those values beside their generic sample collection pass both
    explicitly through the runner.
    """

    indices = _validate_train_indices(dataset, train_indices)
    schema = _resolve_feature_schema(dataset, feature_schema)
    target_count = _resolve_num_targets(dataset, num_targets)
    sums = torch.zeros(target_count, dtype=torch.float64)
    sum_squares = torch.zeros(target_count, dtype=torch.float64)
    counts = torch.zeros(target_count, dtype=torch.int64)

    for index in indices:
        data = dataset[index]
        validate_molecular_data(data, schema, target_count)
        values = data.y[0].to(dtype=torch.float64)
        observed = data.y_mask[0]
        sums[observed] += values[observed]
        sum_squares[observed] += values[observed] * values[observed]
        counts[observed] += 1

    mean = torch.zeros(target_count, dtype=torch.float64)
    scale = torch.ones(target_count, dtype=torch.float64)
    has_values = counts > 0
    mean[has_values] = sums[has_values] / counts[has_values]
    variance = torch.zeros(target_count, dtype=torch.float64)
    variance[has_values] = (
        sum_squares[has_values] / counts[has_values] - mean[has_values] * mean[has_values]
    ).clamp_min(0.0)
    standard_deviation = torch.sqrt(variance)
    usable_scale = (counts > 1) & (standard_deviation >= 1e-12)
    scale[usable_scale] = standard_deviation[usable_scale]

    return TargetScalerState(mean=mean.to(torch.float32), scale=scale.to(torch.float32))


def transform_targets(
    targets: Tensor,
    scaler: TargetScalerState,
    mask: Tensor | None = None,
) -> Tensor:
    """Standardize targets, optionally retaining zero at missing positions."""
    _validate_target_tensor(targets, scaler, "targets")
    valid_mask = _validate_mask(mask, targets)
    mean, scale = _state_for(targets, scaler)
    transformed = (targets - mean) / scale
    if valid_mask is None:
        return transformed
    return torch.where(valid_mask, transformed, torch.zeros_like(transformed))


def inverse_predictions(predictions: Tensor, scaler: TargetScalerState) -> Tensor:
    """Return predictions on the original target scale."""
    _validate_target_tensor(predictions, scaler, "predictions")
    mean, scale = _state_for(predictions, scaler)
    return predictions * scale + mean


def _normalise_state_tensor(value: Tensor, field: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TaskError(f"target scaler {field} must be a torch.Tensor")
    if value.ndim != 1 or value.numel() < 1:
        raise TaskError(f"target scaler {field} must have shape [T] with T >= 1")
    if not torch.is_floating_point(value):
        raise TaskError(f"target scaler {field} must have a floating dtype")
    converted = value.detach().clone().to(dtype=torch.float32, device="cpu")
    if not torch.isfinite(converted).all():
        raise TaskError(f"target scaler {field} must contain finite values")
    return converted


def _validate_train_indices(
    dataset: MolecularDataset,
    train_indices: Sequence[int],
) -> tuple[int, ...]:
    if isinstance(train_indices, (str, bytes)):
        raise TaskError("train_indices must be a non-empty sequence of dataset indices")
    try:
        indices = tuple(train_indices)
    except TypeError as exc:
        raise TaskError("train_indices must be a non-empty sequence of dataset indices") from exc
    if not indices:
        raise TaskError("train_indices must not be empty")
    seen: set[int] = set()
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TaskError(f"train_indices contains a non-integer index: {index!r}")
        if index < 0 or index >= len(dataset):
            raise TaskError(f"train_indices contains index {index} outside dataset bounds")
        if index in seen:
            raise TaskError(f"train_indices contains duplicate index {index}")
        seen.add(index)
    return indices


def _resolve_feature_schema(
    dataset: MolecularDataset,
    explicit_schema: FeatureSchema | None,
) -> FeatureSchema:
    if explicit_schema is not None:
        if not isinstance(explicit_schema, FeatureSchema):
            raise TaskError("feature_schema must be a FeatureSchema or None")
        return explicit_schema
    schema = getattr(dataset, "feature_schema", None)
    if not isinstance(schema, FeatureSchema):
        raise TaskError(
            "feature_schema is required for datasets without a feature_schema attribute"
        )
    return schema


def _resolve_num_targets(
    dataset: MolecularDataset,
    explicit_num_targets: int | None,
) -> int:
    if explicit_num_targets is not None:
        if (
            isinstance(explicit_num_targets, bool)
            or not isinstance(explicit_num_targets, int)
            or explicit_num_targets < 1
        ):
            raise TaskError("num_targets must be a positive integer or None")
        return explicit_num_targets
    summary = getattr(dataset, "summary", None)
    num_targets = getattr(summary, "num_targets", None)
    if (
        isinstance(num_targets, bool)
        or not isinstance(num_targets, int)
        or num_targets < 1
    ):
        raise TaskError(
            "num_targets is required for datasets without a summary.num_targets attribute"
        )
    return num_targets


def _validate_target_tensor(targets: Tensor, scaler: TargetScalerState, field: str) -> None:
    if not isinstance(targets, Tensor):
        raise TaskError(f"{field} must be a torch.Tensor")
    if targets.ndim < 1 or targets.shape[-1] != scaler.mean.numel():
        raise TaskError(f"{field} must have shape [..., {scaler.mean.numel()}]")
    if not torch.is_floating_point(targets):
        raise TaskError(f"{field} must have a floating dtype")


def _validate_mask(mask: Tensor | None, targets: Tensor) -> Tensor | None:
    if mask is None:
        return None
    if not isinstance(mask, Tensor) or mask.shape != targets.shape:
        raise TaskError("target mask must have the same shape as targets")
    if mask.dtype != torch.bool:
        raise TaskError("target mask must have dtype torch.bool")
    return mask


def _state_for(targets: Tensor, scaler: TargetScalerState) -> tuple[Tensor, Tensor]:
    return (
        scaler.mean.to(device=targets.device, dtype=targets.dtype),
        scaler.scale.to(device=targets.device, dtype=targets.dtype),
    )


def _validate_task_batch(
    predictions: Tensor,
    targets: Tensor,
    mask: Tensor | None,
) -> Tensor:
    if not isinstance(predictions, Tensor) or not isinstance(targets, Tensor):
        raise TaskError("predictions and targets must be torch.Tensor values")
    if predictions.shape != targets.shape:
        raise TaskError("predictions and targets must have identical shapes")
    if predictions.ndim < 1 or not torch.is_floating_point(predictions):
        raise TaskError("predictions must be a floating-point tensor")
    if not torch.is_floating_point(targets):
        raise TaskError("targets must be a floating-point tensor")
    valid_mask = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask
    if valid_mask.shape != targets.shape or valid_mask.dtype != torch.bool:
        raise TaskError("target mask must have the same shape as targets and dtype torch.bool")
    observed_targets = targets[valid_mask]
    if observed_targets.numel() and not torch.isfinite(observed_targets).all():
        raise TaskError("observed targets must contain only finite values")
    return valid_mask


__all__ = [
    "BinaryClassificationTaskAdapter",
    "RegressionTaskAdapter",
    "TargetScalerState",
    "TaskAdapter",
    "TaskError",
    "build_task_adapter",
    "fit_target_scaler",
    "inverse_predictions",
    "transform_targets",
]

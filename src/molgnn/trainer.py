"""Model-agnostic training, early stopping, and fit-loop primitives."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from .dataset import DataLoaders
from .evaluator import EvaluationResult, evaluate
from .tasks import TaskAdapter, TaskError


class TrainerError(RuntimeError):
    """Raised when a training step or fit monitor is invalid."""


@dataclass(frozen=True)
class TrainEpochResult:
    """Weighted optimization loss and batch counters for one epoch."""

    optimization_loss: float
    valid_target_count: int
    skipped_empty_target_batches: int
    batch_count: int

    @property
    def loss(self) -> float:
        """Short alias for ``optimization_loss``."""
        return self.optimization_loss


@dataclass(frozen=True)
class EpochRecord:
    """One zero-based fit-loop epoch record."""

    epoch: int
    train_optimization_loss: float
    train_eval_loss: float
    val_loss: float
    metrics: dict[str, float]
    monitor: float
    is_best: bool
    skipped_empty_target_batches: int

    @property
    def epoch_number(self) -> int:
        """One-based epoch number for human-facing artifacts."""
        return self.epoch + 1


@dataclass(frozen=True)
class FitResult:
    """Fit history and the best in-memory model state."""

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_value: float
    stopped_early: bool
    best_state_dict: dict[str, Tensor]
    device: torch.device


@dataclass(frozen=True)
class StrategyResult:
    """Normalized result returned by an optional external training strategy."""

    fit_result: FitResult
    optimizer_state_dict: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.fit_result, FitResult):
            raise TrainerError("strategy result must contain a FitResult")
        if not isinstance(self.optimizer_state_dict, Mapping):
            raise TrainerError("strategy optimizer_state_dict must be a mapping")
        if not self.fit_result.history:
            raise TrainerError("strategy fit result must include at least one epoch")
        if not 0 <= self.fit_result.best_epoch < len(self.fit_result.history):
            raise TrainerError("strategy best_epoch must index its fit history")


@dataclass
class EarlyStoppingState:
    """Strict-improvement early stopping state for min/max monitors."""

    mode: Literal["min", "max"]
    patience: int
    min_delta: float = 0.0
    best_value: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    stopped: bool = False
    seen_finite: bool = False

    def __post_init__(self) -> None:
        if self.mode not in {"min", "max"}:
            raise TrainerError("early stopping mode must be min or max")
        if (
            isinstance(self.patience, bool)
            or not isinstance(self.patience, int)
            or self.patience < 1
        ):
            raise TrainerError("early stopping patience must be a positive integer")
        if self.min_delta < 0 or not math.isfinite(self.min_delta):
            raise TrainerError("early stopping min_delta must be finite and non-negative")

    def update(self, value: float, epoch: int) -> bool:
        """Record a monitor value and return whether it strictly improved."""
        if not math.isfinite(value):
            self.bad_epochs += 1
            self.stopped = self.bad_epochs >= self.patience
            return False
        self.seen_finite = True
        improved = self.best_value is None or (
            value < self.best_value - self.min_delta
            if self.mode == "min"
            else value > self.best_value + self.min_delta
        )
        if improved:
            self.best_value = value
            self.best_epoch = epoch
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.stopped = self.bad_epochs >= self.patience
        return improved


def resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve cpu/cuda/auto without silently falling back from requested CUDA."""
    value = str(requested).lower() if isinstance(requested, str) else str(requested)
    if value == "cpu":
        return torch.device("cpu")
    if value == "cuda":
        if not torch.cuda.is_available():
            raise TrainerError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise TrainerError("device must be cpu, cuda, or auto")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[object],
    optimizer: Optimizer,
    task_adapter: TaskAdapter,
    device: torch.device | str,
    *,
    epoch: int = 0,
) -> TrainEpochResult:
    """Run one optimization epoch with valid-target weighted loss."""
    if not isinstance(model, nn.Module):
        raise TrainerError("model must be a torch.nn.Module")
    training_device = torch.device(device)
    model.train()
    loss_total = 0.0
    valid_target_count = 0
    skipped_empty_target_batches = 0
    batch_count = 0

    for batch_index, batch in enumerate(loader):
        batch_count += 1
        optimizer.zero_grad(set_to_none=True)
        batch_to = getattr(batch, "to", None)
        if not callable(batch_to):
            raise TrainerError(f"batch {batch_index} does not support .to(device)")
        device_batch = batch_to(training_device)
        targets = getattr(device_batch, "y", None)
        mask = getattr(device_batch, "y_mask", None)
        if not isinstance(targets, Tensor) or not isinstance(mask, Tensor):
            raise TrainerError(f"batch {batch_index} is missing y/y_mask tensors")
        predictions = model(device_batch)
        if not isinstance(predictions, Tensor):
            raise TrainerError(
                f"model output at epoch {epoch}, batch {batch_index} must be a Tensor"
            )
        try:
            loss = task_adapter.loss(predictions, targets, mask)
        except TaskError as exc:
            raise TrainerError(
                f"invalid task batch at epoch {epoch}, batch {batch_index}: {exc}"
            ) from exc
        target_count = int(mask.sum().item())
        if loss is None:
            skipped_empty_target_batches += 1
            continue
        if not torch.isfinite(loss).item():
            raise TrainerError(f"non-finite loss at epoch {epoch}, batch {batch_index}")
        loss.backward()
        _validate_gradients(model, epoch=epoch, batch_index=batch_index)
        optimizer.step()
        loss_total += float(loss.item()) * target_count
        valid_target_count += target_count

    weighted_loss = loss_total / valid_target_count if valid_target_count else math.nan
    return TrainEpochResult(
        optimization_loss=weighted_loss,
        valid_target_count=valid_target_count,
        skipped_empty_target_batches=skipped_empty_target_batches,
        batch_count=batch_count,
    )


def fit(
    model: nn.Module,
    loaders: DataLoaders,
    optimizer: Optimizer,
    task_adapter: TaskAdapter,
    *,
    epochs: int,
    patience: int,
    monitor: str = "val_loss",
    monitor_mode: Literal["min", "max"] = "min",
    device: str | torch.device = "auto",
    target_names: Sequence[str] | None = None,
    callbacks: Sequence[Callable[[EpochRecord], None]] = (),
) -> FitResult:
    """Fit using only train/train-eval/validation loaders and early stopping."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise TrainerError("epochs must be a positive integer")
    resolved_device = resolve_device(device)
    model.to(resolved_device)
    stopping = EarlyStoppingState(mode=monitor_mode, patience=patience)
    history: list[EpochRecord] = []
    best_state_dict: dict[str, Tensor] | None = None
    callback_values = (callbacks,) if callable(callbacks) else tuple(callbacks)

    for epoch in range(epochs):
        train_result = train_one_epoch(
            model,
            loaders.train,
            optimizer,
            task_adapter,
            resolved_device,
            epoch=epoch,
        )
        train_eval = evaluate(
            model,
            loaders.train_eval,
            task_adapter,
            resolved_device,
            target_names=target_names,
        )
        validation = evaluate(
            model,
            loaders.validation,
            task_adapter,
            resolved_device,
            target_names=target_names,
        )
        metrics = _epoch_metrics(train_result, train_eval, validation)
        monitor_value = _monitor_value(monitor, metrics)
        is_best = stopping.update(monitor_value, epoch)
        if is_best:
            best_state_dict = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
        record = EpochRecord(
            epoch=epoch,
            train_optimization_loss=train_result.optimization_loss,
            train_eval_loss=train_eval.loss,
            val_loss=validation.loss,
            metrics=metrics,
            monitor=monitor_value,
            is_best=is_best,
            skipped_empty_target_batches=train_result.skipped_empty_target_batches,
        )
        history.append(record)
        for callback in callback_values:
            callback(record)
        if stopping.stopped:
            break

    if not stopping.seen_finite or stopping.best_value is None or stopping.best_epoch is None:
        raise TrainerError(f"monitor '{monitor}' had no finite value during fit")
    if best_state_dict is None:
        raise TrainerError("fit completed without a best model state")
    return FitResult(
        history=tuple(history),
        best_epoch=stopping.best_epoch,
        best_value=stopping.best_value,
        stopped_early=stopping.stopped,
        best_state_dict=best_state_dict,
        device=resolved_device,
    )


def _validate_gradients(model: nn.Module, *, epoch: int, batch_index: int) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise TrainerError(
                f"non-finite gradient at epoch {epoch}, batch {batch_index}, parameter '{name}'"
            )


def _epoch_metrics(
    train_result: TrainEpochResult,
    train_eval: EvaluationResult,
    validation: EvaluationResult,
) -> dict[str, float]:
    metrics = {
        "loss": train_result.optimization_loss,
        "train_optimization_loss": train_result.optimization_loss,
        "train_loss": train_eval.loss,
        "val_loss": validation.loss,
    }
    metrics.update({f"train_{name}": value for name, value in train_eval.metrics.items()})
    metrics.update({f"val_{name}": value for name, value in validation.metrics.items()})
    metrics.update(validation.metrics)
    return metrics


def _monitor_value(monitor: str, metrics: dict[str, float]) -> float:
    if monitor not in metrics:
        raise TrainerError(f"unsupported fit monitor '{monitor}'")
    return metrics[monitor]


__all__ = [
    "EarlyStoppingState",
    "EpochRecord",
    "FitResult",
    "StrategyResult",
    "TrainEpochResult",
    "TrainerError",
    "fit",
    "resolve_device",
    "train_one_epoch",
]

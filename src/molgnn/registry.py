"""Small explicit registry for architecture-only model construction."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from torch import Tensor, nn


class RegistryError(ValueError):
    """Raised when a model cannot be registered or built."""


@dataclass(frozen=True)
class BuildContext:
    """Dataset-derived dimensions injected into model constructors."""

    atom_dim: int
    bond_dim: int
    num_targets: int
    feature_schema_version: str = ""

    def __post_init__(self) -> None:
        for name in ("atom_dim", "bond_dim", "num_targets"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RegistryError(f"BuildContext.{name} must be a positive integer")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[..., nn.Module]
    required_batch_fields: tuple[str, ...] = ()
    graph_transform_name: str | None = None
    prediction_reducer_name: str = "identity"


_REGISTRY: dict[str, ModelSpec] = {}
_T = TypeVar("_T", bound=Callable[..., nn.Module])


def register_model(
    name: str,
    *,
    required_batch_fields: Sequence[str] = (),
    graph_transform_name: str | None = None,
    prediction_reducer_name: str = "identity",
) -> Callable[[_T], _T]:
    """Return a decorator registering one architecture under ``name``."""
    clean_name = _validate_name(name)
    fields = tuple(str(field) for field in required_batch_fields)
    if graph_transform_name is not None:
        graph_transform_name = _validate_name(graph_transform_name)
    prediction_reducer_name = _validate_name(prediction_reducer_name)

    def decorator(factory: _T) -> _T:
        if clean_name in _REGISTRY:
            raise RegistryError(f"model '{clean_name}' is already registered")
        if not callable(factory):
            raise RegistryError("model factory must be callable")
        _REGISTRY[clean_name] = ModelSpec(
            clean_name,
            factory,
            fields,
            graph_transform_name,
            prediction_reducer_name,
        )
        return factory

    return decorator


def available_models() -> tuple[str, ...]:
    """Return registered model names in deterministic order."""
    return tuple(sorted(_REGISTRY))


def get_model_spec(name: str) -> ModelSpec:
    """Return a registered spec or an informative unknown-model error."""
    clean_name = _validate_name(name)
    try:
        return _REGISTRY[clean_name]
    except KeyError as exc:
        available = ", ".join(available_models()) or "<none>"
        raise RegistryError(f"unknown model '{clean_name}'. Available models: {available}") from exc


def build_model(
    name: str,
    parameters: Mapping[str, object] | None,
    context: BuildContext,
) -> nn.Module:
    """Instantiate one registered model with validated parameters and context."""
    spec = get_model_spec(name)
    provided = dict(parameters or {})
    signature = inspect.signature(spec.factory)
    accepted = {
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    unknown = set(provided) - accepted
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise RegistryError(f"unknown parameter(s) for model '{spec.name}': {names}")
    context_values = {
        "atom_dim": context.atom_dim,
        "bond_dim": context.bond_dim,
        "num_targets": context.num_targets,
    }
    kwargs: dict[str, object] = dict(provided)
    for key, value in context_values.items():
        if key in accepted:
            if key in kwargs and kwargs[key] != value:
                raise RegistryError(f"model parameter '{key}' conflicts with BuildContext")
            kwargs[key] = value
    try:
        model = spec.factory(**kwargs)
    except TypeError as exc:
        raise RegistryError(f"invalid parameters for model '{spec.name}': {exc}") from exc
    if not isinstance(model, nn.Module):
        raise RegistryError(f"model factory '{spec.name}' did not return torch.nn.Module")
    return model


def validate_required_batch_fields(batch: object, spec: ModelSpec) -> None:
    """Fail early when a transformed batch cannot satisfy a model contract."""

    missing = [
        field
        for field in spec.required_batch_fields
        if not isinstance(getattr(batch, field, None), Tensor)
    ]
    if missing:
        names = ", ".join(missing)
        raise RegistryError(f"model '{spec.name}' batch is missing tensor field(s): {names}")


def clear_registry() -> None:
    """Clear registrations, primarily for isolated tests."""
    _REGISTRY.clear()


def _validate_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError("model name must be a non-empty string")
    return value.strip()


__all__ = [
    "BuildContext",
    "ModelSpec",
    "RegistryError",
    "available_models",
    "build_model",
    "clear_registry",
    "get_model_spec",
    "register_model",
    "validate_required_batch_fields",
]

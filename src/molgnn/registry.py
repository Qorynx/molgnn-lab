"""Small explicit registry for architecture-only model construction."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeVar

from torch import Tensor, nn


class RegistryError(ValueError):
    """Raised when a model cannot be registered or built."""


GeometryRequirement = Literal["none", "optional", "required"]
# ``topology_2d`` marks 2D-topology-only models whose runtime contract is
# defined by a model-local transform (e.g. 3D Infomax's OGB view).
GeometryRole = Literal["none", "pure_3d", "hybrid", "topology_2d"]


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
    default_parameters: Mapping[str, object] = field(
        default_factory=lambda: MappingProxyType({})
    )
    required_batch_fields: tuple[str, ...] = ()
    optional_batch_fields: tuple[str, ...] = ()
    graph_transform_name: str | None = None
    transform_output_fields: tuple[str, ...] = ()
    prediction_reducer_name: str = "identity"
    geometry_requirement: GeometryRequirement = "none"
    geometry_role: GeometryRole = "none"
    benchmark_enabled: bool = True
    benchmark_order: int = 0


_REGISTRY: dict[str, ModelSpec] = {}
_T = TypeVar("_T", bound=Callable[..., nn.Module])
_CONTEXT_PARAMETER_NAMES = frozenset(
    {"atom_dim", "bond_dim", "num_targets", "feature_schema_version"}
)


def register_model(
    name: str,
    *,
    default_parameters: Mapping[str, object] | None = None,
    required_batch_fields: Sequence[str] = (),
    optional_batch_fields: Sequence[str] = (),
    graph_transform_name: str | None = None,
    transform_output_fields: Sequence[str] = (),
    prediction_reducer_name: str = "identity",
    geometry_requirement: GeometryRequirement = "none",
    geometry_role: GeometryRole = "none",
    benchmark_enabled: bool = True,
    benchmark_order: int = 0,
) -> Callable[[_T], _T]:
    """Return a decorator registering one architecture under ``name``."""
    clean_name = _validate_name(name)
    fields = tuple(_validate_name(field) for field in required_batch_fields)
    if len(set(fields)) != len(fields):
        raise RegistryError("required_batch_fields must not contain duplicates")
    optional_fields = tuple(_validate_name(field) for field in optional_batch_fields)
    if len(set(optional_fields)) != len(optional_fields):
        raise RegistryError("optional_batch_fields must not contain duplicates")
    overlapping_fields = set(fields) & set(optional_fields)
    if overlapping_fields:
        names = ", ".join(sorted(overlapping_fields))
        raise RegistryError(
            f"optional_batch_fields must not duplicate required_batch_fields: {names}"
        )
    if graph_transform_name is not None:
        graph_transform_name = _validate_name(graph_transform_name)
    transform_fields = tuple(_validate_name(field) for field in transform_output_fields)
    if len(set(transform_fields)) != len(transform_fields):
        raise RegistryError("transform_output_fields must not contain duplicates")
    if set(transform_fields) - (set(fields) | set(optional_fields)):
        raise RegistryError(
            "transform_output_fields must be included in required_batch_fields "
            "or optional_batch_fields"
        )
    if transform_fields and graph_transform_name is None:
        raise RegistryError("transform_output_fields requires a graph_transform_name")
    prediction_reducer_name = _validate_name(prediction_reducer_name)
    if geometry_requirement not in {"none", "optional", "required"}:
        raise RegistryError("geometry_requirement must be none, optional, or required")
    if geometry_role not in {"none", "pure_3d", "hybrid", "topology_2d"}:
        raise RegistryError(
            "geometry_role must be none, pure_3d, hybrid, or topology_2d"
        )
    if geometry_requirement == "none" and geometry_role not in {"none", "topology_2d"}:
        raise RegistryError(
            "geometry_role must be none or topology_2d when geometry_requirement is none"
        )
    if geometry_requirement != "none" and geometry_role == "none":
        raise RegistryError(
            "geometry_role must be pure_3d or hybrid when geometry is used"
        )
    defaults = _validated_parameter_mapping(
        default_parameters or {}, field="default_parameters"
    )
    managed_defaults = sorted(set(defaults) & _CONTEXT_PARAMETER_NAMES)
    if managed_defaults:
        names = ", ".join(managed_defaults)
        raise RegistryError(
            f"default parameter(s) managed by BuildContext are not allowed: {names}"
        )
    if not isinstance(benchmark_enabled, bool):
        raise RegistryError("benchmark_enabled must be a boolean")
    if isinstance(benchmark_order, bool) or not isinstance(benchmark_order, int):
        raise RegistryError("benchmark_order must be an integer")

    def decorator(factory: _T) -> _T:
        if clean_name in _REGISTRY:
            raise RegistryError(f"model '{clean_name}' is already registered")
        if not callable(factory):
            raise RegistryError("model factory must be callable")
        spec = ModelSpec(
            name=clean_name,
            factory=factory,
            default_parameters=MappingProxyType(copy.deepcopy(defaults)),
            required_batch_fields=fields,
            optional_batch_fields=optional_fields,
            graph_transform_name=graph_transform_name,
            transform_output_fields=transform_fields,
            prediction_reducer_name=prediction_reducer_name,
            geometry_requirement=geometry_requirement,
            geometry_role=geometry_role,
            benchmark_enabled=benchmark_enabled,
            benchmark_order=benchmark_order,
        )
        resolve_model_parameters(spec, {}, BuildContext(1, 1, 1))
        _REGISTRY[clean_name] = spec
        return factory

    return decorator


def available_models() -> tuple[str, ...]:
    """Return registered model names in deterministic order."""
    return tuple(sorted(_REGISTRY))


def benchmark_models() -> tuple[ModelSpec, ...]:
    """Return benchmark-enabled model specs in stable default execution order."""
    return tuple(
        sorted(
            (spec for spec in _REGISTRY.values() if spec.benchmark_enabled),
            key=lambda spec: (spec.benchmark_order, spec.name),
        )
    )


def resolve_benchmark_models(names: Sequence[str] | None) -> tuple[ModelSpec, ...]:
    """Resolve an explicit ordered subset or all benchmark-enabled models."""
    if names is None:
        return benchmark_models()
    if isinstance(names, (str, bytes)) or not isinstance(names, Sequence):
        raise RegistryError("model selection must be a non-empty sequence of names")
    clean_names = tuple(_validate_name(name) for name in names)
    if not clean_names:
        raise RegistryError("model selection must not be empty")
    if len(set(clean_names)) != len(clean_names):
        raise RegistryError("model selection must not contain duplicates")
    return tuple(get_model_spec(name) for name in clean_names)


def get_model_spec(name: str) -> ModelSpec:
    """Return a registered spec or an informative unknown-model error."""
    clean_name = _validate_name(name)
    try:
        return _REGISTRY[clean_name]
    except KeyError as exc:
        available = ", ".join(available_models()) or "<none>"
        raise RegistryError(
            f"unknown model '{clean_name}'. Available models: {available}"
        ) from exc


def build_model(
    name: str,
    parameters: Mapping[str, object] | None,
    context: BuildContext,
) -> nn.Module:
    """Instantiate one registered model with validated parameters and context."""
    spec = get_model_spec(name)
    resolved = resolve_model_parameters(spec, parameters or {}, context)
    accepted, _ = _accepted_factory_parameters(spec.factory)
    context_values = {
        "atom_dim": context.atom_dim,
        "bond_dim": context.bond_dim,
        "num_targets": context.num_targets,
        "feature_schema_version": context.feature_schema_version,
    }
    kwargs: dict[str, object] = dict(resolved)
    for key, value in context_values.items():
        if key in accepted:
            kwargs[key] = value
    try:
        model = spec.factory(**kwargs)
    except TypeError as exc:
        raise RegistryError(
            f"invalid parameters for model '{spec.name}': {exc}"
        ) from exc
    if not isinstance(model, nn.Module):
        raise RegistryError(
            f"model factory '{spec.name}' did not return torch.nn.Module"
        )
    return model


def resolve_model_parameters(
    spec: ModelSpec,
    overrides: Mapping[str, object] | None,
    context: BuildContext,
) -> Mapping[str, object]:
    """Materialize effective architecture parameters without context-derived values."""
    if not isinstance(spec, ModelSpec):
        raise RegistryError("spec must be a ModelSpec")
    if not isinstance(context, BuildContext):
        raise RegistryError("context must be a BuildContext")
    provided = _validated_parameter_mapping(overrides or {}, field="model parameters")
    managed = sorted(set(provided) & _CONTEXT_PARAMETER_NAMES)
    if managed:
        names = ", ".join(managed)
        raise RegistryError(
            f"model parameter(s) managed by BuildContext cannot be overridden: {names}"
        )

    accepted, accepts_kwargs = _accepted_factory_parameters(spec.factory)
    candidates = {**dict(spec.default_parameters), **provided}
    if not accepts_kwargs:
        unknown = set(candidates) - set(accepted)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise RegistryError(
                f"unknown parameter(s) for model '{spec.name}': {names}"
            )

    resolved: dict[str, object] = {}
    for name, parameter in accepted.items():
        if name in _CONTEXT_PARAMETER_NAMES:
            continue
        if parameter.default is not inspect.Parameter.empty:
            resolved[name] = parameter.default
    resolved.update(spec.default_parameters)
    resolved.update(provided)

    missing = [
        name
        for name, parameter in accepted.items()
        if name not in _CONTEXT_PARAMETER_NAMES
        and parameter.default is inspect.Parameter.empty
        and name not in resolved
    ]
    if missing:
        names = ", ".join(sorted(missing))
        raise RegistryError(
            f"model '{spec.name}' requires default or override parameter(s): {names}"
        )

    return MappingProxyType(copy.deepcopy(resolved))


def validate_required_batch_fields(batch: object, spec: ModelSpec) -> None:
    """Fail early when a transformed batch cannot satisfy a model contract."""

    missing = [
        field
        for field in spec.required_batch_fields
        if not isinstance(getattr(batch, field, None), Tensor)
    ]
    if missing:
        names = ", ".join(missing)
        raise RegistryError(
            f"model '{spec.name}' batch is missing tensor field(s): {names}"
        )


def clear_registry() -> None:
    """Clear registrations, primarily for isolated tests."""
    _REGISTRY.clear()


def _validate_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryError("model name must be a non-empty string")
    return value.strip()


def _validated_parameter_mapping(
    value: Mapping[str, object], *, field: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RegistryError(f"{field} must be a mapping")
    return {_validate_name(key): item for key, item in value.items()}


def _accepted_factory_parameters(
    factory: Callable[..., nn.Module],
) -> tuple[dict[str, inspect.Parameter], bool]:
    signature = inspect.signature(factory)
    accepted = {
        parameter.name: parameter
        for parameter in signature.parameters.values()
        if parameter.name != "self"
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    return accepted, accepts_kwargs


__all__ = [
    "BuildContext",
    "GeometryRequirement",
    "GeometryRole",
    "ModelSpec",
    "RegistryError",
    "available_models",
    "benchmark_models",
    "build_model",
    "clear_registry",
    "get_model_spec",
    "register_model",
    "resolve_benchmark_models",
    "resolve_model_parameters",
    "validate_required_batch_fields",
]

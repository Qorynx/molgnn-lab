"""Load explicitly selected extension hooks for the command-line runner."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


_MISSING = object()


class HookError(ValueError):
    """Raised when a CLI extension hook cannot be loaded safely."""


@dataclass(frozen=True)
class LoadedHook:
    """An imported top-level callback and the selector that identified it."""

    reference: str
    callback: Callable[..., object]


def load_hook(spec: str, *, option: str) -> LoadedHook:
    """Load ``module_or_path.py:top_level_callable`` from a CLI selector."""
    reference, separator, attribute = spec.strip().rpartition(":")
    if not separator or not reference or not attribute:
        raise HookError(
            f"{option} must use 'module_or_path.py:top_level_callable'; got {spec!r}"
        )
    if not attribute.isidentifier():
        raise HookError(
            f"{option} callback must be a top-level callable; got {attribute!r}"
        )

    module = (
        _load_file(reference, option=option)
        if reference.lower().endswith(".py")
        else _load_module(reference, option=option)
    )
    try:
        callback = getattr(module, attribute, _MISSING)
    except Exception as exc:
        raise HookError(
            f"{option} callback {attribute!r} could not be read from {reference!r}: {exc}"
        ) from exc
    if callback is _MISSING:
        raise HookError(
            f"{option} callback {attribute!r} was not found in {reference!r}"
        )
    if not callable(callback):
        raise HookError(
            f"{option} callback {attribute!r} in {reference!r} is not callable"
        )
    return LoadedHook(reference=spec.strip(), callback=callback)


def _load_file(reference: str, *, option: str) -> object:
    """Import an absolute or relative source file and support its siblings."""
    try:
        path = Path(reference).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise HookError(
            f"{option} hook file {reference!r} could not be resolved: {exc}"
        ) from exc
    if not path.is_file():
        raise HookError(f"{option} hook file does not exist: {path}")

    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    module_name = f"_molgnn_hook_{hashlib.sha256(str(path).encode()).hexdigest()[:16]}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise HookError(f"{option} hook file could not be imported: {path}")

    module = importlib.util.module_from_spec(module_spec)
    try:
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise HookError(f"{option} hook import failed for {path}: {exc}") from exc
    return module


def _load_module(reference: str, *, option: str) -> object:
    """Import a dotted Python module reference with a controlled CLI error."""
    if not _is_dotted_module_name(reference):
        raise HookError(
            f"{option} must name a dotted module or a .py file; got {reference!r}"
        )
    try:
        return importlib.import_module(reference)
    except Exception as exc:
        raise HookError(
            f"{option} hook module import failed for {reference!r}: {exc}"
        ) from exc


def _is_dotted_module_name(reference: str) -> bool:
    """Accept importable absolute module names, but not paths or relative imports."""
    return bool(reference) and all(part.isidentifier() for part in reference.split("."))


__all__ = ["HookError", "LoadedHook", "load_hook"]

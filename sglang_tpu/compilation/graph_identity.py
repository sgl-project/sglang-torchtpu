"""Stable full graph names and canonical signatures for TPU compilation."""

import hashlib
import json
from collections.abc import Callable
from enum import Enum
from typing import Any

import torch


def canonical(value: Any, default: Callable[[Any], Any] | None = None) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=default,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _key_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, tuple):
        return [_key_value(item) for item in value]
    return value


def stable_repr(value: Any) -> str:
    text = repr(value)
    if " at 0x" in text:
        raise TypeError(f"{type(value).__name__} has no stable identity: {text}")
    return text


def graph_name(name: str, key: Any) -> str:
    """Identify one specialization of a named compiled function across processes.

    Shared SPMD compilation matches every rank's graph by this string, and the
    lowered-graph store keys by it. Key values that JSON cannot encode (devices,
    pytree structures, types) fall back to their repr; a repr with a memory
    address differs per process, so it raises TypeError.
    """
    return f"{name}:{canonical(_key_value(key), default=stable_repr)}"

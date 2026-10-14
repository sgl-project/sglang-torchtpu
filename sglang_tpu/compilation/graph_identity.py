"""Stable full graph names and canonical signatures for TPU compilation."""

import hashlib
import json
from enum import Enum
from typing import Any

import torch


def canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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

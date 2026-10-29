"""Hooks run on a built model right before the TPU compile policy wraps it.

For per-model setup that must happen before torch.compile traces (e.g. building
a model family's Pallas custom ops) but should not run for every model at
``init_backend()``. Each hook gets the outer model and decides for itself
whether it applies.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

_HOOKS: list[Callable[[torch.nn.Module], None]] = []


def register_pre_compile_hook(hook: Callable[[torch.nn.Module], None]) -> None:
    if hook not in _HOOKS:
        _HOOKS.append(hook)


def run_pre_compile_hooks(model: torch.nn.Module) -> None:
    for hook in _HOOKS:
        hook(model)

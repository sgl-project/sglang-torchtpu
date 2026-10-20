"""Deferred call primitive shared by TPU compilation policies."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from typing import Any

import torch

from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
    has_forward_context,
)


def _capture_forward_context() -> ForwardContext | None:
    return get_forward_context() if has_forward_context() else None


@dataclass(frozen=True)
class PendingCall:
    """A deferred compiled call with its tracing-time forward context.

    ForwardContext is a module global, so copy_context alone cannot retain it
    when warmups are joined after another model runner has been initialized.
    """

    _wait: Callable[[], Any]
    _context: Context = field(default_factory=copy_context, repr=False)
    _forward_context: ForwardContext | None = field(
        default_factory=_capture_forward_context, repr=False
    )
    _grad_enabled: bool = field(default_factory=torch.is_grad_enabled, repr=False)
    _inference_mode: bool = field(
        default_factory=torch.is_inference_mode_enabled, repr=False
    )

    def wait(self) -> Any:
        # Restore inference mode, but never re-enable grad a caller disabled
        # (WarmupQueue.wait runs every warmup under no_grad).
        grad_enabled = self._grad_enabled and torch.is_grad_enabled()
        with (
            (
                forward_context(self._forward_context)
                if self._forward_context is not None
                else nullcontext()
            ),
            torch.inference_mode(self._inference_mode),
            torch.set_grad_enabled(grad_enabled),
        ):
            return self._context.run(self._wait)

    def then(self, callback: Callable[[Any], Any]) -> PendingCall:
        """Apply ``callback`` after this result is ready."""
        return PendingCall(lambda: callback(self.wait()))

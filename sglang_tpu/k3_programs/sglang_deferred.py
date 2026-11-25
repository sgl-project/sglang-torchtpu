# SPDX-License-Identifier: Apache-2.0
"""Keep torch_tpu's deferred work out of the programs' ops.

torch_tpu defers an in-place write through a view or an index (``x[idx] = v``,
``x[:n].copy_(...)``) until something materializes the written tensor, and
the next executed op takes every live deferred write with it. Between two
program steps sglang's scheduler leaves such writes behind -- the request
pool's ``req_index_to_mamba_index_mapping`` on every admission, sampling and
length bookkeeping -- so the program's next whole-model op would compile a
new executable for each mix of them (seconds to tens of seconds a variant).
vLLM-TorchTPU's runner keeps its batch bookkeeping off the device for the
same reason (``MegakernelModelRunner._skip_device_sampling_metadata``).

:class:`DeferredGuard` runs before each program step: it finds the TPU
tensors reachable from the step's sglang objects that hold deferred work and
materializes them as a program of their own (small, compiled once a shape),
without waiting for it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Attribute walks stop at these depths and skip these (large or cyclic) names.
_DEPTH = 3
_SKIP = frozenset(
    {
        "reqs",
        "model",
        "model_runner",
        "tree_cache",
        "caches",
        "plan",
        "program",
        "steps",
        "_prompts",
        "kv_cache",
        "k_buffer",
        "v_buffer",
        "kv_buffer",
        "token_to_kv_pool",
        "full_kv_pool",
        "state_pool",
    }
)


def device_tensors(
    roots: Iterable[tuple[str, Any]], device_type: str = "tpu"
) -> list[tuple[str, torch.Tensor]]:
    """``(path, tensor)`` for every ``device_type`` tensor reachable from
    ``roots`` through attributes, lists, tuples and dicts, up to ``_DEPTH``
    levels."""
    found: list[tuple[str, torch.Tensor]] = []
    seen: set[int] = set()

    def visit(name: str, value: Any, depth: int) -> None:
        if value is None or isinstance(value, (int, float, str, bytes, bool)):
            return
        if id(value) in seen:
            return
        seen.add(id(value))
        if isinstance(value, torch.Tensor):
            if value.device.type == device_type:
                found.append((name, value))
            return
        if depth >= _DEPTH:
            return
        if isinstance(value, (list, tuple)):
            for i, item in enumerate(value[:64]):
                visit(f"{name}[{i}]", item, depth + 1)
            return
        if isinstance(value, dict):
            for key, item in list(value.items())[:64]:
                visit(f"{name}[{key!r}]", item, depth + 1)
            return
        attrs = getattr(value, "__dict__", None)
        if attrs is None:
            slots = getattr(type(value), "__slots__", ())
            attrs = {s: getattr(value, s, None) for s in slots if isinstance(s, str)}
        for key, item in list(attrs.items()):
            if key in _SKIP or key.startswith("__"):
                continue
            visit(f"{name}.{key}", item, depth + 1)

    for name, root in roots:
        visit(name, root, 0)
    return found


class DeferredGuard:
    """Materialize the deferred writes reachable from a step's sglang
    objects before the program runs (see the module docstring)."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        log: bool = False,
        device_type: str = "tpu",
        sync: Any = None,
    ) -> None:
        """``sync``: ``(is_materialized, synchronize)`` (default torch_tpu's)."""
        self.enabled = enabled
        self.log = log
        self.device_type = device_type
        self.flushes = 0
        self._sync = sync
        self._reported: set[str] = set()

    def __call__(self, roots: Iterable[tuple[str, Any]]) -> int:
        """Flush; returns how many tensors held deferred work."""
        if not self.enabled:
            return 0
        tensors = device_tensors(roots, self.device_type)
        if not tensors:
            return 0
        if self._sync is None:
            from torch_tpu._internal.sync import is_materialized, synchronize

            self._sync = (is_materialized, synchronize)
        is_materialized, synchronize = self._sync
        pending = [(name, t) for name, t in tensors if not is_materialized(t)]
        if not pending:
            return 0
        synchronize([t for _, t in pending], wait=False)
        self.flushes += 1
        if self.log:
            new = [name for name, _ in pending if name not in self._reported]
            if new:
                self._reported.update(new)
                logger.info(
                    "K3 programs: flushed deferred writes ahead of a step: %s",
                    ", ".join(
                        f"{name} {tuple(t.shape)}" for name, t in pending if name in new
                    ),
                )
        return len(pending)

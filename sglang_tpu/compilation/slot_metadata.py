# SPDX-License-Identifier: Apache-2.0
"""Build the slot-only part of an attention backend's next metadata ahead of time.

A spec step's draft and verify metadata needs the device sequence lengths for
its rows, and reading them waits for the forward still on the device. The part
that depends only on the request slots (page tables) can be built and uploaded
before that read, while the device is busy. A backend registers how; the spec
worker calls ``prepare`` right before the forward whose rows it cannot build yet.
"""

from typing import Any, Callable, Dict

import torch

_PREPARERS: Dict[type, Callable[..., None]] = {}


def register(backend_cls: type, prepare: Callable[..., None]) -> None:
    _PREPARERS[backend_cls] = prepare


def prepare(
    attn_backend: Any,
    req_pool_indices_cpu: torch.Tensor,
    *,
    reuse: bool = False,
    kv_len_bound: int | None = None,
) -> None:
    """Let ``attn_backend`` build its next forward's slot tables for these host slots.

    The backend's next metadata build consumes them, or rebuilds if its slots differ.
    ``reuse``: keep tables built earlier in the same step for the same slots.
    ``kv_len_bound``: host upper bound on every row's KV length after the forward.
    No-op unless SGLANG_TPU_SPEC_PREBUILD_SLOT_METADATA and the backend registered.
    """
    from sglang_tpu.envs import envs as tpu_envs

    if not tpu_envs.SGLANG_TPU_SPEC_PREBUILD_SLOT_METADATA.get():
        return
    for cls in type(attn_backend).__mro__:
        preparer = _PREPARERS.get(cls)
        if preparer is not None:
            extra = {} if kv_len_bound is None else {"kv_len_bound": kv_len_bound}
            preparer(attn_backend, req_pool_indices_cpu, reuse=reuse, **extra)
            return

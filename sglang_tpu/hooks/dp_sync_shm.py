# SPDX-License-Identifier: Apache-2.0
"""Install the shared-memory DP MLP-sync row exchange (sglang_tpu/dp_sync_shm.py).

Why: the scheduler's per-step DP metadata all-gather runs over gloo (loopback
     TCP) on every rank's serving thread before the forward can be sized.
How: AROUND dp_attn.exchange_sync_rows; pass-through unless
     SGLANG_TPU_DP_SYNC_SHM is set, or where the group is not on one x86 host
     (then the stock gloo exchange runs). The gathered values are identical.
Upstream seam issue: not filed.
Remove when: the core DP sync exposes a pluggable host exchange.
"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookType
from sglang_tpu.dp_sync_shm import ShmExchangeRegistry
from sglang_tpu.envs import envs as tpu_envs

_REGISTRY = ShmExchangeRegistry()


def exchange_sync_rows(original, global_flat, local_row, group):
    """AROUND dp_attn.exchange_sync_rows."""
    if tpu_envs.SGLANG_TPU_DP_SYNC_SHM.get():
        exchange = _REGISTRY.get(group=group, local_row=local_row)
        if exchange is not None:
            exchange.exchange(global_flat, local_row)
            return None
    return original(global_flat, local_row, group=group)


def hook_specs():
    return [
        (
            "sglang.srt.managers.scheduler_components.dp_attn.exchange_sync_rows",
            exchange_sync_rows,
            HookType.AROUND,
        ),
    ]

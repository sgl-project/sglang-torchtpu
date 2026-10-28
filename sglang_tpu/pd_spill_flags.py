"""One switch for offloading prefill KV to decode DRAM, and the flags it implies.

SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM=1 on both roles makes the
decode's DRAM the KV offload tier below the prefill's own RAM. It turns on:
the victim push, the decode-prefix pull and, on hybrid SWA models, a host tier
of SWA windows on both roles (the prefill keeps each path's last window, the
decode keeps that many windows). Each component flag still works on its own; a
component flag set explicitly wins over the default the switch would give it.
"""

from __future__ import annotations

from sglang_tpu.envs import envs as tpu_envs

# SWA host windows the switch provides on a hybrid model when
# SGLANG_TPU_PD_SWA_HOST_WINDOWS is unset: one 128-token window per cached path
# at the window sizes seen so far; size it with the host budget.
_DEFAULT_SWA_HOST_WINDOWS = 128


def spill_enabled() -> bool:
    return tpu_envs.SGLANG_ENABLE_PREFILL_KV_OFFLOAD_TO_DECODE_DRAM.get()


def victim_push_enabled() -> bool:
    return spill_enabled() or tpu_envs.SGLANG_DISAGG_PREFIX_VICTIM_PUSH.get()


def decode_prefix_pull_enabled() -> bool:
    return spill_enabled() or tpu_envs.SGLANG_DISAGG_PREFILL_PULL_DECODE_PREFIX.get()


def swa_host_windows() -> int:
    """Host SWA windows per rank on either role; 0 keeps the full-only host tier."""
    if tpu_envs.SGLANG_TPU_PD_SWA_HOST_WINDOWS.is_set():
        return tpu_envs.SGLANG_TPU_PD_SWA_HOST_WINDOWS.get()
    return _DEFAULT_SWA_HOST_WINDOWS if spill_enabled() else 0


def prefill_swa_path_end_backup_enabled() -> bool:
    return swa_host_windows() > 0


def decode_swa_host_windows() -> int:
    return swa_host_windows()

# SPDX-License-Identifier: Apache-2.0
"""Named in-kernel regions for profiling (MUSE_MK_TRACE)."""

import contextlib
import os

import jax

# Profiling only: set True to name in-kernel regions, visible in an xprof trace
# with libtpu's --xla_enable_custom_call_region_trace=true.
# MUSE_MK_TRACE=1: named regions as on-device trace spans (2: inner regions too).
_TRACE = int(os.environ.get("MUSE_MK_TRACE", "0")) >= 1


# per-chunk attention regions: thousands per step, they distort the timing they show
_TRACE_INNER = int(os.environ.get("MUSE_MK_TRACE", "0")) >= 2
_INNER = frozenset(
    (
        "kv_unpack",
        "qk",
        "softmax",
        "pv",
        "kv_wait",
        "ring_wait",
        "ex_wait",
        "full_expert",
        "ex_gather",
        "bd_w13",
        "bd_w2",
        "ex_scatter",
        "ex_accum",
        "kv_prologue",
        "own_rows",
        "y_row_wait",
        "combine_rows",
        "route_count",
        "route_perm",
        "route_dma",
    )
)


def _scope(name):
    # read at trace time too: a long-lived compile worker imports ops before a sweep's env is set
    from sglang_tpu.envs import envs as tpu_envs

    level = max(int(_TRACE) + int(_TRACE_INNER), tpu_envs.SGLANG_TPU_MK_TRACE_REGIONS.get())
    on = level >= 1 and (level >= 2 or name not in _INNER)
    return jax.named_scope(name) if on else contextlib.nullcontext()

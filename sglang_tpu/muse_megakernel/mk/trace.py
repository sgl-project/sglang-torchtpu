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
    )
)


def _scope(name):
    on = _TRACE and (_TRACE_INNER or name not in _INNER)
    return jax.named_scope(name) if on else contextlib.nullcontext()

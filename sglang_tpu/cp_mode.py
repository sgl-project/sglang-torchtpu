# SPDX-License-Identifier: Apache-2.0
"""Which prefill context-parallel (PCP) mode this server runs.

``gather`` (default) is the all-gather design: every rank of the CP group
all-gathers the chunk's K/V on every layer, writes the whole sequence into its
own fully replicated KV pool and runs ordinary attention over its zigzag rows.

``ring`` stripes the KV cache across the CP ranks by page -- rank ``r`` holds
global page ``l * P + r``, exactly the decode-context-parallel (DCP) layout,
which is why the mode requires ``--dcp-size == --attn-cp-size``; the stripe
group is the attention-CP group, so attention TP may exceed 1.
A zigzag prefill attends the striped cached prefix through the fused ring
kernel (remote DMA of KV blocks between ranks) and the all-gathered chunk in a
new-token phase; every other forward (decode, extends the zigzag strategy does
not take) runs the existing DCP path over the same stripes.

The mode is read from ``SGLANG_TPU_PCP_MODE`` once per call site at startup;
scheduler processes inherit the launcher's environment, so every rank agrees.
"""

from __future__ import annotations

PCP_MODE_GATHER = "gather"
PCP_MODE_RING = "ring"
PCP_MODES = (PCP_MODE_GATHER, PCP_MODE_RING)


def pcp_mode() -> str:
    """The configured PCP mode; raises on a value that is not a mode."""
    from sglang_tpu.envs import envs

    mode = (envs.SGLANG_TPU_PCP_MODE.get() or PCP_MODE_GATHER).strip().lower()
    if mode not in PCP_MODES:
        raise ValueError(
            f"SGLANG_TPU_PCP_MODE must be one of {', '.join(PCP_MODES)}, got {mode!r}."
        )
    return mode


def pcp_ring_requested() -> bool:
    """Whether ``SGLANG_TPU_PCP_MODE=ring`` is set (regardless of topology)."""
    return pcp_mode() == PCP_MODE_RING


def pcp_ring_enabled() -> bool:
    """Whether this process serves prefill context parallelism in ring mode.

    Reads the published parallel topology, so it is only meaningful after
    server-args resolution (backend construction, compile policy, forward).
    """
    if not pcp_ring_requested():
        return False
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().attn_cp_size > 1

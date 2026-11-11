# SPDX-License-Identifier: Apache-2.0
"""The KV pool budget leaves room for the buffered PP hand-off's buffers.

reserve_pp_buffered_handoff
  Why: the buffered hand-off (SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF) allocates
       landing buffers for its receives and holds its queued sends while
       serving, after the KV pool is sized: up to 7 receives plus the send
       depth + 1 transfers at the largest token bucket, and one zero residual
       per bucket (about 1.05 GiB for GPT-OSS-120B at 16384 rows). Sized from
       free memory, the pool can leave no room for them, and the stage runs
       out of memory under load.
  How: AROUND KVCacheConfigurator._profile_available_bytes: with the switch on
       and more than one pipeline stage, subtract the hand-off's worst case
       (pp_scheduler.buffered_handoff_reserve_bytes) from the bytes the KV
       pool may use, as core does for multimodal runtime memory
       (mm_runtime_reservation_gb). The hand-off checks the same bound when it
       starts (PPBufferedHandoff.check_hbm), which also covers a
       --max-total-tokens below the profiled capacity.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1901 (review).
  Remove when: the KV budget takes runtime reservations from the platform.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def reserve_pp_buffered_handoff(original, kvc, *args, **kwargs) -> int:
    available = original(kvc, *args, **kwargs)
    from sglang_tpu.envs import envs as tpu_envs

    if (
        not tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.get()
        or kvc.pp_size < 2
        or kvc.is_draft_worker
    ):
        return available
    from sglang_tpu.pp_scheduler import buffered_handoff_reserve_bytes

    reserved = buffered_handoff_reserve_bytes(
        model=kvc.model,
        model_config=kvc.model_config,
        model_dtype=kvc.model_dtype,
        server_args=kvc.server_args,
        spec_aux_config=kvc.spec_aux_config,
    )
    logger.info(
        "Reserving %.2f GB of the KV budget for the buffered PP hand-off.",
        reserved / (1 << 30),
    )
    return available - reserved

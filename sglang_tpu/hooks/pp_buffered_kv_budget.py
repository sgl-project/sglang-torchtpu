# SPDX-License-Identifier: Apache-2.0
"""A pipeline stage's KV pool budget: its own free memory, less the buffered
PP hand-off's buffers.

size_pp_stage_on_its_own_memory
  Why: the KV budget is free memory reduced to its minimum over every rank of
       the server (get_available_gpu_memory over the world group), so every
       pipeline stage sizes on the stage with the least free memory, usually
       the first (embedding and input encoders). The stages then derive their
       token counts from that one budget and take the minimum, and a stage
       holding more KV layers than the lightest one binds on memory that is not
       its own: the other stages' free memory goes unused.
  How: AROUND KVCacheConfigurator._profile_available_bytes: with the buffered
       hand-off on (its buffers are reserved below; the lockstep wave's are
       not), more than one pipeline stage and no mamba state pool (split from
       the world-minimum budget), add back the stage's own free memory (the
       minimum over its TP group) less the world minimum. The token count
       stays the minimum over stages (_apply_token_constraints), so every
       stage holds the same count.
  Upstream PR or issue: none yet; upstream sizes the same way.
  Remove when: the KV budget reduces free memory over the pipeline stage.

reserve_pp_buffered_handoff
  Why: the buffered hand-off (SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF) allocates
       landing buffers for its receives and holds its queued sends while
       serving, after the KV pool is sized: up to 7 receives plus the send
       depth + 2 transfers at the largest token bucket, and one zero residual
       per bucket (about 1.14 GiB for GPT-OSS-120B at 16384 rows). Sized from
       free memory, the pool can leave no room for them, and the stage runs
       out of memory under load.
  How: AROUND KVCacheConfigurator._profile_available_bytes: with the switch on
       and more than one pipeline stage, subtract this stage's worst case
       (pp_scheduler.buffered_handoff_reserve_bytes: the first stage receives
       nothing) from the bytes the KV pool may use, as
       core does for multimodal runtime memory (mm_runtime_reservation_gb). The
       hand-off checks the same bound when it starts (PPBufferedHandoff.check_hbm),
       which also covers a --max-total-tokens below the profiled capacity.
  Upstream PR or issue: sgl-project/sglang-torchtpu#1901 (review).
  Remove when: the KV budget takes runtime reservations from the platform.
"""

from __future__ import annotations

import functools
import logging

logger = logging.getLogger(__name__)


def pp_stage_kv_budget(original, kvc, *args, **kwargs) -> int:
    """The registered hook: both adjustments, in one AROUND."""
    return reserve_pp_buffered_handoff(
        functools.partial(size_pp_stage_on_its_own_memory, original),
        kvc,
        *args,
        **kwargs,
    )


def size_pp_stage_on_its_own_memory(original, kvc, *args, **kwargs) -> int:
    available = original(kvc, *args, **kwargs)
    from sglang_tpu.envs import envs as tpu_envs

    # Only the buffered hand-off reserves its activation buffers in the budget;
    # a mamba state pool is split from the world-minimum budget inside `original`.
    if (
        not tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.get()
        or kvc.pp_size < 2
        or kvc.is_draft_worker
        or kvc.mambaish_config is not None
    ):
        return available
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.utils.device import get_available_gpu_memory

    parallel = get_parallel()

    # Both reductions run on every rank, in this order. Nothing is allocated
    # since `original` measured, so the world minimum is the one it used.
    stage_free = get_available_gpu_memory(
        kvc.device, kvc.gpu_id, distributed=True, cpu_group=parallel.tp_group.cpu_group
    )
    world_free = get_available_gpu_memory(
        kvc.device,
        kvc.gpu_id,
        distributed=True,
        cpu_group=parallel.world_group.cpu_group,
    )
    headroom = max(stage_free - world_free, 0.0)
    if headroom > 0:
        logger.info(
            "Sizing the KV budget on this pipeline stage's free memory: "
            "%.2f GB, %.2f GB above the server's minimum.",
            stage_free,
            headroom,
        )
    return available + int(headroom * (1 << 30))


def reserve_pp_buffered_handoff(original, kvc, *args, **kwargs) -> int:
    available = original(kvc, *args, **kwargs)
    from sglang_tpu.envs import envs as tpu_envs

    if (
        not tpu_envs.SGLANG_TPU_ENABLE_PP_BUFFERED_HANDOFF.get()
        or kvc.pp_size < 2
        or kvc.is_draft_worker
    ):
        return available
    from sglang.srt.runtime_context import get_parallel
    from sglang_tpu.pp_scheduler import buffered_handoff_reserve_bytes

    reserved = buffered_handoff_reserve_bytes(
        model=kvc.model,
        model_config=kvc.model_config,
        model_dtype=kvc.model_dtype,
        server_args=kvc.server_args,
        spec_aux_config=kvc.spec_aux_config,
        pp_rank=get_parallel().pp_rank,
    )
    logger.info(
        "Reserving %.2f GB of the KV budget for the buffered PP hand-off.",
        reserved / (1 << 30),
    )
    return available - reserved

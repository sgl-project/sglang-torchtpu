# SPDX-License-Identifier: Apache-2.0
"""TPU scheduler initialization adaptations.

Why: Dense TPU models query MoE backend flags at compiled layer boundaries;
     lazy default initialization inside Dynamo tracing is unsupported.
     SGLANG_TPU_PREFILL_DRAIN_INTERVAL is also TPU-specific scheduler config
     that should be read in sglang_tpu rather than importing tpu_envs in
     shared Scheduler.
How: AFTER Scheduler.init_moe_gemm_config; retain the core GEMM setup and
     initialize the flags only when the base method left them unset.
     AFTER Scheduler.__init__; apply SGLANG_TPU_PREFILL_DRAIN_INTERVAL onto
     scheduler._tpu_prefill_drain_interval.
Upstream PR or issue: not filed; tracked in sglang-torchtpu PR #1289.
Remove when: core initializes backend flags for dense models before tracing,
             or exposes a platform/scheduler initialization seam at this point.
"""


def configure_tpu_scheduler(result, scheduler, *args, **kwargs):
    from sglang_tpu.envs import envs as tpu_envs

    scheduler._tpu_prefill_drain_interval = max(
        0, int(tpu_envs.SGLANG_TPU_PREFILL_DRAIN_INTERVAL.get())
    )
    return result


def initialize_dense_moe_config(result, scheduler):
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import get_flags

    moe = get_flags().moe
    if moe.runner_backend is None or moe.a2a_backend is None:
        initialize_moe_config()
    return result

# SPDX-License-Identifier: Apache-2.0
"""TPU scheduler initialization adaptations.

Why: Dense TPU models query MoE backend flags at compiled layer boundaries;
     lazy default initialization inside Dynamo tracing is unsupported.
How: AFTER Scheduler.init_moe_gemm_config; retain the core GEMM setup and
     initialize the flags only when the base method left them unset.
Upstream PR or issue: not filed; tracked in sglang-torchtpu PR #1289.
Remove when: core initializes backend flags for dense models before tracing,
             or exposes a platform initialization seam at this point.
"""


def initialize_dense_moe_config(result, scheduler):
    from sglang.srt.layers.moe.utils import initialize_moe_config
    from sglang.srt.runtime_context import get_flags

    moe = get_flags().moe
    if moe.runner_backend is None or moe.a2a_backend is None:
        initialize_moe_config()
    return result

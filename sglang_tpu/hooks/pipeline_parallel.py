# SPDX-License-Identifier: Apache-2.0
"""TPU pipeline-parallel launch validation.

Why: Core's PP check admits only EAGLE, but the TPU target relays the Eagle3
     aux captures across stages (see sglang_tpu.pp_wave).
How: AROUND check_pipeline_parallel_compat; EAGLE3 gets the TPU checks, every
     other configuration still runs the core check.
Upstream PR or issue: not filed.
Remove when: core asks the platform which speculative algorithms PP supports.
"""

from __future__ import annotations

from typing import Any, Callable


def check_pipeline_parallel_compat(original: Callable[[Any], None], cfg: Any) -> None:
    algorithm = cfg.speculative_algorithm
    if algorithm is None or algorithm.upper() != "EAGLE3":
        original(cfg)
        return
    # The rest of TPU's PP checks run in TpuSRTPlatform.apply_server_args_defaults.
    assert cfg.disable_overlap_schedule, (
        "Pipeline parallelism is not compatible with overlap schedule"
    )
    assert not cfg.enable_multi_layer_eagle, (
        "Pipeline parallelism does not support multi-layer EAGLE3"
    )

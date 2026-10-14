# SPDX-License-Identifier: Apache-2.0
"""TPU pipeline-parallel adaptations.

check_pipeline_parallel_compat
  Why: Core's PP check admits only EAGLE, but the TPU target relays the Eagle3
       aux captures across stages (see sglang_tpu.pp_wave).
  How: AROUND check_pipeline_parallel_compat; EAGLE3 runs the core check as
       EAGLE, every other configuration runs it unchanged.
  Upstream PR or issue: not filed.
  Remove when: core asks the platform which speculative algorithms PP supports.

skip_spec_op_precompile_without_draft
  Why: Under PP only the last stage hosts the draft; the other stages keep a
       draftless spec worker and run no spec ops.
  How: AROUND Scheduler.precompile_tpu_spec_ops; skip it on those stages.
  Upstream PR or issue: not filed.
  Remove when: precompile_tpu_spec_ops moves into sglang_tpu.
"""

from __future__ import annotations

from typing import Any, Callable

from sglang.srt.managers.scheduler import Scheduler


class _AsEagle:
    """A config whose speculative algorithm reads as EAGLE; other fields pass through."""

    __slots__ = ("_cfg",)

    def __init__(self, cfg: Any) -> None:
        self._cfg = cfg

    def __getattr__(self, name: str) -> Any:
        if name == "speculative_algorithm":
            return "EAGLE"
        return getattr(self._cfg, name)


def check_pipeline_parallel_compat(original: Callable[[Any], None], cfg: Any) -> None:
    algorithm = cfg.speculative_algorithm
    if algorithm is not None and algorithm.upper() == "EAGLE3":
        # The rest of TPU's PP checks run in TpuSRTPlatform.apply_server_args_defaults.
        cfg = _AsEagle(cfg)
    original(cfg)


def skip_spec_op_precompile_without_draft(
    original: Callable[[Scheduler], None], scheduler: Scheduler
) -> None:
    if scheduler.draft_worker.draft_worker is None:
        return
    original(scheduler)

# SPDX-License-Identifier: Apache-2.0
"""TPU pipeline-parallel adaptations.

check_pipeline_parallel_compat
  Why: Core's PP check admits only EAGLE and DSPARK, but the TPU target relays
       the Eagle3 and DFlash aux captures across stages (see sglang_tpu.pp_wave).
  How: AROUND check_pipeline_parallel_compat; EAGLE3 runs the core check as
       EAGLE, DFLASH on the experimental worker as DSPARK (a PD prefill that
       only seeds the draft KV), every other configuration runs it unchanged.
  Upstream PR or issue: not filed.
  Remove when: core asks the platform which speculative algorithms PP supports.

skip_spec_op_precompile_without_verify
  Why: Under PP only the last stage hosts the draft; the other stages keep a
       draftless spec worker and run no spec ops. A TpuSpecWorker PD prefill
       never verifies, and these ops serve verify (80 s on a Muse PP8 stage);
       core EAGLE still runs its draft extend op there.
  How: AROUND Scheduler.precompile_tpu_spec_ops; skip it on draftless stages
       and on an experimental-spec --disaggregation-mode prefill server.
  Upstream PR or issue: not filed.
  Remove when: precompile_tpu_spec_ops moves into sglang_tpu.
"""

from __future__ import annotations

from typing import Any, Callable

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_disagg


class _AsAlgorithm:
    """A config whose speculative algorithm reads as another; other fields pass through."""

    __slots__ = ("_cfg", "_algorithm")

    def __init__(self, cfg: Any, algorithm: str) -> None:
        self._cfg = cfg
        self._algorithm = algorithm

    def __getattr__(self, name: str) -> Any:
        if name == "speculative_algorithm":
            return self._algorithm
        return getattr(self._cfg, name)


def check_pipeline_parallel_compat(original: Callable[[Any], None], cfg: Any) -> None:
    from sglang_tpu.experimental import experimental_spec_enabled

    algorithm = cfg.speculative_algorithm
    # The rest of TPU's PP checks run in TpuSRTPlatform.apply_server_args_defaults.
    if algorithm is not None and algorithm.upper() == "EAGLE3":
        cfg = _AsAlgorithm(cfg, "EAGLE")
    elif (
        algorithm is not None
        and algorithm.upper() == "DFLASH"
        and experimental_spec_enabled()
    ):
        cfg = _AsAlgorithm(cfg, "DSPARK")
    original(cfg)


def skip_spec_op_precompile_without_verify(
    original: Callable[[Scheduler], None], scheduler: Scheduler
) -> None:
    from sglang_tpu.experimental import is_experimental_spec

    if scheduler.draft_worker.draft_worker is None or (
        get_disagg().disaggregation_mode == "prefill"
        and is_experimental_spec(scheduler.spec_algorithm)
    ):
        return
    original(scheduler)

# SPDX-License-Identifier: Apache-2.0
"""TPU pipeline-parallel adaptations.

check_pipeline_parallel_compat
  Why: Core's PP check admits only EAGLE, but the TPU target relays the Eagle3
       aux captures across stages (see sglang_tpu.pp_wave).
  How: AROUND check_pipeline_parallel_compat; EAGLE3 runs the core check as
       EAGLE, every other configuration runs it unchanged.
  Upstream PR or issue: not filed.
  Remove when: core asks the platform which speculative algorithms PP supports.
"""

from __future__ import annotations

from typing import Any, Callable


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

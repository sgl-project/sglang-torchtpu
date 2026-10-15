# SPDX-License-Identifier: Apache-2.0
"""Experimental TPU speculative worker selection.

Why: TpuSpecWorker is a TPU-only prototype that core must not import.
How: AFTER SpeculativeAlgorithm.create_worker; return TpuDsparkWorker for DSPARK
     when SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER is set, else keep core's class.
Upstream PR or issue: not filed; prototype only.
Remove when: the experimental worker is dropped or core grows a platform
             spec-worker factory.
"""

from __future__ import annotations

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.experimental import is_experimental_dspark


def select_experimental_spec_worker(
    result: type,
    algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
) -> type[BaseSpecWorker] | None:
    del result, server_args
    if not is_experimental_dspark(algorithm):
        return None
    from sglang_tpu.experimental.speculative.dspark.dspark_worker import TpuDsparkWorker

    return TpuDsparkWorker

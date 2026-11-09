# SPDX-License-Identifier: Apache-2.0
"""Experimental TPU speculative worker selection.

Why: TpuSpecWorker is a TPU-only prototype that core must not import.
How: AFTER SpeculativeAlgorithm.create_worker; when
     SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER or SGLANG_TPU_HOST_PADDING_NO_D2H is
     set, return TpuBlockSpecWorker for DFLASH/DSPARK and TpuChainSpecWorker for
     MTP (NEXTN) and EAGLE3, else keep core's class. A PP stage before the last
     gets the draftless TpuPipelineStageWorker. DSPARK on a DeepSeek V4.1
     checkpoint with a bundled drafter takes TpuDsv41DsparkWorker instead (the
     drafter runs inside the target model, under DP attention).
Upstream PR or issue: not filed; prototype only.
Remove when: the experimental worker is dropped or core grows a platform
             spec-worker factory.
"""

from __future__ import annotations

from sglang.srt.runtime_context import get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.experimental import (
    is_experimental_chain,
    is_experimental_spec_worker,
)


def select_experimental_spec_worker(
    result: type,
    algorithm: SpeculativeAlgorithm,
    server_args: ServerArgs,
) -> type[BaseSpecWorker] | None:
    del result
    parallel = get_parallel()
    if (
        is_experimental_chain(algorithm) or is_experimental_spec_worker(algorithm)
    ) and not parallel.pp_group.is_last_rank:
        from sglang_tpu.experimental.speculative.pipeline_stage_worker import (
            TpuPipelineStageWorker,
        )

        return TpuPipelineStageWorker
    if is_experimental_chain(algorithm):
        from sglang_tpu.experimental.speculative.chain_spec_worker import (
            TpuChainSpecWorker,
        )

        return TpuChainSpecWorker
    if is_experimental_spec_worker(algorithm):
        if algorithm.is_dspark() and _bundled_dsv41_drafter(server_args):
            from sglang_tpu.experimental.speculative.dspark.dsv41_dspark_worker import (
                TpuDsv41DsparkWorker,
            )

            return TpuDsv41DsparkWorker
        from sglang_tpu.experimental.speculative.block_spec_worker import (
            TpuBlockSpecWorker,
        )

        return TpuBlockSpecWorker
    return None


def _bundled_dsv41_drafter(server_args: ServerArgs) -> bool:
    from sglang.srt.configs.model_config import ModelConfig
    from sglang_tpu.dsv41.dspark import dspark_config_of

    hf_config = ModelConfig.from_server_args(server_args).hf_config
    return (
        getattr(hf_config, "model_type", None) == "deepseek_v41"
        and dspark_config_of(hf_config) is not None
    )

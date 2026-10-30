"""A PD prefill pipeline stage before the last: target forward only, no draft.

The last stage hosts the TpuSpecWorker and its draft; the stages before it relay
activations and aux captures (sglang_tpu.pp_wave), so they only run the target
on the same host-padded batches.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_disagg, get_parallel
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.recompilation_checker import (
    snapshot_recompilation_baseline,
)
from sglang_tpu.experimental import register_host_batch_padder


class TpuPipelineStageWorker(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        del server_args, gpu_id, nccl_port
        super().__init__()
        parallel = get_parallel()
        if (
            parallel.pp_size == 1
            or parallel.pp_group.is_last_rank
            or get_disagg().disaggregation_mode != "prefill"
        ):
            raise NotImplementedError(
                "TpuPipelineStageWorker serves the stages before the last of a "
                "--disaggregation-mode prefill server."
            )
        self._target_worker = target_worker
        self._draft_worker = None
        # Every stage pads like the last one, so all stages run the same programs.
        register_host_batch_padder(self._pad_bs)

    def init_cuda_graphs(self) -> None:
        self._warmup_host_uploads()
        # torch_tpu compiles asynchronously; the checker must not count the warmups.
        torch.accelerator.synchronize()
        snapshot_recompilation_baseline()

    def _warmup_host_uploads(self) -> None:
        """HOST_PADDED inputs upload at their batch or token bucket, as on the last stage."""
        runner = self._target_worker.model_runner
        policy = runner.compile_policy
        assert isinstance(policy, DecoderTpuCompilePolicy), type(policy)
        assert policy.t_bucketer is not None and policy.b_bucketer is not None
        # torch_tpu compiles one program per shape even for a plain H2D copy.
        for rows in {*policy.b_bucketer.buckets, *policy.t_bucketer.buckets}:
            for dtype in (torch.int64, torch.int32):
                torch.zeros((rows,), dtype=dtype).to(runner.device, non_blocking=True)
        # prepare_for_extend uploads the extend lengths at the live batch size.
        for bs in range(1, policy._extend_compile_cap() + 1):
            torch.zeros((bs,), dtype=torch.int64).to(runner.device, non_blocking=True)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: Callable[[torch.Tensor], None] | None = None,
        grammar_barrier: Callable[[], None] | None = None,
        *,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> GenerationBatchResult:
        del grammar_barrier
        if not (batch.forward_mode.is_extend() or batch.is_extend_in_batch):
            raise NotImplementedError(
                f"A PD prefill pipeline stage only extends, got {batch.forward_mode}."
            )
        # FULL as on the last stage: the spec compile policy warms only EXTEND(FULL).
        result = self._target_worker.forward_batch_generation(
            batch,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        result.new_seq_lens = batch.seq_lens
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        return result

    def _pad_bs(self, bs: int) -> int:
        compile_policy = self._target_worker.model_runner.compile_policy
        if not isinstance(compile_policy, DecoderTpuCompilePolicy):
            raise TypeError(
                f"TpuPipelineStageWorker needs a DecoderTpuCompilePolicy, got {type(compile_policy)}."
            )
        b_bucketer = compile_policy.b_bucketer
        assert b_bucketer is not None
        return b_bucketer.next_bucket(bs)

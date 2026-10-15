"""DSpark draft model holder for TpuDsparkWorker, in place of a draft TpModelWorker."""

from __future__ import annotations

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_schedule, get_spec
from sglang_tpu.dflash_draft_kv_write import dflash_write_draft_kv
from sglang_tpu.experimental.compilation.dspark_compile_policy import (
    TpuDsparkSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool


class TpuDsparkDraftWorker:
    """Owns the draft ModelRunner; shares the target's req_to_token pool and allocator."""

    def __init__(self, target_runner: ModelRunner) -> None:
        model_config = ModelConfig.from_server_args(
            target_runner.server_args,
            model_path=get_spec().speculative_draft_model_path,
            model_revision=get_spec().speculative_draft_model_revision,
            is_draft_model=True,
            # The draft runs at absolute target positions.
            context_length=target_runner.model_config.context_len,
        )
        # FIXME(junyanxu): ModelRunner requires gpu_id / nccl_port, which TPU never reads;
        # drop the placeholders once core makes them optional.
        self.model_runner = ModelRunner(
            model_config=model_config,
            mem_fraction_static=get_schedule().mem_fraction_static,
            gpu_id=0,
            nccl_port=0,
            server_args=target_runner.server_args,
            is_draft_worker=True,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config: MemoryPoolConfig | None = None,
        req_to_token_pool: ReqToTokenPool | None = None,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None = None,
    ) -> None:
        if req_to_token_pool is None or token_to_kv_pool_allocator is None:
            raise ValueError("The DSpark draft must share the target's KV pools.")
        self.model_runner.req_to_token_pool = req_to_token_pool
        self.model_runner.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.model_runner.alloc_memory_pool(memory_pool_config)

    def init_attention_backends(self) -> None:
        self.model_runner.init_attention_backends()

    def init_cuda_graphs(self) -> None:
        self.model_runner.init_cuda_graphs(capture_decode_cuda_graph=False)
        self._compile_policy().precompile_draft(
            self.model_runner.attn_backend, kv_writer=self._write_draft_kv_eager
        )

    def write_draft_kv(
        self,
        *,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: torch.Tensor | None = None,
        commit_lens: torch.Tensor | None = None,
    ) -> None:
        """Project target hidden into the draft context KV (extend: all rows; commit: accepted)."""
        # Pads to the warmed bucket, then runs the compiled _write_draft_kv_eager.
        self._compile_policy().append_target_hidden_to_draft_kv_by_loc_tpu(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )

    def _write_draft_kv_eager(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: torch.Tensor | None = None,
        commit_lens: torch.Tensor | None = None,
    ) -> None:
        # The fused rpa_v3 pool forbids set_kv_buffer (write_target_hidden_kv);
        # DSpark's dense draft shares DFlash's K/V layout and Pallas writer.
        kv_pool = self.model_runner.token_to_kv_pool
        if not isinstance(kv_pool, TpuFusedMHATokenToKVPool):
            raise TypeError(
                f"TpuDsparkDraftWorker needs TpuFusedMHATokenToKVPool, got {type(kv_pool)}."
            )
        dflash_write_draft_kv(
            draft_model=self._draft_model(),
            kv_pool=kv_pool,
            ctx_hidden=self._draft_model().project_target_hidden(target_hidden),
            ctx_positions=positions,
            ctx_cache_loc=cache_loc,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
            max_num_reqs=self.model_runner.req_to_token_pool.size,
        )

    def _draft_model(self) -> DSparkDraftModel:
        model = self.model_runner.model
        if not isinstance(model, DSparkDraftModel):
            raise TypeError(f"Expected a dense DSparkDraftModel, got {type(model)}.")
        return model

    def _compile_policy(self) -> TpuDsparkSpecDecodeCompilePolicyMixin:
        compile_policy = self.model_runner.compile_policy
        if not isinstance(compile_policy, TpuDsparkSpecDecodeCompilePolicyMixin):
            raise TypeError(
                "The DSpark draft needs TpuDsparkSpecDecodeCompilePolicyMixin, got "
                f"{type(compile_policy).__name__}."
            )
        return compile_policy

    @property
    def draft_runner(self) -> ModelRunner:
        """Read by Scheduler.precompile_tpu_spec_ops."""
        return self.model_runner

    @property
    def graph_memory_usage(self) -> dict[str, float]:
        return self.model_runner.graph_memory_usage

    @property
    def graph_time_usage(self) -> dict[str, float]:
        return self.model_runner.graph_time_usage

    @property
    def weight_load_time(self) -> float:
        return self.model_runner.weight_load_time

    @property
    def preloaded_weights_bytes(self) -> int:
        return self.model_runner.preloaded_weights_bytes

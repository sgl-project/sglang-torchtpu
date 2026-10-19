"""Draft model holder for TpuSpecWorker, in place of a draft TpModelWorker."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.runtime_context import get_schedule, get_spec


class TpuDraftWorker(ABC):
    """Owns the draft ModelRunner; shares the target's req_to_token pool and allocator.

    Attributes:
      num_drafts: Drafts per step (gamma); the target verifies gamma + 1 rows.
    """

    num_drafts: int

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
            raise ValueError("The draft must share the target's KV pools.")
        self.model_runner.req_to_token_pool = req_to_token_pool
        self.model_runner.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.model_runner.alloc_memory_pool(memory_pool_config)

    def init_attention_backends(self) -> None:
        self.model_runner.init_attention_backends()

    def init_cuda_graphs(self) -> None:
        self.model_runner.init_cuda_graphs(capture_decode_cuda_graph=False)
        self.precompile_draft()

    @abstractmethod
    def precompile_draft(self) -> None:
        """Warm every draft forward shape the worker sends, per batch bucket."""

    @property
    def draft_runner(self) -> ModelRunner:
        """Read by Scheduler.precompile_tpu_spec_ops."""
        return self.model_runner

    @property
    def draft_runners(self) -> tuple[ModelRunner, ...]:
        """Read by BaseSpecWorker for weight updates and the draft KV pool, as on EAGLE drafts."""
        return (self.model_runner,)

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

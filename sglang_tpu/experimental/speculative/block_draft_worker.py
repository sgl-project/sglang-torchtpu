"""Block-parallel draft model holder for TpuBlockSpecWorker, in place of a draft TpModelWorker."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import get_schedule, get_spec
from sglang_tpu.dflash_draft_kv_write import dflash_write_draft_kv
from sglang_tpu.experimental.compilation.block_compile_policy import (
    TpuBlockSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

# (step_logits [pbs, V], step_idx) -> sampled [pbs] tokens.
DraftSampler = Callable[[torch.Tensor, int], torch.Tensor]


class TpuBlockDraftWorker(ABC):
    """Owns the draft ModelRunner; shares the target's req_to_token pool and allocator.

    DFlash and DSpark diverge only here: the subclass validates its draft model,
    sets the block geometry and attaches the target modules in __init__, and
    samples drafts from the block hidden.

    Attributes:
      num_drafts: Drafts per step (gamma); the target verifies gamma + 1 rows.
      block_width: Rows per request in the draft block forward [r, m, ..., m].
      mask_token_id: Token filling the block after the root token.
    """

    num_drafts: int
    block_width: int
    mask_token_id: int

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

    @abstractmethod
    def sample_drafts(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        sampler: DraftSampler,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sampled [pbs, gamma] drafts and the [pbs, gamma, V] logits q was drawn from.

        block_hidden is the [pbs, block_width, H] draft block output. The greedy
        drafts come from the draft graph itself (sample_draft_tokens_tpu).
        """

    def alloc_memory_pool(
        self,
        memory_pool_config: MemoryPoolConfig | None = None,
        req_to_token_pool: ReqToTokenPool | None = None,
        token_to_kv_pool_allocator: BaseTokenToKVPoolAllocator | None = None,
    ) -> None:
        if req_to_token_pool is None or token_to_kv_pool_allocator is None:
            raise ValueError("The block draft must share the target's KV pools.")
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
        # dense DFlash and DSpark drafts share one K/V layout and Pallas writer.
        kv_pool = self.model_runner.token_to_kv_pool
        if not isinstance(kv_pool, TpuFusedMHATokenToKVPool):
            raise TypeError(
                f"{type(self).__name__} needs TpuFusedMHATokenToKVPool, got {type(kv_pool)}."
            )
        draft_model = self._draft_model()
        dflash_write_draft_kv(
            draft_model=draft_model,
            kv_pool=kv_pool,
            ctx_hidden=draft_model.project_target_hidden(target_hidden),
            ctx_positions=positions,
            ctx_cache_loc=cache_loc,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
            max_num_reqs=self.model_runner.req_to_token_pool.size,
        )

    def _draft_model(self) -> DFlashDraftModel:
        model = self.model_runner.model
        if not isinstance(model, DFlashDraftModel):
            raise TypeError(f"Expected a dense DFlash-family draft, got {type(model)}.")
        return model

    def _compile_policy(self) -> TpuBlockSpecDecodeCompilePolicyMixin:
        compile_policy = self.model_runner.compile_policy
        if not isinstance(compile_policy, TpuBlockSpecDecodeCompilePolicyMixin):
            raise TypeError(
                "The block draft needs TpuBlockSpecDecodeCompilePolicyMixin, got "
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

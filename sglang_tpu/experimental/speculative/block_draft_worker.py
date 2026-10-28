"""Block-parallel draft model holder for TpuBlockSpecWorker, in place of a draft TpModelWorker."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import torch
from torch import nn

from sglang.srt.layers.utils.common import PPMissingLayer
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import get_disagg
from sglang_tpu.dflash_draft_kv_write import dflash_write_draft_kv
from sglang_tpu.experimental.compilation.block_compile_policy import (
    TpuBlockSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.speculative.draft_worker import TpuDraftWorker
from sglang_tpu.kv_pool import TpuFusedMHATokenToKVPool

# (step_logits [pbs, V], step_idx) -> sampled [pbs] tokens.
DraftSampler = Callable[[torch.Tensor, int], torch.Tensor]


class TpuBlockDraftWorker(TpuDraftWorker):
    """Block-parallel drafter; owns the draft context KV writer.

    DFlash and DSpark diverge only here: the subclass validates its draft model,
    sets the block geometry and attaches the target modules in __init__, and
    samples drafts from the block hidden.

    Attributes:
      block_width: Rows per request in the draft block forward [r, m, ..., m].
      mask_token_id: Token filling the block after the root token.
    """

    block_width: int
    mask_token_id: int

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

    def precompile_draft(self) -> None:
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
    ) -> torch.Tensor:
        """Project target hidden into the draft context KV (extend: all rows; commit: accepted).

        Returns the writer's kv_fence_tail.
        """
        # Pads to the warmed bucket, then runs the compiled _write_draft_kv_eager.
        tail = self._compile_policy().append_target_hidden_to_draft_kv_by_loc_tpu(
            target_hidden=target_hidden,
            cache_loc=cache_loc,
            positions=positions,
            cache_loc_2d=cache_loc_2d,
            commit_lens=commit_lens,
        )
        assert tail is not None
        return tail

    def _write_draft_kv_eager(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: torch.Tensor | None = None,
        commit_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
        # Read inside the compiled writer, after the write: the TPU KV free fence's tail.
        return kv_pool.kv_buffer[-1][0, 0, 0, 0, :1]

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


def target_input_embedding(target_model: nn.Module) -> nn.Module | None:
    """The target embedding the block draft embeds with; None on a PP stage without it."""
    embedding = target_model.get_input_embeddings()
    if not isinstance(embedding, PPMissingLayer):
        return unwrap_lora_layer(embedding)
    # Only the first PP stage holds it; a PD prefill only writes draft KV, never embeds.
    if get_disagg().disaggregation_mode != "prefill":
        raise NotImplementedError(
            "The block draft needs the target embedding, which this pipeline "
            "stage does not hold; PP serves block drafts on a PD prefill only."
        )
    return None

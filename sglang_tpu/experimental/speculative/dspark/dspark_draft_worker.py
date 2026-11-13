"""DSpark drafter: the DFlash block plus a Markov head over the target lm_head.

draft: z_j = l_j + bias(d_{j-1}, h_j), d_0 = r, d_j ~ softmax(z_j / T), so
       q_j = softmax(z_j / T), with l_j = lm_head(h_j) the base logits.

Not ported yet: the confidence verify budget (num_verify_drafts is always
gamma) and MoE/MLA drafts.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dspark_components.dspark_config import (
    resolve_runtime_config,
)
from sglang_tpu.experimental.compilation.block_compile_policy import block_draft_width
from sglang_tpu.experimental.speculative.block_draft_worker import (
    DraftLattice,
    DraftSampler,
    SampledDrafts,
    TpuBlockDraftWorker,
    target_input_embedding,
)


class TpuDsparkDraftWorker(TpuBlockDraftWorker):
    def __init__(self, target_runner: ModelRunner) -> None:
        super().__init__(target_runner)
        draft_model = self.model_runner.model
        if not isinstance(draft_model, DSparkDraftModel):
            raise TypeError(
                f"TpuDsparkDraftWorker needs a dense DSparkDraftModel, got {type(draft_model)}."
            )
        target_model_config = target_runner.model_config
        runtime_config = resolve_runtime_config(
            draft_hf_config=self.model_runner.model_config.hf_config,
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            target_vocab_size=target_model_config.vocab_size,
        )
        # The draft graph's markov chain (sample_draft_tokens_tpu) uses the config gamma.
        if draft_model.gamma != runtime_config.gamma:
            raise ValueError(
                f"Draft config gamma={draft_model.gamma} != runtime gamma={runtime_config.gamma}."
            )
        self.num_drafts = runtime_config.gamma
        self.block_width = block_draft_width(draft_model, self.num_drafts + 1)
        self.mask_token_id = runtime_config.mask_token_id
        self._dspark_model = draft_model
        from sglang_tpu.models.inkling2_mtp import Inkling2ForConditionalGenerationMTP

        # Inkling2 seeds its draft head per request from the last committed target features.
        self._recurrent_draft = (
            draft_model
            if isinstance(draft_model, Inkling2ForConditionalGenerationMTP)
            else None
        )
        _attach_target_vocab(
            draft_model=draft_model,
            target_model=cast(nn.Module, target_runner.model),
            target_hf_text_config=target_model_config.hf_text_config,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ) -> None:
        super().alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )
        if self._recurrent_draft is not None:
            # One recurrent seed per request pool row.
            self._recurrent_draft.allocate_f_rnn(
                req_to_token_pool.req_to_token.shape[0]
            )

    def seed_recurrent_context(
        self,
        *,
        target_hidden: torch.Tensor,
        rows: torch.Tensor,
        req_pool_indices: torch.Tensor,
        live: torch.Tensor,
    ) -> None:
        if self._recurrent_draft is None:
            return
        self._recurrent_draft.store_f_rnn(
            target_hidden=target_hidden,
            rows=rows,
            req_pool_indices=req_pool_indices,
            live=live,
        )

    def warmup_recurrent_context(
        self, *, hidden_rows: tuple[int, ...], num_rows: tuple[int, ...]
    ) -> None:
        if self._recurrent_draft is None:
            return
        device = self._recurrent_draft.f_rnn_buf.device
        features = self._recurrent_draft.num_context_features * int(
            self._recurrent_draft.config.hidden_size
        )
        for tokens, rows in zip(hidden_rows, num_rows):
            # Nothing is live, so the buffer is rewritten unchanged.
            self._recurrent_draft.store_f_rnn(
                target_hidden=torch.zeros(
                    (tokens, features), dtype=torch.bfloat16, device=device
                ),
                rows=torch.zeros((rows,), dtype=torch.int64, device=device),
                req_pool_indices=torch.zeros((rows,), dtype=torch.int64, device=device),
                live=torch.zeros((rows,), dtype=torch.bool, device=device),
            )

    def sample_drafts(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_lattice: DraftLattice | None,
        sampler: DraftSampler,
        req_pool_indices: torch.Tensor,
    ) -> SampledDrafts:
        del draft_lattice
        # Anchor drafts read all gamma rows; the rest drop the anchor row, so
        # draft_hidden[:, j - 1] predicts d_j at position n + j.
        bs = block_hidden.shape[0]
        draft_model = self._dspark_model
        draft_hidden = (
            block_hidden if draft_model.sample_from_anchor else block_hidden[:, 1:]
        )
        with torch.no_grad():
            base_logits, _ = draft_model.compute_base_logits(
                draft_hidden.reshape(bs * self.num_drafts, -1)
            )
        # Eager: gather the seed here, the compiled forward's staging is trace-time only.
        f_rnn = None
        if self._recurrent_draft is not None:
            f_rnn = self._recurrent_draft.f_rnn_buf.index_select(
                0, req_pool_indices.view(-1)[:bs].long()
            )
        draft_tokens, draft_logits = draft_model.markov_head.sample_block(
            base_logits.view(bs, self.num_drafts, -1),
            first_prev_tokens=bonus_tokens,
            hidden_states=draft_hidden,
            sampler=sampler,
            collect_corrected=True,
            f_rnn=f_rnn,
        )
        return SampledDrafts(draft_tokens, draft_logits, None)


def _attach_target_vocab(
    *,
    draft_model: DSparkDraftModel,
    target_model: nn.Module,
    target_hf_text_config: PretrainedConfig,
) -> None:
    """Share the target embedding and lm_head with the draft (dense DSpark)."""
    # HF configs carry this key only for muP targets; the draft trained on the folded head.
    draft_model.logits_mup_width_multiplier = target_hf_text_config.to_dict().get(
        "logits_mup_width_multiplier"
    )
    lm_head = unwrap_lora_layer(target_model.lm_head)
    embedding = target_input_embedding(target_model)
    if embedding is None:
        draft_model.lm_head = lm_head
        return
    draft_model.attach_shared_modules(embed_tokens=embedding, lm_head=lm_head)

"""EAGLE3 drafter: a one-layer draft on the target's aux hidden states, one token per forward.

draft: g is the draft's pre-norm hidden at the last committed row; d_1 comes from
       lm_head(norm(g)) mapped through d2t, and each step feeds the previous token
       and the draft's own pre-norm hidden. Committed rows read the target's
       concatenated aux hidden states, which the draft's fc projects to H.

Not supported: draft trees (topk > 1), multi-layer EAGLE, M-RoPE drafts.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.dspark import gather_and_crop_vocab, project_through_lm_head
from sglang.srt.models.llama_eagle3 import LlamaForCausalLMEagle3
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.pp_draft_embedding import resolve_draft_embed_and_head
from sglang_tpu.experimental.speculative.chain_draft_worker import (
    TpuChainDraftWorker,
)


class TpuEagle3DraftWorker(TpuChainDraftWorker):
    def __init__(self, target_runner: ModelRunner) -> None:
        super().__init__(target_runner)
        draft_model = self.model_runner.model
        if not isinstance(draft_model, LlamaForCausalLMEagle3):
            raise TypeError(
                f"TpuEagle3DraftWorker needs LlamaForCausalLMEagle3, got {type(draft_model)}."
            )
        if self.model_runner.model_config.model_is_mrope:
            raise NotImplementedError(
                "TpuEagle3DraftWorker does not support M-RoPE drafts."
            )
        spec = get_spec()
        if (spec.speculative_eagle_topk or 1) != 1:
            raise NotImplementedError(
                "TpuEagle3DraftWorker drafts chains only; set --speculative-eagle-topk 1."
            )
        if spec.enable_multi_layer_eagle or spec.speculative_adaptive:
            raise NotImplementedError(
                "TpuEagle3DraftWorker does not support multi-layer or adaptive EAGLE."
            )
        assert spec.speculative_num_steps is not None
        self.num_drafts = spec.speculative_num_steps
        config = draft_model.config
        self.carry_size = int(config.hidden_size)
        # fc reads the target's aux layers concatenated: num_aux * target hidden.
        self.target_feature_size = int(draft_model.model.fc.in_features)
        self.target_vocab_size = int(target_runner.model_config.vocab_size)
        _share_target_vocab(draft_model=draft_model, target_runner=target_runner)
        # The fused draft is shared by all TP ranks, so shard bounds come from a rank buffer.
        draft_model.model.embed_tokens.lift_rank_to_input()
        hot_token_id = draft_model.hot_token_id
        if hot_token_id is not None:
            # A model tensor the fused draft reads; registered with direct_compile below.
            draft_model.hot_token_id = hot_token_id.to(self.model_runner.device)
            self.fused_draft_tensor_attrs = ("hot_token_id",)
        self._draft_model = draft_model
        # norm_output drafts already carry the normed hidden that feeds lm_head.
        self._carry_is_normed = bool(draft_model.model.norm_output)

    def first_draft_logits(self, carry: torch.Tensor) -> torch.Tensor:
        model = self._draft_model
        hidden = carry if self._carry_is_normed else model.model.norm(carry)
        lm_head = model.lm_head
        with torch.no_grad():
            return gather_and_crop_vocab(
                project_through_lm_head(hidden, lm_head), lm_head
            )

    def target_tokens(self, draft_ids: torch.Tensor) -> torch.Tensor:
        hot_token_id = self._draft_model.hot_token_id
        if hot_token_id is None:
            return draft_ids
        return hot_token_id.index_select(0, draft_ids)

    def target_logits(self, draft_logits: torch.Tensor) -> torch.Tensor:
        hot_token_id = self._draft_model.hot_token_id
        if hot_token_id is None:
            return draft_logits
        dense = torch.full(
            (draft_logits.shape[0], self.target_vocab_size),
            float("-inf"),
            dtype=draft_logits.dtype,
            device=draft_logits.device,
        )
        return dense.index_copy(1, hot_token_id, draft_logits)


def _share_target_vocab(
    *, draft_model: LlamaForCausalLMEagle3, target_runner: ModelRunner
) -> None:
    """Bind the target embedding (and lm_head when the draft has none), as EagleDraftWorker.init_lm_head does."""
    model_config = target_runner.model_config
    embed, head = resolve_draft_embed_and_head(
        target_model=cast(nn.Module, target_runner.model),
        draft_model=draft_model,
        model_path=model_config.model_path,
        revision=model_config.revision,
        load_config=target_runner.load_config,
    )
    if draft_model.load_lm_head_from_target:
        draft_model.set_embed_and_head(embed, head)
    else:
        draft_model.set_embed(embed)

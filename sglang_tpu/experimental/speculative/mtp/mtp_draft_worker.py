"""MTP drafter: the checkpoint's own next-token layer, one token per forward.

draft: h_1 = layer(fc(cat(norm(embed(t)), norm(target hidden)))) normed at the
       last committed row; each step feeds the previous token and the draft's
       own normed hidden; d_j ~ softmax(lm_head(h_j) / T) over the target vocab.

Not supported: draft trees (topk > 1).
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_nextn import DeepseekV3ForCausalLMNextN
from sglang.srt.models.dspark import gather_and_crop_vocab, project_through_lm_head
from sglang.srt.models.qwen3_5_mtp import Qwen3_5ForCausalLMMTP
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.pp_draft_embedding import resolve_draft_embed_and_head
from sglang_tpu.experimental.speculative.chain_draft_worker import (
    TpuChainDraftWorker,
)

# Each reads the target hidden from spec_info.hidden_states and returns its own
# normed hidden, the logits input of its lm_head. DeepseekV3ForCausalLMNextN
# covers the DeepSeek-style NextN layers (GLM-5.3-Flash included).
MTP_DRAFT_MODELS = (Qwen3_5ForCausalLMMTP, DeepseekV3ForCausalLMNextN)
_MtpDraftModel = Qwen3_5ForCausalLMMTP | DeepseekV3ForCausalLMNextN


class TpuMtpDraftWorker(TpuChainDraftWorker):
    def __init__(self, target_runner: ModelRunner) -> None:
        super().__init__(target_runner)
        draft_model = self.model_runner.model
        if not isinstance(draft_model, MTP_DRAFT_MODELS):
            raise TypeError(
                "TpuMtpDraftWorker needs one of "
                f"{[cls.__name__ for cls in MTP_DRAFT_MODELS]}, got {type(draft_model)}."
            )
        spec = get_spec()
        if spec.speculative_eagle_topk != 1:
            raise NotImplementedError(
                "TpuMtpDraftWorker drafts chains only; set --speculative-eagle-topk 1."
            )
        assert spec.speculative_num_steps is not None
        self.num_drafts = spec.speculative_num_steps
        # Both widths are the hidden size: the draft reads the target's final hidden.
        self.carry_size = int(draft_model.config.hidden_size)
        self.target_feature_size = self.carry_size
        _share_target_vocab(draft_model=draft_model, target_runner=target_runner)
        # The fused draft is shared by all TP ranks, so shard bounds come from a rank buffer.
        draft_model.model.embed_tokens.lift_rank_to_input()
        self._draft_model = draft_model

    def first_draft_logits(self, carry: torch.Tensor) -> torch.Tensor:
        # The carry is the logits input the draft's final norm already produced.
        lm_head = self._draft_model.lm_head
        with torch.no_grad():
            return gather_and_crop_vocab(
                project_through_lm_head(carry, lm_head), lm_head
            )

    def target_tokens(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids

    def target_logits(self, draft_logits: torch.Tensor) -> torch.Tensor:
        return draft_logits


def _share_target_vocab(
    *, draft_model: _MtpDraftModel, target_runner: ModelRunner
) -> None:
    """Bind the target embedding and lm_head, as EAGLEWorkerV2.init_lm_head does for MTP."""
    model_config = target_runner.model_config
    embed, head = resolve_draft_embed_and_head(
        target_model=cast(nn.Module, target_runner.model),
        draft_model=draft_model,
        model_path=model_config.model_path,
        revision=model_config.revision,
        load_config=target_runner.load_config,
    )
    draft_model.set_embed_and_head(embed, head)

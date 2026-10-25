"""DFlash drafter: one block forward, each draft read off the target lm_head on its own.

draft: d_j ~ softmax(lm_head(h_j) / T) independently per row, so q_j is that
       softmax; greedy drafts are the in-graph argmax (sample_draft_tokens_tpu).
DFlash2 instead walks its candidate selector (models/dflash2.py): q_j is the
       softmax over the K candidates of the row d_{j-1} selects, -inf elsewhere.

Not supported: LiLiCorr, Domino projector and trained mask embeddings
(mask_embedding.pt).
"""

from __future__ import annotations

import os
from typing import cast

import torch
from torch import nn

from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.dflash import DFlash2DraftModel, DFlashDraftModel
from sglang.srt.models.dspark import (
    DSparkDraftModel,
    gather_and_crop_vocab,
    project_through_lm_head,
)
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dflash_utils import (
    DFlashDraftConfig,
    parse_dflash_draft_config,
)
from sglang_tpu.experimental.compilation.block_compile_policy import block_draft_width
from sglang_tpu.experimental.models.dflash2 import (
    build_lattice,
)
from sglang_tpu.experimental.models.dflash2 import (
    sample_drafts as sample_dflash2_drafts,
)
from sglang_tpu.experimental.speculative.block_draft_worker import (
    DraftLattice,
    DraftSampler,
    DraftSampling,
    SampledDrafts,
    TpuBlockDraftWorker,
)


class TpuDflashDraftWorker(TpuBlockDraftWorker):
    def __init__(self, target_runner: ModelRunner) -> None:
        super().__init__(target_runner)
        draft_model = self.model_runner.model
        if not isinstance(draft_model, DFlashDraftModel) or isinstance(
            draft_model, DSparkDraftModel
        ):
            raise TypeError(
                f"TpuDflashDraftWorker needs a dense DFlashDraftModel, got {type(draft_model)}."
            )
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.model_runner.model_config.hf_config
        )
        _check_supported(draft_model=draft_model, draft_config=draft_config)
        # ServerArgs resolves it from --speculative-dflash-block-size or the draft config.
        block_size = get_spec().speculative_num_draft_tokens
        assert block_size is not None
        # The grouped convs and the in-graph argmax index the block at this size.
        draft_model.set_block_size(block_size)
        self.num_drafts = block_size - 1
        self.block_width = block_draft_width(draft_model, block_size)
        self.mask_token_id = _resolve_mask_token_id(
            draft_config=draft_config,
            target_vocab_size=target_runner.model_config.vocab_size,
        )
        self._dflash_model = draft_model
        target_model = cast(nn.Module, target_runner.model)
        # Nemotron 3.5 drafts embed with their own checkpoint table.
        if not draft_model.is_nemotron_35_draft:
            draft_model.embed_tokens = unwrap_lora_layer(
                target_model.get_input_embeddings()
            )
        draft_model.lm_head = unwrap_lora_layer(target_model.lm_head)

    def sample_drafts(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_lattice: DraftLattice | None,
        sampler: DraftSampler,
    ) -> SampledDrafts:
        # Row 0 is the root; draft_hidden[:, j - 1] predicts d_j at position n + j.
        lm_head = self._dflash_model.lm_head
        assert lm_head is not None
        if isinstance(self._dflash_model, DFlash2DraftModel):
            if not isinstance(sampler, DraftSampling):
                raise TypeError(
                    "DFlash2 walks its lattice in one program; pass DraftSampling"
                )
            if draft_lattice is None:
                raise ValueError("DFlash2 samples from the draft graph's lattice")
            candidate_ids, scores = draft_lattice
            draft_tokens, q_logits, verify_tokens = sample_dflash2_drafts(
                candidate_ids,
                scores,
                bonus_tokens,
                sampler.temperatures,
                sampler.top_ks,
                sampler.rng_step,
                int(lm_head.org_vocab_size),
                dense=not sampler.target_candidates,
            )
            if sampler.target_candidates:
                return SampledDrafts(
                    draft_tokens, None, verify_tokens, (candidate_ids, q_logits)
                )
            return SampledDrafts(draft_tokens, q_logits, verify_tokens)
        bs = block_hidden.shape[0]
        draft_hidden = block_hidden[:, 1:].reshape(bs * self.num_drafts, -1)
        with torch.no_grad():
            draft_logits = gather_and_crop_vocab(
                project_through_lm_head(draft_hidden, lm_head), lm_head
            ).view(bs, self.num_drafts, -1)
        draft_tokens = torch.stack(
            [sampler(draft_logits[:, j], j) for j in range(self.num_drafts)], dim=1
        )
        return SampledDrafts(draft_tokens, draft_logits, None)

    def warmup_draft_lattice(
        self, *, block_hidden: torch.Tensor, bonus_tokens: torch.Tensor
    ) -> DraftLattice | None:
        if not isinstance(self._dflash_model, DFlash2DraftModel):
            return None
        # Eager, warmup only: the same build_lattice the draft graph traces.
        with torch.no_grad():
            return build_lattice(self._dflash_model, block_hidden[:, 1:], bonus_tokens)


def _check_supported(
    *, draft_model: DFlashDraftModel, draft_config: DFlashDraftConfig
) -> None:
    # Core's TPU DFlash bypasses these silently; reject them instead.
    if draft_model.lilicorr is not None:
        raise NotImplementedError("TpuDflashDraftWorker does not support LiLiCorr.")
    if draft_config.is_domino:
        raise NotImplementedError("TpuDflashDraftWorker does not support Domino.")
    draft_model_path = get_spec().speculative_draft_model_path
    if draft_model_path is not None and os.path.exists(
        os.path.join(draft_model_path, "mask_embedding.pt")
    ):
        raise NotImplementedError(
            "TpuDflashDraftWorker does not merge mask_embedding.pt yet."
        )


def _resolve_mask_token_id(
    *, draft_config: DFlashDraftConfig, target_vocab_size: int
) -> int:
    # Core falls back to the tokenizer (and may grow it); the experimental path
    # needs the id pinned in the draft config, as DSpark does.
    mask_token_id = draft_config.mask_token_id
    if mask_token_id is None:
        raise NotImplementedError(
            "TpuDflashDraftWorker needs dflash_config.mask_token_id in the draft config."
        )
    if mask_token_id >= target_vocab_size:
        raise ValueError(
            f"DFlash mask_token_id={mask_token_id} is outside the target vocab "
            f"size {target_vocab_size}."
        )
    return mask_token_id

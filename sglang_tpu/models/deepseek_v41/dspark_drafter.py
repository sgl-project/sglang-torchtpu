# SPDX-License-Identifier: Apache-2.0
"""TpuBlockSpecWorker drafter for the DeepSeek-V4.1 DSpark MoE draft.

Imported lazily by the drafter factory (hooks.py), so loading the plugin does
not import the experimental speculative package.
"""

from __future__ import annotations

import torch

from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dspark_components.dspark_config import (
    resolve_runtime_config,
)
from sglang_tpu.experimental.speculative.block_draft_worker import (
    DraftLattice,
    DraftSampler,
    SampledDrafts,
    TpuBlockDraftWorker,
    target_input_embedding,
)
from sglang_tpu.models.deepseek_v41.dspark import (
    block_width,
    is_dsv41_dspark_draft,
    swa_commit_records,
    write_context_kv,
)

# CSA2 serves non-causal requests of at most 64 queries.
_MAX_BLOCK_ROWS = 64


class TpuDsv41DsparkDraftWorker(TpuBlockDraftWorker):
    def __init__(self, target_runner: ModelRunner) -> None:
        super().__init__(target_runner)
        model = self.model_runner.model
        if not is_dsv41_dspark_draft(model):
            raise TypeError(f"Expected the V4.1 DSpark draft, got {type(model)}.")
        runtime = resolve_runtime_config(
            draft_hf_config=self.model_runner.model_config.hf_config,
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            target_vocab_size=target_runner.model_config.vocab_size,
        )
        if model.gamma != runtime.gamma:
            raise ValueError(
                f"Draft config gamma={model.gamma} != runtime gamma={runtime.gamma}."
            )
        self.num_drafts = runtime.gamma
        self.block_width = block_width(model)
        window = int(model.config.sliding_window)
        if runtime.verify_num_draft_tokens > min(window, _MAX_BLOCK_ROWS):
            raise ValueError(
                f"DSpark block of {runtime.verify_num_draft_tokens} rows exceeds "
                f"the sliding window {window} or {_MAX_BLOCK_ROWS} CSA2 queries."
            )
        self.mask_token_id = runtime.mask_token_id
        self._dspark_model = model
        target_model = target_runner.model
        with self.draft_context():
            model.attach_shared_modules(
                embed_tokens=target_input_embedding(target_model),
                lm_head=unwrap_lora_layer(target_model.lm_head),
            )

    @property
    def full_to_swa(self) -> torch.Tensor:
        # The V4.1 pool keeps SWA pages, but is not an SWAKVPool.
        return self.model_runner.token_to_kv_pool_allocator.full_to_swa_index_mapping

    def precompile_draft(self) -> None:
        # The KV writer warms on the concatenated target aux hidden.
        main_proj = self._dspark_model.stages[0].main_proj
        self._compile_policy().dflash_target_hidden_size = int(main_proj.input_size)
        super().precompile_draft()

    def sample_drafts(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_lattice: DraftLattice | None,
        sampler: DraftSampler,
    ) -> SampledDrafts:
        del draft_lattice
        model = self._dspark_model
        bs = block_hidden.shape[0]
        draft_hidden = block_hidden if model.sample_from_anchor else block_hidden[:, 1:]
        with torch.no_grad():
            base_logits, _ = model.compute_base_logits(
                draft_hidden.reshape(bs * self.num_drafts, -1)
            )
        draft_tokens, draft_logits = model.markov_head.sample_block(
            base_logits.view(bs, self.num_drafts, -1),
            first_prev_tokens=bonus_tokens,
            hidden_states=draft_hidden,
            sampler=sampler,
            collect_corrected=True,
        )
        return SampledDrafts(draft_tokens, draft_logits, None)

    def _draft_model(self):
        return self._dspark_model

    def _write_draft_kv_eager(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: torch.Tensor | None = None,
        commit_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Callers pass SWA locs (full_to_swa above); publish_swa drops records < 64.
        return write_context_kv(
            self._dspark_model,
            self.model_runner.token_to_kv_pool,
            target_hidden=target_hidden,
            positions=positions,
            records=swa_commit_records(cache_loc, cache_loc_2d, commit_lens),
        )

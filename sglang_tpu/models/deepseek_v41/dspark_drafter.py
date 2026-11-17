# SPDX-License-Identifier: Apache-2.0
"""TpuBlockSpecWorker drafter for the DeepSeek-V4.1 DSpark MoE draft.

TpuDsparkDraftWorker already drives any DSpark draft through the interface
core DSparkWorkerV2 uses; this adds what the MoE draft needs on the CSA2
backend: the target's EP placement, CSA2's block limit, the SWA draft KV
writer, the eager draft embedding under attention DP, rank-local base logits
and the target's Engram history commit.

Imported lazily by the drafter factory (hooks.py), so loading the plugin does
not import the experimental speculative package.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dspark_components.dspark_config import (
    DSparkRuntimeConfig,
)
from sglang_tpu.experimental.speculative.dspark.dspark_draft_worker import (
    TpuDsparkDraftWorker,
)
from sglang_tpu.experimental.speculative.spec_types import AcceptResult, SpecState
from sglang_tpu.models.deepseek_v41 import engram_host
from sglang_tpu.models.deepseek_v41.dspark import (
    draft_base_logits,
    embed_draft_tokens_outside_graph,
    swa_commit_records,
    write_context_kv,
)

# CSA2 serves non-causal requests of at most 64 queries.
_MAX_BLOCK_ROWS = 64


class TpuDsv41DsparkDraftWorker(TpuDsparkDraftWorker):
    # The draft stages are DeepseekV4DecoderLayers: under attention DP they gather
    # tokens across DP ranks and run EP MoE like the target (vllm-torchtpu
    # epic/dsv41 builds them with the target's EP). Idle DP ranks already enter
    # the draft forward through _forward_idle_decode.
    _keeps_target_placement = True

    def _check_draft_model(self, draft_model: nn.Module) -> None:
        if not isinstance(draft_model, DeepseekV4ForCausalLMDSpark):
            raise TypeError(
                f"Expected the V4 DSpark MoE draft, got {type(draft_model)}."
            )

    def _check_runtime_config(self, runtime_config: DSparkRuntimeConfig) -> None:
        window = int(self._dspark_model.config.sliding_window)
        rows = runtime_config.verify_num_draft_tokens
        if rows > min(window, _MAX_BLOCK_ROWS):
            raise ValueError(
                f"DSpark block of {rows} rows exceeds the sliding window {window} "
                f"or {_MAX_BLOCK_ROWS} CSA2 queries."
            )

    def _attach_target_vocab(self, target_runner: ModelRunner) -> None:
        # attach_shared_modules also shards the Markov head over the draft's TP group.
        with self.draft_context():
            super()._attach_target_vocab(target_runner)

    def _base_logits(self, draft_hidden: torch.Tensor) -> torch.Tensor:
        # Rank-local at attention TP 1; core's compute_base_logits gathers over
        # the full TP group.
        return draft_base_logits(self._dspark_model, draft_hidden)

    @property
    def full_to_swa(self) -> torch.Tensor:
        # The V4.1 pool keeps SWA pages, but is not an SWAKVPool.
        return self.model_runner.token_to_kv_pool_allocator.full_to_swa_index_mapping

    def precompile_draft(self) -> None:
        # The KV writer warms on the concatenated target aux hidden.
        model = self._dspark_model
        main_proj = model.stages[0].main_proj
        self._compile_policy().dflash_target_hidden_size = int(main_proj.input_size)
        # After the policy installed its padding prepare, before warmup traces.
        if get_parallel().attn_dp_size > 1:
            model.prepare_forward_batch = embed_draft_tokens_outside_graph(
                model, model.prepare_forward_batch
            )
        super().precompile_draft()

    def after_commit(self, *, state: SpecState, accept: AcceptResult) -> None:
        """Advance the target's Engram history past the committed tokens.

        Core does it in DSparkWorkerV2 (update_after_verify); a verify forward
        commits nothing. Queued, not applied: the host history would read the
        verify outputs back now and block the host for the whole verify. The
        target's host Engram staging applies the queue before its next hash.
        """
        hasher = self._target_runner.ngram_embedding_manager.engram_hasher
        if hasher is None:
            return
        engram_host.defer_engram_commit(
            hasher,
            bonus_tokens=state.bonus_tokens,
            accept_tokens=accept.accept_tokens,
            accept_lens=accept.accept_lens,
            req_pool_indices=state.req_pool_indices,
        )

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

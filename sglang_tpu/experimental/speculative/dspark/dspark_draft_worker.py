"""DSpark drafter: the DFlash block plus a Markov head over the target lm_head.

draft: z_j = l_j + bias(d_{j-1}, h_j), d_0 = r, d_j ~ softmax(z_j / T), so
       q_j = softmax(z_j / T), with l_j = lm_head(h_j) the base logits.

As core DSparkWorkerV2, one drafter serves the dense DSparkDraftModel and the
DeepSeek-V4 MoE DeepseekV4ForCausalLMDSpark; the MoE draft keeps the target's
placement (draft_uses_compressed_kv_attention).

confidence: c_j = sigmoid(head([h_j, E(d_{j-1})]) / T_j), p(d_j accepted | prefix
accepted); TpuBlockSpecWorker schedules per-request verify caps from it.
"""

from __future__ import annotations

from typing import cast

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.configs.draft_parallel import DraftTPExecution
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.dspark_components.dspark_config import (
    draft_uses_compressed_kv_attention,
    resolve_runtime_config,
)
from sglang.srt.speculative.dspark_components.dspark_sts import (
    load_sts_calibration_from_path,
)
from sglang_tpu.experimental.speculative.block_draft_worker import (
    DraftLattice,
    DraftSampler,
    SampledDrafts,
    TpuBlockDraftWorker,
    target_input_embedding,
)
from sglang_tpu.experimental.speculative.spec_ops import schedule_verify_caps


class TpuDsparkDraftWorker(TpuBlockDraftWorker):
    def __init__(self, target_runner: ModelRunner) -> None:
        super().__init__(target_runner)
        draft_model = self.model_runner.model
        if not isinstance(draft_model, (DSparkDraftModel, DeepseekV4ForCausalLMDSpark)):
            raise TypeError(
                f"TpuDsparkDraftWorker needs a DSpark draft, got {type(draft_model)}."
            )
        runtime_config = resolve_runtime_config(
            draft_hf_config=self.model_runner.model_config.hf_config,
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            target_vocab_size=target_runner.model_config.vocab_size,
        )
        # The draft graph's markov chain (sample_draft_tokens_tpu) uses the config gamma.
        if draft_model.gamma != runtime_config.gamma:
            raise ValueError(
                f"Draft config gamma={draft_model.gamma} != runtime gamma={runtime_config.gamma}."
            )
        self.num_drafts = runtime_config.gamma
        # Anchor drafts read all gamma rows, so the block drops the last mask row.
        self.block_width = (
            self.num_drafts if draft_model.sample_from_anchor else self.num_drafts + 1
        )
        self.mask_token_id = runtime_config.mask_token_id
        self._dspark_model = draft_model
        self._load_sts_temperatures()
        # Confidence and the caps split in one program: eager, they dispatch about
        # ten small programs per step from the host.
        self._verify_caps_op = compiled_op(model=draft_model)(self._verify_caps)
        _attach_target_vocab(
            draft_model=draft_model,
            target_model=cast(nn.Module, target_runner.model),
            target_hf_text_config=target_runner.model_config.hf_text_config,
        )

    def _draft_tp_execution(self) -> DraftTPExecution:
        # As core DSparkWorkerV2: the MoE draft's stages gather over DP and run
        # the target's EP MoE, so it keeps the target placement.
        if draft_uses_compressed_kv_attention():
            return DraftTPExecution.TARGET_LAYOUT
        return super()._draft_tp_execution()

    @property
    def has_confidence_head(self) -> bool:
        return getattr(self._dspark_model, "confidence_head", None) is not None

    def _load_sts_temperatures(self) -> None:
        sts_path = get_spec().speculative_dspark_confidence_sts_path
        head = getattr(self._dspark_model, "confidence_head", None)
        if not sts_path or head is None:
            return
        temperatures = load_sts_calibration_from_path(sts_path).temperatures
        if len(temperatures) != self.num_drafts:
            raise ValueError(
                f"DSpark STS calibration has {len(temperatures)} temperatures, "
                f"gamma is {self.num_drafts}."
            )
        head.sts_temperatures = torch.tensor(
            temperatures, dtype=torch.float32, device=head.proj.weight.device
        )

    def verify_confidence(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """[pbs, gamma] float32 p(d_j accepted | d_1..d_{j-1} accepted), as core's compute_confidence."""
        draft_model = self._dspark_model
        head = draft_model.confidence_head
        assert head is not None
        draft_hidden = (
            block_hidden if draft_model.sample_from_anchor else block_hidden[:, 1:]
        )
        prev_embed = None
        if head.with_markov:
            prev_tokens = torch.cat(
                [bonus_tokens[:, None], draft_tokens[:, : self.num_drafts - 1]], dim=1
            )
            prev_embed = draft_model.markov_head.get_prev_embeddings(prev_tokens)
        with torch.no_grad():
            raw = head(draft_hidden, prev_embed)
        return torch.sigmoid(raw.float() / head.sts_temperatures)

    def verify_caps(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        live_mask: torch.Tensor,
        drafts_per_live_row: int,
    ) -> torch.Tensor:
        """[pbs] int32 per-request verify caps scheduled from the draft confidence."""
        return self._verify_caps_op(
            block_hidden, bonus_tokens, draft_tokens, live_mask, drafts_per_live_row
        )

    def _verify_caps(
        self,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        live_mask: torch.Tensor,
        drafts_per_live_row: int,
    ) -> torch.Tensor:
        confidence = self.verify_confidence(
            block_hidden=block_hidden,
            bonus_tokens=bonus_tokens,
            draft_tokens=draft_tokens,
        )
        return schedule_verify_caps(confidence, live_mask, drafts_per_live_row)

    def sample_drafts(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_lattice: DraftLattice | None,
        sampler: DraftSampler,
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
        draft_tokens, draft_logits = draft_model.markov_head.sample_block(
            base_logits.view(bs, self.num_drafts, -1),
            first_prev_tokens=bonus_tokens,
            hidden_states=draft_hidden,
            sampler=sampler,
            collect_corrected=True,
        )
        return SampledDrafts(draft_tokens, draft_logits, None)


def _attach_target_vocab(
    *,
    draft_model: DSparkDraftModel | DeepseekV4ForCausalLMDSpark,
    target_model: nn.Module,
    target_hf_text_config: PretrainedConfig,
) -> None:
    """Share the target embedding and lm_head with the draft, as DSparkWorkerV2."""
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

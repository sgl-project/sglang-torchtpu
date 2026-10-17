"""DSpark on the TpuSpecWorker interface.

draft:  one parallel draft forward of the block [r, m, ..., m] (m = mask token)
        over the draft context KV gives hidden h_1..h_gamma; base logits
        l_j = lm_head(h_j). The Markov head then corrects sequentially:
        z_j = l_j + bias(d_{j-1}, h_j), d_0 = r, d_j ~ softmax(z_j / T), so
        q_j = softmax(z_j / T).
verify: shared TpuSpecWorker.verify over [r, d_1, ..., d_gamma].
commit: project target features of the committed rows into the draft context
        KV; root := bonus.
extend: target forward on new prompt tokens, inject every row's target feature
        into the draft context KV, root := first sampled token.

Not ported yet: the confidence verify budget (num_verify_drafts is always
gamma), MoE/MLA drafts and hybrid (mamba) targets.
"""

from __future__ import annotations

from dataclasses import fields
from typing import cast

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.compilation.compiled_op import warmup_compiled_op
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.lora.layers import unwrap_lora_layer
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    enable_num_token_non_padded,
)
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.dspark import DSparkDraftModel
from sglang.srt.runtime_context import get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.draft_worker_common import (
    make_draft_block_spec_info,
    make_draft_input_v2,
)
from sglang.srt.speculative.dspark_components.dspark_config import (
    resolve_runtime_config,
)
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.recompilation_checker import (
    snapshot_recompilation_baseline,
)
from sglang_tpu.experimental.speculative.base_spec_worker import (
    TpuSpecWorker,
)
from sglang_tpu.experimental.speculative.dflash_info import HostPaddedDraftInput
from sglang_tpu.experimental.speculative.dspark.dspark_draft_worker import (
    TpuDsparkDraftWorker,
)
from sglang_tpu.experimental.speculative.spec_ops import (
    build_draft_block_inputs,
    sample_tokens,
)
from sglang_tpu.experimental.speculative.spec_types import (
    AcceptResult,
    BucketState,
    DraftResult,
    SpecState,
    TpuSpecVerifyInput,
    VerifyRows,
)
from sglang_tpu.experimental.speculative.spec_utils import pad_rows, prompt_positions


class TpuDsparkWorker(TpuSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        # The scheduler passes these to every spec worker; the draft reuses the target's.
        del server_args, gpu_id, nccl_port
        if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
            raise NotImplementedError(
                "TpuDsparkWorker only supports SGLANG_RAGGED_VERIFY_MODE=static."
            )
        target_model_config = target_worker.model_runner.model_config
        draft_worker = TpuDsparkDraftWorker(target_worker.model_runner)
        draft_runner = draft_worker.model_runner
        draft_model = cast(DSparkDraftModel, draft_runner.model)
        # cast does not check at runtime; a non-dense DSpark draft must fail here.
        if not isinstance(draft_model, DSparkDraftModel):
            raise TypeError(
                f"TpuDsparkWorker needs a dense DSparkDraftModel, got {type(draft_model)}."
            )
        if mambaish_config(target_model_config) is not None:
            raise NotImplementedError("TpuDsparkWorker does not support mamba targets.")
        runtime_config = resolve_runtime_config(
            draft_hf_config=draft_runner.model_config.hf_config,
            speculative_num_draft_tokens=get_spec().speculative_num_draft_tokens,
            target_vocab_size=target_model_config.vocab_size,
        )
        super().__init__(target_worker, draft_worker, runtime_config.gamma)
        # prepare_for_decode gathers the verify window and KV reserve at this width.
        if get_spec().speculative_num_draft_tokens != self.num_verify_rows:
            raise ValueError(
                f"speculative_num_draft_tokens={get_spec().speculative_num_draft_tokens} "
                f"must equal gamma + 1 = {self.num_verify_rows}."
            )
        # The draft graph's markov chain (sample_draft_tokens_tpu) uses the config gamma.
        if draft_model.gamma != self.num_drafts:
            raise ValueError(
                f"Draft config gamma={draft_model.gamma} != runtime gamma={self.num_drafts}."
            )
        self._dspark_draft_worker = draft_worker
        self._draft_runner = draft_runner
        self._draft_model = draft_model
        self._mask_token_id = runtime_config.mask_token_id
        # Anchor drafts forward [r, m x (gamma - 1)] and read all gamma rows;
        # the rest forward [r, m x gamma] and drop the anchor row.
        self._sample_from_anchor = draft_model.sample_from_anchor
        self._draft_block_width = (
            self.num_drafts if self._sample_from_anchor else self.num_verify_rows
        )
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=self._draft_block_width, device=self.device
        )
        _attach_target_vocab(
            draft_model=draft_model,
            target_model=cast(nn.Module, target_worker.model_runner.model),
            target_hf_text_config=target_model_config.hf_text_config,
        )
        # Filled per batch bucket once the target compile policy exists.
        self._num_verify_drafts: dict[int, torch.Tensor] = {}

    def init_cuda_graphs(self) -> None:
        super().init_cuda_graphs()
        # No confidence budget yet: every bucket verifies all gamma drafts.
        self._num_verify_drafts = {
            padded_bs: torch.full(
                (padded_bs,), self.num_drafts, dtype=torch.int32, device=self.device
            )
            for padded_bs in self._bucket_sizes()
        }
        self._warmup_draft_block_inputs()
        self._warmup_draft_sampling()
        # The base snapshot ran before these DSpark-only warmups.
        snapshot_recompilation_baseline()

    def draft(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> DraftResult:
        del batch
        state = bucket.state
        bs = state.seq_lens.shape[0]
        block_hidden, graph_draft_tokens = self._forward_draft_block(
            bucket=bucket, rows=rows, verify_input=verify_input
        )
        num_verify_drafts = self._num_verify_drafts[bs]
        temperatures, top_ks = bucket.temperatures, bucket.top_ks
        if temperatures is None or top_ks is None:
            # The draft graph already ran the greedy markov chain.
            return DraftResult(
                draft_tokens=self._tp_sync.sync(
                    SpecTpSyncSite.DSPARK_DRAFT_GREEDY, graph_draft_tokens
                ),
                draft_logits=None,
                num_verify_drafts=num_verify_drafts,
            )
        draft_tokens, draft_logits = self._sample_draft_block(
            block_hidden, state.bonus_tokens, temperatures, top_ks
        )
        return DraftResult(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            num_verify_drafts=num_verify_drafts,
        )

    def _sample_draft_block(
        self,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sampled [pbs, gamma] drafts and their logits from the [pbs, width, H] block."""
        # FIXME(junyanxu): sampled drafting runs eager until the sampling params can
        # enter the draft graph (sampling_info is an ignored direct-compile input).
        # draft_hidden[:, j - 1] predicts d_j at position n + j.
        bs = block_hidden.shape[0]
        draft_hidden = block_hidden if self._sample_from_anchor else block_hidden[:, 1:]
        with torch.no_grad():
            base_logits, _ = self._draft_model.compute_base_logits(
                draft_hidden.reshape(bs * self.num_drafts, -1)
            )

        def sampler(step_logits: torch.Tensor, step_idx: int) -> torch.Tensor:
            del step_idx
            tokens = sample_tokens(
                logits=step_logits, temperatures=temperatures, top_ks=top_ks
            )
            return self._tp_sync.sync(SpecTpSyncSite.DSPARK_DRAFT_SAMPLE, tokens)

        return self._draft_model.markov_head.sample_block(
            base_logits.view(bs, self.num_drafts, -1),
            first_prev_tokens=bonus_tokens,
            hidden_states=draft_hidden,
            sampler=sampler,
            collect_corrected=True,
        )

    def commit(
        self,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> SpecState:
        target_hidden = logits_output.hidden_states
        assert target_hidden is not None
        assert target_hidden.shape[0] == rows.cache_locs.numel(), (
            "verify hidden must keep the batch bucket rows"
        )
        # Padded rows commit 0 tokens, so the writer skips them.
        self._dspark_draft_worker.write_draft_kv(
            target_hidden=target_hidden,
            positions=rows.flat_positions,
            cache_loc=rows.flat_cache_locs,
            cache_loc_2d=rows.cache_locs,
            commit_lens=accept.accept_lens,
        )
        # Verify rows past the new seq_lens become dead target KV; freeing them is the allocator's job.
        return SpecState(
            bonus_tokens=accept.bonus_tokens,
            seq_lens=accept.next_seq_lens,
            req_pool_indices=state.req_pool_indices,
        )

    def extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        result = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        self._tp_sync.sync(SpecTpSyncSite.DSPARK_TARGET, result.next_token_ids)
        # Sampling ran at the batch bucket; the scheduler takes live rows.
        result.next_token_ids = result.next_token_ids[: batch.seq_lens.shape[0]]
        logits_output = result.logits_output
        target_hidden = logits_output.hidden_states
        if target_hidden is None:
            raise RuntimeError("DSpark extend requires target hidden capture.")
        assert logits_output.hidden_states_token_indices is None
        # finalize keeps HOST_PADDED hidden at the token bucket; pad the host rows to
        # match and upload once. Padded rows write slot 0, like the target forward.
        padded_rows = target_hidden.shape[0]
        cache_loc_cpu = batch.out_cache_loc
        assert cache_loc_cpu.device.type == "cpu", cache_loc_cpu.device
        positions = pad_rows(
            prompt_positions(
                prefix_lens=batch.prefix_lens, extend_lens=batch.extend_lens
            ),
            padded_rows,
        ).to(self.device, non_blocking=True)
        cache_loc = pad_rows(cache_loc_cpu, padded_rows).to(
            self.device, non_blocking=True
        )
        # Must inject before returning: the scheduler may update radix afterward,
        # invalidating out_cache_loc.
        self._dspark_draft_worker.write_draft_kv(
            target_hidden=target_hidden, positions=positions, cache_loc=cache_loc
        )
        logits_output.hidden_states = None
        next_state = SpecState(
            bonus_tokens=result.next_token_ids,
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
        )
        result.new_seq_lens = next_state.seq_lens
        result.next_draft_input = self.state_to_spec_info(next_state)
        return result

    def state_to_spec_info(self, state: SpecState) -> DFlashDraftInputV2:
        draft_input = make_draft_input_v2(
            bonus_tokens=state.bonus_tokens, new_seq_lens=state.seq_lens
        )
        return HostPaddedDraftInput(
            **{
                field.name: getattr(draft_input, field.name)
                for field in fields(draft_input)
                if field.init
            }
        )

    def get_spec_state(self, batch: ScheduleBatch) -> SpecState:
        draft_input = batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise TypeError(f"Expected DFlashDraftInputV2, got {type(draft_input)}.")
        return SpecState(
            bonus_tokens=draft_input.bonus_tokens,
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
        )

    def _forward_draft_block(
        self,
        *,
        bucket: BucketState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the block's [pbs, width, H] draft hidden and the graph's greedy [pbs, gamma] drafts."""
        state = bucket.state
        padded_bs = state.seq_lens.shape[0]
        bs = verify_input.seq_lens_cpu.shape[0]
        width = self._draft_block_width
        inputs = build_draft_block_inputs(
            state.bonus_tokens,
            rows,
            state.seq_lens,
            bucket.live_mask,
            self._mask_token_id,
            width,
        )
        # Padded rows carry kv len 0, so RPA keeps them inactive.
        kv_lens_cpu, kv_lens_sum = verify_input.padded_kv_lens_after_rows(width)
        num_live_tokens = bs * width
        num_token_non_padded = (
            torch.tensor(num_live_tokens, dtype=torch.int32, device=self.device)
            if enable_num_token_non_padded()
            else None
        )
        num_pad = padded_bs - bs
        # Writes the block's draft KV into the same slots target verify uses next.
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            out_cache_loc_is_physical=True,
            batch_size=padded_bs,
            input_ids=inputs.input_ids,
            req_pool_indices=state.req_pool_indices,
            req_pool_indices_cpu=verify_input.req_pool_indices_cpu,
            seq_lens=inputs.seq_lens,
            out_cache_loc=inputs.cache_locs,
            seq_lens_sum=kv_lens_sum,
            seq_lens_cpu=kv_lens_cpu,
            positions=inputs.positions,
            input_embeds=None,
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            # Pallas RPA plans the draft block as an extend from host lengths.
            extend_prefix_lens=inputs.extend_prefix_lens,
            extend_seq_lens=inputs.extend_seq_lens,
            extend_prefix_lens_cpu=verify_input.seq_lens_cpu.tolist() + [0] * num_pad,
            extend_seq_lens_cpu=[width] * bs + [0] * num_pad,
            extend_num_tokens=num_live_tokens,
            # Only the local field, as in the warmup batch: direct_compile pins
            # Tensor-vs-None per input.
            num_token_non_padded=num_token_non_padded,
            global_num_token_non_padded_cpu=num_live_tokens,
        )
        # Dynamo guards grad mode; the TPU warmup traced under no_grad.
        with torch.no_grad():
            draft_out = self._draft_runner.forward(forward_batch)
        hidden = draft_out.logits_output.hidden_states
        draft_tokens = draft_out.logits_output.next_draft_tokens
        if hidden is None or draft_tokens is None:
            raise RuntimeError(
                "DSpark draft forward returned no hidden or draft tokens."
            )
        return hidden.view(padded_bs, width, -1), draft_tokens

    def _bucket_sizes(self) -> tuple[int, ...]:
        compile_policy = self.target_worker.model_runner.compile_policy
        assert isinstance(compile_policy, DecoderTpuCompilePolicy)
        assert compile_policy.b_bucketer is not None
        return tuple(compile_policy.b_bucketer.buckets)

    def _warmup_draft_block_inputs(self) -> None:
        # Width and mask token come from the draft config, so the shared spec-op grid cannot warm it.
        with torch.no_grad():
            for padded_bs in self._bucket_sizes():
                block_shape = (padded_bs, self.num_verify_rows)
                warmup_compiled_op(
                    build_draft_block_inputs,
                    torch.zeros((padded_bs,), dtype=torch.int64, device=self.device),
                    VerifyRows(
                        positions=torch.zeros(
                            block_shape, dtype=torch.int64, device=self.device
                        ),
                        cache_locs=torch.zeros(
                            block_shape, dtype=torch.int64, device=self.device
                        ),
                        flat_positions=torch.zeros(
                            (padded_bs * self.num_verify_rows,),
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        flat_cache_locs=torch.zeros(
                            (padded_bs * self.num_verify_rows,),
                            dtype=torch.int64,
                            device=self.device,
                        ),
                    ),
                    torch.zeros((padded_bs,), dtype=torch.int64, device=self.device),
                    torch.ones((padded_bs,), dtype=torch.bool, device=self.device),
                    self._mask_token_id,
                    self._draft_block_width,
                )

    def _warmup_draft_sampling(self) -> None:
        """The eager sampled-draft chain, plus step's verify-token cat and logits cast."""
        hidden_size = self._draft_runner.model_config.hidden_size
        with torch.no_grad():
            for padded_bs in self._bucket_sizes():
                # Materialized like the draft graph's flat hidden output.
                block_hidden = (
                    torch.zeros(
                        (padded_bs * self._draft_block_width, hidden_size),
                        dtype=self._draft_runner.dtype,
                    )
                    .to(self.device)
                    .view(padded_bs, self._draft_block_width, -1)
                )
                bonus_tokens = torch.zeros(
                    (padded_bs,), dtype=torch.int64, device=self.device
                )
                draft_tokens, draft_logits = self._sample_draft_block(
                    block_hidden,
                    bonus_tokens,
                    torch.ones((padded_bs,), dtype=torch.float32).to(self.device),
                    torch.ones((padded_bs,), dtype=torch.int32).to(self.device),
                )
                draft_logits.float().cpu()
                torch.cat([bonus_tokens[:, None], draft_tokens], dim=1).cpu()

    # The base default reads draft_worker.draft_runners, which TpuDsparkDraftWorker lacks.
    def weight_update_runners(self) -> list[tuple[str, ModelRunner]]:
        return [("draft", self._draft_runner)]

    # FIXME(junyanxu): metrics_reporter calls this on every DSPARK decode log;
    # report the block-accept estimate once the confidence budget is ported.
    def block_accept_estimate_log_suffix(self) -> str | None:
        return None

    # FIXME(junyanxu): internal_state calls this on any DSPARK worker for server info;
    # return the info records once the confidence budget is ported.
    def dump_info_records(self) -> dict | None:
        return None


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
    draft_model.attach_shared_modules(
        embed_tokens=unwrap_lora_layer(target_model.get_input_embeddings()),
        lm_head=unwrap_lora_layer(target_model.lm_head),
    )

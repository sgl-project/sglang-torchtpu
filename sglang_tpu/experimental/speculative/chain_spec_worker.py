"""Autoregressive chain drafting (MTP, EAGLE3) on the TpuSpecWorker interface.

draft:  one compiled program per batch bucket: d_1 from the carried hidden g
        of the last committed row, then gamma - 1 decode-shaped draft forwards;
        step j reads (d_j, previous draft hidden) at position n + j - 1 and writes
        draft KV into verify slot j - 1.
verify: shared TpuSpecWorker.verify over [r, d_1, ..., d_gamma].
commit: one draft forward over the N + 1 verify rows, row j reading (target
        feature of row j, accepted token j), rewrites their draft KV; g := the
        draft hidden of the last accepted row.
extend: target forward on the prompt, then a draft forward on the prompt
        shifted by one token (the sampled token, or the next prompt token of a
        chunked request, closes each request); g := its last row.

g lives in a [req pool rows, H] device table, so filter/merge and the overlap
relay never carry it. Under PD, prefill ships g as the EAGLE hidden seed and
decode writes it back into the table. Not ported yet: a draft sliding window.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sglang.srt.compilation.compiled_op import compiled_op, warmup_compiled_op
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.runtime_context import get_disagg, get_spec, process_model_config
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang_tpu.experimental import register_pd_carry_seeder
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.chain_draft_input import (
    chain_draft_forward_input,
)
from sglang_tpu.experimental.speculative.chain_draft_worker import (
    TpuChainDraftWorker,
)
from sglang_tpu.experimental.speculative.chain_info import HostPaddedChainDraftInput
from sglang_tpu.experimental.speculative.draft_forward import forward_draft_rows
from sglang_tpu.experimental.speculative.eagle3.eagle3_draft_worker import (
    TpuEagle3DraftWorker,
)
from sglang_tpu.experimental.speculative.mtp.mtp_draft_worker import (
    TpuMtpDraftWorker,
)
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_drafts_target_only,
    build_chain_block_inputs,
    build_verify_tokens,
    finalize_accept,
    pick,
    select_verify_rows,
)
from sglang_tpu.experimental.speculative.spec_types import (
    AcceptResult,
    BucketState,
    CommitResult,
    DraftResult,
    SpecState,
    TpuSpecVerifyInput,
    VerifyRows,
)
from sglang_tpu.experimental.speculative.spec_utils import pad_rows, padded_h2d

_DRAFT_WORKER_CLS: dict[SpeculativeAlgorithm, type[TpuChainDraftWorker]] = {
    # MTP: core resolves NEXTN to EAGLE.
    SpeculativeAlgorithm.EAGLE: TpuMtpDraftWorker,
    SpeculativeAlgorithm.EAGLE3: TpuEagle3DraftWorker,
}


class TpuChainSpecWorker(TpuSpecWorker):
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
                "TpuChainSpecWorker only supports SGLANG_RAGGED_VERIFY_MODE=static."
            )
        # The verify window plan would lay draft KV out as DFlash's compact window.
        if get_spec().speculative_draft_window_size is not None:
            raise NotImplementedError(
                "TpuChainSpecWorker does not support --speculative-draft-window-size."
            )
        target_runner = target_worker.model_runner
        self._spec_algorithm = target_runner.spec_algorithm
        draft_worker = _DRAFT_WORKER_CLS[self._spec_algorithm](target_runner)
        super().__init__(target_worker, draft_worker, draft_worker.num_drafts)
        if get_spec().speculative_num_draft_tokens != self.num_verify_rows:
            raise ValueError(
                f"speculative_num_draft_tokens={get_spec().speculative_num_draft_tokens} "
                f"must equal gamma + 1 = {self.num_verify_rows}."
            )
        self._chain_draft_worker = draft_worker
        self._draft_runner = draft_worker.model_runner
        self._target_vocab_size = target_runner.model_config.vocab_size
        self._target_is_multimodal = process_model_config().is_multimodal
        # An idle DP rank mirrors its peers' draft syncs without knowing whether they
        # sample; greedy drafts sync one [pbs, N] int64 either way, and sampled
        # requests take the target-only accept, as TPU EAGLE does.
        self._sampled_drafts = not self._dp_attention
        self._disagg_mode = get_disagg().disaggregation_mode
        if self._disagg_mode == "decode":
            register_pd_carry_seeder(self._seed_carry)

    def init_cuda_graphs(self) -> None:
        super().init_cuda_graphs()
        # A PD prefill server never decodes.
        if self._disagg_mode != "prefill":
            self._warmup_draft_ops()
            if not self._sampled_drafts:
                self._warmup_target_only_accept()
        if self._disagg_mode == "decode":
            self._warmup_seed_carry()
        else:
            self._warmup_extend_ops()
            if self._target_is_multimodal:
                self._warmup_multimodal_prefill()
        if (
            self._disagg_mode != "prefill"
            and self.target_worker.model_runner.model_config.model_is_mrope
        ):
            self._warmup_verify_mrope()
        # The base snapshot ran before these chain-draft warmups.
        self._snapshot_warm_baseline()

    def draft(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> DraftResult:
        del batch
        # Host KV bound past the last draft row; RPA reads the exact length per step.
        seq_lens_cpu, _ = verify_input.padded_kv_lens_after_rows(self.num_drafts)
        sampled = self._sampled_drafts
        return self._draft_chain(
            state=bucket.state,
            rows=rows,
            req_pool_indices_cpu=verify_input.req_pool_indices_cpu,
            seq_lens_cpu=seq_lens_cpu,
            temperatures=bucket.temperatures if sampled else None,
            top_ks=bucket.top_ks if sampled else None,
        )

    def _draft_chain(
        self,
        *,
        state: SpecState,
        rows: VerifyRows,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_ks: torch.Tensor | None,
    ) -> DraftResult:
        """The fused draft and its eager prelude; warmup runs it as serving does."""
        padded_bs = state.seq_lens.shape[0]
        drafter = self._chain_draft_worker
        with drafter.draft_context():
            draft_tokens, draft_logits = self._run_fused_draft(
                state=state,
                rows=rows,
                req_pool_indices_cpu=req_pool_indices_cpu,
                seq_lens_cpu=seq_lens_cpu,
                temperatures=temperatures,
                top_ks=top_ks,
            )
        return DraftResult(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            num_verify_drafts=self._all_drafts[padded_bs],
        )

    def _run_fused_draft(
        self,
        *,
        state: SpecState,
        rows: VerifyRows,
        req_pool_indices_cpu: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_ks: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        padded_bs = state.seq_lens.shape[0]
        drafter = self._chain_draft_worker
        forward_batch = drafter.fused_draft_batch(
            spec_algorithm=self._spec_algorithm,
            input_ids=state.bonus_tokens,
            req_pool_indices=state.req_pool_indices,
            req_pool_indices_cpu=req_pool_indices_cpu,
            seq_lens=state.seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            out_cache_loc=rows.cache_locs[:, 0].contiguous(),
        )
        sampled = temperatures is not None and top_ks is not None
        uniforms = (
            # Ranks draw alike from rank 0's uniforms, so the fused chain needs no sync.
            self._tp_sync.sync(
                SpecTpSyncSite.SPEC_DRAFT_SAMPLE,
                torch.rand((padded_bs, self.num_drafts), device=self.device),
            )
            if sampled
            else None
        )
        draft_tokens, draft_logits = drafter.run_fused_draft(
            forward_batch,
            req_pool_indices=state.req_pool_indices,
            positions=rows.positions,
            cache_locs=rows.cache_locs,
            seq_lens=state.seq_lens,
            temperatures=temperatures if sampled else None,
            top_ks=top_ks if sampled else None,
            uniforms=uniforms,
        )
        if not sampled:
            draft_tokens = self._tp_sync.sync(
                SpecTpSyncSite.SPEC_DRAFT_GREEDY, draft_tokens
            )
        return draft_tokens, draft_logits

    def commit(
        self,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> CommitResult:
        del batch
        target_hidden = logits_output.hidden_states
        assert target_hidden is not None
        padded_bs = state.seq_lens.shape[0]
        width = self.num_verify_rows
        # Rows past the bonus carry token 0; their draft KV is past the new seq_lens.
        inputs = build_chain_block_inputs(
            accept.window_tokens, rows, state.seq_lens, accept.live_mask, 0
        )
        with self._chain_draft_worker.draft_context():
            draft_out = forward_draft_rows(
                draft_runner=self._draft_runner,
                spec_algorithm=self._spec_algorithm,
                inputs=inputs,
                req_pool_indices=state.req_pool_indices,
                verify_input=verify_input,
                offset=0,
                width=width,
                spec_info=chain_draft_forward_input(
                    hidden_states=target_hidden, width=width
                ),
                capture_hidden_mode=CaptureHiddenMode.FULL,
            )
        draft_hidden = draft_out.hidden_states
        assert draft_hidden is not None, "Chain draft rows returned no hidden."
        last_hidden = pick(
            draft_hidden.view(padded_bs, width, -1), accept.num_correct_drafts
        )
        self._chain_draft_worker.carry.index_copy_(
            0, state.req_pool_indices, last_hidden
        )
        next_state = SpecState(
            bonus_tokens=accept.bonus_tokens,
            seq_lens=accept.next_seq_lens,
            req_pool_indices=state.req_pool_indices,
        )
        # The draft rows' forward writes the step's last KV; last_hidden is its output.
        return CommitResult(state=next_state, kv_fence_tail=last_hidden)

    def extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        result = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        self._tp_sync.sync(SpecTpSyncSite.SPEC_TARGET, result.next_token_ids)
        if not batch.reqs:
            # A DP-attention rank without prompts (DP padding relabels its batch)
            # still issues the draft prefill's collectives.
            self._forward_idle_draft_prefill(
                batch, target_hidden=result.logits_output.hidden_states
            )
            return self._idle_extend_result(result)
        bs = batch.seq_lens.shape[0]
        # Sampling ran at the batch bucket; the draft reads it there, the scheduler takes live rows.
        bucket_tokens = result.next_token_ids
        result.next_token_ids = bucket_tokens[:bs]
        logits_output = result.logits_output
        target_hidden = logits_output.hidden_states
        if target_hidden is None:
            raise RuntimeError("Chain-draft extend requires target hidden capture.")
        assert logits_output.hidden_states_token_indices is None
        tail_hidden = self._extend_draft(
            batch,
            target_hidden=target_hidden,
            bucket_tokens=bucket_tokens,
            mm_input_embeds=logits_output.mm_input_embeds,
        )
        logits_output.hidden_states = None
        logits_output.mm_input_embeds = None
        next_state = SpecState(
            bonus_tokens=result.next_token_ids,
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
        )
        result.new_seq_lens = next_state.seq_lens
        draft_input = self.state_to_spec_info(next_state)
        # The draft prompt forward writes the step's last KV; tail_hidden is its output.
        draft_input.kv_fence_tail = tail_hidden
        if self._disagg_mode == "prefill":
            _attach_pd_seeds(draft_input, tail_hidden=tail_hidden, bs=bs)
        result.next_draft_input = draft_input
        return result

    def _extend_draft(
        self,
        batch: ScheduleBatch,
        *,
        target_hidden: torch.Tensor,
        bucket_tokens: torch.Tensor,
        mm_input_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Draft forward on the prompt shifted by one token; store and return each request's last hidden."""
        # finalize keeps HOST_PADDED target hidden (and mm embeds) at the token bucket.
        padded_tokens = target_hidden.shape[0]
        draft_ids = self._draft_prompt_ids(
            batch, padded_tokens=padded_tokens, bucket_tokens=bucket_tokens
        )
        forward_batch = ForwardBatch.init_new(
            batch,
            self._draft_runner,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            return_hidden_states_before_norm=False,
        )
        forward_batch.input_ids = draft_ids
        forward_batch.spec_info = chain_draft_forward_input(
            hidden_states=target_hidden, width=1
        )
        forward_batch.return_logprob = False
        if forward_batch.global_num_tokens_cpu is not None:
            # DP attention pads the batch to the synced live counts, but the tokens
            # are already at the token bucket, and the draft (run in its
            # attention-TP group) gathers nothing across DP ranks.
            num_ranks = len(forward_batch.global_num_tokens_cpu)
            forward_batch.global_num_tokens_cpu = [padded_tokens] * num_ranks
            forward_batch.global_num_tokens_gpu = torch.full(
                (num_ranks,), padded_tokens, dtype=torch.int64
            ).to(self.device, non_blocking=True)
        with self._chain_draft_worker.draft_context(), torch.no_grad():
            if mm_input_embeds is not None:
                assert mm_input_embeds.shape[0] == padded_tokens
                # Embeds in hand keep the draft off its mm_inputs path, whose
                # per-slot reads would specialize the program on batch packing.
                forward_batch.input_embeds = self._draft_prompt_embeds(
                    batch, mm_input_embeds=mm_input_embeds, draft_ids=draft_ids
                )
            draft_out = self._draft_runner.forward(forward_batch).logits_output
        _, tail_hidden = _draft_logits_and_hidden(draft_out)
        # LAST capture keeps one row per request at the extend cap; padded rows hit pool row 0.
        cap = tail_hidden.shape[0]
        pool_rows = pad_rows(batch.req_pool_indices_cpu, cap).to(
            self.device, non_blocking=True
        )
        self._chain_draft_worker.carry.index_copy_(0, pool_rows, tail_hidden)
        return tail_hidden

    def _seed_carry(
        self, req_pool_indices_cpu: torch.Tensor, hidden_cpu: torch.Tensor
    ) -> None:
        """Write a PD seed's [bs, H] host hidden into the carry table at the batch bucket."""
        padded_bs = self._pad_bs(req_pool_indices_cpu.shape[0])
        # Padded rows write the reserved pool row 0.
        self._chain_draft_worker.carry.index_copy_(
            0,
            padded_h2d(req_pool_indices_cpu.to(torch.int64), padded_bs, self.device),
            padded_h2d(hidden_cpu.to(self._draft_runner.dtype), padded_bs, self.device),
        )

    def _warmup_seed_carry(self) -> None:
        """_seed_carry at every batch bucket a PD seed batch pads to."""
        carry_size = self._chain_draft_worker.carry_size
        for padded_bs in self._b_bucketer().buckets:
            self._seed_carry(
                torch.zeros((padded_bs,), dtype=torch.int64),
                torch.zeros((padded_bs, carry_size), dtype=self._draft_runner.dtype),
            )

    def _draft_prompt_ids(
        self, batch: ScheduleBatch, *, padded_tokens: int, bucket_tokens: torch.Tensor
    ) -> torch.Tensor:
        """[padded_tokens] draft prompt ids on device: the prompt shifted by one token."""
        input_ids = batch.input_ids
        assert input_ids is not None and input_ids.device.type == "cpu", (
            "HOST_PADDED extend stages prompt tokens on host"
        )
        chunked_row, next_prompt_token = _chunked_tail(batch)
        shifted, row_reqs, is_sampled = shift_prompt_tokens(
            input_ids,
            extend_lens=batch.extend_lens,
            padded_tokens=padded_tokens,
            chunked_row=chunked_row,
            next_prompt_token=next_prompt_token,
            vocab_size=self._target_vocab_size,
        )
        return fill_sampled_tails(
            shifted.to(self.device, non_blocking=True),
            row_reqs.to(self.device, non_blocking=True),
            is_sampled.to(self.device, non_blocking=True),
            bucket_tokens,
        )

    def _draft_prompt_embeds(
        self,
        batch: ScheduleBatch,
        *,
        mm_input_embeds: torch.Tensor,
        draft_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Draft prompt embeds [T, H]: the target's moved up a row, each request's last row re-embedded."""
        cap = self._compile_policy()._extend_compile_cap()
        last_rows = torch.cumsum(torch.tensor(batch.extend_lens), dim=0) - 1
        # Padded slots repeat request 0's last row, rewriting it with the same embed.
        tail_rows = torch.full((cap,), int(last_rows[0]), dtype=torch.int64)
        tail_rows[: last_rows.shape[0]] = last_rows
        tail_rows = tail_rows.to(self.device, non_blocking=True)
        embed_tokens = self._draft_runner.model.get_input_embeddings()
        tail_embeds = embed_tokens(draft_ids.index_select(0, tail_rows))
        return fold_prompt_tail_embeds(mm_input_embeds, tail_rows, tail_embeds)

    def _forward_idle_draft_prefill(
        self, batch: ScheduleBatch, *, target_hidden: torch.Tensor
    ) -> None:
        forward_batch = ForwardBatch.init_new(
            batch,
            self._draft_runner,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            return_hidden_states_before_norm=False,
        )
        forward_batch.spec_info = chain_draft_forward_input(
            hidden_states=target_hidden[:0], width=1
        )
        forward_batch.return_logprob = False
        with self._chain_draft_worker.draft_context(), torch.no_grad():
            self._draft_runner.forward(forward_batch)

    def _idle_extend_result(
        self, result: GenerationBatchResult
    ) -> GenerationBatchResult:
        if result.logits_output is not None:
            result.logits_output.hidden_states = None
        idle = self._idle_result()
        result.next_token_ids = idle.next_token_ids
        result.new_seq_lens = idle.new_seq_lens
        result.next_draft_input = idle.next_draft_input
        return result

    def _forward_idle_step(
        self,
        *,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> None:
        # The active step's forwards in order: draft chain, verify, commit's draft rows.
        seq_lens_cpu, _ = verify_input.padded_kv_lens_after_rows(self.num_drafts)
        draft_result = self._draft_chain(
            state=state,
            rows=rows,
            req_pool_indices_cpu=verify_input.req_pool_indices_cpu,
            seq_lens_cpu=seq_lens_cpu,
            temperatures=None,
            top_ks=None,
        )
        logits_output = self._forward_idle_verify(
            batch=batch,
            state=state,
            rows=rows,
            verify_input=verify_input,
            verify_tokens=build_verify_tokens(
                state.bonus_tokens, draft_result.draft_tokens
            ),
        )
        accept = finalize_accept(
            draft_result.draft_tokens,
            self._device_zeros(state.seq_lens.shape[0]),
            state.bonus_tokens,
            state.seq_lens,
            state.req_pool_indices != 0,
        )
        self.commit(batch, state, rows, verify_input, logits_output, accept)

    def make_verify_input(
        self, *, batch: ScheduleBatch, verify_tokens: torch.Tensor, rows: VerifyRows
    ) -> DFlashVerifyInput:
        return DFlashVerifyInput(
            draft_token=verify_tokens,
            positions=rows.flat_positions,
            draft_token_num=self.num_verify_rows,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            live_seq_lens_cpu=batch.seq_lens_cpu,
        )

    def state_to_spec_info(self, state: SpecState) -> HostPaddedChainDraftInput:
        return HostPaddedChainDraftInput(
            bonus_tokens=state.bonus_tokens.to(torch.int64),
            new_seq_lens=state.seq_lens.to(torch.int64),
        )

    def get_spec_state(self, batch: ScheduleBatch) -> SpecState:
        draft_input = batch.spec_info
        if not isinstance(draft_input, HostPaddedChainDraftInput):
            raise TypeError(
                f"Expected HostPaddedChainDraftInput, got {type(draft_input)}."
            )
        return SpecState(
            bonus_tokens=draft_input.bonus_tokens,
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
        )

    def _warmup_draft_ops(self) -> None:
        """Eager draft/commit ops around the compiled draft programs, per batch bucket."""
        drafter = self._chain_draft_worker
        width = self.num_verify_rows
        with torch.no_grad():
            for pbs in self._b_bucketer().buckets:
                self._warmup_draft_chain(pbs)
                rows = _dummy_verify_rows(pbs, width, self.device)
                accept_tokens = torch.zeros(
                    (pbs, width), dtype=torch.int64, device=self.device
                )
                warmup_compiled_op(
                    build_chain_block_inputs,
                    # Flat, as AcceptResult.window_tokens arrives.
                    accept_tokens.reshape(-1),
                    rows,
                    torch.zeros((pbs,), dtype=torch.int64, device=self.device),
                    torch.ones((pbs,), dtype=torch.bool, device=self.device),
                    0,
                )
                # commit: the last accepted row's hidden back into the table.
                rows_hidden = torch.zeros(
                    (pbs * width, drafter.carry_size),
                    dtype=self._draft_runner.dtype,
                ).to(self.device)
                last = pick(
                    rows_hidden.view(pbs, width, -1),
                    torch.zeros((pbs,), dtype=torch.int64, device=self.device),
                )
                pool_rows = torch.zeros((pbs,), dtype=torch.int64).to(self.device)
                drafter.carry.index_copy_(0, pool_rows, last)

    def _warmup_draft_chain(self, pbs: int) -> None:
        """draft's eager prelude fuses with select_verify_rows' layout, so replay both."""

        def upload(*shape: int, dtype: torch.dtype = torch.int64) -> torch.Tensor:
            return torch.zeros(shape, dtype=dtype).to(self.device, non_blocking=True)

        width = self.num_verify_rows
        seq_lens = upload(pbs)
        rows = select_verify_rows(upload(pbs, 2 * width), upload(pbs), seq_lens)
        state = SpecState(
            bonus_tokens=upload(pbs), seq_lens=seq_lens, req_pool_indices=upload(pbs)
        )
        variants: list[tuple[torch.Tensor | None, torch.Tensor | None]] = [(None, None)]
        if self._sampled_drafts:
            variants.append(
                (upload(pbs, dtype=torch.float32), upload(pbs, dtype=torch.int32))
            )
        for temperatures, top_ks in variants:
            result = self._draft_chain(
                state=state,
                rows=rows,
                req_pool_indices_cpu=torch.zeros((pbs,), dtype=torch.int64),
                seq_lens_cpu=torch.full((pbs,), width, dtype=torch.int64),
                temperatures=temperatures,
                top_ks=top_ks,
            )
            result.draft_tokens.cpu()
            if result.draft_logits is not None:
                self._warmup_sampled_accept(
                    pbs, result.draft_tokens, result.draft_logits
                )

    def _warmup_target_only_accept(self) -> None:
        """Target-only acceptance, target filter included, per batch bucket."""
        vocab_size = self.target_worker.model_runner.model_config.vocab_size
        with torch.no_grad():
            for padded_bs in self._b_bucketer().buckets:
                warmup_compiled_op(
                    accept_drafts_target_only,
                    torch.zeros(
                        (padded_bs * self.num_verify_rows, vocab_size),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    torch.zeros(
                        (padded_bs, self.num_drafts),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    self._all_drafts[padded_bs],
                    *self._sampling_param_warmup_args(padded_bs),
                    self._accept_threshold_single,
                    self._accept_threshold_acc,
                )

    def _warmup_multimodal_prefill(self) -> None:
        """The draft prompt's embed fold per token bucket, and live-row logits padding."""
        compile_policy = self._compile_policy()
        cap = compile_policy._extend_compile_cap()
        assert compile_policy.t_bucketer is not None
        embed_tokens = self._draft_runner.model.get_input_embeddings()
        tail_rows = torch.zeros((cap,), dtype=torch.int64).to(self.device)
        with torch.no_grad():
            for num_tokens in compile_policy.t_bucketer.buckets:
                if num_tokens < cap:
                    continue
                draft_ids = torch.zeros((num_tokens,), dtype=torch.int64).to(
                    self.device
                )
                with self._chain_draft_worker.draft_context():
                    tail_embeds = embed_tokens(draft_ids.index_select(0, tail_rows))
                warmup_compiled_op(
                    fold_prompt_tail_embeds,
                    torch.zeros(
                        (num_tokens, embed_tokens.embedding_dim),
                        dtype=embed_tokens.weight.dtype,
                        device=self.device,
                    ),
                    tail_rows,
                    tail_embeds,
                )
            # A multimodal wrapper's eager logits processor emits live rows;
            # TpuSampler pads them to the batch bucket.
            vocab_size = self._target_vocab_size
            for bs in range(1, cap + 1):
                logits = torch.zeros((bs, vocab_size), dtype=torch.float32).to(
                    self.device
                )
                torch.nn.functional.pad(logits, (0, 0, 0, self._pad_bs(bs) - bs))
        torch.accelerator.synchronize()

    def _warmup_verify_mrope(self) -> None:
        """compute_spec_mrope_positions on the verify rows: text-only, then per-request deltas."""
        width = self.num_verify_rows

        def upload(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.int64).to(self.device)

        with torch.no_grad():
            for padded_bs in self._b_bucketer().buckets:
                # make_verify_input hands over select_verify_rows' flat positions.
                rows = select_verify_rows(
                    upload(padded_bs, 2 * width), upload(padded_bs), upload(padded_bs)
                )
                positions = rows.flat_positions.view(padded_bs, -1)
                positions.flatten().unsqueeze(0).repeat(3, 1).cpu()
                deltas = torch.zeros((padded_bs, 1), dtype=torch.int64)
                (positions + deltas.to(self.device)).flatten().unsqueeze(0).repeat(
                    3, 1
                ).cpu()

    def _warmup_extend_ops(self) -> None:
        """Host uploads and the tail fill of the draft prompt extend, per token and batch bucket."""
        compile_policy = self._compile_policy()
        cap = compile_policy._extend_compile_cap()
        t_bucketer = compile_policy.t_bucketer
        assert t_bucketer is not None
        token_buckets = [t for t in t_bucketer.buckets if t >= cap]
        batch_buckets = sorted({self._pad_bs(bs) for bs in range(1, cap + 1)})
        with torch.no_grad():
            for padded_tokens in token_buckets:
                shifted = torch.zeros((padded_tokens,), dtype=torch.int64).to(
                    self.device
                )
                row_reqs = torch.zeros((padded_tokens,), dtype=torch.int64).to(
                    self.device
                )
                is_sampled = torch.zeros((padded_tokens,), dtype=torch.bool).to(
                    self.device
                )
                for pbs in batch_buckets:
                    warmup_compiled_op(
                        fill_sampled_tails,
                        shifted,
                        row_reqs,
                        is_sampled,
                        torch.zeros((pbs,), dtype=torch.int64, device=self.device),
                    )
            pool_rows = torch.zeros((cap,), dtype=torch.int64).to(self.device)
            tail_hidden = torch.zeros(
                (cap, self._chain_draft_worker.carry_size),
                dtype=self._draft_runner.dtype,
            ).to(self.device)
            self._chain_draft_worker.carry.index_copy_(0, pool_rows, tail_hidden)


@compiled_op()
def fill_sampled_tails(
    shifted: torch.Tensor,
    row_reqs: torch.Tensor,
    is_sampled: torch.Tensor,
    bucket_tokens: torch.Tensor,
) -> torch.Tensor:
    """[T] draft prompt ids: rows closing a finished prompt take that request's sampled token."""
    return torch.where(is_sampled, bucket_tokens.index_select(0, row_reqs), shifted)


@compiled_op()
def fold_prompt_tail_embeds(
    mm_input_embeds: torch.Tensor, tail_rows: torch.Tensor, tail_embeds: torch.Tensor
) -> torch.Tensor:
    """[T, H] embeds moved up by one row (row p embeds token p + 1), tail rows replaced."""
    shifted = torch.cat((mm_input_embeds[1:], mm_input_embeds[-1:]), dim=0)
    return shifted.index_copy(0, tail_rows, tail_embeds.to(shifted.dtype))


def shift_prompt_tokens(
    input_ids: torch.Tensor,
    *,
    extend_lens: Sequence[int],
    padded_tokens: int,
    chunked_row: int | None,
    next_prompt_token: int | None,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host [padded_tokens] shifted ids, owning request per row, and rows awaiting the sampled token.

    Row t of a request reads token t + 1; its last row reads the token the
    target just sampled, or the next prompt token when the request is chunked.
    """
    num_tokens = input_ids.shape[0]
    shifted = torch.zeros((padded_tokens,), dtype=torch.int64)
    # Image placeholders carry hash ids past the vocab; the draft reads those rows
    # from the target's embeds, but must not index its embedding out of range.
    shifted[: num_tokens - 1] = input_ids[1:].clamp(0, vocab_size - 1)
    lens = torch.tensor(list(extend_lens), dtype=torch.int64)
    last_rows = torch.cumsum(lens, dim=0) - 1
    row_reqs = torch.zeros((padded_tokens,), dtype=torch.int64)
    row_reqs[:num_tokens] = torch.repeat_interleave(torch.arange(lens.shape[0]), lens)
    is_sampled = torch.zeros((padded_tokens,), dtype=torch.bool)
    is_sampled[last_rows] = True
    if chunked_row is not None:
        assert next_prompt_token is not None
        tail = int(last_rows[chunked_row])
        is_sampled[tail] = False
        shifted[tail] = next_prompt_token
    return shifted, row_reqs, is_sampled


def _attach_pd_seeds(
    draft_input: HostPaddedChainDraftInput, *, tail_hidden: torch.Tensor, bs: int
) -> None:
    """The per-request seed core PD prefill ships to decode (output_topk_*, hidden_states_tensor).

    A chain decode stores only the hidden and re-derives d_1 from it, so topk is
    a host placeholder. hidden stays at the extend cap: core copies it to host
    whole and reads rows by request, so no slice compiles per bs.
    """
    draft_input.hidden_states = tail_hidden
    draft_input.topk_p = torch.ones((bs, 1), dtype=torch.float32)
    draft_input.topk_index = torch.zeros((bs, 1), dtype=torch.int64)


def _chunked_tail(batch: ScheduleBatch) -> tuple[int | None, int | None]:
    """Index of the chunked request in batch.reqs and its next prompt token, as core EAGLE reads them."""
    next_prompt_token = batch.chunked_req_next_prompt_token
    if next_prompt_token is None:
        return None, None
    for i, req in enumerate(batch.reqs):
        if req is batch.chunked_req:
            return i, int(next_prompt_token)
    return None, None


def _draft_logits_and_hidden(
    logits_output: LogitsProcessorOutput,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits, hidden = logits_output.next_token_logits, logits_output.hidden_states
    if logits is None or hidden is None:
        raise RuntimeError("Chain draft forward returned no logits or hidden.")
    return logits, hidden


def _dummy_verify_rows(pbs: int, width: int, device: torch.device) -> VerifyRows:
    """Distinct tensors, as select_verify_rows returns: Dynamo guards on aliased inputs."""

    def zeros(*shape: int) -> torch.Tensor:
        return torch.zeros(shape, dtype=torch.int64, device=device)

    return VerifyRows(
        positions=zeros(pbs, width),
        cache_locs=zeros(pbs, width),
        flat_positions=zeros(pbs * width),
        flat_cache_locs=zeros(pbs * width),
    )

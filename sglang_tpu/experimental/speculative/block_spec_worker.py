"""Block-parallel drafting (DFlash, DSpark) on the TpuSpecWorker interface.

draft:  one parallel draft forward of the block [r, m, ..., m] (m = mask token)
        over the draft context KV gives hidden h_1..h_gamma; the drafter turns
        them into d, q (TpuBlockDraftWorker.sample_drafts).
verify: shared TpuSpecWorker.verify over [r, d_1, ..., d_gamma].
commit: project target features of the committed rows into the draft context
        KV; root := bonus.
extend: target forward on new prompt tokens, inject every row's target feature
        into the draft context KV, root := first sampled token.
"""

from __future__ import annotations

import functools
import itertools

import torch

from sglang.srt.compilation.compiled_op import warmup_compiled_op
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang_tpu.compilation import slot_metadata
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.recompilation_checker import (
    snapshot_recompilation_baseline,
)
from sglang_tpu.experimental.speculative.base_spec_worker import (
    TpuSpecWorker,
)
from sglang_tpu.experimental.speculative.block_draft_worker import (
    DraftLattice,
    DraftSampling,
    SampledDrafts,
    TpuBlockDraftWorker,
)
from sglang_tpu.experimental.speculative.block_info import HostPaddedBlockDraftInput
from sglang_tpu.experimental.speculative.dflash.dflash_draft_worker import (
    TpuDflashDraftWorker,
)
from sglang_tpu.experimental.speculative.draft_forward import forward_draft_rows
from sglang_tpu.experimental.speculative.dspark.dspark_draft_worker import (
    TpuDsparkDraftWorker,
)
from sglang_tpu.experimental.speculative.spec_ops import (
    TARGET_CANDIDATES,
    build_draft_block_inputs,
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
from sglang_tpu.experimental.speculative.spec_utils import (
    pad_rows,
    padded_h2d,
    prompt_positions,
)
from sglang_tpu.sampler import distributed_top_k_candidate_width_for_model

_DRAFT_WORKER_CLS: dict[SpeculativeAlgorithm, type[TpuBlockDraftWorker]] = {
    SpeculativeAlgorithm.DFLASH: TpuDflashDraftWorker,
    SpeculativeAlgorithm.DSPARK: TpuDsparkDraftWorker,
}


def committed_verify_ids(
    bonus_tokens: torch.Tensor, accept_tokens: torch.Tensor
) -> torch.Tensor:
    """[pbs, N + 2] tokens whose first accept_lens columns are the committed verify rows.

    The verify window is [r, d_1..d_N]; the committed prefix is r and the correct
    drafts, which accept_tokens holds first (then the new bonus, -1 padded).
    """
    return torch.cat([bonus_tokens.view(-1, 1), accept_tokens.clamp_min(0)], dim=1)


class TpuBlockSpecWorker(TpuSpecWorker):
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
                "TpuBlockSpecWorker only supports SGLANG_RAGGED_VERIFY_MODE=static."
            )
        target_runner = target_worker.model_runner
        self._spec_algorithm = target_runner.spec_algorithm
        draft_worker = _DRAFT_WORKER_CLS[self._spec_algorithm](target_runner)
        super().__init__(target_worker, draft_worker, draft_worker.num_drafts)
        # prepare_for_decode gathers the verify window and KV reserve at this width.
        if get_spec().speculative_num_draft_tokens != self.num_verify_rows:
            raise ValueError(
                f"speculative_num_draft_tokens={get_spec().speculative_num_draft_tokens} "
                f"must equal gamma + 1 = {self.num_verify_rows}."
            )
        self._block_draft_worker = draft_worker
        self._draft_runner = draft_worker.model_runner
        self._mask_token_id = draft_worker.mask_token_id
        self._draft_block_width = draft_worker.block_width
        self._draft_block_spec_info = make_draft_block_spec_info(
            draft_token_num=self._draft_block_width, device=self.device
        )

    def init_cuda_graphs(self) -> None:
        super().init_cuda_graphs()
        # Under PP (pp_size > 1), TpuBlockSpecWorker only runs on the last stage
        # of a non-decode (PD prefill) server, which projects extend draft KV and
        # never drafts or samples speculative blocks. Scope this skip strictly to
        # pp_size > 1 and disaggregation_mode != "decode" so PP prefill avoids
        # single-stage decode compilations and unsharded-lm_head draft-sampling
        # HBM spikes on the last stage, while keeping all pp_size == 1 (TP /
        # monolithic) and decode paths completely unchanged.
        is_pp_non_decode = (
            get_parallel().pp_size > 1 and get_disagg().disaggregation_mode != "decode"
        )
        if not is_pp_non_decode:
            self._warmup_draft_block_inputs()
            if self._sampled_drafts:
                self._warmup_draft_sampling()
        self._warmup_extend_draft_inputs()
        self._warmup_folded_drop_draft_kv()
        if not is_pp_non_decode:
            self._warmup_engram_commit_ids()
        if _fused_mixed() and self._dp_attention:
            self._warmup_mixed_fused()
        # The base snapshot ran before these block-draft warmups.
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
        block_hidden, graph_draft_tokens, draft_lattice = self._forward_draft_block(
            state=state,
            live_mask=bucket.live_mask,
            rows=rows,
            verify_input=verify_input,
        )
        # No confidence budget yet: every bucket verifies all gamma drafts.
        num_verify_drafts = self._all_drafts[bs]
        temperatures, top_ks = bucket.temperatures, bucket.top_ks
        if temperatures is None or top_ks is None or not self._sampled_drafts:
            # The draft graph already drafted greedily (sample_draft_tokens_tpu).
            return DraftResult(
                draft_tokens=self._sync_greedy_drafts(graph_draft_tokens),
                draft_logits=None,
                num_verify_drafts=num_verify_drafts,
            )
        sampled = self._sample_draft_block(
            block_hidden,
            state.bonus_tokens,
            draft_lattice,
            temperatures,
            top_ks,
            target_candidates=bucket.target_candidates,
        )
        return DraftResult(
            draft_tokens=sampled.draft_tokens,
            draft_logits=sampled.draft_logits,
            num_verify_drafts=num_verify_drafts,
            verify_tokens=sampled.verify_tokens,
            draft_candidates=sampled.draft_candidates,
        )

    def _sample_draft_block(
        self,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_lattice: DraftLattice | None,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        *,
        target_candidates: bool,
    ) -> SampledDrafts:
        """Sampled [pbs, gamma] drafts and their logits from the [pbs, width, H] block."""
        sampling = DraftSampling(
            temperatures=temperatures,
            top_ks=top_ks,
            sync=functools.partial(
                self._tp_sync.sync, SpecTpSyncSite.SPEC_DRAFT_SAMPLE
            ),
            rng_step=self._rng_step,
            target_candidates=target_candidates,
        )
        with self._block_draft_worker.draft_context():
            return self._block_draft_worker.sample_drafts(
                block_hidden=block_hidden,
                bonus_tokens=bonus_tokens,
                draft_lattice=draft_lattice,
                sampler=sampling,
            )

    def _sync_greedy_drafts(self, draft_tokens: torch.Tensor) -> torch.Tensor:
        return self._tp_sync.sync(SpecTpSyncSite.SPEC_DRAFT_GREEDY, draft_tokens)

    def commit(
        self,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> CommitResult:
        target_hidden = logits_output.hidden_states
        assert target_hidden is not None
        assert target_hidden.shape[0] == rows.cache_locs.numel(), (
            "verify hidden must keep the batch bucket rows"
        )
        swa_candidates = self._draft_swa_candidates(verify_input)
        if swa_candidates is not None:
            rows = select_verify_rows(
                padded_h2d(swa_candidates, rows.cache_locs.shape[0], self.device),
                verify_input.committed_lens,
                state.seq_lens,
                self.num_verify_rows,
            )
        # Padded rows commit 0 tokens, so the writer skips them.
        with self._block_draft_worker.draft_context():
            kv_fence_tail = self._block_draft_worker.write_draft_kv(
                target_hidden=target_hidden,
                positions=rows.flat_positions,
                cache_loc=rows.flat_cache_locs,
                cache_loc_2d=rows.cache_locs,
                commit_lens=accept.accept_lens,
            )
        # As core DSparkWorkerV2: a verify forward commits nothing to the target's
        # per-request Engram history.
        ngram = self.target_worker.model_runner.ngram_embedding_manager
        if ngram.engram_hasher is not None:
            ngram.update_after_verify(
                verify_ids_2d=committed_verify_ids(
                    state.bonus_tokens, accept.accept_tokens
                ),
                req_pool_indices=state.req_pool_indices,
                commit_lens=accept.accept_lens,
            )
        # Verify rows past the new seq_lens become dead target KV; freeing them is the allocator's job.
        next_state = SpecState(
            bonus_tokens=accept.bonus_tokens,
            seq_lens=accept.next_seq_lens,
            req_pool_indices=state.req_pool_indices,
        )
        return CommitResult(state=next_state, kv_fence_tail=kv_fence_tail)

    def _draft_swa_candidates(
        self, verify_input: TpuSpecVerifyInput
    ) -> torch.Tensor | None:
        """Host [bs, 2 * capacity] int64 draft SWA slots of the verify candidate
        window, or None when the draft writes the target's slots."""
        # The window starts at the host committed base; select_verify_rows picks the
        # active rows on device, where seq_lens may lead the host by one commit.
        ring_pool = self._block_draft_worker.swa_ring_pool
        if ring_pool is not None:
            bs = verify_input.seq_lens_cpu.shape[0]
            offsets = torch.arange(2 * self.verify_window_rows, dtype=torch.int64)
            return ring_pool.swa_ring_rows(
                slots=verify_input.req_pool_indices_cpu[:bs, None],
                positions=verify_input.seq_lens_cpu.to(torch.int64)[:, None] + offsets,
            )
        full_to_swa = self._block_draft_worker.full_to_swa
        if full_to_swa is None:
            return None
        return full_to_swa[verify_input.verify_cache_candidates_cpu].to(torch.int64)

    def extend(
        self, batch: ScheduleBatch, *, pp_proxy_tensors: PPProxyTensors | None
    ) -> GenerationBatchResult:
        result = self.target_worker.forward_batch_generation(
            batch,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        result.next_token_ids = self.sync_target_tokens(result.next_token_ids)
        # Sampling ran at the batch bucket; the scheduler takes live rows.
        result.next_token_ids = result.next_token_ids[: batch.seq_lens.shape[0]]
        logits_output = result.logits_output
        target_hidden = logits_output.hidden_states
        if target_hidden is None:
            raise RuntimeError("Block-draft extend requires target hidden capture.")
        assert logits_output.hidden_states_token_indices is None
        kv_fence_tail = self._write_extend_draft_kv(batch, target_hidden)
        logits_output.hidden_states = None
        if not batch.reqs:
            return self._idle_extend_result(result)
        return self._extend_result(batch, result, kv_fence_tail)

    def _write_extend_draft_kv(
        self, batch: ScheduleBatch | None, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Inject an extend's [token bucket, F] target hidden into the draft
        context KV; returns the kv_fence_tail. ``batch`` None or without
        requests: all rows are padding (a DP rank without prompts)."""
        # finalize keeps HOST_PADDED hidden at the token bucket; pad the host rows to
        # match and upload once. Padded rows write slot 0, like the target forward.
        padded_rows = target_hidden.shape[0]
        if batch is None or not batch.reqs:
            # A DP-attention rank without prompts (DP padding relabels its batch)
            # still issues the draft KV write; all its rows are padding.
            compile_policy = self.target_worker.model_runner.compile_policy
            assert isinstance(compile_policy, DecoderTpuCompilePolicy)
            positions_cpu = torch.zeros((padded_rows,), dtype=torch.int64)
            # The writer was warmed with the allocator's dtype; idle batches carry int64.
            cache_loc_cpu = torch.zeros(
                (padded_rows,), dtype=compile_policy.live_cache_loc_dtype()
            )
        else:
            positions_cpu = prompt_positions(
                prefix_lens=batch.prefix_lens, extend_lens=batch.extend_lens
            )
            cache_loc_cpu = batch.out_cache_loc
            assert cache_loc_cpu.device.type == "cpu", cache_loc_cpu.device
            ring_pool = self._block_draft_worker.swa_ring_pool
            full_to_swa = self._block_draft_worker.full_to_swa
            if ring_pool is not None:
                assert not self.ships_pd_hidden, "PD does not serve a draft SWA ring"
                cache_loc_cpu = _extend_ring_rows(
                    ring_pool=ring_pool, batch=batch, positions=positions_cpu
                )
                cache_loc_cpu = self._skip_rows_below_decoder_replay(
                    batch, positions_cpu, cache_loc_cpu
                ).to(batch.out_cache_loc.dtype)
            elif full_to_swa is not None and not self.ships_pd_hidden:
                cache_loc_cpu = full_to_swa[cache_loc_cpu].to(cache_loc_cpu.dtype)
        positions = pad_rows(positions_cpu, padded_rows).to(
            self.device, non_blocking=True
        )
        cache_loc = pad_rows(cache_loc_cpu, padded_rows).to(
            self.device, non_blocking=True
        )
        # Must inject before returning: the scheduler may update radix afterward,
        # invalidating out_cache_loc.
        if self.ships_pd_hidden:
            # The decode projects these rows into its draft KV.
            return self.write_pd_hidden(batch, target_hidden)
        with self._block_draft_worker.draft_context():
            return self._block_draft_worker.write_draft_kv(
                target_hidden=target_hidden,
                positions=positions,
                cache_loc=cache_loc,
            )

    def _extend_result(
        self,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
        kv_fence_tail: torch.Tensor,
    ) -> GenerationBatchResult:
        """An extend's result: root := the sampled token, after the draft KV write."""
        next_state = SpecState(
            bonus_tokens=result.next_token_ids,
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
        )
        result.new_seq_lens = next_state.seq_lens
        next_draft_input = self.state_to_spec_info(next_state)
        next_draft_input.kv_fence_tail = kv_fence_tail
        result.next_draft_input = next_draft_input
        return result

    def inject_mixed_prompt(
        self, prompt: ScheduleBatch | None, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        # The prompt rows of the MIXED forward, folded replay rows dropped, at the
        # prompt's extend token bucket: as extend's write after drop_folded_rows.
        if self.ships_pd_hidden:
            raise NotImplementedError("A PD prefill server does not run mixed steps.")
        return self._write_extend_draft_kv(prompt, target_hidden)

    def mixed_prompt_result(
        self,
        prompt: ScheduleBatch,
        result: GenerationBatchResult,
        kv_fence_tail: torch.Tensor,
    ) -> GenerationBatchResult:
        return self._extend_result(prompt, result, kv_fence_tail)

    def _skip_rows_below_decoder_replay(
        self, batch: ScheduleBatch, positions: torch.Tensor, ring_rows: torch.Tensor
    ) -> torch.Tensor:
        """Under decoder SWA bounded replay the aux rows below each request's
        replay start hold no values (the target decoder skipped them); they are
        never in a draft window, and are not written to the ring either."""
        backend = self.target_worker.model_runner.attn_backend
        # A fused mixed step's prompt part replays as its extend would.
        if (
            not getattr(backend, "decoder_replay", False)
            or batch.forward_mode != ForwardMode.EXTEND
        ):
            return ring_rows
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            scheduled_decoder_plan,
        )

        # The plan of the unfolded batch: its rows are the ones the draft sees.
        plan = scheduled_decoder_plan(batch, folds=False)
        starts = torch.repeat_interleave(
            torch.tensor(plan.replay_starts, dtype=torch.int64),
            torch.tensor(batch.extend_lens, dtype=torch.int64),
        )
        return torch.where(positions >= starts, ring_rows, -1)

    def verify_top_k_width(self) -> int | None:
        # Negotiates on every TP rank here, at init, never inside a compiled verify.
        width = distributed_top_k_candidate_width_for_model(
            self._target_worker.model_runner.model
        )
        if width is not None and width != TARGET_CANDIDATES:
            raise ValueError(
                f"distributed top-k width {width} != TARGET_CANDIDATES {TARGET_CANDIDATES}"
            )
        return width

    def extend_draft_from_hidden(
        self,
        batch: ScheduleBatch,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> None:
        del bonus_tokens
        if (
            self._block_draft_worker.full_to_swa is not None
            or self._block_draft_worker.swa_ring_pool is not None
        ):
            raise NotImplementedError(
                "PD decode-side draft prefill does not support a hybrid-SWA draft."
            )
        positions = pad_rows(
            prompt_positions(
                prefix_lens=batch.prefix_lens, extend_lens=batch.extend_lens
            ),
            target_hidden.shape[0],
        ).to(self.device, non_blocking=True)
        with self._block_draft_worker.draft_context():
            self._block_draft_worker.write_draft_kv(
                target_hidden=target_hidden, positions=positions, cache_loc=cache_loc
            )

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

    def state_to_spec_info(self, state: SpecState) -> HostPaddedBlockDraftInput:
        return HostPaddedBlockDraftInput(
            bonus_tokens=state.bonus_tokens.to(torch.int64),
            new_seq_lens=state.seq_lens.to(torch.int64),
        )

    def get_spec_state(self, batch: ScheduleBatch) -> SpecState:
        draft_input = batch.spec_info
        if not isinstance(draft_input, HostPaddedBlockDraftInput):
            raise TypeError(
                f"Expected HostPaddedBlockDraftInput, got {type(draft_input)}."
            )
        return SpecState(
            bonus_tokens=draft_input.bonus_tokens,
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
        )

    def _forward_draft_block(
        self,
        *,
        state: SpecState,
        live_mask: torch.Tensor,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> tuple[torch.Tensor, torch.Tensor, DraftLattice | None]:
        """The block's [pbs, width, H] draft hidden, greedy [pbs, gamma] drafts and lattice."""
        padded_bs = state.seq_lens.shape[0]
        width = self._draft_block_width
        inputs = build_draft_block_inputs(
            state.bonus_tokens,
            rows,
            state.seq_lens,
            live_mask,
            self._mask_token_id,
            width,
        )
        # Before the read below, which waits for the whole previous verify: the
        # slot tables of this step's draft and verify need no device value.
        kv_len_bound = self._slot_kv_len_bound(verify_input)
        for runner in (self._draft_runner, self.target_worker.model_runner):
            slot_metadata.prepare(
                runner.attn_backend,
                verify_input.req_pool_indices_cpu,
                kv_len_bound=kv_len_bound,
            )
        # Read here, after the inputs are queued: the read waits for the previous verify.
        host_rows = (
            None
            if self._step_host_rows is None
            else self._step_host_rows.draft_rows(width)
        )
        # Writes the block's draft KV into the same slots target verify uses next.
        with self._block_draft_worker.draft_context():
            logits_output = forward_draft_rows(
                draft_runner=self._draft_runner,
                spec_algorithm=self._spec_algorithm,
                inputs=inputs,
                req_pool_indices=state.req_pool_indices,
                verify_input=verify_input,
                offset=0,
                width=width,
                spec_info=self._draft_block_spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                host_rows=host_rows,
            )
        hidden = logits_output.hidden_states
        draft_tokens = logits_output.next_draft_tokens
        if hidden is None or draft_tokens is None:
            raise RuntimeError("Draft forward returned no hidden or draft tokens.")
        return (
            hidden.view(padded_bs, width, -1),
            draft_tokens,
            logits_output.next_draft_lattice,
        )

    def _bucket_sizes(self) -> tuple[int, ...]:
        compile_policy = self.target_worker.model_runner.compile_policy
        assert isinstance(compile_policy, DecoderTpuCompilePolicy)
        assert compile_policy.b_bucketer is not None
        return tuple(compile_policy.b_bucketer.buckets)

    def _warmup_draft_block_inputs(self) -> None:
        # Width and mask token come from the draft config, so the shared spec-op grid cannot warm it.
        with torch.no_grad():
            for width, padded_bs in itertools.product(
                self.verify_widths(), self._bucket_sizes()
            ):
                block_shape = (padded_bs, width)
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
                            (padded_bs * width,),
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        flat_cache_locs=torch.zeros(
                            (padded_bs * width,),
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
                draft_lattice = self._block_draft_worker.warmup_draft_lattice(
                    block_hidden=block_hidden, bonus_tokens=bonus_tokens
                )
                temperatures = torch.ones((padded_bs,), dtype=torch.float32).to(
                    self.device
                )
                top_ks = torch.ones((padded_bs,), dtype=torch.int32).to(self.device)
                for target_candidates in (False, True):
                    sampled = self._sample_draft_block(
                        block_hidden,
                        bonus_tokens,
                        draft_lattice,
                        temperatures,
                        top_ks,
                        target_candidates=target_candidates,
                    )
                    torch.cat(
                        [bonus_tokens[:, None], sampled.draft_tokens], dim=1
                    ).cpu()
                    if sampled.draft_logits is not None:
                        sampled.draft_logits.float().cpu()
                        self._warmup_sampled_accept(
                            padded_bs, sampled.draft_tokens, sampled.draft_logits
                        )
                    if sampled.draft_candidates is not None:
                        self._warmup_candidates_accept(
                            padded_bs, sampled.draft_tokens, sampled.draft_candidates
                        )

    def _warmup_extend_draft_inputs(self) -> None:
        """extend's draft positions / cache_loc uploads at every token bucket."""
        compile_policy = self.target_worker.model_runner.compile_policy
        assert isinstance(compile_policy, DecoderTpuCompilePolicy)
        assert compile_policy.t_bucketer is not None
        with torch.no_grad():
            for tokens in compile_policy.t_bucketer.buckets:
                for dtype in (torch.int64, torch.int32):
                    torch.zeros((tokens,), dtype=dtype).to(
                        self.device, non_blocking=True
                    ).cpu()

    def _warmup_engram_commit_ids(self) -> None:
        """commit's Engram verify ids (committed_verify_ids over finalize_accept
        outputs) and their host copy for the deferred history commit, at every
        decode batch bucket."""
        ngram = self.target_worker.model_runner.ngram_embedding_manager
        if ngram is None or ngram.engram_hasher is None:
            return
        with torch.no_grad():
            for padded_bs in self._bucket_sizes():
                accept = self._warmup_accept(padded_bs, self.num_verify_rows)
                committed_verify_ids(accept.bonus_tokens, accept.accept_tokens).cpu()
                accept.accept_lens.cpu()

    def _warmup_folded_drop_draft_kv(self) -> None:
        """A folded encoder-replay extend hands the draft KV writer hidden rows
        gathered by a host-built index (the V4.1 folded-row drop) instead of the
        target's own output. Warm that gather feeding the writer at every extend
        token bucket, as serving runs it, so a server's first replay batch does
        not compile it."""
        runner = self.target_worker.model_runner
        if (
            not get_exec().features.enable_encoder_swa_bounded_replay
            or not getattr(runner.attn_backend, "folds_encoder_swa_replay", False)
            or self.ships_pd_hidden
        ):
            return
        compile_policy = runner.compile_policy
        assert isinstance(compile_policy, DecoderTpuCompilePolicy)
        assert compile_policy.t_bucketer is not None
        hidden_size = (
            self._block_draft_worker._compile_policy().dflash_target_hidden_size
        )
        loc_dtype = compile_policy.live_cache_loc_dtype()
        with torch.no_grad():
            for tokens in compile_policy.t_bucketer.buckets:
                hidden = torch.zeros(
                    (tokens, hidden_size), dtype=runner.dtype, device=self.device
                )
                # As drop_tpu_dsv41_folded_rows: a host index of the hidden's rows.
                index = torch.full((tokens,), tokens - 1, dtype=torch.int64)
                gathered = hidden[index.to(self.device, non_blocking=True)]
                positions = torch.zeros((tokens,), dtype=torch.int64).to(
                    self.device, non_blocking=True
                )
                # Padding rows write slot 0, as in extend.
                cache_loc = torch.zeros((tokens,), dtype=loc_dtype).to(
                    self.device, non_blocking=True
                )
                with self._block_draft_worker.draft_context():
                    self._block_draft_worker.write_draft_kv(
                        target_hidden=gathered, positions=positions, cache_loc=cache_loc
                    )

    def _warmup_mixed_fused(self) -> None:
        """The eager ops around a fused mixed step's MIXED forward (the forward
        itself warms in the target policy's precompile), at every (token
        bucket, verify bucket) it can run: the logits index upload and the
        verify-token splice, the prompt / verify logits rows, the verify and
        prompt hidden gathers and the prompt's draft KV write at its extend
        bucket (never a fine bucket)."""
        from sglang.srt.speculative.mixed_verify import mixed_verify_enabled
        from sglang_tpu.experimental.speculative import mixed_step

        if not mixed_verify_enabled():
            return
        runner = self.target_worker.model_runner
        compile_policy = runner.compile_policy
        assert isinstance(compile_policy, DecoderTpuCompilePolicy)
        t_bucketer = compile_policy.t_bucketer
        assert t_bucketer is not None
        R = self.num_verify_rows
        cap = compile_policy._extend_compile_cap()
        vocab_size = runner.model_config.vocab_size
        hidden_size = (
            self._block_draft_worker._compile_policy().dflash_target_hidden_size
        )
        loc_dtype = compile_policy.live_cache_loc_dtype()
        device = self.device

        def upload(t: torch.Tensor) -> torch.Tensor:
            return t.to(device, non_blocking=True)

        cases = mixed_step.fused_compile_cases(
            t_bucketer, compile_policy.b_bucketer, get_parallel().attn_tp_size
        )
        with torch.no_grad():
            for num_tokens, ext_cap, pbs in cases:
                num_logit_rows = ext_cap + R * pbs
                upload(torch.zeros((num_logit_rows,), dtype=torch.int64)).cpu()
                ids = upload(torch.zeros((num_tokens,), dtype=torch.int64))
                verify_tokens = torch.zeros(
                    (pbs * R,), dtype=torch.int64, device=device
                )
                mixed_step.splice_verify_tokens(
                    ids,
                    verify_tokens,
                    upload(mixed_step.splice_index(num_tokens, 0, pbs * R)),
                ).cpu()
                logits = upload(
                    torch.zeros((num_logit_rows, vocab_size), dtype=torch.float32)
                )
                logits[:cap].clone().cpu()
                logits[ext_cap:].clone().cpu()
                hidden = upload(
                    torch.zeros((num_tokens, hidden_size), dtype=runner.dtype)
                )
                hidden[
                    upload(mixed_step.verify_hidden_index(0, pbs, R, num_tokens))
                ].cpu()
                for rows in {
                    mixed_step.inject_bucket(t_bucketer, n)
                    for n in t_bucketer.buckets
                    if ext_cap <= n <= num_tokens
                }:
                    gathered = hidden[
                        upload(
                            mixed_step.prompt_hidden_index(None, 0, rows, num_tokens)
                        )
                    ]
                    with self._block_draft_worker.draft_context():
                        self._block_draft_worker.write_draft_kv(
                            target_hidden=gathered,
                            positions=upload(torch.zeros((rows,), dtype=torch.int64)),
                            cache_loc=upload(torch.zeros((rows,), dtype=loc_dtype)),
                        )

    # FIXME(junyanxu): metrics_reporter calls this on every DSPARK decode log;
    # report the block-accept estimate once the confidence budget is ported.
    def block_accept_estimate_log_suffix(self) -> str | None:
        return None

    # FIXME(junyanxu): internal_state calls this on any DSPARK worker for server info;
    # return the info records once the confidence budget is ported.
    def dump_info_records(self) -> dict | None:
        return None


def _extend_ring_rows(*, ring_pool, batch: ScheduleBatch, positions: torch.Tensor):
    """[T] int64 draft ring rows of an extend's tokens. Restarts the ring of each
    newly placed request at its cached prefix: the replayed rows' hidden states
    never reach the draft, so the ring holds nothing below it."""
    if batch.encoder_swa_reset is not None:
        placed = [i for i, reset in enumerate(batch.encoder_swa_reset) if reset]
        if placed:
            ring_pool.set_swa_valid_from(
                batch.req_pool_indices_cpu[placed],
                torch.tensor([batch.prefix_lens[i] for i in placed], dtype=torch.int64),
            )
    lens = torch.tensor(batch.extend_lens, dtype=torch.int64)
    slots = torch.repeat_interleave(batch.req_pool_indices_cpu.to(torch.int64), lens)
    ends = torch.repeat_interleave(
        torch.tensor(batch.prefix_lens, dtype=torch.int64) + lens, lens
    )
    return ring_pool.swa_ring_rows(slots=slots, positions=positions, ends=ends)


def _fused_mixed() -> bool:
    from sglang_tpu.experimental.speculative.mixed_step import fused_mixed_forward

    return fused_mixed_forward()

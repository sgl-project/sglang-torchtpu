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
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
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
        self._warmup_draft_block_inputs()
        if self._sampled_drafts:
            self._warmup_draft_sampling()
        self._warmup_extend_draft_inputs()
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
        full_to_swa = self._block_draft_worker.full_to_swa
        if full_to_swa is not None:
            # A hybrid-SWA draft writes the SWA slots of the same window, mapped on host.
            swa_candidates = full_to_swa[verify_input.verify_cache_candidates_cpu]
            rows = select_verify_rows(
                padded_h2d(
                    swa_candidates.to(torch.int64),
                    rows.cache_locs.shape[0],
                    self.device,
                ),
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
        # Verify rows past the new seq_lens become dead target KV; freeing them is the allocator's job.
        next_state = SpecState(
            bonus_tokens=accept.bonus_tokens,
            seq_lens=accept.next_seq_lens,
            req_pool_indices=state.req_pool_indices,
        )
        return CommitResult(state=next_state, kv_fence_tail=kv_fence_tail)

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
        # finalize keeps HOST_PADDED hidden at the token bucket; pad the host rows to
        # match and upload once. Padded rows write slot 0, like the target forward.
        padded_rows = target_hidden.shape[0]
        if not batch.reqs:
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
            full_to_swa = self._block_draft_worker.full_to_swa
            if full_to_swa is not None and not self.ships_pd_hidden:
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
            kv_fence_tail = self.write_pd_hidden(batch, target_hidden)
        else:
            with self._block_draft_worker.draft_context():
                kv_fence_tail = self._block_draft_worker.write_draft_kv(
                    target_hidden=target_hidden,
                    positions=positions,
                    cache_loc=cache_loc,
                )
        logits_output.hidden_states = None
        if not batch.reqs:
            return self._idle_extend_result(result)
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
        if self._block_draft_worker.full_to_swa is not None:
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

    # FIXME(junyanxu): metrics_reporter calls this on every DSPARK decode log;
    # report the block-accept estimate once the confidence budget is ported.
    def block_accept_estimate_log_suffix(self) -> str | None:
        return None

    # FIXME(junyanxu): internal_state calls this on any DSPARK worker for server info;
    # return the info records once the confidence budget is ported.
    def dump_info_records(self) -> dict | None:
        return None

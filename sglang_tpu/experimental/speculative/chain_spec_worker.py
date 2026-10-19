"""Autoregressive chain drafting (MTP) on the TpuSpecWorker interface.

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
relay never carry it. Not ported yet: PD disaggregation and a draft sliding window.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from sglang.srt.compilation.compiled_op import compiled_op, warmup_compiled_op
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
)
from sglang.srt.runtime_context import get_disagg, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.chain_draft_input import (
    ChainVerifyInput,
    chain_draft_forward_input,
)
from sglang_tpu.experimental.speculative.chain_draft_worker import (
    TpuChainDraftWorker,
)
from sglang_tpu.experimental.speculative.chain_info import HostPaddedChainDraftInput
from sglang_tpu.experimental.speculative.draft_forward import forward_draft_rows
from sglang_tpu.experimental.speculative.mtp.mtp_draft_worker import (
    TpuMtpDraftWorker,
)
from sglang_tpu.experimental.speculative.spec_ops import (
    build_chain_block_inputs,
    pick,
    select_verify_rows,
)
from sglang_tpu.experimental.speculative.spec_types import (
    AcceptResult,
    BucketState,
    DraftResult,
    SpecState,
    TpuSpecVerifyInput,
    VerifyRows,
)
from sglang_tpu.experimental.speculative.spec_utils import pad_rows

_DRAFT_WORKER_CLS: dict[SpeculativeAlgorithm, type[TpuChainDraftWorker]] = {
    SpeculativeAlgorithm.EAGLE: TpuMtpDraftWorker,
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
        if get_disagg().disaggregation_mode != "null":
            raise NotImplementedError(
                "TpuChainSpecWorker does not support PD disaggregation."
            )
        # Idle DP ranks would need to mirror the draft's collectives (_forward_idle_step).
        if is_dp_attention_enabled():
            raise NotImplementedError(
                "TpuChainSpecWorker does not support DP attention."
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

    def init_cuda_graphs(self) -> None:
        super().init_cuda_graphs()
        self._warmup_draft_ops()
        self._warmup_extend_ops()
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
        return self._draft_chain(
            state=bucket.state,
            rows=rows,
            req_pool_indices_cpu=verify_input.req_pool_indices_cpu,
            seq_lens_cpu=seq_lens_cpu,
            temperatures=bucket.temperatures,
            top_ks=bucket.top_ks,
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
        return DraftResult(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            num_verify_drafts=self._all_drafts[padded_bs],
        )

    def commit(
        self,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> SpecState:
        del batch
        target_hidden = logits_output.hidden_states
        assert target_hidden is not None
        padded_bs = state.seq_lens.shape[0]
        width = self.num_verify_rows
        # Rows past the bonus carry token 0; their draft KV is past the new seq_lens.
        draft_out = forward_draft_rows(
            draft_runner=self._draft_runner,
            spec_algorithm=self._spec_algorithm,
            inputs=build_chain_block_inputs(
                accept.window_tokens, rows, state.seq_lens, accept.live_mask, 0
            ),
            req_pool_indices=state.req_pool_indices,
            verify_input=verify_input,
            offset=0,
            width=width,
            spec_info=chain_draft_forward_input(
                hidden_states=target_hidden,
                width=width,
                verify_kv_lens=rows.verify_kv_lens,
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
        return SpecState(
            bonus_tokens=accept.bonus_tokens,
            seq_lens=accept.next_seq_lens,
            req_pool_indices=state.req_pool_indices,
        )

    def extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        result = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        self._tp_sync.sync(SpecTpSyncSite.SPEC_TARGET, result.next_token_ids)
        bs = batch.seq_lens.shape[0]
        # Sampling ran at the batch bucket; the draft reads it there, the scheduler takes live rows.
        bucket_tokens = result.next_token_ids
        result.next_token_ids = bucket_tokens[:bs]
        logits_output = result.logits_output
        target_hidden = logits_output.hidden_states
        if target_hidden is None:
            raise RuntimeError("Chain-draft extend requires target hidden capture.")
        assert logits_output.hidden_states_token_indices is None
        self._extend_draft(
            batch, target_hidden=target_hidden, bucket_tokens=bucket_tokens
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

    def _extend_draft(
        self,
        batch: ScheduleBatch,
        *,
        target_hidden: torch.Tensor,
        bucket_tokens: torch.Tensor,
    ) -> None:
        """Draft forward on the prompt shifted by one token; store each request's last hidden."""
        input_ids = batch.input_ids
        assert input_ids is not None and input_ids.device.type == "cpu", (
            "HOST_PADDED extend stages prompt tokens on host"
        )
        # finalize keeps HOST_PADDED target hidden at the token bucket.
        padded_tokens = target_hidden.shape[0]
        chunked_row, next_prompt_token = _chunked_tail(batch)
        shifted, row_reqs, is_sampled = shift_prompt_tokens(
            input_ids,
            extend_lens=batch.extend_lens,
            padded_tokens=padded_tokens,
            chunked_row=chunked_row,
            next_prompt_token=next_prompt_token,
        )
        draft_ids = fill_sampled_tails(
            shifted.to(self.device, non_blocking=True),
            row_reqs.to(self.device, non_blocking=True),
            is_sampled.to(self.device, non_blocking=True),
            bucket_tokens,
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
        with torch.no_grad():
            draft_out = self._draft_runner.forward(forward_batch).logits_output
        _, tail_hidden = _draft_logits_and_hidden(draft_out)
        # LAST capture keeps one row per request at the extend cap; padded rows hit pool row 0.
        cap = tail_hidden.shape[0]
        pool_rows = pad_rows(batch.req_pool_indices_cpu, cap).to(
            self.device, non_blocking=True
        )
        self._chain_draft_worker.carry.index_copy_(0, pool_rows, tail_hidden)

    def make_verify_input(
        self, *, batch: ScheduleBatch, verify_tokens: torch.Tensor, rows: VerifyRows
    ) -> ChainVerifyInput:
        return ChainVerifyInput(
            draft_token=verify_tokens,
            positions=rows.flat_positions,
            draft_token_num=self.num_verify_rows,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            live_seq_lens_cpu=batch.seq_lens_cpu,
            verify_kv_lens=rows.verify_kv_lens,
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
        for temperatures, top_ks in (
            (None, None),
            (upload(pbs, dtype=torch.float32), upload(pbs, dtype=torch.int32)),
        ):
            result = self._draft_chain(
                state=state,
                rows=rows,
                req_pool_indices_cpu=torch.zeros((pbs,), dtype=torch.int64),
                seq_lens_cpu=torch.full((pbs,), width, dtype=torch.int64),
                temperatures=temperatures,
                top_ks=top_ks,
            )
            result.draft_tokens.cpu()

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


def shift_prompt_tokens(
    input_ids: torch.Tensor,
    *,
    extend_lens: Sequence[int],
    padded_tokens: int,
    chunked_row: int | None,
    next_prompt_token: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Host [padded_tokens] shifted ids, owning request per row, and rows awaiting the sampled token.

    Row t of a request reads token t + 1; its last row reads the token the
    target just sampled, or the next prompt token when the request is chunked.
    """
    num_tokens = input_ids.shape[0]
    shifted = torch.zeros((padded_tokens,), dtype=torch.int64)
    shifted[: num_tokens - 1] = input_ids[1:]
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
    grid = torch.zeros((pbs, width), dtype=torch.int64, device=device)
    flat = torch.zeros((pbs * width,), dtype=torch.int64, device=device)
    return VerifyRows(
        positions=grid,
        cache_locs=grid,
        flat_positions=flat,
        flat_cache_locs=flat,
        verify_kv_lens=torch.zeros((pbs,), dtype=torch.int32, device=device),
    )

"""DeepSeek V4.1 Flash DSpark on the TpuSpecWorker interface (stock target path).

The drafter is bundled in the target checkpoint and lives inside the target
model (``model.dspark``, sglang_tpu/dsv41/dspark.py), so there is no draft
ModelRunner, draft KV pool or draft attention backend:

draft:  one compiled block forward of [r, noise x (K - 1)] per request over the
        drafter's context rings, then the greedy Markov chain (sampled rows draw
        from the Markov-corrected logits instead, eagerly).
verify: the shared TpuSpecWorker.verify over [r, d_1..d_K]; before it, the host
        Engram rows of every verify row (drafted tokens included) are hashed
        from one D2H copy of [committed tail, verify tokens, seq_lens].
commit: inject the committed rows' captured target features (hc-mean of the
        attention input of layers 37-39) into the context rings; advance the
        per-slot committed-token tail Engram reads next step.
extend: target forward with FULL capture; inject every extended row (the last
        RING per request), seed the slot floor and the committed-token tail.

DP attention (V4.1 is DP-local attention + fused EP MoE over all ranks): the
draft forward's MoE and the verify forward are world collectives, so a rank
without requests replays both at the shared batch bucket (_forward_idle_step).
Context injection, Markov sampling and acceptance are rank-local.
"""

from __future__ import annotations

import logging
import weakref

import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    PPProxyTensors,
)
from sglang.srt.models.dspark import run_markov_block
from sglang.srt.runtime_context import get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.dsv41.dspark import RING, DSV41DSparkDrafter
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.block_info import HostPaddedBlockDraftInput
from sglang_tpu.experimental.speculative.spec_ops import (
    build_verify_tokens,
    sample_tokens,
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
    overridden_fields,
    pad_rows,
    prompt_positions,
)

logger = logging.getLogger(__name__)


def extend_rows(
    *,
    prefix_lens: list[int],
    extend_lens: list[int],
    slots: list[int],
    floors: list[int],
    tails: list[list[int]],
    rows: int,
) -> dict[str, torch.Tensor]:
    """Host per-token injection inputs for an extend, zero-padded to rows.

    Only the last RING positions of each request's chunk are written (earlier
    ones would alias ring entries); padded rows write nothing. Floors and tails
    repeat per token so the device op has no batch-size-dependent shape.
    """
    positions = prompt_positions(prefix_lens=prefix_lens, extend_lens=extend_lens)
    lens = torch.tensor(extend_lens, dtype=torch.int64)
    ends = torch.tensor(prefix_lens, dtype=torch.int64) + lens
    per_token = lambda values: torch.repeat_interleave(  # noqa: E731
        torch.as_tensor(values, dtype=torch.int64), lens, dim=0
    )
    write = positions >= per_token(ends) - RING
    return {
        "positions": pad_rows(positions, rows),
        "slots": pad_rows(per_token(slots), rows),
        "write": pad_rows(write, rows),
        "floors": pad_rows(per_token(floors), rows),
        "tails": pad_rows(per_token(tails), rows),
    }


def commit_rows(slots: torch.Tensor, accept_lens: torch.Tensor, rows: int):
    """Request slot and write flag of each verify row; only committed rows write."""
    index = torch.arange(rows, device=slots.device)
    write = (index[None, :] < accept_lens[:, None]).reshape(-1)
    return slots[:, None].expand(-1, rows).reshape(-1), write


def advance_tails(tails, slots, accept_lens, verify_tokens) -> None:
    """Move each slot's committed-token tail past this step's accepted tokens.

    tails[slot] holds positions n - t .. n - 1, verify_tokens [r, d_1..d_K] sit
    at n .. n + K, and accept_len tokens (r, d_1..d_{a}) commit. The new tail
    ends right before the new root n + accept_len.
    """
    rows = verify_tokens.numel() // slots.shape[0]
    history = torch.cat([tails[slots], verify_tokens.view(-1, rows)], dim=1)
    shift = accept_lens.long()[:, None] + torch.arange(
        tails.shape[1], device=slots.device
    )
    tails.index_copy_(0, slots, torch.gather(history, 1, shift))


def committed_tail(req, end: int, count: int) -> list[int]:
    """Tokens at positions end - count .. end - 1 (0 below position 0)."""
    prompt, output = req.origin_input_ids, req.output_ids
    start = max(0, end - count)
    window = list(prompt[start : min(end, len(prompt))])
    if end > len(prompt):
        window.extend(output[max(0, start - len(prompt)) : end - len(prompt)])
    if len(window) != end - start:
        # Not expected for an extend; Engram of this request's first verify
        # rows then sees zeros instead of the missing tokens (logged, not fatal).
        logger.warning("DSpark: request tokens before position %d not on host", end)
        window = window[-(end - start) :] if window else []
    return [0] * (count - len(window)) + [int(t) for t in window]


def dp_verify_counts(global_num_tokens, padded_bs: int):
    """Per-rank MLP-sync request counts for a TARGET_VERIFY forward under DP.

    Every DP rank verifies at the shared bucket padded_bs (the max over ranks,
    TpuSpecWorker._decode_bucket), so its forward has padded_bs * (K + 1) rows.
    The scheduler's counts are each rank's own requests; ForwardBatch scales
    them by K + 1 and pads this rank's rows to its own entry, which is short of
    the forward (negative pad) whenever ranks hold different request counts,
    e.g. 0 / 1 / 2 requests. The verify reports padded_bs for every rank instead.
    """
    if not global_num_tokens:
        return None
    if max(global_num_tokens) > padded_bs:
        raise ValueError(f"DP counts {global_num_tokens} exceed the bucket {padded_bs}")
    return [int(padded_bs)] * len(global_num_tokens)


class ExtendContext:
    """Host side of an extend's context injection, from the scheduled batch only.

    Reads what every ScheduleBatch carries for an extend: reqs, prefix_lens,
    extend_lens and req_pool_indices_cpu (the request slots, aligned with reqs;
    Req itself keeps its slot in req.kv, which a retraction replaces). Tracks
    each request's context floor in a weak map, so finished requests drop out
    and no attribute is set on Req.
    """

    def __init__(self) -> None:
        # Req -> (slot, floor, end of the last injected chunk).
        self._seen: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def floor(self, req, slot: int, start: int, length: int) -> int:
        """First position whose context the drafter injected for this request.

        A continuing chunk (same slot, starting where the last one ended) keeps
        its floor; a new request, a new slot or a restart from a radix prefix
        (retraction) starts at this chunk.
        """
        seen = self._seen.get(req)
        floor = start
        if seen is not None and seen[0] == slot and seen[2] == start:
            floor = seen[1]
        self._seen[req] = (slot, floor, start + length)
        return floor

    def rows(self, batch: ScheduleBatch, *, tail: int, rows: int):
        reqs = list(batch.reqs)
        prefix_lens, extend_lens = list(batch.prefix_lens), list(batch.extend_lens)
        slots = batch.req_pool_indices_cpu[: len(reqs)].tolist()
        if not len(reqs) == len(prefix_lens) == len(extend_lens) == len(slots):
            raise ValueError("Extend batch fields are not aligned with its requests")
        floors = [
            self.floor(req, slot, start, length)
            for req, slot, start, length in zip(reqs, slots, prefix_lens, extend_lens)
        ]
        tails = [
            committed_tail(req, start + length, tail)
            for req, start, length in zip(reqs, prefix_lens, extend_lens)
        ]
        return extend_rows(
            prefix_lens=prefix_lens,
            extend_lens=extend_lens,
            slots=slots,
            floors=floors,
            tails=tails,
            rows=rows,
        )


class TpuDsv41DsparkWorker(TpuSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        del server_args, gpu_id, nccl_port
        if read_ragged_verify_mode() is not RaggedVerifyMode.STATIC:
            raise NotImplementedError(
                "V4.1 DSpark verifies a fixed K (SGLANG_RAGGED_VERIFY_MODE=static)."
            )
        runner = target_worker.model_runner
        drafter = getattr(runner.model, "dspark", None)
        if not isinstance(drafter, DSV41DSparkDrafter):
            raise TypeError("V4.1 DSpark needs the target's bundled drafter")
        # The drafter is part of the target: no draft worker or draft runner.
        super().__init__(target_worker, None, drafter.gamma)
        if get_spec().speculative_num_draft_tokens != self.num_verify_rows:
            raise ValueError(
                f"speculative_num_draft_tokens={get_spec().speculative_num_draft_tokens} "
                f"must equal gamma + 1 = {self.num_verify_rows}."
            )
        self._drafter = drafter
        # The host Engram coordinator is bound when the target allocates its pools,
        # after this worker exists (ModelRunner.init_ngram_embedding_manager).
        self._engram = None
        self._ngram = 1
        self._tails: torch.Tensor | None = None
        self._verify_tokens: torch.Tensor | None = None
        self._extend_context = ExtendContext()
        # Sampled rows draft greedily (target-only acceptance) unless
        # --speculative-use-rejection-sampling asks for drafts sampled from q.
        self._greedy_draft = not get_spec().speculative_use_rejection_sampling
        self._build_ops()

    # ---- state ----------------------------------------------------------------

    def alloc_memory_pool(self, *args, **kwargs):
        super().alloc_memory_pool(*args, **kwargs)
        self._allocate_state()

    def _allocate_state(self) -> None:
        if self._tails is not None:
            return
        runner = self.target_worker.model_runner
        self._engram = getattr(runner, "host_engram", None)
        self._ngram = self._engram.ngram if self._engram is not None else 1
        if self._engram is None and getattr(runner.model.model, "host_engram", None):
            raise RuntimeError("V4.1 DSpark: host Engram is bound to the model only")
        pool = runner.req_to_token_pool
        slots = int(pool.req_to_token.shape[0])
        self._drafter.allocate_state(slots, self.device)
        # Committed tokens at positions n - (ngram - 1) .. n - 1 per request slot.
        self._tails = torch.zeros(
            (slots, max(1, self._ngram - 1)), dtype=torch.int64, device=self.device
        )
        logger.info(
            "V4.1 DSpark: K=%d, %d request slots, draft context %.1f MiB, host Engram %s",
            self.num_drafts,
            slots,
            self._drafter.state_bytes() / 2**20,
            "on" if self._engram is not None else "off",
        )

    def _build_ops(self) -> None:
        drafter, worker = self._drafter, self

        @compiled_op(model=drafter)
        def draft_op(bonus_tokens, seq_lens, slots, live):
            return drafter.draft(bonus_tokens, seq_lens, slots, live)

        @compiled_op(model=drafter)
        def commit_op(aux_hidden, positions, slots, accept_lens, verify_tokens, tails):
            row_slots, write = commit_rows(slots, accept_lens, worker.num_verify_rows)
            fence = drafter.inject(aux_hidden, positions, row_slots, write)
            advance_tails(tails, slots, accept_lens, verify_tokens)
            return fence

        @compiled_op(model=drafter)
        def extend_op(aux_hidden, positions, slots, write, floors, tails_rows, tails):
            drafter.set_floor(slots, floors)
            tails.index_copy_(0, slots, tails_rows)
            return drafter.inject(aux_hidden, positions, slots, write)

        @compiled_op()
        def engram_rows(tails, slots, verify_tokens, seq_lens):
            rows = worker.num_verify_rows
            return torch.cat(
                [tails[slots], verify_tokens.view(-1, rows), seq_lens.long()[:, None]],
                dim=1,
            )

        self._draft_op = draft_op
        self._commit_op = commit_op
        self._extend_op = extend_op
        self._engram_rows = engram_rows

    # ---- draft / verify / commit ------------------------------------------------

    def draft(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> DraftResult:
        del batch, rows, verify_input
        state = bucket.state
        pbs = state.seq_lens.shape[0]
        greedy, hidden = self._draft_op(
            state.bonus_tokens, state.seq_lens, state.req_pool_indices, bucket.live_mask
        )
        if bucket.temperatures is None or bucket.top_ks is None or self._greedy_draft:
            # No draft logits: verify takes the target-only rule for sampled rows.
            draft_tokens, draft_logits = greedy, None
        else:
            draft_tokens, draft_logits = self._sample_drafts(
                hidden, state.bonus_tokens, bucket.temperatures, bucket.top_ks
            )
        return DraftResult(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            num_verify_drafts=self._all_drafts[pbs],
            verify_tokens=build_verify_tokens(state.bonus_tokens, draft_tokens),
        )

    def _sample_drafts(self, hidden, bonus_tokens, temperatures, top_ks):
        """Sampled drafts from the Markov-corrected logits q, as core DSpark draws them."""
        drafter = self._drafter
        base = drafter.base_logits(hidden).view(
            bonus_tokens.shape[0], self.num_drafts, -1
        )

        def sampler(step_logits, step_idx):
            del step_idx
            tokens = sample_tokens(
                logits=step_logits, temperatures=temperatures, top_ks=top_ks
            )
            return self._tp_sync.sync(SpecTpSyncSite.SPEC_DRAFT_SAMPLE, tokens)

        return run_markov_block(
            drafter,
            base,
            first_prev_tokens=bonus_tokens,
            hidden_states=None,
            sampler=sampler,
            collect_corrected=True,
        )

    def verify(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        draft_result: DraftResult,
        verify_input: TpuSpecVerifyInput,
    ) -> tuple[LogitsProcessorOutput, AcceptResult]:
        # commit reads the committed tokens from the verify window.
        self._verify_tokens = draft_result.verify_tokens
        self._stage_verify_engram(len(batch.reqs), bucket.state, draft_result)
        return super().verify(batch, bucket, rows, draft_result, verify_input)

    def _stage_verify_engram(
        self, bs: int, state: SpecState, draft_result: DraftResult
    ) -> None:
        """Hash the verify rows on host: one D2H of [tail, r, d_1..d_K, n] per row.

        The rows op is issued right behind the draft forward. Its D2H waits for
        the drafter, so the copy and hash run when the verify forward first stages
        Engram rows: the verify batch, attention metadata and the segment before
        the first Engram layer are issued while the drafter still runs.
        """
        if self._engram is None:
            return
        from sglang_tpu.engram import verify_inputs

        rows_dev = self._engram_rows(
            self._tails,
            state.req_pool_indices,
            draft_result.verify_tokens,
            state.seq_lens,
        )
        engram, tail = self._engram, self._tails.shape[1]
        rows, ngram = self.num_verify_rows, self._ngram

        def resolve():
            info = rows_dev.cpu()[:bs]
            engram.prepare_verify(
                verify_inputs(
                    info[:, :tail].numpy(),
                    info[:, tail : tail + rows].numpy(),
                    info[:, -1].numpy(),
                    ngram,
                )
            )

        engram.prepare_verify_deferred(resolve)

    def commit(
        self,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> CommitResult:
        del batch, verify_input
        aux_hidden = logits_output.hidden_states
        assert aux_hidden is not None and aux_hidden.shape[0] == rows.cache_locs.numel()
        verify_tokens, self._verify_tokens = self._verify_tokens, None
        fence = self._commit_op(
            aux_hidden,
            rows.flat_positions,
            state.req_pool_indices,
            accept.accept_lens,
            verify_tokens,
            self._tails,
        )
        return CommitResult(
            state=SpecState(
                bonus_tokens=accept.bonus_tokens,
                seq_lens=accept.next_seq_lens,
                req_pool_indices=state.req_pool_indices,
            ),
            kv_fence_tail=fence,
        )

    def _build_verify_batch(
        self,
        *,
        batch: ScheduleBatch,
        bucket_state: SpecState,
        verify_tokens: torch.Tensor,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
        spec_info: SpecInput,
    ):
        # Under DP attention every rank runs the verify at the shared bucket, so
        # every rank's MLP-sync count must be that bucket (see dp_verify_counts).
        counts = dp_verify_counts(
            batch.global_num_tokens if self._dp_attention else None,
            bucket_state.seq_lens.shape[0],
        )
        overrides = (
            {}
            if counts is None
            else {"global_num_tokens": counts, "global_num_tokens_for_logprob": counts}
        )
        with overridden_fields(batch, **overrides):
            return super()._build_verify_batch(
                batch=batch,
                bucket_state=bucket_state,
                verify_tokens=verify_tokens,
                rows=rows,
                verify_input=verify_input,
                spec_info=spec_info,
            )

    def _forward_idle_step(
        self,
        *,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> None:
        # The busy ranks' collectives in order: the draft MoE, then the verify.
        drafts, _ = self._draft_op(
            state.bonus_tokens,
            state.seq_lens,
            state.req_pool_indices,
            state.req_pool_indices != 0,
        )
        if self._engram is not None:
            self._engram.prepare_verify(None)
        self._forward_idle_verify(
            batch=batch,
            state=state,
            rows=rows,
            verify_input=verify_input,
            verify_tokens=build_verify_tokens(state.bonus_tokens, drafts),
        )

    # ---- extend ---------------------------------------------------------------

    def extend(
        self, batch: ScheduleBatch, *, pp_proxy_tensors: PPProxyTensors | None = None
    ) -> GenerationBatchResult:
        # TpuSpecWorker.forward_batch_generation passes pp_proxy_tensors to every
        # spec worker's extend; the in-target V4.1 drafter has no PP support.
        if pp_proxy_tensors is not None:
            raise NotImplementedError(
                "V4.1 DSpark does not support pipeline parallelism."
            )
        result = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        self._tp_sync.sync(SpecTpSyncSite.SPEC_TARGET, result.next_token_ids)
        result.next_token_ids = result.next_token_ids[: batch.seq_lens.shape[0]]
        logits_output = result.logits_output
        aux_hidden = logits_output.hidden_states
        if aux_hidden is None:
            raise RuntimeError("V4.1 DSpark extend needs the captured target features")
        assert logits_output.hidden_states_token_indices is None
        kv_fence_tail = None
        if batch.reqs:
            # Must inject before returning: the scheduler may update radix afterward.
            kv_fence_tail = self._inject_extend(batch, aux_hidden)
        logits_output.hidden_states = None
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

    def _inject_extend(self, batch: ScheduleBatch, aux_hidden: torch.Tensor):
        host = self._extend_context.rows(
            batch, tail=self._tails.shape[1], rows=aux_hidden.shape[0]
        )
        device = {k: v.to(self.device, non_blocking=True) for k, v in host.items()}
        return self._extend_op(
            aux_hidden,
            device["positions"],
            device["slots"],
            device["write"],
            device["floors"],
            device["tails"],
            self._tails,
        )

    # ---- formats ----------------------------------------------------------------

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

    # ---- warmup ---------------------------------------------------------------

    def init_cuda_graphs(self) -> None:
        self._allocate_state()
        super().init_cuda_graphs()
        self._warmup_spec_ops()
        self._warmup_dspark()
        if self._greedy_draft:
            self._warmup_target_only_accept()
        self._snapshot_warm_baseline()

    def _warmup_spec_ops(self) -> None:
        # The scheduler skips its spec-op grid without a draft runner.
        from sglang_tpu.spec_op_precompile import precompile_spec_v2_ops

        config = self.target_worker.model_runner.model_config
        precompile_spec_v2_ops(
            device=torch.device(self.device),
            physical_batch_buckets=self._b_bucketer().buckets,
            num_verify_tokens=self.num_verify_rows,
            vocab_size=config.vocab_size,
            draft_vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            hidden_dtype=torch.bfloat16,
        )

    def _warmup_dspark(self) -> None:
        """Every draft/commit/extend program and its host copies, per bucket."""
        rows = self.num_verify_rows
        aux_width = self._drafter.hidden_size * len(self._drafter.target_layer_ids)
        device = self.device

        def zeros(*shape, dtype=torch.int64):
            return torch.zeros(shape, dtype=dtype).to(device, non_blocking=True)

        with torch.no_grad():
            for pbs in self._b_bucketer().buckets:
                seq_lens, slots, bonus = zeros(pbs), zeros(pbs), zeros(pbs)
                live = slots != 0
                greedy, hidden = self._draft_op(bonus, seq_lens, slots, live)
                verify_tokens = build_verify_tokens(bonus, greedy)
                self._engram_rows(self._tails, slots, verify_tokens, seq_lens).cpu()
                if not self._greedy_draft:
                    drafts, logits = self._sample_drafts(
                        hidden,
                        bonus,
                        torch.ones((pbs,), dtype=torch.float32).to(device),
                        torch.ones((pbs,), dtype=torch.int32).to(device),
                    )
                    logits.float().cpu()
                    build_verify_tokens(bonus, drafts).cpu()
                    self._warmup_sampled_accept(pbs, drafts, logits)
                aux = zeros(pbs * rows, aux_width, dtype=torch.bfloat16)
                self._commit_op(
                    aux,
                    zeros(pbs * rows),
                    slots,
                    zeros(pbs, dtype=torch.int32),
                    verify_tokens,
                    self._tails,
                ).cpu()
            compile_policy = self.target_worker.model_runner.compile_policy
            assert isinstance(compile_policy, DecoderTpuCompilePolicy)
            assert compile_policy.t_bucketer is not None
            tail = self._tails.shape[1]
            for tokens in compile_policy.t_bucketer.buckets:
                host = extend_rows(
                    prefix_lens=[0],
                    extend_lens=[1],
                    slots=[0],
                    floors=[0],
                    tails=[[0] * tail],
                    rows=tokens,
                )
                host["write"].zero_()
                inputs = {k: v.to(device, non_blocking=True) for k, v in host.items()}
                self._extend_op(
                    zeros(tokens, aux_width, dtype=torch.bfloat16),
                    inputs["positions"],
                    inputs["slots"],
                    inputs["write"],
                    inputs["floors"],
                    inputs["tails"],
                    self._tails,
                ).cpu()

    # Core metrics/info hooks call these on every DSPARK worker.
    def block_accept_estimate_log_suffix(self) -> str | None:
        return None

    def dump_info_records(self) -> dict | None:
        return None

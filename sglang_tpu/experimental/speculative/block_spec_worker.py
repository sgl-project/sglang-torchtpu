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

import collections
import functools
import itertools
import time
import weakref
from collections.abc import Callable
from typing import NamedTuple

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
from sglang.srt.runtime_context import get_disagg, get_parallel, get_spec
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.srt.speculative.ragged_verify import (
    RaggedVerifyMode,
    read_ragged_verify_mode,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_tp_sync import SpecTpSyncSite
from sglang_tpu._logging import get_logger
from sglang_tpu.attention_pallas_rpa_base import BaseTpuPallasRpaAttnBackend
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.recompilation_checker import (
    snapshot_recompilation_baseline,
)
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.compilation.block_compile_policy import (
    TpuBlockSpecDecodeCompilePolicyMixin,
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
    build_verify_tokens,
    live_cap_lens,
    pack_verify,
    select_verify_rows,
    truncate_drafts,
    unpack_verify_outputs,
)
from sglang_tpu.experimental.speculative.spec_types import (
    AcceptResult,
    BucketState,
    CommitResult,
    DraftResult,
    PackedVerify,
    SpecState,
    TpuSpecVerifyInput,
    VerifyRows,
)
from sglang_tpu.experimental.speculative.spec_utils import (
    pad_rows,
    padded_h2d,
    prompt_positions,
)
from sglang_tpu.experimental.speculative.verify_width import (
    AdaptiveVerifyWidth,
    verify_width_candidates,
    verify_width_table,
)
from sglang_tpu.sampler import distributed_top_k_candidate_width_for_model


def _packed_rows(packed: PackedVerify, padded_bs: int) -> VerifyRows:
    """The packed verify tokens as [pbs, T / pbs] rows for the shared verify forward."""
    num_tokens = packed.input_ids.shape[0]
    return VerifyRows(
        positions=packed.positions.view(padded_bs, num_tokens // padded_bs),
        cache_locs=packed.cache_locs.view(padded_bs, num_tokens // padded_bs),
        flat_positions=packed.positions,
        flat_cache_locs=packed.cache_locs,
    )


def _unpack_logits_output(
    logits_output: LogitsProcessorOutput, strided_rows: torch.Tensor
) -> None:
    """Return a compact verify's per-token outputs to the strided [pbs * (N + 1)] window."""
    (
        logits_output.next_token_logits,
        logits_output.next_token_global_top_k_logits,
        logits_output.next_token_global_top_k_token_ids,
        logits_output.hidden_states,
    ) = unpack_verify_outputs(
        strided_rows,
        logits_output.next_token_logits,
        logits_output.next_token_global_top_k_logits,
        logits_output.next_token_global_top_k_token_ids,
        logits_output.hidden_states,
    )


class _PendingStep(NamedTuple):
    """A dispatched decode step awaiting its host-side accept counts."""

    bucket: int
    width: int
    bs: int
    extend_ct: int
    # Set when caps vary per request: the scheduler's result copy swaps its
    # cap_lens for a host tensor before on_verify_complete_cpu reads it. Weak, so
    # an unreported step does not pin the result's device outputs.
    result: weakref.ref[GenerationBatchResult] | None = None


logger = get_logger(__name__)

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
        self._ragged_verify_mode = read_ragged_verify_mode()
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
        self._verify_width_table = verify_width_table(self.num_verify_rows)
        candidates = verify_width_candidates(self.num_verify_rows)
        if self._verify_width_table is not None and candidates is not None:
            raise ValueError(
                "Set SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS or "
                "SGLANG_TPU_SPEC_VERIFY_WIDTH_CANDIDATES, not both."
            )
        self._adaptive_width = (
            None
            if candidates is None
            else AdaptiveVerifyWidth(candidates, num_drafts=self.num_drafts)
        )
        if self._verify_width_table is not None or candidates is not None:
            self._check_verify_width_table_supported()
        # The width this decode step uses (begin_decode_step) and the steps whose
        # results the scheduler has not reported yet, oldest first.
        self._step_width = self.num_verify_rows
        self._pending_steps: collections.deque[_PendingStep] = collections.deque(
            maxlen=16
        )
        self._extend_ct = 0
        self._last_completion: tuple[float, int] | None = None
        # [pbs] int32 caps the current decode step scheduled, for its cap_lens.
        self._step_caps: torch.Tensor | None = None
        # cap-accept verifies the full window; compact packs only the scheduled rows.
        self._compact = self._ragged_verify_mode is RaggedVerifyMode.COMPACT
        self._scheduled_caps = self._compact or (
            self._ragged_verify_mode is RaggedVerifyMode.CAP_ACCEPT
        )
        if self._scheduled_caps:
            self._check_scheduled_caps_supported()

    def _check_verify_width_table_supported(self) -> None:
        if self._spec_algorithm != SpeculativeAlgorithm.DSPARK:
            # DFlash2 builds its verify tokens and candidates at gamma.
            raise NotImplementedError(
                "SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS supports only DSPARK."
            )
        if self._target_has_linear_attn:
            # The linear-attention verify checkpoints assume the full width.
            raise NotImplementedError(
                "SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS does not support linear-attention targets."
            )
        if self._adaptive_width is not None and self._dp_attention:
            # DP ranks run one bucket but would each measure and pick a width.
            raise NotImplementedError(
                "SGLANG_TPU_SPEC_VERIFY_WIDTH_CANDIDATES does not support DP attention."
            )

    def _check_scheduled_caps_supported(self) -> None:
        mode = self._ragged_verify_mode.value
        if self._simulate_acc_len > 0:
            # Forced accept lengths would ignore each request's cap; under compact
            # they would commit rows the target never computed.
            raise NotImplementedError(
                f"SGLANG_RAGGED_VERIFY_MODE={mode} does not support SGLANG_SIMULATE_ACC_LEN."
            )
        drafter = self._block_draft_worker
        if (
            not isinstance(drafter, TpuDsparkDraftWorker)
            or not drafter.has_confidence_head
        ):
            raise NotImplementedError(
                f"SGLANG_RAGGED_VERIFY_MODE={mode} needs a DSpark draft with a confidence head."
            )
        if self._verify_width_table is None and self._adaptive_width is None:
            raise ValueError(
                f"SGLANG_RAGGED_VERIFY_MODE={mode} takes its per-bucket budget "
                "(mean verify width) from SGLANG_TPU_SPEC_VERIFY_WIDTH_BY_BS or "
                "SGLANG_TPU_SPEC_VERIFY_WIDTH_CANDIDATES."
            )
        if self._compact and (self._dp_attention or self._target_has_linear_attn):
            raise NotImplementedError(
                "SGLANG_RAGGED_VERIFY_MODE=compact supports neither DP attention "
                "nor linear-attention targets."
            )
        target_backend = self._target_worker.model_runner.attn_backend
        if self._compact and not (
            isinstance(target_backend, BaseTpuPallasRpaAttnBackend)
            and target_backend.supports_ragged_verify
        ):
            # Only the Pallas RPA kernels read the packed rows' ragged layout;
            # another backend (or MLA's own kernel) would split them at
            # draft_token_num per request.
            raise NotImplementedError(
                "SGLANG_RAGGED_VERIFY_MODE=compact needs a Pallas RPA target attention "
                f"backend, got {type(target_backend).__name__}."
            )

    def verify_widths(self) -> tuple[int, ...]:
        if self._scheduled_caps:
            return (self.num_verify_rows,)
        return tuple(
            sorted(
                {w for pbs in self._bucket_sizes() for w in self._budget_widths(pbs)}
            )
        )

    def verify_width_for(self, padded_bs: int) -> int:
        if self._scheduled_caps:
            return self.num_verify_rows
        return self._budget_width(padded_bs)

    def _budget_widths(self, padded_bs: int) -> tuple[int, ...]:
        """Every width a step at this bucket may use, which warmup compiles."""
        if self._adaptive_width is not None:
            return self._adaptive_width.candidates
        if self._verify_width_table is not None:
            return (self._verify_width_table.width_for(padded_bs),)
        return (self.num_verify_rows,)

    def _budget_width(self, padded_bs: int) -> int:
        """This step's mean verify rows per request (uniform width in static mode)."""
        if self._adaptive_width is not None:
            return self._step_width
        return self._budget_widths(padded_bs)[0]

    def _caps_bucket(self, padded_bs: int) -> bool:
        """Whether this step schedules per-request caps; a full-width budget
        verifies the whole window on the plain decode-tile path instead."""
        return (
            self._scheduled_caps
            and self._budget_width(padded_bs) < self.num_verify_rows
        )

    def _caps_shapes(self) -> list[tuple[int, int]]:
        """Every (bucket, budget width) that schedules caps, which warmup compiles."""
        if not self._scheduled_caps:
            return []
        return [
            (pbs, width)
            for pbs in self._bucket_sizes()
            for width in self._budget_widths(pbs)
            if width < self.num_verify_rows
        ]

    def begin_decode_step(self, padded_bs: int, bs: int) -> None:
        # A DP-idle draft schedules caps that no result reports.
        self._step_caps = None
        if self._adaptive_width is None:
            return
        width = self._adaptive_width.width_for_step(
            padded_bs,
            agree=functools.partial(self._tp_group.broadcast_object, src=0),
        )
        if width != self._step_width and self._tp_group.rank_in_group == 0:
            summary = self._adaptive_width.summary(padded_bs)
            logger.info(
                "Adaptive verify width at bucket %d: %d -> %d "
                "(predicted tokens/req %s, step ms %s)",
                padded_bs,
                self._step_width,
                width,
                {
                    w: None if t is None else round(t, 2)
                    for w, (t, _) in summary.items()
                },
                {
                    w: None if s is None else round(s * 1e3, 2)
                    for w, (_, s) in summary.items()
                },
            )
        self._step_width = width
        self._pending_steps.append(
            _PendingStep(
                bucket=padded_bs,
                width=self._step_width,
                bs=bs,
                extend_ct=self._extend_ct,
            )
        )

    def on_verify_complete_cpu(
        self, num_correct_drafts_per_req: list[int], batch_size: int = 0
    ) -> None:
        """Feed the adaptive width: the scheduler reports decode results in dispatch order."""
        if self._adaptive_width is None:
            return
        while self._pending_steps and self._pending_steps[0].bs != batch_size:
            self._pending_steps.popleft()  # a step whose result went unreported
        if not self._pending_steps or batch_size == 0:
            return
        step = self._pending_steps.popleft()
        now = time.perf_counter()
        # An interval spanning a prefill measures the prefill, not this width: the
        # overlap scheduler may dispatch one after this step but before its result.
        last = self._last_completion
        seconds = (
            now - last[0]
            if last is not None and last[1] == step.extend_ct == self._extend_ct
            else None
        )
        self._last_completion = (now, self._extend_ct)
        if step.result is None:
            num_verify_drafts = [step.width - 1] * batch_size
        else:
            result = step.result()
            cap_lens = None if result is None else result.cap_lens
            if (
                cap_lens is None
                or not cap_lens.is_cpu
                or cap_lens.numel() != batch_size
            ):
                # A stray pairing (by batch size) with a step whose result is gone:
                # keep its step time, drop its acceptance rather than guess the caps.
                num_correct_drafts_per_req, num_verify_drafts = [], []
            else:
                num_verify_drafts = [n - 1 for n in cap_lens.tolist()]
        self._adaptive_width.record(
            step.bucket,
            step.width,
            num_correct_drafts=num_correct_drafts_per_req,
            num_verify_drafts=num_verify_drafts,
            seconds=seconds,
        )

    def _pack_decode_result(
        self,
        *,
        bs: int,
        next_state: SpecState,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> GenerationBatchResult:
        result = super()._pack_decode_result(
            bs=bs, next_state=next_state, logits_output=logits_output, accept=accept
        )
        if self._step_caps is None:
            return result
        # The scheduler's async result copy brings these to the host with accept_lens.
        result.cap_lens = live_cap_lens(self._step_caps, bs)
        self._step_caps = None
        if self._adaptive_width is not None:
            self._pending_steps[-1] = self._pending_steps[-1]._replace(
                result=weakref.ref(result)
            )
        return result

    def _schedule_verify_drafts(
        self,
        *,
        block_hidden: torch.Tensor,
        bonus_tokens: torch.Tensor,
        draft_tokens: torch.Tensor,
        live_mask: torch.Tensor,
        budget_width: int,
    ) -> torch.Tensor:
        """[pbs] int32 per-request verify caps from the draft confidence."""
        drafter = self._block_draft_worker
        assert isinstance(drafter, TpuDsparkDraftWorker)
        with drafter.draft_context():
            num_verify_drafts = drafter.verify_caps(
                block_hidden=block_hidden,
                bonus_tokens=bonus_tokens,
                draft_tokens=draft_tokens,
                live_mask=live_mask,
                drafts_per_live_row=budget_width - 1,
            )
        return self._tp_sync.sync(SpecTpSyncSite.SPEC_VERIFY_DRAFTS, num_verify_drafts)

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
            if self._scheduled_caps:
                self._warmup_verify_caps()
            if self._compact:
                self._warmup_compact_verify()
            if (
                self._adaptive_width is not None
                and get_disagg().disaggregation_mode != "prefill"
                and not tpu_envs.SGLANG_TPU_SPEC_VERIFY_WIDTH_DISABLE_PROFILE.get()
            ):
                self._profile_verify_widths()
        self._warmup_extend_draft_inputs()
        # The base snapshot ran before these block-draft warmups.
        snapshot_recompilation_baseline()

    def _profile_verify_widths(self) -> None:
        """Seed the adaptive width with each (bucket, candidate)'s verify-forward time."""
        assert self._adaptive_width is not None
        candidates = self._adaptive_width.candidates
        # cap-accept verifies the full window whatever the budget: time that forward.
        served = {
            width: self.num_verify_rows
            if self._scheduled_caps and not self._compact
            else width
            for width in candidates
        }
        shapes = sorted(
            {
                (pbs, served[width])
                for pbs in self._bucket_sizes()
                for width in candidates
            }
        )
        policy = self._compile_policy()
        assert isinstance(policy, TpuBlockSpecDecodeCompilePolicyMixin), type(policy)
        timed = policy.profile_target_verify(shapes)
        # Rank 0's timings for everyone, so TP ranks rank widths alike.
        timed = self._tp_group.broadcast_object(timed, src=0)
        profile = {
            (pbs, width): timed[(pbs, served[width])]
            for pbs in self._bucket_sizes()
            for width in candidates
        }
        self._adaptive_width.seed(profile)
        if self._tp_group.rank_in_group == 0:
            logger.info(
                "Adaptive verify width profile (verify ms per bucket/width): %s",
                {f"{b}/{w}": round(s * 1e3, 2) for (b, w), s in profile.items()},
            )

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
        width = rows.cache_locs.shape[1]
        block_hidden, graph_draft_tokens, draft_lattice = self._forward_draft_block(
            state=state,
            live_mask=bucket.live_mask,
            rows=self._draft_block_rows(rows, verify_input, state.seq_lens),
            verify_input=verify_input,
        )
        # The draft block always drafts gamma; a narrower step verifies the first width - 1.
        num_verify_drafts = self.num_verify_drafts(bs, width)
        temperatures, top_ks = bucket.temperatures, bucket.top_ks
        if temperatures is None or top_ks is None or not self._sampled_drafts:
            # The draft graph already drafted greedily (sample_draft_tokens_tpu).
            draft_tokens, _ = self._truncate_drafts(
                self._sync_greedy_drafts(graph_draft_tokens), None, width
            )
            if self._caps_bucket(bs):
                num_verify_drafts = self._schedule_verify_drafts(
                    block_hidden=block_hidden,
                    bonus_tokens=state.bonus_tokens,
                    draft_tokens=draft_tokens,
                    live_mask=bucket.live_mask,
                    budget_width=self._budget_width(bs),
                )
                self._step_caps = num_verify_drafts
            return DraftResult(
                draft_tokens=draft_tokens,
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
        if width != self.num_verify_rows:
            assert sampled.verify_tokens is None and sampled.draft_candidates is None
        draft_tokens, draft_logits = self._truncate_drafts(
            sampled.draft_tokens, sampled.draft_logits, width
        )
        if self._caps_bucket(bs):
            num_verify_drafts = self._schedule_verify_drafts(
                block_hidden=block_hidden,
                bonus_tokens=state.bonus_tokens,
                draft_tokens=draft_tokens,
                live_mask=bucket.live_mask,
                budget_width=self._budget_width(bs),
            )
            self._step_caps = num_verify_drafts
        return DraftResult(
            draft_tokens=draft_tokens,
            draft_logits=draft_logits,
            num_verify_drafts=num_verify_drafts,
            verify_tokens=sampled.verify_tokens,
            draft_candidates=sampled.draft_candidates,
        )

    def _draft_block_rows(
        self, rows: VerifyRows, verify_input: TpuSpecVerifyInput, seq_lens: torch.Tensor
    ) -> VerifyRows:
        """Full-width rows for the draft block, which writes draft KV into all gamma + 1 slots."""
        if rows.cache_locs.shape[1] == self.num_verify_rows:
            return rows
        return select_verify_rows(
            verify_input.verify_cache_candidates,
            verify_input.committed_lens,
            seq_lens,
            self.num_verify_rows,
        )

    def _truncate_drafts(
        self,
        draft_tokens: torch.Tensor,
        draft_logits: torch.Tensor | None,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if width == self.num_verify_rows:
            return draft_tokens, draft_logits
        return truncate_drafts(draft_tokens, draft_logits, width - 1)

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

    def verify(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        draft_result: DraftResult,
        verify_input: TpuSpecVerifyInput,
        *,
        grammar_barrier: Callable[[], None] | None = None,
    ) -> tuple[LogitsProcessorOutput, AcceptResult]:
        if not (self._compact and self._caps_bucket(rows.cache_locs.shape[0])):
            return super().verify(
                batch,
                bucket,
                rows,
                draft_result,
                verify_input,
                grammar_barrier=grammar_barrier,
            )
        # The block worker does not support grammar (supports_grammar is False).
        del grammar_barrier
        # compact: the target scores only each request's scheduled rows, packed
        # into bucket x table-width tokens, then the outputs return to the window.
        verify_tokens = draft_result.verify_tokens
        if verify_tokens is None:
            verify_tokens = build_verify_tokens(
                bucket.state.bonus_tokens, draft_result.draft_tokens
            )
        padded_bs = rows.cache_locs.shape[0]
        budget_width = self._budget_width(padded_bs)
        packed = pack_verify(
            verify_tokens,
            rows,
            draft_result.num_verify_drafts,
            bucket.live_mask,
            bucket.state.seq_lens,
            self._packed_tokens(padded_bs, budget_width),
        )
        logits_output = self._forward_target_verify(
            batch=batch,
            bucket_state=bucket.state,
            verify_tokens=packed.input_ids,
            rows=_packed_rows(packed, padded_bs),
            verify_input=verify_input,
            spec_info=DFlashVerifyInput(
                draft_token=packed.input_ids,
                positions=packed.positions,
                # Tokens per request as the compile shim pads them (bucket x width);
                # the backend reads the per-request rows from the layout.
                draft_token_num=budget_width,
                custom_mask=None,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                live_seq_lens_cpu=batch.seq_lens_cpu,
                ragged_verify_layout=packed.layout(),
            ),
            logits_output_mode=self._verify_logits_output_mode(bucket, draft_result),
        )
        _unpack_logits_output(logits_output, packed.strided_rows)
        accept = self._accept(bucket, draft_result, logits_output)
        self._check_rank_agreement(accept)
        return logits_output, accept

    @staticmethod
    def _packed_tokens(padded_bs: int, budget_width: int) -> int:
        """compact verify tokens: the per-live-row draft budget plus roots, at the bucket."""
        return padded_bs * budget_width

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
                rows.cache_locs.shape[1],
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
        # Marks decode intervals that span this prefill (on_verify_complete_cpu).
        self._extend_ct += 1
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
            ring_pool = self._block_draft_worker.swa_ring_pool
            full_to_swa = self._block_draft_worker.full_to_swa
            if ring_pool is not None:
                assert not self.ships_pd_hidden, "PD does not serve a draft SWA ring"
                cache_loc_cpu = _extend_ring_rows(
                    ring_pool=ring_pool, batch=batch, positions=positions_cpu
                ).to(cache_loc_cpu.dtype)
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
            draft_token_num=rows.cache_locs.shape[1],
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
        # The draft block always reads full-width rows (_draft_block_rows).
        width = self.num_verify_rows
        with torch.no_grad():
            for padded_bs in self._bucket_sizes():
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
        self._warmup_narrow_width_drafts()

    def _warmup_narrow_width_drafts(self) -> None:
        """Full-width draft rows and greedy draft truncation for narrower verify widths."""
        narrow = [w for w in self.verify_widths() if w != self.num_verify_rows]
        if not narrow:
            return
        with torch.no_grad():
            for padded_bs in self._bucket_sizes():
                warmup_compiled_op(
                    select_verify_rows,
                    torch.zeros(
                        (padded_bs, 2 * self.verify_window_rows),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    torch.zeros((padded_bs,), dtype=torch.int64, device=self.device),
                    torch.zeros((padded_bs,), dtype=torch.int64, device=self.device),
                    self.num_verify_rows,
                )
                for width in narrow:
                    warmup_compiled_op(
                        truncate_drafts,
                        torch.zeros(
                            (padded_bs, self.num_drafts),
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        None,
                        width - 1,
                    )

    def _warmup_verify_caps(self) -> None:
        """The confidence/caps program and caps sync per scheduling bucket, and
        the result's cap_lens per live bs."""
        hidden_size = self._draft_runner.model_config.hidden_size
        caps_buckets = {pbs for pbs, _ in self._caps_shapes()}
        runner = self.target_worker.model_runner
        max_bs = min(runner.max_running_requests, self._b_bucketer().max_val)
        with torch.no_grad():
            for bs in range(1, max_bs + 1):
                for padded_bs in self._decode_buckets_for(bs):
                    if padded_bs in caps_buckets:
                        warmup_compiled_op(
                            live_cap_lens,
                            torch.zeros(
                                (padded_bs,), dtype=torch.int32, device=self.device
                            ),
                            bs,
                        )
            for padded_bs, budget_width in self._caps_shapes():
                # Materialized like the draft graph's flat hidden output.
                block_hidden = (
                    torch.zeros(
                        (padded_bs * self._draft_block_width, hidden_size),
                        dtype=self._draft_runner.dtype,
                    )
                    .to(self.device)
                    .view(padded_bs, self._draft_block_width, -1)
                )
                live_mask = (
                    torch.zeros((padded_bs,), dtype=torch.int64, device=self.device)
                    != 0
                )
                self._schedule_verify_drafts(
                    block_hidden=block_hidden,
                    bonus_tokens=self._device_zeros(padded_bs),
                    draft_tokens=torch.zeros(
                        (padded_bs, self.num_drafts),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                    live_mask=live_mask,
                    budget_width=budget_width,
                ).cpu()

    def _warmup_compact_verify(self) -> None:
        """compact's pack op and the unpack of every verify output, per batch bucket."""
        runner = self.target_worker.model_runner
        vocab_size = runner.model_config.vocab_size
        hidden_size = self._block_draft_worker.target_hidden_size
        width = self.num_verify_rows
        with torch.no_grad():
            for padded_bs, budget_width in self._caps_shapes():
                num_tokens = self._packed_tokens(padded_bs, budget_width)

                def zeros(
                    *shape: int, dtype: torch.dtype = torch.int64
                ) -> torch.Tensor:
                    return torch.zeros(shape, dtype=dtype, device=self.device)

                packed = pack_verify(
                    zeros(padded_bs * width),
                    VerifyRows(
                        positions=zeros(padded_bs, width),
                        cache_locs=zeros(padded_bs, width),
                        flat_positions=zeros(padded_bs * width),
                        flat_cache_locs=zeros(padded_bs * width),
                    ),
                    zeros(padded_bs, dtype=torch.int32),
                    zeros(padded_bs) != 0,
                    zeros(padded_bs),
                    num_tokens,
                )
                logits = zeros(num_tokens, vocab_size, dtype=torch.float32)
                hidden = zeros(num_tokens, hidden_size, dtype=runner.dtype)
                # Each logits mode leaves a different set of outputs None.
                variants = [(logits, None, None, hidden)]
                if self._verify_top_k_width is not None:
                    top_logits = zeros(
                        num_tokens, TARGET_CANDIDATES, dtype=torch.float32
                    )
                    top_ids = zeros(num_tokens, TARGET_CANDIDATES, dtype=torch.int32)
                    variants += [
                        (None, top_logits, top_ids, hidden),
                        (logits, top_logits, top_ids, hidden),
                    ]
                for variant in variants:
                    warmup_compiled_op(
                        unpack_verify_outputs, packed.strided_rows, *variant
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
                for target_candidates, width in itertools.product(
                    (False, True), self.verify_widths()
                ):
                    sampled = self._sample_draft_block(
                        block_hidden,
                        bonus_tokens,
                        draft_lattice,
                        temperatures,
                        top_ks,
                        target_candidates=target_candidates,
                    )
                    draft_tokens, draft_logits = self._truncate_drafts(
                        sampled.draft_tokens, sampled.draft_logits, width
                    )
                    torch.cat([bonus_tokens[:, None], draft_tokens], dim=1).cpu()
                    if draft_logits is not None:
                        draft_logits.float().cpu()
                        self._warmup_sampled_accept(
                            padded_bs, draft_tokens, draft_logits
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

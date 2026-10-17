"""Minimal mathematical interface for TPU speculative decoding.

One step maps state s = (prefix x_{1:n}, root token r = x_{n+1}, draft context)
to s' through:

  draft:  s -> d, q        draft chain d_1..d_N after r, draft dists q_j
  verify: s, d, q -> p, A  target dists p on [r, d_1..d_N] in one forward,
                           then lossless acceptance A = (correct prefix, bonus)
  commit: s, d, p, A -> s' extend prefix by the accepted tokens, root := bonus,
                           feed target features phi(p-forward) to the draft

TPU only runs chains (no draft trees). EAGLE3 (autoregressive, topk 1), DFlash
(block-parallel) and DSpark (DFlash plus Markov correction and confidence
budget) differ only in how they realize these three maps. All tensors keep
static shapes; raggedness is a mask. Data types live in spec_types, shared
device ops in spec_ops; p and phi are the target's LogitsProcessorOutput.

TpuSpecWorker plugs into the scheduler as a BaseSpecWorker: decode batches go
through ScheduleBatch -> SpecState -> step -> GenerationBatchResult. verify is
shared (every chain algorithm runs the same target TARGET_VERIFY forward);
draft, commit and extend are algorithm specific.

Decode state stays padded to the batch bucket across steps and is re-padded only
when the batch composition changes; only the scheduler-facing result is live bs.
"""

from __future__ import annotations

import contextlib
import itertools
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from sglang.srt.compilation.compiled_op import warmup_compiled_op
from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import get_dp_tp_group, is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_spec
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang.srt.speculative.spec_utils import (
    SIMULATE_ACC_LEN,
    SIMULATE_ACC_METHOD,
    SIMULATE_ACC_TOKEN_MODE,
    gather_spec_extras_func,
    prepare_mamba_track_for_verify,
    sample_simulated_acc_len,
)
from sglang.srt.utils.common import ceil_align
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.recompilation_checker import (
    snapshot_recompilation_baseline,
)
from sglang_tpu.experimental import register_host_batch_padder
from sglang_tpu.experimental.sampling.sampler import HostPaddedTpuSampler
from sglang_tpu.experimental.sampling.sampling_batch_info import (
    HostPaddedSamplingInfo,
    is_host_padded,
)
from sglang_tpu.experimental.speculative.block_draft_worker import TpuBlockDraftWorker
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_drafts_greedy,
    accept_drafts_sampled,
    accept_drafts_sampled_filtered,
    accept_drafts_target_only,
    build_verify_tokens,
    finalize_accept,
    pad_state_rows,
    select_verify_rows,
    target_sampling_probs,
)
from sglang_tpu.experimental.speculative.spec_types import (
    AcceptResult,
    BucketState,
    DraftResult,
    SpecState,
    TpuSpecVerifyInput,
    VerifyRows,
    map_carry,
)
from sglang_tpu.experimental.speculative.spec_utils import overridden_fields
from sglang_tpu.overlap_utils_tpu import flush_direct_handoff_tpu
from sglang_tpu.sampling_mask import tpu_filter_probs

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

# apply_dflash_simulated_acceptance's fixed_token_id, so both paths emit the same tokens.
_SIMULATED_TOKEN = 100


# The target sampler compiles one filter program per need_* combination.
_SAMPLING_WARMUP_VARIANTS = [{"greedy": True}] + [
    {"greedy": False, "top_k": top_k, "top_p": top_p, "min_p": min_p}
    for top_k, top_p, min_p in itertools.product((False, True), repeat=3)
]


class TpuSpecWorker(BaseSpecWorker):
    """Speculative decoding as draft -> verify -> commit.

    Every abstract method is per-algorithm but required: the maps, extend, and
    the batch.spec_info / verify ForwardBatch formats. verify, decode dispatch
    and result packing are shared.
    """

    def __init__(
        self,
        target_worker: TpModelWorker,
        draft_worker: TpuBlockDraftWorker | EagleDraftWorker,
        num_drafts: int,
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        if parallel.pp_size > 1:
            raise NotImplementedError("TpuSpecWorker does not support PP.")
        self._target_worker = target_worker
        runner = target_worker.model_runner
        if get_exec().features.enable_custom_logit_processor:
            # Processors expect len(sampling_info) live rows, not bucket rows.
            raise NotImplementedError(
                "TpuSpecWorker does not support custom logit processors."
            )
        self._bsearch_filtering = envs.SGLANG_TPU_ENABLE_BSEARCH_SAMPLING.get()
        if not isinstance(runner.sampler, HostPaddedTpuSampler):
            # HOST_PADDED sampling params stay on host until TpuSampler uploads
            # them; with full logits it samples exactly like a plain Sampler
            # (--sampling-backend pytorch).
            runner.sampler = HostPaddedTpuSampler()
        self._draft_worker = draft_worker
        self.num_drafts = num_drafts
        self.num_verify_rows = num_drafts + 1
        self.device = target_worker.device
        # Ranks that share one batch: the attention-TP group under DP attention.
        self._tp_sync = SpecTpSync(get_dp_tp_group())
        self._dp_attention = is_dp_attention_enabled()
        if self._dp_attention and (
            envs.SGLANG_ENABLE_DP_SPEC_PREFILL_COORDINATION.get()
            or get_spec().speculative_skip_dp_mlp_sync
        ):
            # Both let DP ranks mix prefill and decode in one step; dispatch and
            # the idle-rank mirror assume every rank runs the same phase.
            raise NotImplementedError(
                "TpuSpecWorker does not support mixed prefill/decode DP steps."
            )
        self._attn_tp_size = parallel.attn_tp_size
        if self._dp_attention and self._tp_sync.enabled(
            SpecTpSyncSite.SPEC_ACCEPT_GREEDY
        ) != self._tp_sync.enabled(SpecTpSyncSite.SPEC_ACCEPT_SAMPLE):
            # An idle DP rank cannot tell which accept its peers ran.
            raise ValueError(
                "SGLANG_SPEC_TP_SYNC must enable both or neither spec-accept site "
                "under DP attention."
            )
        self._bucket: BucketState | None = None
        self._accept_threshold_single = get_spec().speculative_accept_threshold_single
        self._accept_threshold_acc = get_spec().speculative_accept_threshold_acc
        # Filled per batch bucket once the target compile policy exists.
        self._all_drafts: dict[int, torch.Tensor] = {}
        # Benchmark knob: force the accept length (SGLANG_SIMULATE_ACC_LEN), as core DFlash does.
        self._simulate_acc_len = SIMULATE_ACC_LEN
        if self._simulate_acc_len > 0 and SIMULATE_ACC_TOKEN_MODE != "fixed":
            raise NotImplementedError(
                "TpuSpecWorker only supports SGLANG_SIMULATE_ACC_TOKEN_MODE=fixed, "
                f"got {SIMULATE_ACC_TOKEN_MODE!r}."
            )
        # Filled per batch bucket in init_cuda_graphs when simulating.
        self._simulated_drafts: dict[int, torch.Tensor] = {}
        register_host_batch_padder(self._pad_bs)

    def init_cuda_graphs(self) -> None:
        super().init_cuda_graphs()
        self._all_drafts = {
            padded_bs: torch.full(
                (padded_bs,), self.num_drafts, dtype=torch.int32, device=self.device
            )
            for padded_bs in self._b_bucketer().buckets
        }
        self._warmup_logical_bs_ops()
        self._warmup_extend_sampling()
        self._warmup_filtered_target()
        self._warmup_relay_and_results()
        if self._simulate_acc_len > 0:
            self._simulated_drafts = {
                pbs: torch.full(
                    (pbs, self.num_drafts), _SIMULATED_TOKEN, dtype=torch.int64
                ).to(self.device)
                for pbs in self._b_bucketer().buckets
            }
        snapshot_recompilation_baseline()

    def _warmup_logical_bs_ops(self) -> None:
        # torch_tpu compiles one program per shape even for a plain H2D copy.
        max_bs = min(
            self.target_worker.model_runner.max_running_requests,
            self._b_bucketer().max_val,
        )
        live_sizes = range(1, max_bs + 1)
        # Every bucket: DP attention can run any bucket above a rank's own.
        bucket_sizes = list(self._b_bucketer().buckets)
        # Live bs: HOST_PADDED req_pool_indices re-uploaded after filter/merge.
        # Bucket: host-padded sampling params and the plan's padded verify window.
        h2d_shapes = (
            [((bs,), torch.int64) for bs in live_sizes]
            + [
                ((pbs,), dtype)
                for pbs in bucket_sizes
                for dtype in (torch.int64, torch.int32, torch.float32)
            ]
            + [((pbs, 1), torch.float32) for pbs in bucket_sizes]
            + [((pbs, 2 * self.num_verify_rows), torch.int64) for pbs in bucket_sizes]
        )
        with torch.no_grad():
            for bs in live_sizes:
                # _bucket_state pads the relayed live state to each bucket it can run.
                for padded_bs in self._decode_buckets_for(bs):
                    warmup_compiled_op(
                        pad_state_rows,
                        torch.zeros((bs,), dtype=torch.int64, device=self.device),
                        torch.zeros((bs,), dtype=torch.int64, device=self.device),
                        self.make_dummy_carry(bs),
                        padded_bs,
                    )
            for shape, dtype in h2d_shapes:
                torch.zeros(shape, dtype=dtype).to(self.device, non_blocking=True).cpu()
            for pbs in bucket_sizes:
                # _bucket_state derives live_mask from the padded pool rows.
                (torch.zeros((pbs,), dtype=torch.int64, device=self.device) != 0).cpu()

    @abstractmethod
    def draft(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> DraftResult:
        """Sample a draft chain d ~ q after bucket.state.bonus_tokens.

        Why: verify needs the candidate tokens d and, for sampled requests, the
        q they were drawn from (lossless acceptance divides by q).
        Noop: never; an algorithm without drafts is plain decode.
        """

    @abstractmethod
    def commit(
        self,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> SpecState:
        """Commit the accepted tokens: advance the prefix, update the draft context.

        Why: the next draft must condition on the committed tokens' target
        features (DFlash/DSpark inject them into draft KV, EAGLE3 carries them).
        Noop: the draft-context update is a noop for a draft that never reads
        target features; the advanced SpecState must still be returned.
        """

    @abstractmethod
    def extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        """Extend: target forward on new prompt tokens, seed the draft context, set next_draft_input.

        Why: new prompt tokens have no draft context yet, and the first decode
        step reads its root token from next_draft_input.
        Noop: the draft seeding is a noop for a draft that never reads target
        features; the target forward and next_draft_input never are.
        """

    @abstractmethod
    def get_spec_state(self, batch: ScheduleBatch) -> SpecState:
        """Get the live [bs] SpecState from batch.spec_info.

        Why: batch.spec_info is the algorithm's own SpecInput type, which the
        scheduler carries across steps (prepare_for_decode, filter/merge).
        Noop: never; the root tokens live only there.
        """

    @abstractmethod
    def state_to_spec_info(self, state: SpecState) -> SpecInput:
        """SpecState -> the next step's batch.spec_info.

        Why: the inverse of get_spec_state; the scheduler stores the result and
        hands it back next step.
        Noop: never.
        """

    @abstractmethod
    def make_verify_input(
        self, *, batch: ScheduleBatch, verify_tokens: torch.Tensor, rows: VerifyRows
    ) -> SpecInput:
        """The algorithm's own TARGET_VERIFY spec_info for [pbs * (N + 1)] verify rows.

        Why: attention backends and the dynamo tracer key on the algorithm's
        verify type; its compile policy must warm the same structure.
        Noop: never.
        """

    def make_dummy_carry(self, rows: int) -> Any:
        """A zero draft_carry of rows rows on device, shaped and typed like serving.

        Why: warmup and idle results must hand the relay the carry's real
        layout. Noop (None) for drafts that keep their context in draft KV.
        """
        return None

    def verify(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        draft_result: DraftResult,
        verify_input: TpuSpecVerifyInput,
    ) -> tuple[LogitsProcessorOutput, AcceptResult]:
        """Score [r, d] with the target in one forward and accept losslessly.

        The returned LogitsProcessorOutput holds [pbs * (N + 1), ...] verify rows:
        next_token_logits is p, hidden_states is the target feature phi.
        """
        verify_tokens = draft_result.verify_tokens
        if verify_tokens is None:
            verify_tokens = build_verify_tokens(
                bucket.state.bonus_tokens, draft_result.draft_tokens
            )
        logits_output = self._forward_target_verify(
            batch=batch,
            bucket_state=bucket.state,
            verify_tokens=verify_tokens,
            rows=rows,
            verify_input=verify_input,
        )
        target_logits = logits_output.next_token_logits
        assert target_logits.shape[0] == draft_result.draft_tokens.shape[0] * (
            self.num_verify_rows
        ), "verify logits must keep the batch bucket rows"
        if bucket.temperatures is None or bucket.top_ks is None:
            num_correct_drafts, bonus_tokens = accept_drafts_greedy(
                target_logits, draft_result.draft_tokens, draft_result.num_verify_drafts
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_GREEDY
        elif draft_result.draft_logits is None:
            # A deterministic draft is a one-hot q: TPU EAGLE's target-only rule.
            num_correct_drafts, bonus_tokens = accept_drafts_target_only(
                self._filtered_target_probs(target_logits, bucket),
                draft_result.draft_tokens,
                draft_result.num_verify_drafts,
                bucket.top_ks,
                self._accept_threshold_single,
                self._accept_threshold_acc,
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_SAMPLE
        elif bucket.need_filters:
            num_correct_drafts, bonus_tokens = accept_drafts_sampled_filtered(
                self._filtered_target_probs(target_logits, bucket),
                draft_result.draft_tokens,
                draft_result.num_verify_drafts,
                draft_result.draft_logits.float(),
                bucket.temperatures,
                bucket.top_ks,
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_SAMPLE
        else:
            num_correct_drafts, bonus_tokens = accept_drafts_sampled(
                target_logits,
                draft_result.draft_tokens,
                draft_result.num_verify_drafts,
                draft_result.draft_logits.float(),
                bucket.temperatures,
                bucket.top_ks,
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_SAMPLE
        draft_tokens = draft_result.draft_tokens
        if self._simulate_acc_len > 0:
            num_correct_drafts, bonus_tokens, draft_tokens = self._simulated_accept(
                draft_tokens.shape[0]
            )
        # Sampled acceptance draws per-rank randoms; ranks must commit rank 0's outcome.
        self._tp_sync.sync(site, num_correct_drafts)
        self._tp_sync.sync(site, bonus_tokens)
        accept = finalize_accept(
            draft_tokens,
            num_correct_drafts,
            bonus_tokens,
            bucket.state.seq_lens,
            bucket.live_mask,
        )
        return logits_output, accept

    def _simulated_accept(
        self, padded_bs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forced num_correct_drafts, bonus_tokens and draft_tokens, all _SIMULATED_TOKEN."""
        # Host CPU RNG, seeded alike on every rank, as in apply_dflash_simulated_acceptance.
        commit_len = sample_simulated_acc_len(
            self._simulate_acc_len, SIMULATE_ACC_METHOD, self.num_verify_rows
        )
        # Built on host and uploaded: a device fill would bake the value into the program.
        num_correct_drafts = torch.full(
            (padded_bs,), commit_len - 1, dtype=torch.int64
        ).to(self.device, non_blocking=True)
        bonus_tokens = torch.full((padded_bs,), _SIMULATED_TOKEN, dtype=torch.int64).to(
            self.device, non_blocking=True
        )
        return num_correct_drafts, bonus_tokens, self._simulated_drafts[padded_bs]

    def _filtered_target_probs(
        self, target_logits: torch.Tensor, bucket: BucketState
    ) -> torch.Tensor:
        """Target p [pbs * (N + 1), V]: temperature, then the target sampler's filter.

        Like TpuSampler, every cutoff is taken on the unfiltered probabilities and
        the survivors are normalized once.
        """
        assert bucket.temperatures is not None and bucket.top_ks is not None
        assert bucket.top_ps is not None and bucket.min_ps is not None
        probs, top_ks_rows, top_ps_rows, min_ps_rows = target_sampling_probs(
            target_logits,
            bucket.temperatures,
            bucket.top_ks,
            bucket.top_ps,
            bucket.min_ps,
            self.num_verify_rows,
        )
        return _joint_filter_renorm(probs, top_ks_rows, top_ps_rows, min_ps_rows)

    def step(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> tuple[SpecState, LogitsProcessorOutput, AcceptResult]:
        draft_result = self.draft(batch, bucket, rows, verify_input)
        logits_output, accept = self.verify(
            batch, bucket, rows, draft_result, verify_input
        )
        next_state = self.commit(
            batch, bucket.state, rows, verify_input, logits_output, accept
        )
        return next_state, logits_output, accept

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish: Callable[[torch.Tensor], None] | None = None,
        grammar_barrier: Callable[[], None] | None = None,
        *,
        pp_proxy_tensors: PPProxyTensors | None = None,
    ) -> GenerationBatchResult:
        self._check_supported(batch=batch, pp_proxy_tensors=pp_proxy_tensors)
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            result = self.extend(batch)
        elif batch.forward_mode.is_idle():
            if self._dp_attention and max(batch.global_num_tokens or (0,)) > 0:
                # DP peers decode: enter their step's collectives too.
                self._forward_idle_decode(batch)
            result = self._idle_result()
        else:
            result = self._forward_decode(batch)
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        return result

    def _forward_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        bs = batch.seq_lens.shape[0]
        padded_bs = self._decode_bucket(batch, bs)
        verify_input = TpuSpecVerifyInput.from_batch(
            batch, self.num_verify_rows, padded_bs
        )
        bucket = self._bucket_state(batch, padded_bs, verify_input)
        # req_to_token is host-resident on TPU; the host gathered the candidate
        # slot window, so the block is picked on device without a D2H sync.
        rows = select_verify_rows(
            verify_input.verify_cache_candidates,
            verify_input.committed_lens,
            bucket.state.seq_lens,
        )
        next_state, logits_output, accept = self.step(batch, bucket, rows, verify_input)
        # The scheduler stores this key as the next spec_info.future_indices.
        self._bucket = bucket._replace(key=batch.req_pool_indices, state=next_state)
        # Hidden rows stay on device; the overlap result copy would ship them to CPU.
        logits_output.hidden_states = None
        return self._pack_decode_result(
            bs=bs, next_state=next_state, logits_output=logits_output, accept=accept
        )

    def _decode_buckets_for(self, bs: int) -> list[int]:
        """Decode buckets a rank with bs live requests can run (_decode_bucket)."""
        local = self._pad_bs(bs)
        if not self._dp_attention:
            return [local]
        smallest = max(local, self._pad_bs(ceil_align(bs, self._attn_tp_size)))
        return [b for b in self._b_bucketer().buckets if b >= smallest]

    def _decode_bucket(self, batch: ScheduleBatch, bs: int) -> int:
        """The decode batch bucket; DP-attention ranks agree on one from the synced sizes.

        Matches the draft's DP peer rows (requests aligned to attention TP, topk
        1), so the policies' DP padding of draft, verify and draft extend is a no-op.
        """
        if not self._dp_attention:
            return self._pad_bs(bs)
        max_bs = max(batch.global_num_tokens)
        assert max_bs >= bs, (batch.global_num_tokens, bs)
        return self._pad_bs(ceil_align(max_bs, self._attn_tp_size))

    def _bucket_state(
        self,
        batch: ScheduleBatch,
        padded_bs: int,
        verify_input: TpuSpecVerifyInput,
    ) -> BucketState:
        """The previous step's padded state, re-padded after filter/merge/extend or a new bucket."""
        bucket = self._bucket
        # filter_batch/merge_batch rebind future_indices; identity pins the
        # composition. Under DP attention the shared bucket can move by itself.
        if (
            bucket is not None
            and verify_input.future_indices is bucket.key
            and bucket.state.seq_lens.shape[0] == padded_bs
        ):
            return bucket
        state = self.get_spec_state(batch)
        req_pool_indices = verify_input.req_pool_indices
        assert req_pool_indices.dtype == batch.req_pool_indices.dtype
        sampling_info = batch.sampling_info
        assert is_host_padded(sampling_info)
        sampled = not sampling_info.is_all_greedy
        # Host rows padded to the local bucket; DP attention may run a larger one.
        params = sampling_info.host_params_at(padded_bs) if sampled else {}
        bonus_tokens, seq_lens, draft_carry = pad_state_rows(
            state.bonus_tokens, state.seq_lens, state.draft_carry, padded_bs
        )
        return BucketState(
            key=batch.req_pool_indices,
            state=SpecState(
                bonus_tokens=bonus_tokens,
                seq_lens=seq_lens,
                req_pool_indices=req_pool_indices,
                draft_carry=draft_carry,
            ),
            # The plan pads with the reserved pool row 0, which no live request holds.
            live_mask=req_pool_indices != 0,
            temperatures=(
                params["temperatures"].view(-1).to(self.device, non_blocking=True)
                if sampled
                else None
            ),
            top_ks=(
                params["top_ks"].to(self.device, non_blocking=True) if sampled else None
            ),
            top_ps=(
                params["top_ps"].to(self.device, non_blocking=True) if sampled else None
            ),
            min_ps=(
                params["min_ps"].to(self.device, non_blocking=True) if sampled else None
            ),
            need_top_k=sampled and sampling_info.need_top_k_sampling,
            need_top_p=sampled and sampling_info.need_top_p_sampling,
            need_min_p=sampled and sampling_info.need_min_p_sampling,
        )

    def _forward_idle_decode(self, batch: ScheduleBatch) -> None:
        """Run a decode step's forwards on all-padding rows of a DP rank with no requests.

        Every DP rank must issue the same collectives in order, those over its
        own TP group included; the active code at the shared bucket does.
        """
        padded_bs = self._decode_bucket(batch, 0)
        verify_input = TpuSpecVerifyInput.idle(
            num_verify_rows=self.num_verify_rows,
            padded_bs=padded_bs,
            device=self.device,
        )
        # Distinct tensors: the dynamo tracer guards on aliased inputs, and the
        # active step never passes one tensor for two of these fields.
        state = SpecState(
            bonus_tokens=self._device_zeros(padded_bs),
            seq_lens=self._device_zeros(padded_bs),
            req_pool_indices=verify_input.req_pool_indices,
            draft_carry=self.make_dummy_carry(padded_bs),
        )
        rows = select_verify_rows(
            verify_input.verify_cache_candidates,
            verify_input.committed_lens,
            state.seq_lens,
        )
        # The forwards' preparation writes these; the idle batch must stay idle.
        with overridden_fields(
            batch,
            forward_mode=ForwardMode.DECODE,
            spec_info=batch.spec_info,
            input_ids=batch.input_ids,
            out_cache_loc=batch.out_cache_loc,
        ):
            self._forward_idle_step(
                batch=batch, state=state, rows=rows, verify_input=verify_input
            )

    def _forward_idle_step(
        self,
        *,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> None:
        """The decode forwards on idle rows: the target verify, plus any draft forwards."""
        draft_tokens = torch.zeros(
            (state.seq_lens.shape[0], self.num_drafts),
            dtype=torch.int64,
            device=self.device,
        )
        self._forward_idle_verify(
            batch=batch,
            state=state,
            rows=rows,
            verify_input=verify_input,
            verify_tokens=build_verify_tokens(state.bonus_tokens, draft_tokens),
        )

    def _forward_idle_verify(
        self,
        *,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
        verify_tokens: torch.Tensor,
    ) -> LogitsProcessorOutput:
        """verify()'s collectives on idle rows: the target forward and the accept broadcasts.

        A collective over a DP rank's own TP group still needs every DP rank to
        issue it, so the idle rank mirrors the [pbs] int64 accept broadcasts.
        """
        logits_output = self._forward_target_verify(
            batch=batch,
            bucket_state=state,
            verify_tokens=verify_tokens,
            rows=rows,
            verify_input=verify_input,
        )
        site = SpecTpSyncSite.SPEC_ACCEPT_GREEDY
        padded_bs = state.seq_lens.shape[0]
        self._tp_sync.sync(site, self._device_zeros(padded_bs))
        self._tp_sync.sync(site, self._device_zeros(padded_bs))
        return logits_output

    def _device_zeros(self, rows: int) -> torch.Tensor:
        """[rows] int64 zeros by a warmed H2D copy; a device fill compiles per shape."""
        return torch.zeros((rows,), dtype=torch.int64).to(
            self.device, non_blocking=True
        )

    def _pad_bs(self, bs: int) -> int:
        return self._b_bucketer().next_bucket(bs)

    def _warmup_extend_sampling(self) -> None:
        """Extend samples at the batch bucket off the extend cap rows; warm every variant."""
        runner = self.target_worker.model_runner
        cap = self._compile_policy()._extend_compile_cap()
        if self._pad_bs(cap) != cap:
            raise ValueError(
                f"TpuSpecWorker needs the extend compile cap {cap} to be a batch "
                f"bucket; extend would sample {self._pad_bs(cap)} rows off {cap}."
            )
        extend_sizes = range(1, cap + 1)
        vocab_size = runner.model_config.vocab_size
        with torch.no_grad():
            for pbs in sorted({self._pad_bs(bs) for bs in extend_sizes}):
                for variant in _SAMPLING_WARMUP_VARIANTS:
                    # Materialized like the compiled output, then the finalize trim.
                    logits = torch.zeros((cap, vocab_size), dtype=torch.float32).to(
                        self.device
                    )
                    runner.sampler(
                        LogitsProcessorOutput(next_token_logits=logits[:pbs]),
                        self._warmup_sampling_info(pbs, vocab_size, **variant),
                        False,
                        None,
                        None,
                        torch.zeros((pbs,), dtype=torch.int64, device=self.device),
                    ).cpu()
            for bs in extend_sizes:
                # The shim pads the device-resident extend bs fields (seq_lens,
                # req_pool_indices int64; orig_seq_lens int32) to the cap.
                for dtype in (torch.int64, torch.int32):
                    torch.nn.functional.pad(
                        torch.zeros((bs,), dtype=dtype).to(self.device), (0, cap - bs)
                    ).cpu()
                # ModelRunner.sample positions, then the block-draft extend trim.
                (torch.zeros((bs,), dtype=torch.int64).to(self.device) - 1).cpu()
                torch.zeros((self._pad_bs(bs),), dtype=torch.int64, device=self.device)[
                    :bs
                ].cpu()

    def _warmup_filtered_target(self) -> None:
        """Sampled-verify target probs and their joint filter per batch bucket, chained as served."""
        vocab_size = self.target_worker.model_runner.model_config.vocab_size
        rows = self.num_verify_rows
        with torch.no_grad():
            for padded_bs in self._b_bucketer().buckets:
                # Must match the serving call: float32 logits and params, int32 top-k.
                probs = torch.zeros(
                    (padded_bs * rows, vocab_size),
                    dtype=torch.float32,
                    device=self.device,
                )
                # Serving uploads each param separately; Dynamo guards on aliasing.
                temperatures, top_ps, min_ps = (
                    torch.ones((padded_bs,), dtype=torch.float32, device=self.device)
                    for _ in range(3)
                )
                top_ks = torch.ones((padded_bs,), dtype=torch.int32, device=self.device)
                # Feed the op's own outputs to the filter, as serving does: under
                # Dynamo they carry the compiled graph's layout, which the eager
                # renorm after the filter compiles against.
                _joint_filter_renorm(
                    *target_sampling_probs(
                        probs, temperatures, top_ks, top_ps, min_ps, rows
                    )
                )
            # Run the eager filters without copying [pbs * (N + 1), V] rows back.
            torch.accelerator.synchronize()

    def _warmup_relay_and_results(self) -> None:
        """FutureMap relay and result D2H run per live bs outside any compiled op."""
        runner = self.target_worker.model_runner
        max_bs = min(runner.max_running_requests, self._b_bucketer().max_val)
        # A scratch map runs the production publish/stash; live buffers stay untouched.
        future_map = FutureMap(
            self.device, runner.spec_algorithm, runner.req_to_token_pool
        )
        with torch.no_grad():
            for bs in range(1, max_bs + 1):
                indices = torch.zeros((bs,), dtype=torch.int64).to(self.device)
                # Decode relays [:bs] views of the state at each bucket the rank
                # can run; extend relays a compact seq_lens upload.
                relays = [
                    (
                        torch.zeros(
                            (padded_bs,), dtype=torch.int64, device=self.device
                        )[:bs],
                        map_carry(self.make_dummy_carry(padded_bs), lambda t: t[:bs]),
                    )
                    for padded_bs in self._decode_buckets_for(bs)
                ]
                relays.append(
                    (
                        torch.zeros((bs,), dtype=torch.int64).to(self.device),
                        self.make_dummy_carry(bs),
                    )
                )
                for rows, carry in relays:
                    future_map.publish(indices, rows)
                    future_map.stash(
                        indices,
                        RelayPayload.from_draft_input(
                            self.state_to_spec_info(
                                SpecState(
                                    bonus_tokens=rows,
                                    seq_lens=rows,
                                    req_pool_indices=indices,
                                    draft_carry=carry,
                                )
                            )
                        ),
                    )
                    # The TPU EAGLE3 direct handoff defers the store to the next
                    # resolve, which flushes it for host-padded inputs.
                    flush_direct_handoff_tpu(future_map)
                # resolve_seq_lens_cpu / _resolve_spec_extras gathers.
                future_map.new_seq_lens_buf[indices].cpu()
                future_map.output_tokens_buf[indices].cpu()
                if future_map.need_topk:
                    for tensor in gather_spec_extras_func(
                        indices,
                        future_map.topk_p_buf,
                        future_map.topk_index_buf,
                        future_map.output_tokens_buf,
                        future_map.hidden_states_buf
                        if future_map.need_hidden_states
                        else None,
                    ):
                        if tensor is not None:
                            tensor.cpu()
                for padded_bs in self._decode_buckets_for(bs):
                    # _pack_decode_result's rows, then GenerationBatchResult.copy_to_cpu.
                    _live_rows(
                        torch.zeros(
                            (padded_bs, self.num_verify_rows),
                            dtype=torch.int64,
                            device=self.device,
                        ),
                        bs,
                    ).reshape(-1).cpu()
                    _live_rows(
                        torch.zeros(
                            (padded_bs,), dtype=torch.int32, device=self.device
                        ),
                        bs,
                    ).cpu()
            if self._dp_attention:
                for padded_bs in self._b_bucketer().buckets:
                    # _forward_idle_step's all-padding drafts.
                    torch.zeros(
                        (padded_bs, self.num_drafts),
                        dtype=torch.int64,
                        device=self.device,
                    ).cpu()

    def _warmup_sampling_info(
        self,
        padded_bs: int,
        vocab_size: int,
        *,
        greedy: bool,
        top_k: bool = False,
        top_p: bool = False,
        min_p: bool = False,
    ) -> SamplingBatchInfo:
        top_k_value = 1 if greedy else 2 if top_k else TOP_K_ALL
        return HostPaddedSamplingInfo(
            temperatures=torch.ones((padded_bs, 1)),
            top_ps=torch.full((padded_bs,), 0.9 if top_p else 1.0),
            top_ks=torch.full((padded_bs,), top_k_value, dtype=torch.int32),
            min_ps=torch.full((padded_bs,), 0.1 if min_p else 0.0),
            is_all_greedy=greedy,
            is_any_greedy=greedy,
            need_top_p_sampling=top_p,
            need_top_k_sampling=top_k_value != TOP_K_ALL,
            need_min_p_sampling=min_p,
            vocab_size=vocab_size,
            device=self.device,
            host_pad_bs=self._pad_bs,
            real_batch_size=padded_bs,
        )

    def _compile_policy(self) -> DecoderTpuCompilePolicy:
        compile_policy = self.target_worker.model_runner.compile_policy
        if not isinstance(compile_policy, DecoderTpuCompilePolicy):
            raise TypeError(
                f"TpuSpecWorker needs a DecoderTpuCompilePolicy, got {type(compile_policy)}."
            )
        return compile_policy

    def _b_bucketer(self) -> Bucketer:
        b_bucketer = self._compile_policy().b_bucketer
        assert b_bucketer is not None
        return b_bucketer

    @staticmethod
    def _bucket_view(
        batch: ScheduleBatch,
        *,
        bucket_state: SpecState,
        verify_input: TpuSpecVerifyInput,
        kv_rows: int,
        **overrides: Any,
    ) -> contextlib.AbstractContextManager[ScheduleBatch]:
        """batch with its row fields at the bucket, restored on exit, for code that reads them.

        init_new snapshots these; feeding the bucket rows leaves the shim nothing
        to pad. Padded rows hold pool row 0 and host KV length 0, inactive in RPA;
        seq_lens_cpu bounds the KV length once kv_rows rows past seq_lens are
        written. overrides set (and restore) further batch fields.
        """
        padded_bs = bucket_state.seq_lens.shape[0]
        assert verify_input.req_pool_indices_cpu.shape[0] == padded_bs
        seq_lens_cpu, seq_lens_sum = verify_input.padded_kv_lens_after_rows(kv_rows)
        multimodal_inputs = batch.multimodal_inputs
        if multimodal_inputs is not None:
            # Spec M-RoPE positions read one entry per (bucket) row.
            multimodal_inputs = multimodal_inputs + [None] * (
                padded_bs - len(multimodal_inputs)
            )
        view_fields = {
            "multimodal_inputs": multimodal_inputs,
            "seq_lens": bucket_state.seq_lens,
            "seq_lens_cpu": seq_lens_cpu,
            "seq_lens_sum": seq_lens_sum,
            # Ignored by direct compile on TPU; None keeps the shim from padding it per bs.
            "orig_seq_lens": None,
            "req_pool_indices": bucket_state.req_pool_indices,
            "req_pool_indices_cpu": verify_input.req_pool_indices_cpu,
            **overrides,
        }
        return overridden_fields(batch, **view_fields)

    def _build_verify_batch(
        self,
        *,
        batch: ScheduleBatch,
        bucket_state: SpecState,
        verify_tokens: torch.Tensor,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> ForwardBatch:
        """TARGET_VERIFY ForwardBatch already at the [pbs, N + 1] bucket; the shim pads nothing."""
        spec_info = self.make_verify_input(
            batch=batch, verify_tokens=verify_tokens, rows=rows
        )
        batch.out_cache_loc = rows.flat_cache_locs
        with self._bucket_view(
            batch,
            bucket_state=bucket_state,
            verify_input=verify_input,
            kv_rows=self.num_verify_rows,
        ) as view:
            return self._prepare_verify_forward_batch(view, spec_info)

    def _prepare_verify_forward_batch(
        self, batch: ScheduleBatch, spec_info: SpecInput
    ) -> ForwardBatch:
        # The window already owns the verify slots (batch.out_cache_loc), so this
        # never calls a verify input's prepare_for_verify, which may allocate.
        # Attention metadata is left to the runner, which pads to buckets first.
        batch.input_ids = spec_info.draft_token
        batch.spec_info = spec_info
        batch.forward_mode = (
            ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.TARGET_VERIFY
        )
        if not batch.forward_mode.is_idle():
            prepare_mamba_track_for_verify(batch)
        forward_batch = ForwardBatch.init_new(
            batch,
            self.target_worker.model_runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )
        graph_runner = self.target_worker.model_runner.decode_cuda_graph_runner
        if graph_runner and graph_runner.can_run_graph(forward_batch):
            graph_runner.load_batch(forward_batch)
        return forward_batch

    def _forward_target_verify(
        self,
        *,
        batch: ScheduleBatch,
        bucket_state: SpecState,
        verify_tokens: torch.Tensor,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> LogitsProcessorOutput:
        verify_forward_batch = self._build_verify_batch(
            batch=batch,
            bucket_state=bucket_state,
            verify_tokens=verify_tokens,
            rows=rows,
            verify_input=verify_input,
        )
        # Keep logits/hidden on the batch bucket so the compiled accept op needs no re-pad.
        assert verify_forward_batch.spec_info is not None
        verify_forward_batch.spec_info.keep_outputs_padded = True
        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            # prepare_for_verify leaves TPU attention metadata to the runner.
            skip_attn_backend_init=False,
        )
        # Custom logit processors, penalties and logit bias are rejected up front.
        return target_out.logits_output

    def _pack_decode_result(
        self,
        *,
        bs: int,
        next_state: SpecState,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> GenerationBatchResult:
        # The scheduler relays and accounts per live request, so cut the bucket rows here.
        accept_lens = _live_rows(accept.accept_lens, bs)
        new_seq_lens = next_state.seq_lens[:bs]
        logits_output.next_token_logits = logits_output.next_token_logits[
            : bs * self.num_verify_rows
        ]
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=_live_rows(accept.accept_tokens, bs).reshape(-1),
            accept_lens=accept_lens,
            block_accept_lens=accept_lens,
            next_draft_input=self.state_to_spec_info(
                SpecState(
                    bonus_tokens=next_state.bonus_tokens[:bs],
                    seq_lens=new_seq_lens,
                    req_pool_indices=next_state.req_pool_indices[:bs],
                    draft_carry=map_carry(next_state.draft_carry, lambda t: t[:bs]),
                )
            ),
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=self.num_verify_rows,
            new_seq_lens=new_seq_lens,
        )

    def _idle_result(self) -> GenerationBatchResult:
        empty_tokens = torch.empty((0,), dtype=torch.int64, device=self.device)
        empty_state = SpecState(
            bonus_tokens=empty_tokens,
            seq_lens=empty_tokens,
            req_pool_indices=empty_tokens,
            draft_carry=self.make_dummy_carry(0),
        )
        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=empty_tokens,
            accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            block_accept_lens=torch.empty((0,), dtype=torch.int32, device=self.device),
            next_draft_input=self.state_to_spec_info(empty_state),
            can_run_cuda_graph=False,
            speculative_num_draft_tokens=self.num_verify_rows,
            new_seq_lens=empty_tokens,
        )

    def _check_supported(
        self, *, batch: ScheduleBatch, pp_proxy_tensors: PPProxyTensors | None
    ) -> None:
        if pp_proxy_tensors is not None:
            raise NotImplementedError("TpuSpecWorker does not support PP.")
        if batch.has_grammar:
            raise NotImplementedError("TpuSpecWorker does not support grammar.")
        if batch.return_logprob:
            raise NotImplementedError("TpuSpecWorker does not support logprobs.")
        sampling_info = batch.sampling_info
        if sampling_info is None:
            return
        # Their per-request device state is not bucket-padded, so filter/merge
        # would compile per batch-size pair.
        if (
            sampling_info.logit_bias is not None
            or sampling_info.acc_additive_penalties is not None
            or sampling_info.acc_scaling_penalties is not None
            or (
                sampling_info.penalizer_orchestrator is not None
                and sampling_info.penalizer_orchestrator.is_required
            )
        ):
            raise NotImplementedError(
                "TpuSpecWorker does not support penalties or logit_bias."
            )
        # The sort-based target filter keeps different boundary ties than the
        # verify filter, so filtered sampling needs the sort-free path on both.
        if not self._bsearch_filtering and (
            sampling_info.need_top_k_sampling
            or sampling_info.need_top_p_sampling
            or sampling_info.need_min_p_sampling
        ):
            raise NotImplementedError(
                "TpuSpecWorker needs SGLANG_TPU_ENABLE_BSEARCH_SAMPLING for "
                "top-k/top-p/min-p sampling."
            )
        # Seeded sampling reads live-bs positions against bucket-padded seeds.
        if sampling_info.sampling_seed is not None:
            raise NotImplementedError(
                "TpuSpecWorker does not support deterministic sampling."
            )


def _live_rows(tensor: torch.Tensor, bs: int) -> torch.Tensor:
    # The result-copy thread reads these after the next step is queued; a [:bs]
    # view would then need its own program behind that step, so copy it here.
    return tensor if tensor.shape[0] == bs else tensor[:bs].clone()


def _joint_filter_renorm(
    probs: torch.Tensor,
    top_ks: torch.Tensor,
    top_ps: torch.Tensor,
    min_ps: torch.Tensor,
) -> torch.Tensor:
    # Every flag on: disabled rows carry no-op values (TOP_K_ALL, 1.0, 0.0), so
    # this is one program per shape instead of one per flag combination.
    kept = tpu_filter_probs(
        probs,
        top_ks,
        top_ps,
        min_ps,
        need_top_k=True,
        need_top_p=True,
        need_min_p=True,
    )
    return kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-12)

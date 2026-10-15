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
static shapes; raggedness is a mask. Data types live in tpu_spec_types, shared
device ops in tpu_spec_ops; p and phi are the target's LogitsProcessorOutput.

TpuSpecWorker plugs into the scheduler as a BaseSpecWorker: decode batches go
through ScheduleBatch -> SpecState -> step -> GenerationBatchResult. verify is
shared (every chain algorithm runs the same target TARGET_VERIFY forward);
draft, commit and extend are algorithm specific.

Decode state stays padded to the batch bucket across steps and is re-padded only
when the batch composition changes; only the scheduler-facing result is live bs.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.sampling.sampling_params import TOP_K_ALL
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.spec_info import SpecInput
from sglang.srt.speculative.spec_tp_sync import SpecTpSync, SpecTpSyncSite
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.recompilation_checker import (
    snapshot_recompilation_baseline,
)
from sglang_tpu.experimental import register_host_batch_padder
from sglang_tpu.experimental.sampling.sampling_batch_info import (
    HostPaddedSamplingInfo,
    is_host_padded,
)
from sglang_tpu.experimental.speculative.dspark.dspark_draft_worker import (
    TpuDsparkDraftWorker,
)
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_drafts_greedy,
    accept_drafts_sampled,
    build_verify_tokens,
    finalize_accept,
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
from sglang_tpu.experimental.speculative.spec_utils import (
    apply_verify_logits_adjustments,
    pad_rows,
)


class TpuSpecWorker(BaseSpecWorker):
    """Speculative decoding as draft -> verify -> commit.

    Every abstract method is per-algorithm but required: the maps, extend, and
    the batch.spec_info / verify ForwardBatch formats. verify, decode dispatch
    and result packing are shared.
    """

    def __init__(
        self,
        target_worker: TpModelWorker,
        draft_worker: TpuDsparkDraftWorker,
        num_drafts: int,
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        if parallel.enable_dp_attention:
            raise NotImplementedError("TpuSpecWorker does not support DP attention.")
        if parallel.pp_size > 1:
            raise NotImplementedError("TpuSpecWorker does not support PP.")
        self._target_worker = target_worker
        self._draft_worker = draft_worker
        self.num_drafts = num_drafts
        self.num_verify_rows = num_drafts + 1
        self.device = target_worker.device
        self._tp_sync = SpecTpSync(parallel.tp_group)
        self._bucket: BucketState | None = None
        register_host_batch_padder(self._pad_bs)

    def init_cuda_graphs(self) -> None:
        super().init_cuda_graphs()
        self._warmup_logical_bs_ops()
        self._warmup_extend_sampling()
        self._warmup_relay_and_results()
        snapshot_recompilation_baseline()

    def _warmup_logical_bs_ops(self) -> None:
        # torch_tpu compiles one program per shape even for a plain H2D copy.
        max_bs = min(
            self.target_worker.model_runner.max_running_requests,
            self._b_bucketer().max_val,
        )
        live_sizes = range(1, max_bs + 1)
        bucket_sizes = sorted({self._pad_bs(bs) for bs in live_sizes})
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
                # _bucket_state pads device-born bonus_tokens/seq_lens to the bucket.
                pad_rows(
                    torch.zeros((bs,), dtype=torch.int64, device=self.device),
                    self._pad_bs(bs),
                ).cpu()
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
        else:
            if draft_result.draft_logits is None:
                raise ValueError("Sampling acceptance requires draft_logits.")
            num_correct_drafts, bonus_tokens = accept_drafts_sampled(
                target_logits,
                draft_result.draft_tokens,
                draft_result.num_verify_drafts,
                draft_result.draft_logits.float(),
                bucket.temperatures,
                bucket.top_ks,
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_SAMPLE
        # Sampled acceptance draws per-rank randoms; ranks must commit rank 0's outcome.
        self._tp_sync.sync(site, num_correct_drafts)
        self._tp_sync.sync(site, bonus_tokens)
        accept = finalize_accept(
            draft_result.draft_tokens,
            num_correct_drafts,
            bonus_tokens,
            bucket.state.seq_lens,
            bucket.live_mask,
        )
        return logits_output, accept

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
        next_state = self.commit(batch, bucket.state, rows, logits_output, accept)
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
            result = self._idle_result()
        else:
            result = self._forward_decode(batch)
        if on_publish is not None:
            on_publish(result.new_seq_lens)
        return result

    def _forward_decode(self, batch: ScheduleBatch) -> GenerationBatchResult:
        bs = batch.seq_lens.shape[0]
        padded_bs = self._pad_bs(bs)
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

    def _bucket_state(
        self,
        batch: ScheduleBatch,
        padded_bs: int,
        verify_input: TpuSpecVerifyInput,
    ) -> BucketState:
        """The previous step's padded state, re-padded only after filter/merge/extend."""
        bucket = self._bucket
        # filter_batch/merge_batch rebind future_indices; identity pins the composition.
        if bucket is not None and verify_input.future_indices is bucket.key:
            return bucket
        state = self.get_spec_state(batch)
        req_pool_indices = verify_input.req_pool_indices
        assert req_pool_indices.dtype == batch.req_pool_indices.dtype
        sampling_info = batch.sampling_info
        assert is_host_padded(sampling_info)
        assert sampling_info.temperatures.shape[0] == padded_bs
        sampled = not sampling_info.is_all_greedy
        return BucketState(
            key=batch.req_pool_indices,
            state=SpecState(
                bonus_tokens=pad_rows(state.bonus_tokens, padded_bs),
                seq_lens=pad_rows(state.seq_lens.to(torch.int64), padded_bs),
                req_pool_indices=req_pool_indices,
            ),
            # The plan pads with the reserved pool row 0, which no live request holds.
            live_mask=req_pool_indices != 0,
            # SamplingBatchInfo already holds these padded to padded_bs on host.
            temperatures=(
                sampling_info.temperatures.view(-1).to(self.device, non_blocking=True)
                if sampled
                else None
            ),
            top_ks=(
                sampling_info.top_ks.to(self.device, non_blocking=True)
                if sampled
                else None
            ),
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
                for greedy, top_k in ((True, False), (False, False), (False, True)):
                    # Materialized like the compiled output, then the finalize trim.
                    logits = torch.zeros((cap, vocab_size), dtype=torch.float32).to(
                        self.device
                    )
                    runner.sampler(
                        LogitsProcessorOutput(next_token_logits=logits[:pbs]),
                        self._warmup_sampling_info(
                            pbs, vocab_size, greedy=greedy, top_k=top_k
                        ),
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
                # ModelRunner.sample positions, then the DSpark extend trim.
                (torch.zeros((bs,), dtype=torch.int64).to(self.device) - 1).cpu()
                torch.zeros((self._pad_bs(bs),), dtype=torch.int64, device=self.device)[
                    :bs
                ].cpu()

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
                padded_bs = self._pad_bs(bs)
                indices = torch.zeros((bs,), dtype=torch.int64).to(self.device)
                # Decode relays [:bs] views of bucket state; extend relays a
                # compact seq_lens upload.
                bucket_rows = torch.zeros(
                    (padded_bs,), dtype=torch.int64, device=self.device
                )[:bs]
                compact_rows = torch.zeros((bs,), dtype=torch.int64).to(self.device)
                for rows in (bucket_rows, compact_rows):
                    future_map.publish(indices, rows)
                    future_map.stash(
                        indices,
                        RelayPayload.from_draft_input(
                            self.state_to_spec_info(
                                SpecState(
                                    bonus_tokens=rows,
                                    seq_lens=rows,
                                    req_pool_indices=indices,
                                )
                            )
                        ),
                    )
                # resolve_seq_lens_cpu / _resolve_spec_extras gathers.
                future_map.new_seq_lens_buf[indices].cpu()
                future_map.output_tokens_buf[indices].cpu()
                # GenerationBatchResult.copy_to_cpu of _pack_decode_result's rows.
                torch.zeros(
                    (padded_bs, self.num_verify_rows),
                    dtype=torch.int64,
                    device=self.device,
                )[:bs].reshape(-1).cpu()
                torch.zeros((padded_bs,), dtype=torch.int32, device=self.device)[
                    :bs
                ].cpu()

    def _warmup_sampling_info(
        self, padded_bs: int, vocab_size: int, *, greedy: bool, top_k: bool
    ) -> SamplingBatchInfo:
        top_k_value = 1 if greedy else 2 if top_k else TOP_K_ALL
        return HostPaddedSamplingInfo(
            temperatures=torch.ones((padded_bs, 1)),
            top_ps=torch.ones((padded_bs,)),
            top_ks=torch.full((padded_bs,), top_k_value, dtype=torch.int32),
            min_ps=torch.zeros((padded_bs,)),
            is_all_greedy=greedy,
            is_any_greedy=greedy,
            need_top_p_sampling=False,
            need_top_k_sampling=top_k_value != TOP_K_ALL,
            need_min_p_sampling=False,
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
        padded_bs = bucket_state.seq_lens.shape[0]
        # A chain verify is a causal block, so DFlash's linear verify input serves
        # every algorithm; the TPU compile policy warms exactly this structure.
        spec_info = DFlashVerifyInput(
            draft_token=verify_tokens,
            positions=rows.flat_positions,
            draft_token_num=self.num_verify_rows,
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            live_seq_lens_cpu=batch.seq_lens_cpu,
        )
        live_fields = (
            batch.seq_lens,
            batch.seq_lens_cpu,
            batch.seq_lens_sum,
            batch.orig_seq_lens,
            batch.req_pool_indices,
            batch.req_pool_indices_cpu,
        )
        batch.out_cache_loc = rows.flat_cache_locs
        # init_new snapshots these; feed the bucket rows so the shim sees nothing
        # to pad. Padded rows hold pool row 0 and KV length 0, inactive in RPA.
        batch.seq_lens = bucket_state.seq_lens
        batch.req_pool_indices = bucket_state.req_pool_indices
        batch.req_pool_indices_cpu = verify_input.req_pool_indices_cpu
        batch.seq_lens_cpu, batch.seq_lens_sum = verify_input.padded_kv_lens_after_rows(
            self.num_verify_rows
        )
        # Ignored by direct compile on TPU; None keeps the shim from padding it per bs.
        batch.orig_seq_lens = None
        assert batch.req_pool_indices_cpu.shape[0] == padded_bs
        try:
            verify_forward_batch, _ = spec_info.prepare_for_verify(
                batch, self.target_worker
            )
        finally:
            (
                batch.seq_lens,
                batch.seq_lens_cpu,
                batch.seq_lens_sum,
                batch.orig_seq_lens,
                batch.req_pool_indices,
                batch.req_pool_indices_cpu,
            ) = live_fields
        return verify_forward_batch

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
        logits_output = target_out.logits_output
        # Custom logit processors address live rows only; penalties / logit bias are
        # rejected in _check_supported.
        apply_verify_logits_adjustments(
            next_token_logits=logits_output.next_token_logits,
            sampling_info=batch.sampling_info,
            num_verify_rows=self.num_verify_rows,
        )
        return logits_output

    def _pack_decode_result(
        self,
        *,
        bs: int,
        next_state: SpecState,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> GenerationBatchResult:
        # The scheduler relays and accounts per live request, so cut the bucket rows here.
        accept_lens = accept.accept_lens[:bs]
        new_seq_lens = next_state.seq_lens[:bs]
        logits_output.next_token_logits = logits_output.next_token_logits[
            : bs * self.num_verify_rows
        ]
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept.accept_tokens[:bs].reshape(-1),
            accept_lens=accept_lens,
            block_accept_lens=accept_lens,
            next_draft_input=self.state_to_spec_info(
                SpecState(
                    bonus_tokens=next_state.bonus_tokens[:bs],
                    seq_lens=new_seq_lens,
                    req_pool_indices=next_state.req_pool_indices[:bs],
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

    @staticmethod
    def _check_supported(
        *, batch: ScheduleBatch, pp_proxy_tensors: PPProxyTensors | None
    ) -> None:
        if pp_proxy_tensors is not None:
            raise NotImplementedError("TpuSpecWorker does not support PP.")
        if batch.has_grammar:
            raise NotImplementedError("TpuSpecWorker does not support grammar.")
        if batch.return_logprob:
            raise NotImplementedError("TpuSpecWorker does not support logprobs.")
        sampling_info = batch.sampling_info
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
        # Seeded sampling reads live-bs positions against bucket-padded seeds.
        if sampling_info.sampling_seed is not None:
            raise NotImplementedError(
                "TpuSpecWorker does not support deterministic sampling."
            )
        # top-k > 1 is not detectable without a device read; it is unsupported too.
        if not sampling_info.is_all_greedy and (
            sampling_info.need_top_p_sampling or sampling_info.need_min_p_sampling
        ):
            raise NotImplementedError(
                "TpuSpecWorker only supports temperature sampling."
            )

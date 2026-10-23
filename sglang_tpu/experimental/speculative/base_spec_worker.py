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
from torch import nn

from sglang.srt.compilation.compiled_op import warmup_compiled_op
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.environ import envs
from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.layers.dp_attention import get_dp_tp_group, is_dp_attention_enabled
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.overlap_utils import FutureMap, RelayPayload
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.memory_pool import KVCache, MHATokenToKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_memory,
    get_parallel,
    get_schedule,
    get_spec,
    mamba_track_grid,
)
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
from sglang.srt.utils.common import ceil_align, is_pin_memory_available
from sglang_tpu.compilation.bucketer import Bucketer
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy
from sglang_tpu.compilation.recompilation_checker import (
    snapshot_recompilation_baseline,
)
from sglang_tpu.experimental import (
    pd_decode_draft_prefill,
    register_host_batch_padder,
)
from sglang_tpu.experimental.sampling.sampling_batch_info import (
    HostPaddedSamplingInfo,
    is_host_padded,
)
from sglang_tpu.experimental.speculative.draft_forward import set_bucket_dp_metadata
from sglang_tpu.experimental.speculative.draft_info import HostPaddedDraftInput
from sglang_tpu.experimental.speculative.draft_worker import TpuDraftWorker
from sglang_tpu.experimental.speculative.pd_draft_prefill import (
    build_pd_hidden_pool,
    prebuilt_extend_groups,
    prebuilt_extend_view,
    register_pd_draft_prefiller,
)
from sglang_tpu.experimental.speculative.pd_hidden_pool import (
    PdHiddenPool,
    register_pd_hidden_pool,
)
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_drafts_greedy,
    accept_drafts_sampled,
    accept_drafts_sampled_filtered,
    accept_drafts_target_only,
    build_verify_tokens,
    finalize_accept,
    live_result_rows,
    pad_state_rows,
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
    map_carry,
)
from sglang_tpu.experimental.speculative.spec_utils import (
    overridden_fields,
    pad_rows,
    padded_h2d,
)
from sglang_tpu.mamba_pool import (
    TpuMambaPool,
    committed_read_offsets,
    verify_track_steps,
)
from sglang_tpu.overlap_utils_tpu import flush_direct_handoff_tpu
from sglang_tpu.sampler import TpuSampler

if TYPE_CHECKING:
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker

# apply_dflash_simulated_acceptance's fixed_token_id, so both paths emit the same tokens.
_SIMULATED_TOKEN = 100


# The target sampler compiles one filter program per need_* combination.
_SAMPLING_WARMUP_VARIANTS = [{"greedy": True}] + [
    {"greedy": False, "top_k": top_k, "top_p": top_p, "min_p": min_p}
    for top_k, top_p, min_p in itertools.product((False, True), repeat=3)
]


def _check_draft_lm_head_group(draft_model: nn.Module, *, attn_tp_size: int) -> None:
    """Reject a draft head sharded over another group than the draft runs in.

    Under DP attention the draft runs in the attention-TP group. A head built in
    the draft's scope, or a target head under --enable-dp-lm-head, shards over
    that group; a shared target head sharded over the full TP group would draft
    from one shard.
    """
    lm_head = draft_model.lm_head
    if (
        isinstance(lm_head, VocabParallelEmbedding)
        and lm_head.enable_tp
        and lm_head.tp_size != attn_tp_size
    ):
        raise ValueError(
            "Under DP attention the draft shares a target lm_head sharded over the "
            "full TP group; pass --enable-dp-lm-head to shard it over the "
            "attention-TP group the draft runs in."
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
        draft_worker: TpuDraftWorker | EagleDraftWorker,
        num_drafts: int,
    ) -> None:
        super().__init__()
        parallel = get_parallel()
        # A PD prefill only extends; the last stage hosts the draft (TpuPipelineStageWorker the rest).
        if parallel.pp_size > 1 and (
            get_disagg().disaggregation_mode != "prefill"
            or not parallel.pp_group.is_last_rank
        ):
            raise NotImplementedError(
                "TpuSpecWorker supports PP only on the last stage of a "
                "--disaggregation-mode prefill server."
            )
        self._target_worker = target_worker
        runner = target_worker.model_runner
        self._bsearch_filtering = envs.SGLANG_TPU_ENABLE_BSEARCH_SAMPLING.get()
        if not isinstance(runner.sampler, TpuSampler):
            # HOST_PADDED sampling params stay on host until TpuSampler uploads
            # them; with full logits it samples exactly like a plain Sampler
            # (--sampling-backend pytorch).
            runner.sampler = TpuSampler()
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
        # An idle DP rank mirrors its peers' draft syncs without knowing whether they
        # sample; greedy drafts sync one [pbs, gamma] int64 either way, and sampled
        # requests take the target-only accept, as TPU EAGLE does.
        self._sampled_drafts = not self._dp_attention
        if self._dp_attention and self._tp_sync.enabled(
            SpecTpSyncSite.SPEC_ACCEPT_GREEDY
        ) != self._tp_sync.enabled(SpecTpSyncSite.SPEC_ACCEPT_SAMPLE):
            # An idle DP rank cannot tell which accept its peers ran.
            raise ValueError(
                "SGLANG_SPEC_TP_SYNC must enable both or neither spec-accept site "
                "under DP attention."
            )
        if self._dp_attention:
            _check_draft_lm_head_group(
                draft_worker.model_runner.model, attn_tp_size=parallel.attn_tp_size
            )
        self._target_has_linear_attn = (
            mambaish_config(target_worker.model_runner.model_config) is not None
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
        self._simulated_bonus: dict[int, torch.Tensor] = {}
        # Per (batch bucket, forced num_correct_drafts).
        self._simulated_correct: dict[tuple[int, int], torch.Tensor] = {}
        register_host_batch_padder(self._pad_bs)
        # Built in init_cuda_graphs: the target KV pool it pages after does not exist yet.
        self._pd_hidden_pool: PdHiddenPool | None = None

    def _init_pd_hidden_pool(self) -> PdHiddenPool | None:
        """The PD hidden pool when the decode runs the draft prompt pass, else None."""
        disagg_mode = get_disagg().disaggregation_mode
        if not pd_decode_draft_prefill() or disagg_mode == "null":
            return None
        if self._dp_attention:
            # Prebuilt batches land per DP rank; the draft pass would need the idle mirror.
            raise NotImplementedError(
                "SGLANG_TPU_PD_DECODE_DRAFT_PREFILL does not support DP attention."
            )
        pool = build_pd_hidden_pool(self.target_worker.model_runner)
        register_pd_hidden_pool(pool)
        if disagg_mode == "decode":
            register_pd_draft_prefiller(self.prefill_draft_from_pd)
        return pool

    @property
    def ships_pd_hidden(self) -> bool:
        """Whether this PD prefill hands the draft's hidden rows to the decode."""
        return (
            self._pd_hidden_pool is not None
            and get_disagg().disaggregation_mode == "prefill"
        )

    def write_pd_hidden(
        self, batch: ScheduleBatch, target_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Store an extend's target hidden rows for the decode; returns the kv_fence_tail."""
        assert self._pd_hidden_pool is not None
        cache_loc_cpu = batch.out_cache_loc
        assert cache_loc_cpu.device.type == "cpu", cache_loc_cpu.device
        segments = self._pd_hidden_pool.write_segments(
            cache_loc_cpu, target_hidden.shape[0]
        )
        return self._pd_hidden_pool.write(
            target_hidden, segments=segments.to(self.device, non_blocking=True)
        )

    def prefill_draft_from_pd(self, batch: ScheduleBatch) -> None:
        """Draft prompt pass of a PD decode's prebuilt batch, from the shipped hidden rows."""
        assert self._pd_hidden_pool is not None
        compile_policy = self._compile_policy()
        t_bucketer = compile_policy.t_bucketer
        assert t_bucketer is not None
        cap = compile_policy._extend_compile_cap()
        vocab_size = self.target_worker.model_runner.model_config.vocab_size
        for group in prebuilt_extend_groups(
            batch, max_tokens=t_bucketer.max_val, max_reqs=cap
        ):
            view = prebuilt_extend_view(batch, group, vocab_size=vocab_size)
            # The draft graphs and writer were warmed with the allocator's dtype.
            view.out_cache_loc = view.out_cache_loc.to(
                compile_policy.live_cache_loc_dtype()
            )
            padded_tokens = t_bucketer.next_bucket(max(view.extend_num_tokens, cap))
            cache_loc = pad_rows(view.out_cache_loc, padded_tokens).to(
                self.device, non_blocking=True
            )
            bonus_tokens = padded_h2d(
                torch.tensor(
                    [req.output_ids[-1] for req in view.reqs], dtype=torch.int64
                ),
                self._pad_bs(len(view.reqs)),
                self.device,
            )
            self.extend_draft_from_hidden(
                view,
                target_hidden=self._pd_hidden_pool.read(cache_loc),
                cache_loc=cache_loc,
                bonus_tokens=bonus_tokens,
            )

    def extend_draft_from_hidden(
        self,
        batch: ScheduleBatch,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        bonus_tokens: torch.Tensor,
    ) -> None:
        """The draft's prompt pass over an extend batch's target hidden [token bucket, F]."""
        raise NotImplementedError(
            f"{type(self).__name__} does not run the PD decode draft prompt pass."
        )

    def _warmup_pd_hidden(self) -> None:
        """The prefill's hidden write or the decode's read, per extend token bucket."""
        pool = self._pd_hidden_pool
        if pool is None:
            return
        compile_policy = self._compile_policy()
        assert compile_policy.t_bucketer is not None
        cap = compile_policy._extend_compile_cap()
        with torch.no_grad():
            for tokens in compile_policy.t_bucketer.buckets:
                if tokens < cap:
                    continue
                if self.ships_pd_hidden:
                    segments = pool.write_segments(
                        torch.zeros((0,), dtype=torch.int64), tokens
                    )
                    pool.write(
                        torch.zeros(
                            (tokens, pool.feature_size),
                            dtype=pool.buffer.dtype,
                            device=self.device,
                        ),
                        segments=segments.to(self.device),
                    ).cpu()
                else:
                    cache_loc = torch.zeros(
                        (tokens,), dtype=compile_policy.live_cache_loc_dtype()
                    ).to(self.device)
                    pool.read(cache_loc).cpu()
            for pbs in self._b_bucketer().buckets:
                padded_h2d(torch.zeros((0,), dtype=torch.int64), pbs, self.device)

    def init_cuda_graphs(self) -> None:
        super().init_cuda_graphs()
        # Before the Raiden manager, which registers it, and before its warmup.
        self._pd_hidden_pool = self._init_pd_hidden_pool()
        self._all_drafts = {
            padded_bs: torch.full(
                (padded_bs,), self.num_drafts, dtype=torch.int32, device=self.device
            )
            for padded_bs in self._b_bucketer().buckets
        }
        self._warmup_logical_bs_ops()
        self._warmup_extend_sampling()
        self._warmup_relay_and_results()
        self._warmup_pd_hidden()
        if not self._sampled_drafts and get_disagg().disaggregation_mode != "prefill":
            self._warmup_target_only_accept()
        # A PD prefill server never verifies, so its mamba pool has no checkpoints.
        if (
            self._target_has_linear_attn
            and get_disagg().disaggregation_mode != "prefill"
        ):
            self._warmup_linear_attn_commit()
        if self._simulate_acc_len > 0:
            # Uploaded once, so each round only picks one (values stay data,
            # never baked into a program).
            buckets = self._b_bucketer().buckets
            self._simulated_drafts = {
                pbs: torch.full(
                    (pbs, self.num_drafts), _SIMULATED_TOKEN, dtype=torch.int64
                ).to(self.device)
                for pbs in buckets
            }
            self._simulated_bonus = {
                pbs: torch.full((pbs,), _SIMULATED_TOKEN, dtype=torch.int64).to(
                    self.device
                )
                for pbs in buckets
            }
            self._simulated_correct = {
                (pbs, n): torch.full((pbs,), n, dtype=torch.int64).to(self.device)
                for pbs in buckets
                for n in range(self.num_verify_rows)
            }
        self._snapshot_warm_baseline()

    @staticmethod
    def _snapshot_warm_baseline() -> None:
        # torch_tpu compiles asynchronously: wait for the warmups' compiles to land
        # in the cache, or the checker counts them as serving-time compiles.
        torch.accelerator.synchronize()
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
                # Eager ops compile per input layout, so bump finalize_accept's output.
                next_seq_lens = self._warmup_accept(pbs).next_seq_lens
                (next_seq_lens + self.num_verify_rows).to(torch.int32).cpu()

    def _warmup_accept(self, padded_bs: int) -> AcceptResult:
        """finalize_accept outputs at padded_bs, the inputs eager consumers see while serving."""

        def zeros(*shape: int) -> torch.Tensor:
            return torch.zeros(shape, dtype=torch.int64, device=self.device)

        return finalize_accept(
            zeros(padded_bs, self.num_drafts),
            zeros(padded_bs),
            zeros(padded_bs),
            zeros(padded_bs),
            torch.ones((padded_bs,), dtype=torch.bool, device=self.device),
        )

    @property
    def retraction_draft_kv_pools(self) -> tuple[KVCache, ...]:
        # Core backs up only EAGLE draft pools; without these, a request resumed
        # on a PD decode server drafts over whatever KV its new slots hold.
        pools = tuple(
            runner.token_to_kv_pool for runner in self._draft_worker.draft_runners
        )
        disagg = get_disagg()
        if (
            disagg.disaggregation_mode == "decode"
            and disagg.disaggregation_decode_retraction_backup == "cpu_tensor"
        ):
            unsupported = [
                type(pool).__name__
                for pool in pools
                if not isinstance(pool, (MHATokenToKVPool, SWAKVPool))
            ]
            if unsupported:
                raise NotImplementedError(
                    f"TpuSpecWorker cannot back up draft KV pools {unsupported} "
                    "to CPU on decode retraction; pass "
                    "--disaggregation-decode-retraction-backup none to abort "
                    "retracted requests instead."
                )
        return pools

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
    ) -> CommitResult:
        """Commit the accepted tokens: advance the prefix, update the draft context.

        Why: the next draft must condition on the committed tokens' target
        features (DFlash/DSpark inject them into draft KV, EAGLE3 carries them).
        Noop: the draft-context update is a noop for a draft that never reads
        target features; the advanced SpecState must still be returned, with an
        output of the step's last KV write as kv_fence_tail.
        """

    @abstractmethod
    def extend(
        self, batch: ScheduleBatch, *, pp_proxy_tensors: PPProxyTensors | None
    ) -> GenerationBatchResult:
        """Extend: target forward on new prompt tokens, seed the draft context, set next_draft_input.

        next_draft_input.kv_fence_tail must be an output of the last KV write.
        pp_proxy_tensors carries the previous stage's activations on the last PP stage.

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
                target_logits,
                draft_result.draft_tokens,
                draft_result.num_verify_drafts,
                bucket.temperatures,
                bucket.top_ks,
                bucket.top_ps,
                bucket.min_ps,
                self._accept_threshold_single,
                self._accept_threshold_acc,
            )
            site = SpecTpSyncSite.SPEC_ACCEPT_SAMPLE
        elif bucket.need_filters:
            num_correct_drafts, bonus_tokens = accept_drafts_sampled_filtered(
                target_logits,
                draft_result.draft_tokens,
                draft_result.num_verify_drafts,
                draft_result.draft_logits.float(),
                bucket.temperatures,
                bucket.top_ks,
                bucket.top_ps,
                bucket.min_ps,
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
        return (
            self._simulated_correct[(padded_bs, commit_len - 1)],
            self._simulated_bonus[padded_bs],
            self._simulated_drafts[padded_bs],
        )

    def step(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> tuple[CommitResult, LogitsProcessorOutput, AcceptResult]:
        draft_result = self.draft(batch, bucket, rows, verify_input)
        logits_output, accept = self.verify(
            batch, bucket, rows, draft_result, verify_input
        )
        if self._target_has_linear_attn:
            self._commit_linear_attn_states(
                batch=batch, accept=accept, verify_input=verify_input
            )
        committed = self.commit(
            batch, bucket.state, rows, verify_input, logits_output, accept
        )
        return committed, logits_output, accept

    def _commit_linear_attn_states(
        self,
        *,
        batch: ScheduleBatch,
        accept: AcceptResult,
        verify_input: TpuSpecVerifyInput,
    ) -> None:
        """Select each slot's verify checkpoint at its last accepted token, and
        the checkpoints radix tracking snapshots."""
        runner = self.target_worker.model_runner
        attn_backend = runner.attn_backend
        assert isinstance(attn_backend, HybridLinearAttnBackend), type(attn_backend)
        steps_to_track = None
        if batch.mamba_track_indices is not None:
            steps_to_track = verify_track_steps(
                verify_input.committed_lens,
                accept.last_steps,
                mamba_track_grid(batch.tree_cache.page_size),
            )
        # Slots come from the verify forward's bucket metadata; padded rows accept 0,
        # so their -1 step leaves the slot untouched.
        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=accept.last_steps,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=steps_to_track,
            model=runner.model,
        )

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
            result = self.extend(batch, pp_proxy_tensors=pp_proxy_tensors)
        elif batch.forward_mode.is_idle():
            if self._dp_attention and max(batch.global_num_tokens or (0,)) > 0:
                # DP peers decode: enter their step's collectives too.
                self._forward_idle_decode(batch)
            result = self._idle_result()
        else:
            result = self._forward_decode(batch)
        next_draft_input = result.next_draft_input
        assert isinstance(next_draft_input, HostPaddedDraftInput)
        # Rows with requests write KV; idle rows write only the reserved slot 0.
        assert not batch.reqs or next_draft_input.kv_fence_tail is not None
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
        committed, logits_output, accept = self.step(batch, bucket, rows, verify_input)
        next_state = committed.state
        # The scheduler stores this key as the next spec_info.future_indices.
        self._bucket = bucket._replace(key=batch.req_pool_indices, state=next_state)
        # Hidden rows stay on device; the overlap result copy would ship them to CPU.
        logits_output.hidden_states = None
        result = self._pack_decode_result(
            bs=bs, next_state=next_state, logits_output=logits_output, accept=accept
        )
        next_draft_input = result.next_draft_input
        assert isinstance(next_draft_input, HostPaddedDraftInput)
        next_draft_input.kv_fence_tail = committed.kv_fence_tail
        return result

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
            **(_sampled_row_filters(sampling_info) if sampled else {}),
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
        """The active step's forwards and syncs in order on idle rows: draft, verify, commit."""
        # Padded rows have req_pool_indices 0; no temperatures means the greedy draft.
        live_mask = state.req_pool_indices != 0
        idle_bucket = BucketState(
            key=state.req_pool_indices,
            state=state,
            live_mask=live_mask,
            temperatures=None,
            top_ks=None,
        )
        draft_result = self.draft(batch, idle_bucket, rows, verify_input)
        verify_tokens = draft_result.verify_tokens
        if verify_tokens is None:
            verify_tokens = build_verify_tokens(
                state.bonus_tokens, draft_result.draft_tokens
            )
        logits_output = self._forward_idle_verify(
            batch=batch,
            state=state,
            rows=rows,
            verify_input=verify_input,
            verify_tokens=verify_tokens,
        )
        accept = finalize_accept(
            draft_result.draft_tokens,
            self._device_zeros(state.seq_lens.shape[0]),
            state.bonus_tokens,
            state.seq_lens,
            live_mask,
        )
        self.commit(batch, state, rows, verify_input, logits_output, accept)

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

    def sync_target_tokens(self, next_token_ids: torch.Tensor) -> torch.Tensor:
        """extend()'s sync of the target's sampled tokens."""
        if not self._dp_attention:
            return self._tp_sync.sync(SpecTpSyncSite.SPEC_TARGET, next_token_ids)
        # Every DP rank must issue the collective at one shape; an idle rank has no rows.
        cap = self._compile_policy()._extend_compile_cap()
        rows = next_token_ids.shape[0]
        padded = torch.nn.functional.pad(next_token_ids, (0, cap - rows))
        return self._tp_sync.sync(SpecTpSyncSite.SPEC_TARGET, padded)[:rows]

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
        # A multimodal wrapper's eager outer logits processor emits live rows,
        # which TpuSampler pads to the batch bucket (whatever the input modality).
        eager_logits = not self._compile_policy()._compiles_logits_processor()
        with torch.no_grad():
            for pbs in sorted({self._pad_bs(bs) for bs in extend_sizes}):
                for variant in _SAMPLING_WARMUP_VARIANTS:
                    # A target that keeps extend outputs padded samples all cap rows,
                    # materializing the pbs-row params to them.
                    variants = {(pbs, False), (cap, True)}
                    if pbs == 1:
                        # Unpadded [1, 1] params broadcast over the cap rows as-is.
                        variants.add((cap, False))
                    for rows, params_on_cpu in variants:
                        next_token_ids = runner.sampler(
                            LogitsProcessorOutput(
                                next_token_logits=self._warmup_extend_logits(
                                    cap, rows, vocab_size
                                )
                            ),
                            self._warmup_sampling_info(
                                pbs, vocab_size, params_on_cpu=params_on_cpu, **variant
                            ),
                            False,
                            None,
                            None,
                            torch.zeros((rows,), dtype=torch.int64, device=self.device),
                        )
                        # extend() broadcasts these tokens; the collective compiles per shape.
                        self.sync_target_tokens(next_token_ids).cpu()
            if self._dp_attention:
                # A DP rank without prompts syncs no rows.
                self.sync_target_tokens(
                    torch.zeros((0,), dtype=torch.int64, device=self.device)
                ).cpu()
            for bs in extend_sizes:
                # The shim pads the device-resident extend bs fields (seq_lens,
                # req_pool_indices int64; orig_seq_lens int32) to the cap.
                for dtype in (torch.int64, torch.int32):
                    torch.nn.functional.pad(
                        torch.zeros((bs,), dtype=dtype).to(self.device), (0, cap - bs)
                    ).cpu()
                # ModelRunner.sample positions, then the block-draft extend trim; a
                # target that keeps extend outputs padded samples at the cap rows.
                (torch.zeros((bs,), dtype=torch.int64).to(self.device) - 1).cpu()
                for rows in {self._pad_bs(bs), cap}:
                    torch.zeros((rows,), dtype=torch.int64, device=self.device)[
                        :bs
                    ].cpu()
                if eager_logits and self._pad_bs(bs) != bs:
                    torch.nn.functional.pad(
                        torch.zeros((bs, vocab_size), dtype=torch.float32).to(
                            self.device
                        ),
                        (0, 0, 0, self._pad_bs(bs) - bs),
                    ).cpu()

    def _warmup_extend_logits(
        self, cap: int, rows: int, vocab_size: int
    ) -> torch.Tensor:
        """Fresh logits materialized like the compiled cap output, trimmed to rows."""
        # Held by the caller alone, as in serving: the sampler then donates the
        # buffer to its in-place ops, and donation changes the compiled program.
        logits = torch.zeros((cap, vocab_size), dtype=torch.float32).to(self.device)
        return logits if rows == cap else logits[:rows]

    def _sampling_param_warmup_args(
        self, padded_bs: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """[pbs] temperatures, top_ks, top_ps, min_ps typed as _bucket_state uploads them.

        Distinct tensors: serving uploads each param separately, and Dynamo
        guards on aliased inputs.
        """
        temperatures, top_ps, min_ps = (
            torch.ones((padded_bs,), dtype=torch.float32, device=self.device)
            for _ in range(3)
        )
        top_ks = torch.ones((padded_bs,), dtype=torch.int32, device=self.device)
        return temperatures, top_ks, top_ps, min_ps

    def _warmup_sampled_accept(
        self, padded_bs: int, draft_tokens: torch.Tensor, draft_logits: torch.Tensor
    ) -> None:
        """verify's sampled acceptance at padded_bs, with and without target filters."""
        target_logits = torch.zeros(
            (padded_bs * self.num_verify_rows, draft_logits.shape[-1]),
            dtype=torch.float32,
            device=self.device,
        )
        temperatures, top_ks, top_ps, min_ps = self._sampling_param_warmup_args(
            padded_bs
        )
        args = (
            target_logits,
            draft_tokens,
            self._all_drafts[padded_bs],
            draft_logits.float(),
            temperatures,
            top_ks,
        )
        warmup_compiled_op(accept_drafts_sampled, *args)
        warmup_compiled_op(accept_drafts_sampled_filtered, *args, top_ps, min_ps)

    def _warmup_relay_and_results(self) -> None:
        """FutureMap relay and result D2H run per live bs outside any compiled op."""
        runner = self.target_worker.model_runner
        max_bs = min(runner.max_running_requests, self._b_bucketer().max_val)
        cap = self._compile_policy()._extend_compile_cap()
        # A scratch map runs the production publish/stash; live buffers stay untouched.
        future_map = FutureMap(
            self.device, runner.spec_algorithm, runner.req_to_token_pool
        )
        with torch.no_grad():
            for bs in range(1, max_bs + 1):
                indices = torch.zeros((bs,), dtype=torch.int64).to(self.device)
                # Decode relays [:bs] views of finalize_accept's outputs at each bucket
                # the rank can run; extend relays a compact seq_lens upload.
                relays = []
                for padded_bs in self._decode_buckets_for(bs):
                    accept = self._warmup_accept(padded_bs)
                    relays.append(
                        (
                            accept.bonus_tokens[:bs],
                            accept.next_seq_lens[:bs],
                            map_carry(
                                self.make_dummy_carry(padded_bs), lambda t: t[:bs]
                            ),
                        )
                    )
                compact_rows = torch.zeros((bs,), dtype=torch.int64).to(self.device)
                # Extend stashes its [:bs] tokens sampled at the cap rows.
                cap_tokens = torch.zeros((cap,), dtype=torch.int64, device=self.device)
                for bonus_tokens in (compact_rows, cap_tokens[:bs]):
                    relays.append(
                        (bonus_tokens, compact_rows, self.make_dummy_carry(bs))
                    )
                for bonus_tokens, seq_lens, carry in relays:
                    future_map.publish(indices, seq_lens)
                    future_map.stash(
                        indices,
                        RelayPayload.from_draft_input(
                            self.state_to_spec_info(
                                SpecState(
                                    bonus_tokens=bonus_tokens,
                                    seq_lens=seq_lens,
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
                    if padded_bs != bs:
                        # _pack_decode_result's rows.
                        warmup_compiled_op(
                            live_result_rows,
                            torch.zeros(
                                (padded_bs, self.num_verify_rows),
                                dtype=torch.int64,
                                device=self.device,
                            ),
                            torch.zeros(
                                (padded_bs,), dtype=torch.int32, device=self.device
                            ),
                            bs,
                        )
            if self._dp_attention:
                for padded_bs in self._b_bucketer().buckets:
                    # _forward_idle_step's all-padding drafts.
                    torch.zeros(
                        (padded_bs, self.num_drafts),
                        dtype=torch.int64,
                        device=self.device,
                    ).cpu()

    def _warmup_linear_attn_commit(self) -> None:
        """_commit_linear_attn_states' eager ops at every batch bucket."""
        mamba_pool = self.target_worker.model_runner.req_to_token_pool.mamba_pool
        assert isinstance(mamba_pool, TpuMambaPool), type(mamba_pool)
        # Radix tracking snapshots boundary checkpoints after each verify.
        track_grid = (
            mamba_track_grid(get_schedule().page_size)
            if get_exec().mamba.enable_mamba_extra_buffer
            and not get_memory().disable_radix_cache
            else None
        )
        with torch.no_grad():
            for pbs in self._b_bucketer().buckets:
                # int32 like the GDN metadata slots; the null slot 0 is never written.
                slots = torch.zeros((pbs,), dtype=torch.int32, device=self.device)
                accept_lens = torch.zeros((pbs,), dtype=torch.int32, device=self.device)
                slots, steps = slots.clamp(min=0), accept_lens.to(torch.int64) - 1
                warmup_compiled_op(
                    committed_read_offsets,
                    mamba_pool.checkpoint_read_offsets,
                    slots,
                    steps,
                )
                mamba_pool.select_checkpoints(slots, steps)
                if track_grid is not None:
                    mamba_pool.warmup_snapshot_checkpoints(pbs, track_grid)

    def _warmup_sampling_info(
        self,
        padded_bs: int,
        vocab_size: int,
        *,
        greedy: bool,
        top_k: bool = False,
        top_p: bool = False,
        min_p: bool = False,
        params_on_cpu: bool = False,
    ) -> SamplingBatchInfo:
        top_k_value = 1 if greedy else 2 if top_k else TOP_K_ALL
        # Pinned like HostPaddedSamplingInfo's rows: a pageable H2D is its own
        # program and would fuse into the sampler's first op.
        pin = is_pin_memory_available(self.device)
        return HostPaddedSamplingInfo(
            temperatures=torch.ones((padded_bs, 1), pin_memory=pin),
            top_ps=torch.full((padded_bs,), 0.9 if top_p else 1.0, pin_memory=pin),
            top_ks=torch.full(
                (padded_bs,), top_k_value, dtype=torch.int32, pin_memory=pin
            ),
            min_ps=torch.full((padded_bs,), 0.1 if min_p else 0.0, pin_memory=pin),
            is_all_greedy=greedy,
            is_any_greedy=greedy,
            need_top_p_sampling=top_p,
            need_top_k_sampling=top_k_value != TOP_K_ALL,
            need_min_p_sampling=min_p,
            vocab_size=vocab_size,
            device=self.device,
            host_pad_bs=self._pad_bs,
            real_batch_size=padded_bs,
            # Serving's flag survives on_device(), so the sampler pads the params to the logits rows.
            sampling_params_on_cpu=params_on_cpu,
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
        num_rows = batch.input_ids.shape[0]
        set_bucket_dp_metadata(
            forward_batch,
            runner=self.target_worker.model_runner,
            num_tokens=num_rows,
            bs=num_rows // self.num_verify_rows,
            mode=ForwardMode.TARGET_VERIFY,
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

    @staticmethod
    def _live_result_rows(
        accept: AcceptResult, bs: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat [bs * (N + 1)] accept tokens and [bs] accept lengths for the scheduler."""
        if accept.accept_lens.shape[0] == bs:
            return accept.accept_tokens.reshape(-1), accept.accept_lens
        return live_result_rows(accept.accept_tokens, accept.accept_lens, bs)

    def _pack_decode_result(
        self,
        *,
        bs: int,
        next_state: SpecState,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> GenerationBatchResult:
        # The scheduler relays and accounts per live request, so cut the bucket rows here.
        next_token_ids, accept_lens = self._live_result_rows(accept, bs)
        new_seq_lens = next_state.seq_lens[:bs]
        logits_output.next_token_logits = logits_output.next_token_logits[
            : bs * self.num_verify_rows
        ]
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
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

    def _check_supported(
        self, *, batch: ScheduleBatch, pp_proxy_tensors: PPProxyTensors | None
    ) -> None:
        if pp_proxy_tensors is not None and not (
            batch.forward_mode.is_extend() or batch.is_extend_in_batch
        ):
            raise NotImplementedError("TpuSpecWorker under PP only extends.")
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
        if not self._bsearch_filtering and any(
            _sampled_row_filters(sampling_info).values()
        ):
            raise NotImplementedError(
                "TpuSpecWorker needs SGLANG_TPU_ENABLE_BSEARCH_SAMPLING for "
                "top-k/top-p/min-p sampling."
            )
        # Processors expect len(sampling_info) live rows, not bucket rows.
        if sampling_info.has_custom_logit_processor:
            raise NotImplementedError(
                "TpuSpecWorker does not support custom logit processors."
            )
        # Seeded sampling reads live-bs positions against bucket-padded seeds.
        if sampling_info.sampling_seed is not None:
            raise NotImplementedError(
                "TpuSpecWorker does not support deterministic sampling."
            )


def _sampled_row_filters(sampling_info: SamplingBatchInfo) -> dict[str, bool]:
    """Which filters a sampled (top_k > 1) row restricts, from the host params.

    Core's need_*_sampling also counts greedy rows (top_k = 1); those take the
    greedy rule, so they must not switch the whole batch onto the filter.
    """
    top_ks, top_ps, min_ps = (
        sampling_info.top_ks,
        sampling_info.top_ps,
        sampling_info.min_ps,
    )
    sampled = top_ks > 1
    return {
        "need_top_k": bool((sampled & (top_ks < sampling_info.vocab_size)).any()),
        "need_top_p": bool((sampled & (top_ps < 1.0)).any()),
        "need_min_p": bool((sampled & (min_ps > 0.0)).any()),
    }

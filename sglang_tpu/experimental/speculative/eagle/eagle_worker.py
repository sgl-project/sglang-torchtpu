"""EAGLE3 (topk 1) on the TpuSpecWorker interface.

draft:  the seed is the draft output at the last committed row: the first draft
        d_1 and the draft hidden h_1 that predicted it. The fused EAGLE draft
        loop runs N - 1 draft decode steps d_{j+1} = argmax draft(h_j, d_j),
        writing draft KV at n .. n + N - 2. Drafts are greedy, so q is one-hot.
verify: shared TpuSpecWorker.verify over [r, d_1, ..., d_N]; sampled requests
        take the filtered target-only rule, as TPU EAGLE's verify does.
commit: draft extend over the N + 1 verify rows with the target's EAGLE3 aux
        features, refreshing draft KV at n .. n + N; the next seed is its output
        at the last accepted row; root := bonus.
extend: target prefill (FULL capture), then the draft prefill over the prompt
        shifted by one token, whose last row is the sampled token, seeds the
        first chain.

EagleDraftWorker owns the draft model, its attention backends, the fused draft
loop and the draft warmups. Its preparation code builds the draft batches here,
on a batch whose rows are already at the batch bucket, so they match its
exact-bucket warmup structures.

Multimodal targets: the target prefill keeps its image embeddings at the token
bucket, and the draft prefill reads them shifted by one row; verify computes the
target's M-RoPE positions on the bucket rows. The draft itself is text-only.

Not ported: speculative_eagle_topk > 1, speculative_num_steps < 2, adaptive
steps, multi-layer EAGLE, M-RoPE drafts, hybrid (mamba) targets.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import NamedTuple

import torch

from sglang.srt.compilation.compiled_op import compiled_op, warmup_compiled_op
from sglang.srt.configs.hybrid_arch import mambaish_config
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.runtime_context import get_spec, process_model_config
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import get_draft_recurrent_hidden_state_spec
from sglang.srt.speculative.eagle_worker_common import prepare_for_draft
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils.common import is_pin_memory_available
from sglang_tpu.eagle_utils_tpu import fused_draft_extend_select_argmax_func
from sglang_tpu.experimental.speculative.base_spec_worker import TpuSpecWorker
from sglang_tpu.experimental.speculative.eagle_info import (
    HostPaddedEagleDraftInput,
    eagle_verify_input,
)
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_drafts_target_only,
    finalize_accept,
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
from sglang_tpu.experimental.speculative.spec_utils import (
    overridden_fields,
    pad_rows,
)


class EagleSeed(NamedTuple):
    """TpuEagle3Worker's draft_carry: the seed of the next draft chain, [pbs] rows.

    Attributes:
      topk_p: [pbs, 1] float32 ones; the topk 1 chain scores FutureMap relays.
      topk_index: [pbs, 1] int64 first draft token, in the draft vocab.
      hidden_states: [pbs, H] draft hidden state that predicted it.
    """

    topk_p: torch.Tensor
    topk_index: torch.Tensor
    hidden_states: torch.Tensor


@compiled_op()
def split_draft_chain(chain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[pbs, N + 1] chain [r, d_1, ..., d_N] -> drafts [pbs, N] and flat verify rows."""
    return chain[:, 1:].contiguous(), chain.reshape(-1)


@compiled_op()
def draft_extend_inputs(
    accept_tokens: torch.Tensor, seq_lens: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat [pbs * (N + 1)] draft-extend tokens and the post-write lengths [pbs].

    Rows past the accepted prefix hold -1; they only write draft KV past the new
    length, which later steps overwrite before reading, so any valid id serves.
    """
    return accept_tokens.clamp_min(0).reshape(-1), seq_lens + accept_tokens.shape[1]


@compiled_op()
def fill_prefill_tail(
    input_ids: torch.Tensor, tail_src: torch.Tensor, tail_tokens: torch.Tensor
) -> torch.Tensor:
    """Draft prefill tokens [T]: row p takes tail_tokens[tail_src[p]] where tail_src >= 0."""
    picked = tail_tokens.to(input_ids.dtype).index_select(0, tail_src.clamp_min(0))
    return torch.where(tail_src >= 0, picked, input_ids)


@compiled_op()
def shift_rows_up(rows: torch.Tensor) -> torch.Tensor:
    """[T, ...] rows moved up by one, row p holding row p + 1; the last row repeats."""
    return torch.cat((rows[1:], rows[-1:]), dim=0)


@compiled_op()
def select_first_draft(
    next_token_logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy first draft [pbs, 1] and its unit chain score, as topk_p, topk_index."""
    topk_index = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    return torch.ones_like(topk_index, dtype=torch.float32), topk_index


class TpuEagle3Worker(TpuSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        _check_eagle3_config(target_worker)
        draft_worker = EagleDraftWorker(server_args, gpu_id, nccl_port, target_worker)
        super().__init__(target_worker, draft_worker, get_spec().speculative_num_steps)
        self._eagle = draft_worker
        self._draft_runner = draft_worker.draft_runner
        # The fused draft loop advances 1-D positions only.
        if self._draft_runner.model_config.model_is_mrope:
            raise NotImplementedError("TpuEagle3Worker does not support M-RoPE drafts.")
        self._is_multimodal = process_model_config().is_multimodal
        self._hidden_size, self._hidden_dtype = get_draft_recurrent_hidden_state_spec(
            self._draft_runner
        )

    @property
    def last_shared_read_runner(self):
        # Draft extend, the step's last phase, reads the shared buffers.
        return self._draft_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        return (
            self.target_worker.model_runner.attn_backend,
            self._eagle.draft_attn_backend,
            self._eagle.draft_extend_attn_backend or self._draft_runner.attn_backend,
        )

    def init_cuda_graphs(self) -> None:
        super().init_cuda_graphs()
        self._warmup_draft_glue()
        self._warmup_target_only_accept()
        self._warmup_prefill_glue()
        # The base snapshot ran before these EAGLE-only warmups.
        self._snapshot_warm_baseline()

    def draft(
        self,
        batch: ScheduleBatch,
        bucket: BucketState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> DraftResult:
        draft_tokens, verify_tokens = self._draft_chain(
            batch=batch, state=bucket.state, rows=rows, verify_input=verify_input
        )
        return DraftResult(
            draft_tokens=draft_tokens,
            draft_logits=None,
            num_verify_drafts=self._all_drafts[bucket.state.seq_lens.shape[0]],
            verify_tokens=verify_tokens,
        )

    def _draft_chain(
        self,
        *,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The fused draft loop; the chain's drafts [pbs, N] and verify tokens [pbs * (N + 1)]."""
        seed = state.draft_carry
        draft_input = EagleDraftInput(
            topk_p=seed.topk_p,
            topk_index=seed.topk_index,
            hidden_states=seed.hidden_states,
            bonus_tokens=state.bonus_tokens,
            capture_hidden_mode=CaptureHiddenMode.LAST,
            # prepare_for_draft writes the chain's draft KV into the verify rows.
            device_accept_selected_cache_locs=rows.cache_locs,
        )
        with self._draft_context():
            with self._bucket_view(
                batch,
                bucket_state=state,
                verify_input=verify_input,
                kv_rows=0,
                spec_info=draft_input,
                input_ids=None,
                out_cache_loc=None,
            ) as view:
                forward_batch, _ = prepare_for_draft(
                    draft_input=draft_input,
                    req_to_token_pool=self.req_to_token_pool,
                    batch=view,
                    cuda_graph_runner=None,
                    draft_model_runner=self._draft_runner,
                    topk=1,
                    num_steps=self.num_drafts,
                )
            self._eagle._prepare_draft_forward_batch(
                forward_batch, refresh_dp_metadata=True
            )
            with torch.no_grad():
                tree = self._eagle._fused_draft_forward(forward_batch)
        # build_topk1_verify_chain_torch returns the chain [r, d_1, ..., d_N] last.
        return split_draft_chain(tree[-1])

    def commit(
        self,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
        logits_output: LogitsProcessorOutput,
        accept: AcceptResult,
    ) -> SpecState:
        target_hidden = logits_output.hidden_states
        assert target_hidden is not None
        assert target_hidden.shape[0] == rows.cache_locs.numel(), (
            "verify hidden must keep the batch bucket rows"
        )
        seed = self._forward_draft_extend(
            batch=batch,
            state=state,
            rows=rows,
            verify_input=verify_input,
            target_hidden=target_hidden,
            accept=accept,
        )
        # Verify rows past the new seq_lens become dead KV; freeing them is the allocator's job.
        return SpecState(
            bonus_tokens=accept.bonus_tokens,
            seq_lens=accept.next_seq_lens,
            req_pool_indices=state.req_pool_indices,
            draft_carry=seed,
        )

    def extend(self, batch: ScheduleBatch) -> GenerationBatchResult:
        result = self.target_worker.forward_batch_generation(
            batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        if not batch.reqs:
            # A DP-attention rank without prompts (DP padding relabels its batch)
            # still issues the draft prefill's collectives.
            self._forward_idle_draft_prefill(
                batch=batch, target_hidden=result.logits_output.hidden_states
            )
            return self._idle_extend_result(result)
        logits_output = result.logits_output
        target_hidden = logits_output.hidden_states
        if target_hidden is None:
            raise RuntimeError("EAGLE3 extend requires target hidden capture.")
        assert logits_output.hidden_states_token_indices is None
        # Sampling ran at the batch bucket; the draft prefill reads those rows,
        # and must run before returning: the scheduler may update radix
        # afterward, invalidating out_cache_loc.
        seed = self._forward_draft_prefill(
            batch=batch,
            target_hidden=target_hidden,
            mm_input_embeds=logits_output.mm_input_embeds,
            sampled=result.next_token_ids,
        )
        logits_output.hidden_states = None
        logits_output.mm_input_embeds = None
        bs = batch.seq_lens.shape[0]
        result.next_token_ids = result.next_token_ids[:bs]
        next_state = SpecState(
            bonus_tokens=result.next_token_ids,
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
            draft_carry=map_carry(seed, lambda t: t[:bs]),
        )
        result.new_seq_lens = next_state.seq_lens
        result.next_draft_input = self.state_to_spec_info(next_state)
        return result

    def _forward_idle_step(
        self,
        *,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
    ) -> None:
        # The active step's forwards in order: draft chain, verify, draft extend.
        draft_tokens, verify_tokens = self._draft_chain(
            batch=batch, state=state, rows=rows, verify_input=verify_input
        )
        logits_output = self._forward_idle_verify(
            batch=batch,
            state=state,
            rows=rows,
            verify_input=verify_input,
            verify_tokens=verify_tokens,
        )
        accept = finalize_accept(
            draft_tokens,
            self._device_zeros(state.seq_lens.shape[0]),
            state.bonus_tokens,
            state.seq_lens,
            state.req_pool_indices != 0,
        )
        self._forward_draft_extend(
            batch=batch,
            state=state,
            rows=rows,
            verify_input=verify_input,
            target_hidden=logits_output.hidden_states,
            accept=accept,
        )

    def _forward_idle_draft_prefill(
        self, *, batch: ScheduleBatch, target_hidden: torch.Tensor
    ) -> None:
        spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden[:0],
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        with self._draft_context():
            with overridden_fields(batch, spec_info=spec_info):
                forward_batch = ForwardBatch.init_new(
                    batch,
                    self._draft_runner,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                    return_hidden_states_before_norm=False,
                )
            forward_batch.return_logprob = False
            with torch.no_grad():
                self._draft_runner.forward(forward_batch)

    def _idle_extend_result(
        self, result: GenerationBatchResult
    ) -> GenerationBatchResult:
        if result.logits_output is not None:
            result.logits_output.hidden_states = None
            result.logits_output.mm_input_embeds = None
        idle = self._idle_result()
        result.next_token_ids = idle.next_token_ids
        result.new_seq_lens = idle.new_seq_lens
        result.next_draft_input = idle.next_draft_input
        return result

    def get_spec_state(self, batch: ScheduleBatch) -> SpecState:
        draft_input = batch.spec_info
        if not isinstance(draft_input, HostPaddedEagleDraftInput):
            raise TypeError(
                f"Expected HostPaddedEagleDraftInput, got {type(draft_input)}."
            )
        return SpecState(
            bonus_tokens=draft_input.bonus_tokens,
            seq_lens=batch.seq_lens,
            req_pool_indices=batch.req_pool_indices,
            draft_carry=EagleSeed(
                topk_p=draft_input.topk_p,
                topk_index=draft_input.topk_index,
                hidden_states=draft_input.hidden_states,
            ),
        )

    def make_verify_input(
        self, *, batch: ScheduleBatch, verify_tokens: torch.Tensor, rows: VerifyRows
    ) -> EagleVerifyInput:
        return eagle_verify_input(
            draft_token=verify_tokens,
            positions=rows.flat_positions,
            num_drafts=self.num_drafts,
        )

    def state_to_spec_info(self, state: SpecState) -> HostPaddedEagleDraftInput:
        seed = state.draft_carry
        return HostPaddedEagleDraftInput(
            bonus_tokens=state.bonus_tokens.to(torch.int64),
            new_seq_lens=state.seq_lens.to(torch.int64),
            topk_p=seed.topk_p,
            topk_index=seed.topk_index,
            hidden_states=seed.hidden_states,
        )

    def make_dummy_carry(self, rows: int) -> EagleSeed:
        return EagleSeed(
            topk_p=torch.ones((rows, 1), dtype=torch.float32, device=self.device),
            topk_index=torch.zeros((rows, 1), dtype=torch.int64, device=self.device),
            hidden_states=torch.zeros(
                (rows, self._hidden_size), dtype=self._hidden_dtype, device=self.device
            ),
        )

    @contextlib.contextmanager
    def _draft_context(self) -> Iterator[None]:
        # The scopes EAGLEWorkerV2 runs every draft-model phase under.
        with (
            draft_tp_context(self._eagle.draft_owns_attention),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            yield

    def _forward_draft_extend(
        self,
        *,
        batch: ScheduleBatch,
        state: SpecState,
        rows: VerifyRows,
        verify_input: TpuSpecVerifyInput,
        target_hidden: torch.Tensor,
        accept: AcceptResult,
    ) -> EagleSeed:
        """DRAFT_EXTEND_V2 over the verify rows, prepared like prepare_for_draft_extend."""
        padded_bs = state.seq_lens.shape[0]
        width = self.num_verify_rows
        input_ids, post_seq_lens = draft_extend_inputs(
            accept.accept_tokens, state.seq_lens
        )
        spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden,
            num_correct_drafts=accept.num_correct_drafts,
            num_accept_tokens=accept.accept_lens,
            positions=rows.flat_positions,
            num_tokens_per_req=width,
            num_tokens_for_logprob_per_req=width,
        )
        num_pad = padded_bs - verify_input.seq_lens_cpu.shape[0]
        with self._draft_context():
            # Every row extends the whole window, as in the exact-bucket warmup;
            # padded rows keep host length 0, so RPA leaves them inactive. The
            # RPA backend ignores the (host, committed-base) prefix lengths.
            with self._bucket_view(
                batch,
                bucket_state=state._replace(seq_lens=post_seq_lens),
                verify_input=verify_input,
                kv_rows=width,
                spec_info=spec_info,
                input_ids=input_ids,
                out_cache_loc=rows.flat_cache_locs,
                forward_mode=ForwardMode.DRAFT_EXTEND_V2,
                prefix_lens=verify_input.seq_lens_cpu.tolist() + [0] * num_pad,
                extend_lens=[width] * padded_bs,
                extend_num_tokens=padded_bs * width,
                extend_logprob_start_lens=None,
            ) as view:
                forward_batch = ForwardBatch.init_new(
                    view,
                    self._draft_runner,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    return_hidden_states_before_norm=False,
                )
            # init_new keeps HOST_PADDED extend lengths on host, and draft extend
            # arrives planned, so the policy's host-padded upload never runs.
            for name in ("extend_seq_lens", "extend_prefix_lens", "extend_start_loc"):
                setattr(
                    forward_batch,
                    name,
                    getattr(forward_batch, name).to(self.device, non_blocking=True),
                )
            compile_policy = self._draft_runner.compile_policy
            compile_policy.prepare_spec_forward_batch(forward_batch)
            self._draft_runner.attn_backend.init_forward_metadata(forward_batch)
            forward_batch.mark_forward_metadata_ready()
            with torch.no_grad():
                logits_output = self._draft_runner.forward(forward_batch).logits_output
        _, hidden_states, topk_p, topk_index = (
            compile_policy.finalize_spec_forward_batch(
                outputs=fused_draft_extend_select_argmax_func(
                    logits_output.next_token_logits,
                    logits_output.hidden_states,
                    accept.accept_lens,
                ),
                forward_batch=forward_batch,
            )
        )
        return EagleSeed(
            topk_p=topk_p, topk_index=topk_index, hidden_states=hidden_states
        )

    def _forward_draft_prefill(
        self,
        *,
        batch: ScheduleBatch,
        target_hidden: torch.Tensor,
        mm_input_embeds: torch.Tensor | None,
        sampled: torch.Tensor,
    ) -> EagleSeed:
        """EXTEND(LAST) of the draft over the shifted prompt; seeds of the batch-bucket rows."""
        # finalize keeps HOST_PADDED target hidden at the token bucket; build the
        # shifted tokens there too, so the shim pads nothing.
        input_ids_cpu, tail_src_cpu = _shifted_prompt(
            batch=batch,
            padded_tokens=target_hidden.shape[0],
            vocab_size=self.target_worker.model_runner.model_config.vocab_size,
            pin_memory=is_pin_memory_available(self.device),
        )
        cap = self._compile_policy()._extend_compile_cap()
        input_ids = fill_prefill_tail(
            input_ids_cpu.to(self.device, non_blocking=True),
            tail_src_cpu.to(self.device, non_blocking=True),
            pad_rows(sampled, cap),
        )
        spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        live_input_ids = input_ids_cpu[: batch.extend_num_tokens]
        with self._draft_context():
            # init_new reads the live token count off input_ids; the draft then
            # reads the shifted tokens already at the token bucket.
            with overridden_fields(
                batch, input_ids=live_input_ids, spec_info=spec_info
            ):
                forward_batch = ForwardBatch.init_new(
                    batch,
                    self._draft_runner,
                    capture_hidden_mode=CaptureHiddenMode.LAST,
                    return_hidden_states_before_norm=False,
                )
            forward_batch.input_ids = input_ids
            forward_batch.return_logprob = False
            if forward_batch.global_num_tokens_cpu is not None:
                # DP attention pads the batch to the synced live counts, but the
                # tokens are already at the token bucket, and the draft (run in
                # its attention-TP group) gathers nothing across DP ranks.
                num_ranks = len(forward_batch.global_num_tokens_cpu)
                forward_batch.global_num_tokens_cpu = [input_ids.shape[0]] * num_ranks
                forward_batch.global_num_tokens_gpu = torch.full(
                    (num_ranks,), input_ids.shape[0], dtype=torch.int64
                ).to(self.device, non_blocking=True)
            if mm_input_embeds is not None:
                assert mm_input_embeds.shape[0] == target_hidden.shape[0]
                # Row p embeds prompt token p + 1; the draft shim folds each
                # request's last token in (fold_last_token_embeds).
                forward_batch.mm_input_embeds = shift_rows_up(mm_input_embeds)
            with torch.no_grad():
                logits_output = self._draft_runner.forward(forward_batch).logits_output
        topk_p, topk_index = select_first_draft(logits_output.next_token_logits)
        return EagleSeed(
            topk_p=topk_p,
            topk_index=topk_index,
            hidden_states=logits_output.hidden_states,
        )

    def _warmup_draft_glue(self) -> None:
        """The eager ops around the fused draft loop and draft extend, per batch bucket.

        EagleDraftWorker's precompile covers the draft metadata, the fused loop
        and the draft-extend forward.
        """
        width = self.num_verify_rows
        with torch.no_grad():
            for padded_bs in self._b_bucketer().buckets:
                cache_locs = torch.zeros(
                    (padded_bs, width), dtype=torch.int64, device=self.device
                )
                seq_lens = torch.zeros(
                    (padded_bs,), dtype=torch.int64, device=self.device
                )
                # prepare_for_draft: the draft-step slots and positions.
                cache_locs[:, : self.num_drafts].reshape(-1).cpu()
                seq_lens.repeat_interleave(1, dim=0).cpu()
                warmup_compiled_op(split_draft_chain, cache_locs)
                warmup_compiled_op(draft_extend_inputs, cache_locs, seq_lens)
                if self.target_worker.model_runner.model_config.model_is_mrope:
                    # compute_spec_mrope_positions on the verify rows: text-only
                    # batches, then batches with per-request deltas.
                    positions = cache_locs.reshape(-1).view(padded_bs, -1)
                    positions.flatten().unsqueeze(0).repeat(3, 1).cpu()
                    deltas = torch.zeros((padded_bs, 1), dtype=torch.int64)
                    (positions + deltas.to(self.device)).flatten().unsqueeze(0).repeat(
                        3, 1
                    ).cpu()

    def _warmup_target_only_accept(self) -> None:
        """Target-only acceptance per batch bucket; the base warms the target filters."""
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
                    torch.ones((padded_bs,), dtype=torch.int32, device=self.device),
                    self._accept_threshold_single,
                    self._accept_threshold_acc,
                )

    def _warmup_prefill_glue(self) -> None:
        """The draft prefill's token shift and first-draft pick, per token / batch bucket."""
        compile_policy = self._compile_policy()
        cap = compile_policy._extend_compile_cap()
        assert compile_policy.t_bucketer is not None
        draft_vocab_size = (
            getattr(self._draft_runner.model.config, "draft_vocab_size", None)
            or self._draft_runner.model.config.vocab_size
        )
        tail_tokens = torch.zeros((cap,), dtype=torch.int64, device=self.device)
        with torch.no_grad():
            for num_tokens in compile_policy.t_bucketer.buckets:
                if num_tokens < cap:
                    continue
                warmup_compiled_op(
                    fill_prefill_tail,
                    torch.zeros((num_tokens,), dtype=torch.int64).to(self.device),
                    torch.full((num_tokens,), -1, dtype=torch.int64).to(self.device),
                    tail_tokens,
                )
            if self._is_multimodal:
                self._warmup_multimodal_prefill(cap)
            for padded_bs in sorted({self._pad_bs(bs) for bs in range(1, cap + 1)}):
                pad_rows(
                    torch.zeros((padded_bs,), dtype=torch.int64, device=self.device),
                    cap,
                ).cpu()
                # finalize hands the draft prefill's logits on as [:pbs] of the cap rows.
                warmup_compiled_op(
                    select_first_draft,
                    torch.zeros((cap, draft_vocab_size), dtype=torch.float32).to(
                        self.device
                    )[:padded_bs],
                )

    def _warmup_multimodal_prefill(self, cap: int) -> None:
        """The draft prefill's embed shift per token bucket, and live-row logits padding."""
        compile_policy = self._compile_policy()
        runner = self.target_worker.model_runner
        embed_tokens = runner.model.get_input_embeddings()
        for num_tokens in compile_policy.t_bucketer.buckets:
            if num_tokens >= cap:
                warmup_compiled_op(
                    shift_rows_up,
                    torch.zeros(
                        (num_tokens, embed_tokens.embedding_dim),
                        dtype=embed_tokens.weight.dtype,
                        device=self.device,
                    ),
                )
        # The wrapper's eager logits processor emits live rows;
        # TpuSampler pads them.
        vocab_size = runner.model_config.vocab_size
        for bs in range(1, cap + 1):
            logits = torch.zeros((bs, vocab_size), dtype=torch.float32).to(self.device)
            torch.nn.functional.pad(logits, (0, 0, 0, self._pad_bs(bs) - bs))
        torch.accelerator.synchronize()


def _check_eagle3_config(target_worker: TpModelWorker) -> None:
    if mambaish_config(target_worker.model_runner.model_config) is not None:
        raise NotImplementedError("TpuEagle3Worker does not support mamba targets.")
    spec = get_spec()
    if (spec.speculative_eagle_topk or 1) != 1:
        raise NotImplementedError("TpuEagle3Worker only supports EAGLE3 topk 1 chains.")
    if spec.speculative_num_steps < 2:
        raise NotImplementedError(
            "TpuEagle3Worker needs speculative_num_steps >= 2 (the fused draft loop)."
        )
    if spec.speculative_num_draft_tokens != spec.speculative_num_steps + 1:
        raise ValueError(
            f"speculative_num_draft_tokens={spec.speculative_num_draft_tokens} must "
            f"equal speculative_num_steps + 1 = {spec.speculative_num_steps + 1}."
        )
    if spec.enable_multi_layer_eagle or spec.speculative_adaptive:
        raise NotImplementedError(
            "TpuEagle3Worker does not support multi-layer or adaptive EAGLE."
        )


def _shifted_prompt(
    *, batch: ScheduleBatch, padded_tokens: int, vocab_size: int, pin_memory: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Host draft-prefill tokens and tail sources [padded_tokens], int64.

    Row p holds the prompt token after p. Each request's last row is filled on
    device from the sampled token of its row (tail_src; -1 elsewhere), or for a
    non-final chunk from its next prompt token, which keeps the chain consistent
    across chunks.
    """
    prompt = batch.input_ids
    assert prompt.device.type == "cpu", prompt.device
    num_tokens = prompt.shape[0]
    extend_lens = torch.tensor(batch.extend_lens, dtype=torch.int64)
    last_rows = torch.cumsum(extend_lens, dim=0) - 1
    input_ids = torch.zeros((padded_tokens,), dtype=torch.int64, pin_memory=pin_memory)
    # Image placeholders carry hash ids past the vocab; the draft reads their
    # rows from the target's embeds, but must not index embed_tokens out of range.
    input_ids[: num_tokens - 1] = prompt[1:].clamp(0, vocab_size - 1)
    tail_src = torch.full(
        (padded_tokens,), -1, dtype=torch.int64, pin_memory=pin_memory
    )
    tail_src[last_rows] = torch.arange(extend_lens.shape[0])
    next_prompt_token = batch.chunked_req_next_prompt_token
    if next_prompt_token is not None:
        for i, req in enumerate(batch.reqs):
            if req is batch.chunked_req:
                input_ids[last_rows[i]] = next_prompt_token
                tail_src[last_rows[i]] = -1
                break
    return input_ids, tail_src

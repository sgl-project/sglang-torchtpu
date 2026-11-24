"""TPU compile policy for TpuBlockSpecWorker's target and draft runners."""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Callable, Iterator
from typing import Any

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    LogitsOutputMode,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_forward_context,
)
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_disagg, get_exec, get_parallel
from sglang.srt.speculative.draft_worker_common import make_draft_block_spec_info
from sglang.srt.utils import is_tpu
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.attention_rank_inputs import (
    get_attention_rank_tensors,
    use_attention_rank_tensors,
)
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.recompilation_checker import (
    install_recompilation_checker,
)
from sglang_tpu.compilation.warmup_queue import current_warmup_queue
from sglang_tpu.dflash_utils_tpu import register_argmax_draft_top_k
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental import runs_draft_prompt_pass
from sglang_tpu.experimental.compilation.spec_compile_policy import (
    TpuSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.speculative.spec_ops import (
    accept_greedy_top_k_step_impl,
    accept_top_k_candidates_step_impl,
    prepare_dflash_block_step_impl,
    verify_tokens_of,
)
from sglang_tpu.experimental.speculative.spec_types import AcceptResult, VerifyRows

logger = get_logger(__name__)


def unpack_draft_lattice(raw_lattice: Any) -> tuple[Any, torch.Tensor | None]:
    """Unpack DFlashDraftModel.forward's next_draft_lattice into (lattice, verify_tokens)."""
    if (
        isinstance(raw_lattice, tuple)
        and len(raw_lattice) == 2
        and isinstance(raw_lattice[1], torch.Tensor)
        and raw_lattice[1].ndim == 1
    ):
        return raw_lattice[0], raw_lattice[1]
    return raw_lattice, None


@contextlib.contextmanager
def _maybe_guard_stance(stance: Any) -> Iterator[None]:
    if stance is not None and not torch.compiler.is_compiling() and stance.armed():
        with stance:
            yield
        return
    yield


def _prep_and_draft_specialization_key(
    _attention_metadata: Any,
    packed_addressing: torch.Tensor,
    bonus_tokens: torch.Tensor,
    _state_seq_lens: torch.Tensor,
    _live_mask: torch.Tensor,
    _req_pool_indices: torch.Tensor,
    num_token_non_padded: torch.Tensor | None,
    _global_num_tokens_gpu: torch.Tensor | None,
    _global_num_tokens_for_logprob_gpu: torch.Tensor | None,
    *,
    num_verify_rows: int,
    mask_token_id: int,
    draft_width: int,
    has_swa: bool,
    dp_padding_mode: Any = None,
    **_kwargs: Any,
) -> tuple[Any, ...]:
    return (
        int(bonus_tokens.shape[0]),
        tuple(packed_addressing.shape),
        packed_addressing.dtype,
        num_token_non_padded is not None,
        int(num_verify_rows),
        int(mask_token_id),
        int(draft_width),
        bool(has_swa),
        dp_padding_mode,
    )


def _accept_write_greedy_specialization_key(
    *,
    top_ids: torch.Tensor,
    target_hidden: torch.Tensor,
    cache_loc: torch.Tensor,
    cache_loc_2d: torch.Tensor,
    **_kwargs: Any,
) -> tuple[Any, ...]:
    return (
        tuple(top_ids.shape),
        top_ids.dtype,
        tuple(target_hidden.shape),
        target_hidden.dtype,
        tuple(cache_loc.shape),
        cache_loc.dtype,
        tuple(cache_loc_2d.shape),
        cache_loc_2d.dtype,
    )


def _accept_write_candidates_specialization_key(
    *,
    top_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    target_hidden: torch.Tensor,
    cache_loc: torch.Tensor,
    cache_loc_2d: torch.Tensor,
    block_verification: bool,
    **_kwargs: Any,
) -> tuple[Any, ...]:
    return (
        tuple(top_ids.shape),
        top_ids.dtype,
        tuple(draft_ids.shape),
        draft_ids.dtype,
        tuple(target_hidden.shape),
        target_hidden.dtype,
        tuple(cache_loc.shape),
        cache_loc.dtype,
        tuple(cache_loc_2d.shape),
        cache_loc_2d.dtype,
        bool(block_verification),
    )


class TpuBlockSpecDecodeCompilePolicyMixin(TpuSpecDecodeCompilePolicyMixin):
    """The draft warms its block forward and KV writer.

    DFlash and DSpark drafts share the spec inputs and K/V layout, so the target
    verify grid and the draft KV writer carry over. The target scores
    [r, d_1..d_gamma] (draft_token_num = gamma + 1 rows); the draft forwards
    [r, m, ..., m] (the drafter's block_width rows) and never runs EXTEND.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._eager_model_forward: Callable[..., Any] | None = None
        self._kv_writer: Callable[..., torch.Tensor] | None = None
        self._compiled_prep_and_draft: Callable[..., tuple[Any, ...]] | None = None
        self._canonical_draft_spec_info: Any = None
        self._compiled_accept_write_greedy_top_k: (
            Callable[..., tuple[AcceptResult, torch.Tensor]] | None
        ) = None
        self._compiled_accept_write_top_k_candidates: (
            Callable[..., tuple[AcceptResult, torch.Tensor, torch.Tensor]] | None
        ) = None
        self._block_skip_guard_stance: Any = None

    def _get_canonical_draft_spec_info(self, draft_width: int) -> Any:
        info = self._canonical_draft_spec_info
        if info is None or int(info.num_tokens_per_req) != int(draft_width):
            info = make_draft_block_spec_info(
                draft_token_num=draft_width, device=self.model_runner.device
            )
            self._canonical_draft_spec_info = info
        return info

    def install(self) -> bool:
        """Preserve the draft model's uncompiled forward for fused prep+draft tracing."""
        if self.is_draft_worker and self.owner is not None:
            self._eager_model_forward = self.owner.forward
        return super().install()

    def _get_skip_guard_stance(self) -> Any:
        if self.tracer == "direct":
            return None
        if self._block_skip_guard_stance is not None:
            return self._block_skip_guard_stance
        if (
            tpu_envs.SGLANG_TPU_VERIFY_SKIP_GUARD_EVAL.get()
            or tpu_envs.SGLANG_TPU_FAST_COMPILED_OP.get()
        ):
            from sglang_tpu.compilation.compiled_forward_shim import (
                _SkipGuardStance,
            )

            self._block_skip_guard_stance = _SkipGuardStance()
        return self._block_skip_guard_stance

    def _compile_kv_pool_op(
        self,
        fn: Callable[..., Any],
        *,
        specialization_key: Callable[..., tuple[Any, ...]],
    ) -> Callable[..., Any]:
        """Compile a step helper that reads draft weights and mutates draft KV pool."""
        if self.tracer == "direct":
            from sglang_tpu.compilation.direct_compile import direct_compile

            return direct_compile(
                torch.no_grad()(fn),
                specialization_key=specialization_key,
                model=self.model_runner.model,
                model_extra_tensor_attrs=self.get_model_extra_tensor_attrs(),
                kv_pool=self.get_direct_compile_kv_pool(),
            )
        if not is_tpu() or not envs.SGLANG_ENABLE_TORCH_COMPILE.get():
            return torch.no_grad()(fn)
        compiled = torch.compile(
            fn,
            backend=self.backend or current_platform.get_compile_backend(),
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )

        def call_no_grad(*args: Any, **kwargs: Any) -> Any:
            with torch.no_grad():
                return compiled(*args, **kwargs)

        return call_no_grad

    def _ensure_prep_and_draft(self) -> Callable[..., tuple[Any, ...]]:
        if self._compiled_prep_and_draft is not None:
            return self._compiled_prep_and_draft
        eager_forward = self._eager_model_forward or self.model_runner.model.forward
        model = self.model_runner.model

        def _build_traced_batch(
            *,
            draft_inputs: Any,
            req_pool_indices: torch.Tensor,
            num_token_non_padded: torch.Tensor | None,
            global_num_tokens_gpu: torch.Tensor | None,
            global_num_tokens_for_logprob_gpu: torch.Tensor | None,
            padded_bs: int,
            draft_width: int,
            spec_algorithm: Any,
            spec_info: Any,
            dp_padding_mode: Any,
            global_num_tokens_cpu: tuple[int, ...] | None,
            global_num_tokens_for_logprob_cpu: tuple[int, ...] | None,
        ) -> ForwardBatch:
            batch = ForwardBatch(
                forward_mode=ForwardMode.TARGET_VERIFY,
                out_cache_loc_is_physical=True,
                batch_size=padded_bs,
                input_ids=draft_inputs.input_ids,
                req_pool_indices=req_pool_indices,
                req_pool_indices_cpu=None,
                seq_lens=draft_inputs.seq_lens,
                out_cache_loc=draft_inputs.cache_locs,
                seq_lens_sum=0,
                seq_lens_cpu=None,
                positions=draft_inputs.positions,
                input_embeds=None,
                spec_algorithm=spec_algorithm,
                spec_info=spec_info,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                logits_output_mode=LogitsOutputMode.FULL_VOCAB,
                extend_prefix_lens=draft_inputs.extend_prefix_lens,
                extend_seq_lens=draft_inputs.extend_seq_lens,
                extend_num_tokens=padded_bs * draft_width,
                num_token_non_padded=num_token_non_padded,
                is_prefill_only=False,
            )
            if dp_padding_mode is not None:
                batch.dp_padding_mode = dp_padding_mode
                batch.is_extend_in_batch = False
                batch.global_num_tokens_cpu = (
                    list(global_num_tokens_cpu)
                    if global_num_tokens_cpu is not None
                    else None
                )
                batch.original_global_num_tokens_cpu = (
                    list(global_num_tokens_cpu)
                    if global_num_tokens_cpu is not None
                    else None
                )
                batch.global_num_tokens_gpu = global_num_tokens_gpu
                batch.global_num_tokens_for_logprob_cpu = (
                    list(global_num_tokens_for_logprob_cpu)
                    if global_num_tokens_for_logprob_cpu is not None
                    else None
                )
                batch.global_num_tokens_for_logprob_gpu = (
                    global_num_tokens_for_logprob_gpu
                )
            return batch

        def prep_and_draft_core(
            packed_addressing: torch.Tensor,
            bonus_tokens: torch.Tensor,
            state_seq_lens: torch.Tensor,
            live_mask: torch.Tensor,
            req_pool_indices: torch.Tensor,
            num_token_non_padded: torch.Tensor | None,
            global_num_tokens_gpu: torch.Tensor | None,
            global_num_tokens_for_logprob_gpu: torch.Tensor | None,
            *,
            num_verify_rows: int,
            mask_token_id: int,
            draft_width: int,
            has_swa: bool,
            spec_algorithm: Any,
            spec_info: Any,
            dp_padding_mode: Any,
            global_num_tokens_cpu: tuple[int, ...] | None,
            global_num_tokens_for_logprob_cpu: tuple[int, ...] | None,
        ) -> tuple[Any, ...]:
            rows, swa_rows, draft_inputs = prepare_dflash_block_step_impl(
                packed_addressing,
                bonus_tokens,
                state_seq_lens,
                live_mask,
                num_verify_rows,
                mask_token_id,
                draft_width,
                has_swa,
            )
            padded_bs = int(bonus_tokens.shape[0])
            kv_lens = (draft_inputs.seq_lens + draft_width).to(torch.int32)
            traced_spec_info = dataclasses.replace(spec_info, rpa_draft_kv_lens=kv_lens)
            traced_batch = _build_traced_batch(
                draft_inputs=draft_inputs,
                req_pool_indices=req_pool_indices,
                num_token_non_padded=num_token_non_padded,
                global_num_tokens_gpu=global_num_tokens_gpu,
                global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
                padded_bs=padded_bs,
                draft_width=draft_width,
                spec_algorithm=spec_algorithm,
                spec_info=traced_spec_info,
                dp_padding_mode=dp_padding_mode,
                global_num_tokens_cpu=global_num_tokens_cpu,
                global_num_tokens_for_logprob_cpu=global_num_tokens_for_logprob_cpu,
            )
            logits_output = eager_forward(
                draft_inputs.input_ids, draft_inputs.positions, traced_batch
            )
            draft_tokens = logits_output.next_draft_tokens
            hidden_states = logits_output.hidden_states
            assert hidden_states is not None and draft_tokens is not None
            draft_lattice, verify_tokens = unpack_draft_lattice(
                logits_output.next_draft_lattice
            )
            if verify_tokens is None:
                verify_tokens = verify_tokens_of(
                    bonus_tokens, draft_tokens.to(bonus_tokens.dtype)
                )
            return (
                rows,
                swa_rows,
                hidden_states.view(padded_bs, draft_width, -1),
                draft_tokens,
                draft_lattice,
                verify_tokens,
            )

        if self.tracer == "direct":
            from sglang_tpu.compilation.direct_compile import direct_compile
            from sglang_tpu.compilation.hybrid_linear_direct_compile import (
                bind_traced_attention_metadata,
                bind_traced_conv_weights,
                traced_attention_metadata,
            )

            eager_no_grad = torch.no_grad()(eager_forward)

            def trace_prep_and_draft(
                attention_metadata: Any,
                packed_addressing: torch.Tensor,
                bonus_tokens: torch.Tensor,
                state_seq_lens: torch.Tensor,
                live_mask: torch.Tensor,
                req_pool_indices: torch.Tensor,
                num_token_non_padded: torch.Tensor | None,
                global_num_tokens_gpu: torch.Tensor | None,
                global_num_tokens_for_logprob_gpu: torch.Tensor | None,
                *,
                num_verify_rows: int,
                mask_token_id: int,
                draft_width: int,
                has_swa: bool,
                spec_algorithm: Any,
                spec_info: Any,
                dp_padding_mode: Any,
                global_num_tokens_cpu: tuple[int, ...] | None,
                global_num_tokens_for_logprob_cpu: tuple[int, ...] | None,
                attention_rank_tensors: tuple[torch.Tensor, ...],
            ) -> tuple[Any, ...]:
                rows, swa_rows, draft_inputs = prepare_dflash_block_step_impl(
                    packed_addressing,
                    bonus_tokens,
                    state_seq_lens,
                    live_mask,
                    num_verify_rows,
                    mask_token_id,
                    draft_width,
                    has_swa,
                )
                padded_bs = int(bonus_tokens.shape[0])
                kv_lens = (draft_inputs.seq_lens + draft_width).to(torch.int32)
                traced_spec_info = dataclasses.replace(
                    spec_info, rpa_draft_kv_lens=kv_lens
                )
                traced_batch = _build_traced_batch(
                    draft_inputs=draft_inputs,
                    req_pool_indices=req_pool_indices,
                    num_token_non_padded=num_token_non_padded,
                    global_num_tokens_gpu=global_num_tokens_gpu,
                    global_num_tokens_for_logprob_gpu=global_num_tokens_for_logprob_gpu,
                    padded_bs=padded_bs,
                    draft_width=draft_width,
                    spec_algorithm=spec_algorithm,
                    spec_info=traced_spec_info,
                    dp_padding_mode=dp_padding_mode,
                    global_num_tokens_cpu=global_num_tokens_cpu,
                    global_num_tokens_for_logprob_cpu=global_num_tokens_for_logprob_cpu,
                )
                backend = get_forward_context().attn_backend
                with (
                    bind_traced_attention_metadata(
                        backend, attention_metadata, traced_batch.forward_mode
                    ),
                    bind_traced_conv_weights(model),
                    use_attention_rank_tensors(attention_rank_tensors),
                ):
                    logits_output = eager_no_grad(
                        draft_inputs.input_ids, draft_inputs.positions, traced_batch
                    )
                draft_tokens = logits_output.next_draft_tokens
                hidden_states = logits_output.hidden_states
                assert hidden_states is not None and draft_tokens is not None
                draft_lattice, verify_tokens = unpack_draft_lattice(
                    logits_output.next_draft_lattice
                )
                if verify_tokens is None:
                    verify_tokens = verify_tokens_of(
                        bonus_tokens, draft_tokens.to(bonus_tokens.dtype)
                    )
                return (
                    rows,
                    swa_rows,
                    hidden_states.view(padded_bs, draft_width, -1),
                    draft_tokens,
                    draft_lattice,
                    verify_tokens,
                )

            compiled = direct_compile(
                trace_prep_and_draft,
                specialization_key=_prep_and_draft_specialization_key,
                lift_rank=(
                    torch.distributed.is_initialized()
                    and (
                        get_parallel().tp_group.world_size
                        == torch.distributed.get_world_size()
                        or get_exec().graph.use_spmd_compile
                    )
                ),
                model=model,
                model_extra_tensor_attrs=self.get_model_extra_tensor_attrs(),
                kv_pool=self.get_direct_compile_kv_pool(),
                ignored_argnames=self._ignored_argnames(),
            )

            def call_direct(
                *,
                forward_batch: ForwardBatch,
                packed_addressing: torch.Tensor,
                bonus_tokens: torch.Tensor,
                state_seq_lens: torch.Tensor,
                live_mask: torch.Tensor,
                num_verify_rows: int,
                mask_token_id: int,
                draft_width: int,
                has_swa: bool,
            ) -> tuple[Any, ...]:
                backend = get_forward_context().attn_backend
                dp_cpu = forward_batch.global_num_tokens_cpu
                dp_lp_cpu = forward_batch.global_num_tokens_for_logprob_cpu
                return compiled(
                    traced_attention_metadata(backend),
                    packed_addressing,
                    bonus_tokens,
                    state_seq_lens,
                    live_mask,
                    forward_batch.req_pool_indices,
                    forward_batch.num_token_non_padded,
                    forward_batch.global_num_tokens_gpu,
                    forward_batch.global_num_tokens_for_logprob_gpu,
                    num_verify_rows=num_verify_rows,
                    mask_token_id=mask_token_id,
                    draft_width=draft_width,
                    has_swa=has_swa,
                    spec_algorithm=forward_batch.spec_algorithm,
                    spec_info=self._get_canonical_draft_spec_info(draft_width),
                    dp_padding_mode=forward_batch.dp_padding_mode,
                    global_num_tokens_cpu=tuple(dp_cpu) if dp_cpu is not None else None,
                    global_num_tokens_for_logprob_cpu=(
                        tuple(dp_lp_cpu) if dp_lp_cpu is not None else None
                    ),
                    attention_rank_tensors=get_attention_rank_tensors(
                        torch.device(self.model_runner.device)
                    ),
                )

            self._compiled_prep_and_draft = call_direct
            return call_direct

        compiled_dynamo = (
            torch.compile(
                prep_and_draft_core,
                backend=self.backend or current_platform.get_compile_backend(),
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
            )
            if (is_tpu() and envs.SGLANG_ENABLE_TORCH_COMPILE.get())
            else prep_and_draft_core
        )

        def call_dynamo(
            *,
            forward_batch: ForwardBatch,
            packed_addressing: torch.Tensor,
            bonus_tokens: torch.Tensor,
            state_seq_lens: torch.Tensor,
            live_mask: torch.Tensor,
            num_verify_rows: int,
            mask_token_id: int,
            draft_width: int,
            has_swa: bool,
        ) -> tuple[Any, ...]:
            dp_cpu = forward_batch.global_num_tokens_cpu
            dp_lp_cpu = forward_batch.global_num_tokens_for_logprob_cpu
            with torch.no_grad(), _maybe_guard_stance(self._get_skip_guard_stance()):
                return compiled_dynamo(
                    packed_addressing,
                    bonus_tokens,
                    state_seq_lens,
                    live_mask,
                    forward_batch.req_pool_indices,
                    forward_batch.num_token_non_padded,
                    forward_batch.global_num_tokens_gpu,
                    forward_batch.global_num_tokens_for_logprob_gpu,
                    num_verify_rows=num_verify_rows,
                    mask_token_id=mask_token_id,
                    draft_width=draft_width,
                    has_swa=has_swa,
                    spec_algorithm=forward_batch.spec_algorithm,
                    spec_info=self._get_canonical_draft_spec_info(draft_width),
                    dp_padding_mode=forward_batch.dp_padding_mode,
                    global_num_tokens_cpu=tuple(dp_cpu) if dp_cpu is not None else None,
                    global_num_tokens_for_logprob_cpu=(
                        tuple(dp_lp_cpu) if dp_lp_cpu is not None else None
                    ),
                )

        self._compiled_prep_and_draft = call_dynamo
        return call_dynamo

    def run_prep_and_draft(
        self,
        *,
        forward_batch: ForwardBatch,
        packed_addressing: torch.Tensor,
        bonus_tokens: torch.Tensor,
        state_seq_lens: torch.Tensor,
        live_mask: torch.Tensor,
        num_verify_rows: int,
        mask_token_id: int,
        draft_width: int,
        has_swa: bool,
    ) -> tuple[
        VerifyRows,
        VerifyRows | None,
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor] | None,
        torch.Tensor,
    ]:
        """Execute fused prepare_dflash_block_step + draft block forward (Launch 1)."""
        return self._ensure_prep_and_draft()(
            forward_batch=forward_batch,
            packed_addressing=packed_addressing,
            bonus_tokens=bonus_tokens,
            state_seq_lens=state_seq_lens,
            live_mask=live_mask,
            num_verify_rows=num_verify_rows,
            mask_token_id=mask_token_id,
            draft_width=draft_width,
            has_swa=has_swa,
        )

    def _ensure_accept_write_greedy_top_k(
        self, kv_writer: Callable[..., torch.Tensor] | None = None
    ) -> Callable[..., tuple[AcceptResult, torch.Tensor]]:
        if self._compiled_accept_write_greedy_top_k is not None:
            return self._compiled_accept_write_greedy_top_k
        writer = kv_writer or self._kv_writer
        if writer is None:
            raise RuntimeError("accept_and_write_draft_kv needs kv_writer initialized.")

        def fn(
            *,
            top_ids: torch.Tensor,
            draft_tokens: torch.Tensor,
            num_verify_drafts: torch.Tensor,
            seq_lens: torch.Tensor,
            live_mask: torch.Tensor,
            target_hidden: torch.Tensor,
            positions: torch.Tensor,
            cache_loc: torch.Tensor,
            cache_loc_2d: torch.Tensor,
        ) -> tuple[AcceptResult, torch.Tensor]:
            accept = accept_greedy_top_k_step_impl(
                top_ids, draft_tokens, num_verify_drafts, seq_lens, live_mask
            )
            kv_fence_tail = writer(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=accept.accept_lens,
            )
            return accept, kv_fence_tail

        self._compiled_accept_write_greedy_top_k = self._compile_kv_pool_op(
            fn, specialization_key=_accept_write_greedy_specialization_key
        )
        return self._compiled_accept_write_greedy_top_k

    def _ensure_accept_write_top_k_candidates(
        self, kv_writer: Callable[..., torch.Tensor] | None = None
    ) -> Callable[..., tuple[AcceptResult, torch.Tensor, torch.Tensor]]:
        if self._compiled_accept_write_top_k_candidates is not None:
            return self._compiled_accept_write_top_k_candidates
        writer = kv_writer or self._kv_writer
        if writer is None:
            raise RuntimeError("accept_and_write_draft_kv needs kv_writer initialized.")

        def fn(
            *,
            top_logits: torch.Tensor,
            top_ids: torch.Tensor,
            draft_tokens: torch.Tensor,
            num_verify_drafts: torch.Tensor,
            draft_ids: torch.Tensor,
            draft_logits: torch.Tensor,
            temperatures: torch.Tensor,
            top_ks: torch.Tensor,
            seq_lens: torch.Tensor,
            live_mask: torch.Tensor,
            rng_step: torch.Tensor,
            target_hidden: torch.Tensor,
            positions: torch.Tensor,
            cache_loc: torch.Tensor,
            cache_loc_2d: torch.Tensor,
            block_verification: bool,
        ) -> tuple[AcceptResult, torch.Tensor, torch.Tensor]:
            accept, next_rng_step = accept_top_k_candidates_step_impl(
                top_logits,
                top_ids,
                draft_tokens,
                num_verify_drafts,
                draft_ids,
                draft_logits,
                temperatures,
                top_ks,
                seq_lens,
                live_mask,
                rng_step,
                block_verification,
            )
            kv_fence_tail = writer(
                target_hidden=target_hidden,
                cache_loc=cache_loc,
                positions=positions,
                cache_loc_2d=cache_loc_2d,
                commit_lens=accept.accept_lens,
            )
            return accept, next_rng_step, kv_fence_tail

        self._compiled_accept_write_top_k_candidates = self._compile_kv_pool_op(
            fn, specialization_key=_accept_write_candidates_specialization_key
        )
        return self._compiled_accept_write_top_k_candidates

    def accept_and_write_draft_kv_greedy_top_k(
        self,
        *,
        top_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_verify_drafts: torch.Tensor,
        seq_lens: torch.Tensor,
        live_mask: torch.Tensor,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        kv_writer: Callable[..., torch.Tensor] | None = None,
    ) -> tuple[AcceptResult, torch.Tensor]:
        """Run fused greedy top-k accept + draft KV write in one compiled dispatch."""
        fn = self._ensure_accept_write_greedy_top_k(kv_writer)
        with _maybe_guard_stance(self._get_skip_guard_stance()):
            return fn(
                top_ids=top_ids,
                draft_tokens=draft_tokens,
                num_verify_drafts=num_verify_drafts,
                seq_lens=seq_lens,
                live_mask=live_mask,
                target_hidden=target_hidden,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
            )

    def accept_and_write_draft_kv_top_k_candidates(
        self,
        *,
        top_logits: torch.Tensor,
        top_ids: torch.Tensor,
        draft_tokens: torch.Tensor,
        num_verify_drafts: torch.Tensor,
        draft_ids: torch.Tensor,
        draft_logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        seq_lens: torch.Tensor,
        live_mask: torch.Tensor,
        rng_step: torch.Tensor,
        target_hidden: torch.Tensor,
        positions: torch.Tensor,
        cache_loc: torch.Tensor,
        cache_loc_2d: torch.Tensor,
        block_verification: bool,
        kv_writer: Callable[..., torch.Tensor] | None = None,
    ) -> tuple[AcceptResult, torch.Tensor, torch.Tensor]:
        """Run fused sampled candidate top-k accept + draft KV write in one dispatch."""
        fn = self._ensure_accept_write_top_k_candidates(kv_writer)
        with _maybe_guard_stance(self._get_skip_guard_stance()):
            return fn(
                top_logits=top_logits,
                top_ids=top_ids,
                draft_tokens=draft_tokens,
                num_verify_drafts=num_verify_drafts,
                draft_ids=draft_ids,
                draft_logits=draft_logits,
                temperatures=temperatures,
                top_ks=top_ks,
                seq_lens=seq_lens,
                live_mask=live_mask,
                rng_step=rng_step,
                target_hidden=target_hidden,
                positions=positions,
                cache_loc=cache_loc,
                cache_loc_2d=cache_loc_2d,
                block_verification=block_verification,
            )

    def target_verify_logits_output_modes(self) -> tuple[LogitsOutputMode, ...]:
        # TpuBlockSpecWorker.verify_top_k_width reads the same negotiated width.
        if self.is_draft_worker or self._distributed_top_k_width() is None:
            return (LogitsOutputMode.FULL_VOCAB,)
        if tpu_envs.SGLANG_TPU_SPEC_VERIFY_GREEDY_TOPK1.get():
            return (
                LogitsOutputMode.FULL_VOCAB,
                LogitsOutputMode.DISTRIBUTED_TOP_K,
                LogitsOutputMode.DISTRIBUTED_GREEDY,
            )
        return (LogitsOutputMode.FULL_VOCAB, LogitsOutputMode.DISTRIBUTED_TOP_K)

    def precompile_draft(
        self,
        draft_attn_backend: AttentionBackend,
        *,
        kv_writer: Callable[..., torch.Tensor],
        block_width: int,
        num_verify_rows: int | None = None,
        mask_token_id: int = 0,
        has_swa: bool = False,
    ) -> None:
        """Compile the draft block forward and the draft KV writer."""
        if not self.is_draft_worker:
            raise ValueError("precompile_draft runs on the draft runner only.")
        if self.b_bucketer is None:
            raise ValueError("b_bucketer must be initialized before precompile_draft.")
        start = time.perf_counter()
        self._kv_writer = kv_writer
        if self._compiled_kv_appender is None:
            self._compiled_kv_appender = self._compile_kv_appender(kv_writer)
        disagg_mode = get_disagg().disaggregation_mode
        pending: list[PendingCall] = []
        if disagg_mode != "prefill":
            if (
                tpu_envs.SGLANG_TPU_DFLASH_FUSE_STEP_OPS_V2.get()
                and num_verify_rows is not None
            ):
                self._precompile_prep_and_draft(
                    draft_attn_backend,
                    block_width=block_width,
                    num_verify_rows=num_verify_rows,
                    mask_token_id=mask_token_id,
                    has_swa=has_swa,
                )
            else:
                pending = self._precompile_draft_block(
                    draft_attn_backend, block_width=block_width
                )
            self._precompile_draft_append_kv_verify()
        if runs_draft_prompt_pass(disagg_mode):
            self._precompile_draft_append_kv_extend()
        queue = current_warmup_queue()
        if pending and queue is not None:
            # Async block warmups compile with the target's (shared under SPMD);
            # the checker baselines only once they have run.
            queue.calls.extend(pending)
            queue.completions.append(self._install_draft_recompilation_checker)
            logger.info(
                "[spec compile policy] %s DRAFT queued %d warmups in %.1fs.",
                self.algo_label,
                len(pending),
                time.perf_counter() - start,
            )
            return
        for warmup in pending:
            warmup.wait()
        logger.info(
            "[spec compile policy] %s DRAFT precompile done in %.1fs.",
            self.algo_label,
            time.perf_counter() - start,
        )
        self._install_draft_recompilation_checker()

    def _precompile_prep_and_draft(
        self,
        draft_attn_backend: AttentionBackend,
        *,
        block_width: int,
        num_verify_rows: int,
        mask_token_id: int,
        has_swa: bool,
    ) -> None:
        """Warm the fused prepare_dflash_block_step + draft block forward."""
        assert self.b_bucketer is not None
        lm_head = self.model_runner.model.lm_head
        if lm_head is not None:
            register_argmax_draft_top_k(lm_head)
        device = self.model_runner.device
        packed_cols = 2 * num_verify_rows * (2 if has_swa else 1) + 1
        shapes = list(self.b_bucketer.buckets)
        logger.info(
            "[spec compile policy] %s PREP+DRAFT BLOCK: %d shapes (bs=%s, width=%d, has_swa=%s)",
            self.algo_label,
            len(shapes),
            shapes,
            block_width,
            has_swa,
        )

        def run(bs: int) -> str:
            num_tokens = bs * block_width
            spec_info = self._get_canonical_draft_spec_info(block_width)
            args, kwargs = self._make_warmup_inputs(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.TARGET_VERIFY,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                spec_info=spec_info,
            )
            forward_batch = (
                kwargs["forward_batch"] if "forward_batch" in kwargs else args[2]
            )
            prepared = self.model_runner.prepare_dummy_forward_batch(forward_batch)
            if prepared is not forward_batch:
                forward_batch = prepared
            self._seed_dummy_req_to_token(forward_batch)
            if is_dp_attention_enabled():
                self.set_dp_attention_metadata(
                    forward_batch, num_tokens, bs, ForwardMode.TARGET_VERIFY
                )
            setattr(forward_batch, "_tpu_real_b", bs)
            setattr(forward_batch, "_tpu_real_q", num_tokens)
            spec_info.record_padding(
                logical_batch_size=bs,
                physical_batch_size=bs,
                logical_num_tokens=num_tokens,
                physical_num_tokens=num_tokens,
            )
            forward_batch.logits_output_mode = LogitsOutputMode.FULL_VOCAB
            forward_batch.is_prefill_only = False
            draft_attn_backend.init_forward_metadata(forward_batch)
            state_seq_lens = forward_batch.seq_lens.to(torch.int64)
            forward_batch = dataclasses.replace(
                forward_batch,
                seq_lens=state_seq_lens,
            )
            packed_addressing = torch.zeros(
                (bs, packed_cols), dtype=torch.int64, device=device
            )
            bonus_tokens = torch.zeros((bs,), dtype=torch.int64, device=device)
            live_mask = torch.ones((bs,), dtype=torch.bool, device=device)
            with (
                forward_context(
                    ForwardContext(attn_backend=draft_attn_backend),
                    forward_batch=forward_batch,
                ),
                torch.no_grad(),
            ):
                self.run_prep_and_draft(
                    forward_batch=forward_batch,
                    packed_addressing=packed_addressing,
                    bonus_tokens=bonus_tokens,
                    state_seq_lens=state_seq_lens,
                    live_mask=live_mask,
                    num_verify_rows=num_verify_rows,
                    mask_token_id=mask_token_id,
                    draft_width=block_width,
                    has_swa=has_swa,
                )
            return f"bs={bs}, tokens={num_tokens}"

        self._precompile_shapes(f"{self.algo_label} PREP+DRAFT BLOCK", shapes, run)

    def _install_draft_recompilation_checker(self) -> None:
        install_recompilation_checker(
            policies=(self,),
            model_label=f"draft {type(self.model_runner.model).__name__}",
        )

    def _precompile_draft_block(
        self, draft_attn_backend: AttentionBackend, *, block_width: int
    ) -> list[PendingCall]:
        assert self.b_bucketer is not None
        width = block_width
        lm_head = self.model_runner.model.lm_head
        if lm_head is not None:
            register_argmax_draft_top_k(lm_head)
        shapes = [(bs, bs * width) for bs in self.b_bucketer.buckets]
        logger.info(
            "[spec compile policy] %s DRAFT BLOCK: %d shapes (bs=%s, width=%d)",
            self.algo_label,
            len(shapes),
            list(self.b_bucketer.buckets),
            width,
        )

        pending: list[PendingCall] = []

        def run(shape: tuple[int, int]) -> str:
            bs, num_tokens = shape
            warmup = self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.TARGET_VERIFY,
                capture_hidden_mode=CaptureHiddenMode.NULL,
                # Same structure as TpuBlockSpecWorker's draft block input, so the trace matches.
                spec_info=make_draft_block_spec_info(
                    draft_token_num=width, device=self.model_runner.device
                ),
                attn_backend=draft_attn_backend,
            )
            if isinstance(warmup, PendingCall):
                pending.append(warmup)
            return f"bs={bs}, tokens={num_tokens}"

        self._precompile_shapes(f"{self.algo_label} DRAFT BLOCK", shapes, run)
        return pending

"""Speculative-decoding TPU compile policy."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
import tqdm

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.attention_rank_inputs import (
    get_attention_rank_tensors,
    use_attention_rank_tensors,
)
from sglang.srt.hardware_backend.tpu.compilation.compiled_forward_shim import (
    _HOST_PADDED_EXTEND_KEY,
    _REAL_B_KEY,
    _REAL_Q_KEY,
    _dp_attn_apply_bucket_metadata,
    _restore_forward_batch_inputs,
)
from sglang.srt.hardware_backend.tpu.compilation.decoder_compile_policy import (
    _COMPILE_BAR_FORMAT,
    DecoderTpuCompilePolicy,
    _get_hbm_usage_mb,
    _hbm_delta_str,
    _materialize_out,
)
from sglang.srt.hardware_backend.tpu.compilation.recompilation_checker import (
    install_recompilation_checker,
)
from sglang.srt.hardware_backend.tpu.compilation.spec_decode_compile_policy import (
    BaseSpecDecodeCompilePolicyMixin,
    _pad_first_dim,
)
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    get_server_return_hidden_states_mode,
)
from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.platforms import current_platform
from sglang.srt.runtime_context import get_parallel, get_spec, process_model_config
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import get_draft_input_from_target_hidden_dim
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType, SpecPaddingState
from sglang.srt.utils.common import ceil_align

logger = get_logger(__name__)


# Safe values for padded sampling rows.
_TEMPERATURE_FILL = 1.0
_TOP_K_FILL = 1
_TOP_P_FILL = 1.0
# -1 keeps padded results out of real request slots.
_DRAFT_TOKEN_FILL = 0
_RETRIEVE_INDEX_FILL = -1


@dataclass(frozen=True)
class _SpecVerifyPadding:
    """Original tensors needed to restore a logical verify batch."""

    state: SpecPaddingState
    batch: Any
    verify_input: SpecInput
    forward_mode: ForwardMode
    seq_lens: torch.Tensor
    draft_token: torch.Tensor
    retrieve_index: torch.Tensor
    temperatures: torch.Tensor | None
    top_ks: torch.Tensor | None
    top_ps: torch.Tensor | None


def _pad_rows(tensor: torch.Tensor, pad_rows: int, fill: float) -> torch.Tensor:
    """Append filler requests to a per-request tensor."""
    pad_spec = [0, 0] * (tensor.dim() - 1) + [0, pad_rows]
    return F.pad(tensor, pad_spec, value=fill)


def _deferred_padding_state(
    spec_info: SpecInput, logits_rows: int
) -> SpecPaddingState | None:
    """Report the padding still present in the target logits, if any."""
    state = spec_info.padding_state
    if state is None or not spec_info.keep_outputs_padded:
        return None
    if state.physical_num_tokens == state.logical_num_tokens:
        return None
    if logits_rows == state.logical_num_tokens:
        # Outputs were already restored before the deferred sampling hook.
        spec_info.keep_outputs_padded = False
        return None
    if logits_rows != state.physical_num_tokens:
        raise ValueError(
            "Deferred verify logits do not match the recorded padding: "
            f"state={state}, logits rows={logits_rows}."
        )
    return state


def _draft_extend_precompile_bs_buckets(
    bs_buckets: list[int], local_req_capacity: int
) -> list[int]:
    """Limit draft-extend warmup to locally reachable batch buckets."""
    valid = [bs for bs in bs_buckets if bs <= local_req_capacity]
    if not valid:
        raise ValueError(
            "DRAFT_EXTEND_V2 has no precompile batch bucket within the local "
            f"request-pool capacity {local_req_capacity}: buckets={bs_buckets}."
        )
    return valid


def _get_dp_peer_request_rows(
    global_num_tokens: list[int] | tuple[int, ...] | None,
    *,
    tokens_per_request: int,
    attention_tp_size: int,
) -> int:
    peer_tokens = max(global_num_tokens or (), default=0)
    if peer_tokens <= 0:
        return 0
    if tokens_per_request <= 0:
        raise ValueError(
            "DP speculative width requires a positive token count per request, "
            f"got {tokens_per_request}."
        )
    aligned_tokens = ceil_align(peer_tokens, max(attention_tp_size, 1))
    if aligned_tokens % tokens_per_request != 0:
        raise ValueError(
            "DP speculative width must contain whole requests: "
            f"aligned_tokens={aligned_tokens}, "
            f"tokens_per_request={tokens_per_request}."
        )
    return aligned_tokens // tokens_per_request


def _pad_eagle_draft_input(draft_input: EagleDraftInput, request_rows: int) -> int:
    current_rows = 0 if draft_input.topk_p is None else int(draft_input.topk_p.shape[0])
    if current_rows >= request_rows:
        return current_rows
    draft_input.topk_p = _pad_first_dim(draft_input.topk_p, request_rows)
    draft_input.topk_index = _pad_first_dim(draft_input.topk_index, request_rows)
    draft_input.hidden_states = _pad_first_dim(draft_input.hidden_states, request_rows)
    draft_input.bonus_tokens = _pad_first_dim(draft_input.bonus_tokens, request_rows)
    draft_input.draft_probs = _pad_first_dim(draft_input.draft_probs, request_rows)
    return current_rows


def fold_last_token_embeds(
    *, forward_batch: ForwardBatch, embed_tokens: torch.nn.Embedding
) -> torch.Tensor:
    """Return the target's multimodal embeds with each appended next token embedded.

    Feeding the result back as input_embeds is what keeps the draft from
    assembling it inside its graph, where reading forward_batch.mm_inputs makes
    Dynamo guard on each request slot's emptiness. Those guards track how the
    scheduler packs the batch, so no precompiled variant covers them.

    Returns:
      [num_tokens, hidden_size] embeds, already padded to the token bucket.
    """
    # Padding zero-fills the ghost request slots, whose last index would be -1.
    real_b = getattr(forward_batch, _REAL_B_KEY, None)
    last_indices = (forward_batch.extend_start_loc + forward_batch.extend_seq_lens - 1)[
        :real_b
    ].long()
    embeds = forward_batch.mm_input_embeds
    embeds[last_indices] = embed_tokens(forward_batch.input_ids[last_indices])
    return embeds


class Eagle3SpecDecodeCompilePolicyMixin(BaseSpecDecodeCompilePolicyMixin):
    """Add EAGLE3 warmup and batch shaping to a decoder compile policy."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        spec = get_spec()
        self.speculative_num_steps = spec.speculative_num_steps or 0
        self.speculative_algorithm = spec.speculative_algorithm
        self.speculative_eagle_topk = spec.speculative_eagle_topk or 1
        self.is_draft_worker = self.model_runner.is_draft_worker
        self._draft_eager_backbone_forward: Callable[..., Any] | None = None
        self._draft_eager_model_forward: Callable[..., Any] | None = None
        self._draft_eager_logits_forward: Callable[..., Any] | None = None

        if self.speculative_algorithm and self.speculative_algorithm != "EAGLE3":
            raise ValueError(
                f"TPU torch.compile precompilation only supports EAGLE3, "
                f"got {self.speculative_algorithm!r}."
            )

        # Draft tokens verified per request = the TARGET_VERIFY width. This
        # equals speculative_num_steps + 1 only for a topk=1 chain; use the
        # configured value so tree drafts (topk > 1) size correctly.
        self.draft_token_num = spec.speculative_num_draft_tokens or (
            self.speculative_num_steps + 1 if self.speculative_num_steps else 0
        )
        # Width of spec_info.hidden_states the EAGLE3 draft model expects.
        # 0 for non-draft workers (the target never reads it).
        self.eagle_hidden_size = (
            get_draft_input_from_target_hidden_dim(self.model_runner)
            if self.is_draft_worker
            else 0
        )
        # Only a multimodal target ever hands the draft input embeds, and the
        # draft's own model_config is the text-only EAGLE3 checkpoint, so read
        # the target's.
        self.target_is_multimodal = process_model_config().is_multimodal

    def install(self) -> bool:
        """Capture EAGLE's eager regions before the base policy replaces them."""
        if self.is_draft_worker and self.owner is not None:
            self._draft_eager_backbone_forward = self.owner.model.forward
            self._draft_eager_model_forward = self.owner.forward
            self._draft_eager_logits_forward = self.owner.logits_processor.forward
        return super().install()

    def compile_fused_draft_steps(self, fn: Callable[..., Any]) -> Callable[..., Any]:
        """Compile the multi-step EAGLE draft loop with the selected tracer."""
        if self.tracer != "direct":
            return torch.compile(
                fn,
                backend=self.backend or current_platform.get_compile_backend(),
                fullgraph=False,
                dynamic=False,
            )
        return self._direct_compile_fused_draft_steps(fn)

    def _direct_compile_fused_draft_steps(
        self, fn: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Trace the complete EAGLE draft loop without nested compiled calls."""
        if any(
            value is None
            for value in (
                self._draft_eager_backbone_forward,
                self._draft_eager_model_forward,
                self._draft_eager_logits_forward,
            )
        ):
            raise RuntimeError(
                "Direct fused-draft compilation requires an installed draft "
                "model compile policy"
            )

        from sglang.srt.hardware_backend.tpu.compilation.direct_compile import (
            direct_compile,
        )

        fn = torch.no_grad()(fn)
        backbone = self.owner.model
        logits_processor = self.owner.logits_processor

        def trace_fused_draft(
            attention_metadata: Any,
            forward_batch: ForwardBatch,
            *args: Any,
            attention_rank_tensors: tuple[torch.Tensor, ...],
            **kwargs: Any,
        ) -> Any:
            backend = get_forward_context().attn_backend
            previous_metadata = backend.forward_metadata
            previous_model_forward = self.owner.forward
            previous_backbone_forward = backbone.forward
            previous_logits_forward = logits_processor.forward
            backend.forward_metadata = attention_metadata
            # direct_compile executes this Python only while tracing. Restore
            # the eager regions so the captured loop contains model, logits,
            # sampling, and every draft step instead of trying to trace nested
            # PJRT/Dynamo executable launches.
            self.owner.forward = self._draft_eager_model_forward
            backbone.forward = self._draft_eager_backbone_forward
            logits_processor.forward = self._draft_eager_logits_forward
            try:
                with use_attention_rank_tensors(attention_rank_tensors):
                    return fn(forward_batch, *args, **kwargs)
            finally:
                logits_processor.forward = previous_logits_forward
                backbone.forward = previous_backbone_forward
                self.owner.forward = previous_model_forward
                backend.forward_metadata = previous_metadata

        def specialization_key(
            _attention_metadata: Any,
            forward_batch: ForwardBatch,
            *_args: Any,
            **_kwargs: Any,
        ) -> tuple[Any, ...]:
            return (
                forward_batch.forward_mode,
                forward_batch.capture_hidden_mode,
                forward_batch.batch_size,
            )

        compiled = direct_compile(
            trace_fused_draft,
            specialization_key=specialization_key,
            model=self.model_runner.model,
            model_extra_tensor_attrs=self.get_model_extra_tensor_attrs(),
            kv_pool=self.model_runner.token_to_kv_pool,
            ignored_argnames=self._ignored_argnames(),
        )

        @functools.wraps(fn)
        def call(*args: Any, **kwargs: Any) -> Any:
            backend = get_forward_context().attn_backend
            return compiled(
                backend.forward_metadata,
                *args,
                attention_rank_tensors=get_attention_rank_tensors(
                    torch.device(self.model_runner.device)
                ),
                **kwargs,
            )

        return call

    def prepare_spec_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Bucket a draft forward before constructing attention metadata."""
        mode = forward_batch.forward_mode
        if (
            is_dp_attention_enabled()
            and forward_batch.spec_info is not None
            and (mode.is_idle() or forward_batch.seq_lens.numel() == 0)
        ):
            self._prepare_empty_dp_draft_batch(forward_batch, mode)
            return

        if mode.is_idle():
            return
        if mode.is_decode():
            if forward_batch.spec_info is not None:
                forward_batch.spec_info.keep_outputs_padded = True
            self._pad_draft_decode(forward_batch)
        elif mode.is_draft_extend_v2():
            self._pad_draft_extend(forward_batch)

    def refresh_fused_draft_dp_metadata(self, forward_batch: ForwardBatch) -> None:
        """Publish the final padded draft shape before a fused direct call."""
        if not is_dp_attention_enabled() or forward_batch.forward_mode.is_idle():
            return
        padded_batch_size = int(forward_batch.batch_size)
        _dp_attn_apply_bucket_metadata(
            forward_batch, padded_batch_size, padded_batch_size
        )

    def _prepare_empty_dp_draft_batch(
        self, forward_batch: ForwardBatch, original_mode: ForwardMode
    ) -> None:
        """Give an empty DP draft phase the same physical shape as its peers."""
        if (
            original_mode.is_idle()
            and getattr(forward_batch, "_original_forward_mode", None) is None
        ):
            forward_batch._original_forward_mode = original_mode
        # Idle peers must execute the full draft-decode program.
        forward_batch.forward_mode = ForwardMode.DECODE
        rows = self._get_dp_peer_request_rows(forward_batch) or 1
        self._materialize_dp_idle_spec_batch(
            forward_batch, ForwardMode.DECODE, rows=rows
        )

        if isinstance(forward_batch.spec_info, EagleDraftInput):
            self.prepare_spec_draft_input(forward_batch, forward_batch.spec_info)
        forward_batch.spec_info.keep_outputs_padded = True
        self._pad_draft_decode(forward_batch)
        if original_mode.is_decode() or original_mode.is_idle():
            steps = max(self.speculative_num_steps, 1)
            cache_rows = rows * self.speculative_eagle_topk * steps
            if (
                isinstance(forward_batch.out_cache_loc, torch.Tensor)
                and forward_batch.out_cache_loc.shape[0] != cache_rows
            ):
                forward_batch.out_cache_loc = torch.zeros(
                    cache_rows,
                    dtype=forward_batch.out_cache_loc.dtype,
                    device=forward_batch.out_cache_loc.device,
                )

    def prepare_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Bucket once, then record the phase's logical and physical shapes."""
        spec_info = forward_batch.spec_info
        if spec_info is None:
            return super().prepare_forward_batch(forward_batch)

        logical_bs, logical_tokens = self._logical_spec_shape(
            forward_batch,
            fallback_batch_size=int(forward_batch.seq_lens.shape[0]),
            fallback_num_tokens=int(forward_batch.input_ids.shape[0]),
        )
        self._materialize_dp_idle_spec_batch(forward_batch, forward_batch.forward_mode)
        super().prepare_forward_batch(forward_batch)
        spec_info.record_padding(
            logical_batch_size=logical_bs,
            physical_batch_size=int(forward_batch.seq_lens.shape[0]),
            logical_num_tokens=logical_tokens,
            physical_num_tokens=int(forward_batch.input_ids.shape[0]),
        )

    def _logical_spec_shape(
        self,
        forward_batch: ForwardBatch,
        *,
        fallback_batch_size: int,
        fallback_num_tokens: int,
    ) -> tuple[int, int]:
        """Return the scheduler-logical shape for a padded spec phase."""
        spec_info = forward_batch.spec_info
        prior = getattr(spec_info, "padding_state", None)
        if prior is not None:
            return prior.logical_batch_size, prior.logical_num_tokens

        original_mode = getattr(forward_batch, "_original_forward_mode", None)
        if original_mode is not None and original_mode.is_idle():
            return 0, 0

        logical_bs = getattr(forward_batch, "_original_batch_size", None)
        if logical_bs is None:
            logical_bs = fallback_batch_size
        logical_bs = int(logical_bs)

        mode = forward_batch.forward_mode
        if mode.is_target_verify():
            tokens_per_req = int(getattr(spec_info, "draft_token_num", 0))
        elif (
            mode.is_draft_extend_v2() or mode.is_decode() and spec_info.is_draft_input()
        ):
            tokens_per_req = spec_info.num_tokens_per_req
        else:
            tokens_per_req = 0

        if tokens_per_req > 0:
            logical_tokens = logical_bs * tokens_per_req
        elif mode.is_extend():
            unpadded_tokens = getattr(forward_batch, "num_token_non_padded_cpu", None)
            logical_tokens = (
                int(unpadded_tokens)
                if unpadded_tokens is not None
                else fallback_num_tokens
            )
        else:
            logical_tokens = fallback_num_tokens
        return logical_bs, logical_tokens

    def _record_spec_execution_padding(
        self,
        forward_batch: ForwardBatch,
        *,
        fallback_logical_batch_size: int,
        physical_batch_size: int,
        fallback_logical_num_tokens: int,
        physical_num_tokens: int,
    ) -> None:
        """Record scheduler-logical to final-physical speculative padding."""
        spec_info = forward_batch.spec_info
        logical_bs, logical_tokens = self._logical_spec_shape(
            forward_batch,
            fallback_batch_size=fallback_logical_batch_size,
            fallback_num_tokens=fallback_logical_num_tokens,
        )
        spec_info.record_padding(
            logical_batch_size=logical_bs,
            physical_batch_size=physical_batch_size,
            logical_num_tokens=logical_tokens,
            physical_num_tokens=physical_num_tokens,
        )

    def _materialize_dp_idle_spec_batch(
        self,
        forward_batch: ForwardBatch,
        mode: ForwardMode,
        *,
        rows: int = 1,
        pre_write_seq_lens: bool = False,
    ) -> None:
        """Give an empty DP rank the physical rows required by its spec phase."""
        if (
            not is_dp_attention_enabled()
            or forward_batch.spec_info is None
            or forward_batch.seq_lens.numel() > 0
        ):
            return

        if mode.is_draft_extend_v2():
            self._materialize_dp_idle_draft_extend(
                forward_batch, pre_write_seq_lens=pre_write_seq_lens
            )
        elif mode.is_extend():
            max_tokens = max(forward_batch.global_num_tokens_cpu or (), default=0)
            if max_tokens > 0:
                self.materialize_dp_idle_extend(forward_batch, max_tokens)
        else:
            self._materialize_dp_idle_decode(forward_batch, rows=rows)

    @staticmethod
    def _materialize_dp_idle_decode(
        forward_batch: ForwardBatch, *, rows: int = 1
    ) -> None:
        """Represent an idle draft-decode rank with physical dummy sequences."""
        if forward_batch.seq_lens.numel() > 0:
            return

        rows = max(int(rows), 1)
        device = forward_batch.input_ids.device
        seq_dtype = forward_batch.seq_lens.dtype
        forward_batch.batch_size = rows
        forward_batch.seq_lens = torch.ones(rows, dtype=seq_dtype, device=device)
        forward_batch.seq_lens_cpu = torch.ones(rows, dtype=seq_dtype)
        forward_batch.seq_lens_sum = rows
        forward_batch.input_ids = torch.zeros(
            rows, dtype=forward_batch.input_ids.dtype, device=device
        )
        forward_batch.positions = torch.zeros(rows, dtype=torch.int64, device=device)
        if isinstance(forward_batch.out_cache_loc, torch.Tensor):
            forward_batch.out_cache_loc = torch.zeros(
                rows, dtype=forward_batch.out_cache_loc.dtype, device=device
            )
        if isinstance(forward_batch.req_pool_indices, torch.Tensor):
            pool_dtype = forward_batch.req_pool_indices.dtype
            forward_batch.req_pool_indices = torch.zeros(
                rows, dtype=pool_dtype, device=device
            )
            forward_batch.req_pool_indices_cpu = torch.zeros(rows, dtype=pool_dtype)

    @staticmethod
    def _materialize_dp_idle_draft_extend(
        forward_batch: ForwardBatch, *, pre_write_seq_lens: bool = False
    ) -> None:
        """Build fixed-width extend metadata for a logically idle draft rank."""
        tokens_per_req = int(getattr(forward_batch.spec_info, "num_tokens_per_req", 0))
        max_tokens = max(forward_batch.global_num_tokens_cpu or (), default=0)
        if tokens_per_req <= 0 or max_tokens <= 0:
            return
        if max_tokens % tokens_per_req != 0:
            raise ValueError(
                "DP idle DRAFT_EXTEND_V2 requires whole requests: "
                f"max_tokens={max_tokens}, tokens_per_req={tokens_per_req}."
            )

        batch_size = max_tokens // tokens_per_req
        device = forward_batch.input_ids.device
        seq_dtype = forward_batch.seq_lens.dtype
        seq_len = 0 if pre_write_seq_lens else tokens_per_req
        forward_batch.batch_size = batch_size
        forward_batch.extend_num_tokens = max_tokens
        forward_batch.seq_lens = torch.full(
            (batch_size,), seq_len, dtype=seq_dtype, device=device
        )
        forward_batch.seq_lens_cpu = torch.full((batch_size,), seq_len, dtype=seq_dtype)
        forward_batch.seq_lens_sum = seq_len * batch_size
        forward_batch.extend_seq_lens = torch.full(
            (batch_size,), tokens_per_req, dtype=torch.int32, device=device
        )
        forward_batch.extend_prefix_lens = torch.zeros(
            batch_size, dtype=torch.int32, device=device
        )
        forward_batch.extend_start_loc = torch.arange(
            0, max_tokens, tokens_per_req, dtype=torch.int32, device=device
        )
        forward_batch.extend_seq_lens_cpu = [tokens_per_req] * batch_size
        forward_batch.extend_prefix_lens_cpu = [0] * batch_size
        forward_batch.extend_logprob_start_lens_cpu = [0] * batch_size
        forward_batch.input_ids = torch.zeros(
            max_tokens, dtype=forward_batch.input_ids.dtype, device=device
        )
        forward_batch.positions = torch.arange(
            max_tokens, dtype=torch.int64, device=device
        )
        if isinstance(getattr(forward_batch, "out_cache_loc", None), torch.Tensor):
            forward_batch.out_cache_loc = torch.zeros(
                max_tokens,
                dtype=forward_batch.out_cache_loc.dtype,
                device=device,
            )
        if isinstance(forward_batch.req_pool_indices, torch.Tensor):
            pool_dtype = forward_batch.req_pool_indices.dtype
            forward_batch.req_pool_indices = torch.zeros(
                batch_size, dtype=pool_dtype, device=device
            )
            forward_batch.req_pool_indices_cpu = torch.zeros(
                batch_size, dtype=pool_dtype
            )

        spec_info = forward_batch.spec_info
        if isinstance(spec_info, EagleDraftExtendInput):
            spec_info.input_ids = forward_batch.input_ids
            spec_info.seq_lens = forward_batch.seq_lens
            spec_info.seq_lens_cpu = forward_batch.seq_lens_cpu
            spec_info.req_pool_indices = forward_batch.req_pool_indices
            spec_info.positions = forward_batch.positions
            if spec_info.hidden_states is not None:
                spec_info.hidden_states = spec_info.hidden_states.new_zeros(
                    (max_tokens, *spec_info.hidden_states.shape[1:])
                )
            if spec_info.num_correct_drafts is not None:
                spec_info.num_correct_drafts = spec_info.num_correct_drafts.new_zeros(
                    batch_size
                )
            if spec_info.num_accept_tokens is not None:
                spec_info.num_accept_tokens = spec_info.num_accept_tokens.new_full(
                    (batch_size,), tokens_per_req
                )
            spec_info.num_accept_tokens_cpu = [tokens_per_req] * batch_size
            if spec_info.bonus_tokens is not None:
                spec_info.bonus_tokens = spec_info.bonus_tokens.new_zeros(batch_size)

    def normalize_spec_forward_batch(self, forward_batch: ForwardBatch) -> None:
        """Materialize an idle post-verify draft before sequence bookkeeping."""
        if (
            not forward_batch.forward_mode.is_idle()
            or not is_dp_attention_enabled()
            or not isinstance(forward_batch.spec_info, EagleDraftExtendInput)
            or int(forward_batch.spec_info.num_tokens_per_req) <= 1
            or max(forward_batch.global_num_tokens_cpu or (), default=0) <= 0
        ):
            return

        forward_batch._original_forward_mode = forward_batch.forward_mode
        forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        forward_batch.spec_info.keep_outputs_padded = True
        # The caller applies the shared sequence-length update afterward.
        self._materialize_dp_idle_spec_batch(
            forward_batch,
            ForwardMode.DRAFT_EXTEND_V2,
            pre_write_seq_lens=True,
        )

    def _normalize_dp_idle_forward(self, forward_batch: ForwardBatch) -> None:
        """Match an idle speculative rank to the active rank's phase."""
        spec_info = forward_batch.spec_info
        if spec_info is None:
            return super()._normalize_dp_idle_forward(forward_batch)

        if forward_batch._original_forward_mode is None:
            forward_batch._original_forward_mode = forward_batch.forward_mode
        # Preserve physical rows until speculative bookkeeping completes.
        spec_info.keep_outputs_padded = True

        if getattr(spec_info, "is_verify_input", lambda: False)():
            forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
            return

        is_draft_extend = (
            getattr(spec_info, "spec_input_type", None)
            == SpecInputType.EAGLE_DRAFT_EXTEND
        )
        if is_draft_extend and int(spec_info.num_tokens_per_req) == 1:
            # Draft prefill is an ordinary EXTEND.
            forward_batch.forward_mode = ForwardMode.EXTEND
        elif is_draft_extend:
            forward_batch.forward_mode = ForwardMode.DRAFT_EXTEND_V2
        else:
            forward_batch.forward_mode = ForwardMode.DECODE

        self._materialize_dp_idle_spec_batch(forward_batch, forward_batch.forward_mode)

    def _get_dp_peer_request_rows(self, forward_batch: ForwardBatch) -> int:
        """Return the compile-bucket request width shared by all DP ranks."""
        if not is_dp_attention_enabled() or self.b_bucketer is None:
            return 0
        request_rows = _get_dp_peer_request_rows(
            forward_batch.global_num_tokens_cpu,
            tokens_per_request=self.speculative_eagle_topk,
            attention_tp_size=get_parallel().attn_tp_size,
        )
        return self.b_bucketer.next_bucket(request_rows) if request_rows else 0

    def get_dp_padded_token_counts(
        self,
        forward_batch: ForwardBatch,
        model_runner: Any,
        global_num_tokens: list[int],
        global_num_tokens_for_logprob: list[int],
        *,
        attn_tp_size: int,
        cp_align_size: int,
    ) -> tuple[list[int], list[int]]:
        if not is_dp_attention_enabled() or forward_batch.spec_info is None:
            return super().get_dp_padded_token_counts(
                forward_batch,
                model_runner,
                global_num_tokens,
                global_num_tokens_for_logprob,
                attn_tp_size=attn_tp_size,
                cp_align_size=cp_align_size,
            )
        fixed_width = (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        )
        bucket_width = fixed_width or (
            model_runner.is_draft_worker
            and forward_batch.forward_mode.is_decode()
            and forward_batch.spec_info.is_draft_input()
        )
        if not fixed_width:
            (
                global_num_tokens,
                global_num_tokens_for_logprob,
            ) = super().get_dp_padded_token_counts(
                forward_batch,
                model_runner,
                global_num_tokens,
                global_num_tokens_for_logprob,
                attn_tp_size=attn_tp_size,
                cp_align_size=cp_align_size,
            )
        if bucket_width:
            shared_width = self._bucket_dp_sync_token_width(
                forward_batch, max(global_num_tokens)
            )
            global_num_tokens = [shared_width] * len(global_num_tokens)
        return global_num_tokens, global_num_tokens_for_logprob

    def prepare_spec_draft_input(
        self, forward_batch: ForwardBatch, draft_input: EagleDraftInput
    ) -> None:
        request_rows = self._get_dp_peer_request_rows(forward_batch)
        if request_rows <= 0:
            return
        seed_rows = _pad_eagle_draft_input(draft_input, request_rows)
        if seed_rows >= request_rows:
            return
        original_mode = getattr(forward_batch, "_original_forward_mode", None)
        logical_rows = (
            0 if original_mode is not None and original_mode.is_idle() else seed_rows
        )
        draft_input.keep_outputs_padded = True
        draft_input.record_padding(
            logical_batch_size=logical_rows,
            physical_batch_size=request_rows,
            logical_num_tokens=logical_rows * self.speculative_eagle_topk,
            physical_num_tokens=request_rows * self.speculative_eagle_topk,
        )

    def _bucket_dp_sync_token_width(
        self, forward_batch: ForwardBatch, num_tokens: int
    ) -> int:
        """Round a speculative DP width up to a runnable TPU graph shape."""
        if self.b_bucketer is None:
            return num_tokens
        spec_info = forward_batch.spec_info
        tokens_per_req = spec_info.num_tokens_per_req
        if tokens_per_req <= 0 or num_tokens <= 0:
            return num_tokens
        physical_tokens = int(forward_batch.input_ids.shape[0])
        req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
        if isinstance(req_pool_indices, torch.Tensor):
            physical_tokens = max(
                physical_tokens,
                int(req_pool_indices.shape[0]) * tokens_per_req,
            )
        required_tokens = max(num_tokens, physical_tokens)
        rows = (required_tokens + tokens_per_req - 1) // tokens_per_req
        return self.b_bucketer.next_bucket(rows) * tokens_per_req

    def _dp_peer_draft_extend_rows(self, forward_batch: ForwardBatch) -> int:
        """Return the shared post-verify draft-extend request width."""
        if not is_dp_attention_enabled():
            return 0
        max_tokens = max(forward_batch.global_num_tokens_cpu or (), default=0)
        tokens_per_req = self.draft_token_num
        if max_tokens <= 0 or tokens_per_req <= 0:
            return 0
        return -(-max_tokens // tokens_per_req)

    def precompile_draft(
        self,
        draft_attn_backend: Any,
        draft_extend_attn_backend: Any,
        fused_draft_forward: Callable[[ForwardBatch], Any] | None = None,
    ) -> None:
        """Compile DRAFT-model forward families at server warmup."""
        if not self.is_draft_worker or self.b_bucketer is None:
            return
        overall = time.perf_counter()
        self._precompile_draft_prefill_extend()
        self._precompile_draft_decode(draft_attn_backend)
        if fused_draft_forward is not None:
            self._precompile_fused_draft_decode(draft_attn_backend, fused_draft_forward)
        self._precompile_draft_extend(draft_extend_attn_backend)
        logger.info(
            "[spec compile policy] DRAFT precompile done in %.1fs.",
            time.perf_counter() - overall,
        )
        # Include draft warmup in the shared target/draft baseline. Later TPU
        # speculative-op and KV-offload warmup refresh it when they finish.
        install_recompilation_checker(
            policies=(self,),
            model_label=f"draft {type(self.model_runner.model).__name__}",
        )

    def _precompile_shapes(
        self, desc: str, items: list[Any], run: Callable[[Any], str]
    ) -> None:
        compile_range = (
            tqdm.tqdm(items, desc=f"Compiling {desc}", bar_format=_COMPILE_BAR_FORMAT)
            if get_parallel().tp_rank == 0
            else items
        )
        for item in compile_range:
            hbm_before = _get_hbm_usage_mb()
            start = time.perf_counter()
            shape_desc = run(item)
            if get_parallel().tp_rank == 0:
                elapsed = time.perf_counter() - start
                hbm_str = _hbm_delta_str(hbm_before, _get_hbm_usage_mb())
                compile_range.set_description(
                    f"Compiled {desc} ({shape_desc}) in {elapsed:.1f}s{hbm_str}"
                )

    def _precompile_draft_prefill_extend(self) -> None:
        """Compile draft prefill EXTEND(LAST) token buckets at the batch cap.

        Runtime EXTEND padding uses this same fixed batch cap to avoid compiling
        the token-bucket by batch-bucket cross product.
        """
        assert self.t_bucketer is not None
        assert self.b_bucketer is not None
        bs = self._extend_compile_cap()
        # A multimodal prefill hands the draft the target's embeds, which the
        # draft reads instead of calling embed_tokens -- a separate graph. A
        # multimodal target still serves text-only requests, so both stay live.
        variants = ("plain", "mm_embeds") if self.target_is_multimodal else ("plain",)
        valid_shapes = [
            (t, bs, variant)
            for t in self.t_bucketer.buckets
            if t >= bs
            for variant in variants
        ]
        logger.info(
            "[spec compile policy] DRAFT EXTEND(LAST): %d shapes "
            "(bs=%d, token_buckets=%s, variants=%s)",
            len(valid_shapes),
            bs,
            list(self.t_bucketer.buckets),
            list(variants),
        )

        def run(shape: tuple[int, int, str]) -> str:
            num_tokens, bs, variant = shape
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.EXTEND,
                variant=variant,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_info=self._make_draft_prefill_spec_info(),
            )
            return f"tokens={num_tokens}, bs={bs}, variant={variant}"

        self._precompile_shapes("DRAFT EXTEND(LAST)", valid_shapes, run)

    def _precompile_draft_decode(self, draft_attn_backend: Any) -> None:
        """Compile the multi-step draft DECODE(LAST) across the bs buckets."""
        if draft_attn_backend is None:
            return
        assert self.b_bucketer is not None
        step_backends = list(draft_attn_backend.attn_backends)
        bs_buckets = list(self.b_bucketer.buckets)
        shapes = [(bs, step) for bs in bs_buckets for step in range(len(step_backends))]
        logger.info(
            "[spec compile policy] DRAFT DECODE: %d shapes (bs=%s, steps=%d, topk=%d)",
            len(shapes),
            bs_buckets,
            len(step_backends),
            self.speculative_eagle_topk,
        )

        def run(shape: tuple[int, int]) -> str:
            bs, step = shape
            self._run_one(
                num_tokens=bs * self.speculative_eagle_topk,
                bs=bs,
                mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_info=self._make_draft_decode_spec_info(),
                attn_backend=step_backends[step],
            )
            return f"bs={bs}, step={step}"

        self._precompile_shapes("DRAFT DECODE", shapes, run)

    def _precompile_fused_draft_decode(
        self,
        draft_attn_backend: Any,
        fused_draft_forward: Callable[[ForwardBatch], Any],
    ) -> None:
        """Compile the fused draft loop for every decode batch bucket."""
        assert self.b_bucketer is not None
        bs_buckets = list(self.b_bucketer.buckets)
        logger.info(
            "[spec compile policy] FUSED DRAFT: %d shapes (bs=%s)",
            len(bs_buckets),
            bs_buckets,
        )

        def run(bs: int) -> str:
            args, kwargs = self._make_warmup_inputs(
                num_tokens=bs * self.speculative_eagle_topk,
                bs=bs,
                mode=ForwardMode.DECODE,
                capture_hidden_mode=CaptureHiddenMode.LAST,
                spec_info=self._make_fused_draft_decode_spec_info(bs),
            )
            forward_batch = kwargs.get("forward_batch", args[2])
            # A live fused draft reserves one KV output slot per request and
            # draft step. Generic DECODE warmup only allocates one per request.
            forward_batch.out_cache_loc = forward_batch.out_cache_loc.repeat_interleave(
                self.speculative_num_steps
            )
            self._seed_dummy_req_to_token(forward_batch)
            if is_dp_attention_enabled():
                self._set_dp_attention_metadata(
                    forward_batch,
                    bs * self.speculative_eagle_topk,
                    bs,
                    ForwardMode.DECODE,
                )
            # Serving buckets and marks the draft input before attention
            # metadata is built. Keep padded outputs at their physical width
            # for every model call in the fused loop.
            self.prepare_spec_forward_batch(forward_batch)
            self.model_runner.model.prepare_forward_batch(forward_batch)
            draft_attn_backend.init_forward_metadata(forward_batch)
            forward_batch.mark_forward_metadata_ready()
            # The live draft path has consumed this seed before entering the
            # fused loop. Match its state so Dynamo reuses the warmup graph.
            forward_batch.input_ids = None
            # The live fused loop bypasses ModelRunner.forward, so its batch carries
            # None here; direct_compile pins each input's type at trace time.
            forward_batch.num_token_non_padded = None
            with torch.no_grad():
                out = fused_draft_forward(forward_batch)
                _materialize_out(
                    out,
                    draft_attn_backend.attn_backends[0],
                    self.model_runner.token_to_kv_pool,
                )
            return f"bs={bs}"

        self._precompile_shapes("FUSED DRAFT", bs_buckets, run)

    def _precompile_draft_extend(self, attn_backend: Any) -> None:
        """Compile DRAFT_EXTEND_V2(FULL) across the bs buckets."""
        assert self.b_bucketer is not None
        all_bs_buckets = list(self.b_bucketer.buckets)
        local_req_capacity = int(self.model_runner.req_to_token_pool.size)
        bs_buckets = _draft_extend_precompile_bs_buckets(
            all_bs_buckets, local_req_capacity
        )
        tokens_per_req = self.draft_token_num
        logger.info(
            "[spec compile policy] DRAFT_EXTEND_V2(FULL): %d shapes "
            "(bs=%s, tokens_per_req=%d, local_req_capacity=%d)",
            len(bs_buckets),
            bs_buckets,
            tokens_per_req,
            local_req_capacity,
        )

        def run(bs: int) -> str:
            num_tokens = bs * tokens_per_req
            self._run_one(
                num_tokens=num_tokens,
                bs=bs,
                mode=ForwardMode.DRAFT_EXTEND_V2,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                spec_info=self._make_draft_extend_spec_info(),
                attn_backend=attn_backend,
            )
            return f"bs={bs}, tokens={num_tokens}"

        self._precompile_shapes("DRAFT_EXTEND_V2(FULL)", bs_buckets, run)

    def _make_draft_prefill_spec_info(self) -> EagleDraftExtendInput:
        """EagleDraftExtendInput stub for draft prefill EXTEND(LAST)."""
        spec_info = EagleDraftExtendInput.create_idle_input(
            device=self.model_runner.device,
            hidden_size=self.eagle_hidden_size or 1,
            dtype=self.model_runner.dtype,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        spec_info.num_tokens_per_req = 1
        spec_info.num_tokens_for_logprob_per_req = 1
        return spec_info

    def _make_draft_decode_spec_info(self) -> EagleDraftInput:
        """EagleDraftInput stub for multi-step draft DECODE(LAST)."""
        model_config = self.model_runner.model_config
        hidden_size = (
            getattr(model_config, "spec_hidden_size", 0) or model_config.hidden_size
        )
        spec_info = self._make_idle_draft_input(
            hidden_size=hidden_size,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        spec_info.num_tokens_per_req = self.speculative_eagle_topk
        spec_info.num_tokens_for_logprob_per_req = self.speculative_eagle_topk
        return spec_info

    def _make_fused_draft_decode_spec_info(self, bs: int) -> EagleDraftInput:
        """Build batch-shaped EAGLE seeds consumed by fused loop."""
        spec_info = self._make_draft_decode_spec_info()
        device = self.model_runner.device
        spec_info.topk_p = torch.ones(
            (bs, self.speculative_eagle_topk), device=device, dtype=torch.float32
        )
        spec_info.topk_index = torch.zeros(
            (bs, self.speculative_eagle_topk), device=device, dtype=torch.int64
        )
        # FutureMap stores serving tokens in its int64 relay buffer. The fused
        # draft graph also builds the verify chain from these seeds.
        spec_info.bonus_tokens = torch.zeros(bs, device=device, dtype=torch.int64)
        spec_info.hidden_states = torch.zeros(
            (bs, spec_info.hidden_states.shape[1]),
            device=device,
            dtype=self.model_runner.dtype,
        )
        return spec_info

    def _make_draft_extend_spec_info(self) -> EagleDraftExtendInput:
        """EagleDraftExtendInput stub for DRAFT_EXTEND_V2 precompilation."""
        spec_info = EagleDraftExtendInput.create_idle_input(
            device=self.model_runner.device,
            hidden_size=self.eagle_hidden_size or 1,
            dtype=self.model_runner.dtype,
            capture_hidden_mode=CaptureHiddenMode.LAST,
        )
        spec_info.num_tokens_per_req = self.draft_token_num
        spec_info.num_tokens_for_logprob_per_req = self.draft_token_num
        return spec_info

    def make_null_spec_info(self) -> EagleDraftInput:
        """Build a minimal idle spec_info stub for non-spec forward modes."""
        return self._make_idle_draft_input(
            hidden_size=self.eagle_hidden_size or 1,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

    def _make_null_spec_info(self) -> EagleDraftInput:
        return self.make_null_spec_info()

    def _make_idle_draft_input(
        self, *, hidden_size: int, capture_hidden_mode: CaptureHiddenMode
    ) -> EagleDraftInput:
        """Build a warmup stub."""
        device = self.model_runner.device
        return EagleDraftInput(
            bonus_tokens=torch.empty(0, device=device, dtype=torch.int64),
            hidden_states=torch.empty(
                (0, hidden_size), device=device, dtype=self.model_runner.dtype
            ),
            topk_p=torch.empty(
                (0, self.speculative_eagle_topk),
                device=device,
                dtype=torch.float32,
            ),
            topk_index=torch.empty(
                (0, self.speculative_eagle_topk),
                device=device,
                dtype=torch.int64,
            ),
            draft_probs=None,
            capture_hidden_mode=capture_hidden_mode,
        )

    def make_verify_spec_info(self) -> EagleVerifyInput:
        """Build an EagleVerifyInput for TARGET_VERIFY precompilation."""
        return EagleVerifyInput.create_idle_input(
            topk=self.speculative_eagle_topk,
            spec_steps=self.speculative_num_steps,
            num_verify_tokens=self.draft_token_num,
            device=self.model_runner.device,
        )

    def _make_verify_spec_info(self) -> EagleVerifyInput:
        return self.make_verify_spec_info()

    def _dummy_kv_slot(self) -> int:
        return 1

    def _dummy_tokens_per_req(
        self, mode: ForwardMode, spec_info: SpecInput | None
    ) -> int:
        if mode.is_target_verify() and spec_info is not None:
            return spec_info.draft_token_num
        if mode.is_draft_extend_v2() and self.draft_token_num:
            return self.draft_token_num
        return 1

    def _dummy_mm_inputs(self, bs: int) -> list[None]:
        return [None] * bs

    def _make_warmup_inputs(
        self,
        *,
        num_tokens: int,
        bs: int,
        mode: ForwardMode,
        variant: str = "plain",
        capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL,
        spec_info: SpecInput | None = None,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Attach the target embeds a multimodal draft extend arrives with."""
        args, kwargs = super()._make_warmup_inputs(
            num_tokens=num_tokens,
            bs=bs,
            mode=mode,
            variant=variant,
            capture_hidden_mode=capture_hidden_mode,
            spec_info=spec_info,
        )
        if variant == "mm_embeds":
            forward_batch = kwargs.get("forward_batch", args[2])
            embed_tokens = self.model_runner.model.get_input_embeddings()
            forward_batch.mm_input_embeds = torch.zeros(
                (num_tokens, embed_tokens.embedding_dim),
                dtype=embed_tokens.weight.dtype,
                device=self.model_runner.device,
            )
        return args, kwargs

    def _dummy_out_cache_loc_dtype(self, mode: ForwardMode) -> torch.dtype:
        if (
            mode.is_target_verify()
            or mode.is_draft_extend_v2()
            or (self.is_draft_worker and mode.is_decode())
        ):
            return torch.int64
        return super()._dummy_out_cache_loc_dtype(mode)

    def _seed_dummy_req_to_token(self, forward_batch: ForwardBatch) -> None:
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        if forward_batch.forward_mode.is_draft_extend_v2():
            rows = forward_batch.req_pool_indices.to("cpu")
            ctx = int(forward_batch.seq_lens[0].item())
            cols = torch.arange(ctx, dtype=req_to_token.dtype)
            vals = rows.to(req_to_token.dtype).view(-1, 1) * ctx + cols.view(1, -1)
            req_to_token[rows, :ctx] = vals
        else:
            num_tokens = int(forward_batch.input_ids.shape[0])
            req_to_token[1, :num_tokens] = 1

    def finalize_forward_batch(self, outputs: Any, forward_batch: ForwardBatch) -> Any:
        """Slice padded spec outputs back to real request shapes."""
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        real_b = getattr(forward_batch, _REAL_B_KEY, None)
        if real_q is None or real_b is None:
            return outputs

        next_token_logits = getattr(outputs, "next_token_logits", None)
        hidden_states = getattr(outputs, "hidden_states", None)
        if next_token_logits is None and hidden_states is None:
            # Inner decoder boundaries return raw tensors/tuples. Delegate
            # cooperatively so a model-specific policy later in the MRO (for
            # example Qwen3-VL) can slice those outputs and restore metadata.
            return super().finalize_forward_batch(outputs, forward_batch)
        original_mode = getattr(forward_batch, "_original_forward_mode", None)
        preserve_idle_outputs = original_mode is not None and original_mode.is_idle()
        spec_info = forward_batch.spec_info
        forward_mode = getattr(forward_batch, "forward_mode", None)
        preserve_deferred_verify_outputs = (
            forward_mode is not None
            and forward_mode.is_target_verify()
            and spec_info is not None
            and spec_info.keep_outputs_padded
        )
        preserve_padded_outputs = (
            preserve_idle_outputs or preserve_deferred_verify_outputs
        )
        if (
            not preserve_padded_outputs
            and next_token_logits is not None
            and real_b < next_token_logits.shape[0]
        ):
            # TARGET_VERIFY outputs bs * draft_token_num logits (one per draft
            # token across all requests); slicing to real_b would drop verify
            # tokens, so use real_q instead.
            logit_rows = (
                real_q
                if (forward_mode is not None and forward_mode.is_target_verify())
                else real_b
            )
            if logit_rows < next_token_logits.shape[0]:
                outputs.next_token_logits = next_token_logits[:logit_rows]

        # A host-padded target prefill's per-token hidden states only feed the
        # draft extend, which runs at the same token bucket: trimming here made
        # the draft pad them back, one program per prompt length. Per-request
        # (LAST) captures still trim; they become the next cycle's draft seed.
        keep_padded_hidden = (
            getattr(forward_batch, _HOST_PADDED_EXTEND_KEY, False)
            and forward_batch.capture_hidden_mode == CaptureHiddenMode.FULL
            and get_server_return_hidden_states_mode() == CaptureHiddenMode.NULL
        )
        if (
            hidden_states is not None
            and not preserve_padded_outputs
            and not keep_padded_hidden
        ):
            # Capture-LAST (draft extend) emits one hidden row per
            # sequence (batch-dim), so slice to real_b. Every other capture mode
            # emits one row per token (real_q).
            cap = getattr(forward_batch, "capture_hidden_mode", None)
            n_real = real_b if (cap is not None and cap.is_last()) else real_q
            if n_real < hidden_states.shape[0]:
                outputs.hidden_states = hidden_states[:n_real]

        _restore_forward_batch_inputs(forward_batch, real_q, real_b)
        if spec_info is not None and not preserve_padded_outputs:
            spec_info.keep_outputs_padded = False
        return outputs

    def _pad_draft_decode(self, forward_batch: ForwardBatch) -> None:
        """Pad the speculative draft-decode batch to a torch.compile bs bucket."""
        b_bucketer = self.b_bucketer
        if b_bucketer is None:
            return
        real_bs = int(forward_batch.seq_lens.shape[0])
        # Draft seeds and later model outputs must use the same DP width.
        pad_b = max(
            b_bucketer.next_bucket(real_bs),
            self._get_dp_peer_request_rows(forward_batch),
        )
        if pad_b == real_bs:
            return

        forward_batch.seq_lens = _pad_first_dim(forward_batch.seq_lens, pad_b)
        seq_lens_cpu = forward_batch.seq_lens_cpu
        if isinstance(seq_lens_cpu, torch.Tensor):
            forward_batch.seq_lens_cpu = _pad_first_dim(seq_lens_cpu, pad_b)
        forward_batch.req_pool_indices = _pad_first_dim(
            forward_batch.req_pool_indices, pad_b
        )
        forward_batch.req_pool_indices_cpu = _pad_first_dim(
            forward_batch.req_pool_indices_cpu, pad_b
        )
        forward_batch.positions = _pad_first_dim(
            forward_batch.positions,
            forward_batch.positions.shape[0] // real_bs * pad_b,
        )
        mrope_positions = forward_batch.mrope_positions
        if (
            mrope_positions is not None
            and mrope_positions.shape[-1] < forward_batch.positions.shape[0]
        ):
            pad_tokens = forward_batch.positions.shape[0] - mrope_positions.shape[-1]
            forward_batch.mrope_positions = torch.cat(
                [
                    mrope_positions,
                    mrope_positions.new_zeros(
                        (*mrope_positions.shape[:-1], pad_tokens)
                    ),
                ],
                dim=-1,
            )
        forward_batch.out_cache_loc = _pad_first_dim(
            forward_batch.out_cache_loc,
            forward_batch.out_cache_loc.shape[0] // real_bs * pad_b,
        )

        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if isinstance(mm_inputs, list) and len(mm_inputs) < pad_b:
            forward_batch.mm_inputs = mm_inputs + [None] * (pad_b - len(mm_inputs))

        # EagleDraftInput seeds consumed by select_top_k_tokens.
        spec_info = forward_batch.spec_info
        if isinstance(spec_info, EagleDraftInput):
            _pad_eagle_draft_input(spec_info, pad_b)
            logical_bs, logical_tokens = self._logical_spec_shape(
                forward_batch,
                fallback_batch_size=real_bs,
                fallback_num_tokens=real_bs * self.speculative_eagle_topk,
            )
            spec_info.record_padding(
                logical_batch_size=logical_bs,
                physical_batch_size=pad_b,
                logical_num_tokens=logical_tokens,
                physical_num_tokens=pad_b * self.speculative_eagle_topk,
            )

        forward_batch.batch_size = pad_b
        setattr(forward_batch, _REAL_B_KEY, real_bs)

    def finalize_spec_forward_batch(
        self, outputs: Any, forward_batch: ForwardBatch
    ) -> Any:
        """Trim draft outputs and restore the logical batch state."""
        real_bs = getattr(forward_batch, _REAL_B_KEY, None)
        spec_info = forward_batch.spec_info
        if spec_info is not None and spec_info.keep_outputs_padded:
            # Keep the shared physical width through the complete draft loop.
            return outputs
        if real_bs is None:
            if spec_info is not None:
                spec_info.keep_outputs_padded = False
            return outputs
        forward_batch.batch_size = real_bs
        if spec_info is not None:
            spec_info.keep_outputs_padded = False
        # draft_probs is absent when rejection sampling is disabled.
        return tuple(t[:real_bs] if t is not None else None for t in outputs)

    def _pad_draft_extend(self, forward_batch: ForwardBatch) -> None:
        """Bucket V2 draft-extend before attention metadata is constructed."""
        if self.b_bucketer is None:
            return
        draft_extend_input = forward_batch.spec_info
        real_bs = int(forward_batch.seq_lens.shape[0])
        real_tokens = int(forward_batch.input_ids.shape[0])
        pad_b = self.b_bucketer.next_bucket(
            max(real_bs, self._dp_peer_draft_extend_rows(forward_batch))
        )
        pad_tokens = pad_b * self.draft_token_num
        # Padding must not disguise missing active-row query counts.
        query_lens_cpu = forward_batch.extend_seq_lens_cpu
        if query_lens_cpu is None or len(query_lens_cpu) < real_bs:
            forward_batch.extend_seq_lens_cpu = None
        if pad_b != real_bs:
            # Draft extend has a fixed width, so it needs no token bucket.
            # This also grows its owned token- and request-axis tensors.
            forward_batch._pad_inputs_to_size(
                model_runner=self.model_runner,
                num_tokens=pad_tokens,
                bs=pad_b,
            )
            # Pad any remaining extend metadata to the compiled batch size.
            # Outputs for padded requests are discarded.
            for field in ("extend_seq_lens", "extend_prefix_lens", "extend_start_loc"):
                value = getattr(forward_batch, field, None)
                if isinstance(value, torch.Tensor) and value.shape[0] < pad_b:
                    setattr(
                        forward_batch, field, F.pad(value, (0, pad_b - value.shape[0]))
                    )
            forward_batch.batch_size = pad_b
            # Intentionally leave _REAL_Q_KEY unset. draft extend consumes the
            # output later using its own select-index gather instead of generic
            # prefix unpadding.
            setattr(forward_batch, _REAL_B_KEY, real_bs)

        # Prefill may leave CPU metadata padded to a different batch size.
        # Draft extend skips model.prepare_forward_batch, so resize these lists
        # here even when the current batch already matches a bucket.
        for field in (
            "extend_seq_lens_cpu",
            "extend_prefix_lens_cpu",
            "extend_logprob_start_lens_cpu",
        ):
            value = getattr(forward_batch, field, None)
            if value is None:
                if field == "extend_seq_lens_cpu":
                    continue
                value = []
            elif isinstance(value, torch.Tensor):
                value = value.tolist()
            setattr(forward_batch, field, (value + [0] * pad_b)[:pad_b])

        mm_inputs = getattr(forward_batch, "mm_inputs", None)
        if isinstance(mm_inputs, list):
            forward_batch.mm_inputs = (mm_inputs + [None] * pad_b)[:pad_b]

        if isinstance(draft_extend_input, EagleDraftExtendInput):
            self._record_spec_execution_padding(
                forward_batch,
                fallback_logical_batch_size=real_bs,
                physical_batch_size=pad_b,
                fallback_logical_num_tokens=real_tokens,
                physical_num_tokens=pad_tokens,
            )

    def should_defer_verify_output_unpadding(
        self, batch: Any, verify_input: SpecInput
    ) -> bool:
        """Decide whether TPU can defer target-output unpadding until sampling."""
        idle_with_active_dp_peer = (
            batch.forward_mode.is_idle()
            and is_dp_attention_enabled()
            and max(getattr(batch, "global_num_tokens_cpu", None) or (), default=0) > 0
        )
        verify_input.keep_outputs_padded = (
            not batch.forward_mode.is_idle() or idle_with_active_dp_peer
        )
        return verify_input.keep_outputs_padded

    def prepare_spec_verify(
        self,
        batch: Any,
        verify_input: SpecInput,
        next_token_logits: torch.Tensor,
    ) -> _SpecVerifyPadding | None:
        """Temporarily pad scheduler inputs to match deferred target logits."""
        state = _deferred_padding_state(verify_input, int(next_token_logits.shape[0]))
        if state is None:
            return None

        pad_bs = state.physical_batch_size - state.logical_batch_size
        sampling_info = batch.sampling_info

        # Preserve original tensors so aliases can be restored exactly.
        padding = _SpecVerifyPadding(
            state=state,
            batch=batch,
            verify_input=verify_input,
            forward_mode=batch.forward_mode,
            seq_lens=batch.seq_lens,
            draft_token=verify_input.draft_token,
            retrieve_index=verify_input.retrieve_index,
            temperatures=sampling_info.temperatures,
            top_ks=sampling_info.top_ks,
            top_ps=sampling_info.top_ps,
        )

        # Build every replacement before publishing any of them, so an error
        # cannot leave the caller with a partially padded batch.
        seq_lens = _pad_rows(padding.seq_lens, pad_bs, 0)
        draft_token = _pad_rows(
            padding.draft_token,
            pad_bs * verify_input.draft_token_num,
            _DRAFT_TOKEN_FILL,
        )
        retrieve_index = _pad_rows(padding.retrieve_index, pad_bs, _RETRIEVE_INDEX_FILL)
        temperatures = (
            _pad_rows(padding.temperatures, pad_bs, _TEMPERATURE_FILL)
            if padding.temperatures is not None
            else None
        )
        top_ks = (
            _pad_rows(padding.top_ks, pad_bs, _TOP_K_FILL)
            if padding.top_ks is not None
            else None
        )
        top_ps = (
            _pad_rows(padding.top_ps, pad_bs, _TOP_P_FILL)
            if padding.top_ps is not None
            else None
        )

        # Publish padded tensors only while eagle_sample runs.
        # A logically idle DP peer must take the same sampling/bookkeeping
        # path as its active peer after its tensors have been materialized.
        # Restore IDLE before scheduler bookkeeping observes the batch again.
        if batch.forward_mode.is_idle():
            batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.seq_lens = seq_lens
        verify_input.draft_token = draft_token
        verify_input.retrieve_index = retrieve_index
        sampling_info.temperatures = temperatures
        sampling_info.top_ks = top_ks
        sampling_info.top_ps = top_ps
        # Penalties, logit bias, and grammar data stay logical and are applied
        # only to the logical logits prefix. retrieve_next_token /
        # retrieve_next_sibling are sgl_kernel-only arguments, and draft_probs
        # is accepted but ignored by the torch rejection sampler.
        return padding

    def finalize_spec_verify(
        self,
        padding: _SpecVerifyPadding | None,
        verify_result: tuple[
            torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
        ],
        bookkeeping_result: tuple[torch.Tensor | None, torch.Tensor | None],
    ) -> tuple[tuple[Any, Any, Any], tuple[Any, Any]]:
        """Restore logical inputs and trim all padded verification outputs."""
        if padding is None:
            return verify_result, bookkeeping_result
        predict, accept_lens, accept_index = verify_result
        new_seq_lens, bonus_tokens = bookkeeping_result
        sampling_info = padding.batch.sampling_info
        padding.batch.forward_mode = padding.forward_mode
        padding.batch.seq_lens = padding.seq_lens
        padding.verify_input.draft_token = padding.draft_token
        padding.verify_input.retrieve_index = padding.retrieve_index
        padding.verify_input.keep_outputs_padded = False
        sampling_info.temperatures = padding.temperatures
        sampling_info.top_ks = padding.top_ks
        sampling_info.top_ps = padding.top_ps
        state = padding.state
        logical_bs = state.logical_batch_size
        # The scheduler copies predict and accept_lens to the host from the
        # result-copy thread, after the next step is already queued. Copying a
        # slice first needs a device program to make it contiguous, which would
        # then wait behind that whole step; materialize both here instead.
        return (
            (
                (
                    predict[: state.logical_num_tokens].clone()
                    if predict is not None
                    else None
                ),
                accept_lens[:logical_bs].clone() if accept_lens is not None else None,
                accept_index[:logical_bs] if accept_index is not None else None,
            ),
            (
                new_seq_lens[:logical_bs] if new_seq_lens is not None else None,
                bonus_tokens[:logical_bs] if bonus_tokens is not None else None,
            ),
        )


class SpecDecoderTpuCompilePolicy(
    Eagle3SpecDecodeCompilePolicyMixin, DecoderTpuCompilePolicy
):
    """Speculative-decoding policy over the base decoder policy."""

# SPDX-License-Identifier: Apache-2.0
"""TPU compiled-forward adaptation for bucketized static shapes.

This module pads ForwardBatch tensors and Python-list metadata to configured
token/batch buckets before attention metadata is built. It wraps compiled
model forwards to normalize default kwargs at the Dynamo boundary and slices
padded decoder outputs back to real request shapes.
"""

from __future__ import annotations

import functools
import inspect
import time
from typing import TYPE_CHECKING, Any, Callable, List, Optional

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.runtime_context import get_exec, get_parallel, get_schedule
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.utils.common import ceil_align, next_power_of_2, print_warning_once
from sglang_tpu._logging import get_logger
from sglang_tpu.compilation.bucketer import (
    Bucketer,
    PowerBucketer,
)
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.compilation.recompilation_checker import (
    snapshot_recompilation_baseline,
)
from sglang_tpu.cp_forward import (
    cp_local_embeds,
    cp_token_inputs,
    get_tpu_cp_layout,
    pinned_extend_seq_lens_cpu,
)
from sglang_tpu.envs import envs as tpu_envs

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool_host import TpuFusedMHATokenToKVPoolHost
    from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy

logger = get_logger(__name__)

DEFAULT_TPU_COMPILE_MAX_BS = 64

# Fields whose token dim gets padded to the token bucket.
_TOKEN_FIELDS = (
    "input_ids",
    "positions",
    "out_cache_loc",
    "input_embeds",
    # Draft extend inherits the target's real-length embeds, so re-pad them.
    "mm_input_embeds",
    "mrope_positions",
)

# Token dim of each field above. MROPE positions are (3, num_tokens). Every
# other field has tokens on dim 0.
_TOKEN_FIELD_DIM = {"mrope_positions": -1}

_MAMBA_TRACK_FIELDS = (
    "mamba_track_indices",
    "mamba_track_mask",
    "mamba_track_seqlens",
)

# Fields whose batch dim gets padded to the bs bucket.
_BS_FIELDS = (
    "seq_lens",
    "orig_seq_lens",
    "req_pool_indices",
    "req_pool_indices_cpu",
    "extend_seq_lens",
    "extend_prefix_lens",
    "extend_start_loc",
    "encoder_lens",
    *_MAMBA_TRACK_FIELDS,
)

# Python-list batch metadata can become Dynamo guards; pad to bucket length.
_BS_CPU_LIST_FIELDS = (
    "extend_seq_lens_cpu",
    "extend_prefix_lens_cpu",
    "extend_logprob_start_lens_cpu",
)

_REAL_Q_KEY = "_tpu_real_q"
_REAL_B_KEY = "_tpu_real_b"
_DEFAULT_KWARG_TYPES = (type(None), bool, int, float, str)


def _extend_batch_cap(
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
    configured_cap: int | None = None,
) -> int:
    """Return the largest reachable fixed batch shape for EXTEND."""
    return min(
        t_bucketer.max_val,
        b_bucketer.max_val,
        configured_cap or b_bucketer.max_val,
    )


def _static_default_kwargs(fn: Callable[..., Any]) -> dict[str, Any]:
    signature = getattr(fn, "__signature__", None)
    if signature is None:
        return {}
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
        and isinstance(param.default, _DEFAULT_KWARG_TYPES)
    }


def _matches_default(value: Any, default: Any) -> bool:
    if default is None:
        return value is None
    return type(value) is type(default) and value == default


def _cp_localize_embeds_in_args(
    args: tuple[Any, ...], cp_layout: Any
) -> tuple[Any, ...]:
    """Narrow a positionally-passed ``input_embeds`` to this rank's rows.

    ``input_embeds`` is the compiled inner forward's 4th parameter, so for a
    model that fills it positionally rather than by keyword it lands at
    ``args[0]``, where the keyword lookup at the call site cannot see it.
    Arity is unchanged.
    """
    if not args or not isinstance(args[0], torch.Tensor):
        return args
    return (cp_local_embeds(args[0], cp_layout),) + args[1:]


def _localize_cp_inputs(
    forward_batch: Any,
    *,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    cp_layout: Any,
    embeds_tokens: bool,
) -> tuple[torch.Tensor, torch.Tensor, tuple[Any, ...]]:
    """Present this rank's zigzag rows to a context-parallel graph call.

    The graph embeds the global ids (identical on every rank) and runs the
    model on the rank's rows; every token-indexed field on the batch becomes
    local. Returns the graph's ``input_ids``, ``positions`` and ``args``;
    ``kwargs`` is updated in place.
    """
    input_ids, local_ids, positions = cp_token_inputs(
        input_ids=input_ids, positions=positions, layout=cp_layout
    )
    # Embeddings a caller supplies are global, and the precompile ladder builds
    # them local like every other token-indexed input; leaving them global is a
    # serve-time recompile, which `fail_on_recompile` turns into a crash on the
    # first request.
    supplied_embeds = kwargs.get("input_embeds") is not None
    if supplied_embeds:
        kwargs["input_embeds"] = cp_local_embeds(
            input_embeds=kwargs["input_embeds"], layout=cp_layout
        )
    else:
        args = _cp_localize_embeds_in_args(args=args, cp_layout=cp_layout)
    if supplied_embeds or not embeds_tokens:
        # Later stages take their activations from the proxy, and a caller that
        # supplies embeddings has no lookup to run; the ids only size the rows.
        input_ids = local_ids
    else:
        # The embedding takes the global ids and keeps the rank's rows.
        kwargs["cp_local_rows"] = cp_layout.local_rows
    forward_batch.input_ids = local_ids
    forward_batch.positions = positions
    return input_ids, positions, args


def _fit_pp_proxy_rows(proxy: Any, input_ids: Any, forward_batch: Any) -> Any:
    """Bring a pipeline proxy to the padded batch's token axis.

    The ICI wave delivers the previous stage's activations at its fixed buffer
    size (the largest token bucket), so a longer proxy is trimmed to this
    stage's padded token count; a shorter one means the two stages padded to
    different buckets and is rejected.
    """
    reference = input_ids if input_ids is not None else forward_batch.input_ids
    if reference is None:
        return proxy
    expected = int(reference.shape[0])
    tensors = {}
    trimmed = False
    for name, tensor in proxy.tensors.items():
        if isinstance(tensor, torch.Tensor):
            rows = int(tensor.shape[0])
            if rows < expected:
                raise ValueError(
                    f"pp_proxy_tensors[{name!r}] has {rows} rows but the padded "
                    f"batch has {expected} input tokens; the previous pipeline "
                    "stage and this one padded to different token buckets."
                )
            if rows > expected:
                tensor = tensor[:expected]
                trimmed = True
        tensors[name] = tensor
    return type(proxy)(tensors) if trimmed else proxy


def _fit_pp_proxy_in_args(
    args: tuple[Any, ...], input_ids: Any, forward_batch: Any
) -> tuple[Any, ...]:
    """Trim a pipeline proxy that arrived as a positional argument.

    Most models hand the proxy to the compiled decoder as the
    ``pp_proxy_tensors`` keyword, which the kwarg lookup above covers; several
    (glm4_moe, kimi_k3, minimax_m3, sarvam_moe, ...) pass it positionally. A
    proxy that slips past the lookup reaches direct_compile at the wave's full
    row count, which the specialization key does not cover.
    """
    for index, value in enumerate(args):
        if isinstance(value, PPProxyTensors):
            fitted = _fit_pp_proxy_rows(value, input_ids, forward_batch)
            if fitted is not value:
                return args[:index] + (fitted,) + args[index + 1 :]
            break
    return args


def _zero_pad_dim0(t: torch.Tensor, target_size: int) -> torch.Tensor:
    """Zero-pad a tensor along dim 0 to target_size."""
    if t.shape[0] >= target_size:
        return t
    pad_rows = target_size - t.shape[0]
    return F.pad(t, (0,) * (2 * (t.ndim - 1)) + (0, pad_rows))


def _dp_attn_select_bucket_shape(
    forward_batch,
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
    max_tokens_per_rank: int,
    real_q: int,
    real_b: int,
) -> tuple[int, int]:
    """Pick the padded (pad_t, pad_b) bucket every DP rank agrees on."""

    is_decode = forward_batch.forward_mode.is_decode()
    is_idle = forward_batch.forward_mode.is_idle()
    has_fixed_tokens_per_request = (
        forward_batch.forward_mode.is_target_verify()
        or forward_batch.forward_mode.is_draft_extend_v2()
    )

    group_is_decode = is_decode or (
        is_idle and not getattr(forward_batch, "is_extend_in_batch", False)
    )
    if group_is_decode:
        # Decode runs one query token per sequence, so the token axis equals
        # the batch axis and both pad with the bs bucketer (decode graphs are
        # precompiled at (bs, bs) shapes). pad_b also uses max_tokens_per_rank
        # so the decode BATCH bucket is identical across ranks.
        pad_t = b_bucketer.next_bucket(max(real_q, max_tokens_per_rank))
        pad_b = b_bucketer.next_bucket(max(real_b, max_tokens_per_rank))
    elif has_fixed_tokens_per_request:
        # Fixed-width spec modes bucket whole requests, not independent tokens.
        spec_info = forward_batch.spec_info
        tokens_per_req = spec_info.num_tokens_per_req
        if tokens_per_req <= 0:
            raise ValueError(
                "Fixed-width speculative forwards require "
                "spec_info.num_tokens_per_req > 0, "
                f"got {tokens_per_req}."
            )
        max_bs_per_rank = (max_tokens_per_rank + tokens_per_req - 1) // tokens_per_req
        pad_b = b_bucketer.next_bucket(max(real_b, max_bs_per_rank))
        pad_t = pad_b * tokens_per_req
    else:
        # Extend pads to one fixed batch shape. It cannot contain more requests
        # than tokens, so cap the shape by the token ceiling when decode
        # concurrency is larger. Precompile mirrors this choice.
        # TODO(wyzhang): Support multiple batch size buckets for extend/prefill.
        pad_b = _extend_batch_cap(t_bucketer, b_bucketer)
        pad_t = t_bucketer.next_bucket(max(real_q, max_tokens_per_rank, pad_b))
    return pad_t, pad_b


def _dp_attn_apply_bucket_metadata(forward_batch, pad_t: int, pad_b: int) -> None:
    """Update the DP-attention state on forward_batch for the chosen bucket."""
    from sglang.srt.layers.dp_attention import (
        set_dp_buffer_len,
        set_is_extend_in_batch,
    )

    dp_size = len(forward_batch.global_num_tokens_cpu)
    input_ids = getattr(forward_batch, "input_ids", None)
    device = (
        input_ids.device
        if isinstance(input_ids, torch.Tensor)
        else forward_batch.seq_lens.device
    )

    forward_batch.global_num_tokens_cpu = [pad_t] * dp_size
    forward_batch.global_num_tokens_gpu = torch.full(
        (dp_size,), pad_t, dtype=torch.int64, device=device
    )
    if forward_batch.global_num_tokens_for_logprob_cpu is not None:
        # Verify gathers one logit row per token; other modes gather per request.
        padded_logprob_rows = (
            pad_t if forward_batch.forward_mode.is_target_verify() else pad_b
        )
        forward_batch.global_num_tokens_for_logprob_cpu = [
            padded_logprob_rows
        ] * dp_size
        forward_batch.global_num_tokens_for_logprob_gpu = torch.full(
            (dp_size,), padded_logprob_rows, dtype=torch.int64, device=device
        )
    # Clear cached local offsets so they recompute from the padded counts.
    forward_batch.dp_local_start_pos = None
    forward_batch.dp_local_num_tokens = None
    forward_batch.global_dp_buffer_len = pad_t * dp_size
    set_dp_buffer_len(pad_t * dp_size, pad_t, True, [pad_t] * dp_size)
    # Traced code guards on this flags-tier value; precompile dummies bypass
    # the DP sync that normally publishes it, so publish here for both paths.
    set_is_extend_in_batch(
        getattr(
            forward_batch,
            "is_extend_in_batch",
            forward_batch.forward_mode.is_extend(),
        )
    )


def _dp_attn_compute_padding(
    forward_batch,
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
    real_q: int,
    real_b: int,
) -> tuple[int, int, int, int]:
    """Compute the DP-attention padding (no side effects).

    Every rank must run the same shape, so the bucket is chosen by taking the
    max per-rank token count and padding up to the next bucket size. Returns
    (pad_t, pad_b, real_q_out, real_b_out).
    """
    # TPU only supports the MAX_LEN layout: equal-length per-rank blocks.
    # SUM_LEN is ragged and not supported on TPU.
    if not forward_batch.dp_padding_mode.is_max_len():
        raise NotImplementedError(
            "TPU DP Attention supports DpPaddingMode.MAX_LEN only, but got "
            f"{forward_batch.dp_padding_mode}."
        )

    dp_rank = get_parallel().attn_dp_rank
    # This rank's real live token count (used below for decode/idle real_q/b).
    # global_num_tokens_cpu is padded.
    # original_global_num_tokens_cpu is the pre-padding counts.
    # Explicit None check (not `or`): a malformed empty list must fail at the
    # bounds check below, not silently fall back to the padded counts.
    real_counts = getattr(forward_batch, "original_global_num_tokens_cpu", None)
    if real_counts is None:
        real_counts = forward_batch.global_num_tokens_cpu
    if not 0 <= dp_rank < len(real_counts):
        raise RuntimeError(
            f"DP rank {dp_rank} out of range for per-rank token counts "
            f"(len {len(real_counts)})."
        )
    real_local_tokens = int(real_counts[dp_rank])

    max_tokens_per_rank = max(forward_batch.global_num_tokens_cpu)
    pad_t, pad_b = _dp_attn_select_bucket_shape(
        forward_batch, t_bucketer, b_bucketer, max_tokens_per_rank, real_q, real_b
    )

    if forward_batch.forward_mode.is_decode() or forward_batch.forward_mode.is_idle():
        # ForwardBatch.prepare_mlp_sync_batch (upstream, before this shim) already
        # padded decode/idle shapes up to the cross-rank max.
        real_q_out = real_b_out = real_local_tokens
    else:
        real_q_out, real_b_out = real_q, real_b

    return pad_t, pad_b, real_q_out, real_b_out


def _dp_attn_prepare_padding(
    forward_batch,
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
    real_q: int,
    real_b: int,
) -> tuple[int, int, int, int]:
    """Compute the DP-attention padding and apply the DP-specific batch state.

    Returns (pad_t, pad_b, real_q_out, real_b_out). The caller pads tensors to
    the bucket sizes.
    """
    pad_t, pad_b, real_q_out, real_b_out = _dp_attn_compute_padding(
        forward_batch, t_bucketer, b_bucketer, real_q, real_b
    )
    _dp_attn_apply_bucket_metadata(forward_batch, pad_t, pad_b)
    return pad_t, pad_b, real_q_out, real_b_out


def needs_input_logprob(forward_batch: Any) -> bool:
    """True when this batch wants INPUT (prompt / echo) logprobs on a prefill.

    Output (next-token) logprobs are computed by the eager Sampler after the
    compiled forward returns next_token_logits, so they never reach the graph.
    Input logprobs are different: the LogitsProcessor gathers them over
    per-request slices of the prefill, which is data-dependent and cannot be
    expressed with static shapes. They are served instead by capturing every
    prefill row (CaptureHiddenMode.FULL) and running that gather eagerly after
    the forward -- see CompiledForwardShim.__call__.
    """
    if not forward_batch.return_logprob:
        return False
    mode = forward_batch.forward_mode
    if not mode.is_extend() or mode.is_target_verify():
        return False
    extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu or []
    start_lens_cpu = forward_batch.extend_logprob_start_lens_cpu or []
    return any(
        extend_len - start_len > 0
        for extend_len, start_len in zip(extend_seq_lens_cpu, start_lens_cpu)
    )


# Logprob request metadata that the compiled forward reads, and that Dynamo
# therefore guards on. Warmup only ever compiles the no-logprob values, so a
# request asking for logprobs would miss every cached graph and recompile --
# fatal once the scheduler arms set_stance("fail_on_recompile"). None of it is
# needed inside the graph: the eager Sampler and _attach_input_logprobs read the
# real values after the forward returns.
_PINNED_LOGPROB_FIELDS: dict[str, Any] = {
    "return_logprob": False,
    "top_logprobs_nums": None,
    "token_ids_logprobs": None,
    "extend_input_logprob_token_ids_gpu": None,
}


def _pin_fields(forward_batch: Any, pinned: dict[str, Any]) -> dict[str, Any]:
    """Set each field to its pinned value; return only the ones actually changed.

    Writing nothing when a value already matches keeps the no-logprob path at a
    handful of attribute reads.
    """
    saved: dict[str, Any] = {}
    for field, value in pinned.items():
        current = getattr(forward_batch, field)
        if current is value:
            continue
        saved[field] = current
        setattr(forward_batch, field, value)
    return saved


def _restore_fields(forward_batch: Any, saved: dict[str, Any]) -> None:
    for field, value in saved.items():
        setattr(forward_batch, field, value)


def _attach_input_logprobs(
    *, logits_output: Any, forward_batch: Any, model: Any
) -> None:
    """Gather input (prompt) logprobs eagerly from the captured prefill rows.

    Re-runs the model's own LogitsProcessor rather than reimplementing the
    gather: the off-by-one between a position's logits and the token it predicts
    is already baked into extend_input_logprob_token_ids_gpu, and duplicating
    that is how logprobs end up silently attached to the wrong tokens. Every
    model reaches its logits through this same `logits_processor(input_ids,
    hidden_states, lm_head, forward_batch)` call, so this stays model-agnostic.

    Copies every input_* field the gather produces -- chosen-token logprobs,
    top-k, and logprobs for caller-named token ids. The scheduler indexes each
    directly, so a field left behind is a None index downstream.
    """
    hidden_states = logits_output.hidden_states
    if hidden_states is None:
        raise RuntimeError(
            "input logprobs need CaptureHiddenMode.FULL, but the compiled "
            "forward returned no hidden states"
        )
    logits_processor = model.logits_processor
    previous = logits_processor.compute_input_logprobs
    logits_processor.compute_input_logprobs = True
    try:
        eager = logits_processor(
            forward_batch.input_ids, hidden_states, model.lm_head, forward_batch
        )
    finally:
        logits_processor.compute_input_logprobs = previous
    if eager.input_token_logprobs is None:
        raise RuntimeError(
            "the logits processor returned no input_token_logprobs despite "
            "compute_input_logprobs being set"
        )
    # Exact, not an approximation of the gather's gate: from_forward_batch guards
    # extend_return_top_logprob on is_extend / return_logprob / not target_verify,
    # all of which needs_input_logprob already required to get here, leaving only
    # the any(). Empty top-k past that means the pin was not restored.
    if any(n > 0 for n in forward_batch.top_logprobs_nums or []) and (
        eager.input_top_logprobs_val is None
    ):
        raise RuntimeError(
            "the batch asked for input top logprobs but the logits processor "
            "returned none; the pinned top_logprobs_nums was not restored before "
            "the eager gather"
        )
    # Keep this in step with LogitsProcessorOutput's input_* fields.
    logits_output.input_token_logprobs = eager.input_token_logprobs
    logits_output.input_top_logprobs_val = eager.input_top_logprobs_val
    logits_output.input_top_logprobs_idx = eager.input_top_logprobs_idx
    logits_output.input_token_ids_logprobs_val = eager.input_token_ids_logprobs_val
    logits_output.input_token_ids_logprobs_idx = eager.input_token_ids_logprobs_idx
    logits_output.input_logprobs_copy_done = eager.input_logprobs_copy_done


def _pad_tensor(t: torch.Tensor, target_size: int, *, dim: int = 0) -> torch.Tensor:
    """Pad a tensor along dim to target_size."""
    if t.shape[dim] >= target_size:
        return t
    dim = dim % t.ndim
    padding = (0,) * (2 * (t.ndim - dim - 1)) + (0, target_size - t.shape[dim])
    return F.pad(t, padding)


def _pad_mamba_tracking(forward_batch: Any, pad_b: int) -> None:
    """Upload scheduler-owned tracking metadata only at the bucketed shape."""
    originals = getattr(forward_batch, "_tpu_mamba_tracking_cpu", None)
    if originals is None:
        originals = {}
    for field in _MAMBA_TRACK_FIELDS:
        val = getattr(forward_batch, field, None)
        if not isinstance(val, torch.Tensor):
            continue
        if val.device.type == "cpu":
            originals[field] = val
            setattr(
                forward_batch,
                field,
                _zero_pad_dim0(val, pad_b).to(forward_batch.seq_lens.device),
            )
        elif val.shape[0] != pad_b:
            # Already-padded inputs make repeated preparation harmless. Never
            # hide an early upload by reading it back or padding it on TPU.
            raise ValueError("TPU Mamba tracking metadata must be padded on CPU")
    forward_batch._tpu_mamba_tracking_cpu = originals


def _pad_forward_batch_inputs(
    forward_batch: Any,
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
    extend_bs: int | None = None,
) -> tuple[int, int]:
    """Pad ForwardBatch tensors to next bucket shape and return original sizes."""
    is_decode = forward_batch.forward_mode.is_decode()
    is_draft_extend_v2 = forward_batch.forward_mode.is_draft_extend_v2()
    real_q = forward_batch.input_ids.shape[0]
    real_b = forward_batch.seq_lens.shape[0]
    if (
        tpu_envs.SGLANG_TPU_HOST_PADDING.get()
        and is_decode
        and forward_batch.spec_info is None
    ):
        # Relay-gathered decode tokens arrive bucketed; lengths stay logical.
        real_q = real_b

    from sglang.srt.layers.dp_attention import is_dp_attention_enabled

    if is_dp_attention_enabled():
        # DP Attention needs all ranks to pad to the same bucket.
        pad_t, pad_b, real_q_out, real_b_out = _dp_attn_prepare_padding(
            forward_batch, t_bucketer, b_bucketer, real_q, real_b
        )
    elif is_decode:
        pad_t = b_bucketer.next_bucket(real_q)
        pad_b = b_bucketer.next_bucket(real_b)
        real_q_out, real_b_out = real_q, real_b
    elif forward_batch.forward_mode.is_mixed():
        extend_cap = _extend_batch_cap(t_bucketer, b_bucketer, extend_bs)
        mixed_cap = _extend_batch_cap(t_bucketer, b_bucketer, b_bucketer.max_val)
        pad_b = extend_cap if real_b <= extend_cap else mixed_cap
        if real_b > pad_b:
            raise ValueError(
                f"TPU mixed batch size {real_b} exceeds its compile cap {pad_b}. "
                "Increase --torch-compile-max-bs to cover the largest mixed batch."
            )
        pad_t = t_bucketer.next_bucket(max(real_q, pad_b))
        real_q_out, real_b_out = real_q, real_b
    elif (
        forward_batch.forward_mode.is_target_verify()
        and forward_batch.spec_info is not None
    ):
        # Verify is decode-like: batch follows the bs bucket (not pinned to the
        # cap), with a fixed speculative width per request.
        pad_b = b_bucketer.next_bucket(real_b)
        pad_t = pad_b * forward_batch.spec_info.num_tokens_per_req
        real_q_out, real_b_out = real_q, real_b
    elif is_draft_extend_v2 and forward_batch.spec_info is not None:
        pad_b = b_bucketer.next_bucket(real_b)
        num_tokens_per_req = forward_batch.spec_info.num_tokens_per_req
        assert num_tokens_per_req > 0, (
            f"DRAFT_EXTEND_V2 requires num_tokens_per_req > 0, got {num_tokens_per_req}"
        )
        pad_t = pad_b * num_tokens_per_req
        real_q_out, real_b_out = real_q, real_b
    else:
        # Prefill/extend is precompiled at a fixed batch cap to avoid compiling
        # the full token_bucket x batch_bucket cross product.
        pad_b = _extend_batch_cap(t_bucketer, b_bucketer, extend_bs)
        if real_b > pad_b:
            raise ValueError(
                f"TPU extend batch size {real_b} exceeds its compile cap {pad_b}. "
                "Set --prefill-max-requests to at least the largest extend batch."
            )
        pad_t = t_bucketer.next_bucket(max(real_q, pad_b))
        real_q_out, real_b_out = real_q, real_b

    _pad_batch_fields(
        forward_batch,
        pad_t=pad_t,
        pad_b=pad_b,
        real_b=real_b,
        is_draft_extend_v2=is_draft_extend_v2,
    )
    return real_q_out, real_b_out


def _pad_batch_fields(
    forward_batch: Any,
    *,
    pad_t: int | None,
    pad_b: int,
    real_b: int,
    is_draft_extend_v2: bool,
) -> None:
    """Pad the batch's tensors and lists in place to ``(pad_t, pad_b)``.

    ``pad_t`` None leaves every token-indexed field at its live row count
    (context-parallel batches present their local rows to the graph instead);
    ``real_b`` is the live request count the CPU mirrors must cover.
    """
    if pad_t is not None:
        _pad_token_fields(forward_batch, pad_t, is_draft_extend_v2)

    _pad_mamba_tracking(forward_batch, pad_b)
    for field in _BS_FIELDS:
        if field in _MAMBA_TRACK_FIELDS:
            continue
        val = getattr(forward_batch, field, None)
        if isinstance(val, torch.Tensor) and val.shape[0] < pad_b:
            setattr(forward_batch, field, _zero_pad_dim0(val, pad_b))

    # Normalize tensor/None values to padded lists so every extend batch
    # presents the same type to the compiled graph (the precompile dummy uses
    # lists, and DP attention derives tensors when converting decode batches
    # to extend). Non-extend batches keep the upstream None.
    for field in _BS_CPU_LIST_FIELDS:
        val = getattr(forward_batch, field, None)
        if field == "extend_seq_lens_cpu" and (val is None or len(val) < real_b):
            # The attention backend prefers this mirror over the device query
            # lengths whenever it covers the batch, so padding a missing or
            # stale one out to pad_b would hand the draft/extend path zeros or
            # an earlier batch's counts for real requests. Clearing it keeps
            # _extend_query_lens on the device lengths instead.
            forward_batch.extend_seq_lens_cpu = None
            continue
        if val is None:
            if not (forward_batch.forward_mode.is_extend() or is_draft_extend_v2):
                continue
            val = []
        elif isinstance(val, torch.Tensor):
            val = val.tolist()
        if len(val) < pad_b:
            val = val + [0] * (pad_b - len(val))
        setattr(forward_batch, field, val)

    # mm_inputs is per request. Keep it aligned with padded extend_* CPU lists
    # so eager multimodal embedding code can index by batch position.
    # Normalize None to avoid a separate idle-rank Dynamo guard.
    mm_inputs = forward_batch.mm_inputs
    if mm_inputs is None:
        mm_inputs = []
    if len(mm_inputs) < pad_b:
        forward_batch.mm_inputs = mm_inputs + [None] * (pad_b - len(mm_inputs))

    forward_batch.batch_size = pad_b


def _pad_token_fields(forward_batch: Any, pad_t: int, is_draft_extend_v2: bool) -> None:
    # Host-staged inputs stay on CPU through padding and upload at bucket shape.
    # Device-produced inputs retain their device; padding never reads them back.
    if (
        tpu_envs.SGLANG_TPU_HOST_PADDING.get()
        and forward_batch.forward_mode == ForwardMode.EXTEND
        and forward_batch.input_ids.device.type != "cpu"
    ):
        print_warning_once(
            "Host padding is on, but prefill tokens reached padding on the device, "
            "which may compile a program per prompt length."
        )
    for field in _TOKEN_FIELDS:
        val = getattr(forward_batch, field, None)
        dim = _TOKEN_FIELD_DIM.get(field, 0)
        if isinstance(val, torch.Tensor) and val.shape[dim] < pad_t:
            val = _pad_tensor(val, pad_t, dim=dim)
            setattr(forward_batch, field, val)

    if is_draft_extend_v2:
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc.shape[0] < pad_t:
            forward_batch.out_cache_loc = _zero_pad_dim0(out_cache_loc, pad_t)

    # Pad spec_info tensors whose token dim must match input_ids after padding
    # (spec target verify/extend feeds spec_info.hidden_states through the
    # compiled forward). Guarded so non-spec runs are unaffected.
    spec_info = forward_batch.spec_info
    if spec_info is not None:
        hs = getattr(spec_info, "hidden_states", None)
        if isinstance(hs, torch.Tensor) and hs.ndim >= 1 and hs.shape[0] < pad_t:
            pad_rows = pad_t - hs.shape[0]
            spec_info.hidden_states = F.pad(
                hs, (0,) * (2 * (hs.ndim - 1)) + (0, pad_rows)
            )


def _restore_forward_batch_inputs(forward_batch: Any, real_q: int, real_b: int) -> None:
    """Undo in-place TPU bucketing before eager outer-model processing.

    Restore real token and batch shapes so downstream logits and sampling do
    not observe the compiled inner decoder's padded ghost slots.
    """
    for field in _TOKEN_FIELDS:
        value = getattr(forward_batch, field, None)
        if (
            tpu_envs.SGLANG_TPU_HOST_PADDING.get()
            and field == "mrope_positions"
            and isinstance(value, torch.Tensor)
            and forward_batch.forward_mode.is_decode()
            and forward_batch.spec_info is None
        ):
            # Ordinary decode consumes mRoPE only inside the decoder. Leave
            # its device tensor bucketed instead of creating an unused slice.
            continue
        dim = _TOKEN_FIELD_DIM.get(field, 0)
        if isinstance(value, torch.Tensor) and value.shape[dim] > real_q:
            setattr(forward_batch, field, value.narrow(dim, 0, real_q))

    for field in _BS_FIELDS:
        value = getattr(forward_batch, field, None)
        if field in _MAMBA_TRACK_FIELDS:
            # Restore the host snapshot, avoiding a variable-size TPU slice
            # when unpadding the completed forward.
            originals = getattr(forward_batch, "_tpu_mamba_tracking_cpu", None)
            if originals is not None and field in originals:
                value = originals[field]
                setattr(forward_batch, field, value)
        if isinstance(value, torch.Tensor) and value.shape[0] > real_b:
            setattr(forward_batch, field, value[:real_b])
    forward_batch._tpu_mamba_tracking_cpu = None

    for field in _BS_CPU_LIST_FIELDS:
        value = getattr(forward_batch, field, None)
        if (isinstance(value, torch.Tensor) and value.shape[0] > real_b) or (
            isinstance(value, list) and len(value) > real_b
        ):
            setattr(forward_batch, field, value[:real_b])

    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    if isinstance(mm_inputs, list) and len(mm_inputs) > real_b:
        forward_batch.mm_inputs = mm_inputs[:real_b]

    mm_input_embeds = getattr(forward_batch, "mm_input_embeds", None)
    if isinstance(mm_input_embeds, torch.Tensor) and mm_input_embeds.shape[0] > real_q:
        forward_batch.mm_input_embeds = mm_input_embeds[:real_q]

    spec_info = getattr(forward_batch, "spec_info", None)
    hidden_states = getattr(spec_info, "hidden_states", None)
    if isinstance(hidden_states, torch.Tensor) and hidden_states.shape[0] > real_q:
        spec_info.hidden_states = hidden_states[:real_q]

    forward_batch.batch_size = real_b


# Warming above this many pow2 blocks is correct but means warmup runs a
# (near-)full-pool transfer, so we surface it rather than silently clamp.
_KV_OFFLOAD_WARMUP_WARN_BLOCKS: int = 2048


def precompile_kv_offload(host_pool: TpuFusedMHATokenToKVPoolHost) -> None:
    """Dummy-run the real reload/backup transfer methods across pow2 block
    buckets to pre-compile their XLA programs.
    """
    device_pool = host_pool.device_pool
    page_size = host_pool.page_size
    device = device_pool.kv_buffer[0].device

    # Serving pads page counts UP to the next pow2
    device_blocks = device_pool.kv_buffer[0].shape[0]
    max_blocks = next_power_of_2(device_blocks)  # next pow2 >= capacity
    if max_blocks > _KV_OFFLOAD_WARMUP_WARN_BLOCKS:
        logger.warning(
            "[kv-offload warmup] device pool has %d blocks; warming pow2 "
            "buckets up to %d (a near-full-pool transfer). Startup will be "
            "longer, but this avoids a mid-serve JIT recompile.",
            device_blocks,
            max_blocks,
        )

    start = time.perf_counter()
    num_buckets = 0
    n = 1
    while n <= max_blocks:
        num_tokens = n * page_size
        # Page-aligned indices into page 0. Host indices are host-born (CPU),
        # device indices live on-device -- matching what move_indices() yields.
        host_indices = torch.zeros(num_tokens, dtype=torch.long)
        device_indices = torch.zeros(num_tokens, dtype=torch.long, device=device)

        # Run the ACTUAL transfer methods so the warmed HLOs == the served HLOs.
        host_pool.load_to_device_all(
            device_pool, host_indices, device_indices, "torch_tpu"
        )
        host_pool.backup_from_device_all_layer(
            device_pool, host_indices, device_indices, "torch_tpu"
        )

        num_buckets += 1
        n <<= 1

    logger.info(
        "[kv-offload warmup] pre-compiled reload/backup HLOs for %d pow2 block "
        "buckets (<= %d blocks) in %.1fs.",
        num_buckets,
        max_blocks,
        time.perf_counter() - start,
    )
    snapshot_recompilation_baseline()


class CompiledForwardShim:
    """Adapt compiled forward calls and outputs for padded execution."""

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        policy: DecoderTpuCompilePolicy,
        warmup_async_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.fn = compiled_fn
        self.warmup_async_fn = warmup_async_fn
        self.policy = policy
        self._recompilation_checker = None
        self._model_label = ""
        # Read once: the flag decides which extend variant warmup compiled, so
        # it is frozen for this shim's lifetime. Caching it also keeps the
        # disabled path at a single attribute read per forward.
        self._prompt_logprob_enabled = envs.SGLANG_TPU_ENABLE_PROMPT_LOGPROB.get()
        runner = policy.model_runner
        # Draft models read the multimodal embeds inside their forward.
        self._draft_worker = bool(
            runner is not None and getattr(runner, "is_draft_worker", False)
        )
        # Prefill context parallelism: above 1, extend batches present their
        # local zigzag rows to the graph and a batch-independent
        # extend_seq_lens_cpu (see __call__).
        cp_strategy = get_cp_strategy()
        self._attn_cp_size = cp_strategy.cp_size if cp_strategy is not None else 1
        signature = getattr(compiled_fn, "__signature__", None)
        if signature is not None:
            self.__signature__ = signature
        self._default_kwargs = _static_default_kwargs(compiled_fn)
        functools.update_wrapper(self, compiled_fn)

    @staticmethod
    def _prompt_logprob_rejection(enabled: bool) -> str | None:
        """Why this server cannot serve input logprobs, or None if it can.

        Consulted only for requests that actually ask for them, so none of this
        lands on anyone else's forward. Intake asks the same question first.
        """
        if not enabled:
            return (
                "input/prompt logprobs (echo) need "
                "SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1 on the TPU compiled "
                "backend; only output (next-token) logprobs are served by "
                "default."
            )
        parallel = get_parallel()
        # get_attention_dp_size() asserts on live DP state; read the parallel leaves.
        attn_dp_size = parallel.attn_dp_size
        # _attach_input_logprobs re-runs the LogitsProcessor, whose logits path
        # enters the attention-TP/DP collectives. Wanting input logprobs is a
        # property of the local batch, so under DP attention a rank holding
        # such a request would join a collective its peers never reach and the
        # worker group would hang.
        if attn_dp_size > 1:
            return (
                "input/prompt logprobs are not supported with DP attention "
                "(attn_dp_size > 1): the eager gather re-enters the logits "
                "collectives, which only the ranks holding such a request "
                "would reach. TODO: reduce the request flag across the DP "
                "group and have every rank run the gather."
            )
        # Non-final pipeline stages return PPProxyTensors, which carries no
        # hidden states for the gather to read.
        if parallel.pp_size > 1:
            return (
                "input/prompt logprobs are not supported with pipeline "
                "parallelism (pp_size > 1): non-final stages return "
                "PPProxyTensors, which carries no hidden states. TODO: run "
                "the gather on the last stage only."
            )
        return None

    def set_recompilation_checker(self, checker: Any, model_label: str = "") -> None:
        """Set the serving-time recompilation checker."""
        self._recompilation_checker = checker
        self._model_label = model_label

    def _is_first_pp_stage(self) -> bool:
        pp_group = self.policy.model_runner.pp_group
        return pp_group is not None and bool(pp_group.is_first_rank)

    def wait_for_hicache_reload(self) -> None:
        """Wait for the current HiCache reload outside compiled regions."""
        model_runner = self.policy.model_runner
        if model_runner is not None:
            counter = model_runner.token_to_kv_pool.layer_transfer_counter
            if counter is not None:
                counter.wait()

    def __call__(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        forward_batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        return self._call(self.fn, input_ids, positions, forward_batch, *args, **kwargs)

    def warmup_async(self, *args: Any, **kwargs: Any) -> Any:
        """Queue direct compilation and defer execution until ``wait``."""
        return self._call(self.warmup_async_fn, *args, **kwargs)

    def _call(
        self,
        fn: Callable[..., Any],
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        forward_batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # HiCache reload no-stale gate: block until the reload's host->device
        # scatters have landed. A compiled caller must invoke the eager gate
        # before entering its graph; tracing the mutable counter would guard on
        # its lifecycle state and recompile on the first real reload.
        if not torch.compiler.is_compiling():
            self.wait_for_hicache_reload()

        if self._default_kwargs and kwargs:
            # Dynamo treats omitted kwargs and explicit defaults as different
            # call shapes; drop explicit default values.
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if not (
                    k in self._default_kwargs
                    and _matches_default(v, self._default_kwargs[k])
                )
            }
        # The fused draft loop inlines this call into its own Dynamo graph.
        # Reading the checker there would put it in that graph's guards, so
        # attaching it later would recompile the graph this very tool exists to
        # protect. Compiles from inside it surface on the next eager forward.
        if torch.compiler.is_compiling():
            return self._forward(
                fn, input_ids, positions, forward_batch, *args, **kwargs
            )

        checker = self._recompilation_checker
        if checker is None:
            return self._forward(
                fn, input_ids, positions, forward_batch, *args, **kwargs
            )
        # Read at entry: the padded shapes are what the graph compiled for.
        pad_q = int(forward_batch.input_ids.shape[0])
        pad_b = int(forward_batch.seq_lens.shape[0])
        try:
            return self._forward(
                fn, input_ids, positions, forward_batch, *args, **kwargs
            )
        finally:
            checker.check(
                context_fn=lambda: self._describe_forward(
                    forward_batch, pad_q=pad_q, pad_b=pad_b
                )
            )

    def _describe_forward(self, forward_batch: Any, *, pad_q: int, pad_b: int) -> str:
        # Dynamic keys, unset on eagle3 draft extend (as in _unpad_decoder_outputs).
        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        real_b = getattr(forward_batch, _REAL_B_KEY, None)
        real = f" real=q{real_q},b{real_b}" if real_q is not None else ""
        mode = forward_batch.forward_mode.name
        return f"{self._model_label} {mode}{real} pad=q{pad_q},b{pad_b}"

    def _forward(
        self,
        fn: Callable[..., Any],
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        forward_batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        keep_padded = self.policy.keep_outputs_padded(forward_batch)
        # EAGLE3 draft extend: EagleDraftExtendInput also rides DRAFT_EXTEND_V2,
        # which is_extend() excludes.
        if (
            forward_batch.mm_input_embeds is not None
            and isinstance(forward_batch.spec_info, EagleDraftExtendInput)
            and forward_batch.forward_mode.is_extend()
        ):
            # Local import: the eagle3 policy imports this module.
            from sglang_tpu.compilation.spec_decode_compile_policy_eagle3 import (
                fold_last_token_embeds,
            )

            forward_batch.input_embeds = fold_last_token_embeds(
                forward_batch=forward_batch,
                embed_tokens=self.policy.model_runner.model.get_input_embeddings(),
            )
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds
        # Pin is_prefill_only for the compiled forward. An idle batch computes
        # it as True (all() over zero requests) while real batches are False;
        # the compiled forward reads the value, so it must be identical across
        # all batches to avoid a serve-time recompile. The worker's sampling
        # gate reads it after the forward — a prefill-only batch
        # (max_new_tokens=0) has no sampling metadata to run the sampler
        # with — so the original value is restored even if the forward
        # raises.
        orig_prefill_only = forward_batch.is_prefill_only
        forward_batch.is_prefill_only = False
        # Input (prompt) logprobs, when enabled. FULL capture is then presented
        # on *every* extend, not only the batches asking for logprobs:
        # capture_hidden_mode is part of the graph's guard, so two values would
        # mean two extend variants (measured: 828s of precompile against 113s).
        # That is precisely why this is opt-in -- returning every prefill row
        # costs ~8% of prefill, and nobody who did not ask should pay it. With
        # the flag off nothing below runs, the batch is never touched, and the
        # extend graph is the NULL-capture one warmup has always compiled.
        wants_input_logprob = needs_input_logprob(forward_batch)
        if wants_input_logprob:
            rejection = self._prompt_logprob_rejection(self._prompt_logprob_enabled)
            if rejection is not None:
                raise NotImplementedError(rejection)
        saved: dict[str, Any] = {}
        if self._prompt_logprob_enabled and forward_batch.forward_mode.is_extend():
            saved["capture_hidden_mode"] = forward_batch.capture_hidden_mode
            forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            # LogitsMetadata reads len(extend_logprob_start_lens_cpu), so the
            # graph guards on it. One zero per padded request is both a stable
            # length and a request for every position; the real starts, which
            # the eager gather needs, go back in the finally below.
            saved["extend_logprob_start_lens_cpu"] = (
                forward_batch.extend_logprob_start_lens_cpu
            )
            forward_batch.extend_logprob_start_lens_cpu = [0] * int(
                forward_batch.seq_lens.shape[0]
            )
        saved.update(_pin_fields(forward_batch, _PINNED_LOGPROB_FIELDS))
        if not self._prompt_logprob_enabled and hasattr(
            forward_batch, "extend_logprob_start_lens_cpu"
        ):
            # Only the eager input-logprob gather reads these; keep them out of the guards.
            saved.update(
                _pin_fields(forward_batch, {"extend_logprob_start_lens_cpu": None})
            )
        # Keep the multimodal embeds out of a target graph's guards; re-attached below.
        mm_embeds = (
            None
            if self._draft_worker
            else getattr(forward_batch, "mm_input_embeds", None)
        )
        if mm_embeds is not None:
            saved["mm_input_embeds"] = mm_embeds
            forward_batch.mm_input_embeds = None
        live_extend_seq_lens_cpu = None
        if self._attn_cp_size > 1:
            cp_layout = get_tpu_cp_layout(forward_batch)
            # The graph tests the extend lengths (cp_strategy) to pick the
            # context-parallel or the plain program; one constant per
            # program family keeps that test off the live lengths. The live
            # list goes back at whatever length the output unpadding leaves.
            if isinstance(forward_batch.extend_seq_lens_cpu, list):
                live_extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu
                forward_batch.extend_seq_lens_cpu = pinned_extend_seq_lens_cpu(
                    len(live_extend_seq_lens_cpu),
                    cp_size=self._attn_cp_size,
                    cp_active=cp_layout is not None,
                )
            if cp_layout is not None:
                saved["input_ids"] = forward_batch.input_ids
                saved["positions"] = forward_batch.positions
                input_ids, positions, args = _localize_cp_inputs(
                    forward_batch,
                    input_ids=input_ids,
                    positions=positions,
                    args=args,
                    kwargs=kwargs,
                    cp_layout=cp_layout,
                    embeds_tokens=self._is_first_pp_stage(),
                )
        try:
            # A proxy holds the previous stage's local rows; fit it after localizing.
            pp_proxy_tensors = kwargs.get("pp_proxy_tensors")
            if pp_proxy_tensors is not None:
                if self._is_first_pp_stage():
                    # The first stage embeds its own input; its graph takes no proxy.
                    kwargs.pop("pp_proxy_tensors")
                else:
                    kwargs["pp_proxy_tensors"] = _fit_pp_proxy_rows(
                        proxy=pp_proxy_tensors,
                        input_ids=input_ids,
                        forward_batch=forward_batch,
                    )
            elif any(isinstance(value, PPProxyTensors) for value in args):
                if not self._is_first_pp_stage():
                    args = _fit_pp_proxy_in_args(
                        args=args, input_ids=input_ids, forward_batch=forward_batch
                    )
            out = fn(input_ids, positions, forward_batch, *args, **kwargs)
            if isinstance(out, PendingCall):
                finalize_fields = {key: getattr(forward_batch, key) for key in saved}
                finalize_fields["is_prefill_only"] = forward_batch.is_prefill_only

                def finalize_pending(result: Any) -> Any:
                    previous = _pin_fields(forward_batch, finalize_fields)
                    try:
                        return self._finalize_forward(
                            result, forward_batch, keep_padded=keep_padded
                        )
                    finally:
                        _restore_fields(forward_batch, previous)

                out = out.then(finalize_pending)
            else:
                out = self._finalize_forward(
                    out, forward_batch, keep_padded=keep_padded
                )
        finally:
            forward_batch.is_prefill_only = orig_prefill_only
            _restore_fields(forward_batch, saved)
            if live_extend_seq_lens_cpu is not None:
                forward_batch.extend_seq_lens_cpu = live_extend_seq_lens_cpu[
                    : len(forward_batch.extend_seq_lens_cpu)
                ]
        if mm_embeds is not None:
            real_q = getattr(forward_batch, _REAL_Q_KEY, None)
            if (
                not keep_padded
                and real_q is not None
                and mm_embeds.shape[0] > real_q
                and not self.policy.keeps_padded_mm_embeds(forward_batch)
            ):
                mm_embeds = mm_embeds[:real_q]
            forward_batch.mm_input_embeds = mm_embeds

        def finish(result: Any) -> Any:
            # Re-attach the embeds the graph did not see; a hidden-state boundary
            # leaves them on the batch for the wrapper's eager tail.
            if (
                mm_embeds is not None
                and isinstance(result, LogitsProcessorOutput)
                and result.mm_input_embeds is None
            ):
                result.mm_input_embeds = mm_embeds
            return self._finish_forward(result, forward_batch, wants_input_logprob)

        if isinstance(out, PendingCall):
            return out.then(finish)
        return finish(out)

    def _finalize_forward(
        self, out: Any, forward_batch: Any, *, keep_padded: bool
    ) -> Any:
        spec_info = forward_batch.spec_info
        if not keep_padded and (spec_info is None or not spec_info.keep_outputs_padded):
            out = self.policy.finalize_forward_batch(out, forward_batch)
        return out

    def _finish_forward(
        self, out: Any, forward_batch: Any, wants_input_logprob: bool = False
    ) -> Any:
        # After the restore: LogitsMetadata.from_forward_batch branches on
        # capture_hidden_mode, so the gather must see the caller's value.
        if wants_input_logprob:
            _attach_input_logprobs(
                logits_output=out,
                forward_batch=forward_batch,
                model=self.policy.model_runner.model,
            )

        return out


def reconcile_tpu_batch_limits(
    torch_compile_max_bs: int | None,
    tpu_compile_bs: Optional[List[int]],
    max_running_requests: int | None,
    attn_dp_size: int = 1,
    attn_tp_size: int = 1,
) -> tuple[int, Optional[List[int]], int]:
    """Make the TPU compile grid cover scheduler concurrency without dead shapes.

    The grid serves one attention-DP worker, and each worker's scheduler runs
    at most ``max_running_requests // attn_dp_size`` requests (the same
    division kv_cache_configurator applies), so all coupling here uses that
    per-worker concurrency, not the global cap.

    ``attn_tp_size`` is the attention-TP width under attention DP and 1
    otherwise: attention DP rounds every worker's decode rows up to a multiple
    of it before the bucket lookup (``get_dp_padded_token_counts``), so the
    grid has to reach the rounded per-worker cap.
    """
    per_worker_mrr = (
        max(max_running_requests // attn_dp_size, 1)
        if max_running_requests is not None
        else None
    )
    explicit_compile_limits = [
        limit
        for limit in (
            torch_compile_max_bs,
            max(tpu_compile_bs) if tpu_compile_bs else None,
        )
        if limit is not None
    ]
    if explicit_compile_limits:
        tcmb = max(explicit_compile_limits)
    elif per_worker_mrr is not None:
        # A worker's scheduler never batches more than its concurrency cap, so
        # buckets above it are dead shapes: they cost compile time and, for
        # long contexts, SMEM for the attention page table.
        tcmb = per_worker_mrr
    else:
        tcmb = DEFAULT_TPU_COMPILE_MAX_BS

    if per_worker_mrr is None:
        per_worker_mrr = tcmb
        max_running_requests = tcmb * attn_dp_size
    per_worker_rows = ceil_align(per_worker_mrr, attn_tp_size)
    tcmb = max(tcmb, per_worker_rows)

    if tpu_compile_bs and max(tpu_compile_bs) < per_worker_rows:
        tpu_compile_bs = sorted(set(tpu_compile_bs) | {per_worker_rows})

    return tcmb, tpu_compile_bs, max_running_requests


def build_bucketers(
    server_args: Any, max_t: int, max_b: int
) -> tuple[Bucketer, Bucketer]:
    """Build token and batch size bucketers."""
    token_paddings = server_args.tpu_compile_tokens
    if token_paddings:
        # Keep the smallest configured bucket covering the extend limit to
        # preserve its shape alignment. Leave undersized lists unchanged.
        max_bucket = min((t for t in token_paddings if t >= max_t), default=max_t)
        kept = [t for t in token_paddings if t <= max_bucket]
        dropped = [t for t in token_paddings if t > max_bucket]
        if dropped:
            logger.warning(
                "[compiled forward shim] Dropping --tpu-compile-tokens %s above "
                "the last needed token bucket %d (max extend length %d).",
                dropped,
                max_bucket,
                max_t,
            )
        token_bucketer: Bucketer = Bucketer(name="TokenBucketer", buckets=kept)
    else:
        token_bucketer = PowerBucketer(
            name="TokenBucketer", min_val=64, max_val=max_t, power=2
        )

    from sglang_tpu.pp_chunks import uses_dynamic_chunks

    if uses_dynamic_chunks():
        from sglang_tpu.pp_chunks import (
            chunk_buckets,
            chunk_granularity,
        )

        granularity = chunk_granularity(get_schedule().page_size, max_t, True)
        token_bucketer = Bucketer(
            name="TokenBucketer",
            buckets=sorted(
                set(
                    token_bucketer.buckets + chunk_buckets(max_t, granularity) + [max_t]
                )
            ),
        )

    # tpu_compile_bs is reconciled in resolution; read the resolved leaf.
    bs_paddings = get_exec().graph.tpu_compile_bs
    if bs_paddings:
        bs_bucketer: Bucketer = Bucketer(
            name="BatchSizeBucketer", buckets=list(bs_paddings)
        )
    else:
        bs_bucketer = PowerBucketer(
            name="BatchSizeBucketer", min_val=1, max_val=max_b, power=2
        )

    return token_bucketer, bs_bucketer

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
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F

from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.bucketer import (
    Bucketer,
    PowerBucketer,
)
from sglang.srt.utils.common import next_power_of_2

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool_host import TpuFusedMHATokenToKVPoolHost

logger = get_logger(__name__)

DEFAULT_TPU_COMPILE_MAX_BS = 64

# Fields whose token dim gets padded to the token bucket.
_TOKEN_FIELDS = ("input_ids", "positions", "out_cache_loc")

# Fields whose batch dim gets padded to the bs bucket.
_BS_FIELDS = (
    "seq_lens",
    "orig_seq_lens",
    "req_pool_indices",
    "extend_seq_lens",
    "extend_prefix_lens",
    "encoder_lens",
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


def _zero_pad_1d(t: torch.Tensor, target_size: int) -> torch.Tensor:
    """Zero-pad a 1-D tensor to target_size."""
    if t.shape[0] >= target_size:
        return t
    return F.pad(t, (0, target_size - t.shape[0]))


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
    else:
        # Extend always pads to the max batch bucket. Precompile mirrors this in
        # DecoderTpuCompilePolicy._precompile_extend (which compiles extend only
        # at b_bucketer.max_val), so the two must change together.
        # TODO(wyzhang): Support multiple batch size buckets for extend/prefill.
        pad_b = b_bucketer.max_val
        pad_t = t_bucketer.next_bucket(max(real_q, max_tokens_per_rank, pad_b))
    return pad_t, pad_b


def _dp_attn_apply_bucket_metadata(forward_batch, pad_t: int, pad_b: int) -> None:
    """Update the DP-attention state on forward_batch for the chosen bucket."""
    from sglang.srt.layers.dp_attention import set_dp_buffer_len

    dp_size = len(forward_batch.global_num_tokens_cpu)
    device = forward_batch.input_ids.device

    forward_batch.global_num_tokens_cpu = [pad_t] * dp_size
    forward_batch.global_num_tokens_gpu = torch.full(
        (dp_size,), pad_t, dtype=torch.int64, device=device
    )
    if forward_batch.global_num_tokens_for_logprob_cpu is not None:
        forward_batch.global_num_tokens_for_logprob_cpu = [pad_b] * dp_size
        forward_batch.global_num_tokens_for_logprob_gpu = torch.full(
            (dp_size,), pad_b, dtype=torch.int64, device=device
        )
    # Clear cached local offsets so they recompute from the padded counts.
    forward_batch.dp_local_start_pos = None
    forward_batch.dp_local_num_tokens = None
    forward_batch.global_dp_buffer_len = pad_t * dp_size
    set_dp_buffer_len(pad_t * dp_size, pad_t, True, [pad_t] * dp_size)


def _dp_attn_compute_padding(
    forward_batch,
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
    real_q: int,
    real_b: int,
) -> tuple[int, int, int, int, int]:
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

    from sglang.srt.layers.dp_attention import get_attention_dp_rank

    dp_rank = get_attention_dp_rank()
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

    Returns (pad_t, pad_b, real_q_out, real_b_out). the caller pads tensors to
    the bucket sizes.
    """
    pad_t, pad_b, real_q_out, real_b_out = _dp_attn_compute_padding(
        forward_batch, t_bucketer, b_bucketer, real_q, real_b
    )
    _dp_attn_apply_bucket_metadata(forward_batch, pad_t, pad_b)
    return pad_t, pad_b, real_q_out, real_b_out


def _pad_forward_batch_inputs(
    forward_batch: Any,
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
) -> tuple[int, int]:
    """Pad ForwardBatch tensors to next bucket shape and return original sizes."""
    if forward_batch.return_logprob:
        # Output (next-token) logprobs are computed by the eager Sampler AFTER
        # the compiled forward returns next_token_logits, so they don't affect
        # the compiled graph's shapes. Only INPUT/prompt logprobs (echo) run
        # inside the LogitsProcessor over dynamically-pruned states, which would
        # break the static-shape compiled forward. Reject only that case.
        needs_input_logprob = False
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            extend_seq_lens_cpu = forward_batch.extend_seq_lens_cpu or []
            start_lens_cpu = forward_batch.extend_logprob_start_lens_cpu or []
            for extend_len, start_len in zip(extend_seq_lens_cpu, start_lens_cpu):
                if extend_len - start_len > 0:
                    needs_input_logprob = True
                    break
        if needs_input_logprob:
            raise NotImplementedError(
                "input/prompt logprobs (echo) are not supported on the TPU "
                "compiled backend yet; only output (next-token) logprobs are."
            )

    is_decode = forward_batch.forward_mode.is_decode()
    real_q = forward_batch.input_ids.shape[0]
    real_b = forward_batch.seq_lens.shape[0]

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
    elif (
        forward_batch.forward_mode.is_target_verify()
        and forward_batch.spec_info is not None
    ):
        # Verify is decode-like: batch follows the bs bucket (not pinned to the
        # cap), with a fixed draft_token_num query tokens per request.
        pad_b = b_bucketer.next_bucket(real_b)
        pad_t = pad_b * forward_batch.spec_info.draft_token_num
        real_q_out, real_b_out = real_q, real_b
    else:
        # Prefill/extend is precompiled at a fixed batch cap to avoid compiling
        # the full token_bucket x batch_bucket cross product.
        pad_b = b_bucketer.max_val
        pad_t = t_bucketer.next_bucket(max(real_q, pad_b))
        real_q_out, real_b_out = real_q, real_b

    for field in _TOKEN_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, torch.Tensor) and val.shape[0] < pad_t:
            setattr(forward_batch, field, _zero_pad_1d(val, pad_t))

    # MROPE positions are (3, num_tokens); pad along the token dim.
    mrope_pos = getattr(forward_batch, "mrope_positions", None)
    if isinstance(mrope_pos, torch.Tensor) and mrope_pos.shape[-1] < pad_t:
        forward_batch.mrope_positions = F.pad(
            mrope_pos, (0, pad_t - mrope_pos.shape[-1])
        )

    for field in _BS_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, torch.Tensor) and val.shape[0] < pad_b:
            setattr(forward_batch, field, _zero_pad_1d(val, pad_b))

    # Normalize tensor/None values to padded lists so every extend batch
    # presents the same type to the compiled graph (the precompile dummy uses
    # lists, and DP attention derives tensors when converting decode batches
    # to extend). Non-extend batches keep the upstream None.
    for field in _BS_CPU_LIST_FIELDS:
        val = getattr(forward_batch, field, None)
        if val is None:
            if not forward_batch.forward_mode.is_extend():
                continue
            val = []
        elif isinstance(val, torch.Tensor):
            val = val.tolist()
        if len(val) < pad_b:
            val = val + [0] * (pad_b - len(val))
        setattr(forward_batch, field, val)

    # mm_inputs is per request. Keep it aligned with padded extend_* CPU lists
    # so eager multimodal embedding code can index by batch position.
    mm_inputs = getattr(forward_batch, "mm_inputs", None)
    if isinstance(mm_inputs, list) and len(mm_inputs) < pad_b:
        forward_batch.mm_inputs = mm_inputs + [None] * (pad_b - len(mm_inputs))

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

    forward_batch.batch_size = pad_b

    return real_q_out, real_b_out


def _restore_forward_batch_inputs(forward_batch: Any, real_q: int, real_b: int) -> None:
    """Undo in-place TPU bucketing before eager outer-model processing.

    Restore real token and batch shapes so downstream logits and sampling do
    not observe the compiled inner decoder's padded ghost slots.
    """
    for field in _TOKEN_FIELDS:
        value = getattr(forward_batch, field, None)
        if isinstance(value, torch.Tensor) and value.shape[0] > real_q:
            setattr(forward_batch, field, value[:real_q])

    mrope_positions = getattr(forward_batch, "mrope_positions", None)
    if isinstance(mrope_positions, torch.Tensor) and mrope_positions.shape[-1] > real_q:
        forward_batch.mrope_positions = mrope_positions[..., :real_q]

    for field in _BS_FIELDS:
        value = getattr(forward_batch, field, None)
        if isinstance(value, torch.Tensor) and value.shape[0] > real_b:
            setattr(forward_batch, field, value[:real_b])

    for field in _BS_CPU_LIST_FIELDS:
        value = getattr(forward_batch, field, None)
        if isinstance(value, torch.Tensor) and value.shape[0] > real_b:
            setattr(forward_batch, field, value[:real_b])
        elif isinstance(value, list) and len(value) > real_b:
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


class CompiledForwardShim:
    """Adapt compiled forward calls and outputs for padded execution."""

    def __init__(
        self,
        compiled_fn: Callable[..., Any],
        policy: CompilePolicy,
    ) -> None:
        self.fn = compiled_fn
        self.policy = policy
        self._recompilation_checker = None
        self._model_label = ""
        signature = getattr(compiled_fn, "__signature__", None)
        if signature is not None:
            self.__signature__ = signature
        self._default_kwargs = _static_default_kwargs(compiled_fn)
        functools.update_wrapper(self, compiled_fn)

    def set_recompilation_checker(self, checker: Any, model_label: str = "") -> None:
        """Set the serving-time recompilation checker."""
        self._recompilation_checker = checker
        self._model_label = model_label

    def __call__(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        forward_batch: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # HiCache reload no-stale gate: the producer's reload scatters aren't in the traced graph, so wait until they've landed before launching.
        # TODO: rename layer_transfer_counter to transfer_counter or similar after integrating to upstream as it represents different things on tpu and other hardware backends.
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        counter = getattr(pool, "layer_transfer_counter", None)
        if counter is not None:
            counter.wait()

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
        try:
            out = self.fn(input_ids, positions, forward_batch, *args, **kwargs)
        finally:
            forward_batch.is_prefill_only = orig_prefill_only
        out = self.policy.unpad_outputs(out, forward_batch)

        # Check for serving-time XLA recompilation (once per forward call).
        # Gated by SGLANG_TPU_CHECK_RECOMPILATION=1 (off by default).
        if self._recompilation_checker is not None:
            fm = getattr(forward_batch, "forward_mode", "N/A")
            self._recompilation_checker.check(
                f"{self._model_label} {fm.name if hasattr(fm, 'name') else fm}"
            )

        return out


def resolve_compile_caps(
    *,
    chunked_prefill_size: int | None,
    max_prefill_tokens: int | None,
    context_len: int | None,
    torch_compile_max_bs: int | None,
    max_running_requests: int | None,
) -> tuple[int, int]:
    """Resolve (max_t, max_b) limits for compilation bucketing."""
    candidates = [
        c
        for c in (chunked_prefill_size, max_prefill_tokens, context_len)
        if c and c > 0
    ]
    if not candidates:
        raise ValueError(
            "Cannot determine max_t for TPU compile bucketing: "
            "chunked_prefill_size, max_prefill_tokens, and context_len are all unset."
        )
    # TPU serving should compile the requested scheduler capacity by default for
    # practical benchmark sizes (for example max_running_requests=64). Avoid
    # blindly expanding to very large upstream defaults unless the user raises
    # torch_compile_max_bs or pins --tpu-compile-bs.
    max_running_compile_bs = (
        min(max_running_requests, DEFAULT_TPU_COMPILE_MAX_BS)
        if max_running_requests
        else 1
    )
    max_b = max(torch_compile_max_bs or 1, max_running_compile_bs)
    if max_running_requests and max_b < max_running_requests:
        logger.warning(
            "[compiled forward shim] TPU compile bs cap=%d < max_running_requests=%d; "
            "batches above %d are outside the TPU compile grid. Raise "
            "--torch-compile-max-bs or pin --tpu-compile-bs to cover them.",
            max_b,
            max_running_requests,
            max_b,
        )
    return min(candidates), max_b


def build_bucketers(
    server_args: Any, max_t: int, max_b: int
) -> tuple[Bucketer, Bucketer]:
    """Build token and batch size bucketers."""
    token_paddings = server_args.tpu_compile_tokens
    if token_paddings:
        # Token buckets are consumed only by the EXTEND path, and extend never
        # runs longer than max_t (= min(chunked_prefill_size, max_prefill_tokens,
        # context_len)). Drop any user buckets above it -- they'd compile dead
        # shapes that serving can never reach.
        capped = [t for t in token_paddings if t <= max_t]
        dropped = [t for t in token_paddings if t > max_t]
        if dropped:
            logger.warning(
                "[compiled forward shim] Dropping --tpu-compile-tokens %s above the "
                "max extend length %d; extend is capped there (chunked prefill / "
                "context), so those buckets would never be used.",
                dropped,
                max_t,
            )
        token_bucketer: Bucketer = Bucketer(
            name="TokenBucketer", buckets=sorted(capped) or [max_t]
        )
    else:
        token_bucketer = PowerBucketer(
            name="TokenBucketer", min_val=64, max_val=max_t, power=2
        )

    bs_paddings = server_args.tpu_compile_bs
    if bs_paddings:
        bs_bucketer: Bucketer = Bucketer(
            name="BatchSizeBucketer", buckets=list(bs_paddings)
        )
    else:
        bs_bucketer = PowerBucketer(
            name="BatchSizeBucketer", min_val=1, max_val=max_b, power=2
        )

    return token_bucketer, bs_bucketer

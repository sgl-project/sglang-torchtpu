"""TPU compiled-forward adaptation for bucketized static shapes.

This module pads ForwardBatch tensors and Python-list metadata to
configured token/batch buckets before attention metadata is built. It
wraps compiled model forwards to normalize default kwargs at the Dynamo
boundary and slices padded outputs back to real request shapes.
"""

from __future__ import annotations

import functools
import inspect
from typing import Callable

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.compilation.bucketer import (
    Bucketer,
    PowerBucketer,
)

logger = get_logger(__name__)

DEFAULT_TPU_COMPILE_MAX_BS = 64

# Fields whose token dim gets padded to the token bucket.
_TOKEN_FIELDS = ("input_ids", "positions")

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


def _static_default_kwargs(fn: Callable) -> dict:
    signature = getattr(fn, "__signature__", None)
    if signature is None:
        return {}
    return {
        name: param.default
        for name, param in signature.parameters.items()
        if param.default is not inspect.Parameter.empty
        and isinstance(param.default, _DEFAULT_KWARG_TYPES)
    }


def _matches_default(value, default) -> bool:
    if default is None:
        return value is None
    return type(value) is type(default) and value == default


def _zero_pad_1d(t: torch.Tensor, target_size: int) -> torch.Tensor:
    """Zero-pad a 1-D tensor to target_size."""
    if t.shape[0] >= target_size:
        return t
    return F.pad(t, (0, target_size - t.shape[0]))


def _pad_forward_batch_inputs(
    forward_batch,
    t_bucketer: Bucketer,
    b_bucketer: Bucketer,
) -> tuple[int, int]:
    """Pad ForwardBatch tensors to next bucket shape and return original sizes."""
    if forward_batch.return_logprob:
        raise NotImplementedError(
            "return_logprob is not supported on the TPU backend yet."
        )

    is_decode = forward_batch.forward_mode.is_decode()
    real_q = forward_batch.input_ids.shape[0]
    real_b = forward_batch.seq_lens.shape[0]

    if is_decode:
        pad_t = b_bucketer.next_bucket(real_q)
        pad_b = b_bucketer.next_bucket(real_b)
    else:
        # Prefill/extend is precompiled at a fixed batch cap to avoid compiling
        # the full token_bucket x batch_bucket cross product.
        pad_b = b_bucketer.max_val
        pad_t = t_bucketer.next_bucket(max(real_q, pad_b))

    for field in _TOKEN_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, torch.Tensor) and val.shape[0] < pad_t:
            setattr(forward_batch, field, _zero_pad_1d(val, pad_t))

    for field in _BS_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, torch.Tensor) and val.shape[0] < pad_b:
            setattr(forward_batch, field, _zero_pad_1d(val, pad_b))

    for field in _BS_CPU_LIST_FIELDS:
        val = getattr(forward_batch, field, None)
        if isinstance(val, list) and len(val) < pad_b:
            setattr(forward_batch, field, val + [0] * (pad_b - len(val)))

    forward_batch.batch_size = pad_b

    return real_q, real_b


class CompiledForwardShim:
    """Adapt compiled forward calls and outputs for padded execution."""

    def __init__(self, compiled_fn):
        self.fn = compiled_fn
        signature = getattr(compiled_fn, "__signature__", None)
        if signature is not None:
            self.__signature__ = signature
        self._default_kwargs = _static_default_kwargs(compiled_fn)
        functools.update_wrapper(self, compiled_fn)

    def __call__(self, input_ids, positions, forward_batch, *args, **kwargs):
        # HiCache reload no-stale gate: the producer's reload scatters aren't in
        # the traced graph, so wait until they've landed before launching. The
        # forward reads all layers at once and the producer signals in order
        # (last layer last), so one wait on the final layer covers the batch.
        pool = getattr(forward_batch, "token_to_kv_pool", None)
        counter = getattr(pool, "layer_transfer_counter", None)
        if counter is not None:
            counter.wait_until(pool.layer_num - 1)

        if self._default_kwargs and kwargs:
            # Dynamo treats omitted kwargs and explicit defaults as
            # different call shapes; drop explicit default values.
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if not (
                    k in self._default_kwargs
                    and _matches_default(v, self._default_kwargs[k])
                )
            }
        out = self.fn(input_ids, positions, forward_batch, *args, **kwargs)

        real_q = getattr(forward_batch, _REAL_Q_KEY, None)
        real_b = getattr(forward_batch, _REAL_B_KEY, None)
        if real_q is None or real_b is None:
            return out

        next_token_logits = getattr(out, "next_token_logits", None)
        if next_token_logits is not None and real_b < next_token_logits.shape[0]:
            out.next_token_logits = next_token_logits[:real_b]

        hidden_states = getattr(out, "hidden_states", None)
        if hidden_states is not None and real_q < hidden_states.shape[0]:
            out.hidden_states = hidden_states[:real_q]

        # Restore real batch_size so downstream consumers (sampler,
        # logprob, scheduler hooks) don't iterate over ghost slots.
        forward_batch.batch_size = real_b

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


class TpuCompiledForwardShim:
    """Installs bucket padding and compiled-forward adaptation."""

    def __init__(self, model_runner, compiled_fns: dict[str, Callable]):
        self.model_runner = model_runner
        self.model = model_runner.model
        self._compiled_fns = compiled_fns

        server_args = model_runner.server_args
        max_t, max_b = resolve_compile_caps(
            chunked_prefill_size=server_args.chunked_prefill_size,
            max_prefill_tokens=server_args.max_prefill_tokens,
            context_len=model_runner.model_config.context_len,
            torch_compile_max_bs=getattr(server_args, "torch_compile_max_bs", None),
            max_running_requests=model_runner.max_running_requests,
        )

        self.t_bucketer, self.b_bucketer = build_bucketers(
            server_args, max_t=max_t, max_b=max_b
        )

    def install(self) -> TpuCompiledForwardShim:
        # TODO(wyzhang): Speculative decoding bypasses prepare_forward_batch via
        # skip_attn_backend_init=True. Speculative decoding is not yet supported
        # on TPU; when added, either integrate padding into the batch builder
        # or ensure speculative runners invoke prepare_forward_batch.
        existing_prep = getattr(self.model, "prepare_forward_batch", None)
        t_bucketer = self.t_bucketer
        b_bucketer = self.b_bucketer

        def _prepare(forward_batch):
            if existing_prep is not None:
                existing_prep(forward_batch)
            real_q, real_b = _pad_forward_batch_inputs(
                forward_batch, t_bucketer, b_bucketer
            )
            setattr(forward_batch, _REAL_Q_KEY, real_q)
            setattr(forward_batch, _REAL_B_KEY, real_b)

        self.model.prepare_forward_batch = _prepare

        for attr, compiled_fn in self._compiled_fns.items():
            setattr(self.model, attr, CompiledForwardShim(compiled_fn))

        logger.info(
            "[compiled forward shim] Installed compiled-forward adaptation on %s.{%s} "
            "(t-bucketer=%s, b-bucketer=%s).",
            type(self.model).__name__,
            ", ".join(self._compiled_fns),
            self.t_bucketer.name,
            self.b_bucketer.name,
        )
        return self


def build_bucketers(server_args, max_t: int, max_b: int) -> tuple[Bucketer, Bucketer]:
    """Build token and batch size bucketers."""
    token_paddings = server_args.tpu_compile_tokens
    if token_paddings:
        token_bucketer: Bucketer = Bucketer(
            name="TokenBucketer", buckets=list(token_paddings)
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

"""Host-side batch-composition statistics for TPU profiling spans and JSONL logs.

Kept outside ``sglang_tpu.hooks`` so compilation and auto-profile modules can
share target-policy and ``batch_id`` state without importing
``sglang_tpu/hooks/__init__.py`` (which eagerly imports model-specific hook
modules).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode

# Process-local target compile policy and active scheduler batch_id handoff.
_target_compile_policy: Any = None
_active_batch_id: Optional[int] = None


def set_target_compile_policy(policy: Any) -> None:
    """Register the worker's target ``DecoderTpuCompilePolicy`` for bucket lookup."""
    global _target_compile_policy
    _target_compile_policy = policy


def set_active_batch_id(batch_id: Optional[int]) -> None:
    """Record the scheduler's ``batch_id`` for the immediately following forward."""
    global _active_batch_id
    _active_batch_id = batch_id


def consume_active_batch_id() -> Optional[int]:
    """Pop the scheduler-supplied ``batch_id`` once for the target forward span."""
    global _active_batch_id
    batch_id = _active_batch_id
    _active_batch_id = None
    return batch_id


def _as_cpu_int_list(values: Any) -> Optional[list[int]]:
    """Convert a host sequence or CPU tensor to ``list[int]`` without D2H sync."""
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            return None
        return [int(x) for x in values.tolist()]
    return [int(x) for x in values]


def _extract_min_kv_len(batch: Any, mode: Any) -> int:
    seq_lens = _as_cpu_int_list(getattr(batch, "seq_lens_cpu", None))
    if mode is not None and mode.is_decode() and seq_lens:
        return min(seq_lens)

    # For EXTEND and MIXED, read prefix + extend lengths first: on a MIXED
    # ScheduleBatch, seq_lens_cpu covers only the prefill subset while
    # prefix_lens/extend_lens include all prefill + running decode requests.
    ext_prefix = _as_cpu_int_list(
        getattr(batch, "extend_prefix_lens_cpu", None)
    ) or _as_cpu_int_list(getattr(batch, "prefix_lens", None))
    ext_seq = _as_cpu_int_list(
        getattr(batch, "extend_seq_lens_cpu", None)
    ) or _as_cpu_int_list(getattr(batch, "extend_lens", None))
    if ext_prefix and ext_seq:
        return min(p + q for p, q in zip(ext_prefix, ext_seq))

    return min(seq_lens) if seq_lens else 0


def _determine_phase(num_prefill_tokens: int, num_decode_tokens: int) -> str:
    if num_prefill_tokens > 0 and num_decode_tokens == 0:
        return "PREFILL_ONLY"
    if num_decode_tokens > 0 and num_prefill_tokens == 0:
        return "DECODE_ONLY"
    if num_prefill_tokens > 0 and num_decode_tokens > 0:
        return "MIXED"
    return "UNKNOWN"


def _compute_padded_num_tokens(
    batch: Any,
    policy: Any,
    total_tokens: int,
    num_reqs: int,
) -> int:
    """Return the padded token bucket matching ``CompiledForwardShim``."""
    if total_tokens <= 0 or policy is None:
        return total_tokens
    t_bucketer = getattr(policy, "t_bucketer", None)
    b_bucketer = getattr(policy, "b_bucketer", None)
    mode = getattr(batch, "forward_mode", None)
    if t_bucketer is None or b_bucketer is None or mode is None:
        return total_tokens

    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang_tpu.compilation.compiled_forward_shim import (
        _dp_attn_select_bucket_shape,
        forward_pad_shape,
        speculative_tokens_per_req,
    )

    try:
        if is_dp_attention_enabled():
            counts = (
                _as_cpu_int_list(getattr(batch, "original_global_num_tokens_cpu", None))
                or _as_cpu_int_list(getattr(batch, "global_num_tokens_cpu", None))
                or _as_cpu_int_list(getattr(batch, "global_num_tokens", None))
            )
            max_tokens = max(max(counts), total_tokens) if counts else total_tokens
            pad_t, _ = _dp_attn_select_bucket_shape(
                batch, t_bucketer, b_bucketer, max_tokens, total_tokens, num_reqs
            )
            return pad_t
        pad_t, _ = forward_pad_shape(
            mode,
            total_tokens,
            num_reqs,
            speculative_tokens_per_req(mode, getattr(batch, "spec_info", None)),
            t_bucketer,
            b_bucketer,
            getattr(policy, "extend_bs", None),
        )
        return pad_t
    except (AssertionError, ValueError):
        return total_tokens


def get_batch_composition_stats(
    batch: Any,
    batch_id: int = 0,
) -> dict[str, Any]:
    """Compute host-side batch composition statistics without D2H transfers."""
    mode = getattr(batch, "forward_mode", None)
    raw_bs = getattr(batch, "batch_size", 0)
    bs = raw_bs() if callable(raw_bs) else raw_bs
    num_reqs = bs if isinstance(bs, int) else 0
    num_prefill_tokens = 0
    num_decode_tokens = 0

    ext_seq = _as_cpu_int_list(getattr(batch, "extend_seq_lens_cpu", None))
    if ext_seq is None:
        ext_seq = _as_cpu_int_list(getattr(batch, "extend_lens", None))
    if mode is not None and mode.is_decode():
        num_decode_tokens = num_reqs
    elif mode is not None and mode == ForwardMode.MIXED and ext_seq is not None:
        for q in ext_seq:
            if q == 1:
                num_decode_tokens += 1
            else:
                num_prefill_tokens += q
    elif mode is not None and mode.is_prefill():
        ext_num = getattr(batch, "extend_num_tokens", None)
        if ext_num is not None:
            num_prefill_tokens = int(ext_num)
        elif ext_seq is not None:
            num_prefill_tokens = sum(ext_seq)
        else:
            num_prefill_tokens = num_reqs

    total_tokens = num_prefill_tokens + num_decode_tokens
    attn_backend = getattr(batch, "attn_backend", None)
    model_runner = getattr(attn_backend, "model_runner", None)
    policy = getattr(model_runner, "compile_policy", None) or _target_compile_policy
    padded_tokens = _compute_padded_num_tokens(batch, policy, total_tokens, num_reqs)

    return {
        "num_reqs": num_reqs,
        "num_prefill_tokens": num_prefill_tokens,
        "num_decode_tokens": num_decode_tokens,
        "phase": _determine_phase(num_prefill_tokens, num_decode_tokens),
        "batch_id": batch_id,
        "total_num_scheduled_tokens": total_tokens,
        "padded_total_num_scheduled_tokens": padded_tokens,
        "min_kv_len": _extract_min_kv_len(batch, mode),
    }


def append_span_metadata(span_name: str, metadata: Mapping[str, Any]) -> str:
    """Append ``#k1=v1,k2=v2#`` TraceMe metadata to ``span_name``.

    If ``span_name`` already ends with a ``#...#`` metadata block, merges only
    keys not already present into that trailing block so chained hooks do not
    emit duplicate keys (such as ``num_reqs``).
    """
    if not metadata:
        return span_name
    existing_keys: set[str] = set()
    has_trailing_block = span_name.endswith("#") and "#" in span_name[:-1]
    if has_trailing_block and not span_name.endswith("##"):
        block_start = span_name.rfind("#", 0, -1)
        for entry in span_name[block_start + 1 : -1].split(","):
            if "=" in entry:
                existing_keys.add(entry.split("=", 1)[0])
    pairs = [f"{k}={v}" for k, v in metadata.items() if k not in existing_keys]
    if not pairs:
        return span_name
    encoded = ",".join(pairs)
    if span_name.endswith("##"):
        return f"{span_name[:-1]}{encoded}#"
    if has_trailing_block:
        return f"{span_name[:-1]},{encoded}#"
    return f"{span_name}#{encoded}#"

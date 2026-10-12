# SPDX-License-Identifier: Apache-2.0
"""Torch glue for prefill context parallelism on TPU.

The compiled forward of every context-parallel rank runs on that rank's
zigzag token rows (``cp_layout``): the forward batch is padded to the local
token bucket, the attention backend all-gathers K/V per layer and writes the
whole sequence into its own KV pool, and the logits step reduces the last-token
rows so every rank samples identical tokens. All rank-dependent data enter the
compiled program as tensors built on the host; nothing here bakes the rank
into a traced graph.

Every rank of a context-parallel group must admit the same prefill batch in the
same step, because the per-layer K/V all-gather is a group collective; the ranks
agree by running identical admission code over identical state.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, NamedTuple

import numpy as np
import torch

from sglang.srt.layers.logits_processor import LogitsMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import get_forward_context
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.cp_layout import (
    TpuCpHostLayout,
    build_tpu_cp_layout,
    max_local_rows,
)
from sglang_tpu.kv_row_write import write_rows_torch

_LAYOUT_ATTR = "_tpu_cp_layout"
_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)


class TpuCpBatchLayout(NamedTuple):
    """Per-batch context-parallel layout carried on the forward batch."""

    host: TpuCpHostLayout
    # Device int64 [local_rows_padded]: global token index of each local row
    # (padding rows point at token 0; their contents are never consumed).
    local_rows: torch.Tensor


def get_tpu_cp_layout(forward_batch: Any) -> TpuCpBatchLayout | None:
    """Layout attached by ``prepare_tpu_cp_forward_batch``; None otherwise."""
    return getattr(forward_batch, _LAYOUT_ATTR, None)


# ---------------------------------------------------------------------------
# Forward-batch preparation (host side, before attention metadata)
# ---------------------------------------------------------------------------


def _int_list(values: Any, count: int, name: str) -> list[int]:
    if values is None:
        raise ValueError(f"{name} is required for a context-parallel extend batch.")
    values = [int(x) for x in values]
    if len(values) < count:
        raise ValueError(f"{name} has {len(values)} entries for {count} requests.")
    return values[:count]


def prepare_tpu_cp_forward_batch(
    forward_batch: Any,
    *,
    t_bucketer: Any,
    b_bucketer: Any,
    extend_bs: int | None,
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> tuple[int, int]:
    """Bucket a context-parallel extend batch on its local rows.

    Builds the zigzag layout from the batch's CPU lengths, picks the local
    token bucket from the largest per-rank row count (identical on every
    rank), pads the request-indexed fields to the request bucket, and attaches
    the layout to the batch. Token-indexed fields keep the global row count;
    the compiled-forward shim gathers the local rows from them.

    Returns the real (global) token count and the real request count.
    """
    from sglang_tpu.compilation.compiled_forward_shim import (
        _extend_batch_cap,
        _pad_batch_fields,
    )

    real_q = int(forward_batch.input_ids.shape[0])
    real_b = int(forward_batch.seq_lens.shape[0])
    pad_b = _extend_batch_cap(t_bucketer, b_bucketer, extend_bs)
    if real_b > pad_b:
        raise ValueError(
            f"TPU extend batch size {real_b} exceeds its compile cap {pad_b}. "
            "Set --prefill-max-requests to at least the largest extend batch."
        )
    extend_lens = _int_list(
        forward_batch.extend_seq_lens_cpu, real_b, "extend_seq_lens_cpu"
    )
    prefix_lens = _int_list(
        forward_batch.extend_prefix_lens_cpu, real_b, "extend_prefix_lens_cpu"
    )
    if sum(extend_lens) != real_q:
        raise ValueError(
            f"extend_seq_lens_cpu sums to {sum(extend_lens)} but input_ids has "
            f"{real_q} rows; context parallelism needs a pure extend batch."
        )

    local_rows_padded = t_bucketer.next_bucket(
        max(max_local_rows(extend_lens, cp_size), pad_b)
    )
    host = build_tpu_cp_layout(
        extend_lens=extend_lens,
        prefix_lens=prefix_lens,
        cp_size=cp_size,
        cp_rank=cp_rank,
        pad_b=pad_b,
        local_rows_padded=local_rows_padded,
    )
    local_rows = torch.from_numpy(np.clip(host.local_rows, 0, None)).to(
        device, non_blocking=True
    )

    _pad_batch_fields(
        forward_batch,
        pad_t=None,
        pad_b=pad_b,
        real_b=real_b,
        is_draft_extend_v2=False,
    )
    metadata = forward_batch.attn_cp_metadata
    metadata.per_rank_logical_token = [local_rows_padded] * cp_size
    metadata.per_rank_actual_token = [local_rows_padded] * cp_size
    metadata.max_rank_len = [local_rows_padded] * cp_size

    setattr(
        forward_batch, _LAYOUT_ATTR, TpuCpBatchLayout(host=host, local_rows=local_rows)
    )
    return real_q, real_b


def cp_token_inputs(
    input_ids: torch.Tensor, positions: torch.Tensor, layout: TpuCpBatchLayout
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(global_ids, local_ids, local_positions)`` for a context-parallel forward.

    ``global_ids`` is the whole batch's token ids padded with zeros to
    ``cp_size * local_rows_padded`` rows; it is identical on every rank, so the
    vocab-parallel embedding's all-reduce inside the compiled forward sees the
    same inputs everywhere. ``local_ids`` and ``local_positions`` are this
    rank's zigzag rows.
    """
    rows = layout.local_rows
    global_rows = layout.host.cp_size * layout.host.local_rows_padded
    global_ids = torch.nn.functional.pad(
        input_ids, (0, global_rows - input_ids.shape[0])
    )
    return global_ids, input_ids.index_select(0, rows), positions.index_select(0, rows)


def cp_local_embeds(
    input_embeds: torch.Tensor, layout: TpuCpBatchLayout
) -> torch.Tensor:
    """This rank's zigzag rows of a global ``[tokens, hidden]`` embedding.

    A caller that hands the compiled forward its own embeddings hands them at
    the batch's global token count, while every other token-indexed input the
    graph takes is local. The zero pad matches ``cp_token_inputs``: local rows
    index the ``cp_size * local_rows_padded`` row space, which is wider than
    the batch's own padded token count.
    """
    global_rows = layout.host.cp_size * layout.host.local_rows_padded
    padded = torch.nn.functional.pad(
        input_embeds, (0, 0, 0, global_rows - input_embeds.shape[0])
    )
    return padded.index_select(0, layout.local_rows)


def wrap_cp_embedding_forward(fn: Any, embed_tokens: Any) -> Any:
    """Model forward that embeds the global token ids and runs the model on
    this rank's rows.

    With ``cp_local_rows`` set, ``input_ids`` holds the global (padded) ids:
    they are embedded in full, the local rows are selected from the
    embeddings, and ``fn`` runs on those rows through ``input_embeds``.
    Without it the call passes straight through.
    """

    @functools.wraps(fn)
    def forward(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        *args: Any,
        cp_local_rows: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if cp_local_rows is None:
            return fn(input_ids, positions, forward_batch, *args, **kwargs)
        input_embeds = embed_tokens(input_ids).index_select(0, cp_local_rows)
        local_ids = input_ids.index_select(0, cp_local_rows)
        if args:
            # ``input_embeds`` is fn's 4th positional parameter, so a caller that
            # fills it positionally already occupies args[0]; passing the keyword
            # as well binds it twice. Overwrite the slot rather than appending to
            # it, because dynamo keys its warmup entries on len(args).
            return fn(
                local_ids, positions, forward_batch, input_embeds, *args[1:], **kwargs
            )
        return fn(
            local_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            **kwargs,
        )

    return forward


def pinned_extend_seq_lens_cpu(
    num_reqs: int, *, cp_size: int, cp_active: bool
) -> list[int]:
    """Batch-independent stand-in for ``extend_seq_lens_cpu`` inside the graph.

    The compiled forward decides between the context-parallel and the plain
    program by testing the entries (``cp_strategy.TpuZigzagCPStrategy``); feeding
    one value per program family keeps that test static instead of guarding on
    the live lengths of each batch.
    """
    return [2 * cp_size if cp_active else 0] * num_reqs


# ---------------------------------------------------------------------------
# Per-layer K/V all-gather and full-sequence write (inside the compiled graph)
# ---------------------------------------------------------------------------


def fused_kv_rows(
    k: torch.Tensor, v: torch.Tensor, kv_cache: torch.Tensor
) -> torch.Tensor:
    """Rows ``[T, packed, packing, padded_head_dim]`` of ``k``/``v`` (each
    ``[T, heads, head_dim]``) in the fused pool layout of ``kv_cache``.

    fp8 tensors are handled as their byte view so the interleave, padding and
    reshape run on uint8.
    """
    if k.dtype in _FP8_DTYPES:
        k = k.view(torch.uint8)
        v = v.view(torch.uint8)
    num_tokens, heads, head_dim = k.shape
    packed, packing, padded_head_dim = kv_cache.shape[2:]
    rows = torch.stack([k, v], dim=2).reshape(num_tokens, 2 * heads, head_dim)
    pad_dim = padded_head_dim - head_dim
    pad_heads = packed * packing - 2 * heads
    if pad_dim or pad_heads:
        rows = torch.nn.functional.pad(rows, (0, pad_dim, 0, pad_heads))
    return rows.reshape(num_tokens, packed, packing, padded_head_dim)


def all_gather_and_write_kv(
    *,
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    write_segments: torch.Tensor,
    write_num_segments: torch.Tensor,
    group: Any,
    quantize: (
        Callable[[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] | None
    ) = None,
) -> None:
    """All-gather this layer's local K/V over the context-parallel group and
    write the gathered rows into ``kv_cache`` along ``write_segments``.

    The rows are rank-major (``[cp * T]``); the segments are their runs of
    consecutive KV slots (padding rows have none). The write covers this rank's
    own rows too; the attention kernel rewrites them with identical bytes
    afterwards. Runs are DMAed into the layer buffer in place: an XLA scatter
    here would re-lay out the whole pool around it on every layer.

    For an fp8 pool ``k``/``v`` are the compute-dtype tensors and ``quantize``
    maps the gathered K/V to the pool dtype.
    """
    if quantize is None and kv_cache.dtype in (*_FP8_DTYPES, torch.uint8):
        raise ValueError(
            f"A {kv_cache.dtype} KV pool needs `quantize` to write gathered K/V rows."
        )
    # K/V are gathered as [T, heads, head_dim] in the compute dtype and laid out
    # afterwards: on libtpu 0.0.47 an 8-bit all-gather gave wrong tokens with the
    # SparseCore gather offload on and halted with it off, and an all-gather of the
    # bf16 pool rows stalls warmup at attention CP 8.
    # TODO(yinlin09): gather the fp8 rows directly once that runs correctly.
    k_all = k.new_empty((group.world_size * k.shape[0], *k.shape[1:]))
    v_all = v.new_empty((group.world_size * v.shape[0], *v.shape[1:]))
    group.all_gather_into_tensor(k_all, k.contiguous())
    group.all_gather_into_tensor(v_all, v.contiguous())
    if quantize is not None:
        k_all, v_all = quantize(k_all, v_all)
    gathered = fused_kv_rows(k_all, v_all, kv_cache).contiguous().view(kv_cache.dtype)
    write_rows_torch(write_num_segments, write_segments, gathered, kv_cache)


# ---------------------------------------------------------------------------
# Logits on the last-token rows (inside the compiled graph)
# ---------------------------------------------------------------------------


def cp_last_token_rows(
    hidden_states: torch.Tensor,
    *,
    last_rows: torch.Tensor,
    last_owner_mask: torch.Tensor,
    group: Any,
) -> torch.Tensor:
    """``[pad_b, H]`` last-token hidden states, identical on every rank.

    Each rank contributes the rows it owns (the owner of each request's last
    non-empty block) and zeros elsewhere; a sum all-reduce over the group
    broadcasts them. ``torch.where`` rather than a 0/1 scale keeps an overflow on one
    rank from turning into NaN on the others.
    """
    rows = hidden_states.index_select(0, last_rows)
    zero = torch.zeros((), dtype=rows.dtype, device=rows.device)
    rows = torch.where(last_owner_mask.to(torch.bool)[:, None], rows, zero)
    return group.all_reduce(rows)


class TpuCpLogitsForward:
    """``LogitsProcessor.forward`` replacement for context-parallel prefill.

    For a batch carrying context-parallel attention metadata it reduces the
    last-token rows across ranks and runs the processor on those ``pad_b``
    rows with one query token per request; any other batch goes straight to
    the original forward.
    """

    def __init__(self, logits_processor: Any) -> None:
        self._logits_processor = logits_processor
        self._forward = logits_processor.forward

    def __call__(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        lm_head: Any,
        logits_metadata: Any,
        aux_hidden_states: Any = None,
        hidden_states_before_norm: torch.Tensor | None = None,
    ) -> Any:
        attn_metadata = get_forward_context().attn_backend.forward_metadata
        last_rows = attn_metadata.cp_last_rows
        if last_rows is None or not isinstance(logits_metadata, ForwardBatch):
            return self._forward(
                input_ids,
                hidden_states,
                lm_head,
                logits_metadata,
                aux_hidden_states,
                hidden_states_before_norm,
            )
        rows = cp_last_token_rows(
            hidden_states,
            last_rows=last_rows,
            last_owner_mask=attn_metadata.cp_last_owner_mask,
            group=get_parallel().attn_cp_group,
        )
        metadata = LogitsMetadata.from_forward_batch(
            logits_metadata,
            compute_input_logprobs=self._logits_processor.compute_input_logprobs,
        )
        # One query row per request, so the extend last-index gather is the
        # identity over the reduced rows.
        metadata.extend_seq_lens = torch.ones_like(last_rows, dtype=torch.int32)
        return self._forward(
            input_ids,
            rows,
            lm_head,
            metadata,
            aux_hidden_states,
            hidden_states_before_norm,
        )


def install_cp_logits_forward(logits_processor: Any) -> TpuCpLogitsForward:
    """Install ``TpuCpLogitsForward`` on ``logits_processor`` (idempotent)."""
    current = logits_processor.forward
    if isinstance(current, TpuCpLogitsForward):
        return current
    wrapper = TpuCpLogitsForward(logits_processor)
    logits_processor.forward = wrapper
    return wrapper

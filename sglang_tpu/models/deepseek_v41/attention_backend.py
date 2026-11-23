# SPDX-License-Identifier: Apache-2.0
"""TPU attention backend for DeepSeek-V4.1 (compress ratios 0, 1 and 2).

Core ``MQALayer`` drives each layer; on a non-CUDA platform it calls:

- ``prepare_mqa_qkv``: q_norm, wq_b and RoPE on q; kv_norm and RoPE on kv.
- ``forward_low_ratio_sources`` (ratio 1/2 kv_source and index_source layers):
  compress, publish main KV and index keys, run the indexer selection.
- ``forward``: SWA write of this step's kv, then CSA2 over the SWA window plus the
  selected compressed records of the layer's kv_source. CSA2 inverse-rotates its
  output, so ``apply_mqa_inv_rope`` is the identity.

The control plane is sglang's: per-token request rows, ``out_cache_loc``, the full
page table ``req_to_token[req, ::256] // 256``, the SWA page table through
``full_to_swa_index_mapping`` at 64-token pages, ``c1/c2_out_loc = full_loc // r``
on group-completing tokens (core ``_low_ratio_compression_metadata``) and the
indexer page table from core ``expand_index_page_table``. The ratio-2 compressor
ports core ``_low_ratio_pair_partners`` on the pool's request-scoped pair state,
with the ring write through the donated ``csa2_ops.write_rows``.

Padding tokens (``out_cache_loc == 0`` or past the batch's real tokens) never write
a cache (record -1) and produce zeros.

Under ``--enable-encoder-swa-bounded-replay`` the pool keeps SWA in a request window
(``request_window.py``) and core folds each prefix hit's last <=128 cached tokens in
front of its extend (``encoder_swa_replay.fold_encoder_swa_replay``). Those replay
rows rebuild the window: they write SWA and attend from the hit's replay start
(``encoder_swa_row_floor``), but publish no compressed record and touch no pair
state, since the cache hit already holds both (``encoder_swa_compress_rows``).

Kernels come from ``ops/csa2.py`` and ``ops/indexer.py``; tests replace them
through ``csa2_ops`` / ``indexer_ops``.

IMPORTANT: Do NOT add ``from __future__ import annotations``; see
``sglang_tpu/dsv4/dsv4_attention_backend.py``.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang_tpu.compilation import host_input_mirror
from sglang_tpu.compilation.compiled_forward_shim import _REAL_B_KEY
from sglang_tpu.models.deepseek_v41.decoder_replay import (
    DecoderTail,
    TailView,
    tail_gather_scatter_enabled,
    tail_inverse_rows,
)

# The pool owns the page layout the kernels read; the backend only indexes it.
from sglang_tpu.models.deepseek_v41.memory_pool import (
    FULL_PAGE_TOKENS,
    LOW_RATIOS,
    PAGE_RECORDS,
    SWA_PAGE_TOKENS,
)
from sglang_tpu.models.deepseek_v41.request_window import (
    DSV41TPURequestWindow,
    WindowPlan,
    decode_window_pages,
    extend_window_pages,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# =============================================================================
# RoPE
# =============================================================================


def rope_table(freqs: torch.Tensor) -> torch.Tensor:
    """Lane-packed fp32 ``[ceil(max_pos / 2), 128]`` ``[cos | sin]`` table of a
    layer's ``freqs_cis`` (complex, or already converted by
    ``_prewarm_model_caches``); see ``get_or_build_rope_lane_table``."""
    if freqs.is_complex():
        from sglang_tpu.dsv4.dsv4_memory_pool import get_or_build_rope_lane_table

        return get_or_build_rope_lane_table(freqs)
    return freqs


def rope_reference(
    x: torch.Tensor,
    positions: torch.Tensor,
    table: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Core ``rope_tail`` numerics on a plain or lane-packed ``[cos | sin]``
    table: rotate the last ``rope_dim`` features as interleaved (even, odd)
    complex pairs, in fp32."""
    from sglang_tpu.dsv4.dsv4_memory_pool import lookup_cos_sin

    rows = lookup_cos_sin(table, positions).float()
    rope_dim = rows.shape[-1]
    half = rope_dim // 2
    cos, sin = rows[:, :half], rows[:, half:]
    if inverse:
        sin = -sin
    shape = (x.shape[0], *([1] * (x.ndim - 2)), half)
    cos, sin = cos.view(shape), sin.view(shape)
    tail = x[..., -rope_dim:].float().unflatten(-1, (half, 2))
    even, odd = tail[..., 0], tail[..., 1]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1)
    return torch.cat((x[..., :-rope_dim], rotated.flatten(-2).to(x.dtype)), -1)


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """The V4 Pallas RoPE on TPU (same pairing and table), ``rope_reference`` elsewhere."""
    table = rope_table(freqs)
    if x.device.type == "tpu":
        from sglang_tpu.dsv4.dsv4_attention_backend import apply_tpu_dsv4_rope

        return apply_tpu_dsv4_rope(x, positions, table)
    return rope_reference(x, positions, table)


# =============================================================================
# Per-step metadata
# =============================================================================


class LowRatioMetadata(NamedTuple):
    # [T] int32 compressed write record: full_loc // r on group-completing real
    # tokens, else -1.
    out_loc: torch.Tensor
    # [T] int32 visible compressed records (pos + 1) // r; 0 on padding.
    lengths: torch.Tensor
    # [B, C * 256 / r / 128] int32 indexer page table.
    index_pages: torch.Tensor


# A dataclass like DSV4TPUForwardMetadata: layers set fields on it inside the
# compiled forward, where msgspec.Struct mutation is untested.
@dataclass
class DSV41TPUForwardMetadata:
    """Built on the host once per forward; shared by every layer of the step."""

    # [T] int32 token positions (padding rows as the batch padded them).
    positions: torch.Tensor
    # [B + 1] int32 request row boundaries; each request's rows are contiguous.
    query_start_loc: torch.Tensor
    # [T] int32 batch row of each token, -1 for padding; the indexer's requests.
    token_rows: torch.Tensor
    # [T] int32 req_pool_idx of each token (core token_req_indices), 0 for padding.
    token_req_pool: torch.Tensor
    # [T] bool: a real token with a cache slot.
    valid: torch.Tensor
    # [T] bool: a valid token that compresses (not a folded replay row).
    compress_valid: torch.Tensor
    # [T] int32 scheduler full_loc; TPU lowering rejects 64-bit integer vectors.
    out_cache_loc: torch.Tensor
    # [B, C] int32 full page ids, C = ceil(context_len / 256); also main KV pages.
    full_pages: torch.Tensor
    # [B, 4C] int32 SWA 64-token page ids; 0 is the null page (evicted or unmapped).
    swa_pages: torch.Tensor
    # [B] int32 earliest visible SWA position: a folded hit's replay start, else 0.
    swa_minimum: torch.Tensor
    low_ratio: Dict[int, LowRatioMetadata]
    # Direct tracer only: (local row ids, staged rows) of the host-staged Engram
    # lookup (engram_host), carried here as graph inputs because the tracer
    # rebuilds ForwardBatch from its dataclass fields only. None otherwise.
    engram: Optional[tuple] = None
    # Request-window row moves of an extend forward (history in, window out).
    swa_window_plan: Optional[WindowPlan] = None
    # Decoder SWA bounded replay (decoder_replay.py): the decoder rows' own
    # metadata, and [D] int32 full-batch rows to gather them from (padding rows:
    # a row the decoder does not keep). None outside a decoder-replay extend.
    tail: Optional["DSV41TPUForwardMetadata"] = None
    tail_gather: Optional[torch.Tensor] = None
    # [T] int32: the decoder row each full-batch row comes back from, or -1 (a row
    # the decoder skipped), so the decoder rows come back by a gather instead of
    # a scatter. None with SGLANG_TPU_DSV41_TAIL_GATHER_SCATTER=0.
    tail_inverse: Optional[torch.Tensor] = None
    # The decoder row bucket. Static pytree aux data, so the traced structure
    # carries it.
    tail_rows: Optional[int] = None

    # Filled lazily during the forward.
    swa_meta: Optional[torch.Tensor] = None
    swa_bounds: Optional[torch.Tensor] = None
    swa_write_records: Optional[torch.Tensor] = None
    # Latest index_source selection per ratio, read by every later layer of that
    # ratio (core sparse_page_indices(ratio)); [T, 512] int32, valid prefix.
    selected: Dict[int, torch.Tensor] = field(default_factory=dict)
    # Candidate blocks published by config.candidate_source_layer_id.
    candidates: Optional[torch.Tensor] = None
    # (ratio, index source layer) -> indexer_ops.prepare_candidates result,
    # gathered once and read by every candidate consumer of that source.
    prepared_candidates: Dict[tuple, tuple] = field(default_factory=dict)
    # Gather group's first layer -> its gathered main KV (csa2_ops.attention with
    # keep_gathered), read by the group's later layers and dropped at its last.
    gathered_main: Dict[int, tuple] = field(default_factory=dict)


# Host-built fields: the direct tracer's graph inputs. The rest is per-forward
# scratch the layers fill, so it is left out of the pytree and every unflatten
# starts it empty: warmup and serving then flatten to the same structure.
_METADATA_INPUT_FIELDS = (
    "positions",
    "query_start_loc",
    "token_rows",
    "token_req_pool",
    "valid",
    "compress_valid",
    "out_cache_loc",
    "full_pages",
    "swa_pages",
    "swa_minimum",
    "low_ratio",
    "engram",
    "swa_window_plan",
    "tail",
    "tail_gather",
    "tail_inverse",
)


def _fused_ep_enabled() -> bool:
    from sglang_tpu.models.deepseek_v41.fused_ep_op import is_dsv41_fused_ep_enabled

    return is_dsv41_fused_ep_enabled()


def _register_metadata_pytree() -> None:
    try:
        from jax import tree_util
    except ModuleNotFoundError:  # CPU unit tests without JAX
        return
    # Keyed by attribute: direct_compile resolves every leaf of a deferred call
    # through its key path and accepts only GetAttr / Dict / Sequence keys.
    tree_util.register_pytree_with_keys(
        DSV41TPUForwardMetadata,
        lambda m: (
            tuple(
                (tree_util.GetAttrKey(f), getattr(m, f)) for f in _METADATA_INPUT_FIELDS
            ),
            m.tail_rows,
        ),
        lambda tail_rows, children: DSV41TPUForwardMetadata(
            **dict(zip(_METADATA_INPUT_FIELDS, children)), tail_rows=tail_rows
        ),
    )


_register_metadata_pytree()


def _host_int64(values: torch.Tensor) -> torch.Tensor:
    """``values`` as int64 on the host, copied first and cast there.

    ``.to("cpu", torch.int64)`` on a device tensor of another dtype casts on
    the device before the copy: a program per shape, so a compile per batch
    size or token count at serving time.
    """
    return values.detach().to("cpu").to(torch.int64)


def _host_extend_positions(
    forward_batch: ForwardBatch, num_tokens: int
) -> Optional[torch.Tensor]:
    """[num_tokens] int64 positions of a pure extend, from its host extend lists.

    Core computes them on the device at the live token count, so reading them
    back (an int64 copy to the host) is a program per prompt length under
    decoder replay. They are prefix + arange(extend) per request, then zeros in
    the padding rows. None when the host lists cannot reproduce them.
    """
    if forward_batch.forward_mode != ForwardMode.EXTEND:
        return None
    # Explicit positions (spec inputs) are not prefix + arange.
    if (
        getattr(getattr(forward_batch, "spec_info", None), "positions", None)
        is not None
    ):
        return None
    prefix = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    extend = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if prefix is None or extend is None or len(prefix) != len(extend):
        return None
    if sum(int(n) for n in extend) > num_tokens:
        return None
    positions = torch.zeros(num_tokens, dtype=torch.int64)
    offset = 0
    for start, count in zip(prefix, extend):
        start, count = int(start), int(count)
        positions[offset : offset + count] = torch.arange(start, start + count)
        offset += count
    return positions


def _extend_positions(forward_batch: ForwardBatch, num_tokens: int) -> torch.Tensor:
    from sglang_tpu.envs import envs as tpu_envs

    # The compile policy's host mirror is the exact uploaded tensor: prefer it.
    # Positions derived from the extend length lists serve only batches it did
    # not mirror, as those lists can be stale where the mirror is not.
    if host_input_mirror.host_copy(forward_batch, "positions") is not None:
        return _host_or_read(forward_batch, "positions", num_tokens)
    positions = _host_extend_positions(forward_batch, num_tokens)
    if positions is None:
        return _as_cpu_int64(forward_batch.positions, num_tokens)
    if tpu_envs.SGLANG_TPU_DSV41_CHECK_HOST_POSITIONS.get():
        read = _as_cpu_int64(forward_batch.positions, num_tokens)
        if not torch.equal(read, positions):
            raise AssertionError(
                "DSV4.1 host extend positions differ from the device: "
                f"host={positions.tolist()} device={read.tolist()}"
            )
    return positions


def _as_cpu_int64(values, length: int) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        out = _host_int64(values)
    else:
        out = torch.tensor(list(values), dtype=torch.int64)
    out = out[:length]
    if out.shape[0] < length:
        out = F.pad(out, (0, length - out.shape[0]))
    return out


def _host_or_read(forward_batch: ForwardBatch, name: str, length: int) -> torch.Tensor:
    """[length] int64 host value of ``forward_batch.<name>``.

    The compile policy keeps the host tensor it uploaded at the bucket shape
    (``host_input_mirror``); that copy replaces the blocking device read. Fields it
    did not upload, or that changed since, are read from the device as before.
    """
    from sglang_tpu.envs import envs as tpu_envs

    device_value = getattr(forward_batch, name)
    host_value = host_input_mirror.host_copy(forward_batch, name)
    if host_value is None:
        return _as_cpu_int64(device_value, length)
    host_value = _as_cpu_int64(host_value, length)
    if tpu_envs.SGLANG_TPU_DSV41_CHECK_HOST_METADATA.get():
        read = _as_cpu_int64(device_value, length)
        if not torch.equal(read, host_value):
            raise AssertionError(
                f"DSV4.1 host {name} differs from the device value: "
                f"host={host_value.tolist()} device={read.tolist()}"
            )
        _HOST_CHECKS[name] = _HOST_CHECKS.get(name, 0) + 1
        total = sum(_HOST_CHECKS.values())
        if total == 1 or total % 1000 == 0:
            logger.info("DSV4.1 host metadata matched the device: %s", _HOST_CHECKS)
    return host_value


# Host copies compared with the device under SGLANG_TPU_DSV41_CHECK_HOST_METADATA.
_HOST_CHECKS: Dict[str, int] = {}


def _request_token_counts(forward_batch: ForwardBatch, bs: int) -> torch.Tensor:
    """[B] int64 query rows of each batch row; bucket-padding rows get 0."""
    mode = forward_batch.forward_mode
    real_bs = min(bs, int(getattr(forward_batch, _REAL_B_KEY, bs)))
    counts = torch.zeros(bs, dtype=torch.int64)
    if mode.is_idle():
        return counts
    if mode.is_decode():
        counts[:real_bs] = 1
        return counts
    if mode.is_target_verify():
        # extend_seq_lens is not set on verify batches; every request has the same width.
        spec_info = forward_batch.spec_info
        assert spec_info is not None, "a verify batch carries spec_info"
        counts[:real_bs] = int(spec_info.draft_token_num)
        return counts
    if mode.is_extend():
        lens = forward_batch.extend_seq_lens_cpu
        if lens is None:
            lens = forward_batch.extend_seq_lens
        assert lens is not None, "an extend batch carries extend_seq_lens"
        counts[:real_bs] = _as_cpu_int64(lens, bs)[:real_bs]
        return counts
    raise NotImplementedError(
        f"DeepSeek-V4.1 TPU attention serves extend, decode and target verify, not {mode.name}"
    )


def _full_page_table(
    req_to_token: torch.Tensor, req_pool_indices: torch.Tensor, columns: int
) -> torch.Tensor:
    """[B, columns] int64: page p of a request is req_to_token[req, 256 p] // 256."""
    starts = req_to_token[req_pool_indices.to(req_to_token.device), ::FULL_PAGE_TOKENS]
    starts = _host_int64(starts[:, :columns])
    if starts.shape[1] < columns:
        starts = F.pad(starts, (0, columns - starts.shape[1]))
    return starts // FULL_PAGE_TOKENS


def _swa_page_table(
    full_pages: torch.Tensor, full_to_swa_index_mapping: Optional[torch.Tensor]
) -> torch.Tensor:
    """[B, 4C] int64: each logical page's SWA start split into 64-token pages.

    A page whose SWA peer is gone (mapping 0) maps to the null page 0."""
    if full_to_swa_index_mapping is None:
        swa_starts = full_pages * FULL_PAGE_TOKENS
    else:
        mapping = full_to_swa_index_mapping
        lookup = (full_pages * FULL_PAGE_TOKENS).clamp(0, mapping.shape[0] - 1)
        swa_starts = _host_int64(mapping[lookup.to(mapping.device)])
    sub_pages = FULL_PAGE_TOKENS // SWA_PAGE_TOKENS
    pages = swa_starts[..., None] // SWA_PAGE_TOKENS + torch.arange(sub_pages)
    pages = torch.where((swa_starts > 0)[..., None], pages, 0)
    return pages.reshape(full_pages.shape[0], -1)


def _low_ratio_metadata(
    *,
    ratio: int,
    positions: torch.Tensor,
    out_cache_loc: torch.Tensor,
    valid: torch.Tensor,
    compress_valid: torch.Tensor,
    full_pages: torch.Tensor,
) -> LowRatioMetadata:
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _low_ratio_compression_metadata,
    )
    from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table

    seq_lens_casual = positions + 1
    out_loc, _ = _low_ratio_compression_metadata(ratio, seq_lens_casual, out_cache_loc)
    return LowRatioMetadata(
        out_loc=out_loc.masked_fill(~compress_valid, -1).to(torch.int32),
        lengths=torch.where(valid, seq_lens_casual // ratio, 0).to(torch.int32),
        index_pages=expand_index_page_table(
            full_pages.to(torch.int32),
            full_page_size=FULL_PAGE_TOKENS,
            compress_ratio=ratio,
            index_page_size=PAGE_RECORDS,
        ).to(torch.int32),
    )


def _fold_masks(
    forward_batch: ForwardBatch,
    *,
    num_tokens: int,
    query_start_loc: torch.Tensor,
    valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """([B] int64 SWA floor, [T] compress mask) of a folded extend; padding rows,
    past the folded rows, have floor 0."""
    bs = query_start_loc.shape[0] - 1
    row_floor = forward_batch.encoder_swa_row_floor
    if row_floor is None:
        return torch.zeros(bs, dtype=torch.int64), valid
    floor_rows = _as_cpu_int64(row_floor, num_tokens)
    first_rows = query_start_loc[:-1].clamp(max=max(num_tokens - 1, 0))
    counts = query_start_loc[1:] - query_start_loc[:-1]
    floor = torch.where(counts > 0, floor_rows[first_rows], 0)
    compress = torch.zeros(num_tokens, dtype=torch.bool)
    keep = _host_int64(forward_batch.encoder_swa_compress_rows)
    compress[keep] = True
    return floor, valid & compress


def _window_swa_table(
    *,
    forward_batch: ForwardBatch,
    window: DSV41TPURequestWindow,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    req_pool_indices: torch.Tensor,
    minimum: torch.Tensor,
    columns: int,
) -> Tuple[torch.Tensor, Optional[WindowPlan]]:
    """[B, columns] int64 SWA table of a request-window forward and its row moves."""
    if not forward_batch.forward_mode.is_extend():
        return decode_window_pages(req_pool_indices, columns), None
    counts = query_start_loc[1:] - query_start_loc[:-1]
    last = max(positions.shape[0] - 1, 0)
    first = torch.where(counts > 0, positions[query_start_loc[:-1].clamp(max=last)], 0)
    return extend_window_pages(
        slots=req_pool_indices,
        first=first,
        counts=counts,
        minimum=minimum,
        columns=columns,
        base_page=window.workspace_first_page,
        capacity_pages=window.workspace_pages,
    )


def build_forward_metadata(
    forward_batch: ForwardBatch,
    *,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    request_window: Optional[DSV41TPURequestWindow],
    swa_ring_valid_from: Optional[torch.Tensor],
    max_context_len: int,
    low_ratios: Sequence[int],
    device: torch.device,
    decoder_replay: bool = False,
    decoder_low_ratios: Sequence[int] = (),
) -> DSV41TPUForwardMetadata:
    """Host-side metadata from the sglang batch, moved to ``device`` once.

    ``decoder_replay``: also build the decoder rows' metadata (``tail``)."""
    bs = int(forward_batch.batch_size)
    num_tokens = forward_batch.positions.shape[0]
    positions = _extend_positions(forward_batch, num_tokens)
    out_cache_loc = _host_or_read(forward_batch, "out_cache_loc", num_tokens)

    counts = _request_token_counts(forward_batch, bs)
    query_start_loc = F.pad(torch.cumsum(counts, 0), (1, 0))
    real_tokens = int(query_start_loc[-1])
    assert real_tokens <= num_tokens, f"{real_tokens=} > {num_tokens=}"
    req_pool_indices = _host_or_read(forward_batch, "req_pool_indices", bs)
    host_input_mirror.release(forward_batch)
    token_rows, valid, token_req_pool = _token_rows(
        counts, out_cache_loc, req_pool_indices, num_tokens
    )

    swa_minimum, compress_valid = _fold_masks(
        forward_batch,
        num_tokens=num_tokens,
        query_start_loc=query_start_loc,
        valid=valid,
    )

    columns = max(1, -(-int(max_context_len) // FULL_PAGE_TOKENS))
    full_pages = _full_page_table(req_to_token, req_pool_indices, columns)

    def put(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, non_blocking=True)

    full_pages_device = put(full_pages.to(torch.int32))
    metadata = _assemble_metadata(
        forward_batch=forward_batch,
        positions=positions,
        query_start_loc=query_start_loc,
        token_rows=token_rows,
        token_req_pool=token_req_pool,
        valid=valid,
        compress_valid=compress_valid,
        out_cache_loc=out_cache_loc,
        req_pool_indices=req_pool_indices,
        swa_minimum=swa_minimum,
        full_pages=full_pages,
        full_pages_device=full_pages_device,
        request_window=request_window,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        swa_ring_valid_from=swa_ring_valid_from,
        columns=columns,
        low_ratios=low_ratios,
        put=put,
    )
    if not decoder_replay:
        return metadata

    from sglang_tpu.models.deepseek_v41.decoder_replay import plan_tail_rows

    if getattr(forward_batch, "decoder_rows_bucket", None) == 0:
        # No rank keeps a decoder row: the model skips the decoder layers.
        metadata.tail_rows = 0
        return metadata
    plan = plan_tail_rows(
        fill_lens=getattr(forward_batch, "decoder_fill_lens", None),
        query_start_loc=query_start_loc,
        positions=positions,
        valid=valid,
        floors=swa_minimum,
        bucket=getattr(forward_batch, "decoder_rows_bucket", None),
        warmup=getattr(forward_batch, "decoder_replay_warmup", False),
    )
    tail_positions = positions[plan.gather]
    tail_qsl = F.pad(torch.cumsum(plan.counts, 0), (1, 0))
    real = torch.arange(plan.gather.shape[0]) < plan.num_real
    tail_loc = torch.where(real, out_cache_loc[plan.gather], 0)
    tail_token_rows, tail_valid, tail_req_pool = _token_rows(
        plan.counts, tail_loc, req_pool_indices, plan.gather.shape[0]
    )
    metadata.tail_rows = int(plan.gather.shape[0])
    metadata.tail = _assemble_metadata(
        forward_batch=forward_batch,
        positions=torch.where(real, tail_positions, 0),
        query_start_loc=tail_qsl,
        token_rows=tail_token_rows,
        token_req_pool=tail_req_pool,
        valid=tail_valid,
        # The decoder layers own no compressor.
        compress_valid=torch.zeros_like(tail_valid),
        out_cache_loc=tail_loc,
        req_pool_indices=req_pool_indices,
        swa_minimum=plan.replay_starts,
        full_pages=full_pages,
        full_pages_device=full_pages_device,
        request_window=request_window,
        full_to_swa_index_mapping=full_to_swa_index_mapping,
        swa_ring_valid_from=None,
        columns=columns,
        low_ratios=decoder_low_ratios,
        put=put,
    )
    metadata.tail_gather = put(plan.gather.to(torch.int32))
    if tail_gather_scatter_enabled():
        metadata.tail_inverse = put(
            tail_inverse_rows(plan.gather, plan.num_real, int(positions.shape[0]))
        )
    return metadata


def _token_rows(
    counts: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_pool_indices: torch.Tensor,
    num_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """([T] batch row or -1, [T] valid, [T] req_pool_idx or 0) of request-
    contiguous rows; rows past the counts and rows without a slot are invalid."""
    bs = counts.shape[0]
    real_tokens = int(counts.sum())
    token_rows = torch.full((num_tokens,), -1, dtype=torch.int64)
    token_rows[:real_tokens] = torch.repeat_interleave(torch.arange(bs), counts)
    valid = (token_rows >= 0) & (out_cache_loc > 0)
    token_rows = token_rows.masked_fill(~valid, -1)
    token_req_pool = req_pool_indices[token_rows.clamp_min(0)].masked_fill(~valid, 0)
    return token_rows, valid, token_req_pool


def _assemble_metadata(
    *,
    forward_batch: ForwardBatch,
    positions: torch.Tensor,
    query_start_loc: torch.Tensor,
    token_rows: torch.Tensor,
    token_req_pool: torch.Tensor,
    valid: torch.Tensor,
    compress_valid: torch.Tensor,
    out_cache_loc: torch.Tensor,
    req_pool_indices: torch.Tensor,
    swa_minimum: torch.Tensor,
    full_pages: torch.Tensor,
    full_pages_device: torch.Tensor,
    request_window: Optional[DSV41TPURequestWindow],
    full_to_swa_index_mapping: Optional[torch.Tensor],
    swa_ring_valid_from: Optional[torch.Tensor],
    columns: int,
    low_ratios: Sequence[int],
    put,
) -> DSV41TPUForwardMetadata:
    """SWA table, request-window moves and low-ratio tables of one row set."""
    window_plan = None
    if swa_ring_valid_from is not None:
        # DSpark draft ring: every logical page is the slot's ring; rows before
        # valid_from belong to the slot's previous owner.
        swa_pages = decode_window_pages(
            req_pool_indices, columns * (FULL_PAGE_TOKENS // SWA_PAGE_TOKENS)
        )
        swa_minimum = torch.maximum(swa_minimum, swa_ring_valid_from[req_pool_indices])
    elif request_window is None:
        swa_pages = _swa_page_table(full_pages, full_to_swa_index_mapping)
    else:
        swa_pages, window_plan = _window_swa_table(
            forward_batch=forward_batch,
            window=request_window,
            positions=positions,
            query_start_loc=query_start_loc,
            req_pool_indices=req_pool_indices,
            minimum=swa_minimum,
            columns=columns * (FULL_PAGE_TOKENS // SWA_PAGE_TOKENS),
        )
    low_ratio = {
        ratio: _low_ratio_metadata(
            ratio=ratio,
            positions=positions,
            out_cache_loc=out_cache_loc,
            valid=valid,
            compress_valid=compress_valid,
            full_pages=full_pages,
        )
        for ratio in low_ratios
    }
    return DSV41TPUForwardMetadata(
        positions=put(positions.to(torch.int32)),
        query_start_loc=put(query_start_loc.to(torch.int32)),
        token_rows=put(token_rows.to(torch.int32)),
        token_req_pool=put(token_req_pool.to(torch.int32)),
        valid=put(valid),
        compress_valid=put(compress_valid),
        out_cache_loc=put(out_cache_loc.to(torch.int32)),
        full_pages=full_pages_device,
        swa_pages=put(swa_pages.to(torch.int32)),
        swa_minimum=put(swa_minimum.to(torch.int32)),
        low_ratio={
            ratio: LowRatioMetadata(*(put(t) for t in meta))
            for ratio, meta in low_ratio.items()
        },
        swa_window_plan=(
            None if window_plan is None else WindowPlan(*(put(t) for t in window_plan))
        ),
    )


# =============================================================================
# Backend
# =============================================================================


_MAX_BLOCK_ROWS = 64


def _check_supported(config) -> None:
    from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

    parallel = get_parallel()
    for name in ("pp_size", "attn_cp_size", "attn_dcp_size", "attn_tp_size"):
        if getattr(parallel, name) != 1:
            raise NotImplementedError(
                f"DeepSeek-V4.1 TPU attention needs {name} == 1 "
                "(attention TP1 per DP rank)"
            )
    spec = get_spec()
    algorithm = SpeculativeAlgorithm.from_string(spec.speculative_algorithm)
    if not (algorithm.is_none() or algorithm.is_dspark()):
        raise NotImplementedError(
            f"DeepSeek-V4.1 TPU attention serves DSPARK speculation only, not {algorithm}"
        )
    # CSA2 serves non-causal requests of at most 64 queries; the draft block and
    # the verify window are gamma + 1 rows.
    rows = spec.speculative_num_draft_tokens
    if algorithm.is_dspark() and rows is not None and rows > _MAX_BLOCK_ROWS:
        raise ValueError(
            f"DSpark verify of {rows} rows exceeds "
            f"the {_MAX_BLOCK_ROWS} non-causal CSA2 queries"
        )
    if get_exec().features.enable_decoder_swa_bounded_replay:
        _check_decoder_replay(config)
    if set(config.compress_ratios) - {0, *LOW_RATIOS}:
        raise ValueError(
            "DeepSeek-V4.1 TPU attention serves ratios 0/1/2, got "
            f"{sorted(set(config.compress_ratios))}"
        )
    if int(config.sliding_window) != 128 or int(config.head_dim) != 512:
        raise ValueError("CSA2 requires sliding_window 128 and head_dim 512")
    _check_index_sources(config)


def _check_decoder_replay(config) -> None:
    """The decoder layers (past the last kv_source layer) may own no compressor,
    index keys or Engram: the decoder rows read only the sources' records."""
    late = max(config.kv_source_layer_ids) + 1
    owners = [
        layer_id
        for layer_id in (
            *config.kv_source_layer_ids,
            *(getattr(config, "engram_layer_ids", None) or ()),
        )
        if layer_id >= late
    ]
    if owners or set(config.compress_ratios[late : config.num_hidden_layers]) - {0, 1}:
        raise ValueError(
            "decoder SWA bounded replay needs stateless decoder layers; "
            f"layers {owners} past {late - 1} own state"
        )
    candidate = getattr(config, "candidate_source_layer_id", -1)
    if candidate >= late:
        raise ValueError(
            f"the candidate source layer {candidate} is a decoder layer (>= {late})"
        )


def _check_index_sources(config) -> None:
    """Every ratio 1/2 layer attends with the selection of the latest preceding
    index_source of its ratio (core reads sparse_page_indices(ratio))."""
    ratios = list(config.compress_ratios)
    for layer_id, ratio in enumerate(ratios):
        if ratio in LOW_RATIOS and not any(
            s <= layer_id and ratios[s] == ratio for s in config.index_source_layer_ids
        ):
            raise ValueError(f"layer {layer_id} has no ratio-{ratio} index source")


def main_gather_groups(
    *,
    compress_ratios: Sequence[int],
    index_source_layer_ids: Sequence[int],
    source_layer_of,
    layer_ids: Sequence[int],
) -> Dict[int, Tuple[int, int]]:
    """layer_id -> (first, last) layer of its main-KV gather group.

    Ratio 1/2 layers that read one main KV source with one selection gather the
    same records: only kv_source layers publish main KV and only index_source
    layers select, both before their own attention, so a group's first layer
    is the one that changes either.
    """
    members: Dict[Tuple[int, int], list] = {}
    for layer_id in layer_ids:
        ratio = compress_ratios[layer_id]
        if ratio not in LOW_RATIOS:
            continue
        selector = max(
            s
            for s in index_source_layer_ids
            if s <= layer_id and compress_ratios[s] == ratio
        )
        key = (source_layer_of(layer_id), selector)
        members.setdefault(key, []).append(layer_id)
    return {
        layer_id: (layers[0], layers[-1])
        for layers in members.values()
        for layer_id in layers
    }


class DSV41TPUAttnBackend(AttentionBackend):
    """DeepSeek-V4.1 CSA2 attention on sglang request/page metadata."""

    # Encoder replay rows run inside the extend forward (see the module docstring).
    folds_encoder_swa_replay: bool = True
    # Set by decoder_replay.clear_full_layout_head around the model forward.
    in_model_forward: bool = False

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        model_config = model_runner.model_config
        config = model_config.hf_text_config
        _check_supported(config)
        self.model_runner = model_runner
        # build_forward_metadata reads the host copies the forwards record.
        from sglang_tpu.compilation import host_input_mirror

        host_input_mirror.register_consumer()
        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.request_window = self.token_to_kv_pool.request_window
        self.max_context_len = int(model_config.context_len)
        self.page_size = FULL_PAGE_TOKENS
        self.index_topk = int(config.index_topk)
        self.forward_metadata: Optional[DSV41TPUForwardMetadata] = None
        # The DSpark draft runs TARGET_VERIFY as one non-causal block per request.
        self.is_dspark_draft = (
            model_runner.is_draft_worker and model_runner.spec_algorithm.is_dspark()
        )
        # The draft carries the target config, but its stages are all ratio 0.
        self.low_ratios = (
            ()
            if self.is_dspark_draft
            else tuple(r for r in LOW_RATIOS if r in config.compress_ratios)
        )
        # The draft has no index layers, hence no candidate consumers.
        self.last_candidate_consumer = (
            {} if self.is_dspark_draft else self._last_candidate_consumers(config)
        )
        # layer_id -> (group's first layer, group's last layer) of main-KV gathers.
        self.gather_groups = (
            {}
            if self.is_dspark_draft
            else main_gather_groups(
                compress_ratios=config.compress_ratios,
                index_source_layer_ids=config.index_source_layer_ids,
                source_layer_of=self.token_to_kv_pool.source_layer_of,
                layer_ids=[
                    i
                    for i, item in enumerate(self.token_to_kv_pool.layer_mapping)
                    if item is not None
                ],
            )
        )
        from sglang.srt.runtime_context import get_exec

        # Decoder SWA bounded replay (decoder_replay.py): target extends only.
        self.decoder_replay = (
            get_exec().features.enable_decoder_swa_bounded_replay
            and not self.is_dspark_draft
        )
        late = max(config.kv_source_layer_ids or (0,)) + 1
        self.decoder_low_ratios = tuple(
            r
            for r in self.low_ratios
            if r in config.compress_ratios[late : config.num_hidden_layers]
        )
        if self.decoder_replay:
            # Decoder-replay tails gather per layer (_attend_main), so no gather
            # group may carry an encoder layer's gather into the decoder layers.
            self.gather_groups = {
                layer_id: (first, min(last, late - 1))
                for layer_id, (first, last) in self.gather_groups.items()
                if layer_id < late
            }
        # layer_id -> fp32 [cos | sin] table; forward only sees the RadixAttention.
        # Attention layer id -> the module owning that layer's freqs_cis buffer.
        # The table is read through the owner at call time, so the direct tracer
        # sees the buffer it lifted, not a tensor captured at init.
        self.rope_owners: Dict[int, torch.nn.Module] = {}
        self.csa2_ops, self.indexer_ops = self._register_ops()
        # CSA2's fast main-record path reads exactly this many selected ids.
        self.selection_width = self.csa2_ops.MAX_SELECTED
        if self.index_topk > self.selection_width:
            raise ValueError(
                f"CSA2 attends at most {self.selection_width} compressed records, "
                f"got index_topk={self.index_topk}"
            )
        self._prewarm_model_caches()
        # The DSpark draft has no inner .model, Engram layers or image-token mask.
        if not self.is_dspark_draft:
            self._drop_text_only_image_mask(model_runner)

    @staticmethod
    def _drop_text_only_image_mask(model_runner) -> None:
        """Without multimodal serving no prompt holds image tokens, so the forward's
        image-token torch.where in each Engram gate is the identity.

        The vision tower, MoE correction bias and hasher read vision_n_layers at
        construction, already done; the forward reads it, and each Engram layer
        its image_token_id, only for that where, whose fused select has no valid
        TPU window ("Cannot find a valid window").
        """
        model = model_runner.model
        if model is None or model_runner.model_config.is_multimodal:
            return
        config = model.model.config
        if config.vision_n_layers > 0:
            config.vision_n_layers = 0
        for layer in model.model.layers:
            if layer.engram is not None:
                layer.engram.image_token_id = None

    @staticmethod
    def _register_ops():
        # Registration must precede torch.compile tracing.
        from sglang_tpu.dsv4.dsv4_attention_backend import _register_rope_op
        from sglang_tpu.models.deepseek_v41.ops import csa2, indexer

        _register_rope_op()  # apply_rope's Pallas RoPE op
        csa2.register_csa2_ops()
        indexer.register_indexer_ops()
        return csa2, indexer

    def _prewarm_model_caches(self) -> None:
        """Convert complex freqs_cis to lane-packed [cos | sin] tables outside any
        trace and record each attention layer's table owner by layer id."""
        model = self.model_runner.model
        if model is None:
            return
        for module in model.modules():
            freqs = module._buffers.get("freqs_cis")
            if freqs is None:
                continue
            table = rope_table(freqs)
            if freqs.is_complex():
                module.register_buffer("freqs_cis", table, persistent=False)
            attn = module._modules.get("attn_mqa")
            if attn is not None:
                self.rope_owners[attn.layer_id] = module

    # ------------------------------------------------------------------ #
    # Runtime hooks (mirror DSV4TPUAttnBackend)
    # ------------------------------------------------------------------ #

    def support_triton(self) -> bool:
        return False

    def get_or_create_metadata(
        self, forward_batch: ForwardBatch
    ) -> DSV41TPUForwardMetadata:
        if self.forward_metadata is None:
            self.init_forward_metadata(forward_batch)
        return self.forward_metadata

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = build_forward_metadata(
            forward_batch,
            req_to_token=self.req_to_token,
            full_to_swa_index_mapping=self.token_to_kv_pool.full_to_swa_index_mapping,
            request_window=self.request_window,
            swa_ring_valid_from=self.token_to_kv_pool.swa_valid_from,
            max_context_len=self.max_context_len,
            low_ratios=self.low_ratios,
            device=self.device,
            decoder_replay=self._uses_decoder_replay(forward_batch),
            decoder_low_ratios=self.decoder_low_ratios,
        )

    def scheduled_decoder_rows(self, batch, *, folds: bool) -> int:
        """This rank's MLP-sync decoder row count (core dp_attn asks a backend
        with ``decoder_replay``)."""
        from sglang_tpu.models.deepseek_v41.decoder_replay import (
            scheduled_decoder_rows,
        )

        return scheduled_decoder_rows(batch, folds=folds)

    def _uses_decoder_replay(self, forward_batch: ForwardBatch) -> bool:
        # MIXED (prefill + decode rows) is not served (validated at startup).
        return (
            self.decoder_replay
            and forward_batch.forward_mode == ForwardMode.EXTEND
            and not getattr(forward_batch, "encoder_swa_replay", False)
        )

    # ------------------------------------------------------------------ #
    # Decoder SWA bounded replay (core late-layer tail interface)
    # ------------------------------------------------------------------ #

    # Set by decoder_replay.scatter_dspark_capture for the causal-LM head of the
    # same forward: the rows are back in the full layout.
    head_full_layout: Optional[ForwardBatch] = None

    # The forward whose decoder rows the model is on (DecoderTail.rows re-gathers
    # the DP-wide ids of those rows with it).
    tail_forward_batch: Optional[ForwardBatch] = None
    # The decoder rows' metadata while the model is on them.
    _decoder_tail: Optional[DSV41TPUForwardMetadata] = None

    @property
    def skips_decoder(self) -> bool:
        """No DP rank keeps a decoder row in this forward (bucket 0)."""
        metadata = self.forward_metadata
        return metadata is not None and metadata.tail_rows == 0

    @property
    def tail_forward_metadata(self) -> Optional[TailView]:
        """Core's model reads ``.late_layer_tail`` during the forward. After it,
        TpModelWorker's encoder-replay fold drop reads ``None`` as no compact
        tail: every row is back in the full layout before the head."""
        if not self.in_model_forward:
            return None
        if self.head_full_layout is not None:
            from sglang_tpu.models.deepseek_v41.decoder_replay import FullLayoutTail

            return TailView(late_layer_tail=FullLayoutTail(self.head_full_layout))
        metadata = self.forward_metadata
        if metadata is None or metadata.tail is None:
            return TailView(late_layer_tail=None)
        return TailView(
            late_layer_tail=DecoderTail(
                gather=metadata.tail_gather,
                positions=metadata.tail.positions,
                num_tokens=metadata.positions.shape[0],
                backend=self,
                inverse=metadata.tail_inverse,
            )
        )

    def enter_late_layer_tail(self, forward_batch: ForwardBatch) -> tuple:
        """Switch the decoder layers onto the decoder rows' metadata, carrying the
        candidate source's selection and candidates to those rows."""
        from sglang.srt.layers import dp_attention
        from sglang.srt.layers.dp_attention import (
            get_dp_global_num_tokens,
            get_global_dp_buffer_len,
            get_local_dp_buffer_len,
            is_dp_attention_enabled,
            set_dp_buffer_len,
        )

        full = self.forward_metadata
        tail = full.tail
        saved = (full, None)
        self.tail_forward_batch = forward_batch
        gather = full.tail_gather.to(torch.int64)
        keep = tail.valid[:, None]
        for ratio, selected in full.selected.items():
            if ratio in self.decoder_low_ratios:
                tail.selected[ratio] = torch.where(
                    keep, selected.index_select(0, gather), -1
                )
        if full.candidates is not None:
            tail.candidates = torch.where(
                keep, full.candidates.index_select(0, gather), -1
            )
        self.forward_metadata = tail
        self._decoder_tail = tail
        if is_dp_attention_enabled():
            rows = int(gather.shape[0])
            from sglang.srt.runtime_context import get_parallel

            dp_size = get_parallel().attn_dp_size
            saved = (
                full,
                (
                    # The DP gather sizes as published for the whole forward,
                    # restored exactly on exit (padded counts, padding mode and
                    # the device counts included).
                    (
                        get_global_dp_buffer_len(),
                        get_local_dp_buffer_len(),
                        dp_attention.is_dp_max_padding(),
                        get_dp_global_num_tokens(),
                        dp_attention._DpGatheredBufferWrapper.get_dp_global_num_tokens_gpu(),
                    ),
                    forward_batch.global_num_tokens_cpu,
                    forward_batch.global_num_tokens_gpu,
                    forward_batch.global_dp_buffer_len,
                ),
            )
            counts = [rows] * dp_size
            # Fused EP masks MoE padding rows by num_token_non_padded (this
            # rank's real rows at the head of its hidden states). The tail
            # keeps its real rows first (plan_tail_rows), so its count is the
            # number of valid tail rows; restored on exit.
            saved_non_padded = getattr(forward_batch, "num_token_non_padded", None)
            if saved_non_padded is not None and _fused_ep_enabled():
                forward_batch.num_token_non_padded = (
                    tail.valid.sum()
                    .to(saved_non_padded.dtype)
                    .reshape(saved_non_padded.shape)
                )
            saved = (saved[0], saved[1] + (saved_non_padded,))
            forward_batch.global_num_tokens_cpu = counts
            forward_batch.global_num_tokens_gpu = torch.full(
                (dp_size,), rows, dtype=torch.int64, device=gather.device
            )
            forward_batch.global_dp_buffer_len = rows * dp_size
            forward_batch.dp_local_start_pos = None
            forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                rows * dp_size,
                rows,
                True,
                counts,
                forward_batch.global_num_tokens_gpu,
            )
        return saved

    def exit_late_layer_tail(self, saved: tuple, forward_batch: ForwardBatch) -> None:
        from sglang.srt.layers.dp_attention import set_dp_buffer_len

        full, dp = saved
        self.forward_metadata = full
        self._decoder_tail = None
        self.tail_forward_batch = None
        if dp is None:
            return
        published, counts, counts_gpu, buffer_len, non_padded = dp
        if non_padded is not None:
            forward_batch.num_token_non_padded = non_padded
        forward_batch.global_num_tokens_cpu = counts
        forward_batch.global_num_tokens_gpu = counts_gpu
        forward_batch.global_dp_buffer_len = buffer_len
        forward_batch.dp_local_start_pos = None
        forward_batch.dp_local_num_tokens = None
        set_dp_buffer_len(*published)

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        self.init_forward_metadata(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        pass

    def init_forward_metadata_replay_cuda_graph(
        self, *args, forward_batch: Optional[ForwardBatch] = None, **kwargs
    ):
        if forward_batch is not None:
            self.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    def store_cache(self, layer_id: int, swa_k=None, forward_batch=None, **kwargs):
        # forward writes the SWA pages right before CSA2 reads them.
        pass

    # ------------------------------------------------------------------ #
    # MQALayer hooks
    # ------------------------------------------------------------------ #

    def prepare_mqa_qkv(
        self,
        layer,
        x_linear: torch.Tensor,
        q_lora: torch.Tensor,
        qkv_a: Optional[torch.Tensor],
        positions: torch.Tensor,
        q_out: Optional[torch.Tensor] = None,
    ):
        """Returns (normalized q_lora, RoPE'd q [T, H, 512], normed RoPE'd kv [T, 512])."""
        assert not layer.q_head_norm, "DeepSeek-V4.1 has no per-head q norm"
        # A module reference, never a tensor: safe to keep across traces.
        self.rope_owners.setdefault(layer.layer_id, layer)
        q_lora = layer.q_norm(q_lora)
        q, _ = layer.wq_b(q_lora)
        q = apply_rope(
            q.view(-1, layer.n_local_heads, layer.head_dim), positions, layer.freqs_cis
        )
        if qkv_a is not None:
            kv = qkv_a[..., layer.q_lora_rank :]
        else:
            kv, _ = layer.wkv(x_linear)
        kv = apply_rope(layer.kv_norm(kv), positions, layer.freqs_cis)
        if q_out is not None:
            q_out.copy_(q)
            q = q_out
        return q_lora, q, kv

    def apply_mqa_inv_rope(self, layer, o: torch.Tensor, positions: torch.Tensor):
        # CSA2 applies the inverse RoPE to its output.
        return o

    def forward_low_ratio_sources(
        self,
        *,
        layer,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        run_compressor: bool = True,
        run_indexer: bool = True,
    ) -> None:
        """Ratio 1/2 kv_source and index_source work, before the layer's attention."""
        if forward_batch.forward_mode.is_idle() or not x.shape[0]:
            return
        metadata = self.get_or_create_metadata(forward_batch)
        assert x.shape[0] == metadata.positions.shape[0], (
            f"{x.shape[0]} rows, metadata for {metadata.positions.shape[0]}"
        )
        if run_compressor and layer.compressor is not None:
            self._compress_and_publish(layer=layer, x=x, metadata=metadata)
        if run_indexer and layer.indexer is not None:
            self._select(layer=layer, x=x, q_lora=q_lora, metadata=metadata)

    def _compress_and_publish(self, *, layer, x, metadata) -> None:
        """Core _low_ratio_compress_torch + _low_ratio_write_group, publishing
        through the CSA2 and indexer ops."""
        pool = self.token_to_kv_pool
        ratio = layer.compress_ratio
        compressor = layer.compressor
        records = metadata.low_ratio[ratio].out_loc
        if ratio == 1:
            # Core's ratio-1 projection is the BF16 nn.Linear; accumulate in FP32
            # as the vllm-torchtpu compressor does (torch.mm on FP32 operands).
            kv = torch.mm(x.float(), compressor.wkv.weight.float().t())
            pooled, group_pos = kv, metadata.positions
        else:
            kv, score = compressor.project(x)  # FP32 (linear_bf16_fp32)
            pooled, group_pos = self._pool_pairs(
                layer=layer, kv=kv, score=score, metadata=metadata
            )
        # Core finish() rounds to BF16 before RMSNorm; keep FP32 through the norm
        # and round once afterwards, like vllm-torchtpu save_compress_norm.
        latent = compressor.norm(pooled.float()).to(torch.bfloat16)

        # Index keys come from the pre-RoPE latent.
        indexer = layer.indexer
        if indexer is not None and indexer.owns_k:
            keys = apply_rope(
                indexer.k_norm(indexer.wk(latent)), group_pos, layer.freqs_cis
            )
            index_k = pool.get_index_k(layer.layer_id)
            index_k.copy_(
                self.indexer_ops.write_index_k(
                    index_k, records, keys, pool.index_k_layout(layer.layer_id)
                )
            )
            # Candidate K gathered before this write is stale.
            metadata.prepared_candidates.pop(pool.index_source(layer.layer_id), None)

        latent = apply_rope(latent, group_pos, layer.freqs_cis)
        words, scales = pool.get_main_kv(layer.layer_id)
        new_words, new_scales = self.csa2_ops.publish_main(
            words, scales, records, latent
        )
        words.copy_(new_words)
        scales.copy_(new_scales)

    def _pool_pairs(self, *, layer, kv, score, metadata):
        """Each odd token pools with its even partner, read from this batch or from
        the request's pair-state ring; the group position is the even token."""
        pos = metadata.positions
        partner_kv, partner_score = self._pair_partners(
            layer_id=layer.layer_id,
            kv=kv,
            score=score,
            req=metadata.token_req_pool,
            pos=pos,
            # Folded replay rows keep the cached pair state of their hit.
            pad=~metadata.compress_valid,
        )
        pooled = layer.compressor.pool_pairs(
            torch.stack([partner_kv, kv], dim=1),
            torch.stack([partner_score, score], dim=1),
        )
        return pooled, torch.where(pos % 2 == 1, pos - 1, pos)

    def _pair_partners(self, *, layer_id, kv, score, req, pos, pad):
        """Core ``_low_ratio_pair_partners``, writing the ring through the
        donated ``write_rows`` op after every read of the old ring.

        Core's in-place ``kv_score_buffer[state_loc] = value`` is not written
        back in the compiled TPU graph (84c46f2eaf)."""
        state = self.token_to_kv_pool.get_attention_compress_states(layer_id)
        ring = state.ring_size
        num_tokens = pos.shape[0]
        if not num_tokens:
            return kv, score

        read_pos = (pos - 1).masked_fill(pad, -1)
        carried = state.get_state_by_state_loc(
            state.translate_from_req_position_to_state_loc(req, read_pos)
        )
        in_batch = torch.zeros_like(pad)
        in_batch[1:] = (
            (req[1:] == req[:-1]) & (pos[1:] == pos[:-1] + 1) & ~pad[1:] & ~pad[:-1]
        )
        partner_kv = torch.where(in_batch[:, None], torch.roll(kv, 1, 0), carried.kv)
        partner_score = torch.where(
            in_batch[:, None], torch.roll(score, 1, 0), carried.score
        )

        # Keep only a request's last ring_size rows so each write hits a distinct
        # live slot; dropped and padding rows skip the write.
        keep = ~pad
        if num_tokens > ring:
            keep[:-ring] &= (req[:-ring] != req[ring:]) | pad[ring:]
        write_pos = pos.masked_fill(~keep, -1)
        slots = state.translate_from_req_position_to_state_loc(req, write_pos)
        buffer = state.kv_score_buffer.kv_score
        values = torch.cat([kv, score], dim=-1)
        buffer.copy_(self.csa2_ops.write_rows(buffer, slots, values, carried.kv_score))
        return partner_kv, partner_score

    def _select(self, *, layer, x, q_lora, metadata) -> None:
        """Indexer query (core wq_b, head weights; RoPE on the last 64 dims), then
        the FP8 top-k over the layer's source index keys."""
        pool = self.token_to_kv_pool
        indexer = layer.indexer
        ratio = layer.compress_ratio
        low = metadata.low_ratio[ratio]
        num_tokens = x.shape[0]
        q, _ = indexer.wq_b(q_lora)
        q = q.view(num_tokens, indexer.n_heads, indexer.index_head_dim)
        q = apply_rope(q, metadata.positions, layer.freqs_cis)
        weights = indexer.head_weights_raw(x).float() * indexer.head_weight_scale
        has_candidates = indexer.candidate_topk_blocks > 0
        publish = has_candidates and indexer.is_candidate_source
        consume = has_candidates and indexer.uses_candidates
        if consume:
            if metadata.candidates is None:
                raise RuntimeError(
                    f"layer {layer.layer_id} consumes candidates before their source"
                )
            candidates = metadata.candidates
        else:
            # A source publishes into this buffer's width; other layers ignore it.
            width = indexer.candidate_topk_blocks if publish else 1
            candidates = torch.full(
                (num_tokens, width),
                -1,
                dtype=torch.int32,
                device=x.device,
            )
        block_size = max(1, indexer.candidate_block_size)
        width = low.index_pages.shape[1] * pool.index_k_layout(layer.layer_id)[3]
        if consume and self.indexer_ops.uses_prepared_candidates(
            count=candidates.shape[1], block_size=block_size, width=width
        ):
            selected = self._select_prepared(
                layer=layer,
                q=q,
                weights=weights,
                candidates=candidates,
                metadata=metadata,
                block_size=block_size,
                width=width,
            )
            blocks = candidates
        else:
            selected, blocks = self.indexer_ops.select(
                q,
                weights,
                pool.get_index_k(layer.layer_id),
                low.index_pages,
                metadata.token_rows,
                low.lengths,
                candidates,
                pool.index_k_layout(layer.layer_id),
                topk=self.index_topk,
                block_size=block_size,
                publish=publish,
                consume=consume,
            )
        if publish:
            metadata.candidates = blocks
        if selected.shape[1] < self.selection_width:
            selected = F.pad(
                selected, (0, self.selection_width - selected.shape[1]), value=-1
            )
        metadata.selected[ratio] = selected

    def _select_prepared(
        self, *, layer, q, weights, candidates, metadata, block_size, width
    ) -> torch.Tensor:
        """Candidate-consumer top-k; the candidate K gather runs once per source
        and batch and lives until its last consumer, as the epic's stage cache."""
        pool = self.token_to_kv_pool
        low = metadata.low_ratio[layer.compress_ratio]
        key = pool.index_source(layer.layer_id)
        prepared = metadata.prepared_candidates.get(key)
        if prepared is None:
            prepared = self.indexer_ops.prepare_candidates(
                pool.get_index_k(layer.layer_id),
                low.index_pages,
                candidates,
                metadata.token_rows,
                pool.index_k_layout(layer.layer_id),
                block_size=block_size,
            )
            metadata.prepared_candidates[key] = prepared
        selected = self.indexer_ops.select_prepared(
            q,
            weights,
            low.lengths,
            prepared,
            width=width,
            topk=self.index_topk,
            block_size=block_size,
        )
        # Drop the gathered K after its last reader, as the epic's per-stage
        # scope does; kept on the metadata it outlives the forward (8.86 GB at
        # 2048 tokens) and the next program fails to load.
        if layer.layer_id >= self.last_candidate_consumer.get(key, -1):
            metadata.prepared_candidates.pop(key, None)
        return selected

    def _last_candidate_consumers(self, config) -> Dict[tuple, int]:
        """(ratio, index source) -> last index layer reading its candidates."""
        last: Dict[tuple, int] = {}
        source = config.candidate_source_layer_id
        for layer_id in sorted(config.index_source_layer_ids):
            if (
                0 <= source < layer_id
                and config.compress_ratios[layer_id] in LOW_RATIOS
            ):
                last[self.token_to_kv_pool.index_source(layer_id)] = layer_id
        return last

    # ------------------------------------------------------------------ #
    # Attention
    # ------------------------------------------------------------------ #

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_idle():
            return q.new_zeros(q.shape[0], q.shape[1], layer.v_head_dim)
        return self._attention(
            q=q, k=k, layer=layer, forward_batch=forward_batch, **kwargs
        )

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        return self._attention(q=q, k=k, layer=layer, forward_batch=forward_batch, **kw)

    def forward_decode(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kw):
        return self._attention(q=q, k=k, layer=layer, forward_batch=forward_batch, **kw)

    def _swa_metadata(self, metadata: DSV41TPUForwardMetadata) -> None:
        if metadata.swa_meta is not None:
            return
        meta, bounds, records = self.csa2_ops.swa_metadata(
            metadata.swa_pages,
            metadata.query_start_loc,
            metadata.positions,
            metadata.swa_minimum,
            non_causal=self.is_dspark_draft,
        )
        # Bounds are passed to attention only for the non-causal draft block.
        metadata.swa_meta = meta
        metadata.swa_bounds = bounds if self.is_dspark_draft else None
        metadata.swa_write_records = records.masked_fill(~metadata.valid, -1)

    def _attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        compress_ratio: int = 0,
        attn_sink: Optional[torch.Tensor] = None,
        **_,
    ) -> torch.Tensor:
        """SWA write of this step's kv (normed + RoPE'd), then CSA2 over SWA and the
        source layer's selected main records; the output is inverse-rotated.

        Core passes save_kv_cache=False with k = kv because prepare_mqa_qkv
        returned kv; this backend always owns the SWA write."""
        if not q.shape[0]:
            return q.new_empty(q.shape[0], q.shape[1], layer.v_head_dim)
        metadata = self.get_or_create_metadata(forward_batch)
        pool = self.token_to_kv_pool
        layer_id = layer.layer_id
        self._swa_metadata(metadata)

        swa = pool.get_swa_pages(layer_id)
        kv = k.reshape(k.shape[0], -1).contiguous()
        records = metadata.swa_write_records
        plan = metadata.swa_window_plan
        if plan is not None:
            # One publish also copies ring history into the workspace pages the
            # chunk attends and the chunk's last window back into the ring.
            history = swa.reshape(-1, kv.shape[1])[plan.history_src]
            kv = torch.cat((kv, history, kv[plan.commit_token]))
            records = torch.cat((records, plan.history_dst, plan.commit_dst))
        swa.copy_(self.csa2_ops.publish_swa(swa, records, kv))

        if attn_sink is None:
            sink = torch.zeros(q.shape[1], dtype=torch.float32, device=q.device)
        else:
            sink = attn_sink[: q.shape[1]].float().contiguous()
        cos_sin = rope_table(self.rope_owners[layer_id].freqs_cis)
        if compress_ratio:
            out = self._attend_main(
                q=q.contiguous(),
                swa=swa,
                sink=sink,
                cos_sin=cos_sin,
                layer=layer,
                compress_ratio=compress_ratio,
                metadata=metadata,
            )
        else:
            out = self.csa2_ops.attention(
                q.contiguous(),
                swa,
                metadata.swa_meta,
                metadata.swa_bounds,
                metadata.query_start_loc,
                metadata.positions,
                sink,
                cos_sin,
                0,
                float(layer.scaling),
            )
        return out.masked_fill(~metadata.valid[:, None, None], 0)

    def _attend_main(
        self, *, q, swa, sink, cos_sin, layer, compress_ratio, metadata
    ) -> torch.Tensor:
        """Main-KV attention; a gather group's first layer gathers the selected
        records and keeps them for the group's later layers."""
        layer_id = layer.layer_id
        first, last = self.gather_groups.get(layer_id, (layer_id, layer_id))
        # Decoder-replay tails take the extend tiling (ops.csa2.attention), which
        # keeps no gathered KV, so each tail layer gathers its own records.
        extend_tiling = metadata is self._decoder_tail
        if extend_tiling:
            first = last = layer_id
        common = dict(
            swa_pages=swa,
            swa_meta=metadata.swa_meta,
            query_start_loc=metadata.query_start_loc,
            positions=metadata.positions,
            sink=sink,
            cos_sin=cos_sin,
            compress_ratio=compress_ratio,
            scale=float(layer.scaling),
        )
        if layer_id != first:
            gathered = metadata.gathered_main.get(first)
            if gathered is None:
                raise RuntimeError(
                    f"layer {layer_id} reuses the main KV of layer {first}, "
                    "which did not gather it"
                )
            if layer_id >= last:
                # Released after its last reader; kept, it outlives the forward.
                metadata.gathered_main.pop(first)
            return self.csa2_ops.attention_gathered(q, gathered, **common)

        selected = metadata.selected.get(compress_ratio)
        if selected is None:
            raise RuntimeError(
                f"layer {layer_id} (ratio {compress_ratio}) attends before its "
                "index source selected"
            )
        pool = self.token_to_kv_pool
        words, scales = pool.get_main_kv(layer_id)
        main = (
            words,
            scales,
            selected,
            metadata.full_pages,
            pool.main_page_records(layer_id),
        )
        keep = last > layer_id
        result = self.csa2_ops.attention(
            q,
            common["swa_pages"],
            common["swa_meta"],
            None,
            common["query_start_loc"],
            common["positions"],
            sink,
            cos_sin,
            compress_ratio,
            common["scale"],
            main=main,
            keep_gathered=keep,
            extend_tiling=extend_tiling,
        )
        if not keep:
            return result
        out, gathered = result
        metadata.gathered_main[first] = gathered
        return out

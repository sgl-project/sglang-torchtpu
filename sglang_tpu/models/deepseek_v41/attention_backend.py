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
with the ring write as a functional update copied back (as the Engram history fix).

Padding tokens (``out_cache_loc == 0`` or past the batch's real tokens) never write
a cache (record -1) and produce zeros.

Kernels come from ``ops/csa2.py`` and ``ops/indexer.py``; tests replace them
through ``csa2_ops`` / ``indexer_ops``.

IMPORTANT: Do NOT add ``from __future__ import annotations``; see
``sglang_tpu/dsv4/dsv4_attention_backend.py``.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, NamedTuple, Optional, Sequence

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

FULL_PAGE_TOKENS = 256
SWA_PAGE_TOKENS = 64
INDEX_PAGE_RECORDS = 128
# CSA2's fast main-record path reads exactly 512 selected ids per query.
SELECTION_WIDTH = 512
LOW_RATIOS = (1, 2)
# compiled_forward_shim sets this on bucket-padded batches.
_REAL_B_KEY = "_tpu_real_b"


# =============================================================================
# RoPE
# =============================================================================


def rope_table(freqs: torch.Tensor) -> torch.Tensor:
    """fp32 ``[max_pos, rope_dim]`` ``[cos | sin]`` table of a layer's ``freqs_cis``
    (complex, or already converted by ``_prewarm_model_caches``)."""
    if freqs.is_complex():
        from sglang_tpu.dsv4.dsv4_memory_pool import get_or_build_cos_sin_cache

        return get_or_build_cos_sin_cache(freqs)
    return freqs


def rope_reference(
    x: torch.Tensor,
    positions: torch.Tensor,
    table: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Core ``rope_tail`` numerics on a ``[cos | sin]`` table: rotate the last
    ``rope_dim`` features as interleaved (even, odd) complex pairs, in fp32."""
    rope_dim = table.shape[-1]
    half = rope_dim // 2
    rows = table[positions].float()
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
    # [T] int32 scheduler full_loc; TPU lowering rejects 64-bit integer vectors.
    out_cache_loc: torch.Tensor
    # [B, C] int32 full page ids, C = ceil(context_len / 256); also main KV pages.
    full_pages: torch.Tensor
    # [B, 4C] int32 SWA 64-token page ids; 0 is the null page (evicted or unmapped).
    swa_pages: torch.Tensor
    # [B] int32 earliest visible SWA position (no replay: 0).
    swa_minimum: torch.Tensor
    low_ratio: Dict[int, LowRatioMetadata]

    # Filled lazily during the forward.
    swa_meta: Optional[torch.Tensor] = None
    swa_bounds: Optional[torch.Tensor] = None
    swa_write_records: Optional[torch.Tensor] = None
    # Latest index_source selection per ratio, read by every later layer of that
    # ratio (core sparse_page_indices(ratio)); [T, 512] int32, valid prefix.
    selected: Dict[int, torch.Tensor] = field(default_factory=dict)
    # Candidate blocks published by config.candidate_source_layer_id.
    candidates: Optional[torch.Tensor] = None


def _as_cpu_int64(values, length: int) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        out = values.detach().to(device="cpu", dtype=torch.int64)
    else:
        out = torch.tensor(list(values), dtype=torch.int64)
    out = out[:length]
    if out.shape[0] < length:
        out = F.pad(out, (0, length - out.shape[0]))
    return out


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
    if mode.is_extend() and not mode.is_target_verify():
        lens = forward_batch.extend_seq_lens_cpu
        if lens is None:
            lens = forward_batch.extend_seq_lens
        assert lens is not None, "an extend batch carries extend_seq_lens"
        counts[:real_bs] = _as_cpu_int64(lens, bs)[:real_bs]
        return counts
    raise NotImplementedError(
        f"DeepSeek-V4.1 TPU attention serves extend and decode, not {mode.name}"
    )


def _full_page_table(
    req_to_token: torch.Tensor, req_pool_indices: torch.Tensor, columns: int
) -> torch.Tensor:
    """[B, columns] int64: page p of a request is req_to_token[req, 256 p] // 256."""
    starts = req_to_token[req_pool_indices.to(req_to_token.device), ::FULL_PAGE_TOKENS]
    starts = starts[:, :columns].to("cpu", torch.int64)
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
        swa_starts = mapping[lookup.to(mapping.device)].to("cpu", torch.int64)
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
    full_pages: torch.Tensor,
) -> LowRatioMetadata:
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        _low_ratio_compression_metadata,
    )
    from sglang.srt.layers.attention.dsv4.metadata import expand_index_page_table

    seq_lens_casual = positions + 1
    out_loc, _ = _low_ratio_compression_metadata(ratio, seq_lens_casual, out_cache_loc)
    return LowRatioMetadata(
        out_loc=out_loc.masked_fill(~valid, -1).to(torch.int32),
        lengths=torch.where(valid, seq_lens_casual // ratio, 0).to(torch.int32),
        index_pages=expand_index_page_table(
            full_pages.to(torch.int32),
            full_page_size=FULL_PAGE_TOKENS,
            compress_ratio=ratio,
            index_page_size=INDEX_PAGE_RECORDS,
        ).to(torch.int32),
    )


def build_forward_metadata(
    forward_batch: ForwardBatch,
    *,
    req_to_token: torch.Tensor,
    full_to_swa_index_mapping: Optional[torch.Tensor],
    max_context_len: int,
    low_ratios: Sequence[int],
    device: torch.device,
) -> DSV41TPUForwardMetadata:
    """Host-side metadata from the sglang batch, moved to ``device`` once."""
    bs = int(forward_batch.batch_size)
    positions = forward_batch.positions.detach().to("cpu", torch.int64)
    num_tokens = positions.shape[0]
    out_cache_loc = _as_cpu_int64(forward_batch.out_cache_loc, num_tokens)

    counts = _request_token_counts(forward_batch, bs)
    query_start_loc = F.pad(torch.cumsum(counts, 0), (1, 0))
    real_tokens = int(query_start_loc[-1])
    assert real_tokens <= num_tokens, f"{real_tokens=} > {num_tokens=}"
    token_rows = torch.full((num_tokens,), -1, dtype=torch.int64)
    token_rows[:real_tokens] = torch.repeat_interleave(torch.arange(bs), counts)
    valid = (token_rows >= 0) & (out_cache_loc > 0)
    token_rows = token_rows.masked_fill(~valid, -1)
    req_pool_indices = _as_cpu_int64(forward_batch.req_pool_indices, bs)
    token_req_pool = req_pool_indices[token_rows.clamp_min(0)].masked_fill(~valid, 0)

    columns = max(1, -(-int(max_context_len) // FULL_PAGE_TOKENS))
    full_pages = _full_page_table(req_to_token, req_pool_indices, columns)
    swa_pages = _swa_page_table(full_pages, full_to_swa_index_mapping)
    low_ratio = {
        ratio: _low_ratio_metadata(
            ratio=ratio,
            positions=positions,
            out_cache_loc=out_cache_loc,
            valid=valid,
            full_pages=full_pages,
        )
        for ratio in low_ratios
    }

    def put(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, non_blocking=True)

    return DSV41TPUForwardMetadata(
        positions=put(positions.to(torch.int32)),
        query_start_loc=put(query_start_loc.to(torch.int32)),
        token_rows=put(token_rows.to(torch.int32)),
        token_req_pool=put(token_req_pool.to(torch.int32)),
        valid=put(valid),
        out_cache_loc=put(out_cache_loc.to(torch.int32)),
        full_pages=put(full_pages.to(torch.int32)),
        swa_pages=put(swa_pages.to(torch.int32)),
        swa_minimum=put(torch.zeros(bs, dtype=torch.int32)),
        low_ratio={
            ratio: LowRatioMetadata(*(put(t) for t in meta))
            for ratio, meta in low_ratio.items()
        },
    )


# =============================================================================
# Backend
# =============================================================================


def _check_supported(config) -> None:
    from sglang.srt.runtime_context import get_exec, get_parallel, get_spec

    parallel = get_parallel()
    for name in ("pp_size", "attn_cp_size", "attn_dcp_size", "attn_tp_size"):
        if getattr(parallel, name) != 1:
            raise NotImplementedError(
                f"DeepSeek-V4.1 TPU attention needs {name} == 1 "
                "(attention TP1 per DP rank)"
            )
    if get_spec().speculative_algorithm is not None:
        raise NotImplementedError("DeepSeek-V4.1 TPU attention does not run MTP yet")
    features = get_exec().features
    if (
        features.enable_encoder_swa_bounded_replay
        or features.enable_decoder_swa_bounded_replay
    ):
        raise NotImplementedError("DeepSeek-V4.1 TPU attention has no SWA replay")
    if set(config.compress_ratios) - {0, *LOW_RATIOS}:
        raise ValueError(
            "DeepSeek-V4.1 TPU attention serves ratios 0/1/2, got "
            f"{sorted(set(config.compress_ratios))}"
        )
    if int(config.sliding_window) != 128 or int(config.head_dim) != 512:
        raise ValueError("CSA2 requires sliding_window 128 and head_dim 512")
    _check_index_sources(config)


def _check_index_sources(config) -> None:
    """Every ratio 1/2 layer attends with the selection of the latest preceding
    index_source of its ratio (core reads sparse_page_indices(ratio))."""
    ratios = list(config.compress_ratios)
    for layer_id, ratio in enumerate(ratios):
        if ratio in LOW_RATIOS and not any(
            s <= layer_id and ratios[s] == ratio for s in config.index_source_layer_ids
        ):
            raise ValueError(f"layer {layer_id} has no ratio-{ratio} index source")


class DSV41TPUAttnBackend(AttentionBackend):
    """DeepSeek-V4.1 CSA2 attention on sglang request/page metadata."""

    def __init__(self, model_runner: "ModelRunner"):
        super().__init__()
        model_config = model_runner.model_config
        config = model_config.hf_text_config
        _check_supported(config)
        if (
            getattr(model_runner.server_args, "enable_torch_compile", False)
            and getattr(model_runner.server_args, "tpu_compile_tracer", None)
            == "direct"
        ):
            # The direct tracer reads the KV pool through get_state(), which
            # DSV41TPUTokenToKVPool does not implement.
            raise NotImplementedError(
                "DeepSeek-V4.1 TPU attention does not support "
                "--tpu-compile-tracer direct yet; use the default dynamo tracer"
            )
        self.model_runner = model_runner
        self.device = model_runner.device
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.max_context_len = int(model_config.context_len)
        self.page_size = FULL_PAGE_TOKENS
        self.index_topk = int(config.index_topk)
        self.low_ratios = tuple(r for r in LOW_RATIOS if r in config.compress_ratios)
        self.forward_metadata: Optional[DSV41TPUForwardMetadata] = None
        # layer_id -> fp32 [cos | sin] table; forward only sees the RadixAttention.
        self.rope_tables: Dict[int, torch.Tensor] = {}
        self.csa2_ops, self.indexer_ops = self._register_ops()
        self._prewarm_model_caches()
        self._drop_text_only_image_mask(model_runner)

    @staticmethod
    def _drop_text_only_image_mask(model_runner) -> None:
        """Without multimodal serving no prompt holds image tokens, so the forward's
        image-token torch.where after each Engram layer is the identity.

        The vision tower, MoE correction bias and hasher read vision_n_layers at
        construction, already done; only that where reads it in the forward, and
        its fused select has no valid TPU window ("Cannot find a valid window").
        """
        model = model_runner.model
        if model is None or model_runner.model_config.is_multimodal:
            return
        config = model.model.config
        if config.model_type == "deepseek_v41" and config.vision_n_layers > 0:
            config.vision_n_layers = 0

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
        """Convert complex freqs_cis to [cos | sin] tables outside any trace and
        record each attention layer's table by layer id."""
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
                self.rope_tables[attn.layer_id] = table

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
            max_context_len=self.max_context_len,
            low_ratios=self.low_ratios,
            device=self.device,
        )

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
        if layer.layer_id not in self.rope_tables:
            self.rope_tables[layer.layer_id] = rope_table(layer.freqs_cis)
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
            pad=~metadata.valid,
        )
        pooled = layer.compressor.pool_pairs(
            torch.stack([partner_kv, kv], dim=1),
            torch.stack([partner_score, score], dim=1),
        )
        return pooled, torch.where(pos % 2 == 1, pos - 1, pos)

    def _pair_partners(self, *, layer_id, kv, score, req, pos, pad):
        """Core ``_low_ratio_pair_partners`` with the ring write as a functional
        update copied back into the pool buffer.

        Core writes ``kv_score_buffer[state_loc] = value`` in place; inside the
        compiled TPU graph that indexed assignment is the pattern the Engram
        history fix found not written back (84c46f2eaf), which would leave an
        odd token pooling with a stale partner. Reads all precede the write."""
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
        # live slot; dropped and padding rows land on the scratch row -1.
        keep = ~pad
        if num_tokens > ring:
            keep[:-ring] &= (req[:-ring] != req[ring:]) | pad[ring:]
        write_pos = pos.masked_fill(~keep, -1)
        slots = state.translate_from_req_position_to_state_loc(req, write_pos)
        buffer = state.kv_score_buffer.kv_score
        slots = torch.where(slots < 0, buffer.shape[0] - 1, slots).to(torch.int64)
        values = torch.cat([kv, score], dim=-1).to(buffer.dtype)
        updated = buffer.index_put((slots,), values)
        # Row -1 reads as an empty partner: kv 0, score -inf (core clear()).
        half = buffer.shape[-1] // 2
        empty = torch.cat(
            [
                buffer.new_zeros(1, half),
                buffer.new_full((1, half), float("-inf")),
            ],
            dim=-1,
        )
        updated = torch.cat([updated[:-1], empty], dim=0)
        buffer.copy_(updated)
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
            block_size=max(1, indexer.candidate_block_size),
            publish=publish,
            consume=consume,
        )
        if publish:
            metadata.candidates = blocks
        if selected.shape[1] < SELECTION_WIDTH:
            selected = F.pad(
                selected, (0, SELECTION_WIDTH - selected.shape[1]), value=-1
            )
        metadata.selected[ratio] = selected

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
        )
        metadata.swa_meta, metadata.swa_bounds = meta, bounds
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
        swa.copy_(self.csa2_ops.publish_swa(swa, metadata.swa_write_records, kv))

        if attn_sink is None:
            sink = torch.zeros(q.shape[1], dtype=torch.float32, device=q.device)
        else:
            sink = attn_sink[: q.shape[1]].float().contiguous()
        main = None
        if compress_ratio:
            selected = metadata.selected.get(compress_ratio)
            if selected is None:
                raise RuntimeError(
                    f"layer {layer_id} (ratio {compress_ratio}) attends before its "
                    "index source selected"
                )
            words, scales = pool.get_main_kv(layer_id)
            main = (
                words,
                scales,
                selected,
                metadata.full_pages,
                pool.main_page_records(layer_id),
            )
        out = self.csa2_ops.attention(
            q.contiguous(),
            swa,
            metadata.swa_meta,
            metadata.swa_bounds,
            metadata.query_start_loc,
            metadata.positions,
            sink,
            self.rope_tables[layer_id],
            compress_ratio,
            float(layer.scaling),
            main=main,
        )
        return out.masked_fill(~metadata.valid[:, None, None], 0)

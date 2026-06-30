# SPDX-License-Identifier: Apache-2.0
"""TPU Pallas attention backend for SGLang.

Drop-in replacement for the `torch_native` attention backend on torch_tpu.
It keeps `torch_native`'s exact data flow (per-request gather from the flat
token-indexed KV pool, with control tensors host-staged via `.tolist()` so the
torch_tpu device queue does not wedge under TP>1 — same reason as the
`_patch_compute_position_torch` / `_patch_torch_native_attn_backend` patches),
and only swaps the hot attention math: the slow `scaled_dot_product_attention`
MATH-backend fallback is replaced by the vendored Pallas `flash_attention`
kernel.

Each TP rank runs attention purely on its local heads (QKV projection is
column-parallel; the all-reduce happens after o_proj), so no JAX mesh /
shard_map is needed here — unlike the torchtpu-vllm path this is ported from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import torch
from jax.experimental.pallas.ops.tpu.paged_attention.paged_attention_kernel import (
    paged_attention,
)

from sglang.srt.hardware_backend.tpu.kernels.flash_attention import (
    SegmentIds,
    flash_attention,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


# Cache of torch custom ops keyed by (causal, sm_scale). flash_attention bakes
# causal/sm_scale as static jax.jit args, so one op def per distinct pair.
_FLASH_OPS: dict[tuple[bool, float], object] = {}


def _get_flash_op(causal: bool, sm_scale: float):
    key = (bool(causal), float(sm_scale))
    op = _FLASH_OPS.get(key)
    if op is not None:
        return op

    from torch_tpu._internal import pallas

    def _jax_flash(q, k, v):
        # q, k, v: [batch, num_heads, seq_len, head_dim].
        return flash_attention(
            q,
            k,
            v,
            causal=causal,
            sm_scale=sm_scale,
            vmem_limit_bytes=None,
        )

    # jax_op's _verify_signature requires every param + return annotated with
    # real jax.Array/POD objects. This module uses `from __future__ import
    # annotations`, which would stringize inline annotations ('jax.Array') and
    # fail that check, so set the real objects explicitly.
    _jax_flash.__annotations__ = {
        "q": jax.Array,
        "k": jax.Array,
        "v": jax.Array,
        "return": jax.Array,
    }

    op_name = f"sglang_tpu::flash_attn_c{int(causal)}_s{key[1]!r}".replace(
        ".", "_"
    ).replace("-", "m")
    op = pallas.jax_op(op_name, _jax_flash)

    def _fake_flash(q, k, v):
        return torch.empty_like(q)

    op.register_fake(_fake_flash)
    _FLASH_OPS[key] = op
    return op


_BLOCK_Q = 128

# Decode: jax pallas paged_attention. page_size fixed; pages_per_seq derived
# from the (bucketed) max kv length. The op bakes pages_per_compute_block as
# a static jax arg, so cache one op per ppb value.
_PAGE_SIZE = 64
_PAGED_OPS: dict[int, object] = {}


def _get_paged_op(pages_per_compute_block: int):
    op = _PAGED_OPS.get(pages_per_compute_block)
    if op is not None:
        return op

    from torch_tpu._internal import pallas

    def _jax_paged(q, k_pages, v_pages, lengths, page_indices):
        return paged_attention(
            q,
            k_pages,
            v_pages,
            lengths,
            page_indices,
            pages_per_compute_block=pages_per_compute_block,
        )

    _jax_paged.__annotations__ = {
        "q": jax.Array,
        "k_pages": jax.Array,
        "v_pages": jax.Array,
        "lengths": jax.Array,
        "page_indices": jax.Array,
        "return": jax.Array,
    }

    op_name = f"sglang_tpu::paged_attn_ppb{pages_per_compute_block}"
    op = pallas.jax_op(op_name, _jax_paged)

    def _fake_paged(q, k_pages, v_pages, lengths, page_indices):
        return torch.empty_like(q)

    op.register_fake(_fake_paged)
    _PAGED_OPS[pages_per_compute_block] = op
    return op


def _bucket_kv_len(max_len: int) -> int:
    """Round the batch's max kv length up to a power-of-two multiple of
    _PAGE_SIZE, to bound the number of distinct compiled shapes."""
    n = _PAGE_SIZE
    while n < max_len:
        n *= 2
    return n


def _attend(op, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Call the Pallas flash op with q's seq dim padded up to block_q.

    The kernel requires q_seq_len >= block_q (128); SGLang's ragged per-request
    seqs are far smaller (prompt ~27, decode 1). kv needs no padding (the
    kernel sets block_k == kv_seq_len in the single-step path, which always
    divides). Attention is independent per q-row, so padding q at the end and
    slicing the real rows back off is exact — causal correctness holds because
    kv is unpadded and the real q rows keep their positions.

    q, k, v: [1, num_heads, seq, head_dim]. Returns [1, num_heads, q_seq, hd].
    """
    q_seq = q.shape[2]
    q_pad = max(_BLOCK_Q, ((q_seq + _BLOCK_Q - 1) // _BLOCK_Q) * _BLOCK_Q)
    if q_pad != q_seq:
        padded = torch.zeros(
            (q.shape[0], q.shape[1], q_pad, q.shape[3]),
            dtype=q.dtype,
            device=q.device,
        )
        padded[:, :, :q_seq, :] = q
        q = padded
    out = op(q, k, v)
    return out[:, :, :q_seq, :]


def _maybe_expand_kv_heads(
    kv: torch.Tensor, q_head_num: int, kv_head_num: int
) -> torch.Tensor:
    """Expand GQA KV heads to match Q heads.

    The Pallas flash kernel requires num_heads(q) == num_heads(k) == num_heads(v)
    (it raises on a head-count mismatch). SDPA's `enable_gqa=True` did this
    implicitly; here we materialize it. `kv` is [num_kv_heads, seq, head_dim];
    repeat_interleave maps q-head g -> kv-head g // group, matching SDPA.
    """
    if q_head_num == kv_head_num:
        return kv
    group = q_head_num // kv_head_num
    return kv.repeat_interleave(group, dim=0)


class TpuPallasAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        pass

    def _run_flash_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # Single explicit host sync of all loop-control tensors (TP-safe).
        seq_lens_l = seq_lens.tolist()
        extend_prefix_lens_l = extend_prefix_lens.tolist()
        extend_seq_lens_l = extend_seq_lens.tolist()
        req_pool_indices_l = req_pool_indices.tolist()

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)
        q_head_num = query.shape[0]

        op = _get_flash_op(bool(causal), float(scaling))

        start_q = 0
        for seq_idx in range(len(seq_lens_l)):
            extend_seq_len_q = extend_seq_lens_l[seq_idx]
            prefill_seq_len_q = extend_prefix_lens_l[seq_idx]

            seq_len_kv = seq_lens_l[seq_idx]
            end_q = start_q + extend_seq_len_q

            per_req_query = query[:, start_q:end_q, :]
            # Zero (not empty): the prefix rows (0..prefill_seq_len_q) get
            # masked off the final output, but uninitialized values can carry
            # NaN/Inf into the flash kernel's softmax intermediates.
            per_req_query_redudant = torch.zeros(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )
            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            req_pool_idx = req_pool_indices_l[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_key = _maybe_expand_kv_heads(
                per_req_key, q_head_num, per_req_key.shape[0]
            )
            per_req_value = _maybe_expand_kv_heads(
                per_req_value, q_head_num, per_req_value.shape[0]
            )

            per_req_out_redudant = (
                _attend(
                    op,
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q = end_q
        return output

    def _run_paged_forward_decode(
        self,
        query: torch.Tensor,  # [batch, Hq, D] (one query token per request)
        output: torch.Tensor,  # [batch, Hq, Dv]
        k_cache: torch.Tensor,  # [num_slots, Hkv, D]
        v_cache: torch.Tensor,  # [num_slots, Hkv, Dv]
        req_to_token: torch.Tensor,  # [max_reqs, max_ctx] int
        req_pool_indices: torch.Tensor,  # [batch] int
        seq_lens: torch.Tensor,  # [batch] int
        scaling: float = None,
    ):
        """Single batched paged_attention call (no per-request Python loop,
        no q>=128 padding). Each request's KV is gathered from the flat
        token-indexed pool and presented to the kernel as its own run of
        fixed-size pages; the kernel does length masking and GQA natively.
        """
        batch = query.shape[0]

        # One explicit host sync (TP-safe, same reason as the other tpu
        # patches): the kv-length bucket must be a Python int.
        seq_lens_l = seq_lens.tolist()
        max_len = max(seq_lens_l) if seq_lens_l else _PAGE_SIZE
        # The real per-request seq length must fit in the allocated per-req
        # slot row; the scheduler/allocator guarantees this by construction.
        # We assert so any future regression surfaces here, not as silent
        # KV truncation passed to the kernel.
        assert max_len <= req_to_token.shape[1], (
            f"seq_len {max_len} exceeds req_to_token capacity "
            f"{req_to_token.shape[1]}"
        )
        Lb = max(_PAGE_SIZE, _bucket_kv_len(max_len))
        pps = Lb // _PAGE_SIZE
        ppb = min(8, pps)

        # Vectorized gather of per-request token slots from the flat pool.
        # Slots past the real length are stale but masked by `lengths`.
        #
        # Invariant: req_to_token.shape[1] >= Lb. `_bucket_kv_len` is monotone
        # and maps any input to a power-of-two multiple of `_PAGE_SIZE`, so
        # this holds whenever `req_to_token.shape[1]` is itself such a value.
        # `_patch_req_to_token_pool_max_context_len` enforces that at pool
        # allocation. Asserting here means any future regression surfaces
        # loudly instead of right-padding with slot 0 — a live KV slot owned
        # by another request, whose data would leak into attention if length
        # masking ever drifted.
        assert req_to_token.shape[1] >= Lb, (
            f"req_to_token.shape[1] ({req_to_token.shape[1]}) must be >= Lb "
            f"({Lb}); is _patch_req_to_token_pool_max_context_len applied?"
        )
        tok = req_to_token[req_pool_indices.long()][:, :Lb]  # [batch, Lb]

        k_g = k_cache[tok.long()]  # [batch, Lb, Hkv, D]
        v_g = v_cache[tok.long()]  # [batch, Lb, Hkv, Dv]
        if k_g.dtype != query.dtype:
            k_g = k_g.to(query.dtype)
            v_g = v_g.to(query.dtype)

        Hkv, D, Dv = k_g.shape[2], k_g.shape[3], v_g.shape[3]

        # [batch, Lb, Hkv, *] -> [Hkv, batch*pps, PAGE_SIZE, *]; global page
        # index g = b*pps + pp, matching page_indices below.
        k_pages = (
            k_g.reshape(batch, pps, _PAGE_SIZE, Hkv, D)
            .permute(3, 0, 1, 2, 4)
            .reshape(Hkv, batch * pps, _PAGE_SIZE, D)
        )
        v_pages = (
            v_g.reshape(batch, pps, _PAGE_SIZE, Hkv, Dv)
            .permute(3, 0, 1, 2, 4)
            .reshape(Hkv, batch * pps, _PAGE_SIZE, Dv)
        )
        page_indices = torch.arange(
            batch * pps, device=query.device, dtype=torch.int32
        ).reshape(batch, pps)
        lengths = seq_lens.to(torch.int32)

        # paged_attention has no sm_scale arg -> pre-scale q.
        q_scaled = (query * scaling).to(query.dtype)

        op = _get_paged_op(ppb)
        out = op(q_scaled, k_pages, v_pages, lengths, page_indices)
        output.copy_(out.to(output.dtype))
        return output

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        # The Pallas flash kernel (kernels/flash_attention.py:120) raises
        # NotImplementedError if qk_head_dim != v_head_dim. Fail loudly here
        # rather than crashing deep inside the kernel — fail-fast surfaces the
        # incompatibility instead of surprising users of MLA-style models.
        # Use `--attention-backend torch_native` for those.
        assert layer.qk_head_dim == layer.v_head_dim, (
            f"pallas_tpu_flash attention backend requires qk_head_dim == "
            f"v_head_dim, got qk={layer.qk_head_dim}, v={layer.v_head_dim}. "
            f"Re-run with --attention-backend torch_native."
        )
        o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_flash_forward_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        **kwargs,
    ):
        # Symmetric with forward_extend: forbid qk != v upfront so the
        # whole-request path (prefill + decode) gives a consistent error.
        assert layer.qk_head_dim == layer.v_head_dim, (
            f"pallas_tpu_flash attention backend requires qk_head_dim == "
            f"v_head_dim, got qk={layer.qk_head_dim}, v={layer.v_head_dim}. "
            f"Re-run with --attention-backend torch_native."
        )

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_paged_forward_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            scaling=layer.scaling,
        )

        return o

    def support_triton(self):
        return False


# ---------------------------------------------------------------------------
# Vision encoder: non-causal flash attention with segment_ids
# ---------------------------------------------------------------------------
_VISION_FLASH_OPS: dict[float, object] = {}


def _get_vision_flash_op(sm_scale: float):
    """Return a cached pallas op for non-causal flash attention with segment IDs."""
    key = float(sm_scale)
    op = _VISION_FLASH_OPS.get(key)
    if op is not None:
        return op

    from torch_tpu._internal import pallas

    def _jax_vision_flash(q, k, v, seg_ids):
        return flash_attention(
            q,
            k,
            v,
            segment_ids=SegmentIds(q=seg_ids, kv=seg_ids),
            causal=False,
            sm_scale=sm_scale,
            vmem_limit_bytes=None,
        )

    _jax_vision_flash.__annotations__ = {
        "q": jax.Array,
        "k": jax.Array,
        "v": jax.Array,
        "seg_ids": jax.Array,
        "return": jax.Array,
    }

    op_name = f"sglang_tpu::vision_flash_s{key!r}".replace(".", "_").replace("-", "m")
    op = pallas.jax_op(op_name, _jax_vision_flash)

    def _fake_vision_flash(q, k, v, seg_ids):
        return torch.empty_like(q)

    op.register_fake(_fake_vision_flash)
    _VISION_FLASH_OPS[key] = op
    return op

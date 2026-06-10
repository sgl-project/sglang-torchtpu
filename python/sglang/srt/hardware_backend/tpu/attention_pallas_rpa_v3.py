"""Pallas ragged-paged-attention backend for the TPU torch_tpu platform.

This backend wraps the JAX/Pallas v3 ragged_paged_attention kernel
(``sglang.srt.layers.attention.pallas.ragged_paged_attention_v3``) and
exposes it as a PyTorch custom op so it can be invoked from PyTorch and
traced through ``torch.compile`` with ``backend="tpu"``.

Design notes
------------

* The kernel writes K/V into the fused KV cache itself. The backend therefore
  ignores ``save_kv_cache``: the cache is always updated.
* The fused KV cache must be allocated by
  :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`.
* All attention metadata (cu_q_lens, cu_kv_lens, page_indices, distribution)
  is built once per forward pass in :meth:`init_forward_metadata` and reused
  by every layer.
* Only Qwen3-dense-style models are exercised today (causal, GQA, no
  speculative decoding, no SWA). The kernel itself supports more, so
  enabling extra features is a matter of plumbing flags through the
  ``_KernelHandle`` cache key.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here.
# ``torch_tpu._internal.pallas.jax_op`` inspects the wrapped function's
# annotations at registration time (``param.annotation is jax.Array``); under
# PEP 563 those would be plain strings and validation would reject them.
# Forward-reference types used only at type-check time are spelled as
# explicit string literals (e.g. ``"RadixAttention"``).

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


@dataclass
class PallasForwardMetadata:
    """Per-forward-pass metadata shared by every attention layer."""

    # i32[max_num_seqs]  total KV length per sequence (after appending new K/V)
    kv_lens: torch.Tensor
    # i32[max_num_seqs + 1]  cumulative sum of query lengths (starts at 0)
    cu_q_lens: torch.Tensor
    # i32[max_num_seqs + 1]  cumulative sum of page-aligned KV lengths
    cu_kv_lens: torch.Tensor
    # i32[max_num_seqs * pages_per_seq]  flat page-indices look-up table
    page_indices: torch.Tensor
    # i32[3]  (decode_end, prefill_end, mixed_end) sequence-index ranges
    distribution: torch.Tensor


_PALLAS_RPA_OP = None


def _quantize_kv_to_fp8(
    t: torch.Tensor, scale: float, fp8_dtype: torch.dtype
) -> torch.Tensor:
    """Static per-tensor quantization of K or V into ``fp8_dtype``.

    Mirrors the GPU path's ``set_kv_buffer`` quantization (``cache_k.div_(scale)
    .to(fp8)``) and torchtpu-vllm's ``quantize_kv``: divide by the per-tensor
    scale, clamp to the fp8 representable range, then cast. The kernel undoes
    this by multiplying the attention scores / output by the same scale.
    """
    finfo = torch.finfo(fp8_dtype)
    scaled = t.to(torch.float32) / scale
    scaled = torch.clamp(scaled, finfo.min, finfo.max)
    return scaled.to(fp8_dtype)


def _build_pallas_kernel_op():
    """Lazily register the ragged_paged_attention kernel as a torch custom op.

    Only one custom op is registered per process; the kernel's
    Pallas-internal block-size selection is shape-aware, so shape-dependent
    specialisation happens inside the kernel rather than at the
    ``torch.library`` boundary. We override ``register_fake`` to return
    shape-matching empties without re-tracing the JAX function — this is
    what lets the dynamic token dimension flow through ``torch.compile``.
    """
    global _PALLAS_RPA_OP
    if _PALLAS_RPA_OP is not None:
        return _PALLAS_RPA_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang.srt.layers.attention.pallas.ragged_paged_attention_v3 import (
        ragged_paged_attention as _jax_rpa,
    )

    op_name = "sglang_tpu_pallas::ragged_paged_attention"

    def _wrapped(
        queries: jax.Array,
        keys: jax.Array,
        values: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        sm_scale: float = 1.0,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> tuple[jax.Array, jax.Array]:
        return _jax_rpa(
            queries,
            keys,
            values,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            None,  # custom_mask
            None,  # attention_sink
            causal=1,
            sm_scale=sm_scale,
            sliding_window=sliding_window,
            soft_cap=soft_cap,
            # For fp8 KV cache: K/V are quantized to fp8 before this call, so
            # the kernel dequantizes on read by multiplying the attention
            # scores / output by these per-tensor scales. ``q_scale`` is left
            # unset (None) so queries stay bf16 — only K/V are quantized,
            # matching the GPU backends and torchtpu-vllm.
            k_scale=k_scale,
            v_scale=v_scale,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=(9, 10, 11, 12, 13),
        donate_argnums=(3,),
        input_output_aliases={3: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("kv_cache_fused",),
        device_types=["tpu"],
    )
    def custom_op(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        cu_kv_lens: torch.Tensor,
        distribution: torch.Tensor,
        sm_scale: float = 1.0,
        sliding_window: Optional[int] = None,
        soft_cap: Optional[float] = None,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
    ) -> torch.Tensor:
        output, _updated_kv = jax_kernel(
            queries,
            keys,
            values,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            sm_scale,
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
        )
        return output

    def _fake_impl(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        *args,
        **kwargs,
    ):
        return torch.empty_like(queries)

    custom_op.register_fake(_fake_impl)

    _PALLAS_RPA_OP = custom_op
    return custom_op


class TpuPallasRpaV3AttnBackend(AttentionBackend):
    """SGLang attention backend backed by the Pallas v3 ragged-paged-attention kernel.

    Sibling to :class:`TpuPallasAttnBackend` (``attention_pallas.py``), which
    wraps the Pallas flash + paged-attention kernels. Selected via
    ``--attention-backend pallas_tpu_rpa_v3`` (the TPU platform default).
    Requires the fused KV layout supplied by
    :class:`sglang.srt.hardware_backend.tpu.kv_pool.TpuFusedMHATokenToKVPool`;
    :func:`patches._patch_mha_kv_pool_for_rpa_v3` swaps that pool in when this
    backend is active.
    """

    # The torch custom op exposes KV updates as an in-place mutation, so syncing
    # model outputs during TPU precompile is enough to execute the graph.
    requires_kv_sync_after_compile = False

    def __init__(self, model_runner: "ModelRunner") -> None:
        super().__init__()
        self.device = model_runner.device
        self.page_size = model_runner.page_size
        self.req_to_token_pool = model_runner.req_to_token_pool

        # fp8 KV cache: when the resolved cache dtype is fp8, K/V are quantized
        # before the kernel call and the kernel dequantizes on read via
        # ``k_scale``/``v_scale``. ``None`` for bf16 caches (the default path).
        # e4m3 is the default/tested fp8 storage format (TPU's natural one);
        # e5m2 is accepted for parity. With per-tensor scaling, e4m3's extra
        # mantissa bit gives better KV-cache accuracy.
        kv_cache_dtype = getattr(model_runner, "kv_cache_dtype", None)
        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        self.kv_cache_quant_dtype: Optional[torch.dtype] = (
            kv_cache_dtype if kv_cache_dtype in fp8_dtypes else None
        )

        self.forward_metadata: Optional[PallasForwardMetadata] = None
        # Register the custom op eagerly so the (Python-introspection-heavy)
        # ``jax_op`` registration does not happen inside a ``torch.compile``
        # traced region — dynamo can't handle ``inspect.signature`` on the
        # wrapped JAX function. The forward path then just looks the op up.
        self._op = _build_pallas_kernel_op()

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        self.forward_metadata = self._build_metadata(forward_batch)

    def _build_metadata(self, forward_batch: ForwardBatch) -> PallasForwardMetadata:
        """Build per-forward-pass metadata for the Pallas kernel.

        Layout choice
        -------------
        The kernel uses ``cu_kv_lens[i]`` only as a *page-indices offset*
        (``start_kv_page_idx = cdiv(cu_kv_lens[i], page_size)``). It uses
        ``kv_lens[i]`` for the actual sequence length, and ``page_indices``
        is read as a flat 1D table of size
        ``sum_i cdiv(kv_lens[i], page_size)`` in the *ragged* world.

        We use a *rectangular* layout: every sequence reserves
        ``pages_per_seq`` slots in ``page_indices``, and we set
        ``cu_kv_lens[i] = i * pages_per_seq * page_size``. The kernel reads
        ``page_indices[i * pages_per_seq + kv_p_start]`` which is exactly
        the page for token ``kv_p_start`` of sequence ``i``.
        """
        page_size = self.page_size
        device = forward_batch.seq_lens.device
        bs = int(forward_batch.seq_lens.shape[0])
        req_to_token = self.req_to_token_pool.req_to_token
        max_context_len = req_to_token.shape[1]
        pages_per_seq = max_context_len // page_size

        seq_lens = forward_batch.seq_lens.to(torch.int32)

        if forward_batch.forward_mode.is_extend():
            assert forward_batch.extend_seq_lens is not None, (
                "extend_seq_lens is required for the Pallas attention backend in "
                "extend mode."
            )
            extend_seq_lens = forward_batch.extend_seq_lens.to(torch.int32)
        else:
            # Decode: q_len == 1 per sequence.
            extend_seq_lens = torch.ones(bs, dtype=torch.int32, device=device)

        req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)

        cu_q_lens = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        cu_q_lens[1:] = torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32)

        # cu_kv_lens[i] = i * pages_per_seq * page_size so the kernel addresses
        # ``page_indices[i * pages_per_seq + j]`` for token j of seq i.
        cu_kv_lens = torch.arange(bs + 1, dtype=torch.int32, device=device) * (
            pages_per_seq * page_size
        )

        per_seq = req_to_token[req_pool_indices]  # [bs, max_ctx]
        # ``req_to_token.shape[1]`` may not be an exact multiple of
        # ``page_size`` (SGLang adds extra slack tokens), in which case
        # ``per_seq[:, ::page_size]`` would emit one extra trailing column.
        # Trim to exactly ``pages_per_seq`` entries per row so the flat
        # ``page_indices`` shape matches the rectangular layout the kernel
        # expects.
        page_indices = (
            per_seq[:, : pages_per_seq * page_size : page_size] // page_size
        ).to(torch.int32)
        page_indices = page_indices.reshape(bs * pages_per_seq).contiguous()

        num_active = torch.full((), bs, dtype=torch.int32, device=device)
        if forward_batch.forward_mode.is_decode():
            distribution = num_active.repeat(3).to(torch.int32)
        else:
            zero = torch.zeros((), dtype=torch.int32, device=device)
            distribution = torch.stack([zero, num_active, num_active]).to(torch.int32)

        return PallasForwardMetadata(
            kv_lens=seq_lens,
            cu_q_lens=cu_q_lens,
            cu_kv_lens=cu_kv_lens,
            page_indices=page_indices,
            distribution=distribution,
        )

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        return self._forward(q, k, v, layer, forward_batch)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # During torch.compile, rotary_emb may produce a 3D query; flatten so
        # the view below matches the kernel's expected layout. This mirrors
        # TorchNativeAttnBackend.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)
        return self._forward(q, k, v, layer, forward_batch)

    def _forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        assert self.forward_metadata is not None, (
            "Pallas attention backend was not initialised for this forward pass; "
            "did you forget to call init_forward_metadata()?"
        )

        num_tokens = q.shape[0]
        q = q.view(num_tokens, layer.tp_q_head_num, layer.qk_head_dim)
        k = k.view(num_tokens, layer.tp_k_head_num, layer.qk_head_dim)
        v = v.view(num_tokens, layer.tp_v_head_num, layer.v_head_dim)

        kv_cache = forward_batch.token_to_kv_pool.get_fused_kv_buffer(layer.layer_id)

        op = self._op

        sliding_window = getattr(layer, "sliding_window_size", None)
        if sliding_window is not None and sliding_window <= 0:
            sliding_window = None

        soft_cap = layer.logit_cap if getattr(layer, "logit_cap", 0) else None

        # fp8 KV cache: quantize new K/V to fp8 here (the kernel writes them to
        # the fp8 cache itself and reads them back for the current tokens), and
        # hand the per-tensor scales to the kernel for dequant-on-read. The
        # scales come from the same ``k_scale_float``/``v_scale_float`` the GPU
        # backends use; they default to 1.0 when the checkpoint carries no KV
        # scales (e.g. a bf16-weight model run with --kv-cache-dtype fp8_e4m3).
        if self.kv_cache_quant_dtype is not None:
            k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
            v_scale = layer.v_scale_float if layer.v_scale_float is not None else 1.0
            k = _quantize_kv_to_fp8(k, k_scale, self.kv_cache_quant_dtype)
            v = _quantize_kv_to_fp8(v, v_scale, self.kv_cache_quant_dtype)
        else:
            k_scale = None
            v_scale = None

        out = op(
            q,
            k,
            v,
            kv_cache,
            self.forward_metadata.kv_lens,
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
            float(layer.scaling),
            sliding_window,
            soft_cap,
            k_scale,
            v_scale,
        )

        return out.reshape(num_tokens, layer.tp_q_head_num * layer.v_head_dim)

    # ------------------------------------------------------------------ #
    # Capability flags
    # ------------------------------------------------------------------ #

    def support_triton(self) -> bool:
        return False

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 0

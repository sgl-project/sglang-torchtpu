# SPDX-License-Identifier: Apache-2.0
"""Pallas MLA (Multi-head Latent Attention) backend for the TPU torch_tpu platform.

Wraps the JAX/Pallas MLA v2 ragged-paged-attention kernel
(``sglang_tpu.layers.attention.pallas.mla_ragged_paged_attention_v2``, vendored
from sglang-jax, itself adapted from vllm/tpu-inference) and exposes it as a
PyTorch custom op, mirroring ``attention_pallas_rpa_v3``. Selected via
``--attention-backend pallas_tpu_mla``; used by MLA models (DeepSeek V2/V3).

Design notes
------------

* Only the *absorbed* MLA path is supported: the DeepSeek dispatch handler for
  this backend returns ``AttnForwardMethod.MLA`` for every forward mode, and
  ``pallas_tpu_mla`` is registered in ``FORWARD_ABSORB_CORE_ATTENTION_BACKENDS``
  so ``forward_absorb_core`` calls ``attn_mqa`` in the split-rope form::

      attn_mqa(q_nope_out,  # [T, H, kv_lora_rank]     (q_nope · W_UK)
               k_nope,      # [T, 1, kv_lora_rank]     (latent, post layernorm)
               k_nope,      # v (ignored; same latent)
               forward_batch,
               q_rope=q_pe, # [T, H, qk_rope_head_dim] (post rotary)
               k_rope=k_pe) # [T, 1, qk_rope_head_dim] (post rotary)

  which maps 1:1 onto the kernel inputs (ql_nope, q_pe, new_kv_c, new_k_pe).
  The kernel attends over the latent cache (MQA) and returns o_latent
  ``[T, H, kv_lora_rank]``; the model folds W_UV into it afterwards.
* The kernel writes the new latent KV into the paged cache itself (the cache
  is donated to the pallas call), so ``save_kv_cache`` is ignored and the pool
  (:class:`sglang_tpu.mem_cache.kv_pool.TpuFusedMLATokenToKVPool`)
  rejects out-of-band writes.
* All attention metadata is identical to the RPA v3 backend's (the MLA v2
  kernel was adapted to the same ragged ``cu_kv_lens``-offset page-indices
  addressing), so this class reuses ``TpuPallasRpaV3AttnBackend``'s
  ``init_forward_metadata`` machinery wholesale via inheritance.
"""

# NOTE: do *not* add ``from __future__ import annotations`` here — see the
# annotation-introspection note in attention_pallas_rpa_v3.py.

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang_tpu.layers.attention.attention_pallas_rpa_base import (
    _quantize_kv_to_fp8,
)
from sglang_tpu.layers.attention.attention_pallas_rpa_v3 import (
    TpuPallasRpaV3AttnBackend,
)
from sglang_tpu.mem_cache.kv_pool import NOPE_MLA_ROPE_DIM
from sglang_tpu.utils.pallas_export import tpu_info_for_cpu_export

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention

logger = logging.getLogger(__name__)

_PALLAS_MLA_OP = None

_MLA_DECODE_BATCH_SIZE = 4


def _build_pallas_mla_kernel_op():
    """Lazily register the MLA v2 kernel as a torch custom op.

    Mirrors ``attention_pallas_rpa_v3._build_pallas_kernel_op``: one custom op
    per process; block-size selection is shape-aware inside the kernel; the
    fake impl returns a shape-matching empty so the dynamic token dimension
    flows through ``torch.compile``.
    """
    global _PALLAS_MLA_OP
    if _PALLAS_MLA_OP is not None:
        return _PALLAS_MLA_OP

    import jax  # noqa: F401  (force loud failure if JAX is missing)
    from torch_tpu._internal import pallas as _torch_pallas

    from sglang_tpu.layers.attention.pallas.mla_ragged_paged_attention_v2 import (
        mla_ragged_paged_attention as _jax_mla,
    )

    op_name = "sglang_tpu_pallas::mla_ragged_paged_attention"

    def _wrapped(
        ql_nope: jax.Array,
        q_pe: jax.Array,
        new_kv_c: jax.Array,
        new_k_pe: jax.Array,
        kv_cache_fused: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        cu_kv_lens: jax.Array,
        distribution: jax.Array,
        sm_scale: float = 1.0,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        segments: str = "all",
        decode_query_size: int = 1,
    ) -> tuple[jax.Array, jax.Array]:
        return _jax_mla(
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            kv_cache_fused,
            kv_lens,
            page_indices,
            cu_q_lens,
            cu_kv_lens,
            distribution,
            sm_scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            decode_batch_size=_MLA_DECODE_BATCH_SIZE,
            segments=segments,
            decode_query_size=decode_query_size,
        )

    jax_kernel = _torch_pallas.custom_jax_kernel(
        _wrapped,
        name=f"jax_{op_name}",
        static_argnums=(10, 11, 12, 13, 14),
        donate_argnums=(4,),
        input_output_aliases={4: 1},
        warn_deprecated=False,
    )

    @torch.library.custom_op(
        op_name,
        mutates_args=("kv_cache_fused",),
        device_types=["tpu"],
    )
    def custom_op(
        ql_nope: torch.Tensor,
        q_pe: torch.Tensor,
        new_kv_c: torch.Tensor,
        new_k_pe: torch.Tensor,
        kv_cache_fused: torch.Tensor,
        kv_lens: torch.Tensor,
        page_indices: torch.Tensor,
        cu_q_lens: torch.Tensor,
        cu_kv_lens: torch.Tensor,
        distribution: torch.Tensor,
        sm_scale: float = 1.0,
        k_scale: Optional[float] = None,
        v_scale: Optional[float] = None,
        segments: str = "all",
        decode_query_size: int = 1,
    ) -> torch.Tensor:
        # JAX is CPU-only here, so Mosaic needs the TPU generation patched in.
        # Inside the custom op like the RPA base: Dynamo can't trace the patch's lock.
        with tpu_info_for_cpu_export():
            output, _updated_kv = jax_kernel(
                ql_nope,
                q_pe,
                new_kv_c,
                new_k_pe,
                kv_cache_fused,
                kv_lens,
                page_indices,
                cu_q_lens,
                cu_kv_lens,
                distribution,
                sm_scale,
                k_scale,
                v_scale,
                segments,
                decode_query_size,
            )
        return output

    def _fake_impl(
        ql_nope: torch.Tensor,
        *args,
        **kwargs,
    ):
        return torch.empty_like(ql_nope)

    custom_op.register_fake(_fake_impl)

    _PALLAS_MLA_OP = custom_op
    return custom_op


class TpuPallasMlaAttnBackend(TpuPallasRpaV3AttnBackend):
    """SGLang attention backend backed by the Pallas MLA v2 kernel.

    Inherits the full per-forward metadata machinery (kv_lens / cu_q_lens /
    cu_kv_lens / page_indices / distribution) from the RPA v3 backend — the
    MLA kernel consumes the identical layout — and overrides only the op
    registration and the per-layer forward.
    """

    def _build_kernel_op(self):
        # Hook called by ``super().__init__()`` to populate ``self._op``. Both
        # forward modes go through ``_forward_mla``, so register the MLA v2
        # kernel op here — the parent never builds the unused RPA v3 op.
        return _build_pallas_mla_kernel_op()

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #

    def _kernel_segments(self, forward_batch: ForwardBatch, is_decode: bool) -> str:
        """MLA v2 passes the ``distribution`` built from ``is_decode`` can
        populate: decode ``[N, N, N]`` leaves MIXED empty, extend ``[0, 0, N]``
        both decode ranges. Spec servers keep every pass outside DECODE: which
        distribution verify / draft-extend build is not part of the
        compiled-graph key.
        """
        if self.decode_query_size > 1 and not forward_batch.forward_mode.is_decode():
            return "all"
        return "decode" if is_decode else "mixed"

    @staticmethod
    def _make_rpa_distribution_cpu(is_decode: bool, real_bs: int) -> torch.Tensor:
        """MLA v2 ``distribution`` semantics differ from RPA v3 for extend.

        The kernel runs three passes over ``(i, j, k) = distribution``:
        decode ``[0:i]``, prefill-only ``[i:j]`` and mixed ``[j:k]`` — but its
        PREFILL pass is dead code (kernel.py TODO), so extend batches must be
        routed through MIXED: ``[0, 0, N]`` (sglang-jax mla_backend.py does
        the same). The parent's RPA v3 shape ``[0, N, N]`` would leave every
        prefill sequence unprocessed — attention output uninitialised and the
        latent cache never written. Decode is ``[N, N, N]`` in both kernels.
        """
        num_active = torch.tensor(real_bs, dtype=torch.int32)
        if is_decode:
            return num_active.repeat(3)
        zero = torch.zeros((), dtype=torch.int32)
        return torch.stack([zero, zero, num_active])

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
        return self._forward_mla(q, k, layer, forward_batch, **kwargs)

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
        return self._forward_mla(q, k, layer, forward_batch, **kwargs)

    def _forward_mla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        out = self._op(
            *self._mla_kernel_args(
                q=q,
                k=k,
                layer=layer,
                forward_batch=forward_batch,
                q_rope=q_rope,
                k_rope=k_rope,
            ),
            # Only this dense kernel takes ``segments``. The sparse op does not.
            segments=self.forward_metadata.segments,
            decode_query_size=self.forward_metadata.decode_query_size,
        )
        # o_latent [T, H, kv_lora_rank] -> [T, H * kv_lora_rank]; the model
        # views it back to 3D and folds W_UV.
        # TODO(sglang-torchtpu#1072): avoid this reshape. The model undoes it
        # right away, and it adds latency.
        return out.reshape(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)

    def _mla_kernel_args(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: ForwardBatch,
        q_rope: Optional[torch.Tensor],
        k_rope: Optional[torch.Tensor],
    ) -> tuple:
        """The MLA op's positional arguments for one layer."""
        assert self.forward_metadata is not None, (
            "Pallas MLA attention backend was not initialised for this forward "
            "pass; did you forget to call init_forward_metadata()?"
        )
        assert q_rope is not None and k_rope is not None, (
            "pallas_tpu_mla requires the split-rope absorbed call form "
            "(q_rope=/k_rope= kwargs). Is 'pallas_tpu_mla' registered in "
            "FORWARD_ABSORB_CORE_ATTENTION_BACKENDS?"
        )

        # For the MLA attn_mqa layer: v_head_dim == kv_lora_rank, and
        # qk_head_dim == kv_lora_rank + qk_rope_head_dim.
        kv_lora_rank = layer.v_head_dim
        rope_dim = layer.qk_head_dim - layer.v_head_dim
        num_tokens = q.shape[0]

        # q arrives as [T, H, kv_lora_rank] from forward_absorb_core but is
        # NON-contiguous (bmm(...).transpose(0, 1)); k / k_rope are slices of
        # latent_cache ([T, 1, kv_lora_rank] / [T, 1, rope_dim], row stride
        # 576). ``reshape`` to the same logical shape keeps those strides, and
        # the torch->JAX pallas bridge hands the raw buffer to the kernel, so
        # every input must be made dense explicitly or the kernel reads
        # scrambled data.
        ql_nope = q.reshape(num_tokens, layer.tp_q_head_num, kv_lora_rank).contiguous()
        new_kv_c = k.reshape(num_tokens, kv_lora_rank).contiguous()
        if rope_dim == 0:
            # NoPE model: the kernels need a rope segment, so attend with zeros.
            q_pe = q.new_zeros(num_tokens, layer.tp_q_head_num, NOPE_MLA_ROPE_DIM)
            new_k_pe = k.new_zeros(num_tokens, NOPE_MLA_ROPE_DIM)
        else:
            q_pe = q_rope.reshape(
                num_tokens, layer.tp_q_head_num, rope_dim
            ).contiguous()
            new_k_pe = k_rope.reshape(num_tokens, rope_dim).contiguous()

        # fp8 KV cache: the kernel writes new_kv_c / new_k_pe into the cache
        # itself and requires them in the cache dtype. Quantize per-tensor with
        # the layer's k_scale (1.0 when the checkpoint has none) and pass the
        # same scale for K and V: MLA's V is the same latent tensor, so it
        # must be dequantized with the scale it was quantized with. Q stays
        # bf16 (the combination upstream tests).
        if self.kv_cache_quant_dtype is not None:
            k_scale = layer.k_scale_float if layer.k_scale_float is not None else 1.0
            v_scale = k_scale
            new_kv_c = _quantize_kv_to_fp8(new_kv_c, k_scale, self.kv_cache_quant_dtype)
            new_k_pe = _quantize_kv_to_fp8(new_k_pe, k_scale, self.kv_cache_quant_dtype)
        else:
            k_scale = None
            v_scale = None

        return (
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            *self._kv_cache_buffers(layer.layer_id),
            self._kv_lens(forward_batch),
            self.forward_metadata.page_indices,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
            float(layer.scaling),
            k_scale,
            v_scale,
        )

    def _kv_cache_buffers(self, layer_id: int) -> tuple:
        """The layer's KV cache buffers, in the order the op takes them.

        The sparse backend overrides this: its pool has two buffers per layer.
        """
        # Backend-cached pool: upstream moved token_to_kv_pool off ForwardBatch.
        return (self.token_to_kv_pool.get_fused_kv_buffer(layer_id),)

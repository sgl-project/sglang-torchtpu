# SPDX-License-Identifier: Apache-2.0
"""Kimi Delta Attention (KDA) linear-attention backend for the TPU torch_tpu platform.

Runs Kimi-K3's KDA layers on the fused Conv1D + gated-delta-rule Pallas kernel in
``AttentionMode.KDA`` (per-channel gate, in-kernel gate activation and beta
sigmoid). Metadata, state donation and the export guard are shared with
:class:`TpuGDNAttnBackend`; only the per-layer call differs.
"""

from typing import TYPE_CHECKING

import torch

from sglang_tpu.attention_gdn_pallas import TpuGDNAttnBackend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention


class TpuKDAAttnBackend(TpuGDNAttnBackend):
    """TPU KDA linear-attention backend backed by the fused Pallas kernel."""

    def _run(
        self,
        layer: "RadixLinearAttention",
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        num_tokens = mixed_qkv.shape[0]
        n_v, d_k, d_v = layer.num_v_heads, layer.head_k_dim, layer.head_v_dim
        # Kimi-Linear / Bailing bind [dim, K]; Kimi-K3 binds its depthwise
        # [dim, 1, K] parameter itself.
        conv_weight = layer.conv_weights
        if conv_weight.ndim == 2:
            conv_weight = conv_weight.unsqueeze(1)
        # The kernel takes the raw forget gate [T, n_v * d_k] and raw beta [T, n_v];
        # the model hands them over with leading batch / per-head axes. Fused
        # projections pass split views, and the op reads raw buffers, so make
        # every input dense.
        state_rows, read_offsets = self._state_rows()
        out = self._op(
            mixed_qkv.contiguous(),
            b.reshape(num_tokens, n_v).contiguous(),
            a.reshape(num_tokens, n_v * d_k).contiguous(),
            layer_cache.conv[0],
            layer_cache.temporal,
            conv_weight,
            layer.bias,
            layer.A_log,
            # Per-channel under KDA; Mosaic rejects the flat [n_v * d_k] form.
            layer.dt_bias.reshape(n_v, d_k),
            self._query_start_loc,
            state_rows,
            self._distribution,
            self._seq_lens,
            layer.num_k_heads,
            n_v,
            d_k,
            d_v,
            conv_weight.shape[-1],
            True,
            layer.lower_bound,
            batched_only=self._is_decode,
            prefill_only=not self._is_decode,
            # Kimi-K3 passes its fused [q, k, v, g] projection as mixed_qkv.
            fused_qkvg=getattr(layer, "fused_qkvg", False),
            # Kimi-K3 packs conv / dt_bias / A_log into one f32 operand post-load.
            packed_weights=getattr(layer, "packed_kernel_weights", None),
            read_offsets=read_offsets,
            num_spec_tokens=self._num_spec_tokens,
        )
        # Keep the batch axis: the model squeezes dim 0 after o_norm, which would
        # eat the token axis of a one-token decode.
        return out.view(1, num_tokens, n_v, d_v)

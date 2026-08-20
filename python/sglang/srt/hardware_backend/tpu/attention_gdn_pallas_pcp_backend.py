# SPDX-License-Identifier: Apache-2.0
"""Gated-DeltaNet backend for the prefix-context-parallel path.

Differs from the dense backend in three ways, all forced by the fused kernel:

* it takes ``hidden_states`` and the QKVZ projection weight, not a projected
  ``mixed_qkv`` -- the kernel does the fp8 projection itself, inside the same
  pipeline that drives the head-shard exchange;
* it returns ``z`` alongside the attention output, because the kernel produces
  it as part of that projection and the layer's gated norm needs it;
* state lives in a byte-addressed pool rather than two dense tensors.

Slot semantics are unchanged: the same ``state_indices`` the dense path uses
address the same request slots, so the pool is a storage swap, not a
bookkeeping one.
"""

# NOTE: no ``from __future__ import annotations`` -- the op registered below
# infers its schema from real annotations.

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.srt.hardware_backend.tpu._logging import get_logger
from sglang.srt.hardware_backend.tpu.attention_gdn_pallas import TpuGDNAttnBackend
from sglang.srt.hardware_backend.tpu.attention_gdn_pallas_pcp import (
    build_fused_gdn_pcp_op,
    invoke_fused_gdn_pcp_op,
)
from sglang.srt.hardware_backend.tpu.attention_pallas_pcp_streaming_op import (
    get_pcp_streaming_mesh,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = get_logger(__name__)


class TpuGDNPcpAttnBackend(TpuGDNAttnBackend):
    """Fused context-parallel GDN.

    One op serves every linear layer: the geometry a GDN layer specializes on
    (head counts, head dims, conv kernel, pool block size, group size) is the
    same for all of them, so the op is built once at construction and the
    per-layer difference is only which weights and which slice of the pool it
    is handed.
    """

    def __init__(
        self, model_runner: "ModelRunner", *, pcp_size: int, interleave_size: int
    ):
        super().__init__(model_runner)
        self._pcp_size = int(pcp_size)
        self._interleave_size = int(interleave_size)
        self._mesh = get_pcp_streaming_mesh()
        self._pcp_op = None
        self._pooled_state = None

    def attach_pooled_state(self, pooled_state) -> None:
        """Bind the byte-addressed state pool this backend reads and writes."""
        self._pooled_state = pooled_state

    def attach_to_model(self, model) -> int:
        """Point every Gated-DeltaNet layer at this backend.

        The layer has to call in through a different entry point than the
        dense path (it hands over hidden states and gets Z back), so it reads
        a plain attribute set here rather than resolving a backend mid-forward
        -- the compiled forward must see a constant.
        """
        attached = 0
        first_layer = None
        for module in model.modules():
            if hasattr(module, "_pcp_gdn_backend") and hasattr(module, "in_proj_qkvz"):
                module._pcp_gdn_backend = self
                first_layer = first_layer or module.attn
                attached += 1
        if attached == 0:
            raise RuntimeError(
                "Context-parallel GDN is enabled but no Gated-DeltaNet layer "
                "was found to attach to."
            )
        # Every linear layer shares head counts, head dims, conv kernel and
        # pool block size, so one op serves all of them.
        self._build_op(first_layer)
        logger.info("Context-parallel GDN attached to %d linear layers.", attached)
        return attached

    def _build_op(self, layer: "RadixLinearAttention") -> None:
        """Register the op. Runs at attach time, never inside a forward.

        Registration JAX-exports the kernel, which must not happen inside a
        torch.compile trace.
        """
        if self._pooled_state is None:
            raise RuntimeError(
                "The context-parallel GDN backend has no state pool; "
                "attach_pooled_state must run before attach_to_model."
            )
        self._pcp_op = build_fused_gdn_pcp_op(
            name="sglang_tpu_pallas::gdn_attention_pcp",
            mesh=self._mesh,
            n_kq=layer.num_k_heads,
            n_v=layer.num_v_heads,
            d_k=layer.head_k_dim,
            d_v=layer.head_v_dim,
            kernel_size=layer.conv_weights.shape[-1],
            pool_block_tokens=self._pooled_state.pool_block_tokens,
            pcp_size=self._pcp_size,
            interleave_size=self._interleave_size,
        )

    def forward_gdn_pcp(
        self,
        *,
        layer: "RadixLinearAttention",
        hidden_states: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        qkvz_weight: torch.Tensor,
        qkvz_weight_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run one linear-attention layer. Returns ``(core_attn_out, z)``."""
        op = self._pcp_op
        pool = self._pooled_state.layer_state(layer.layer_id)
        out, z = invoke_fused_gdn_pcp_op(
            op,
            pool=pool,
            hidden_states=hidden_states,
            qkvz_weight=qkvz_weight,
            qkvz_weight_scale=qkvz_weight_scale,
            b=b,
            a=a,
            # RadixLinearAttention keeps the conv weight as [dim, K]; the
            # kernel wants the depthwise [dim, 1, K].
            conv_weight=layer.conv_weights.unsqueeze(1),
            conv_bias=layer.bias,
            a_log=layer.A_log,
            dt_bias=layer.dt_bias,
            state_indices=self._state_indices,
            query_start_loc=self._query_start_loc,
            distribution=self._distribution,
            seq_lens=self._seq_lens,
        )
        return out, z

    def _run(self, layer, mixed_qkv, a, b):  # pragma: no cover - guarded
        raise RuntimeError(
            "The context-parallel GDN backend does not take a projected "
            "mixed_qkv: the fused kernel performs the QKVZ projection itself. "
            "The layer must call forward_gdn_pcp with hidden_states."
        )

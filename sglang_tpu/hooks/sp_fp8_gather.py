# SPDX-License-Identifier: Apache-2.0
"""1-byte FP8 all-gather of the sequence-parallel stage input on TPU.

sp_gather_stage_input
  Why: on an input-scattered batch (--enable-attn-tp-input-scattered) every
       decoder stage reduce-scatters its output onto each rank's slice, updates
       the residual and reads the stage input there, then all-gathers that bf16
       input back to all rows. The gather moves 2 bytes per element although its
       FP8 consumers (dense FFN, shared and routed experts, MLA latent
       projections) quantize each row to 1 byte anyway. With
       SGLANG_TPU_SP_FP8_ALL_GATHER=1 each rank quantizes its slice per row
       (the same amax/fp8-max scale as the FP8 linear's activation quantizer)
       and the gather moves the 1-byte values plus one fp32 scale per row; the
       receiver dequantizes to bf16. The residual is never gathered and stays
       bf16.
  How: AROUND layer_boundary.prepare.sp_gather_stage_input.
  Upstream PR or issue: https://github.com/sgl-project/sglang-torchtpu/issues/2335
  Remove when: the stage consumers take the FP8 rows and scales directly
       (no dequantize), via the boundary's quantized-read path.
"""

from __future__ import annotations

import torch

from sglang_tpu.envs import envs

_FP8_DTYPE = torch.float8_e4m3fn


def quantize_rows_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row FP8 quantization: ``x ~= q * scale`` with ``scale`` [rows, 1] fp32.

    Same arithmetic as ``sglang_tpu.fp8_linear._quantize_activation_per_row``,
    so a consumer that re-quantizes the dequantized row recovers (up to bf16
    rounding of the dequantized values) the same scale and bytes."""
    dtype_max = torch.finfo(_FP8_DTYPE).max
    scale = torch.clamp(
        x.abs().amax(dim=1, keepdim=True).float() / dtype_max, min=1e-12
    )
    q = (x.float() / scale).to(_FP8_DTYPE)
    return q, scale


def fp8_gather_rows(
    hidden_states: torch.Tensor, shard: torch.Tensor, all_gather_into_tensor
) -> None:
    """All-gather ``shard`` into ``hidden_states`` as FP8 rows plus per-row scales.

    ``all_gather_into_tensor(output, input)`` gathers over the attention-TP group
    in rank order, so row block ``r`` of the output is rank ``r``'s shard."""
    rows, dim = shard.shape
    total_rows = hidden_states.shape[0]
    q, scale = quantize_rows_fp8(shard)
    # Gather the bytes as uint8 (same itemsize): collectives need not support
    # float8, and the reinterpretation is free.
    q_all = torch.empty((total_rows, dim), dtype=torch.uint8, device=shard.device)
    all_gather_into_tensor(q_all, q.view(torch.uint8))
    scale_all = torch.empty((total_rows, 1), dtype=scale.dtype, device=shard.device)
    all_gather_into_tensor(scale_all, scale)
    hidden_states.copy_(
        (q_all.view(_FP8_DTYPE).float() * scale_all).to(hidden_states.dtype)
    )


def sp_gather_stage_input(original, hidden_states: torch.Tensor, shard: torch.Tensor):
    if (
        not envs.SGLANG_TPU_SP_FP8_ALL_GATHER.get()
        or shard.dtype not in (torch.bfloat16, torch.float16, torch.float32)
        or shard.dim() != 2
        or shard.shape[0] == 0
    ):
        return original(hidden_states, shard)
    from sglang.srt.layers.dp_attention import attn_tp_all_gather_into_tensor

    fp8_gather_rows(hidden_states, shard, attn_tp_all_gather_into_tensor)

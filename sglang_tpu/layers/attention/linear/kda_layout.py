# SPDX-License-Identifier: Apache-2.0
"""Layout of the packed KDA kernel weights.

Kimi-K3 packs its conv weight, dt_bias and A_log into one f32 operand after
load (``pack_kda_weights``); the GDN/KDA Pallas kernel reads it back
(``layers/attention/pallas/gdn/memory_ref.unpack_kda_weights``). Pure Python at
import, so the model imports it on hosts without JAX.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def kda_packed_weights_shape(
    kernel_size: int, dim_size: int, num_v_heads: int, kq_head_dim: int
) -> tuple[int, int]:
    """[rows, width] of the packed f32 KDA weights: conv [K, dim] in rows
    [0, K), dt_bias [n_v, d_k] in rows [K, K + n_v), A_log [n_v] in row
    K + n_v. Rows round up to the f32 sublane tile (8), width to the lane
    tile (128)."""
    rows = -(-(kernel_size + num_v_heads + 1) // 8) * 8
    width = -(-max(dim_size, kq_head_dim, num_v_heads) // 128) * 128
    return rows, width


def pack_kda_weights(
    conv_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    a_log: torch.Tensor,
    num_v_heads: int,
    kq_head_dim: int,
) -> torch.Tensor:
    """Host f32 array in the ``kda_packed_weights_shape`` layout from the
    depthwise conv weight [dim, 1, K], dt_bias (n_v * d_k values) and A_log
    (n_v values). Built on the host and moved once: slice-assigning into a
    device tensor leaves a view chain that torch_tpu re-materializes before
    every forward."""
    import torch

    dim, kernel_size = conv_weight.shape[0], conv_weight.shape[-1]
    rows, width = kda_packed_weights_shape(kernel_size, dim, num_v_heads, kq_head_dim)
    packed = torch.zeros(rows, width, dtype=torch.float32)
    with torch.no_grad():
        packed[:kernel_size, :dim] = (
            conv_weight.detach().float().reshape(dim, kernel_size).t().cpu()
        )
        packed[kernel_size : kernel_size + num_v_heads, :kq_head_dim] = (
            dt_bias.detach().float().reshape(num_v_heads, kq_head_dim).cpu()
        )
        packed[kernel_size + num_v_heads, :num_v_heads] = (
            a_log.detach().float().reshape(num_v_heads).cpu()
        )
    return packed

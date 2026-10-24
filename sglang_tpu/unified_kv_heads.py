# SPDX-License-Identifier: Apache-2.0
"""Merged sliding-window KV heads of the unified KV pool (see unified_kv_pool).

A leaf module: the attention backends import it when they load, which happens
before out-of-tree plugins may swap the KV pool base classes, so it must not
import any KV pool module.
"""

from __future__ import annotations

from typing import Tuple

import torch


def merge_swa_kv_heads(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``k``/``v`` ``[T, Hkv, D]`` into 1 KV head ``[T, 1, Hkv * D]``
    and block-diagonalize ``q`` ``[T, Hq, D]`` -> ``[T, Hq, Hkv * D]``.

    Uses a one-hot ``torch.eye`` multiply so ``torch_tpu`` avoids strided
    slice concatenations.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_kv_heads == 1:
        return q, k, v
    merged_dim = num_kv_heads * head_dim
    eye = torch.eye(num_kv_heads, dtype=q.dtype, device=q.device).view(
        1, num_kv_heads, 1, num_kv_heads, 1
    )
    q_merged = (
        q.reshape(num_tokens, num_kv_heads, num_q_heads // num_kv_heads, 1, head_dim)
        * eye
    ).reshape(num_tokens, num_q_heads, merged_dim)
    return (
        q_merged,
        k.reshape(num_tokens, 1, merged_dim),
        v.reshape(num_tokens, 1, merged_dim),
    )


def split_swa_output(out: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Unpack block-diagonal SWA attention output ``[T, Hq, Hkv * D]`` -> ``[T, Hq, D]``."""
    if num_kv_heads == 1:
        return out
    num_tokens, num_q_heads, merged_dim = out.shape
    head_dim = merged_dim // num_kv_heads
    eye = torch.eye(num_kv_heads, dtype=out.dtype, device=out.device).view(
        1, num_kv_heads, 1, num_kv_heads, 1
    )
    return (
        (
            out.reshape(
                num_tokens,
                num_kv_heads,
                num_q_heads // num_kv_heads,
                num_kv_heads,
                head_dim,
            )
            * eye
        )
        .sum(dim=3)
        .reshape(num_tokens, num_q_heads, head_dim)
    )

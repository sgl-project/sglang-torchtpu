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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    target_kv_heads: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack ``k``/``v`` ``[T, Hkv, D]`` into ``target_kv_heads`` KV heads
    ``[T, target_kv_heads, (Hkv // target_kv_heads) * D]`` and block-diagonalize
    ``q`` within each target KV head group.

    Uses a one-hot ``torch.eye`` multiply so ``torch_tpu`` avoids strided
    slice concatenations.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if num_kv_heads == target_kv_heads:
        return q, k, v
    group_factor = num_kv_heads // target_kv_heads
    merged_dim = group_factor * head_dim
    eye = torch.eye(group_factor, dtype=q.dtype, device=q.device).view(
        1, 1, group_factor, 1, group_factor, 1
    )
    q_merged = (
        q.reshape(
            num_tokens,
            target_kv_heads,
            group_factor,
            num_q_heads // num_kv_heads,
            1,
            head_dim,
        )
        * eye
    ).reshape(num_tokens, num_q_heads, merged_dim)
    return (
        q_merged,
        k.reshape(num_tokens, target_kv_heads, merged_dim),
        v.reshape(num_tokens, target_kv_heads, merged_dim),
    )


def split_swa_output(
    out: torch.Tensor, num_kv_heads: int, target_kv_heads: int = 1
) -> torch.Tensor:
    """Unpack block-diagonal SWA attention output back to ``[T, Hq, D]``."""
    if num_kv_heads == target_kv_heads:
        return out
    group_factor = num_kv_heads // target_kv_heads
    num_tokens, num_q_heads, merged_dim = out.shape
    head_dim = merged_dim // group_factor
    eye = torch.eye(group_factor, dtype=out.dtype, device=out.device).view(
        1, 1, group_factor, 1, group_factor, 1
    )
    return (
        (
            out.reshape(
                num_tokens,
                target_kv_heads,
                group_factor,
                num_q_heads // num_kv_heads,
                group_factor,
                head_dim,
            )
            * eye
        )
        .sum(dim=4)
        .reshape(num_tokens, num_q_heads, head_dim)
    )

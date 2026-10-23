# SPDX-License-Identifier: Apache-2.0
"""Merged sliding-window KV heads of the unified KV pool (see unified_kv_pool).

A leaf module: the attention backends import it when they load, which happens
before out-of-tree plugins may swap the KV pool base classes, so it must not
import any KV pool module.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def merge_swa_kv_heads(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a sliding-window layer's KV heads as one head of their joint width.

    ``q`` is [tokens, q_heads, head_dim], ``k`` and ``v`` [tokens, kv_heads,
    head_dim]. The KV heads are concatenated, and each query head sits in its
    own KV head's slice with zeros elsewhere, so its scores over the merged head
    are its scores over its own KV head.
    """
    num_tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    width = num_kv_heads * head_dim
    # Query head i reads KV head i // per_group. torch.cat, not torch.stack:
    # stacking sliced views returns wrong values on torch_tpu.
    groups = q.reshape(num_tokens, num_kv_heads, num_q_heads // num_kv_heads, head_dim)
    q = torch.cat(
        [
            F.pad(
                groups[:, g : g + 1], (g * head_dim, (num_kv_heads - 1 - g) * head_dim)
            )
            for g in range(num_kv_heads)
        ],
        dim=1,
    )
    return (
        q.reshape(num_tokens, num_q_heads, width),
        k.reshape(num_tokens, 1, width),
        v.reshape(num_tokens, 1, width),
    )


def split_swa_output(out: torch.Tensor, num_kv_heads: int) -> torch.Tensor:
    """Each query head's output from its own KV head's slice of the merged
    head's output [tokens, q_heads, kv_heads * head_dim]."""
    num_tokens, num_q_heads, width = out.shape
    head_dim = width // num_kv_heads
    # [tokens, KV head of the query, query in group, KV slice, head_dim]
    groups = out.reshape(
        num_tokens, num_kv_heads, num_q_heads // num_kv_heads, num_kv_heads, head_dim
    )
    out = torch.cat([groups[:, g : g + 1, :, g] for g in range(num_kv_heads)], dim=1)
    return out.reshape(num_tokens, num_q_heads, head_dim)

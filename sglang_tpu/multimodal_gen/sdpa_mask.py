# SPDX-License-Identifier: Apache-2.0
"""SDPA with query-broadcast masks, for TPU workers.

Runtime patch, tracked debt:
- Why: torch_tpu SDPA returns NaN for masks that broadcast over queries, e.g.
  [B, 1, 1, S] key padding; reproduced on torch_tpu 0.2.0rc4 (libtpu 0.0.50).
- How: :func:`install` wraps ``torch.nn.functional.scaled_dot_product_attention``
  to expand such a mask's query dim (a view) for TPU tensors. Module-level
  aliases bound before worker init keep the unpatched function; hunyuan3d_vae's
  is the only one and passes no mask.
- Upstream PR or issue: none (torch_tpu bug).
- Remove when: test_diffusion_tpu_sdpa_mask's canary stops reproducing the NaN.
"""

from __future__ import annotations

import functools

import torch


def expand_query_broadcast_mask(query, attn_mask):
    """Expand a mask's size-1 query dim to the query length on TPU."""
    if (
        attn_mask is not None
        and query.device.type == "tpu"
        and attn_mask.dim() >= 2
        and attn_mask.shape[-2] == 1
        and query.shape[-2] > 1
    ):
        attn_mask = attn_mask.expand(
            *attn_mask.shape[:-2], query.shape[-2], attn_mask.shape[-1]
        )
    return attn_mask


def install() -> None:
    """Route SDPA masks through :func:`expand_query_broadcast_mask`; idempotent."""
    sdpa = torch.nn.functional.scaled_dot_product_attention
    if getattr(sdpa, "_sglang_tpu_mask_patch", False):
        return

    @functools.wraps(sdpa)
    def scaled_dot_product_attention(query, key, value, attn_mask=None, *a, **kw):
        attn_mask = expand_query_broadcast_mask(query, attn_mask)
        return sdpa(query, key, value, attn_mask, *a, **kw)

    scaled_dot_product_attention._sglang_tpu_mask_patch = True
    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention

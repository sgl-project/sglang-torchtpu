# SPDX-License-Identifier: Apache-2.0
"""ViT attention on TPU: the Pallas flash kernel with segment IDs.

Registered as the ``"tpu"`` multimodal attention backend from ``init_backend()``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from sglang.srt.layers.attention.vision import (
    SingletonCache,
    VisionSdpaAttention,
    resolve_seqlens,
)


def _build_tpu_segment_ids(
    cu_seqlens: torch.Tensor | SingletonCache | None,
    bsz: int,
    seq_len: int,
    *,
    flatten_batch: bool,
    device: torch.device,
) -> torch.Tensor:
    cu_seqlens = resolve_seqlens(cu_seqlens, bsz, seq_len, device=device).to(
        device=device, dtype=torch.int32
    )
    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]

    if flatten_batch and bsz == 1:
        starts = cu_seqlens[:-1].to(torch.int64)
        markers = torch.zeros(seq_len + 1, device=device, dtype=torch.int32)
        markers = markers.scatter_add(
            0,
            starts,
            torch.ones(starts.shape, device=device, dtype=torch.int32),
        )
        return (markers.cumsum(0, dtype=torch.int32)[:seq_len] - 1).unsqueeze(0)

    if seq_lens.shape[0] != bsz:
        raise ValueError(
            f"Expected {bsz} sequence lengths for a dense batch, got "
            f"{seq_lens.shape[0]}"
        )
    positions = torch.arange(seq_len, device=device, dtype=torch.int32)
    return (positions.unsqueeze(0) >= seq_lens.unsqueeze(1)).to(torch.int32)


class VisionTpuAttention(nn.Module):
    _BLOCK_SIZE = 128
    supports_segment_ids = True

    def __init__(
        self,
        head_dim,
        num_heads,
        num_kv_heads,
        dropout=0.0,
        flatten_batch=False,
        softmax_in_single_precision=False,
        softmax_scale=None,
        **kwargs,
    ):
        super().__init__()
        self.head_size = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.flatten_batch = flatten_batch
        self.scale = (
            softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(head_dim)
        )
        # Lazy: the Pallas kernel module imports jax, which CPU-only hosts lack.
        from sglang_tpu.attention_pallas import _get_vision_flash_op

        self._flash_op = _get_vision_flash_op(float(self.scale))
        self._sdpa_fallback = VisionSdpaAttention(
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
            flatten_batch=False,
            softmax_in_single_precision=softmax_in_single_precision,
            softmax_scale=softmax_scale,
        )

    def forward(
        self,
        q,
        k,
        v,
        bsz,
        cu_seqlens=None,
        attention_mask=None,
        segment_ids=None,
        **kwargs,
    ):
        if attention_mask is not None and segment_ids is not None:
            raise ValueError(
                "TPU vision attention cannot combine attention_mask and segment_ids"
            )

        if k.shape[1] != v.shape[1] or k.shape[1] != q.shape[1]:
            if k.shape[1] != v.shape[1] or q.shape[1] % k.shape[1] != 0:
                raise ValueError(
                    "TPU vision attention requires matching KV heads and an "
                    "integer query-to-KV head ratio"
                )
            repeats = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(repeats, dim=1)
            v = v.repeat_interleave(repeats, dim=1)

        if attention_mask is not None:
            return self._sdpa_fallback(
                q,
                k,
                v,
                bsz,
                cu_seqlens=cu_seqlens,
                attention_mask=attention_mask,
            )

        s = q.shape[0] // bsz
        if segment_ids is None:
            segment_ids = _build_tpu_segment_ids(
                cu_seqlens,
                bsz,
                s,
                flatten_batch=self.flatten_batch,
                device=q.device,
            )
        else:
            segment_ids = segment_ids.to(device=q.device, dtype=torch.int32)
        q, k, v = [rearrange(x, "(b s) h d -> b h s d", b=bsz) for x in [q, k, v]]
        pad = (-s) % self._BLOCK_SIZE
        if pad > 0:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            num_segs = segment_ids.max() + 1
            pad_ids = num_segs.expand(bsz, pad)
            segment_ids = torch.cat([segment_ids, pad_ids], dim=1)
        output = self._flash_op(q, k, v, segment_ids)
        if pad > 0:
            output = output[:, :, :s, :]
        output = output.contiguous()
        output = rearrange(output, "b h s d -> (b s) h d")
        return output

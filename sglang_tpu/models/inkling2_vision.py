# SPDX-License-Identifier: Apache-2.0
"""Inkling2's PE-Lang vision encoder on TPU, in fixed per-image patch buckets.

The math follows ``sglang.tml.vision.Inkling2VisionEncoder`` (the reference and
this module's test oracle): a 23-layer pre-LN ViT over 20x20 native patches with
a CLS token, bilinear position-grid interpolation, 2D RoPE and LayerScale, then a
2x2 merge and a GELU projector to the decoder width. Everything shape-dependent
(interpolation weights, RoPE angles, attention segments, the merge order) is
computed on the host per image; the device sees one padded bucket of rows, so
each bucket compiles once.
"""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.layers.attention.vision import VisionAttention, VisionTpuAttention
from sglang.srt.utils import add_prefix

# ViT rows (native patches plus one CLS per image) a bucket holds; the processor
# downscales images that do not fit the largest.
ROW_BUCKETS = (512, 2048, 8192, 16384, 32768, 65536)

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def row_bucket(rows: int) -> int:
    for bucket in ROW_BUCKETS:
        if rows <= bucket:
            return bucket
    raise ValueError(
        f"Inkling2 image needs {rows} vision rows; maximum is {ROW_BUCKETS[-1]}"
    )


def _apply_rope(query, key, frequencies, x_shape=None):
    """Rotate (even, odd) pairs by the per-row angles; frequencies [rows, dim/2, 2]."""
    cosine = frequencies[..., 0].unsqueeze(1)
    sine = frequencies[..., 1].unsqueeze(1)

    def rotate(value):
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        even, odd = pairs.unbind(dim=-1)
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        )
        return rotated.flatten(-2).to(value.dtype)

    return rotate(query), rotate(key)


class Inkling2VisionLayer(nn.Module):
    def __init__(
        self, width: int, heads: int, mlp_ratio: float, eps: float, prefix: str
    ):
        super().__init__()
        head_dim = width // heads
        self.ln_1 = nn.LayerNorm(width, eps=eps)
        self.self_attn = VisionAttention(
            embed_dim=width,
            num_heads=heads,
            projection_size=width,
            use_qkv_parallel=True,
            qkv_bias=True,
            proj_bias=True,
            softmax_scale=head_dim**-0.5,
            flatten_batch=True,
            customized_position_embedding_applier=_apply_rope,
            # Every rank encodes whole images, as the reference does on its rank 0.
            use_data_parallel=True,
            prefix=add_prefix("self_attn", prefix),
        )
        if type(self.self_attn.qkv_backend) is VisionTpuAttention:
            # Splash skips the padding blocks and keeps VMEM flat past 16k rows.
            from sglang_tpu.attention.vision_splash import VisionTpuSplashAttention

            self.self_attn.qkv_backend = VisionTpuSplashAttention(
                head_dim, heads, heads, flatten_batch=True, softmax_scale=head_dim**-0.5
            )
        if not getattr(self.self_attn.qkv_backend, "supports_segment_ids", False):
            raise ValueError(
                "Inkling2 vision needs an mm attention backend that honors segment_ids; "
                f"{self.self_attn.qkv_backend_name!r} ignores them"
            )
        self.ln_2 = nn.LayerNorm(width, eps=eps)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(width, int(width * mlp_ratio))
        self.mlp.c_proj = nn.Linear(int(width * mlp_ratio), width)
        self.ls_1 = nn.Module()
        self.ls_1.gamma = nn.Parameter(torch.ones(width))
        self.ls_2 = nn.Module()
        self.ls_2.gamma = nn.Parameter(torch.ones(width))

    def forward(self, hidden, frequencies, segment_ids):
        attn = self.self_attn(
            self.ln_1(hidden), segment_ids=segment_ids, position_embeddings=frequencies
        ).squeeze(0)
        hidden = hidden + attn * self.ls_1.gamma
        mlp = self.mlp.c_proj(F.gelu(self.mlp.c_fc(self.ln_2(hidden))))
        return hidden + mlp * self.ls_2.gamma


class Inkling2VisionEncoderTpu(nn.Module):
    """Submodule names mirror the checkpoint's ``model.visual.*`` tensors."""

    def __init__(
        self, vision_config: dict, text_hidden_size: int, prefix: str = "visual"
    ):
        super().__init__()
        cfg = vision_config
        width, heads = int(cfg["hidden_size"]), int(cfg["num_attention_heads"])
        self.width = width
        self.head_dim = width // heads
        self.patch_size = int(cfg["native_patch_size"])
        self.merge_size = int(cfg["adapter_merge_size"])
        self.temporal_patch_size = int(cfg["temporal_patch_size"])
        self.posemb_grid_size = int(cfg["posemb_grid_size"])
        self.text_hidden_size = text_hidden_size
        if bool(cfg["projector_bias"]):
            raise ValueError("Inkling2 vision on TPU expects projector_bias=False")

        self.backbone = nn.Module()
        self.backbone.embeddings = nn.Module()
        self.backbone.embeddings.patch_embedding = nn.Linear(
            3 * self.patch_size**2, width, bias=False
        )
        self.backbone.embeddings.position_embedding = nn.Parameter(
            torch.empty(self.posemb_grid_size**2, width)
        )
        self.backbone.embeddings.class_embedding = nn.Parameter(torch.empty(width))
        self.backbone.embeddings.cls_position_embedding = nn.Parameter(
            torch.empty(width)
        )
        self.backbone.ln_pre = nn.LayerNorm(width, eps=float(cfg["layer_norm_eps"]))
        self.backbone.layers = nn.ModuleList(
            Inkling2VisionLayer(
                width,
                heads,
                float(cfg["mlp_ratio"]),
                float(cfg["layer_norm_eps"]),
                add_prefix(f"backbone.layers.{i}", prefix),
            )
            for i in range(int(cfg["num_hidden_layers"]))
        )
        projection = int(cfg["projection_hidden_dim"])
        self.proj_in = nn.Linear(self.merge_size**2 * width, projection, bias=False)
        self.proj_out = nn.Linear(projection, text_hidden_size, bias=False)

    def forward(
        self,
        pixels,  # [bucket, 3 * p * p] bf16; zero rows at CLS and padding
        pos_weights_y,  # [bucket, grid] f32 interpolation weights (zero at CLS/padding)
        pos_weights_x,  # [bucket, grid] f32
        cls_mask,  # [bucket, 1] bf16, one at CLS rows
        frequencies,  # [bucket, head_dim/2, 2] f32 (cos, sin); zero angle at CLS
        segment_ids,  # [1, bucket] int32, one segment per image, padding apart
        merge_order,  # [bucket // 4 * 4] int32 rows per merged token in (dy, dx) order
    ):
        emb = self.backbone.embeddings
        hidden = emb.patch_embedding(pixels)
        grid = self.posemb_grid_size
        table = emb.position_embedding.float().reshape(grid, grid, self.width)
        positional = torch.einsum("ni,nj,ijd->nd", pos_weights_y, pos_weights_x, table)
        hidden = hidden + positional.to(hidden.dtype)
        cls = (emb.class_embedding + emb.cls_position_embedding).to(hidden.dtype)
        hidden = hidden * (1 - cls_mask) + cls_mask * cls
        hidden = self.backbone.ln_pre(hidden)
        for layer in self.backbone.layers:
            hidden = layer(hidden, frequencies, segment_ids)
        merged = hidden[merge_order.long()].reshape(-1, self.merge_size**2 * self.width)
        return self.proj_out(F.gelu(self.proj_in(merged)))


def _interp_weights(indices: torch.Tensor, target: int, size: int) -> torch.Tensor:
    """Reference ``_position_embeddings`` weights: triangle kernel over the grid axis."""
    center = (size / target) * (indices.float() + 0.5)
    distance = torch.arange(size, dtype=torch.float32)[None, :] + 0.5
    weights = (1 - (distance - center[:, None]).abs()).clamp(min=0)
    return weights / weights.sum(dim=1, keepdim=True)


@functools.lru_cache(maxsize=256)
def vision_metadata(
    grid: tuple[tuple[int, int], ...],
    *,
    merge: int,
    head_dim: int,
    posemb_grid: int,
    bucket: int,
):
    """Host-side per-bucket inputs for a batch of images with merged grids (h, w)."""
    rows = sum(h * w * merge * merge + 1 for h, w in grid)
    if rows > bucket:
        raise ValueError("vision rows exceed the bucket")
    wy = torch.zeros(bucket, posemb_grid)
    wx = torch.zeros(bucket, posemb_grid)
    cls_mask = torch.zeros(bucket, 1)
    angles = torch.zeros(bucket, head_dim // 2)
    segments = torch.full((bucket,), len(grid), dtype=torch.int32)
    merge_order = []
    frequencies = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 4, dtype=torch.float32) / head_dim)
    )
    offset = 0
    for image, (h, w) in enumerate(grid):
        height, width = h * merge, w * merge
        count = height * width
        native = torch.arange(count)
        y, x = native // width, native % width
        start = offset + 1
        wy[start : start + count] = _interp_weights(y, height, posemb_grid)
        wx[start : start + count] = _interp_weights(x, width, posemb_grid)
        angles[start : start + count] = torch.cat(
            (
                (x + 1).float()[:, None] * frequencies,
                (y + 1).float()[:, None] * frequencies,
            ),
            dim=-1,
        )
        cls_mask[offset] = 1
        segments[offset : start + count] = image
        for hh in range(h):
            for ww in range(w):
                for dy in range(merge):
                    for dx in range(merge):
                        merge_order.append(
                            start + (hh * merge + dy) * width + ww * merge + dx
                        )
        offset = start + count
    merged_bucket = bucket // (merge * merge)
    merge_order += [bucket - 1] * (merged_bucket * merge * merge - len(merge_order))
    return (
        wy,
        wx,
        cls_mask.to(torch.bfloat16),
        torch.stack((angles.cos(), angles.sin()), dim=-1),
        segments.unsqueeze(0),
        torch.tensor(merge_order, dtype=torch.int32),
    )


def prepare_vision_input(
    patches: torch.Tensor,
    grid: list[tuple[int, int]],
    *,
    patch_size: int,
    merge: int,
    head_dim: int,
    posemb_grid: int,
):
    """Host preprocessing of the processor's [N, T, 40, 40, 3] bf16 patches into one bucket.

    Returns the encoder's positional inputs and the merged-token count.
    """
    if patches.device.type != "cpu":
        raise ValueError(
            "Inkling2 vision preprocessing keeps image patches on the host"
        )
    mean = torch.tensor(_CLIP_MEAN, dtype=torch.float32).reshape(1, 1, 1, 1, 3)
    std = torch.tensor(_CLIP_STD, dtype=torch.float32).reshape(1, 1, 1, 1, 3)
    # Reference: renormalize in the patch dtype, average the temporal copies, NCHW.
    x = patches * std.to(patches.dtype) + mean.to(patches.dtype)
    x = (2.0 * x - 1.0).mean(dim=1).permute(0, 3, 1, 2)
    rows = []
    for image, (h, w) in zip(x.split([h * w for h, w in grid]), grid):
        native = image.reshape(h, w, 3, merge, patch_size, merge, patch_size)
        native = native.permute(0, 3, 1, 5, 4, 6, 2).reshape(
            -1, patch_size * patch_size * 3
        )
        rows.append(native.new_zeros((1, native.shape[1])))
        rows.append(native)
    pixels = torch.cat(rows)
    bucket = row_bucket(pixels.shape[0])
    pixels = F.pad(pixels, (0, 0, 0, bucket - pixels.shape[0]))
    meta = vision_metadata(
        tuple(grid),
        merge=merge,
        head_dim=head_dim,
        posemb_grid=posemb_grid,
        bucket=bucket,
    )
    return (pixels, *meta), sum(h * w for h, w in grid)

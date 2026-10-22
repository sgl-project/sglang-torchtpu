# SPDX-License-Identifier: Apache-2.0
"""Muse Spark NaViT on TPU, adapted from upstream PR #44666 (26305be7f410).

Patch/grid metadata stays on CPU. The device boundary takes only fixed-size
padded tensors; real patches and padding have distinct attention segments.
The checkpoint's xy positional interpolation and pixel shuffle are preserved.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.muse_spark import MuseSparkVisionConfig
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.utils import add_prefix


def _rope_frequencies(
    head_dim: int,
    grid_h: int,
    grid_w: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    if head_dim % 4:
        raise ValueError("NaViT attention head size must be divisible by four")
    half_dim = head_dim // 2
    exponents = torch.arange(0, half_dim, 2, dtype=torch.float32, device=device)
    inverse = theta ** (-exponents / half_dim)
    pos_h = torch.arange(1, grid_h + 1, dtype=torch.float32, device=device)
    pos_w = torch.arange(1, grid_w + 1, dtype=torch.float32, device=device)
    mesh_h, mesh_w = torch.meshgrid(pos_h, pos_w, indexing="ij")
    freq_h = torch.einsum("hw,d->hwd", mesh_h, inverse)
    freq_w = torch.einsum("hw,d->hwd", mesh_w, inverse)
    angles = torch.cat((freq_w, freq_h), dim=-1)
    return torch.stack((angles.cos(), angles.sin()), dim=-1)


def _patch_grid_positions(
    grid_h: int,
    grid_w: int,
    device: torch.device,
) -> torch.Tensor:
    """Return fp32 grid-sample coordinates for the checkpoint's xy (non-ij) layout."""
    pos_h = torch.linspace(
        -1 + 1 / grid_h,
        1 - 1 / grid_h,
        grid_h,
        device=device,
        dtype=torch.float32,
    )
    pos_w = torch.linspace(
        -1 + 1 / grid_w,
        1 - 1 / grid_w,
        grid_w,
        device=device,
        dtype=torch.float32,
    )
    return torch.stack(torch.meshgrid(pos_h, pos_w, indexing="xy"), dim=-1).reshape(
        -1, 2
    )


def _apply_rope(
    query: torch.Tensor,
    key: torch.Tensor,
    frequencies: torch.Tensor,
    x_shape=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cosine = frequencies[..., 0].unsqueeze(1)
    sine = frequencies[..., 1].unsqueeze(1)

    def rotate(value: torch.Tensor) -> torch.Tensor:
        pairs = value.float().reshape(*value.shape[:-1], -1, 2)
        first, second = pairs.unbind(dim=-1)
        rotated = torch.stack(
            (first * cosine - second * sine, first * sine + second * cosine),
            dim=-1,
        ).flatten(-2)
        return rotated.to(value.dtype)

    return rotate(query), rotate(key)


def _pixel_shuffle_permutation(
    offset: int, grid_h: int, grid_w: int, factor: int, device: torch.device
) -> tuple[torch.Tensor, int]:
    if grid_h % factor or grid_w % factor:
        raise ValueError(
            "preprocessed image grid is not divisible by downsample factor"
        )
    order = torch.arange(offset, offset + grid_h * grid_w, device=device)
    order = (
        order.reshape(grid_h // factor, factor, grid_w // factor, factor)
        .permute(0, 2, 1, 3)
        .reshape(-1)
    )
    return order, offset + grid_h * grid_w


def _sparse_permutation(
    offset: int,
    grid_h: int,
    grid_w: int,
    block_h: int,
    block_w: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], int]:
    padded_h = math.ceil(grid_h / block_h) * block_h
    padded_w = math.ceil(grid_w / block_w) * block_w
    order = torch.arange(offset, offset + grid_h * grid_w, device=device).reshape(
        grid_h, grid_w
    )
    order = F.pad(order, (0, padded_w - grid_w, 0, padded_h - grid_h), value=-1)
    rows, cols = padded_h // block_h, padded_w // block_w
    order = (
        order.reshape(rows, block_h, cols, block_w)
        .permute(0, 2, 1, 3)
        .reshape(rows * cols, block_h * block_w)
    )
    lengths = (order != -1).sum(dim=1).tolist()
    order = order[order != -1]
    return order, lengths, offset + grid_h * grid_w


class NaViTMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc1 = ReplicatedLinear(
            hidden_size,
            intermediate_size,
            bias=True,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = ReplicatedLinear(
            intermediate_size,
            hidden_size,
            bias=True,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value, _ = self.fc1(value)
        value, _ = self.fc2(F.gelu(value))
        return value


class NaViTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        use_rope: bool,
        prefix: str = "",
        num_dummy_heads: int = 0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("vision hidden size must be divisible by attention heads")
        self.norm1 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = VisionAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            num_dummy_heads=num_dummy_heads,
            head_dim=hidden_size // num_heads,
            projection_size=hidden_size,
            use_qkv_parallel=True,
            qkv_bias=True,
            proj_bias=True,
            softmax_scale=(hidden_size // num_heads) ** -0.5,
            flatten_batch=True,
            customized_position_embedding_applier=(_apply_rope if use_rope else None),
            # Each rank encodes whole images; see _batch_encode_per_image_misses.
            use_data_parallel=True,
            prefix=add_prefix("attn", prefix),
        )
        # Padding and attention windows are isolated only through segment_ids.
        if not getattr(self.attn.qkv_backend, "supports_segment_ids", False):
            raise ValueError(
                "Muse vision needs an mm attention backend that honors segment_ids; "
                f"{self.attn.qkv_backend_name!r} ignores them"
            )
        self.use_rope = use_rope
        self.norm2 = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = NaViTMLP(
            hidden_size,
            intermediate_size,
            add_prefix("mlp", prefix),
        )

    def forward(
        self,
        value: torch.Tensor,
        frequencies: torch.Tensor,
        metadata: torch.Tensor,
    ) -> torch.Tensor:
        attn_output = self.attn(
            self.norm1(value),
            segment_ids=metadata,
            position_embeddings=frequencies if self.use_rope else None,
        )
        value = value + attn_output.squeeze(0)
        return value + self.mlp(self.norm2(value))


class NaViTAdapter(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.fc1 = ReplicatedLinear(
            input_size,
            output_size,
            bias=False,
            prefix=add_prefix("fc1", prefix),
        )
        self.fc2 = ReplicatedLinear(
            output_size,
            output_size,
            bias=False,
            prefix=add_prefix("fc2", prefix),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value, _ = self.fc1(value)
        value, _ = self.fc2(F.gelu(value))
        return F.gelu(value)


class NaViT(nn.Module):
    """Variable-resolution image/video encoder using only public PyTorch ops."""

    def __init__(
        self,
        config: MuseSparkVisionConfig,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.patch_temporal = config.patch_temporal
        self.downsample_factor = config.merge_size

        if config.hidden_act != "gelu" or config.rope_parameters["rope_type"] not in (
            "default",
            "axial",
        ):
            raise ValueError("Muse vision supports gelu and axial RoPE only")
        if len(config.layer_types) != config.num_hidden_layers or set(
            config.layer_types
        ) - {"window_attention", "full_attention"}:
            raise ValueError("Invalid Muse vision layer_types")
        latent_size = config.hidden_size
        self.output_size = latent_size * config.merge_size**2
        self.head_dim = latent_size // config.num_attention_heads

        patch_input_size = (
            3 * config.patch_size * config.patch_size * config.patch_temporal
        )
        self.patch_embedder = nn.Module()
        self.patch_embedder.patch_embedding = ReplicatedLinear(
            patch_input_size,
            latent_size,
            bias=False,
            prefix=add_prefix("patch_embedder.patch_embedding", prefix),
        )
        self.patch_embedder.position_embedding_table = nn.Embedding(
            config.pos_emb_height * config.pos_emb_width, latent_size
        )
        nn.init.normal_(
            self.patch_embedder.position_embedding_table.weight, std=latent_size**-0.5
        )
        self.ln_pre = nn.LayerNorm(latent_size, eps=config.layer_norm_eps)
        self.ln_post = nn.LayerNorm(latent_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList(
            NaViTBlock(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                True,
                add_prefix(f"layers.{i}", prefix),
                num_dummy_heads=getattr(config, "num_dummy_heads", 0),
                norm_eps=config.layer_norm_eps,
            )
            for i in range(config.num_hidden_layers)
        )

    def forward(
        self,
        patches,
        pos_indices,
        pos_weights,
        frequencies,
        full_segments,
        window_segments,
        output_order,
    ):
        value, _ = self.patch_embedder.patch_embedding(patches)
        # Equivalent to grid_sample(..., align_corners=False, padding_mode="zeros")
        # without relying on a grid_sample lowering for PrivateUse1.
        table = self.patch_embedder.position_embedding_table.weight.float()
        positional = (table[pos_indices.long()] * pos_weights[..., None]).sum(1)
        value = self.ln_pre(value + positional.to(value.dtype))
        for kind, layer in zip(self.config.layer_types, self.layers):
            segments = full_segments if kind == "full_attention" else window_segments
            value = layer(value, frequencies, segments)
        value = self.ln_post(value)[output_order.long()]
        factor_squared = self.downsample_factor**2
        return (
            value.reshape(-1, factor_squared, value.shape[-1])
            .transpose(1, 2)
            .flatten(1)
        )


# Raw-patch encoder shapes; serving picks only buckets that warmup compiled.
PATCH_BUCKETS = (1024, 4096, 8192, 16384)


def patch_bucket(count):
    for bucket in PATCH_BUCKETS:
        if count <= bucket:
            return bucket
    raise ValueError(
        f"Muse image has {count} raw patches; maximum is {PATCH_BUCKETS[-1]}"
    )


@functools.lru_cache(maxsize=64)
def vision_metadata(grid_h, grid_w, head_dim, pos_h, pos_w, theta, factor, bucket):
    """CPU-only static metadata, including zero-padded bilinear interpolation."""
    count = grid_h * grid_w
    if bucket < count or bucket % (factor**2):
        raise ValueError("invalid Muse patch bucket")
    device = torch.device("cpu")
    order, lengths, _ = _sparse_permutation(0, grid_h, grid_w, pos_h, pos_w, device)
    coords = _patch_grid_positions(grid_h, grid_w, device)
    x = (coords[:, 0] + 1) * pos_w / 2 - 0.5
    y = (coords[:, 1] + 1) * pos_h / 2 - 0.5
    x0, y0 = x.floor().long(), y.floor().long()
    indices, weights = [], []
    for dy, dx in ((0, 0), (0, 1), (1, 0), (1, 1)):
        ix, iy = x0 + dx, y0 + dy
        valid = (ix >= 0) & (ix < pos_w) & (iy >= 0) & (iy < pos_h)
        weight = (1 - (x - ix).abs()) * (1 - (y - iy).abs()) * valid
        indices.append(iy.clamp(0, pos_h - 1) * pos_w + ix.clamp(0, pos_w - 1))
        weights.append(weight)
    pos_indices = F.pad(torch.stack(indices, 1)[order], (0, 0, 0, bucket - count))
    pos_weights = F.pad(torch.stack(weights, 1)[order], (0, 0, 0, bucket - count))
    frequencies = _rope_frequencies(head_dim, grid_h, grid_w, theta, device).reshape(
        count, head_dim // 2, 2
    )[order]
    frequencies = F.pad(frequencies, (0, 0, 0, 0, 0, bucket - count))
    full = F.pad(
        torch.zeros(count, dtype=torch.int32), (0, bucket - count), value=1
    ).unsqueeze(0)
    window = torch.repeat_interleave(
        torch.arange(len(lengths), dtype=torch.int32), torch.tensor(lengths)
    )
    window = F.pad(window, (0, bucket - count), value=len(lengths)).unsqueeze(0)
    shuffled, _ = _pixel_shuffle_permutation(0, grid_h, grid_w, factor, device)
    output_order = F.pad(order.argsort()[shuffled], (0, bucket - count))
    return order, pos_indices, pos_weights, frequencies, full, window, output_order


def prepare_vision_input(feature, config):
    """Patchify on CPU before transferring a fixed bucket to the encoder."""
    if feature.device.type != "cpu":
        raise ValueError("Muse TPU preprocessing must retain CPU image features")
    if feature.ndim == 3:
        feature = feature.unsqueeze(0).expand(config.patch_temporal, -1, -1, -1)
    if feature.ndim != 4 or feature.shape[0] != config.patch_temporal:
        raise ValueError("expected one image or a patch_temporal video clip")
    frames, channels, h, w = feature.shape
    p = config.patch_size
    if h % (p * config.merge_size) or w % (p * config.merge_size):
        raise ValueError("Muse image size must align to the merged patch size")
    gh, gw = h // p, w // p
    n, bucket = gh * gw, patch_bucket(gh * gw)
    meta = vision_metadata(
        gh,
        gw,
        config.hidden_size // config.num_attention_heads,
        config.pos_emb_height,
        config.pos_emb_width,
        config.rope_parameters["rope_theta"],
        config.merge_size,
        bucket,
    )
    patches = (
        feature.reshape(frames, channels, gh, p, gw, p)
        .permute(2, 4, 0, 1, 3, 5)
        .reshape(n, -1)
    )
    patches = F.pad(patches[meta[0]], (0, 0, 0, bucket - n))
    return (patches, *meta[1:]), n // config.merge_size**2

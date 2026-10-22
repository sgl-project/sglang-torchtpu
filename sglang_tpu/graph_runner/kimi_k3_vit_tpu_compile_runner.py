"""Kimi-K3 vision encoder runner + policy for TPU ``torch.compile``.

Kimi-K3's tower is MoonViT-shaped (interpolated position embedding, complex
2-D RoPE, a 2x2 temporal-pool merger, then the projector), but run eagerly
every stage takes a per-image-grid shape: a new grid compiled the patch
embedding, the position-embedding interpolation, the RoPE lookup, every
encoder layer, the merger and the projector while serving.

Like Qwen3-VL's ``ViTTpuCompileRunner``, everything grid-dependent is staged on
the host, one image at a time, at a fixed patch bucket:

* pixels, reordered so every 2x2 merge window is 4 consecutive rows;
* 16 taps and weights of each patch's position embedding, matching the
  tower's ``F.interpolate`` mode: bilinear (the checkpoint's) uses 4 of them,
  bicubic (``kimi_k25._bicubic_interp_axis``) all 16;
* the RoPE cos / sin of each patch's (row, column);
* segment ids, padding in a trailing segment of its own.

The compiled boundary (``KimiK3VitTpuCompileRunner.forward``) then runs the
patch embedding (the stride-14 conv as a matmul), the position-embedding
gather, the encoder layers (segment-id Pallas attention, real-arithmetic RoPE),
the merge (a reshape, thanks to the window order) and the projector. Its
output is unpadded on the host, where the TPU embed merge wants it anyway.

Video grids (t > 1), images past the largest bucket and towers with another
interpolation mode keep the tower's own eager path.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import multimodal_encoder_runs_here
from sglang_tpu.graph_runner.vit_tpu_compile_runner import (
    VisionTpuCompilePolicy,
    _compute_patch_counts,
    _validate_patch_buckets,
)

logger = logging.getLogger(__name__)

# The merger pools 2x2 windows, so every image covers a multiple of 4 patches;
# square buckets with an even side keep that.
_MERGE_SIZE = 2
_MERGE_AREA = _MERGE_SIZE * _MERGE_SIZE


def patch_buckets() -> list[int]:
    """The patch-count ladder: even-sided squares, bounded by the ViT envs."""
    if envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.is_set():
        # Set but empty => () => warms nothing, i.e. precompile disabled.
        buckets = sorted({int(x) for x in envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.get()})
        return _validate_patch_buckets(buckets, _MERGE_SIZE)
    return _compute_patch_counts(
        _MERGE_SIZE, max_patches=envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.get()
    )


def rotate_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Real-arithmetic form of Kimi-K3's complex ``apply_rope``.

    Adjacent pairs (x[2i], x[2i+1]) rotate by angle i, exactly as
    ``view_as_real(view_as_complex(x) * (cos + i*sin))``.

    Args:
        x: ``[S, num_heads, head_dim]``.
        cos, sin: ``[S, head_dim/2]``.
    """
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    pairs = x.float().view(*x.shape[:-1], -1, 2)
    real, imag = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
    return rotated.flatten(-2).type_as(x)


def _bilinear_interp_axis(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray]:
    """1D taps + weights of ``F.interpolate(mode="bilinear", align_corners=False)``.

    Each output row blends the two source rows around its half-pixel source
    coordinate, which aten clamps at 0 for linear modes; at the far edge both
    taps are the last row. Two zero-weight slots give
    ``kimi_k25._bicubic_interp_axis``'s ``[out_size, 4]`` layout, so both modes
    stage the same 16 (h, w) taps.
    """
    scale = in_size / out_size
    src = np.maximum((np.arange(out_size, dtype=np.float64) + 0.5) * scale - 0.5, 0)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, in_size - 1)
    frac = src - lo
    taps = np.pad(np.stack([lo, hi], axis=1), ((0, 0), (0, 2)))
    coeffs = np.pad(np.stack([1.0 - frac, frac], axis=1), ((0, 0), (0, 2)))
    return taps, coeffs


def _axis_taps_for(
    mode: str,
) -> Optional[Callable[[int, int], tuple[np.ndarray, np.ndarray]]]:
    """The per-axis taps of the tower's position-embedding interpolation mode,
    or None for a mode the runner does not reproduce."""
    if mode == "bilinear":
        return _bilinear_interp_axis
    if mode == "bicubic":
        from sglang.srt.models.kimi_k25 import _bicubic_interp_axis

        return _bicubic_interp_axis
    return None


def merge_window_order(h: int, w: int) -> torch.Tensor:
    """Row-major patch indices of an ``h x w`` grid, regrouped so each 2x2
    merge window is 4 consecutive entries, windows in row-major order --
    the order ``tpool_patch_merger`` emits for a single frame."""
    rows = torch.arange(h).view(h // _MERGE_SIZE, _MERGE_SIZE, 1, 1)
    cols = torch.arange(w).view(1, 1, w // _MERGE_SIZE, _MERGE_SIZE)
    index = rows * w + cols  # [h/2, 2, w/2, 2]
    return index.permute(0, 2, 1, 3).reshape(-1)


@dataclass(frozen=True)
class StagedImage:
    """One image's compiled-boundary inputs on the host, at its bucket."""

    pixels: torch.Tensor  # [B, 3 * p * p]
    pos_idx: torch.Tensor  # [B, 16] int32 into the flattened pos-embed grid
    pos_weight: torch.Tensor  # [B, 16] float32 interpolation weights (0 on padding)
    cos: torch.Tensor  # [B, head_dim/2]
    sin: torch.Tensor  # [B, head_dim/2]
    segment_ids: torch.Tensor  # [1, B] int32
    real_patches: int

    def args(self, device: torch.device, dtype: torch.dtype) -> tuple:
        return (
            self.pixels.to(device=device, dtype=dtype),
            self.pos_idx.to(device=device),
            self.pos_weight.to(device=device),
            self.cos.to(device=device),
            self.sin.to(device=device),
            self.segment_ids.to(device=device),
        )


class KimiK3VitTpuCompileRunner:
    """Runs Kimi-K3's vision tower + projector over one staged patch bucket.

    Attached to the encoder by ``sglang_tpu.hooks.kimi_k3_vision``, which also
    binds the tower and the projector. Eager until
    ``KimiK3VisionTpuCompilePolicy.install`` compiles ``forward``; both use
    the same math, so a compiled server serves what an eager one does.
    """

    def __init__(self, encoder: Any, attention_cls: Any = None) -> None:
        self.encoder = encoder
        self.owner = encoder
        self.tower = None
        self.projector = None
        if attention_cls is None:
            from sglang.srt.layers.attention.vision import VisionTpuAttention

            attention_cls = VisionTpuAttention
        # The layers hold no attention module of their own on TPU (their eager
        # path calls SDPA per segment), and the segment-id attention has no
        # weights, so one instance per layer lives here.
        self.attention = [
            attention_cls(
                head_dim=block.hidden_size_per_attention_head,
                num_heads=block.num_heads,
                num_kv_heads=block.num_heads,
            )
            for block in encoder.blocks
        ]
        self._rope_freqs: Optional[torch.Tensor] = None
        self._patch_buckets: Optional[list[int]] = None

    def bind(self, tower: Any, projector: Any) -> None:
        self.tower = tower
        self.projector = projector

    @property
    def ready(self) -> bool:
        return (
            self.tower is not None
            and self.projector is not None
            and tuple(self.tower.merge_kernel_size) == (_MERGE_SIZE, _MERGE_SIZE)
            # Any other mode keeps the tower's own eager interpolation.
            and _axis_taps_for(self.tower.patch_embed.pos_emb.interpolation_mode)
            is not None
            and all(
                getattr(attention, "supports_segment_ids", False)
                for attention in self.attention
            )
        )

    # --- compiled boundary --------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """The encoder layers + ``final_layernorm`` over ``[S, D]`` patches."""
        seq_len = x.shape[0]
        for block, attention in zip(self.encoder.blocks, self.attention):
            residual = x
            hidden = block.norm0(x)
            qkv = block.wqkv(hidden).view(
                seq_len, 3, block.num_heads, block.hidden_size_per_attention_head
            )
            q, k, v = torch.unbind(qkv, dim=1)
            q = rotate_interleaved(q, cos, sin)
            k = rotate_interleaved(k, cos, sin)
            attn = attention(q, k, v, 1, segment_ids=segment_ids)
            x = residual + block.wo(attn.flatten(start_dim=-2))
            x = x + block.mlp(block.norm1(x))
        return self.encoder.final_layernorm(x)

    def forward(
        self,
        pixels: torch.Tensor,
        pos_idx: torch.Tensor,
        pos_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One staged image: ``[B, 3*p*p]`` pixels -> ``[B/4, text_hidden]``."""
        patch_embed = self.tower.patch_embed
        proj = patch_embed.proj
        x = F.linear(pixels, proj.weight.flatten(1), proj.bias)
        table = patch_embed.pos_emb.weight.flatten(0, 1)
        pos = (table[pos_idx].float() * pos_weight.unsqueeze(-1)).sum(dim=1)
        x = x + pos.to(x.dtype)
        x = self.encode(x, cos, sin, segment_ids)
        # Window order makes the 2x2 merge a reshape.
        merged = x.reshape(x.shape[0] // _MERGE_AREA, _MERGE_AREA * x.shape[1])
        return self.projector(merged)

    # --- host staging -------------------------------------------------------

    def _host_rope_freqs(self) -> torch.Tensor:
        rope = self.encoder.rope_2d
        if self._rope_freqs is None:
            dim_range = torch.arange(0, rope.dim, 4)[: rope.dim // 4].float()
            self._rope_freqs = 1.0 / (rope.theta_base ** (dim_range / rope.dim))
        return self._rope_freqs

    def stage(self, pixels: torch.Tensor, h: int, w: int, bucket: int) -> StagedImage:
        """Host inputs of one single-frame ``h x w`` image padded to ``bucket``."""
        real = h * w
        if real > bucket or h % _MERGE_SIZE or w % _MERGE_SIZE:
            raise ValueError(f"cannot stage a {h}x{w} grid into a {bucket} bucket")
        order = merge_window_order(h, w)
        rows, cols = order // w, order % w

        staged = pixels.reshape(real, -1)[order]
        pad = bucket - real
        staged = F.pad(staged, (0, 0, 0, pad))

        pos_emb = self.tower.patch_embed.pos_emb
        src_h, src_w = pos_emb.weight.shape[:2]
        axis_taps = _axis_taps_for(pos_emb.interpolation_mode)
        taps_h, coeffs_h = axis_taps(src_h, h)
        taps_w, coeffs_w = axis_taps(src_w, w)
        taps_h = torch.from_numpy(taps_h)[rows]  # [real, 4]
        taps_w = torch.from_numpy(taps_w)[cols]
        coeffs_h = torch.from_numpy(coeffs_h)[rows]
        coeffs_w = torch.from_numpy(coeffs_w)[cols]
        pos_idx = (taps_h[:, :, None] * src_w + taps_w[:, None, :]).reshape(real, 16)
        pos_weight = (coeffs_h[:, :, None] * coeffs_w[:, None, :]).reshape(real, 16)
        pos_idx = F.pad(pos_idx, (0, 0, 0, pad)).to(torch.int32)
        pos_weight = F.pad(pos_weight, (0, 0, 0, pad)).to(torch.float32)

        freqs = self._host_rope_freqs()
        # Interleaved (x, y) angle pairs, as Rope2DPosEmbRepeated lays them out.
        angles = torch.stack(
            (torch.outer(cols.float(), freqs), torch.outer(rows.float(), freqs)),
            dim=-1,
        ).reshape(real, -1)
        cos = F.pad(torch.cos(angles), (0, 0, 0, pad))
        sin = F.pad(torch.sin(angles), (0, 0, 0, pad))

        segment_ids = torch.zeros(1, bucket, dtype=torch.int32)
        segment_ids[0, real:] = 1
        return StagedImage(staged, pos_idx, pos_weight, cos, sin, segment_ids, real)

    def _bucket_for(self, patches: int) -> Optional[int]:
        if self._patch_buckets is None:
            self._patch_buckets = patch_buckets()
        return next((b for b in self._patch_buckets if b >= patches), None)

    def run_images(
        self, pixels: torch.Tensor, shapes: Sequence[tuple[int, int, int]]
    ) -> Optional[list[torch.Tensor]]:
        """Projected features of every image (host tensors), or None when some
        image needs the tower's eager path (video, or past the ladder)."""
        return self._run_staged(pixels, shapes, self._bucket_for, self.forward)

    def _run_staged(
        self, pixels, shapes, bucket_for, fn
    ) -> Optional[list[torch.Tensor]]:
        buckets = []
        for t, h, w in shapes:
            bucket = bucket_for(h * w) if t == 1 else None
            if bucket is None:
                return None
            buckets.append(bucket)
        param = self.tower.patch_embed.proj.weight
        outputs = []
        start = 0
        with torch.no_grad():
            for (_, h, w), bucket in zip(shapes, buckets):
                staged = self.stage(pixels[start : start + h * w], h, w, bucket)
                start += h * w
                result = fn(*staged.args(param.device, param.dtype))
                # Unpad on the host: device slices would compile per length.
                outputs.append(result.cpu()[: staged.real_patches // _MERGE_AREA])
        return outputs


class KimiK3VisionTpuCompilePolicy(VisionTpuCompilePolicy):
    """Compiles ``KimiK3VitTpuCompileRunner.forward`` once per patch bucket."""

    def __init__(
        self,
        encoder: Any | None = None,
        *,
        name: str = "kimi_k3_vit_tower",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder, name=name, **kwargs)
        self._warned_oversize: set[int] = set()

    def _runner_ready_for_compile(self, runner: Any) -> bool:
        # PD decode, language-only and LM-only servers never forward the tower,
        # nor does one with multimodal input off; they skip the bucket ladder.
        return (
            self.model_runner.model_config.is_multimodal
            and multimodal_encoder_runs_here()
            and bool(getattr(runner, "ready", False))
        )

    def compute_patch_buckets(self) -> list[int]:
        return patch_buckets()

    def make_warmup_inputs(self, patch_count: int) -> tuple[torch.Tensor, ...]:
        runner = self._resolve_runner()
        grid = math.isqrt(patch_count)
        proj = runner.tower.patch_embed.proj
        pixels = torch.zeros(patch_count, proj.weight[0].numel())
        staged = runner.stage(pixels, grid, grid, patch_count)
        return staged.args(proj.weight.device, proj.weight.dtype)

    def run_images(
        self, pixels: torch.Tensor, shapes: Sequence[tuple[int, int, int]]
    ) -> Optional[list[torch.Tensor]]:
        runner = self._resolve_runner()

        def bucket_for(patches: int) -> Optional[int]:
            bucket = self._bucket_for(patches)
            if bucket is None and patches not in self._warned_oversize:
                self._warned_oversize.add(patches)
                buckets = self.resolve_patch_buckets()
                logger.warning(
                    "KimiK3VisionTpuCompilePolicy: %d patches exceed the largest "
                    "precompiled bucket %s; the tower runs this call eagerly. Add "
                    "a bucket to SGLANG_TPU_VIT_PRECOMPILE_SIZES or raise "
                    "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.",
                    patches,
                    buckets[-1] if buckets else None,
                )
            return bucket

        return runner._run_staged(pixels, shapes, bucket_for, self.compiled_fn)

    def _install_run_adapter(self, runner: Any) -> None:
        runner.run_images = self.run_images

    # The staged path does its own padding and unpadding (run_images); the
    # base class's run() is not used.
    def prepare_call(self, *raw: Any):
        raise NotImplementedError("use run_images")

    def unpad_outputs(self, result: Any, ctx: Any):
        raise NotImplementedError("use run_images")

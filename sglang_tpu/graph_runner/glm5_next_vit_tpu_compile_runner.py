# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3-Flash vision encoder runner + policy for TPU ``torch.compile``.

``Glm5NextVisionModel`` inherits its forward pass from ``GlmOcrVisionModel``
(and ``rot_pos_emb`` from ``Glm4vVisionModel``): without a TPU compile runner,
every new image grid runs ``rot_pos_emb``, ``cu_seqlens``, the 24 vision
blocks, the 2x2 ``Conv2dLayer`` downsampler and the SwiGLU patch merger
eagerly across variable patch counts.

Like ``KimiK3VitTpuCompileRunner``, all grid-dependent inputs are staged on the
host, one frame at a time, padded to a fixed square patch bucket aligned to the
2x2 spatial merger:

* patchified pixels padded to ``[B, C * T * P * P]``;
* 2D RoPE ``cos`` / ``sin`` tables ``[B, head_dim]`` indexed in 2x2 merge-window
  order on the host;
* ``segment_ids`` ``[1, B]`` placing padded patches in segment 1 so they never
  attend to real patches in segment 0.

The compiled boundary (``Glm5NextVitTpuCompileRunner.forward``) runs
``patch_embed``, the 24 ``Glm5NextVisionBlock`` layers (passing ``segment_ids``
directly to ``VisionAttention``), ``post_layernorm``, ``downsample`` and
``merger``, then unpads ``[: real_patches // 4]`` on the host.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs as sglang_envs
from sglang.srt.layers.dp_attention import multimodal_encoder_runs_here
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.graph_runner.vit_tpu_compile_runner import (
    VisionTpuCompilePolicy,
    _compute_patch_counts,
    _validate_patch_buckets,
)

logger = logging.getLogger(__name__)

_MERGE_SIZE = 2
_MERGE_AREA = _MERGE_SIZE * _MERGE_SIZE


def patch_buckets(merge_size: int = _MERGE_SIZE) -> list[int]:
    """Square patch-count bucket ladder aligned to ``merge_size``."""
    if tpu_envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.is_set():
        buckets = sorted(
            {int(x) for x in tpu_envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.get()}
        )
        return _validate_patch_buckets(buckets, merge_size)
    return _compute_patch_counts(
        merge_size,
        max_patches=tpu_envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.get(),
    )


def _rot_pos_ids(h: int, w: int, merge_size: int = _MERGE_SIZE) -> torch.Tensor:
    """2D ``(h, w)`` RoPE position indices in ``merge_size x merge_size`` window order."""
    hpos_ids = (
        torch.arange(h, dtype=torch.long)
        .unsqueeze(1)
        .expand(-1, w)
        .reshape(h // merge_size, merge_size, w // merge_size, merge_size)
        .permute(0, 2, 1, 3)
        .flatten()
    )
    wpos_ids = (
        torch.arange(w, dtype=torch.long)
        .unsqueeze(0)
        .expand(h, -1)
        .reshape(h // merge_size, merge_size, w // merge_size, merge_size)
        .permute(0, 2, 1, 3)
        .flatten()
    )
    return torch.stack([hpos_ids, wpos_ids], dim=-1)


def _expand_frames(grid_thw: Any) -> list[tuple[int, int]]:
    """Expand ``[N, 3]`` ``(t, h, w)`` rows into ``t`` single-frame ``(h, w)`` tuples."""
    rows = grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else list(grid_thw)
    frames: list[tuple[int, int]] = []
    for t, h, w in rows:
        frames.extend([(int(h), int(w))] * int(t))
    return frames


@dataclass(frozen=True)
class StagedFrame:
    """One frame's compiled-boundary inputs on the host, padded to ``bucket``."""

    pixels: torch.Tensor  # [B, C * T * P * P]
    rotary_pos_emb_cos: torch.Tensor  # [B, head_dim]
    rotary_pos_emb_sin: torch.Tensor  # [B, head_dim]
    segment_ids: torch.Tensor  # [1, B] int32
    real_patches: int

    def args(self, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, ...]:
        return (
            self.pixels.to(device=device, dtype=dtype),
            self.rotary_pos_emb_cos.to(device=device),
            self.rotary_pos_emb_sin.to(device=device),
            self.segment_ids.to(device=device),
        )


class Glm5NextVitTpuCompileRunner:
    """Runs ``Glm5NextVisionModel`` over one host-staged patch bucket per frame.

    Attached as ``visual.graph_runners`` by ``sglang_tpu.hooks.glm5_next_vision``.
    Eager until ``Glm5NextVisionTpuCompilePolicy.install`` compiles ``forward``;
    both paths execute the exact same math.
    """

    def __init__(self, vit: Any) -> None:
        self.vit = vit
        self.owner = vit
        self._rotary_cos_sin_cache_cpu: Optional[torch.Tensor] = None
        self._patch_buckets: Optional[list[int]] = None

    @property
    def ready(self) -> bool:
        blocks = getattr(self.vit, "blocks", ())
        return (
            len(blocks) > 0
            and getattr(self.vit, "spatial_merge_size", None) == _MERGE_SIZE
            and all(
                getattr(
                    getattr(getattr(block, "attn", None), "qkv_backend", None),
                    "supports_segment_ids",
                    False,
                )
                for block in blocks
            )
        )

    # --- compiled boundary --------------------------------------------------

    def encode(
        self,
        x: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """The 24 ``Glm5NextVisionBlock`` layers + ``post_layernorm`` over ``[S, 1, H]``."""
        seq_len, bsz, hidden = x.shape
        for block in self.vit.blocks:
            x2d = x.reshape(-1, hidden)
            hidden_states = block.norm1(x2d).reshape(seq_len, bsz, hidden)
            hidden_states = hidden_states.transpose(0, 1)
            attn = block.attn(
                hidden_states,
                cu_seqlens=None,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                segment_ids=segment_ids,
            )
            attn = attn.transpose(0, 1)
            attn2d = attn.reshape(-1, hidden)
            x_norm_2d, x_after_add_2d = block.norm2(x2d, residual=attn2d)
            x_norm = x_norm_2d.reshape(seq_len, bsz, hidden)
            x_after_add = x_after_add_2d.reshape(seq_len, bsz, hidden)
            mlp_out = block.mlp(x_norm)
            x = x_after_add + mlp_out
        return self.vit.post_layernorm(x)

    def forward(
        self,
        pixels: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """One staged frame: ``[B, C*T*P*P]`` pixels -> ``[B/4, out_hidden_size]``."""
        vit = self.vit
        x = vit.patch_embed(pixels).unsqueeze(1)
        x = self.encode(x, rotary_pos_emb_cos, rotary_pos_emb_sin, segment_ids)
        merge = vit.spatial_merge_size
        x = x.view(-1, merge, merge, x.shape[-1]).permute(0, 3, 1, 2)
        x = vit.downsample(x).view(-1, vit.out_hidden_size)
        return vit.merger(x)

    # --- host staging -------------------------------------------------------

    def _host_rotary_cache(self) -> torch.Tensor:
        if self._rotary_cos_sin_cache_cpu is None:
            self._rotary_cos_sin_cache_cpu = (
                self.vit.rotary_pos_emb.cos_sin_cache.detach().to("cpu", copy=True)
            )
        return self._rotary_cos_sin_cache_cpu

    def stage(
        self, pixels: torch.Tensor, h: int, w: int, bucket: int
    ) -> StagedFrame:
        """Host inputs for one ``h x w`` frame padded to ``bucket`` patches."""
        merge_size = self.vit.spatial_merge_size
        real = h * w
        if h <= 0 or w <= 0 or real > bucket or h % merge_size or w % merge_size:
            raise ValueError(f"cannot stage a {h}x{w} grid into a {bucket} bucket")
        pad = bucket - real
        staged_pixels = F.pad(pixels.reshape(real, -1), (0, 0, 0, pad))

        pos_ids = _rot_pos_ids(h, w, merge_size)
        padded_pos_ids = pos_ids.new_zeros((bucket, 2))
        padded_pos_ids[:real] = pos_ids

        max_grid = max(h, w)
        cos_table, sin_table = self._host_rotary_cache()[:max_grid].chunk(2, dim=-1)
        cos = cos_table[padded_pos_ids].flatten(1)
        sin = sin_table[padded_pos_ids].flatten(1)
        cos = torch.cat([cos, cos], dim=-1)
        sin = torch.cat([sin, sin], dim=-1)

        segment_ids = torch.zeros(1, bucket, dtype=torch.int32)
        segment_ids[0, real:] = 1
        return StagedFrame(staged_pixels, cos, sin, segment_ids, real)

    def _bucket_for(self, patches: int) -> Optional[int]:
        if self._patch_buckets is None:
            self._patch_buckets = patch_buckets(self.vit.spatial_merge_size)
        return next((b for b in self._patch_buckets if b >= patches), None)

    def run_grid(
        self, pixels: torch.Tensor, grid_thw: Any
    ) -> Optional[torch.Tensor]:
        """Merged features for ``grid_thw``, or ``None`` if any frame exceeds the ladder."""
        return self._run_staged(pixels, grid_thw, self._bucket_for, self.forward)

    def _run_staged(
        self,
        pixels: torch.Tensor,
        grid_thw: Any,
        bucket_for: Callable[[int], Optional[int]],
        fn: Callable[..., torch.Tensor],
    ) -> Optional[torch.Tensor]:
        merge_size = self.vit.spatial_merge_size
        frames = _expand_frames(grid_thw)
        buckets: list[int] = []
        for h, w in frames:
            if h <= 0 or w <= 0 or h % merge_size or w % merge_size:
                return None
            bucket = bucket_for(h * w)
            if bucket is None:
                return None
            buckets.append(bucket)

        device = self.vit.device
        dtype = self.vit.dtype
        return_on_device = (
            getattr(self.vit, "use_data_parallel", False)
            or sglang_envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.get()
        )
        if not frames:
            target = device if return_on_device else torch.device("cpu")
            return torch.empty(
                (0, self.vit.out_hidden_size), device=target, dtype=dtype
            )

        pixels_cpu = pixels.cpu() if pixels.device.type != "cpu" else pixels
        outputs: list[torch.Tensor] = []
        start = 0
        with torch.no_grad():
            for (h, w), bucket in zip(frames, buckets):
                patches = h * w
                staged = self.stage(
                    pixels_cpu[start : start + patches], h, w, bucket
                )
                start += patches
                result = fn(*staged.args(device, dtype))
                # Unpad on the host so variable post-merge lengths never compile on TPU.
                outputs.append(result.cpu()[: staged.real_patches // _MERGE_AREA])

        merged = torch.cat(outputs, dim=0)
        if return_on_device:
            return merged.to(device)
        return merged


class Glm5NextVisionTpuCompilePolicy(VisionTpuCompilePolicy):
    """Compiles ``Glm5NextVitTpuCompileRunner.forward`` once per patch bucket."""

    spmd_safe = True

    def __init__(
        self,
        vit: Any | None = None,
        *,
        name: str = "glm5_next_vit_encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(vit, name=name, **kwargs)
        self._warned_oversize: set[int] = set()

    def _runner_ready_for_compile(self, runner: Any) -> bool:
        is_multimodal = (
            True
            if self.model_runner is None
            else bool(self.model_runner.model_config.is_multimodal)
        )
        return (
            is_multimodal
            and multimodal_encoder_runs_here()
            and bool(getattr(runner, "ready", False))
        )

    def get_compile_model(self) -> torch.nn.Module | None:
        return self.vit if isinstance(self.vit, torch.nn.Module) else None

    def compute_patch_buckets(self) -> list[int]:
        merge_size = getattr(self.vit, "spatial_merge_size", _MERGE_SIZE)
        return patch_buckets(merge_size)

    def make_warmup_inputs(self, patch_count: int) -> tuple[torch.Tensor, ...]:
        runner = self._resolve_runner()
        vit = runner.vit
        grid = math.isqrt(patch_count)
        patch_embed = vit.patch_embed
        in_features = (
            patch_embed.in_channels
            * patch_embed.temporal_patch_size
            * patch_embed.patch_size**2
        )
        pixels = torch.zeros(patch_count, in_features, dtype=vit.dtype)
        staged = runner.stage(pixels, grid, grid, patch_count)
        return staged.args(vit.device, vit.dtype)

    def run_grid(
        self, pixels: torch.Tensor, grid_thw: Any
    ) -> Optional[torch.Tensor]:
        runner = self._resolve_runner()

        def bucket_for(patches: int) -> Optional[int]:
            bucket = self._bucket_for(patches)
            if bucket is None and patches not in self._warned_oversize:
                self._warned_oversize.add(patches)
                buckets = self.resolve_patch_buckets()
                logger.warning(
                    "Glm5NextVisionTpuCompilePolicy: %d patches exceed the largest "
                    "precompiled bucket %s; the vision encoder runs this call "
                    "eagerly. Add a bucket to SGLANG_TPU_VIT_PRECOMPILE_SIZES or "
                    "raise SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.",
                    patches,
                    buckets[-1] if buckets else None,
                )
            return bucket

        return runner._run_staged(pixels, grid_thw, bucket_for, self.compiled_fn)

    def _install_run_adapter(self, runner: Any) -> None:
        runner.run_grid = self.run_grid

    def prepare_call(self, *raw: Any):
        raise NotImplementedError("use run_grid")

    def unpad_outputs(self, result: Any, ctx: Any):
        raise NotImplementedError("use run_grid")

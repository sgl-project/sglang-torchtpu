# SPDX-License-Identifier: Apache-2.0
"""Gemma4 vision encoder runner + policy for TPU ``torch.compile``.

Gemma4 gets its own runner rather than reusing ``ViTTpuCompileRunner``: patches
arrive batched with per-patch ``(x, y)`` ids instead of packed behind
``cu_seqlens``, and it pools by ``pooling_kernel_size`` instead of a spatial
merger. The bucket/padding lifecycle still comes from ``VisionTpuCompilePolicy``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from msgspec import Struct

from sglang.srt.environ import envs
from sglang_tpu.multimodal.vit_tpu_compile_runner import (
    VisionPaddingContext,
    VisionTpuCompilePolicy,
)

logger = logging.getLogger(__name__)

# Rungs this encoder warms when the env override is unset. Each costs one
# compile; the gap between them is the padding a small image pays. A guess,
# not a measurement.
_NUM_PATCH_BUCKETS = 6


class Gemma4VisionRunContext(Struct, frozen=True):
    """Threaded from ``prepare_call`` to ``unpad_outputs``."""

    padding: VisionPaddingContext


class Gemma4VisionTpuCompileRunner:
    """Eager until the policy installs the compiled boundary over ``forward``."""

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self.owner = encoder

    @property
    def pooling_window(self) -> int:
        """Patches per pooled token; every bucket must be a whole number of these."""
        return self.encoder.pooling_kernel_size**2

    def forward(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [batch, patches_bucket, patch_pixels] and [batch, patches_bucket, 2],
        # position id -1 marking padding; out is (pooled, pooler_mask).
        return self.encoder.forward(pixel_values, pixel_position_ids)

    def run(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Entry point used until the policy installs its padded ``run``."""
        return self.forward(pixel_values, pixel_position_ids)


class Gemma4VisionTpuCompilePolicy(VisionTpuCompilePolicy):
    """Patch-count ladder and padded call seam for the Gemma4 vision encoder."""

    def __init__(
        self,
        encoder: Any | None = None,
        *,
        name: str = "gemma4_vision_encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder, name=name, **kwargs)
        self._warned_oversize: set[int] = set()

    def _runner_ready_for_compile(self, runner: Any) -> bool:
        return isinstance(runner, Gemma4VisionTpuCompileRunner)

    def _install_run_adapter(self, runner: Any) -> None:
        def _run(
            pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.run(pixel_values, pixel_position_ids)

        runner.run = _run

    def _pooling_window(self) -> int:
        return self._resolve_runner().pooling_window

    def compute_patch_buckets(self) -> list[int]:
        window = self._pooling_window()
        if envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.is_set():
            # Set but empty => () => warms nothing, i.e. precompile disabled.
            buckets = sorted(
                {int(x) for x in envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.get()}
            )
            for bucket in buckets:
                if bucket <= 0 or bucket % window:
                    raise ValueError(
                        "SGLANG_TPU_VIT_PRECOMPILE_SIZES entries must be positive "
                        f"multiples of the pooling window {window}; got {bucket}."
                    )
            return buckets
        max_patches = (
            envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.get() // window * window
        )
        num_buckets = envs.SGLANG_TPU_VIT_NUM_PRECOMPILE_BUCKETS.get()
        if num_buckets is None:
            num_buckets = _NUM_PATCH_BUCKETS
        elif num_buckets <= 0:
            raise ValueError(
                "SGLANG_TPU_VIT_NUM_PRECOMPILE_BUCKETS must be positive; got "
                f"{num_buckets}. Set SGLANG_TPU_VIT_PRECOMPILE_SIZES to an empty "
                "value to disable precompile instead."
            )
        step = max(window, max_patches // num_buckets // window * window)
        buckets = list(range(step, max_patches + 1, step))
        if buckets and buckets[-1] != max_patches:
            buckets.append(max_patches)
        return buckets

    def make_warmup_inputs(self, patch_count: int) -> tuple[torch.Tensor, ...]:
        encoder = self._resolve_runner().encoder
        param = next(encoder.parameters())
        patch_pixels = encoder.patch_embedder.input_proj.weight.shape[-1]
        pixel_values = torch.zeros(
            1, patch_count, patch_pixels, device=param.device, dtype=param.dtype
        )
        # Exercise the padding path: mark the tail as padding, as a real call would.
        position_ids = torch.zeros(
            1, patch_count, 2, device=param.device, dtype=torch.int32
        )
        if patch_count > 1:
            position_ids[0, -1] = -1
        return (pixel_values, position_ids)

    def prepare_call(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
    ) -> tuple[tuple[torch.Tensor, ...], Gemma4VisionRunContext]:
        seq_len = pixel_values.shape[-2]
        bucket_len = self._bucket_for(seq_len)
        padded = bucket_len is not None and bucket_len > seq_len

        if padded:
            pad = bucket_len - seq_len
            pixel_values = F.pad(pixel_values, (0, 0, 0, pad))
            # -1 is the tower's own padding marker: it drives both the attention
            # mask and the pooler mask, so padded patches change no output.
            pixel_position_ids = F.pad(pixel_position_ids, (0, 0, 0, pad), value=-1)

        ctx = Gemma4VisionRunContext(
            padding=VisionPaddingContext(
                real_seq_len=seq_len,
                bucket_len=bucket_len,
                padded=padded,
                merge_sq=self._pooling_window(),
            )
        )
        return (pixel_values, pixel_position_ids), ctx

    def unpad_outputs(
        self, result: tuple[torch.Tensor, torch.Tensor], ctx: Gemma4VisionRunContext
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled, pooler_mask = result
        if ctx.padding.padded:
            real_pooled_len = ctx.padding.real_seq_len // ctx.padding.merge_sq
            pooled = pooled[:, :real_pooled_len]
            pooler_mask = pooler_mask[:, :real_pooled_len]
        # Stays on device: the projector runs next with device weights.
        return pooled, pooler_mask

    def run(
        self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = pixel_values.shape[-2]
        if self._bucket_for(seq_len) is None and self.eager_fn is not None:
            # Past the top bucket, and recompiling is not allowed once warmup
            # has finished (fail_on_recompile), so run this image uncompiled.
            if seq_len not in self._warned_oversize:
                self._warned_oversize.add(seq_len)
                buckets = self.resolve_patch_buckets()
                logger.warning(
                    "Gemma4VisionTpuCompilePolicy: patch count %d exceeds the "
                    "largest precompiled bucket %s; running this image eagerly. "
                    "Add a bucket to SGLANG_TPU_VIT_PRECOMPILE_SIZES or raise "
                    "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.",
                    seq_len,
                    buckets[-1] if buckets else None,
                )
            with torch.no_grad():
                return self.eager_fn(pixel_values, pixel_position_ids)
        return super().run(pixel_values, pixel_position_ids)

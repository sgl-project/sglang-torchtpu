"""ViT TPU compile runner with explicit patch-bucket padding."""

from __future__ import annotations

import inspect
import logging
import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch._dynamo
import torch.nn.functional as F
from torch_tpu._internal.sync.sync import synchronize as tpu_sync

from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang.srt.hardware_backend.tpu.patches import _register_tpu_compile_backend

logger = logging.getLogger(__name__)

# Unlike the upstream CUDA and NPU runners, TPU precompiles the vision encoder
# for fixed patch counts. Real images are often non-square, so runtime pads them
# to the nearest bucket. MMMU reached 3400 patches; 4900 = 70^2 leaves headroom
# without warming a huge grid.
_DEFAULT_MAX_PATCHES = 4900


def _compute_patch_counts(
    merge_size: int, min_patches: int = 196, max_patches: int = _DEFAULT_MAX_PATCHES
) -> list[int]:
    """Compute square-image patch-count buckets aligned to merger windows."""
    counts = []
    min_grid = max(
        merge_size, math.ceil(math.sqrt(min_patches) / merge_size) * merge_size
    )
    grid_dim = min_grid
    while True:
        patch_count = grid_dim * grid_dim
        if patch_count > max_patches:
            break
        counts.append(patch_count)
        grid_dim += merge_size
    return counts


def _validate_patch_buckets(buckets: list[int], merge_size: int) -> list[int]:
    merge_sq = merge_size**2
    for bucket in buckets:
        grid = math.isqrt(bucket)
        if bucket <= 0 or bucket % merge_sq != 0 or grid * grid != bucket:
            raise ValueError(
                "SGLANG_TPU_VIT_PRECOMPILE_SIZES entries must be positive "
                f"perfect squares divisible by {merge_sq}; got {bucket}."
            )
    return buckets


def _cu_seqlens_to_segment_ids(
    cu_seqlens: torch.Tensor,
    total_patches: int,
    device: torch.device,
) -> torch.Tensor:
    """Build segment IDs on the host, then transfer them to the patch device."""
    if cu_seqlens.device.type != "cpu":
        raise ValueError("TPU vision cu_seqlens must remain on CPU")

    cu = cu_seqlens.to(dtype=torch.int64)
    if cu.ndim != 1 or cu.numel() == 0 or int(cu[0]) != 0:
        raise ValueError("cu_seqlens must be a non-empty 1D tensor starting at zero")
    if int(cu[-1]) != total_patches:
        raise ValueError(
            f"cu_seqlens[-1]={int(cu[-1])} != total_patches={total_patches}"
        )

    lengths = cu[1:] - cu[:-1]
    if bool(torch.any(lengths < 0)):
        raise ValueError("cu_seqlens must be nondecreasing")
    segment_ids = torch.repeat_interleave(
        torch.arange(lengths.numel(), dtype=torch.int32), lengths
    ).unsqueeze(0)
    return segment_ids.to(device=device, non_blocking=True)


@dataclass(frozen=True)
class VisionPaddingContext:
    real_seq_len: int
    bucket_len: int | None
    padded: bool
    merge_sq: int


class VisionTpuCompilePolicy(CompilePolicy):
    """Complete compile policy for compiled Qwen3-VL ViT blocks."""

    platform = "tpu"
    backend = "tpu"
    fullgraph = True
    dynamic = False

    def __init__(
        self,
        vit: Any | None = None,
        *,
        model_runner: Any | None = None,
        owner: Any | None = None,
        method_name: str = "forward",
        method: Any | None = None,
        name: str = "qwen3_vl_vit_encoder",
        label: str | None = None,
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            owner=owner,
            method_name=method_name,
            method=method,
            name=name,
            label=label,
        )
        if vit is None and owner is not None:
            vit = getattr(owner, "vit", owner)
        self.vit = vit
        self.compiled_fn = None
        self._patch_buckets: list[int] | None = None
        self._warned_oversize: set[int] = set()

    def _resolve_runner(self) -> Any | None:
        if self.owner is None:
            return None
        return getattr(self.owner, "graph_runners", self.owner)

    def install(self) -> bool:
        runner = self._resolve_runner()
        if runner is None:
            logger.warning("VisionTpuCompilePolicy: missing owner; cannot install.")
            return False
        if not getattr(runner, "_supports_segment_ids", False):
            logger.info(
                "VisionTpuCompilePolicy: skipped because segment_ids are not "
                "supported."
            )
            return False

        torch._dynamo.config.accumulated_cache_size_limit = 1024
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1024

        _register_tpu_compile_backend()
        method = getattr(runner, self.method_name)
        self.compiled_fn = torch.compile(
            torch.no_grad()(method),
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )
        setattr(runner, self.method_name, self.compiled_fn)

        def _run(
            x: torch.Tensor,
            cu_seqlens: torch.Tensor,
            position_embeddings: torch.Tensor | None = None,
            rotary_pos_emb_cos: torch.Tensor | None = None,
            rotary_pos_emb_sin: torch.Tensor | None = None,
            cu_window_seqlens: torch.Tensor | None = None,
            output_indices: torch.Tensor | None = None,
        ) -> torch.Tensor:
            return self.run(
                x.unsqueeze(1),
                cu_seqlens,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
                output_indices=output_indices,
            )

        runner.run = _run
        logger.info(
            "VisionTpuCompilePolicy: installed %s on %s.%s.",
            self.name,
            type(runner).__name__,
            self.method_name,
        )
        return True

    def make_warmup_inputs(
        self,
        patch_count: int,
        device: torch.device,
        dtype: torch.dtype,
        embed_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vit = self.vit
        x = torch.zeros(patch_count, 1, embed_dim, device=device, dtype=dtype)
        grid = int(math.isqrt(patch_count))
        cos, sin = vit.rot_pos_emb([[1, grid, grid]])
        cu_seqlens = torch.tensor([0, patch_count], device=device, dtype=torch.int32)
        segment_ids = torch.zeros(1, patch_count, device=device, dtype=torch.int32)

        merge_sq = getattr(vit, "spatial_merge_size", 2) ** 2
        real = max(merge_sq, patch_count - merge_sq)
        if patch_count > real:
            segment_ids[0, real:] = 1

        return x, cu_seqlens, cos, sin, segment_ids

    def pad_inputs(
        self,
        x_3d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        VisionPaddingContext,
    ]:
        seq_len = x_3d.shape[0]
        merge_sq = getattr(self.vit, "spatial_merge_size", 2) ** 2
        bucket_len = self._bucket_for(seq_len)
        padded = bucket_len is not None and bucket_len > seq_len
        cos = rotary_pos_emb_cos
        sin = rotary_pos_emb_sin

        if padded:
            pad = bucket_len - seq_len
            x_3d = F.pad(x_3d, (0, 0, 0, 0, 0, pad))
            cos = F.pad(cos, (0, 0, 0, pad))
            sin = F.pad(sin, (0, 0, 0, pad))
            segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x_3d.device)
            num_real_segments = int(cu_seqlens.shape[0]) - 1
            pad_segment = torch.full(
                (1, pad), num_real_segments, dtype=torch.int32, device=x_3d.device
            )
            segment_ids = torch.cat([segment_ids, pad_segment], dim=1)
        else:
            if bucket_len is None and seq_len not in self._warned_oversize:
                self._warned_oversize.add(seq_len)
                buckets = self.resolve_patch_buckets()
                logger.warning(
                    "ViTTpuCompileRunner: patch count %d exceeds the largest "
                    "precompiled bucket %s; this image triggers a one-time ViT "
                    "recompile. Raise SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.",
                    seq_len,
                    buckets[-1] if buckets else None,
                )
            segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x_3d.device)

        graph_cu_seqlens = torch.tensor(
            [0, x_3d.shape[0]], device=x_3d.device, dtype=torch.int32
        )
        context = VisionPaddingContext(
            real_seq_len=seq_len,
            bucket_len=bucket_len,
            padded=padded,
            merge_sq=merge_sq,
        )
        return x_3d, graph_cu_seqlens, cos, sin, segment_ids, context

    def unpad_outputs(
        self, result: torch.Tensor, context: VisionPaddingContext
    ) -> torch.Tensor:
        if context.padded:
            result = result[: context.real_seq_len // context.merge_sq]
        return result

    def resolve_patch_buckets(self) -> list[int]:
        if self._patch_buckets is not None:
            return self._patch_buckets

        env_val = os.environ.get("SGLANG_TPU_VIT_PRECOMPILE_SIZES", None)
        if env_val is not None:
            if env_val.strip() == "":
                self._patch_buckets = []
                return self._patch_buckets
            buckets = sorted({int(x.strip()) for x in env_val.split(",") if x.strip()})
            buckets = _validate_patch_buckets(
                buckets, getattr(self.vit, "spatial_merge_size", 2)
            )
        else:
            merge_size = getattr(self.vit, "spatial_merge_size", 2)
            max_patches = int(
                os.environ.get(
                    "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES",
                    str(_DEFAULT_MAX_PATCHES),
                )
            )
            buckets = _compute_patch_counts(merge_size, max_patches=max_patches)

        self._patch_buckets = buckets
        return buckets

    def _bucket_for(self, seq_len: int) -> int | None:
        for bucket in self.resolve_patch_buckets():
            if bucket >= seq_len:
                return bucket
        return None

    def precompile(self, patch_counts: list[int] | None = None) -> None:
        """Warm the torch.compile cache for common patch counts."""
        owner = self.owner
        runner = self._resolve_runner()
        if owner is None or runner is None:
            logger.warning("VisionTpuCompilePolicy.precompile: missing owner.")
            return
        if not getattr(runner, "_supports_segment_ids", False):
            logger.info(
                "VisionTpuCompilePolicy.precompile: skipped because segment_ids "
                "are not supported."
            )
            return

        if patch_counts is None:
            patch_counts = self.resolve_patch_buckets()
        if not patch_counts:
            logger.info(
                "VisionTpuCompilePolicy.precompile: no patch counts to warm "
                "(disabled via SGLANG_TPU_VIT_PRECOMPILE_SIZES='')."
            )
            return

        if self.compiled_fn is None:
            logger.info(
                "VisionTpuCompilePolicy.precompile: skipped because policy is "
                "not installed."
            )
            return

        vit = self.vit
        device = vit.patch_embed.proj.weight.device
        dtype = vit.patch_embed.proj.weight.dtype
        embed_dim = vit.blocks[0].norm1.normalized_shape[0]

        logger.info(
            "VisionTpuCompilePolicy.precompile: warming %d patch counts %s "
            "(embed_dim=%d).",
            len(patch_counts),
            patch_counts,
            embed_dim,
        )

        init_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.no_grad():
                for index, patch_count in enumerate(patch_counts):
                    logger.info(
                        "VisionTpuCompilePolicy.precompile: [%d/%d] "
                        "patch_count=%d ...",
                        index + 1,
                        len(patch_counts),
                        patch_count,
                    )
                    inputs = self.make_warmup_inputs(
                        patch_count, device, dtype, embed_dim
                    )
                    result = self.compiled_fn(*inputs)
                    tpu_sync([result], wait=True)
        finally:
            torch.set_default_dtype(init_dtype)

        logger.info("VisionTpuCompilePolicy.precompile: done.")

    def run(
        self,
        x_3d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        output_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        runner = self._resolve_runner()
        if runner is None or self.compiled_fn is None:
            raise RuntimeError("VisionTpuCompilePolicy.run called before install().")
        (
            x_3d,
            graph_cu_seqlens,
            cos,
            sin,
            segment_ids,
            context,
        ) = self.pad_inputs(x_3d, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin)

        if not getattr(runner, "_logged_shapes", False):
            runner._logged_shapes = True
            logger.info(
                "ViTTpuCompileRunner.run: seq_len=%d bucket=%s padded=%s "
                "x=%s cos=%s sin=%s seg=%s cu=%s",
                context.real_seq_len,
                context.bucket_len,
                context.padded,
                tuple(x_3d.shape),
                tuple(cos.shape),
                tuple(sin.shape),
                tuple(segment_ids.shape),
                tuple(graph_cu_seqlens.shape),
            )

        result = self.compiled_fn(x_3d, graph_cu_seqlens, cos, sin, segment_ids)
        result = self.unpad_outputs(result, context)

        if output_indices is not None:
            result = result.index_select(0, output_indices)

        return result


class ViTTpuCompileRunner:
    """Vision-encoder runner for TPU using ``torch.compile``."""

    def __init__(self, vit: Any) -> None:
        self.vit = vit
        self._compiled_fn = None

        self._deepstack_visual_indexes = set(
            getattr(vit, "deepstack_visual_indexes", []) or []
        )
        self._deepstack_merger_list = getattr(vit, "deepstack_merger_list", None)

        first_block = vit.blocks[0] if len(vit.blocks) > 0 else None
        if first_block is not None:
            sig = inspect.signature(first_block.forward)
            self._supports_segment_ids = (
                "segment_ids" in sig.parameters
                and self._all_blocks_use_segment_id_attention()
            )
        else:
            self._supports_segment_ids = False
        if not self._supports_segment_ids:
            logger.warning(
                "ViTTpuCompileRunner: active ViT attention does not support "
                "segment_ids; "
                "falling back to eager ViT execution."
            )

    def _all_blocks_use_segment_id_attention(self) -> bool:
        for block in self.vit.blocks:
            attention = getattr(block, "attn", None)
            backend = getattr(attention, "qkv_backend", None)
            if not getattr(backend, "supports_segment_ids", False):
                return False
        return True

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled ViT boundary: blocks + merger over a padded patch bucket.

        Named ``forward`` so the encoder boundary matches the module-level
        compile annotation convention used by the decoder.
        """
        vit = self.vit
        deepstack_features = []
        num_deepstack = 0

        for layer_num, block in enumerate(vit.blocks):
            x = block(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
                segment_ids=segment_ids,
            )
            if layer_num in self._deepstack_visual_indexes:
                deepstack = self._deepstack_merger_list[num_deepstack](x)
                deepstack_features.append(deepstack)
                num_deepstack += 1

        x = vit.merger(x)
        if deepstack_features:
            return torch.cat([x] + deepstack_features, dim=1)
        return x

    def _blocks_and_merger_fallback(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
    ) -> torch.Tensor:
        vit = self.vit
        deepstack_features = []
        num_deepstack = 0

        for layer_num, block in enumerate(vit.blocks):
            x = block(
                x,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb_cos=rotary_pos_emb_cos,
                rotary_pos_emb_sin=rotary_pos_emb_sin,
            )
            if layer_num in self._deepstack_visual_indexes:
                deepstack = self._deepstack_merger_list[num_deepstack](x)
                deepstack_features.append(deepstack)
                num_deepstack += 1

        x = vit.merger(x)
        if deepstack_features:
            return torch.cat([x] + deepstack_features, dim=1)
        return x

    def run(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: torch.Tensor | None = None,
        rotary_pos_emb_cos: torch.Tensor | None = None,
        rotary_pos_emb_sin: torch.Tensor | None = None,
        cu_window_seqlens: torch.Tensor | None = None,
        output_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run eagerly until the compile policy replaces this entry point.

        Qwen3-VL calls ``graph_runners.run`` for TPU vision execution even before
        policy installation. Segment-aware TPU attention therefore needs the
        same segment IDs in this eager path as it receives after installation.
        """
        x_3d = x.unsqueeze(1)

        if self._supports_segment_ids:
            seq_len = x_3d.shape[0]
            segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x_3d.device)
            graph_cu_seqlens = torch.tensor(
                [0, seq_len], device=x_3d.device, dtype=torch.int32
            )
            result = self.forward(
                x_3d,
                graph_cu_seqlens,
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
                segment_ids,
            )
        else:
            result = self._blocks_and_merger_fallback(
                x_3d, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin
            )
        if output_indices is not None:
            result = result.index_select(0, output_indices)
        return result

"""TPU vision-encoder runners: the eager boundary the compile policy replaces."""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, Protocol, TypeVar

import torch
from torch import nn

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


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


def maybe_move_result_to_host(result: torch.Tensor) -> torch.Tensor:
    """Land the encoder output on the host unless it is kept on device.

    Both the eager and the compiled path must agree here, or the multimodal
    embedding cache ends up mixing devices across the two modes.
    """
    # Move while the shape is still the bucket: every consumer from here on is
    # variable-length and would recompile per length.
    if envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.get():
        return result
    return result.to("cpu")


class BlockedEncoder(Protocol):
    """The block stack every runner probes; the bound on EncoderT."""

    blocks: Sequence[nn.Module]


# The concrete encoders cannot be imported here (the model modules import this
# one), so each runner declares the surface it touches as a Protocol instead.
EncoderT = TypeVar("EncoderT", bound=BlockedEncoder)


class EncoderTpuRunner(ABC, Generic[EncoderT]):
    """Shared half of the TPU vision-encoder runners.

    Holds the encoder and probes whether its blocks can take segment_ids, which
    is what decides between the segment-aware boundary and the eager fallback.
    Subclasses own the boundary itself; its signature is per-encoder, so the two
    abstract methods below only pin down the roles.
    """

    def __init__(self, encoder: EncoderT, *, unsupported_warning: str) -> None:
        self.encoder: EncoderT = encoder
        self.supports_segment_ids: bool = self._probe_segment_ids()
        if not self.supports_segment_ids:
            logger.warning(unsupported_warning)

    def _probe_segment_ids(self) -> bool:
        """Whether every block takes segment_ids and has a backend honoring it."""
        blocks = self.encoder.blocks
        if len(blocks) == 0:
            return False
        if "segment_ids" not in inspect.signature(blocks[0].forward).parameters:
            return False
        for block in blocks:
            attention = getattr(block, "attn", None)
            backend = getattr(attention, "qkv_backend", None)
            if not getattr(backend, "supports_segment_ids", False):
                return False
        return True

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """The compiled region: shape-static over one padded patch bucket."""

    @abstractmethod
    def run_eager(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Eager boundary for one encoder; the signature is per-encoder."""

    def run(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Serve entry point, eager until the policy replaces it on install.

        Owns the host contract so the eager path lands where the installed
        policy's does. Override to append work that must run after the D2H.
        """
        return maybe_move_result_to_host(self.run_eager(*args, **kwargs))


class Qwen3VLRotaryPosEmb(Protocol):
    cos_sin_cache: torch.Tensor


class Qwen3VLViTEncoder(BlockedEncoder, Protocol):
    """What Qwen3VLMoeVisionModel must expose for this runner.

    deepstack_visual_indexes and deepstack_merger_list are deliberately absent:
    a deepstack-less encoder is supported, so they stay optional and are read
    with a default rather than declared here.
    """

    spatial_merge_size: int
    merger: nn.Module
    rotary_pos_emb: Qwen3VLRotaryPosEmb

    def rot_pos_ids(self, h: int, w: int, merge_size: int) -> torch.Tensor: ...


class ViTTpuCompileRunner(EncoderTpuRunner[Qwen3VLViTEncoder]):
    """Vision-encoder runner for TPU using torch.compile."""

    def __init__(self, encoder: Qwen3VLViTEncoder) -> None:
        super().__init__(
            encoder,
            unsupported_warning=(
                "ViTTpuCompileRunner: active ViT attention does not support "
                "segment_ids; falling back to eager ViT execution."
            ),
        )

        # Published by the TPU compile policy on install; empty until then,
        # which makes bucket_for_patches the identity.
        self.padded_patch_buckets: tuple[int, ...] = ()
        self._warned_oversize_patches: set[int] = set()
        # Host mirror of the rotary cache, taken on first use: some loaders
        # build the model under a meta device, where it holds no values yet.
        self._rotary_cos_sin_cache_cpu: torch.Tensor | None = None

        # Optional surface, so it is read with a default instead of being
        # declared on Qwen3VLViTEncoder: a deepstack-less encoder is supported.
        self._deepstack_visual_indexes: set[int] = set(
            getattr(encoder, "deepstack_visual_indexes", None) or []
        )
        self._deepstack_merger_list: nn.ModuleList | None = getattr(
            encoder, "deepstack_merger_list", None
        )

    def bucket_for_patches(self, total_patches: int) -> int:
        """Padded patch count for total_patches, or itself when unbucketed."""
        for bucket in self.padded_patch_buckets:
            if bucket >= total_patches:
                return bucket
        if (
            self.padded_patch_buckets
            and total_patches not in self._warned_oversize_patches
        ):
            self._warned_oversize_patches.add(total_patches)
            logger.warning(
                "ViTTpuCompileRunner: patch count %d exceeds the largest "
                "precompiled bucket %d; this image triggers a one-time ViT "
                "recompile. Raise SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES, or "
                "lower max_pixels via --mm-process-config.",
                total_patches,
                self.padded_patch_buckets[-1],
            )
        return total_patches

    def _host_rotary_cache(self) -> torch.Tensor:
        if self._rotary_cos_sin_cache_cpu is None:
            self._rotary_cos_sin_cache_cpu = (
                self.encoder.rotary_pos_emb.cos_sin_cache.detach().to("cpu", copy=True)
            )
        return self._rotary_cos_sin_cache_cpu

    def _rot_pos_emb_cpu(
        self, grid_thw: list[list[int]], *, bucket_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Host-side rot_pos_emb; cos/sin come back padded to bucket_len."""
        pos_ids_per_image = []
        for t, h, w in grid_thw:
            base = self.encoder.rot_pos_ids(h, w, self.encoder.spatial_merge_size)
            pos_ids_per_image.append(base if t == 1 else base.repeat(t, 1))
        pos_ids = torch.cat(pos_ids_per_image, dim=0)

        padded_pos_ids = pos_ids.new_zeros((bucket_len, pos_ids.shape[1]))
        padded_pos_ids[: pos_ids.shape[0]] = pos_ids

        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        cos, sin = self._host_rotary_cache()[:max_grid_size].chunk(2, dim=-1)
        return cos[padded_pos_ids].flatten(1), sin[padded_pos_ids].flatten(1)

    def forward(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled ViT boundary: blocks + merger over a padded patch bucket.

        Named forward so the encoder boundary matches the module-level
        compile annotation convention used by the decoder.
        """
        encoder = self.encoder
        deepstack_features = []
        num_deepstack = 0

        for layer_num, block in enumerate(encoder.blocks):
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

        x = encoder.merger(x)
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
        encoder = self.encoder
        deepstack_features = []
        num_deepstack = 0

        for layer_num, block in enumerate(encoder.blocks):
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

        x = encoder.merger(x)
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
        """Eager entry point plus Qwen3-VL's output-row subset.

        The selection runs after the base lands the result, matching the order
        the installed policy's unpad_outputs uses.
        """
        result = super().run(x, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin)
        if output_indices is not None:
            result = result.index_select(0, output_indices.to(result.device))
        return result

    def run_eager(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor | None,
        rotary_pos_emb_sin: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run eagerly until the compile policy replaces this entry point.

        Qwen3-VL calls graph_runners.run for TPU vision execution even before
        policy installation. Segment-aware TPU attention therefore needs the
        same segment IDs in this eager path as it receives after installation.
        """
        x_3d = x.unsqueeze(1)
        if not self.supports_segment_ids:
            return self._blocks_and_merger_fallback(
                x_3d, cu_seqlens, rotary_pos_emb_cos, rotary_pos_emb_sin
            )

        seq_len = x_3d.shape[0]
        segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x_3d.device)
        graph_cu_seqlens = torch.tensor(
            [0, seq_len], device=x_3d.device, dtype=torch.int32
        )
        return self.forward(
            x_3d,
            graph_cu_seqlens,
            rotary_pos_emb_cos,
            rotary_pos_emb_sin,
            segment_ids,
        )

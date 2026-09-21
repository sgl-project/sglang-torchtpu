"""MoonViT (Kimi-K2.5) ViT encoder runner + policy for TPU ``torch.compile``.

MoonViT diverges from Qwen3-VL's ViT (complex ``freqs_cis`` RoPE, a temporal-pool
merger that returns a list, no ``spatial_merge_size``/deepstack), so it gets its
own runner/policy rather than being contorted onto ``VisionTpuCompilePolicy``.
The model-agnostic bucketing / padding / segment-id helpers are reused from
``vit_tpu_compile_runner`` so there is a single copy.

Compiled boundary = the encoder **block loop + ``final_layernorm``** only. The
eager preamble (patch embed, pos embed, RoPE precompute, ``cu_seqlens``) and the
eager postamble (``tpool_patch_merger`` temporal pooling) stay outside the graph.
RoPE is passed as a ``kimi_k25.RealRoPE`` (cos, sin) pair so no
complex op enters the graph; per-image boundaries + padding are carried by a
fixed-shape ``segment_ids`` instead of a variable-length ``cu_seqlens``.
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Any

import torch
import torch.nn.functional as F

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
    VisionPaddingContext,
    VisionTpuCompilePolicy,
    _compute_patch_counts,
    _cu_seqlens_to_segment_ids,
    _validate_patch_buckets,
)

logger = logging.getLogger(__name__)


class MoonVitTpuCompileRunner:
    """Runs MoonViT's encoder blocks + ``final_layernorm`` on TPU.

    Eager until ``MoonVitTpuCompilePolicy.install`` replaces ``run`` with the
    compiled boundary. The merge (``tpool_patch_merger``) is applied by the
    caller after this runner, so the runner never touches it.
    """

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self.owner = encoder
        self._logged_shapes = False

        # Local import: kimi_k25 imports this module while still executing, so a
        # top-level import back into it would be order-dependent.
        from sglang.srt.models.kimi_k25 import RealRoPE

        self._real_rope = RealRoPE

        blocks = encoder.blocks
        first_block = blocks[0] if len(blocks) > 0 else None
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
                "MoonVitTpuCompileRunner: MoonViT attention does not support "
                "segment_ids; the caller must keep the eager cu_seqlens loop."
            )

    @property
    def supports_segment_ids(self) -> bool:
        """False => the caller must not use this runner; nothing would mask."""
        return self._supports_segment_ids

    def _all_blocks_use_segment_id_attention(self) -> bool:
        for block in self.encoder.blocks:
            attention = getattr(block, "attn", None)
            backend = getattr(attention, "qkv_backend", None)
            if not getattr(backend, "supports_segment_ids", False):
                return False
        return True

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compiled MoonViT boundary over a padded patch bucket.

        Args:
            x: ``[S_bucket, D]`` patch features (pos-embed already added).
            cos, sin: ``[S_bucket, head_dim/2]`` real RoPE tensors.
            segment_ids: ``[1, S_bucket]`` int32 per-patch image id (padding is a
                distinct trailing id so it never attends to real tokens).
        Returns:
            ``[1, S_bucket, D]`` (matches the eager block-loop output shape).
        """
        encoder = self.encoder
        for block in encoder.blocks:
            x = block(
                x,
                None,  # cu_seqlens unused: segment_ids drives TPU attention
                None,  # max_seqlen unused
                rope_freqs_cis=self._real_rope(cos, sin),
                segment_ids=segment_ids,
            )
        return encoder.final_layernorm(x)

    def run(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Eager entry point (used until the policy installs the compiled fn).

        Builds segment_ids from ``cu_seqlens`` so eager and compiled paths feed
        the segment-aware TPU attention identically.
        """
        seq_len = x.shape[0]
        segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x.device)
        return self.forward(x, cos, sin, segment_ids)


class MoonVitTpuCompilePolicy(VisionTpuCompilePolicy):
    """Compile policy for the MoonViT encoder block loop + final_layernorm.

    Only the model-specific half of the shared vision lifecycle: MoonViT's
    compiled boundary takes ``(x, cos, sin, segment_ids)`` -- no cu_seqlens, no
    spatial_merge_size -- and its temporal-pool merger runs eagerly outside the
    graph. Install, warmup and the padded run orchestration come from the base.
    """

    def __init__(
        self,
        encoder: Any | None = None,
        *,
        name: str = "moonvit_vit_encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder, name=name, **kwargs)
        self._warned_oversize: set[int] = set()

    def _runner_ready_for_compile(self, runner: Any) -> bool:
        return getattr(runner, "supports_segment_ids", False)

    def run(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Run the compiled encoder once per image.

        ``x`` holds every image of the batch packed along the patch axis, with
        ``cu_seqlens`` marking the boundaries. Compiled graphs exist only for
        the precompiled patch-count buckets and the packed total rarely matches
        one, so each image is sliced out and run at its own bucket. Attention
        does not cross image boundaries, so the result is unchanged.

        The slices are cloned because torch_tpu passes a compiled function the
        full backing buffer of a view rather than the view itself.
        """
        bounds = cu_seqlens.tolist()
        if len(bounds) <= 2:
            return self._run_one(x, cos, sin, cu_seqlens)
        outs = []
        for start, end in zip(bounds[:-1], bounds[1:]):
            single = torch.tensor([0, end - start], dtype=torch.int32)
            outs.append(
                self._run_one(
                    x[start:end].clone(),
                    cos[start:end].clone(),
                    sin[start:end].clone(),
                    single,
                )
            )
        # Each output is [1, S_i, D]; concatenate along the patch axis.
        return torch.cat(outs, dim=outs[0].ndim - 2)

    def _run_one(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = x.shape[0]
        if self._bucket_for(seq_len) is None and self.eager_fn is not None:
            # No precompiled bucket fits this image, and recompiling is not
            # allowed once warmup has finished (fail_on_recompile), so run the
            # uncompiled encoder for it.
            if seq_len not in self._warned_oversize:
                self._warned_oversize.add(seq_len)
                buckets = self.resolve_patch_buckets()
                logger.warning(
                    "MoonVitTpuCompilePolicy: patch count %d exceeds largest "
                    "precompiled bucket %s; running this image eagerly. Add a "
                    "bucket to SGLANG_TPU_VIT_PRECOMPILE_SIZES or raise "
                    "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.",
                    seq_len,
                    buckets[-1] if buckets else None,
                )
            segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x.device)
            with torch.no_grad():
                return self.eager_fn(x, cos, sin, segment_ids)
        return super().run(x, cos, sin, cu_seqlens)

    def _install_run_adapter(self, runner: Any) -> None:
        def _run(
            x: torch.Tensor,
            cos: torch.Tensor,
            sin: torch.Tensor,
            cu_seqlens: torch.Tensor,
        ) -> torch.Tensor:
            return self.run(x, cos, sin, cu_seqlens)

        runner.run = _run

    def compute_patch_buckets(self) -> list[int]:
        # merge_size=1: MoonViT merges patches outside the compiled region, so
        # there is no merger divisibility constraint to satisfy.
        if envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.is_set():
            # Set but empty => () => warms nothing, i.e. precompile disabled.
            buckets = sorted(
                {int(x) for x in envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.get()}
            )
            return _validate_patch_buckets(buckets, 1)
        return _compute_patch_counts(
            1, max_patches=envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.get()
        )

    def make_warmup_inputs(self, patch_count: int) -> tuple[torch.Tensor, ...]:
        encoder = self.vit
        param = next(encoder.parameters())
        device, dtype = param.device, param.dtype
        embed_dim = encoder.blocks[0].hidden_dim

        grid = int(math.isqrt(patch_count))
        x = torch.zeros(patch_count, embed_dim, device=device, dtype=dtype)
        grid_thws = torch.tensor([[1, grid, grid]], dtype=torch.int32, device=device)
        freqs_cis = encoder.rope_2d.get_freqs_cis(grid_thws=grid_thws, device=device)
        segment_ids = torch.zeros(1, patch_count, device=device, dtype=torch.int32)
        # Exercise the padding-mask path: mark the tail as a second segment.
        if patch_count > 1:
            segment_ids[0, patch_count - 1] = 1
        return (
            x,
            freqs_cis.real.contiguous(),
            freqs_cis.imag.contiguous(),
            segment_ids,
        )

    def prepare_call(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[tuple[torch.Tensor, ...], VisionPaddingContext]:
        seq_len = x.shape[0]
        bucket_len = self._bucket_for(seq_len)
        padded = bucket_len is not None and bucket_len > seq_len

        if padded:
            pad = bucket_len - seq_len
            x = F.pad(x, (0, 0, 0, pad))
            cos = F.pad(cos, (0, 0, 0, pad))
            sin = F.pad(sin, (0, 0, 0, pad))
            segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x.device)
            num_real_segments = int(cu_seqlens.shape[0]) - 1
            pad_segment = torch.full(
                (1, pad), num_real_segments, dtype=torch.int32, device=x.device
            )
            segment_ids = torch.cat([segment_ids, pad_segment], dim=1)
        else:
            # seq_len matches a bucket exactly, so no padding is needed.
            # Oversized images never reach this point; _run_one handles them.
            segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x.device)

        ctx = VisionPaddingContext(
            real_seq_len=seq_len,
            bucket_len=bucket_len,
            padded=padded,
            merge_sq=1,  # merge happens eagerly outside the compiled region
        )
        return (x, cos, sin, segment_ids), ctx

    def unpad_outputs(
        self, result: torch.Tensor, ctx: VisionPaddingContext
    ) -> torch.Tensor:
        # result is [1, S_bucket, D]; padding is along the patch axis (dim 1).
        if ctx.padded:
            result = result[:, : ctx.real_seq_len]
        return result

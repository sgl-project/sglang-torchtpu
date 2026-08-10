"""MoonViT (Kimi-K2.5) ViT encoder runner + policy for TPU ``torch.compile``.

MoonViT diverges from Qwen3-VL's ViT (complex ``freqs_cis`` RoPE, a temporal-pool
merger that returns a list, no ``spatial_merge_size``/deepstack), so it gets its
own runner/policy rather than being contorted onto ``VisionTpuCompilePolicy``.
The model-agnostic bucketing / padding / segment-id helpers are reused from
``vit_tpu_compile_runner`` so there is a single copy.

Compiled boundary = the encoder **block loop + ``final_layernorm``** only. The
eager preamble (patch embed, pos embed, RoPE precompute, ``cu_seqlens``) and the
eager postamble (``tpool_patch_merger`` temporal pooling) stay outside the graph.
RoPE is passed as a real ``(cos, sin)`` tuple (see ``kimi_k25.apply_rope``) so no
complex op enters the graph; per-image boundaries + padding are carried by a
fixed-shape ``segment_ids`` instead of a variable-length ``cu_seqlens``.
"""

from __future__ import annotations

import inspect
import logging
import math
import os
from typing import Any

import torch
import torch.nn.functional as F
from torch_tpu._internal.sync.sync import synchronize as tpu_sync

from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang.srt.hardware_backend.tpu.graph_runner._vit_tpu_common import (
    _DEFAULT_MAX_PATCHES,
    VisionPaddingContext,
    _compute_patch_counts,
    _cu_seqlens_to_segment_ids,
    _validate_patch_buckets,
)
from sglang.srt.hardware_backend.tpu.patches import _register_tpu_compile_backend

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
                rope_freqs_cis=(cos, sin),
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


class MoonVitTpuCompilePolicy(CompilePolicy):
    """Compile policy for the MoonViT encoder block loop + final_layernorm."""

    platform = "tpu"
    backend = "tpu"
    fullgraph = True
    dynamic = False

    def __init__(
        self,
        encoder: Any | None = None,
        *,
        model_runner: Any | None = None,
        owner: Any | None = None,
        method_name: str = "forward",
        method: Any | None = None,
        name: str = "moonvit_vit_encoder",
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
        self.encoder = encoder if encoder is not None else owner
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
            logger.warning("MoonVitTpuCompilePolicy: missing owner; cannot install.")
            return False
        if not getattr(runner, "_supports_segment_ids", False):
            logger.info("MoonVitTpuCompilePolicy: skipped (segment_ids unsupported).")
            return False

        torch._dynamo.config.accumulated_cache_size_limit = 1024
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1024

        _register_tpu_compile_backend()
        self.compiled_fn = torch.compile(
            torch.no_grad()(runner.forward),
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )
        runner.run = self.run
        logger.info("MoonVitTpuCompilePolicy: installed on %s.", type(runner).__name__)
        return True

    def resolve_patch_buckets(self) -> list[int]:
        if self._patch_buckets is not None:
            return self._patch_buckets

        env_val = os.environ.get("SGLANG_TPU_VIT_PRECOMPILE_SIZES", None)
        # MoonViT does not merge patches inside the compiled region, so the only
        # divisibility constraint is the attention block size (handled by the
        # backend's internal 128-pad); validate against merge_size=1.
        if env_val is not None:
            if env_val.strip() == "":
                self._patch_buckets = []
                return self._patch_buckets
            buckets = sorted({int(x.strip()) for x in env_val.split(",") if x.strip()})
            buckets = _validate_patch_buckets(buckets, 1)
        else:
            max_patches = int(
                os.environ.get(
                    "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES",
                    str(_DEFAULT_MAX_PATCHES),
                )
            )
            buckets = _compute_patch_counts(1, max_patches=max_patches)

        self._patch_buckets = buckets
        return buckets

    def _bucket_for(self, seq_len: int) -> int | None:
        for bucket in self.resolve_patch_buckets():
            if bucket >= seq_len:
                return bucket
        return None

    def pad_inputs(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, VisionPaddingContext
    ]:
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
            if bucket_len is None and seq_len not in self._warned_oversize:
                self._warned_oversize.add(seq_len)
                buckets = self.resolve_patch_buckets()
                logger.warning(
                    "MoonVitTpuCompileRunner: patch count %d exceeds largest "
                    "precompiled bucket %s; one-time recompile. Raise "
                    "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.",
                    seq_len,
                    buckets[-1] if buckets else None,
                )
            segment_ids = _cu_seqlens_to_segment_ids(cu_seqlens, seq_len, x.device)

        context = VisionPaddingContext(
            real_seq_len=seq_len,
            bucket_len=bucket_len,
            padded=padded,
            merge_sq=1,  # merge happens eagerly outside the compiled region
        )
        return x, cos, sin, segment_ids, context

    def unpad_outputs(
        self, result: torch.Tensor, context: VisionPaddingContext
    ) -> torch.Tensor:
        # result is [1, S_bucket, D]; padding is along the patch axis (dim 1).
        if context.padded:
            result = result[:, : context.real_seq_len]
        return result

    def make_warmup_inputs(self, patch_count: int, device, dtype, embed_dim):
        grid = int(math.isqrt(patch_count))
        x = torch.zeros(patch_count, embed_dim, device=device, dtype=dtype)
        grid_thws = torch.tensor([[1, grid, grid]], dtype=torch.int32, device=device)
        freqs_cis = self.encoder.rope_2d.get_freqs_cis(
            grid_thws=grid_thws, device=device
        )
        cos = freqs_cis.real.contiguous()
        sin = freqs_cis.imag.contiguous()
        segment_ids = torch.zeros(1, patch_count, device=device, dtype=torch.int32)
        # Exercise the padding-mask path: mark the tail as a second segment.
        if patch_count > 1:
            segment_ids[0, patch_count - 1] = 1
        return x, cos, sin, segment_ids

    def precompile(self, patch_counts: list[int] | None = None) -> None:
        runner = self._resolve_runner()
        if runner is None or self.compiled_fn is None:
            logger.info("MoonVitTpuCompilePolicy.precompile: not installed; skip.")
            return
        if not getattr(runner, "_supports_segment_ids", False):
            return

        if patch_counts is None:
            patch_counts = self.resolve_patch_buckets()
        if not patch_counts:
            logger.info("MoonVitTpuCompilePolicy.precompile: no patch counts to warm.")
            return

        encoder = self.encoder
        param = next(encoder.parameters())
        device, dtype = param.device, param.dtype
        embed_dim = encoder.blocks[0].hidden_dim

        init_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float32)
        try:
            with torch.no_grad():
                for index, patch_count in enumerate(patch_counts):
                    logger.info(
                        "MoonVitTpuCompilePolicy.precompile: [%d/%d] patch_count=%d",
                        index + 1,
                        len(patch_counts),
                        patch_count,
                    )
                    x, cos, sin, seg = self.make_warmup_inputs(
                        patch_count, device, dtype, embed_dim
                    )
                    result = self.compiled_fn(x, cos, sin, seg)
                    tpu_sync([result], wait=True)
        finally:
            torch.set_default_dtype(init_dtype)
        logger.info("MoonVitTpuCompilePolicy.precompile: done.")

    def run(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        if self.compiled_fn is None:
            raise RuntimeError("MoonVitTpuCompilePolicy.run called before install().")
        x, cos, sin, segment_ids, context = self.pad_inputs(x, cos, sin, cu_seqlens)

        runner = self._resolve_runner()
        if runner is not None and not getattr(runner, "_logged_shapes", False):
            runner._logged_shapes = True
            logger.info(
                "MoonVitTpuCompileRunner.run: seq_len=%d bucket=%s padded=%s "
                "x=%s cos=%s seg=%s",
                context.real_seq_len,
                context.bucket_len,
                context.padded,
                tuple(x.shape),
                tuple(cos.shape),
                tuple(segment_ids.shape),
            )

        result = self.compiled_fn(x, cos, sin, segment_ids)
        return self.unpad_outputs(result, context)

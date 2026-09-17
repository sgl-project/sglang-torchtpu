"""TPU compile policies for the vision encoders.

One policy per encoder boundary, plus the model-agnostic lifecycle they share:
install, precompile warmup, and the padded run orchestration. The runners they
drive live in hardware_backend/tpu/graph_runner.
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import torch
import torch._dynamo
import torch.nn.functional as F
from torch_tpu._internal.sync.sync import synchronize as tpu_sync

from sglang.srt.compilation import torch_compile_bundle
from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.tpu.graph_runner.vit_tpu_compile_runner import (
    _cu_seqlens_to_segment_ids,
    maybe_move_result_to_host,
)
from sglang.srt.hardware_backend.tpu.patches import _register_tpu_compile_backend

logger = logging.getLogger(__name__)

# The concrete encoders cannot be imported here (the model modules import this
# one), so each policy declares the surface it touches as a Protocol instead.
EncoderT = TypeVar("EncoderT")


# Unlike the upstream CUDA and NPU runners, TPU precompiles the vision encoder
# for fixed patch counts. Real images are often non-square, so runtime pads them
# to the nearest bucket. MMMU reached 3400 patches; 4900 = 70^2 leaves headroom
# without warming a huge grid.
_DEFAULT_MAX_PATCHES = 4900

_precompiled_max_patch_count: int | None = None


def precompiled_max_patch_count() -> int:
    """Largest ViT patch count that has a precompiled graph.

    The env vars only bound the ladder -- its real ceiling is the largest square
    under the cap (4900 = 70^2 by default) -- so the warmed value wins and the
    envs are a fallback for before precompile runs.
    """
    if _precompiled_max_patch_count is not None:
        return _precompiled_max_patch_count
    explicit_sizes = envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.get()
    if explicit_sizes:
        return max(int(size) for size in explicit_sizes)
    return envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.get()


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


@dataclass(frozen=True)
class VisionPaddingContext:
    real_seq_len: int
    bucket_len: int | None
    padded: bool
    merge_sq: int


class VisionTpuCompilePolicy(CompilePolicy, Generic[EncoderT]):
    """Model-agnostic compile policy for TPU vision encoders.

    Owns the full compile lifecycle -- install, precompile warmup, and the
    padded run orchestration -- for any ViT encoder that precompiles fixed
    patch-count buckets on TPU. Model-specific behavior (the bucket ladder,
    warmup inputs, and the padded call/uncall seam) is expressed through hooks
    that subclasses override; the base never reads encoder attributes itself.
    """

    platform = "tpu"
    backend = "tpu"
    fullgraph = True
    dynamic = False

    def __init__(
        self,
        encoder: EncoderT | None = None,
        *,
        model_runner: Any | None = None,
        owner: Any | None = None,
        method_name: str = "forward",
        method: Any | None = None,
        name: str | None = None,
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
        # The annotated module is the encoder itself, so owner doubles as it.
        self.encoder: EncoderT = encoder if encoder is not None else owner
        self.compiled_fn: Callable[..., torch.Tensor] | None = None
        self._patch_buckets: list[int] | None = None
        self._graph_cu_seqlens_cache: dict[int, torch.Tensor] = {}
        self._logged_shapes = False

    def _resolve_runner(self) -> Any | None:
        if self.owner is None:
            return None
        return getattr(self.owner, "graph_runners", self.owner)

    def install(self) -> bool:
        runner = self._resolve_runner()
        if runner is None:
            logger.warning("VisionTpuCompilePolicy: missing owner; cannot install.")
            return False
        if not self._runner_ready_for_compile(runner):
            logger.info(
                "VisionTpuCompilePolicy: skipped because the runner does not "
                "support the compiled boundary."
            )
            return False

        torch._dynamo.config.accumulated_cache_size_limit = 1024
        if hasattr(torch._dynamo.config, "cache_size_limit"):
            torch._dynamo.config.cache_size_limit = 1024

        _register_tpu_compile_backend()
        method = getattr(runner, self.method_name)
        server_args = self.model_runner.server_args
        use_bundle = (
            server_args.unsafe_save_torch_compile_bundle is not None
            or server_args.unsafe_load_torch_compile_bundle is not None
        )
        compiler = torch_compile_bundle.Compiler if use_bundle else torch.compile
        self.compiled_fn = compiler(
            torch.no_grad()(method),
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )
        setattr(runner, self.method_name, self.compiled_fn)

        self._install_run_adapter(runner)
        logger.info(
            "VisionTpuCompilePolicy: installed %s on %s.%s.",
            self.name,
            type(runner).__name__,
            self.method_name,
        )
        return True

    def resolve_patch_buckets(self) -> list[int]:
        if self._patch_buckets is None:
            self._patch_buckets = self.compute_patch_buckets()
        return self._patch_buckets

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
        if not self._runner_ready_for_compile(runner):
            logger.info(
                "VisionTpuCompilePolicy.precompile: skipped because the runner "
                "does not support the compiled boundary."
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

        global _precompiled_max_patch_count
        _precompiled_max_patch_count = max(patch_counts)

        logger.info(
            "VisionTpuCompilePolicy.precompile: warming %d patch counts %s.",
            len(patch_counts),
            patch_counts,
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
                    self.warmup_once(patch_count)
        finally:
            torch.set_default_dtype(init_dtype)

        logger.info("VisionTpuCompilePolicy.precompile: done.")

    def run(self, *raw: Any) -> torch.Tensor:
        if self.compiled_fn is None:
            raise RuntimeError("VisionTpuCompilePolicy.run called before install().")

        args, ctx = self.prepare_call(*raw)

        if not self._logged_shapes:
            self._logged_shapes = True
            logger.info(
                "VisionTpuCompilePolicy.run: compiled-fn arg shapes=%s",
                tuple(
                    tuple(a.shape) if isinstance(a, torch.Tensor) else None
                    for a in args
                ),
            )

        result = maybe_move_result_to_host(self.compiled_fn(*args))
        return self.unpad_outputs(result, ctx)

    # --- Model-specific hooks -------------------------------------------------
    # Subclasses override these to describe one vision encoder. The generic
    # lifecycle above calls them; it never reads encoder attributes directly.

    @abstractmethod
    def compute_patch_buckets(self) -> list[int]:
        """Patch-count bucket ladder for this encoder (cached by the caller).

        Owns any env override the encoder honors. Return [] to disable
        precompile.
        """

    def warmup_once(self, patch_count: int) -> None:
        """Warm one patch count, blocking until it lands.

        Defaults to the compiled region. Override when the encoder has an
        eager prologue or epilogue that is shape-static per bucket too.
        """
        result = self.compiled_fn(*self.make_warmup_inputs(patch_count))
        tpu_sync([result], wait=True)

    def make_warmup_inputs(self, patch_count: int) -> tuple[Any, ...]:
        """Compiled-fn argument tuple, for the default warmup_once.

        Reads whatever encoder state it needs (device, dtype, embedding dim)
        directly; the base holds no opinion about the encoder's module tree.
        Unneeded when a subclass overrides warmup_once.
        """
        raise NotImplementedError

    @abstractmethod
    def prepare_call(self, *raw: Any) -> tuple[tuple[Any, ...], Any]:
        """Adapt caller inputs into the compiled-fn arg tuple + a run context.

        Returns (args, ctx): args is splatted into self.compiled_fn; ctx is
        threaded through to unpad_outputs. This is the single model-specific
        seam for the compiled call -- padding, segment IDs, and any eager
        pre-embedding all happen here.
        """

    @abstractmethod
    def unpad_outputs(self, result: Any, ctx: Any) -> Any:
        """Turn the padded compiled output into what the caller expects.

        Slices the real token count back out of the bucket, plus any
        encoder-specific output adaptation.
        """

    def _runner_ready_for_compile(self, runner: Any) -> bool:
        """Whether runner can execute the compiled boundary.

        Defaults to yes. Override to gate install/precompile on an encoder
        capability, returning False to fall back to eager vision.
        """
        return True

    def _install_run_adapter(self, runner: Any) -> None:
        """Wire the serve entry point onto runner (default: no-op).

        Subclasses that must adapt the caller's run signature to self.run
        replace runner.run here.
        """


class Qwen3VLPatchEmbed(Protocol):
    in_channels: int
    temporal_patch_size: int
    patch_size: int


class Qwen3VLVisionEncoder(Protocol):
    """What Qwen3VLMoeVisionModel must expose for this policy."""

    spatial_merge_size: int
    patch_embed: Qwen3VLPatchEmbed
    dtype: torch.dtype

    def forward_with_tpu_graph(
        self, x: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor: ...


@dataclass(frozen=True)
class _Qwen3VLRunContext:
    """Run context threaded from prepare_call to unpad_outputs."""

    padding: VisionPaddingContext
    output_indices: torch.Tensor | None


class Qwen3VLVisionTpuCompilePolicy(VisionTpuCompilePolicy[Qwen3VLVisionEncoder]):
    """Compile policy for compiled Qwen3-VL ViT blocks."""

    def __init__(
        self,
        encoder: Qwen3VLVisionEncoder | None = None,
        *,
        name: str = "qwen3_vl_vit_encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder, name=name, **kwargs)

    def _runner_ready_for_compile(self, runner: Any) -> bool:
        # Without segment_ids support the blocks cannot take what the compiled
        # boundary passes them.
        return getattr(runner, "supports_segment_ids", False)

    def _install_run_adapter(self, runner: Any) -> None:
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
                output_indices,
            )

        runner.run = _run
        # Padding moved upstream to the encoder, which needs the ladder to size
        # its host-side buffers. Only reached once install succeeds.
        runner.padded_patch_buckets = tuple(self.resolve_patch_buckets())

    def compute_patch_buckets(self) -> list[int]:
        merge_size = self.encoder.spatial_merge_size
        # Set but empty disables precompile: EnvTuple drops empty entries, so the
        # override is (), which validates trivially and warms nothing.
        if envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.is_set():
            buckets = sorted(
                {int(x) for x in envs.SGLANG_TPU_VIT_PRECOMPILE_SIZES.get()}
            )
            return _validate_patch_buckets(buckets, merge_size)
        return _compute_patch_counts(
            merge_size,
            max_patches=envs.SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.get(),
        )

    def warmup_once(self, patch_count: int) -> None:
        # A square grid lands exactly on the bucket, so the eager prologue and
        # epilogue warm at the shapes a real image of this bucket will use.
        encoder = self.encoder
        patch_embed = encoder.patch_embed
        in_features = (
            patch_embed.in_channels
            * patch_embed.temporal_patch_size
            * patch_embed.patch_size**2
        )
        grid = int(math.isqrt(patch_count))
        encoder.forward_with_tpu_graph(
            torch.zeros(patch_count, in_features, dtype=encoder.dtype),
            torch.tensor([[1, grid, grid]], dtype=torch.int32),
        )

    def prepare_call(
        self,
        x_3d: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb_cos: torch.Tensor,
        rotary_pos_emb_sin: torch.Tensor,
        output_indices: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], _Qwen3VLRunContext]:
        # The caller pads to a bucket on the host, so x_3d is already bucket
        # sized; cu_seqlens still carries the real segmentation.
        bucket_len = x_3d.shape[0]
        real_seq_len = int(cu_seqlens[-1])
        merge_sq = self.encoder.spatial_merge_size**2

        # Closing the tail with one more boundary gives the padded patches a
        # segment of their own, so they cannot attend to real ones.
        padded_cu_seqlens = torch.cat([cu_seqlens, cu_seqlens.new_tensor([bucket_len])])
        segment_ids = _cu_seqlens_to_segment_ids(
            padded_cu_seqlens, bucket_len, x_3d.device
        )
        ctx = _Qwen3VLRunContext(
            padding=VisionPaddingContext(
                real_seq_len=real_seq_len,
                bucket_len=bucket_len,
                padded=bucket_len > real_seq_len,
                merge_sq=merge_sq,
            ),
            output_indices=output_indices,
        )
        return (
            (
                x_3d,
                self._graph_cu_seqlens(bucket_len, x_3d.device),
                rotary_pos_emb_cos,
                rotary_pos_emb_sin,
                segment_ids,
            ),
            ctx,
        )

    def _graph_cu_seqlens(self, bucket_len: int, device: torch.device) -> torch.Tensor:
        """Single-segment indptr for the compiled boundary, cached per bucket."""
        cached = self._graph_cu_seqlens_cache.get(bucket_len)
        if cached is None:
            cached = torch.tensor([0, bucket_len], device=device, dtype=torch.int32)
            self._graph_cu_seqlens_cache[bucket_len] = cached
        return cached

    def unpad_outputs(
        self, result: torch.Tensor, ctx: _Qwen3VLRunContext
    ) -> torch.Tensor:
        padding = ctx.padding
        if padding.padded:
            result = result[: padding.real_seq_len // padding.merge_sq]
        # Qwen3-VL's caller asks for a subset of the merged output rows.
        if ctx.output_indices is not None:
            result = result.index_select(0, ctx.output_indices.to(result.device))
        return result


class MoonVitBlock(Protocol):
    hidden_dim: int


class MoonVitRope2D(Protocol):
    def get_freqs_cis(
        self, *, grid_thws: torch.Tensor, device: torch.device
    ) -> torch.Tensor: ...


class MoonVitEncoder(Protocol):
    """What MoonViT3dEncoder must expose for this policy."""

    blocks: Sequence[MoonVitBlock]
    rope_2d: MoonVitRope2D

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...


class MoonVitTpuCompilePolicy(VisionTpuCompilePolicy[MoonVitEncoder]):
    """Compile policy for the MoonViT encoder block loop + final_layernorm.

    Only the model-specific half of the shared vision lifecycle: MoonViT's
    compiled boundary takes (x, cos, sin, segment_ids) -- no cu_seqlens, no
    spatial_merge_size -- and its temporal-pool merger runs eagerly outside the
    graph. Install, warmup and the padded run orchestration come from the base.
    """

    def __init__(
        self,
        encoder: MoonVitEncoder | None = None,
        *,
        name: str = "moonvit_vit_encoder",
        **kwargs: Any,
    ) -> None:
        super().__init__(encoder, name=name, **kwargs)
        self._warned_oversize: set[int] = set()

    def _runner_ready_for_compile(self, runner: Any) -> bool:
        return getattr(runner, "supports_segment_ids", False)

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
        encoder = self.encoder
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
            if bucket_len is None and seq_len not in self._warned_oversize:
                self._warned_oversize.add(seq_len)
                buckets = self.resolve_patch_buckets()
                logger.warning(
                    "MoonVitTpuCompilePolicy: patch count %d exceeds largest "
                    "precompiled bucket %s; one-time recompile. Raise "
                    "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES.",
                    seq_len,
                    buckets[-1] if buckets else None,
                )
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

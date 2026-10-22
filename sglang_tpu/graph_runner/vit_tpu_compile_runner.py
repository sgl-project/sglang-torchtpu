"""ViT TPU compile runner with explicit patch-bucket padding."""

from __future__ import annotations

import inspect
import logging
import math
import time
from abc import abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch._dynamo
from torch_tpu._internal.sync.sync import synchronize as tpu_sync

from sglang.srt.compilation import torch_compile_bundle
from sglang.srt.compilation.support_torch_compile import CompilePolicy
from sglang.srt.environ import envs
from sglang.srt.observability.compile_timing import CompileTimingRecorder
from sglang.srt.runtime_context import get_parallel
from sglang_tpu.compilation.pending_call import PendingCall
from sglang_tpu.patches import _register_tpu_compile_backend

logger = logging.getLogger(__name__)

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
    """Model-agnostic compile policy for TPU vision encoders.

    Owns the full compile lifecycle -- install, precompile warmup, and the
    padded ``run`` orchestration -- for any ViT encoder that precompiles fixed
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
        vit: Any | None = None,
        *,
        model_runner: Any | None = None,
        owner: Any | None = None,
        method_name: str = "forward",
        method: Any | None = None,
        name: str | None = None,
        label: str | None = None,
        compile_timing_recorder: CompileTimingRecorder | None = None,
    ) -> None:
        super().__init__(
            model_runner=model_runner,
            owner=owner,
            method_name=method_name,
            method=method,
            name=name,
            label=label,
            compile_timing_recorder=compile_timing_recorder,
        )
        if vit is None and owner is not None:
            vit = getattr(owner, "vit", owner)
        self.vit = vit
        self.compiled_fn = None
        # The uncompiled forward, saved by install(). Images larger than the
        # largest precompiled bucket run through it; see run() and
        # input_patch_count(), or MoonVitTpuCompilePolicy._run_one.
        self.eager_fn = None
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
        self.eager_fn = method
        server_args = self.model_runner.server_args
        use_bundle = (
            server_args.unsafe_save_torch_compile_bundle is not None
            or server_args.unsafe_load_torch_compile_bundle is not None
        )
        self.compiled_fn = self._compile_method(method, use_bundle=use_bundle)
        setattr(runner, self.method_name, self.compiled_fn)

        self._install_run_adapter(runner)
        logger.info(
            "VisionTpuCompilePolicy: installed %s on %s.%s.",
            self.name,
            type(runner).__name__,
            self.method_name,
        )
        return True

    def _compile_method(self, method: Any, *, use_bundle: bool) -> Any:
        if (
            self.model_runner.server_args.tpu_compile_tracer == "direct"
            and envs.SGLANG_DIRECT_COMPILE_ASYNC.get()
        ):
            model = self.get_compile_model()
            if model is not None:
                if use_bundle:
                    raise ValueError(
                        "Async direct vision compilation does not support bundles"
                    )
                from sglang_tpu.compilation.direct_compile import (
                    direct_compile,
                )

                return direct_compile(
                    torch.no_grad()(method),
                    name=self.name,
                    spmd_safe=self.spmd_safe,
                    specialization_key=lambda *args: tuple(
                        (tuple(t.shape), t.dtype) for t in args
                    ),
                    model=model,
                    lift_rank=(
                        torch.distributed.is_initialized()
                        and get_parallel().tp_group.world_size
                        == torch.distributed.get_world_size()
                    ),
                )
        compiler = torch_compile_bundle.Compiler if use_bundle else torch.compile
        return compiler(
            torch.no_grad()(method),
            backend=self.backend,
            fullgraph=self.fullgraph,
            dynamic=self.dynamic,
        )

    def get_compile_model(self) -> torch.nn.Module | None:
        """Opt into direct compilation by supplying the compiled region's state.

        All parameters and buffers read by the region must be registered here.
        The default keeps Dynamo compilation and model-specific eager warmups.
        """
        return None

    def resolve_patch_buckets(self) -> list[int]:
        if self._patch_buckets is None:
            self._patch_buckets = self.compute_patch_buckets()
        return self._patch_buckets

    def _bucket_for(self, seq_len: int) -> int | None:
        for bucket in self.resolve_patch_buckets():
            if bucket >= seq_len:
                return bucket
        return None

    def precompile(
        self, patch_counts: list[int] | None = None
    ) -> list[PendingCall] | None:
        """Warm common patch counts, returning any deferred executions."""
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
        pending_warmups = []
        try:
            with torch.no_grad():
                for index, patch_count in enumerate(patch_counts):
                    logger.info(
                        "VisionTpuCompilePolicy.precompile: [%d/%d] patch_count=%d ...",
                        index + 1,
                        len(patch_counts),
                        patch_count,
                    )
                    start = time.perf_counter()
                    pending = self.warmup_once(patch_count)
                    pending = self._record_warmup(
                        patch_count, time.perf_counter() - start, pending
                    )
                    if pending is not None:
                        pending_warmups.append(pending)
        finally:
            torch.set_default_dtype(init_dtype)

        logger.info(
            "VisionTpuCompilePolicy.precompile: submitted; %d deferred warmups.",
            len(pending_warmups),
        )
        return pending_warmups or None

    def _record_warmup(
        self, patch_count: int, duration_seconds: float, pending: PendingCall | None
    ) -> PendingCall | None:
        if self.compile_timing_recorder is None:
            return pending
        record = partial(
            self.compile_timing_recorder.record,
            component="vision",
            phase="vision_encode",
            policy=type(self).__name__,
            num_tokens=patch_count,
        )
        if pending is None:
            record(duration_seconds=duration_seconds)
            return None

        def wait_and_record():
            start = time.perf_counter()
            result = pending.wait()
            # Match decoder timing: submission plus blocking wait/execution,
            # excluding time spent submitting other overlapping warmups.
            record(duration_seconds=duration_seconds + time.perf_counter() - start)
            return result

        return PendingCall(wait_and_record)

    def run(self, *raw: Any) -> torch.Tensor:
        if self.compiled_fn is None:
            raise RuntimeError("VisionTpuCompilePolicy.run called before install().")

        args, ctx = self.prepare_call(*raw)

        patch_count = self.input_patch_count(*raw)
        if (
            patch_count is not None
            and self._bucket_for(patch_count) is None
            and self.eager_fn is not None
        ):
            # Past the top bucket, and recompiling is not allowed once warmup has
            # finished (fail_on_recompile), so run these images uncompiled.
            with torch.no_grad():
                return self.unpad_outputs(self.eager_fn(*args), ctx)

        if not self._logged_shapes:
            self._logged_shapes = True
            logger.info(
                "VisionTpuCompilePolicy.run: compiled-fn arg shapes=%s",
                tuple(
                    tuple(a.shape) if isinstance(a, torch.Tensor) else None
                    for a in args
                ),
            )

        return self.unpad_outputs(self.compiled_fn(*args), ctx)

    # --- Model-specific hooks -------------------------------------------------
    # Subclasses override these to describe one vision encoder. The generic
    # lifecycle above calls them; it never reads encoder attributes directly.

    @abstractmethod
    def compute_patch_buckets(self) -> list[int]:
        """Patch-count bucket ladder for this encoder (cached by the caller).

        Owns any env override the encoder honors. Return ``[]`` to disable
        precompile.
        """

    def warmup_once(self, patch_count: int) -> PendingCall | None:
        """Warm one patch count, deferring execution for direct compilation.

        Defaults to the compiled region. Override when the encoder has an
        eager prologue or epilogue that is shape-static per bucket too.
        """
        from sglang_tpu.compilation.direct_compile import (
            CompiledFunction,
        )

        inputs = self.make_warmup_inputs(patch_count)
        if isinstance(self.compiled_fn, CompiledFunction):
            return self.compiled_fn.warmup_async(*inputs).then(self._sync_warmup)
        self._sync_warmup(self.compiled_fn(*inputs))

    @staticmethod
    def _sync_warmup(result: torch.Tensor | tuple[torch.Tensor, ...]) -> None:
        # A boundary may return several tensors (features and a mask): land each.
        tensors = list(result) if isinstance(result, (tuple, list)) else [result]
        tpu_sync(tensors, wait=True)

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

        Returns ``(args, ctx)``: ``args`` is splatted into ``self.compiled_fn``;
        ``ctx`` is threaded through to ``unpad_outputs``. This is the single
        model-specific seam for the compiled call -- padding, segment IDs, and
        any eager pre-embedding all happen here.
        """

    @abstractmethod
    def unpad_outputs(self, result: Any, ctx: Any) -> Any:
        """Turn the padded compiled output into what the caller expects.

        Slices the real token count back out of the bucket, plus any
        encoder-specific output adaptation.
        """

    def input_patch_count(self, *raw: Any) -> int | None:
        """Patch count of a ``run`` input, so ``run`` can spot one past the ladder.

        Defaults to None, which always calls the compiled fn. When this returns
        a count no bucket fits, ``run`` calls ``eager_fn`` with
        ``prepare_call``'s arguments instead, so ``prepare_call`` must accept
        such an input unpadded.
        """
        return None

    def _runner_ready_for_compile(self, runner: Any) -> bool:
        """Whether ``runner`` can execute the compiled boundary.

        Defaults to yes. Override to gate install/precompile on an encoder
        capability, returning False to fall back to eager vision.
        """
        return True

    def _install_run_adapter(self, runner: Any) -> None:
        """Wire the serve entry point onto ``runner`` (default: no-op).

        Subclasses that must adapt the caller's ``run`` signature to
        ``self.run`` replace ``runner.run`` here.
        """


@dataclass(frozen=True)
class _Qwen3VLRunContext:
    """Run context threaded from ``prepare_call`` to ``unpad_outputs``."""

    padding: VisionPaddingContext
    output_indices: torch.Tensor | None


class Qwen3VLVisionTpuCompilePolicy(VisionTpuCompilePolicy):
    """Compile policy for compiled Qwen3-VL ViT blocks."""

    def __init__(
        self, vit: Any | None = None, *, name: str = "qwen3_vl_vit_encoder", **kwargs
    ) -> None:
        super().__init__(vit, name=name, **kwargs)

    def _runner_ready_for_compile(self, runner: Any) -> bool:
        # ``_supports_segment_ids`` is set by ``ViTTpuCompileRunner``; without it
        # the blocks cannot take the segment_ids the compiled boundary passes.
        return getattr(runner, "_supports_segment_ids", False)

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

    def input_patch_count(self, x_3d: torch.Tensor, *rest: Any) -> int:
        # The caller pads to a bucket on the host, so a length that matches no
        # bucket means the images are past the top one.
        # ViTTpuCompileRunner.bucket_for_patches has already warned.
        return x_3d.shape[0]

    def compute_patch_buckets(self) -> list[int]:
        merge_size = self.vit.spatial_merge_size
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

    def get_compile_model(self) -> torch.nn.Module | None:
        # The compiled boundary (ViTTpuCompileRunner.forward) reads the blocks,
        # merger and deepstack mergers, all registered submodules of the vision
        # tower. Supplying it opts this encoder into native async direct
        # compilation under --tpu-compile-tracer direct, so the bucket ladder
        # compiles concurrently instead of one bucket at a time.
        return self.vit if isinstance(self.vit, torch.nn.Module) else None

    def warmup_once(self, patch_count: int) -> PendingCall | None:
        from sglang_tpu.compilation.direct_compile import (
            CompiledFunction,
        )

        # A square grid lands exactly on the bucket, so the eager prologue and
        # epilogue warm at the shapes a real image of this bucket will use.
        vit = self.vit
        patch_embed = vit.patch_embed
        in_features = (
            patch_embed.in_channels
            * patch_embed.temporal_patch_size
            * patch_embed.patch_size**2
        )
        grid = int(math.isqrt(patch_count))
        x = torch.zeros(patch_count, in_features, dtype=vit.dtype)
        grid_thw = torch.tensor([[1, grid, grid]], dtype=torch.int32)
        if not isinstance(self.compiled_fn, CompiledFunction):
            vit.forward_with_tpu_graph(x, grid_thw)
            return None

        # Direct compile: run the eager prologue now, then submit the compiled
        # boundary with the exact argument tuple run() passes at serve time so
        # the specialization is reused. unpad_outputs is skipped: a square grid
        # is never trimmed, so the synchronous path does not warm it either.
        x, cu_seqlens, cos, sin = vit.prepare_tpu_graph_inputs(x, grid_thw)
        args, _ = self.prepare_call(x.unsqueeze(1), cu_seqlens, cos, sin)
        return self.compiled_fn.warmup_async(*args).then(self._sync_warmup)

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
        merge_sq = self.vit.spatial_merge_size**2

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
        # Land on the host while the shape is still the bucket: every consumer
        # from here on is variable-length and would recompile per length.
        if not envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.get():
            result = result.to("cpu")
        if padding.padded:
            result = result[: padding.real_seq_len // padding.merge_sq]
        # Qwen3-VL's caller asks for a subset of the merged output rows.
        if ctx.output_indices is not None:
            result = result.index_select(0, ctx.output_indices.to(result.device))
        return result


class ViTTpuCompileRunner:
    """Vision-encoder runner for TPU using ``torch.compile``."""

    def __init__(self, vit: Any) -> None:
        self.vit = vit
        self._compiled_fn = None

        # Published by the TPU compile policy on install; empty until then,
        # which makes bucket_for_patches the identity.
        self.padded_patch_buckets: tuple[int, ...] = ()
        self._warned_oversize_patches: set[int] = set()
        # Host mirror of the rotary cache, taken on first use: some loaders
        # build the model under a meta device, where it holds no values yet.
        self._rotary_cos_sin_cache_cpu: torch.Tensor | None = None

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
                "precompiled bucket %d; running these images through the "
                "uncompiled ViT, which is slower. Raise "
                "SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES, or lower max_pixels via "
                "--mm-process-config.",
                total_patches,
                self.padded_patch_buckets[-1],
            )
        return total_patches

    def _host_rotary_cache(self) -> torch.Tensor:
        if self._rotary_cos_sin_cache_cpu is None:
            self._rotary_cos_sin_cache_cpu = (
                self.vit.rotary_pos_emb.cos_sin_cache.detach().to("cpu", copy=True)
            )
        return self._rotary_cos_sin_cache_cpu

    def _rot_pos_emb_cpu(
        self, grid_thw: list[list[int]], *, bucket_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Host-side rot_pos_emb; cos/sin come back padded to bucket_len."""
        pos_ids_per_image = []
        for t, h, w in grid_thw:
            base = self.vit.rot_pos_ids(h, w, self.vit.spatial_merge_size)
            pos_ids_per_image.append(base if t == 1 else base.repeat(t, 1))
        pos_ids = torch.cat(pos_ids_per_image, dim=0)

        padded_pos_ids = pos_ids.new_zeros((bucket_len, pos_ids.shape[1]))
        padded_pos_ids[: pos_ids.shape[0]] = pos_ids

        max_grid_size = max(max(h, w) for _, h, w in grid_thw)
        cos, sin = self._host_rotary_cache()[:max_grid_size].chunk(2, dim=-1)
        return cos[padded_pos_ids].flatten(1), sin[padded_pos_ids].flatten(1)

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
        # The vision blocks add their residuals in place (``x += attn``). Work
        # on a copy so the compiled region never mutates its input: Dynamo
        # functionalizes that, but direct_compile only allows mutation through
        # its KV pool and aborts when an input placeholder carries deferred ops.
        x = x.clone()
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
        # Same host contract as the installed policy's unpad_outputs, so the
        # embedding cache never mixes devices across the two modes.
        if not envs.SGLANG_TPU_MM_EMBED_ON_DEVICE.get():
            result = result.to("cpu")
        if output_indices is not None:
            result = result.index_select(0, output_indices.to(result.device))
        return result

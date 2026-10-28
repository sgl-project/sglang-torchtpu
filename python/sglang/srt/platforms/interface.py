"""
SGLang SRT Hardware Platform Abstraction.

Defines SRTPlatform — the base class for SRT (LLM inference) platform
backends.  SRTPlatform inherits DeviceMixin for shared device operations
and adds SRT-specific subsystem factory methods, capability flags, and
configuration lifecycle hooks.

Out-of-tree platforms register via setuptools entry_points under the
"sglang.srt.platforms" group and should subclass SRTPlatform.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Callable, Optional, Type

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

if TYPE_CHECKING:
    import torch

    from sglang.srt.layers.quantization.base_config import QuantizationConfig

# Re-export for convenience
__all__ = ["SRTPlatform", "PlatformEnum"]


class SRTPlatform(DeviceMixin):
    """
    Base class for SRT hardware platform backends.

    Inherits device identity queries and operations from DeviceMixin.
    Adds SRT-specific factory methods, capability flags, and lifecycle hooks.

    OOT platforms should subclass SRTPlatform and override the methods
    relevant to their hardware.
    """

    # SRT-specific class-level attribute
    supported_quantization: list[str] = []

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply platform-specific default values to server arguments.

        Called after ServerArgs is parsed.
        """
        pass

    supports_mamba_extra_buffer: bool = False

    def supports_mamba_prefix_cache(self, model_config) -> bool:
        """Whether this model's recurrent backend can reuse prefix checkpoints.

        Called for models with recurrent state. Preserve the shared cache
        behavior by default; plugins may restrict their supported backends.
        """
        return True

    def validate_mamba_prefix_cache(self, view, hf_config=None) -> None:
        """Validate platform-specific recurrent prefix-cache constraints.

        ``hf_config`` is provided for extra-buffer strategies. The default
        adds no restrictions to the shared backend validation.
        """
        pass

    # ------------------------------------------------------------------
    # Subsystem factory methods
    # ------------------------------------------------------------------

    def get_default_attention_backend(self) -> str:
        """Return the default attention backend name for this platform."""
        raise NotImplementedError

    def get_default_speculative_draft_attention_backend(self, algorithm: str) -> str:
        """Return the default draft attention backend for an algorithm."""
        raise NotImplementedError

    def get_graph_runner_cls(self) -> type:
        """Return the graph runner class for this platform."""
        raise NotImplementedError

    def get_full_graph_backend_cls(self) -> type[Any]:
        """Return the full device-graph backend class for this platform."""
        return None

    def get_mha_kv_pool_cls(self) -> type:
        """Return the MHA KV pool class for this platform."""
        raise NotImplementedError

    def get_mha_kv_bytes_per_token(
        self,
        *,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        kv_cache_dtype: torch.dtype,
    ) -> Optional[int]:
        """Per-token, per-layer bytes of this platform's MHA pool; ``None`` means
        the generic ``num_kv_heads * (head_dim + v_head_dim) * itemsize``."""
        del num_kv_heads, head_dim, v_head_dim, kv_cache_dtype
        return None

    def get_mha_kv_pool_host_cls(self, device_pool) -> type:
        """Return the host (L2) MHA KV pool class matching ``device_pool``.

        Default: delegate to ``get_mha_host_pool_cls``, which picks the host
        pool class from the device pool's K/V dims (e.g. asymmetric vs. default
        layout). Platforms override this to return a device-specific host pool
        (e.g. TPU's fused host pool).
        """
        from sglang.srt.mem_cache.pool_host.mha import get_mha_host_pool_cls

        return get_mha_host_pool_cls(device_pool)

    def get_hicache_explicit_size_context(self) -> AbstractContextManager[None]:
        """Scope host-pool construction for an explicit HiCache ratio or size.

        The context covers target and draft pools. By default, each host pool
        performs its own admission check; platforms may share a budget instead.
        Automatic sizing manages its own budget and does not call this hook.
        """
        return nullcontext()

    def get_mamba_pool_cls(self) -> type:
        """Return the recurrent (GDN / mamba) state pool class for this platform."""
        from sglang.srt.mem_cache.memory_pool import MambaPool

        return MambaPool

    def get_mla_kv_pool_cls(self) -> type:
        """Return the MLA KV pool class for this platform."""
        raise NotImplementedError

    def get_dsa_kv_pool_cls(self) -> type:
        """Return the DSA KV pool class for this platform (DeepSeek V3.2)."""
        raise NotImplementedError

    def get_paged_allocator_cls(self) -> type:
        """Return the paged allocator class for this platform."""
        raise NotImplementedError

    def get_zigzag_cp_strategy_cls(self) -> type:
        """Return the zigzag prefill context-parallel strategy class."""
        from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy

        return ZigzagCPStrategy

    def runs_eager_extend_cp(self) -> bool:
        """Whether the eager runner shards a context-parallel extend itself."""
        return True

    def get_compile_backend(self, mode: str | None = None) -> str:
        """Return the compilation backend identifier.

        ``mode`` is an optional hint for the platform (e.g. "npugraph_ex").
        """
        return "inductor"

    def get_batch_input_device(self, batch) -> Optional[str]:
        """Placement of host-created forward inputs before backend preparation.

        None preserves each caller's normal device. Scheduler state and
        device-produced tokens keep their own devices.
        """
        return None

    def supports_torch_compile(self) -> bool:
        """Whether this platform consumes annotation-driven compile policies."""
        return False

    def get_traced_rank(
        self, rank: int, *, group_size: int | None = None
    ) -> torch.Tensor | None:
        """Return the traced rank, optionally within a contiguous parallel group."""
        return None

    def slice_tensor_for_rank(
        self, tensor: torch.Tensor, dim: int, parts: int, rank: int
    ) -> torch.Tensor:
        """Select one evenly divided tensor shard for this rank."""
        assert tensor.shape[dim] % parts == 0
        return tensor.split(tensor.shape[dim] // parts, dim=dim)[rank].contiguous()

    def get_piecewise_backend_cls(self) -> type:
        """Return the piecewise compilation backend class for this platform."""
        raise NotImplementedError

    def get_speculative_cache_locs_fn(
        self,
    ) -> Optional[Callable[..., torch.Tensor]]:
        """Return a platform implementation for speculative KV-cache locations."""
        return None

    def get_quantization_config(
        self, quantization: str
    ) -> Optional[Type[QuantizationConfig]]:
        """Return hardware-specific quantization config for the specific
        quantization scheme, raise an error if not supported or return None
        to use the default config."""
        return None

    # ------------------------------------------------------------------
    # Capability flags (safe conservative defaults)
    # ------------------------------------------------------------------

    def supports_fp8(self) -> bool:
        """Whether this platform supports FP8 quantization."""
        return False

    def supports_speculative_algorithm(self, algorithm: str) -> bool:
        """Whether this platform supports the named speculative algorithm."""
        return False

    def supports_speculative_draft_attention_backend(
        self, algorithm: str, backend: str
    ) -> bool:
        """Whether this platform supports a draft backend for an algorithm."""
        return False

    def supports_speculative_dp_attention(self, algorithm: str) -> bool:
        """Whether this platform serves the named speculative algorithm under DP attention.

        Consulted only where core rejects an algorithm with DP attention by default.
        """
        return False

    def support_cuda_graph(self) -> bool:
        """Whether this platform supports device graph capture and replay.
        Controls CUDA graph (CudaGraphRunner) for the decode path.
        OOT platforms that support graph-style capture should return True.
        """
        return False

    def support_piecewise_cuda_graph(self) -> bool:
        """Whether this platform supports piecewise CUDA graph.

        Controls PiecewiseCudaGraphRunner for the prefill/extend path
        (torch.compile backend).
        """
        return False

    def supports_decode_context_parallel(self) -> bool:
        """Whether this platform runs decode context parallelism; CUDA, HIP and
        NPU are accepted by initialize_model_parallel directly."""
        return False

    def dcp_shards_every_kv_pool(self) -> bool:
        """Whether decode context parallelism widens every KV allocator (SWA
        included) by attn_dcp_size and shards every layer's K/V heads across
        attn_tp_size // attn_dcp_size, not only the full-attention paged pool."""
        return False

    # TODO(yinlin09): needs an upstream seam issue; carry it as a sglang_tpu/hooks/
    # entry until the seam PR merges upstream.
    def supports_moe_dp_pipeline_stages(self) -> bool:
        """Whether a pipeline stage may replicate its experts over its own
        devices (--moe-dp-size > 1 with --pp-size > 1)."""
        return False

    def keeps_swa_index_mapping_on_host(self) -> bool:
        """Whether the SWA allocator's full-to-SWA index mapping is host-resident,
        so SWA segment frees require host indices."""
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """One-time backend initialization.  Called in each worker."""
        pass

    def on_warmup_complete(self) -> None:
        """Called once on the scheduler thread, the first time it goes idle after
        a forward (the server warmup request has drained). Platforms that
        compile ahead of serving freeze or monitor compilation here."""
        pass

    def get_compilation_cache_misses(self) -> Optional[int]:
        """Return a monotonic platform compilation-cache miss counter.

        A platform may return ``None`` when its runtime exposes no such
        counter. Implementations should not synchronize device execution.
        """
        return None

    def get_recompilation_counters(self) -> dict[str, Any]:
        """Return platform-specific serving-time compilation counters for internal state readback."""
        return {}

    # TODO(yinlin09): needs an upstream seam issue; carry it as a sglang_tpu/hooks/
    # entry until the seam PR merges upstream.
    def prepare_model_parallel_groups(self) -> None:
        """Called on every rank once the model-parallel groups are built,
        before their first collective."""
        pass

    # ------------------------------------------------------------------
    # BaseFusedOp integration
    # ------------------------------------------------------------------

    def get_dispatch_key_name(self) -> str:
        """Return the dispatch key name for BaseFusedOp
        (``sglang.kernels.fused_op``).

        Determines which ``forward_<key>()`` method is selected on an
        out-of-tree platform. E.g. "cuda", "npu", "hip", "xpu", "cpu".
        Forwards registered via ``BaseFusedOp.register_oot_forward`` with
        this key take precedence over the method lookup.
        """
        return "native"

"""
SGLang SRT Hardware Platform Abstraction.

Defines SRTPlatform — the base class for SRT (LLM inference) platform
backends.  SRTPlatform inherits DeviceMixin for shared device operations
and adds SRT-specific subsystem factory methods, a capability declaration,
and configuration lifecycle hooks.

Out-of-tree platforms register via setuptools entry_points under the
"sglang.srt.platforms" group and should subclass SRTPlatform.
"""

from __future__ import annotations

import logging
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Type

import msgspec

from sglang.srt.platforms.device_mixin import DeviceMixin, PlatformEnum

if TYPE_CHECKING:
    import torch

    from sglang.srt.compilation.helper_compile import HelperCompilePolicy
    from sglang.srt.layers.quantization.base_config import QuantizationConfig

# Re-export for convenience
__all__ = [
    "KVPoolKind",
    "PlatformCapabilities",
    "SRTPlatform",
    "PlatformEnum",
    "require_out_of_tree_impl",
    "reject_out_of_tree_path",
]

logger = logging.getLogger(__name__)

KVPoolKind = Literal["mha", "mla", "dsa"]

_LEGACY_KV_POOL_HOOKS: dict[str, str] = {
    "mha": "get_mha_kv_pool_cls",
    "mla": "get_mla_kv_pool_cls",
    "dsa": "get_dsa_kv_pool_cls",
}
_warned_legacy_kv_pool_hooks: set[tuple[type, str]] = set()


class PlatformCapabilities(msgspec.Struct, frozen=True, kw_only=True):
    """What the platform can do.

    Core reads these before it constructs anything and picks a code path
    the platform can serve, instead of announcing a decision and asking the
    platform to honor it. Every field defaults to the conservative answer.

    ``supports_triton``: Triton kernels can launch on this device. When
    False, core uses the torch-native allocator and req-to-token writers.
    ``graph_capture``: the decode graph runner (``get_graph_runner_cls``)
    runs. ``piecewise_graph``: the prefill piecewise compilation backend
    runs.
    ``hicache_device_kernels``: the sgl_kernel HiCache transfer / write-back
    kernels are available; otherwise the host pools skip their staging
    buffers.
    """

    supports_triton: bool = False
    graph_capture: bool = False
    piecewise_graph: bool = False
    hicache_device_kernels: bool = False
    # Fork-local fields below; each default is today's non-TPU behaviour.
    # DP gather/scatter supports only equal per-rank blocks (DpPaddingMode.MAX_LEN).
    uniform_dp_padding: bool = False
    # Compiled programs must be identical on every rank; no rank-dependent branches.
    # Requires uniform_dp_padding: rank-uniform gather/scatter use equal slots.
    rank_uniform_graphs: bool = False
    # Device tensors can be sent point-to-point without a host copy.
    device_p2p: bool = True
    # Dynamo can trace raw torch.distributed (c10d functional) collectives.
    traces_raw_collectives: bool = True

    def __post_init__(self):
        if self.rank_uniform_graphs and not self.uniform_dp_padding:
            raise ValueError("rank_uniform_graphs requires uniform_dp_padding")


class SRTPlatform(DeviceMixin):
    """
    Base class for SRT hardware platform backends.

    Inherits device identity queries and operations from DeviceMixin.
    Adds SRT-specific factory methods, a capability declaration, and
    lifecycle hooks.

    OOT platforms subclass SRTPlatform and override the methods relevant to
    their hardware.
    """

    # SRT-specific class-level attributes
    supported_quantization: list[str] = []
    capabilities: PlatformCapabilities = PlatformCapabilities()

    # ------------------------------------------------------------------
    # Configuration lifecycle
    # ------------------------------------------------------------------

    def apply_server_args_defaults(self, server_args) -> None:
        """Apply platform-specific default values to server arguments.

        Called after ServerArgs is parsed.
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

    def initialize_scheduler_worker(
        self, server_args, tp_rank, pp_rank=0, dp_rank=None
    ):
        """Initialize backend process state before worker plugin/model loading."""
        pass

    def get_scheduler_extension_cls(self) -> type:
        from sglang.srt.managers.scheduler_components.backend_extension import (
            SchedulerExtension,
        )

        return SchedulerExtension

    def get_graph_runner_cls(self) -> Optional[type]:
        """Return the graph runner class, or None for the in-tree default.

        ``None`` means "no platform opinion": the caller falls back to the
        in-tree device-keyed selection.
        """
        return None

    def get_full_graph_backend_cls(self) -> type[Any]:
        """Return the full device-graph backend class for this platform."""
        return None

    def get_kv_pool_cls(self, *, kind: KVPoolKind) -> Optional[type]:
        """Return this platform's KV pool class for ``kind``, or None.

        The class must subclass the in-tree pool for that kind
        (``MHATokenToKVPool`` / ``MLATokenToKVPool`` / ``DSATokenToKVPool``)
        and is constructed by the configurator with exactly the keyword
        arguments the in-tree class receives, at every site the in-tree
        class is built (standalone, the SWA composite, the hybrid-linear
        full-attention leaf). Override ``_create_buffers`` to change how the
        backing storage is carved; the rest of the pool (addressing, index
        translation, PD registration) stays core's. ``None`` (the default)
        means the in-tree class, which allocates on ``device`` and is
        correct for any torch device.

        A platform that still overrides the removed
        ``get_{mha,mla,dsa}_kv_pool_cls`` hooks is served from them, with a
        deprecation warning.
        """
        hook = _LEGACY_KV_POOL_HOOKS[kind]
        legacy = getattr(self, hook, None)
        if legacy is None:
            return None
        key = (type(self), hook)
        if key not in _warned_legacy_kv_pool_hooks:
            _warned_legacy_kv_pool_hooks.add(key)
            logger.warning(
                "%s.%s() is deprecated and will be removed; override "
                "get_kv_pool_cls(kind=%r) instead.",
                type(self).__name__,
                hook,
                kind,
            )
        return legacy()

    def get_paged_allocator_cls(self) -> Optional[type]:
        """Return the paged allocator class, or None for the in-tree default.

        Honored at every paged-allocator construction site, including
        inside ``SWATokenToKVPoolAllocator``. The in-tree allocator already
        falls back to torch-native kernels when ``capabilities.supports_triton``
        is False, so most platforms need no override.
        """
        return None

    def get_compile_backend(self, mode: str | None = None) -> str:
        """Return the compilation backend identifier.

        ``mode`` is an optional hint for the platform (e.g. "npugraph_ex").
        """
        return "inductor"

    def get_piecewise_backend_cls(self) -> Optional[type]:
        """Return the piecewise compilation backend class, or None for the
        in-tree device-keyed default."""
        return None

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
    # Speculative decoding support (safe conservative defaults)
    # ------------------------------------------------------------------

    def supports_speculative_algorithm(self, algorithm: str) -> bool:
        """Whether this platform supports the named speculative algorithm."""
        return False

    def supports_speculative_draft_attention_backend(
        self, algorithm: str, backend: str
    ) -> bool:
        """Whether this platform supports a draft backend for an algorithm."""
        return False

    def support_mamba_cache_extra_buffer(self) -> bool:
        """Whether this out-of-tree platform supports Mamba cache snapshots."""
        return False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def init_backend(self) -> None:
        """One-time backend initialization.  Called in each worker."""
        pass

    def post_load_model(self, model: torch.nn.Module) -> None:
        """Called once after the model's weights are loaded, in the worker.

        The place to relocate parameters, swap quant methods, or install
        forward hooks on an in-tree model class; mutate ``model`` in place.
        """
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

    def get_default_vision_attention_backend(self) -> Optional[str]:
        """Default multimodal (ViT) attention backend; None keeps the built-in
        per-device choice in ``VisionAttention._determine_attention_backend``."""
        return None

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

    def get_zigzag_cp_strategy_cls(self) -> type:
        """Return the zigzag prefill context-parallel strategy class."""
        from sglang.srt.layers.cp.zigzag import ZigzagCPStrategy

        return ZigzagCPStrategy

    def runs_eager_extend_cp(self) -> bool:
        """Whether the eager runner shards a context-parallel extend itself."""
        return True

    def get_batch_input_device(self, batch) -> Optional[str]:
        """Placement of host-created forward inputs before backend preparation.

        None preserves each caller's normal device. Scheduler state and
        device-produced tokens keep their own devices.
        """
        return None

    def supports_torch_compile(self) -> bool:
        """Whether this platform consumes annotation-driven compile policies."""
        return False

    def get_helper_compile_policy(self) -> HelperCompilePolicy:
        """How standalone tensor helpers compile; the default is torch.compile."""
        from sglang.srt.compilation.helper_compile import DEFAULT_HELPER_COMPILE_POLICY

        return DEFAULT_HELPER_COMPILE_POLICY

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

    def get_spec_worker_cls(self, algorithm, server_args, *, default: type) -> type:
        """Return the worker class for a builtin speculative algorithm.

        ``default`` is the class core selects for ``algorithm``. A platform may
        return a different class satisfying ``SpecWorkerContract``
        (``speculative/spec_worker_contract.py``) that serves the same algorithm.
        """
        return default

    def supports_decode_context_parallel(self) -> bool:
        """Whether this platform runs decode context parallelism; CUDA, HIP and
        NPU are accepted by initialize_model_parallel directly."""
        return False

    def dcp_shards_every_kv_pool(self) -> bool:
        """Whether decode context parallelism widens every KV allocator (SWA
        included) by attn_dcp_size and shards every layer's K/V heads across
        attn_tp_size // attn_dcp_size, not only the full-attention paged pool."""
        return False

    def supports_moe_dp_pipeline_stages(self) -> bool:
        """Whether a pipeline stage may replicate its experts over its own
        devices (--moe-dp-size > 1 with --pp-size > 1)."""
        return False

    def keeps_swa_index_mapping_on_host(self) -> bool:
        """Whether the SWA allocator's full-to-SWA index mapping is host-resident,
        so SWA segment frees require host indices."""
        return False

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

    def prepare_model_parallel_groups(self) -> None:
        """Called on every rank once the model-parallel groups are built,
        before their first collective."""
        pass


def require_out_of_tree_impl(
    platform: SRTPlatform, *, hook: str, subsystem: str
) -> None:
    """Reject "no platform opinion" from an out-of-tree platform; no-op in-tree."""
    if not platform.is_out_of_tree():
        return
    raise NotImplementedError(
        f"Out-of-tree platform {type(platform).__name__} (device "
        f"{platform.device_name!r}) provides no {subsystem}. Override "
        f"{hook} on your SRTPlatform subclass: the in-tree fallback is "
        f"device-keyed and would hand this device the CUDA implementation."
    )


def reject_out_of_tree_path(platform: SRTPlatform, *, subsystem: str) -> None:
    """Reject an in-tree-only path that has no platform seam; no-op in-tree."""
    if not platform.is_out_of_tree():
        return
    raise NotImplementedError(
        f"Out-of-tree platform {type(platform).__name__} (device "
        f"{platform.device_name!r}) reached {subsystem}, which has no platform "
        f"seam yet and would otherwise build in-tree classes that assume CUDA."
    )

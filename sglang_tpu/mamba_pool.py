# SPDX-License-Identifier: Apache-2.0
"""TPU-native recurrent (GDN / mamba) state pool.

The GPU :class:`~sglang.srt.mem_cache.memory_pool.MambaPool` allocates its
conv/ssm buffers with CUDA-specific machinery (memory-saver regions, custom
NVLink mem pools, fp8 quirks, CUDA-only speculative buffers) and in the layout
expected by the Triton GDN kernels. This subclass reallocates the two state
buffers on the TPU device in the layout expected by the fused Pallas kernel
:func:`sglang_tpu.attention.pallas.gdn.fused_conv1d_gdn`:

* ``conv``:      ``[size + 1, kernel_size - 1, conv_dim]``    (one per layer)
* ``recurrent``: ``[size + 1, num_v_heads, d_k, d_v]``        (one per layer)

versus the GPU layouts ``[..., conv_dim, kernel_size - 1]`` and
``[..., num_v_heads, d_v, d_k]``. Allocating kernel-native here means the TPU
GDN backend passes the per-layer table straight to the op with no per-forward
transpose.


Slot 0 is a reserved null block (padded / invalid tokens); real slots are
``1 .. size``. The kernel masks reads on fresh requests via
``has_initial_state`` (derived from ``seq_lens - query_lens``), so slot reuse
needs **no** explicit clearing — hence ``clear_slots`` is a no-op here and no
dynamic in-place scatter is required on device.

Selected by ``TpuSRTPlatform.get_mamba_pool_cls`` and instantiated from
``HybridReqToTokenPool._init_mamba_pool``.
"""

import logging
from typing import List, Optional

import msgspec
import torch

from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.mem_cache.memory_pool import MambaPool, get_tensor_size_bytes
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

GB = 1024**3


class TpuPerLayerState(msgspec.Struct, frozen=True):
    """GDN state held as one independent tensor per layer.

    ``conv[kind][layer]`` and ``temporal[layer]`` each own their whole buffer, so
    :meth:`at_layer_idx` hands the Pallas bridge a real tensor rather than a view
    into a stacked pool. That is the entire purpose of this container -- see the
    module docstring for what the view costs.

    Presents the same ``at_layer_idx`` / ``mem_usage_bytes`` surface as
    :class:`MambaPool.State` and returns a plain ``MambaPool.State`` per layer, so
    every consumer downstream of ``mamba2_layer_cache`` is unchanged.
    """

    conv: List[List[torch.Tensor]]  # [state_kind][layer]
    temporal: List[torch.Tensor]  # [layer]

    def at_layer_idx(self, layer: int) -> "MambaPool.State":
        # A list index, not a tensor slice: the result owns its storage.
        return MambaPool.State(
            conv=[per_layer[layer] for per_layer in self.conv],
            temporal=self.temporal[layer],
        )

    def mem_usage_bytes(self) -> int:
        # get_tensor_size_bytes recurses through nested lists.
        return get_tensor_size_bytes(self.conv) + get_tensor_size_bytes(self.temporal)


class TpuMambaPool(MambaPool):
    """MambaPool variant that allocates kernel-native state buffers on TPU.

    Only the state *buffers* (``self.mamba_cache``) live here; per-request slot
    allocation is handled separately by ``MambaSlotAllocator``.
    """

    def __init__(
        self,
        *,
        size: int,
        spec_state_size: int,
        cache_params: BaseLinearStateParams,
        mamba_layer_ids: List[int],
        device: str,
        enable_memory_saver: bool = False,
        speculative_num_draft_tokens: Optional[int] = None,
        speculative_eagle_topk: Optional[int] = None,
        enable_linear_replayssm: bool = False,
        linear_replayssm_cache_len: int = 16,
        envelope_layout: bool = False,
        **kwargs,
    ):
        # NOTE: deliberately does NOT call super().__init__ — the base
        # allocator is CUDA/NPU/CPU-specific. We reproduce only the pieces the
        # TPU GDN path needs.
        if speculative_num_draft_tokens is not None:
            raise NotImplementedError(
                "Speculative decoding is not supported for the TPU GDN state pool yet."
            )
        if enable_linear_replayssm:
            raise NotImplementedError(
                "GDN ReplaySSM ring buffers are not supported on the TPU GDN "
                "state pool yet."
            )
        if envelope_layout:
            raise NotImplementedError(
                "Page-major envelope mamba layout is CUDA-only; not supported on TPU."
            )

        conv_state_shapes = cache_params.shape.conv  # list of (conv_dim, K-1)
        # temporal (GPU layout): (num_v_heads, head_dim=d_v, state_size=d_k)
        n_v, d_v, d_k = cache_params.shape.temporal

        num_mamba_layers = len(mamba_layer_ids)
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )
        self.size = size
        self.device = device
        self.num_mamba_layers = num_mamba_layers
        # TPU does not use NVLink custom mem pools.
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        # ReplaySSM is disabled on TPU (guarded above), but the shared paths
        # read these unconditionally -- HybridReqToTokenPool.alloc does a bare
        # ``self.mamba_pool.replayssm_cache_base is not None`` on every extend,
        # so a missing attr is an AttributeError on the first request rather
        # than a disabled feature. Mirror the disabled-state attrs the unified
        # pool sets for the same reason (see unified_memory_pool.py).
        self.enable_linear_replayssm = False
        self.enable_linear_replayssm_spec = False
        self.linear_replayssm_cache_len = linear_replayssm_cache_len
        self.replayssm_is_kda = False
        self.replayssm_write_pos = None
        self.replayssm_spec_write_pos = None
        self.replayssm_spec_fold = False
        self.replayssm_cache_base = None
        self.replayssm_is_flush = None

        # ssm (recurrent) fp32 is NOT a TPU-specific choice: mainline defaults
        # the ssm state to fp32 too (mamba_utils.get_mamba2_state_dtype), since
        # the recurrent state accumulates over the sequence and drifts in bf16.
        # conv fp32 IS the only TPU-specific deviation (mainline defaults conv
        # to bf16): the fused GDN kernel upcasts the conv input to fp32
        # internally regardless, so a bf16 conv store buys memory but no
        # accuracy — deferred as a later memory optimization.
        conv_dtype = torch.float32
        ssm_dtype = torch.float32

        # One tensor per layer, never a stacked [num_layers, ...] buffer -- see
        # the module docstring. Total bytes are identical; what changes is that
        # each layer's state owns its storage, so the Pallas bridge's write-back
        # aliases in place instead of lowering to slice + dynamic-update-slice.
        #
        # conv: kernel-native [size+1, K-1, conv_dim] per layer
        # (GPU stores [..., conv_dim, K-1]; we swap the trailing two axes).
        conv_state = [
            [
                torch.zeros(
                    (size + 1, conv_dim_km1[1], conv_dim_km1[0]),
                    dtype=conv_dtype,
                    device=device,
                )
                for _ in range(num_mamba_layers)
            ]
            for conv_dim_km1 in conv_state_shapes
        ]

        # recurrent: kernel-native [size+1, n_v, d_k, d_v] per layer
        # (GPU stores [..., n_v, d_v, d_k]; we swap the trailing two axes).
        temporal_state = [
            torch.zeros(
                (size + 1, n_v, d_k, d_v),
                dtype=ssm_dtype,
                device=device,
            )
            for _ in range(num_mamba_layers)
        ]

        self.mamba_cache = TpuPerLayerState(conv=conv_state, temporal=temporal_state)
        self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB
        logger.info(
            "TPU Mamba/GDN state cache allocated: max_size=%d, layers=%d, "
            "per-layer conv=%s%s, per-layer recurrent=%s (mem=%.3fGB)",
            size,
            num_mamba_layers,
            list(conv_state[0][0].shape),
            "" if len(conv_state) == 1 else f" x{len(conv_state)}",
            list(temporal_state[0].shape),
            self.mem_usage,
        )

    def clear_slots(self, indices: torch.Tensor):
        """No-op on TPU.

        The fused GDN kernel zeroes a slot's contribution whenever
        ``has_initial_state`` is False (a fresh request), so reused slots need
        no explicit clearing and we avoid a dynamic in-place device scatter.
        Note that the masking is derived from ``query_lens``, so anything that
        corrupts ``query_start_loc`` makes the predicate read True for a fresh
        request and resurrects the previous occupant's state.
        """
        return

    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        raise NotImplementedError(
            "Recurrent-state copy-on-write (radix/prefix cache) is not "
            "supported on TPU yet. Radix cache is auto-disabled for hybrid "
            "GDN models on TPU (see kv_cache_builder.build_kv_cache); reaching "
            "here means prefix caching was force-enabled for this pool."
        )

    # The four below are inherited from MambaPool, which indexes the layer axis
    # as `state[:, slots]` on a stacked tensor. This pool has no layer axis to
    # index -- state is one tensor per layer -- so the inherited versions would
    # silently address the *slot* axis instead. The two RDMA ones additionally
    # walk `vars(self.mamba_cache)`, which a msgspec.Struct does not support.
    # None is reachable on TPU today (host/hierarchical cache and RDMA
    # disaggregation are both unsupported here), so they raise rather than
    # pretend.

    def get_cpu_copy(self, indices):
        raise NotImplementedError(
            "Host-offload of GDN state is not supported on TPU: TpuMambaPool "
            "stores one tensor per layer, not a stacked [num_layers, ...] pool, "
            "so MambaPool.get_cpu_copy's `[:, indices]` does not apply."
        )

    def load_cpu_copy(self, mamba_cache_cpu, indices):
        raise NotImplementedError(
            "Host-offload of GDN state is not supported on TPU: TpuMambaPool "
            "stores one tensor per layer, not a stacked [num_layers, ...] pool, "
            "so MambaPool.load_cpu_copy's `[:, indices]` does not apply."
        )

    def get_contiguous_buf_infos(self):
        raise NotImplementedError(
            "RDMA registration of GDN state is not supported on TPU: "
            "TpuMambaPool stores one tensor per layer, so the per-layer strides "
            "MambaPool.get_contiguous_buf_infos derives from a stacked buffer "
            "do not exist."
        )

    def get_state_dim_per_tensor(self):
        raise NotImplementedError(
            "RDMA registration of GDN state is not supported on TPU; see "
            "TpuMambaPool.get_contiguous_buf_infos. The TP-sliceable axis this "
            "reports as dim 2 of a stacked buffer is dim 1 of a per-layer one."
        )

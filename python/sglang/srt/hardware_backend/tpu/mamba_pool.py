# SPDX-License-Identifier: Apache-2.0
"""TPU-native recurrent (GDN / mamba) state pool.

The GPU :class:`~sglang.srt.mem_cache.memory_pool.MambaPool` allocates its
conv/ssm buffers with CUDA-specific machinery (memory-saver regions, custom
NVLink mem pools, fp8 quirks, CUDA-only speculative buffers) and in the layout
expected by the Triton GDN kernels. This subclass reallocates the two state
buffers on the TPU device in the layout expected by the fused Pallas kernel
:func:`sglang.srt.layers.attention.pallas.gdn.fused_conv1d_gdn`:

* ``conv``:      ``[num_layers, size + 1, kernel_size - 1, conv_dim]``
* ``recurrent``: ``[num_layers, size + 1, num_v_heads, d_k, d_v]``

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

import torch

from sglang.srt.configs.mamba_utils import BaseLinearStateParams
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

GB = 1024**3


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
                "Speculative decoding is not supported for the TPU GDN state "
                "pool yet."
            )
        if enable_linear_replayssm:
            raise NotImplementedError(
                "GDN ReplaySSM ring buffers are not supported on the TPU GDN "
                "state pool yet."
            )
        if envelope_layout:
            raise NotImplementedError(
                "Page-major envelope mamba layout is CUDA-only; not supported "
                "on TPU."
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
        # ReplaySSM is disabled on TPU (guarded above); keep the attrs the
        # shared backend-metadata path reads.
        self.enable_linear_replayssm = False
        self.linear_replayssm_cache_len = linear_replayssm_cache_len
        self.replayssm_is_kda = False
        self.replayssm_write_pos = None

        # ssm (recurrent) fp32 is NOT a TPU-specific choice: mainline defaults
        # the ssm state to fp32 too (mamba_utils.get_mamba2_state_dtype), since
        # the recurrent state accumulates over the sequence and drifts in bf16.
        # conv fp32 IS the only TPU-specific deviation (mainline defaults conv
        # to bf16): the fused GDN kernel upcasts the conv input to fp32
        # internally regardless, so a bf16 conv store buys memory but no
        # accuracy — deferred as a later memory optimization.
        conv_dtype = torch.float32
        ssm_dtype = torch.float32

        # conv: kernel-native [num_layers, size+1, K-1, conv_dim]
        # (GPU stores [..., conv_dim, K-1]; we swap the trailing two axes).
        conv_state = [
            torch.zeros(
                (num_mamba_layers, size + 1, conv_dim_km1[1], conv_dim_km1[0]),
                dtype=conv_dtype,
                device=device,
            )
            for conv_dim_km1 in conv_state_shapes
        ]

        # recurrent: kernel-native [num_layers, size+1, n_v, d_k, d_v]
        # (GPU stores [..., n_v, d_v, d_k]; we swap the trailing two axes).
        temporal_state = torch.zeros(
            (num_mamba_layers, size + 1, n_v, d_k, d_v),
            dtype=ssm_dtype,
            device=device,
        )

        self.mamba_cache = self.State(conv=conv_state, temporal=temporal_state)
        self.mem_usage = self.mamba_cache.mem_usage_bytes() / GB
        logger.info(
            "TPU Mamba/GDN state cache allocated: max_size=%d, layers=%d, "
            "conv=%s%s, recurrent=%s (mem=%.3fGB)",
            size,
            num_mamba_layers,
            list(conv_state[0].shape),
            "" if len(conv_state) == 1 else f" x{len(conv_state)}",
            list(temporal_state.shape),
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

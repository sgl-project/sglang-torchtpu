# SPDX-License-Identifier: Apache-2.0
"""TPU-native recurrent (GDN / mamba) state pool.

The GPU :class:`~sglang.srt.mem_cache.memory_pool.MambaPool` allocates its
conv/ssm buffers with CUDA-specific machinery (memory-saver regions, custom
NVLink mem pools, fp8 quirks, CUDA-only speculative buffers) and in the layout
expected by the Triton GDN kernels. This subclass reallocates the two state
buffers on the TPU device in the layout expected by the fused Pallas kernel
:func:`sglang_tpu.attention.pallas.gdn.fused_conv1d_gdn`:

* ``conv``:      ``[size + 1, kernel_size - 1, 1, conv_dim]`` (one per layer)
* ``recurrent``: ``[size + 1, num_v_heads, d_k, d_v]``        (one per layer)

versus the GPU layouts ``[..., conv_dim, kernel_size - 1]`` and
``[..., num_v_heads, d_v, d_k]``. Allocating kernel-native here means the TPU
GDN backend passes the per-layer table straight to the op with no per-forward
transpose.


Slot 0 is reserved for padding and real slots are ``1 .. size``, one state row
each. With speculation, rows past the slots hold one ``checkpoint_width`` window
per request pool row, one state per verify input, so radix snapshots cost one
row as core's mamba budget assumes. A per-slot read offset names the committed
state: ``AT_SLOT`` is the slot's own row (after extend, a radix copy or a P/D
transfer), and ``t >= 0`` is row ``t`` of the request's window (after a verify).

The kernel masks reads on fresh requests via ``has_initial_state`` (derived
from ``seq_lens - query_lens``), so slot reuse needs no full-state clearing.
``clear_slots`` only resets the small checkpoint-offset vector when present.

Selected by ``TpuSRTPlatform.get_mamba_pool_cls`` and instantiated from
``HybridReqToTokenPool._init_mamba_pool``.
"""

import logging
import math
from typing import List, Optional

import msgspec
import torch

from sglang.srt.compilation.compiled_op import compiled_op
from sglang.srt.configs.mamba_utils import (
    BaseLinearStateParams,
    KimiLinearStateShape,
)
from sglang.srt.mem_cache.memory_pool import MambaPool, get_tensor_size_bytes
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

GB = 1024**3

# Read offset of a slot whose committed state is its own row, not a window row.
AT_SLOT = -1


@compiled_op()
def committed_read_offsets(read_offsets, indices, steps):
    """(int64 slots, int32 read offsets) select_checkpoints commits, in one
    launch. Negative steps (inactive requests) and the null slot keep their
    previous offset."""
    indices = indices.to(torch.int64)
    steps = steps.to(torch.int32)
    previous = read_offsets.index_select(0, indices)
    return indices, torch.where((indices > 0) & (steps >= 0), steps, previous)


@compiled_op()
def verify_track_steps(committed_lens, last_steps, interval: int):
    """Window step whose checkpoint lands on a track-interval boundary, or -1.

    committed_lens are the lengths before the verify. Matches core's post-verify
    commit (spec_utils._verify_commit_step_indices) for linear chains, where the
    window step is the accepted token's index.
    """
    pre = committed_lens.to(torch.int64)
    post = pre + last_steps.to(torch.int64) + 1
    crossed = pre // interval != post // interval
    return torch.where(crossed, post // interval * interval - pre - 1, -1)


@compiled_op()
def checkpoint_snapshot_pairs(
    track_slots, req_pool_indices, steps, window_start: int, width: int, bucket: int
):
    """(src rows, dst slots, count) copying window checkpoint steps into track
    slots, valid pairs first and padded to bucket. A negative step or a track
    slot <= 0 (no buffer, or padding) skips its row."""
    valid = (steps >= 0) & (track_slots > 0)
    order = torch.argsort((~valid).to(torch.int32), stable=True)
    rows = window_start + req_pool_indices.clamp(min=0) * width + steps.clamp(min=0)
    src = rows.to(torch.int32).index_select(0, order)
    dst = track_slots.to(torch.int32).index_select(0, order)
    padding = bucket - src.shape[0]
    src = torch.nn.functional.pad(src, (0, padding))
    dst = torch.nn.functional.pad(dst, (0, padding))
    return src, dst, valid.sum(dtype=torch.int32).reshape(1)


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
        self.checkpoint_width = speculative_num_draft_tokens or 1
        if self.checkpoint_width < 1:
            raise ValueError("GDN checkpoint width must be positive")
        if self.checkpoint_width > 1 and (speculative_eagle_topk or 1) != 1:
            raise NotImplementedError("TPU GDN supports linear-chain speculation only")
        self.spec_state_size = spec_state_size
        if self.checkpoint_width > 1:
            # Windows are indexed by request pool row, 0 .. spec_state_size.
            self.window_start: Optional[int] = size + 1
            state_slots = (
                self.window_start + (spec_state_size + 1) * self.checkpoint_width
            )
            self.checkpoint_read_offsets = torch.full(
                (size + 1,), AT_SLOT, dtype=torch.int32, device=device
            )
        else:
            self.window_start = None
            state_slots = size + 1
            self.checkpoint_read_offsets = None
        if enable_linear_replayssm:
            raise NotImplementedError(
                "GDN ReplaySSM ring buffers are not supported on the TPU GDN "
                "state pool yet."
            )
        if envelope_layout:
            raise NotImplementedError(
                "Page-major envelope mamba layout is CUDA-only; not supported on TPU."
            )

        # (conv_dim, K-1) per conv kind; Kimi already stores (K-1, conv_dim).
        conv_state_shapes = cache_params.shape.conv
        if isinstance(cache_params.shape, KimiLinearStateShape):
            conv_state_shapes = [(dim, km1) for km1, dim in conv_state_shapes]
        # temporal (GPU layout): (num_v_heads, head_dim=d_v, state_size=d_k)
        n_v, d_v, d_k = cache_params.shape.temporal

        num_mamba_layers = len(mamba_layer_ids)
        # Read by the inherited P/D state-registration helpers.
        self.mamba_layer_ids = list(mamba_layer_ids)
        self.conv_shard_groups = cache_params.shape.conv_shard_groups
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
        # conv: kernel-native [size+1, K-1, 1, conv_dim] per layer (GPU stores
        # [..., conv_dim, K-1]). The kernel reshapes the pool to this rank-4
        # form, which on TPU re-tiles the whole pool twice per layer call
        # unless it is stored so. Same bytes as [size+1, K-1, conv_dim].
        conv_state = [
            [
                torch.zeros(
                    (state_slots, conv_dim_km1[1], 1, conv_dim_km1[0]),
                    dtype=conv_dtype,
                    device=device,
                )
                for _ in range(num_mamba_layers)
            ]
            for conv_dim_km1 in conv_state_shapes
        ]

        # recurrent: kernel-native [state_slots, n_v, d_k, d_v] per layer
        # (GPU stores [..., n_v, d_v, d_k]; we swap the trailing two axes).
        temporal_state = [
            torch.zeros(
                (state_slots, n_v, d_k, d_v),
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

    @property
    def slot_allocator_device(self):
        # Allocation, radix ownership and COW pair collection are host work.
        # Keeping IDs here avoids variable-size TPU casts/pads and host readbacks.
        return "cpu"

    @classmethod
    def cache_bytes_per_request(cls, cache_params):
        # Both TPU state regions use fp32, including the conv taps.
        elements = sum(math.prod(shape) for shape in cache_params.shape.conv)
        elements += math.prod(cache_params.shape.temporal)
        return elements * 4 * len(cache_params.layers)

    def get_speculative_mamba2_params_all_layers(self) -> TpuPerLayerState:
        """Verify checkpoints live in the main cache."""
        return self.mamba_cache

    def select_checkpoints(self, indices, last_correct_step_indices):
        """Record the state after the final accepted verify input, without copying HBM."""
        if self.checkpoint_read_offsets is None:
            raise RuntimeError("GDN checkpoint storage was not allocated")
        indices, offsets = committed_read_offsets(
            self.checkpoint_read_offsets,
            indices.to(self.device),
            last_correct_step_indices.to(self.device),
        )
        self.checkpoint_read_offsets.index_copy_(0, indices, offsets)

    def clear_slots(self, indices: torch.Tensor):
        """Point the slots' committed state back at their own rows.

        The fused GDN kernel zeroes a slot's contribution whenever
        ``has_initial_state`` is False (a fresh request), so reused slots need
        no full-state clearing. Only the small checkpoint-offset vector is reset.
        Note that the masking is derived from ``query_lens``, so anything that
        corrupts ``query_start_loc`` makes the predicate read True for a fresh
        request and resurrects the previous occupant's state.
        """
        if self.checkpoint_read_offsets is not None:
            self.checkpoint_read_offsets.index_fill_(
                0, indices.to(device=self.device, dtype=torch.int64), AT_SLOT
            )

    def window_bases(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        """First state row of each request's checkpoint window."""
        assert self.window_start is not None, "no checkpoint windows without spec"
        return self.window_start + req_pool_indices.clamp(min=0) * self.checkpoint_width

    def committed_state_rows(
        self, slots: torch.Tensor, req_pool_indices: torch.Tensor
    ) -> torch.Tensor:
        """State row holding each slot's committed state."""
        if self.checkpoint_read_offsets is None:
            return slots
        offsets = self.checkpoint_read_offsets.index_select(0, slots.to(torch.int64))
        return torch.where(
            offsets < 0, slots, self.window_bases(req_pool_indices) + offsets
        )

    def checkpoint_rows(self, req_pool_indices: torch.Tensor) -> torch.Tensor:
        """[num_reqs, checkpoint_width] window rows a verify writes."""
        steps = torch.arange(
            self.checkpoint_width, dtype=req_pool_indices.dtype, device=self.device
        )
        return self.window_bases(req_pool_indices).unsqueeze(1) + steps

    @torch.no_grad()
    def snapshot_checkpoints(
        self,
        track_slots: torch.Tensor,
        req_pool_indices: torch.Tensor,
        steps: torch.Tensor,
    ) -> None:
        """Copy window checkpoint steps[i] of request i into track_slots[i] (radix
        snapshot after a verify); a negative step or track slot skips that row."""
        track_slots = track_slots.to(self.device)
        states = self._copy_states()
        if states[0].device.type == "cpu":
            valid = ((steps >= 0) & (track_slots > 0)).nonzero(as_tuple=True)[0]
            src = (self.window_bases(req_pool_indices) + steps)[valid].long()
            for state in states:
                state.index_copy_(
                    0, track_slots[valid].long(), state.index_select(0, src)
                )
            return
        from sglang_tpu.mamba_state_copy import (
            compiled_state_copy,
            copy_bucket_size,
        )

        src, dst, count = checkpoint_snapshot_pairs(
            track_slots,
            req_pool_indices,
            steps,
            self.window_start,
            self.checkpoint_width,
            copy_bucket_size(steps.shape[0]),
        )
        compiled_state_copy()(states, src, dst, count)

    def warmup_snapshot_checkpoints(self, batch_size: int, track_grid: int) -> None:
        """Compile and load the post-verify radix snapshot for one padded batch size."""
        from sglang.srt.compilation.compiled_op import warmup_compiled_op
        from sglang_tpu.mamba_state_copy import copy_bucket_size

        req_pool_indices = torch.zeros(
            batch_size, dtype=torch.int64, device=self.device
        )
        committed_lens = torch.zeros_like(req_pool_indices)
        last_steps = torch.full_like(req_pool_indices, -1)
        warmup_compiled_op(verify_track_steps, committed_lens, last_steps, track_grid)
        warmup_compiled_op(
            checkpoint_snapshot_pairs,
            torch.zeros(batch_size, dtype=torch.int32, device=self.device),
            req_pool_indices,
            last_steps,
            self.window_start,
            self.checkpoint_width,
            copy_bucket_size(batch_size),
        )
        # A queued warmup only compiles; executing loads the programs, as serving
        # would on its first verify. No step crosses a boundary, so nothing is copied.
        steps = verify_track_steps(committed_lens, last_steps, track_grid)
        track_slots = torch.zeros(batch_size, dtype=torch.int32)
        self.snapshot_checkpoints(track_slots, req_pool_indices, steps)

    def _copy_states(self):
        return [
            state
            for states in [*self.mamba_cache.conv, self.mamba_cache.temporal]
            for state in states
        ]

    def get_state(self) -> list[torch.Tensor]:
        """The per-layer conv and recurrent tensors the GDN/KDA kernels mutate."""
        return self._copy_states()

    def set_state(self, state: list[torch.Tensor]) -> None:
        """Install a list laid out like get_state."""
        num_layers = self.num_mamba_layers
        num_kinds = len(self.mamba_cache.conv)
        assert len(state) == (num_kinds + 1) * num_layers, len(state)
        self.mamba_cache = TpuPerLayerState(
            conv=[
                state[kind * num_layers : (kind + 1) * num_layers]
                for kind in range(num_kinds)
            ],
            temporal=state[num_kinds * num_layers :],
        )

    @torch.no_grad()
    def copy_from(self, src_indices: torch.Tensor, dst_indices: torch.Tensor):
        if src_indices.numel() == 0:
            return
        # Slot rows only: with extra_buffer radix caching, copies run between
        # snapshot slots and a request's slot before its extend.
        states = self._copy_states()
        if states[0].device.type == "cpu":
            # CPU reference for layout and allocator tests.
            for state in states:
                state.index_copy_(
                    0, dst_indices.long(), state.index_select(0, src_indices.long())
                )
            return
        from sglang_tpu.mamba_state_copy import (
            compiled_state_copy,
            copy_bucket_size,
        )

        # The allocator and radix tree retain authoritative CPU IDs. Never read
        # device IDs back here: only bucketed int32 tensors cross to the TPU.
        if src_indices.device.type != "cpu" or dst_indices.device.type != "cpu":
            raise ValueError("TPU state copies require CPU allocator slot IDs")
        count = src_indices.numel()
        padding = copy_bucket_size(count) - count
        src = torch.nn.functional.pad(src_indices.to(torch.int32), (0, padding))
        dst = torch.nn.functional.pad(dst_indices.to(torch.int32), (0, padding))
        copy_count = torch.tensor([count], dtype=torch.int32, device=self.device)
        compiled_state_copy()(
            states, src.to(self.device), dst.to(self.device), copy_count
        )

    @torch.no_grad()
    def warmup_state_copy(self, max_requests: int):
        if getattr(self, "_state_copy_warmed", False):
            return
        from torch_tpu._internal.sync.sync import synchronize

        from sglang_tpu.mamba_state_copy import (
            compiled_state_copy,
            copy_bucket_size,
        )

        states = self._copy_states()
        # Pool initialization must finish before its buffers are first donated.
        synchronize(states, wait=True)
        size = 8
        copy_count = torch.zeros(1, dtype=torch.int32, device=self.device)
        while size <= copy_bucket_size(max_requests):
            # Distinct tensors avoid Dynamo's src-is-dst specialization.
            src = torch.zeros(size, dtype=torch.int32, device=self.device)
            dst = torch.zeros(size, dtype=torch.int32, device=self.device)
            compiled_state_copy()(states, src, dst, copy_count)
            size *= 4
        synchronize(states, wait=True)
        self._state_copy_warmed = True
        logger.info(
            "TPU Mamba state copies warmed through bucket %d",
            copy_bucket_size(max_requests),
        )

    # The two below are inherited from MambaPool, which indexes the layer axis
    # as `state[:, slots]` on a stacked tensor. This pool has no layer axis to
    # index -- state is one tensor per layer -- so the inherited versions would
    # silently address the *slot* axis instead. Host offload of GDN state is
    # unsupported on TPU, so they raise rather than pretend.

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

    def _iter_transfer_state_entries(self):
        """One [state_slots, ...] tensor per layer, laid out as get_state.

        P/D registration reads these. Slice axes follow the kernel-native
        layouts: conv [slots, K-1, 1, conv_dim], recurrent [slots, n_v, d_k, d_v].
        """
        for conv in self.mamba_cache.conv:
            for tensor, layer_id in zip(conv, self.mamba_layer_ids, strict=True):
                yield "conv", tensor, 2, layer_id
        for tensor, layer_id in zip(
            self.mamba_cache.temporal, self.mamba_layer_ids, strict=True
        ):
            yield "temporal", tensor, 0, layer_id

    def committed_row(self, slot: int) -> int:
        """State row a P/D transfer reads or writes for slot.

        Extend leaves its state in the slot row, and P/D decode resets the slot's
        read offset to AT_SLOT before its first verify, so both sides use it.
        """
        return slot

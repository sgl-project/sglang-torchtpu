# SPDX-License-Identifier: Apache-2.0
# Adapted from https://github.com/sgl-project/sglang-jax/blob/main/python/sgl_jax/srt/mem_cache/memory_pool.py
"""TPU fused KV cache pool used by the Pallas ragged-paged-attention backend.

The Pallas v3 kernel expects KV cache laid out as a single 5D buffer per
layer with shape::

    [num_blocks, page_size, num_kv_heads_x2 // kv_packing, kv_packing, head_dim]

where ``num_kv_heads_x2`` is ``2 * num_kv_heads`` (K and V interleaved) and
``kv_packing = 32 // bits_per_dtype_elem`` is the number of dtype elements
that fit in a 32-bit lane (e.g. 2 for bfloat16). The kernel performs the
KV-cache write itself during attention, so SGLang's ``set_kv_buffer`` is a
no-op on the pallas path — calling it would only undo the kernel's writes
when ``save_kv_cache=True``.

To stay compatible with the rest of SGLang (radix cache, HiCache, etc.)
the pool still exposes ``get_key_buffer`` / ``get_value_buffer`` as logical
views into the fused buffer; the contiguous storage is the 5D fused
buffer.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import torch

from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    MLATokenToKVPool,
    get_tensor_size_bytes,
)
from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

GB = 1024 * 1024 * 1024


def _torch_dtype_packing(dtype: torch.dtype, packing_bits: int = 32) -> int:
    """Number of ``dtype`` elements that fit in a ``packing_bits``-bit lane."""
    bits = torch.empty((), dtype=dtype).element_size() * 8
    if packing_bits % bits != 0:
        raise ValueError(
            f"dtype bitwidth must divide {packing_bits}, but got {bits} for {dtype}."
        )
    return packing_bits // bits


# Rope width the TPU MLA kernels attend with when the model has none (NoPE, e.g.
# GLM-5.3-Flash): the backend feeds zeros, so q_pe . k_pe = 0.
NOPE_MLA_ROPE_DIM = 64


def _zeros_kv_buffer(shape, dtype: torch.dtype, device) -> torch.Tensor:
    """Preallocate one layer's KV buffer as a genuine ``dtype`` base tensor.

    Called once per layer at pool construction; nothing allocates later. The
    buffer must not be a dtype view (``uint8`` storage ``.view(fp8)``):
    torch_tpu passes a compiled graph the real device buffer only when the
    tensor dtype equals the storage dtype, otherwise the attention kernel's
    KV donation is dropped and every extend materializes a second copy of the
    pool (+6.6 GB for a 6.4 GB fp8 pool, OOM at tp=16).
    """
    buf = torch.zeros(shape, dtype=dtype, device=device)
    assert buf._base is None and buf.dtype == dtype
    return buf


def _seq_along_lane_head_dim(*, head_dim: int, kv_packing: int) -> int:
    """Head-dim width of the SEQ_ALONG_LANE layout, rounded to a sublane tile."""
    # Under TorchTPU, JAX is initialised CPU-only so Pallas can export
    # without a competing TPU client; `tpu_info_for_cpu_export` swaps in the
    # physical device's metadata for the duration of the query.
    from jax.experimental.pallas import tpu as pltpu

    from sglang_tpu.pallas_export import (
        tpu_info_for_cpu_export,
    )

    with tpu_info_for_cpu_export():
        num_sublanes = pltpu.get_tpu_info().num_sublanes
    tile = num_sublanes * kv_packing
    return _cdiv(head_dim, tile) * tile


def kv_pack_heads(num_kv_heads: int, head_dim: int, align: int) -> int:
    """Heads stored side by side in one ``align``-lane row (SGLANG_TPU_KV_PACK_HEADS), or 1.

    Only exact fits pack (e.g. two 64-wide heads in 128 lanes): the row then holds no
    padding, half the bytes of one zero-padded row per head.
    """
    if (
        tpu_envs.SGLANG_TPU_KV_PACK_HEADS.get()
        and num_kv_heads > 1
        and num_kv_heads * head_dim == align
    ):
        return num_kv_heads
    return 1


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class TpuFusedMHATokenToKVPool(MHATokenToKVPool):
    """MHA KV pool with the fused 5D layout consumed by the Pallas kernel.

    Only the ``num_blocks * page_size`` axis is addressable by ``loc``; we
    keep an additional pad of one ``page_size`` so that padded tokens
    (``loc == 0``) write into a dummy page, mirroring the upstream MHA pool
    behaviour.
    """

    HEAD_DIM_ALIGNMENT = 128
    # Whether this layout can store exactly-fitting heads side by side (kv_pack_heads).
    PACKS_HEADS = True
    # HiCache io backends (--hicache-io-backend) that can offload this pool to
    # host; hicache_hook refuses the rest at startup. raiden copies whole pages
    # by their dim-0 block id, so it serves any block-major pool exposing
    # get_fused_kv_buffer; torch_tpu's host pool stores this 5D row verbatim.
    # A subclass that changes the buffer layout must restate this. Listing an
    # io backend asserts it moves all of the pool's KV state: a pool that keeps
    # quantization scales outside get_fused_kv_buffer must not list it.
    kv_offload_io_backends: frozenset[str] = frozenset({"torch_tpu", "raiden"})
    # Whether --enable-unified-memory may share this class's per-layer buffers
    # between the full-attention and sliding-window layers: the attention
    # backends reading it must take each sliding-window group's page table.
    # A subclass with its own layout or backends must restate this.
    supports_unified_kv_pool: bool = True

    @classmethod
    def kv_bytes_per_token_per_layer(
        cls,
        *,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        store_dtype: torch.dtype,
    ) -> int:
        """Bytes one token occupies in one layer's buffer; must track
        ``_create_buffers``, since the KV budget is spent before the pool exists."""
        del v_head_dim  # V shares K's padded slot; see __init__'s width check.
        if (
            cls.PACKS_HEADS
            and kv_pack_heads(num_kv_heads, head_dim, cls.HEAD_DIM_ALIGNMENT) > 1
        ):
            num_kv_heads, head_dim = 1, num_kv_heads * head_dim
        packing = _torch_dtype_packing(store_dtype)
        num_kv_heads_x2 = _cdiv(num_kv_heads * 2, packing) * packing
        padded_head_dim = (
            _cdiv(head_dim, cls.HEAD_DIM_ALIGNMENT) * cls.HEAD_DIM_ALIGNMENT
        )
        return (
            num_kv_heads_x2 * padded_head_dim * torch._utils._element_size(store_dtype)
        )

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        head_num: int,
        head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        v_head_dim: Optional[int] = None,
        swa_head_num: Optional[int] = None,
        swa_head_dim: Optional[int] = None,
        swa_v_head_dim: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_alt_stream: bool = True,
        enable_kv_cache_copy: bool = False,
        allocation_label: Optional[str] = None,
    ):
        # We deliberately do *not* call ``MHATokenToKVPool.__init__`` here:
        # its ``_create_buffers`` allocates the legacy two-buffer (K / V) layout
        # which is incompatible with the Pallas kernel. Instead we initialize
        # the ``KVCache`` base directly, then build the fused buffers ourselves.
        from sglang.srt.mem_cache.memory_pool import (
            KVCache,  # local import to avoid cycles
        )

        KVCache.__init__(
            self,
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        # The Pallas RPA kernel reads and writes the KV cache as a
        # floating-point buffer: it statically validates the cache dtype is
        # floating, and TPU natively supports fp8 (e4m3/e5m2) as a *storage*
        # dtype. ``KVCache.__init__`` remaps fp8 -> uint8 storage to work
        # around CUDA's missing ``index_put`` for float8; that workaround does
        # not apply here because the kernel performs the cache writes itself
        # (``set_kv_buffer`` is a no-op on the steady-state path). Keep the
        # real fp8 dtype as the storage dtype so the kernel accepts the buffer
        # and ``kv_packing`` (computed from ``store_dtype`` below) is 4 for fp8.
        # For bf16 this is a no-op (``store_dtype`` already equals ``dtype``).
        self.store_dtype = self.dtype
        # Accounting tag SWAKVPool passes to its sub-pools ("Full" / "SWA").
        self.allocation_label = allocation_label

        self.head_num = swa_head_num if swa_head_num is not None else head_num
        self.head_dim = swa_head_dim if swa_head_dim is not None else head_dim
        self.v_head_dim = (
            swa_v_head_dim
            if swa_v_head_dim is not None
            else v_head_dim
            if v_head_dim is not None
            else head_dim
        )
        # Asymmetric K/V head dims (e.g. MiMo / DeepSeek-style qk=192, v=128):
        # the fused KV buffer stores a single shared head_dim, so V is padded up
        # to head_dim before the kernel writes it (see
        # ``attention_pallas_rpa_v3._forward``) and the attention output is
        # sliced back to v_head_dim. Only v_head_dim <= head_dim is
        # representable; a wider V cannot fit the qk-width slot.
        # Packed heads (kv_pack_heads): the buffer holds one head of the heads' total
        # width; attention backends read kv_pack_heads and lay q / k / v out to match.
        self.kv_pack_heads = (
            kv_pack_heads(self.head_num, self.head_dim, self.HEAD_DIM_ALIGNMENT)
            if self.PACKS_HEADS
            else 1
        )
        if self.kv_pack_heads > 1:
            if self.v_head_dim != self.head_dim:
                raise NotImplementedError("packed KV heads need v_head_dim == head_dim")
            self.head_num, self.head_dim = 1, self.head_num * self.head_dim
            self.v_head_dim = self.head_dim
        if self.v_head_dim > self.head_dim:
            raise NotImplementedError(
                "TpuFusedMHATokenToKVPool supports v_head_dim <= head_dim "
                f"(got v_head_dim={self.v_head_dim} > head_dim={self.head_dim}); "
                "the fused KV buffer stores V padded to the qk head_dim."
            )

        self.padded_head_dim = (
            _cdiv(self.head_dim, self.HEAD_DIM_ALIGNMENT) * self.HEAD_DIM_ALIGNMENT
        )
        self.kv_packing = _torch_dtype_packing(self.store_dtype)
        # K and V are interleaved along the kv-heads axis, then packed in 32-bit lanes.
        self.num_kv_heads_x2 = (
            _cdiv(self.head_num * 2, self.kv_packing) * self.kv_packing
        )

        if self.size % self.page_size != 0:
            raise ValueError(
                f"size ({self.size}) must be divisible by page_size ({self.page_size}) "
                "for the TPU fused KV pool."
            )
        # Pad with one extra page so loc == 0 (the padded-token slot) is harmless.
        self.num_blocks = self.size // self.page_size + 1

        self._create_buffers()
        # The KV budget was drawn at this per-token price before the pool existed.
        if self.kv_buffer:
            priced = self.kv_bytes_per_token_per_layer(
                num_kv_heads=self.head_num,
                head_dim=self.head_dim,
                v_head_dim=self.v_head_dim,
                store_dtype=self.store_dtype,
            )
            assert (
                self.kv_buffer[0].nbytes == self.num_blocks * self.page_size * priced
            ), f"KV layer buffer {self.kv_buffer[0].shape} != {priced} B/token"

        self.device_module = torch.get_device_module(self.device)
        # No alt_stream on TPU.
        self.alt_stream = None
        self._kv_copy_config = None

        self._finalize_allocation_log(size)

        # Bookkeeping used by store-cache helpers; the fused buffer keeps a
        # single contiguous row per slot of width ``num_kv_heads_x2 * head_dim``.
        self.row_dim = self.head_num * self.head_dim
        self.same_kv_dim = True

    def _synchronize_device(self) -> None:
        device = getattr(self, "device", None)
        if device is not None:
            device_type = torch.device(device).type
        elif getattr(self, "kv_buffer", None):
            device_type = self.kv_buffer[0].device.type
        else:
            device_type = None

        if device_type == "cpu" or device_type is None:
            return
        torch.accelerator.synchronize()

    def get_state(self) -> list[torch.Tensor]:
        """Return the fused KV buffers used by forward execution."""
        return self.kv_buffer

    def set_state(self, state: list[torch.Tensor]) -> None:
        """Replace the fused KV buffers used by forward execution."""
        self.kv_buffer = state

    # ------------------------------------------------------------------ #
    # Buffer creation / accessors
    # ------------------------------------------------------------------ #

    def _create_buffers(self):
        kv_heads_packed = self.num_kv_heads_x2 // self.kv_packing
        shape = (
            self.num_blocks,
            self.page_size,
            kv_heads_packed,
            self.kv_packing,
            self.padded_head_dim,
        )
        logger.info(
            "Allocating fused TPU KV buffers: layers=%d, shape per layer=%s, dtype=%s",
            self.layer_num,
            shape,
            self.store_dtype,
        )
        # The CUDA-only paths the upstream MHATokenToKVPool wraps allocation
        # in (``torch_memory_saver.region`` and ``torch.cuda.use_mem_pool`` for
        # Mooncake PD disaggregation) are inert on TPU — skip them and just
        # allocate. The parent ``KVCache.__init__`` still populates
        # ``memory_saver_adapter`` / ``enable_custom_mem_pool`` as no-ops so
        # inherited methods that touch them keep working.
        # Each layer's buffer must be a genuine ``store_dtype`` base tensor
        # (no dtype view), or the kernel's donation is lost -- see
        # ``_zeros_kv_buffer``.
        self.kv_buffer = [
            _zeros_kv_buffer(shape, self.store_dtype, self.device)
            for _ in range(self.layer_num)
        ]

        # Mirrors used by the rest of SGLang. These are recomputed lazily; the
        # fused buffer is the source of truth.
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [int(np.prod(x.shape[1:])) * x.dtype.itemsize for x in self.kv_buffer],
            device=self.device,
        )

    def _clear_buffers(self):
        del self.kv_buffer

    def get_kv_size_bytes(self):
        size_bytes = 0
        for buf in self.kv_buffer:
            size_bytes += get_tensor_size_bytes(buf)
        # Report as a (k_size, v_size) tuple for compatibility with the
        # base-class accounting; the buffers are fused so we split evenly.
        return size_bytes // 2, size_bytes // 2

    # ------------------------------------------------------------------ #
    # CPU offload / restore
    # ------------------------------------------------------------------ #

    def _raiden_backup(self):
        """The pool's retraction backup, built once when budgeted."""
        if not hasattr(self, "_raiden_backup_obj"):
            from sglang_tpu.raiden_kv_backup import (
                maybe_build_backup,
            )

            # A staged backup restores whole pages verbatim, while
            # SWAKVPool.load_cpu_copy row-filters its SWA member's payload
            # whenever the resume allocation maps fewer window tokens than the
            # save did. That member takes the row path instead.
            swa_member = self.allocation_label == "SWA"
            self._raiden_backup_obj = None if swa_member else maybe_build_backup(self)
        return self._raiden_backup_obj

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        """Copy the KV rows at ``indices`` out to host.

        Overrides the base version, which stores K and V in two separate
        tensors, while this pool uses one fused KV tensor per layer. With a
        raiden retraction budget, whole pages go to pinned host staging by
        DMA instead and come back the same way; rows that do not fill their
        pages sequentially use the row path.
        """
        backup = self._raiden_backup()
        if backup is not None:
            self._synchronize_device()
            handle = backup.try_save(indices)
            if handle is not None:
                return handle
        # Indices might be on host and need to be copied to device.
        indices = indices.to(device=self.kv_buffer[0].device, dtype=torch.long)
        self._synchronize_device()
        kv_cache_cpu = []
        chunk_size = self.cpu_offloading_chunk_size
        for layer_id in range(self.layer_num):
            kv_cache_cpu.append([])
            layer_kv = self.kv_buffer[layer_id].flatten(0, 1)
            for i in range(0, len(indices), chunk_size):
                chunk_indices = indices[i : i + chunk_size]
                kv_cpu = layer_kv.index_select(0, chunk_indices).to(
                    "cpu", non_blocking=True
                )
                kv_cache_cpu[-1].append(kv_cpu)
        self._synchronize_device()
        return kv_cache_cpu

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        """Copy the KV rows at ``indices`` from host to device.

        Inverse of `get_cpu_copy`. A raiden-staged payload DMAs straight
        back into the pages holding ``indices``. Row payloads are written a
        whole page at a time through `write_pages_torch`, which updates the
        layer buffer in place; a page that ``indices`` only partly covers is
        read back first and merged on device.
        """
        from sglang_tpu.kv_page_write import write_pages_torch
        from sglang_tpu.raiden_kv_backup import (
            RaidenKVBackupHandle,
        )

        if isinstance(kv_cache_cpu, RaidenKVBackupHandle):
            self._synchronize_device()
            self._raiden_backup().restore(kv_cache_cpu, indices)
            self._synchronize_device()
            return

        indices = indices.to(device="cpu", dtype=torch.long)
        if indices.numel() == 0:
            return
        page = self.page_size
        blocks = indices // page
        offsets = indices % page
        uniq_blocks, block_pos = torch.unique(blocks, return_inverse=True)
        full = torch.bincount(block_pos, minlength=uniq_blocks.numel()) == page
        full_rows = full[block_pos]
        # Position of each row's page within the full / partial page batches.
        full_pos = torch.cumsum(full.long(), 0) - 1
        part_pos = torch.cumsum((~full).long(), 0) - 1
        device = self.kv_buffer[0].device
        full_blocks = uniq_blocks[full].to(device)
        part_blocks = uniq_blocks[~full].to(device)
        full_page_pos = full_pos[block_pos][full_rows]
        full_offsets = offsets[full_rows]
        part_page_pos = part_pos[block_pos][~full_rows].to(device)
        part_offsets = offsets[~full_rows].to(device)

        self._synchronize_device()
        for layer_id in range(self.layer_num):
            rows = torch.cat(kv_cache_cpu[layer_id], dim=0)
            assert rows.shape[0] == indices.numel()
            dst = self.kv_buffer[layer_id]
            if full_blocks.numel():
                pages = torch.empty(
                    (full_blocks.numel(), page, *rows.shape[1:]), dtype=rows.dtype
                )
                pages[full_page_pos, full_offsets] = rows[full_rows]
                write_pages_torch(full_blocks, pages.to(device, non_blocking=True), dst)
            if part_blocks.numel():
                pages = dst[part_blocks]
                pages[part_page_pos, part_offsets] = rows[~full_rows].to(
                    device, non_blocking=True
                )
                write_pages_torch(part_blocks, pages, dst)
        self._synchronize_device()

    def filter_cpu_copy(self, kv_cache_cpu, row_mask: torch.Tensor):
        """Keep only the rows of a `get_cpu_copy` payload selected by ``row_mask``.

        Raiden-staged payloads are whole pages and are never filtered: only
        the sliding-window pool's payload goes through here, and that pool
        stays on the row path.

        Each layer's chunks are single fused KV tensors with the token axis
        first; the rows are masked along that axis and re-split into
        ``cpu_offloading_chunk_size`` chunks, the format `load_cpu_copy` reads.
        """
        chunk_size = self.cpu_offloading_chunk_size
        filtered = []
        for layer_chunks in kv_cache_cpu:
            if len(layer_chunks) == 0:
                filtered.append([])
                continue
            layer_kv = torch.cat(layer_chunks, dim=0)[row_mask]
            filtered.append(
                [
                    layer_kv[i : i + chunk_size]
                    for i in range(0, len(layer_kv), chunk_size)
                ]
            )
        return filtered

    # ``get_fused_kv_buffer`` is the canonical accessor on TPU; the Pallas
    # backend uses it directly.
    def get_fused_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_buffer[layer_id - self.start_layer]

    def _logical_kv_view(self, layer_id: int) -> torch.Tensor:
        """Return a logical [num_tokens, num_kv_heads*2, head_dim] view.

        Used by HiCache / radix cache helpers that want to read K and V back
        out separately. Note the head dim has padding to a multiple of 128.
        """
        buf = self.kv_buffer[layer_id - self.start_layer]
        n_blocks, page_size, packed, packing, hd = buf.shape
        return buf.view(n_blocks * page_size, packed * packing, hd)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait()
        view = self._logical_kv_view(layer_id)
        # K heads are the even indices of the interleaved KV axis.
        return view[:, 0 : 2 * self.head_num : 2, : self.head_dim]

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait()
        view = self._logical_kv_view(layer_id)
        return view[:, 1 : 2 * self.head_num : 2, : self.head_dim]

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def _get_key_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_key_buffer(layer_id)

    def _get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_value_buffer(layer_id)

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def set_kv_buffer(self, *args, **kwargs):
        """The Pallas RPA v3 kernel owns KV cache writes (via the donated
        ``kv_cache_fused`` argument), so this method must never be called.

        Same precedent as :class:`NoOpMHATokenToKVPool.set_kv_buffer` in
        ``mem_cache/memory_pool.py`` and ``MLATokenToKVPool.set_kv_buffer`` in
        sglang-jax — pools whose writes are owned by the attention kernel
        reject out-of-band writes loudly rather than silently scattering data
        that the next kernel call would overwrite.
        """
        raise NotImplementedError(
            "TpuFusedMHATokenToKVPool.set_kv_buffer was called. The Pallas "
            "ragged-paged-attention v3 kernel writes the fused KV buffer "
            "itself via the donated kv_cache_fused argument; any out-of-band "
            "write here would be overwritten on the next forward. If you hit "
            "this, the calling code path is incompatible with the rpa_v3 "
            "backend — either skip the write or switch to --attention-backend "
            "pallas_tpu_flash (which uses the unfused MHATokenToKVPool)."
        )

    # ------------------------------------------------------------------ #
    # Misc compatibility
    # ------------------------------------------------------------------ #

    def get_contiguous_buf_infos(self):
        kv_data_ptrs = [buf.data_ptr() for buf in self.kv_buffer]
        kv_data_lens = [buf.nbytes for buf in self.kv_buffer]
        # One item = one page. The fused buffers are page-major -- every layout
        # under this class leads with ``num_blocks`` (see ``_create_buffers`` here
        # and in TpuSeqAlongLaneKVPool) -- so ``buf[0]`` already spans the page's
        # ``page_size`` token slots. Multiplying by ``page_size`` again counted
        # them twice and inflated every KV-transfer byte count by that factor.
        kv_item_lens = [buf[0].nbytes for buf in self.kv_buffer]
        return kv_data_ptrs, kv_data_lens, kv_item_lens


class TpuSeqAlongLaneKVPool(TpuFusedMHATokenToKVPool):
    """MHA KV pool in the SEQ_ALONG_LANE layout the batched-RPA kernel reads.

    Selected when SGLANG_TPU_USE_SEQ_ALONG_LANE_KV is set. Same paged
    bookkeeping as the fused pool, different buffer shape -- it must match
    ``batched_rpa.wrapper.get_kv_cache_shape`` for that layout::

        [num_blocks, num_kv_heads * 2,
         align(head_dim, num_sublanes * kv_packing) // kv_packing, kv_packing,
         page_size]

    K and V interleave along axis 1 (K at ``2h``, V at ``2h + 1``) and the token
    index sits on the *lane* axis, so a page's tokens are contiguous in lanes
    rather than in rows. The head-dim axis is split into
    ``(hd // kv_packing, kv_packing)`` so a 32-bit word holds ``kv_packing``
    consecutive head-dim elements.

    ``num_kv_heads * 2`` is a plain axis here rather than something packed into
    32-bit lanes, so it carries no padding. At one KV head per rank that makes
    an fp8 cache half the size of the fused layout's, which rounds two heads up
    to ``kv_packing``. Host copies retain this native packing while translating
    between token indices and their page/lane coordinates.
    """

    PACKS_HEADS = False

    # raiden's page copy is layout-agnostic, and torch_tpu's host pool reads
    # its page shape from get_fused_kv_buffer, so it mirrors this layout too.
    kv_offload_io_backends: frozenset[str] = frozenset({"torch_tpu", "raiden"})
    supports_unified_kv_pool: bool = False

    @classmethod
    def kv_bytes_per_token_per_layer(
        cls,
        *,
        num_kv_heads: int,
        head_dim: int,
        v_head_dim: int,
        store_dtype: torch.dtype,
    ) -> int:
        """Like the base method, but head_dim rounds to a sublane tile and the
        ``num_kv_heads * 2`` axis never rounds."""
        del v_head_dim  # V interleaves into K's axis at the same width.
        packing = _torch_dtype_packing(store_dtype)
        return (
            num_kv_heads
            * 2
            * _seq_along_lane_head_dim(head_dim=head_dim, kv_packing=packing)
            * torch._utils._element_size(store_dtype)
        )

    def _head_dim_lanes(self) -> int:
        return _seq_along_lane_head_dim(
            head_dim=self.head_dim, kv_packing=self.kv_packing
        )

    def _create_buffers(self):
        shape = (
            self.num_blocks,
            self.head_num * 2,
            self._head_dim_lanes() // self.kv_packing,
            self.kv_packing,
            self.page_size,
        )
        logger.info(
            "Allocating SEQ_ALONG_LANE TPU KV buffers: layers=%d, shape per layer=%s, "
            "dtype=%s",
            self.layer_num,
            shape,
            self.store_dtype,
        )
        # Genuine ``store_dtype`` bases, as in the fused pool above.
        self.kv_buffer = [
            _zeros_kv_buffer(shape, self.store_dtype, self.device)
            for _ in range(self.layer_num)
        ]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self.data_strides = torch.tensor(
            [int(np.prod(x.shape[1:])) * x.dtype.itemsize for x in self.kv_buffer],
            device=self.device,
        )

    def _logical_kv_view(self, layer_id: int) -> torch.Tensor:
        """[num_tokens, num_kv_heads * 2, head_dim], as a copy.

        The token axis is minormost in this layout, so a logical row is
        strided; materialising it costs a transpose. Only the diagnostic and
        cache-inspection helpers below use it, never the forward path.
        """
        buf = self.kv_buffer[layer_id - self.start_layer]
        n_blocks, kv_heads_x2, hd_words, packing, page_size = buf.shape
        return buf.permute(0, 4, 1, 2, 3).reshape(
            n_blocks * page_size, kv_heads_x2, hd_words * packing
        )

    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        """Copy selected token lanes to the fused pool's host-row format."""
        backup = self._raiden_backup()
        if backup is not None:
            self._synchronize_device()
            handle = backup.try_save(indices)
            if handle is not None:
                return handle

        indices = indices.to(device="cpu", dtype=torch.long)
        self._synchronize_device()
        if indices.numel() == 0:
            return [[] for _ in range(self.layer_num)]

        device = self.kv_buffer[0].device
        chunk_specs = []
        chunk_size = self.cpu_offloading_chunk_size
        for i in range(0, len(indices), chunk_size):
            chunk_indices = indices[i : i + chunk_size]
            blocks, block_pos = torch.unique(
                chunk_indices // self.page_size, return_inverse=True
            )
            chunk_specs.append(
                (blocks.to(device), block_pos, chunk_indices % self.page_size)
            )

        pending_pages = []
        for layer_id in range(self.layer_num):
            layer_pages = []
            for blocks, block_pos, offsets in chunk_specs:
                pages = self.kv_buffer[layer_id].index_select(0, blocks)
                layer_pages.append(
                    (pages.to("cpu", non_blocking=True), block_pos, offsets)
                )
            pending_pages.append(layer_pages)

        self._synchronize_device()
        return [
            [pages[block_pos, ..., offsets] for pages, block_pos, offsets in chunks]
            for chunks in pending_pages
        ]

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        """Restore host rows with page-bounded device writes."""
        from sglang_tpu.kv_page_write import write_pages_torch
        from sglang_tpu.raiden_kv_backup import (
            RaidenKVBackupHandle,
        )

        if isinstance(kv_cache_cpu, RaidenKVBackupHandle):
            self._synchronize_device()
            self._raiden_backup().restore(kv_cache_cpu, indices)
            self._synchronize_device()
            return

        indices = indices.to(device="cpu", dtype=torch.long)
        if indices.numel() == 0:
            return
        page = self.page_size
        blocks = indices // page
        offsets = indices % page
        uniq_blocks, block_pos = torch.unique(blocks, return_inverse=True)
        full = torch.bincount(block_pos, minlength=uniq_blocks.numel()) == page
        full_rows = full[block_pos]
        full_pos = torch.cumsum(full.long(), 0) - 1
        part_pos = torch.cumsum((~full).long(), 0) - 1
        device = self.kv_buffer[0].device
        full_blocks = uniq_blocks[full].to(device)
        part_blocks = uniq_blocks[~full].to(device)
        full_page_pos = full_pos[block_pos][full_rows]
        full_offsets = offsets[full_rows]
        part_page_pos = part_pos[block_pos][~full_rows].to(device)
        part_offsets = offsets[~full_rows].to(device)

        self._synchronize_device()
        for layer_id in range(self.layer_num):
            rows = torch.cat(kv_cache_cpu[layer_id], dim=0)
            assert rows.shape[0] == indices.numel()
            dst = self.kv_buffer[layer_id]
            if full_blocks.numel():
                pages = torch.empty(
                    (full_blocks.numel(), *rows.shape[1:], page), dtype=rows.dtype
                )
                pages[full_page_pos, ..., full_offsets] = rows[full_rows]
                write_pages_torch(full_blocks, pages.to(device, non_blocking=True), dst)
            if part_blocks.numel():
                pages = dst[part_blocks]
                pages[part_page_pos, ..., part_offsets] = rows[~full_rows].to(
                    device, non_blocking=True
                )
                write_pages_torch(part_blocks, pages, dst)
        self._synchronize_device()


class TpuFusedMLATokenToKVPool(MLATokenToKVPool):
    """MLA latent KV pool with the paged 4D layout consumed by the Pallas MLA kernel.

    The MLA v2 kernel (``layers/attention/pallas/mla_ragged_paged_attention_v2``)
    reads and writes the latent cache as a single paged buffer per layer::

        [num_pages,
         page_size // kv_packing,
         kv_packing,
         align(kv_lora_rank, 128) + align(qk_rope_head_dim, 128)]

    where ``kv_packing = 32 // bits_per_dtype_elem`` (2 for bf16). The last dim
    pads the nope (latent) and rope segments *independently* to 128 — for
    DeepSeek (512 + 64) that is 512 + 128 = 640, with the rope values living at
    offset ``nope_dim`` (512), not ``kv_lora_rank + 0``. Like the MHA twin
    above, the kernel performs the cache write itself (the buffer is donated to
    the pallas call), so ``set_kv_buffer`` / ``set_mla_kv_buffer`` must never
    be called.
    """

    HEAD_DIM_ALIGNMENT = 128

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_nsa: bool = False,
        override_kv_cache_dim: Optional[int] = None,
    ):
        # Bypass ``MLATokenToKVPool.__init__``: its ``_create_buffers``
        # allocates the flat [size, 1, 576] layout, which the Pallas MLA
        # kernel cannot address. Initialize the ``KVCache`` base directly and
        # build the paged buffers ourselves (same pattern as
        # ``TpuFusedMHATokenToKVPool``).
        from sglang.srt.mem_cache.memory_pool import (
            KVCache,  # local import to avoid cycles
        )

        if use_nsa or override_kv_cache_dim is not None:
            raise NotImplementedError(
                "NSA / kv_cache_dim overrides are not supported by the TPU "
                "fused MLA pool."
            )

        KVCache.__init__(
            self,
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        # Keep the real (possibly fp8) dtype as storage: the kernel owns cache
        # writes, so the CUDA fp8-as-uint8 index_put workaround does not apply
        # (see TpuFusedMHATokenToKVPool.__init__ for the full rationale).
        self.store_dtype = self.dtype

        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_nsa = False
        self.nsa_kv_cache_store_fp8 = False
        # Logical (upstream) cache dim: nope + rope unpadded. Kept for
        # compatibility with consumers of the flat MLA pool.
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        # Physical segment dims: each aligned to 128 *independently* (the
        # kernel slices the last axis at ``nope_dim``, so a combined align
        # would misplace the rope segment for kv_lora_rank % 128 != 0).
        self.nope_dim = (
            _cdiv(kv_lora_rank, self.HEAD_DIM_ALIGNMENT) * self.HEAD_DIM_ALIGNMENT
        )
        rope_width = qk_rope_head_dim or NOPE_MLA_ROPE_DIM
        self.rope_dim = (
            _cdiv(rope_width, self.HEAD_DIM_ALIGNMENT) * self.HEAD_DIM_ALIGNMENT
        )
        self.padded_kv_cache_dim = self.nope_dim + self.rope_dim

        self.kv_packing = _torch_dtype_packing(self.store_dtype)
        if self.size % self.page_size != 0:
            raise ValueError(
                f"size ({self.size}) must be divisible by page_size "
                f"({self.page_size}) for the TPU fused MLA pool."
            )
        if self.page_size % self.kv_packing != 0:
            raise ValueError(
                f"page_size ({self.page_size}) must be divisible by kv_packing "
                f"({self.kv_packing}) for the TPU fused MLA pool."
            )
        # Pad with one extra page so loc == 0 (the padded-token slot) is
        # harmless; the TPU allocator hands out page IDs starting at 1.
        self.num_pages = self.size // self.page_size + 1

        self._create_buffers()

        self.device_module = torch.get_device_module(self.device)

        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.kv_buffer],
            dtype=torch.uint64,
            device=self.device,
        )
        self._finalize_allocation_log(size)

    # ------------------------------------------------------------------ #
    # Buffer creation / accessors
    # ------------------------------------------------------------------ #

    def _create_buffers(self):
        shape = (
            self.num_pages,
            self.page_size // self.kv_packing,
            self.kv_packing,
            self.padded_kv_cache_dim,
        )
        logger.info(
            "Allocating fused TPU MLA KV buffers: layers=%d, shape per layer=%s, "
            "dtype=%s (nope_dim=%d, rope_dim=%d)",
            self.layer_num,
            shape,
            self.store_dtype,
            self.nope_dim,
            self.rope_dim,
        )
        # Genuine ``store_dtype`` bases (no uint8 dtype view), or the MLA
        # kernel's KV donation is dropped and the extend graph materializes a
        # second pool -- see ``_zeros_kv_buffer``.
        self.kv_buffer = [
            _zeros_kv_buffer(shape, self.store_dtype, self.device)
            for _ in range(self.layer_num)
        ]

    def _clear_buffers(self):
        del self.kv_buffer

    # ``get_fused_kv_buffer`` is the canonical accessor on TPU; the Pallas MLA
    # backend passes it straight to the kernel (donated).
    def get_fused_kv_buffer(self, layer_id: int) -> torch.Tensor:
        return self.kv_buffer[layer_id - self.start_layer]

    def _logical_latent_view(self, layer_id: int) -> torch.Tensor:
        """Token-ordered logical view ``[num_tokens, 1, padded_kv_cache_dim]``.

        Flattening (page, row, col) of the paged layout recovers token order:
        token ``t`` of page ``p`` lives at ``[p, t // kv_packing,
        t % kv_packing]``, so the flat index is exactly ``p * page_size + t``.
        """
        buf = self.kv_buffer[layer_id - self.start_layer]
        n_pages, rows, packing, dim = buf.shape
        return buf.view(n_pages * rows * packing, 1, dim)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        view = self._logical_latent_view(layer_id)
        if self.nope_dim == self.kv_lora_rank:
            # Rope segment is adjacent to the unpadded latent: the logical
            # [nope + rope] layout is a contiguous prefix slice.
            return view[..., : self.kv_cache_dim]
        raise NotImplementedError(
            "get_key_buffer on the TPU fused MLA pool requires "
            "kv_lora_rank % 128 == 0 (rope segment starts at nope_dim, which "
            f"!= kv_lora_rank: {self.nope_dim} != {self.kv_lora_rank})."
        )

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id - self.start_layer)
        return self._logical_latent_view(layer_id)[..., : self.kv_lora_rank]

    def get_kv_buffer(self, layer_id: int):
        return self.get_key_buffer(layer_id), self.get_value_buffer(layer_id)

    def get_kv_size_bytes(self):
        assert hasattr(self, "kv_buffer")
        kv_size_bytes = 0
        for kv_cache in self.kv_buffer:
            kv_size_bytes += get_tensor_size_bytes(kv_cache)
        return kv_size_bytes

    def get_contiguous_buf_infos(self):
        kv_data_ptrs = [buf.data_ptr() for buf in self.kv_buffer]
        kv_data_lens = [buf.nbytes for buf in self.kv_buffer]
        # One item = one page (buf[0] is a full page in the paged layout).
        kv_item_lens = [buf[0].nbytes for buf in self.kv_buffer]
        return kv_data_ptrs, kv_data_lens, kv_item_lens

    # ------------------------------------------------------------------ #
    # CPU offload / restore — unsupported on the paged MLA layout
    # ------------------------------------------------------------------ #

    def get_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError(
            "CPU offloading / HiCache is unsupported on the paged TPU fused MLA "
            "layout: the base MLATokenToKVPool implementation indexes the KV "
            "buffer by token id, which assumes the flat [size, 1, kv_cache_dim] "
            "layout, not this pool's [num_pages, page_size // kv_packing, "
            "kv_packing, dim] paged buffer. TODO: map token id -> (page, slot) "
            "for the 4-D index before enabling offload with pallas_tpu_mla."
        )

    def load_cpu_copy(self, *args, **kwargs):
        raise NotImplementedError(
            "CPU offloading / HiCache is unsupported on the paged TPU fused MLA "
            "layout (see TpuFusedMLATokenToKVPool.get_cpu_copy)."
        )

    def get_mla_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError(
            "get_mla_kv_buffer is unsupported on the paged TPU fused MLA pool: "
            "the base implementation both indexes the KV buffer by token id "
            "(wrong for the paged layout) and calls a Triton kernel that is "
            "unavailable on TPU."
        )

    def move_kv_cache(self, *args, **kwargs):
        raise NotImplementedError(
            "move_kv_cache is unsupported on the paged TPU fused MLA layout: the "
            "base implementation relocates rows via token-id indexing "
            "(kv_cache[loc]), which addresses the num_pages axis on this paged "
            "buffer. TODO: map token id -> (page, slot) before enabling the "
            "spec-decode accept path with pallas_tpu_mla."
        )

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #

    def set_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError(
            "TpuFusedMLATokenToKVPool.set_kv_buffer was called. The Pallas MLA "
            "kernel writes the latent KV cache itself via the donated cache "
            "argument; any out-of-band write would be overwritten on the next "
            "forward. If you hit this, the calling code path is incompatible "
            "with the pallas_tpu_mla backend (e.g. the non-absorbed MHA "
            "prefill path) — the DeepSeek dispatch handler must return "
            "AttnForwardMethod.MLA for every forward mode."
        )

    def set_mla_kv_buffer(self, *args, **kwargs):
        raise NotImplementedError(
            "TpuFusedMLATokenToKVPool.set_mla_kv_buffer was called; the Pallas "
            "MLA kernel owns cache writes (see set_kv_buffer)."
        )


class TpuSplitMLATokenToKVPool(MLATokenToKVPool):
    """MLA KV pool in the layout the sparse MLA kernels read.

    The kernels keep each layer's cache in two paged ``uint8`` buffers:

    * NoPE, ``[num_pages, page_size, 4, 128]``: 512 bytes per token.
    * RoPE, ``[num_pages, page_size // 4, 4, 128]``: 64 bytes per token, padded
      to 128.

    The values are FP8, stored as raw bytes.

    Each layer that runs its own DSA indexer also gets an index-key buffer. It
    uses the same page table. A layer that reuses the previous layer's top-k
    result has no index-key buffer. With a pooled indexer (``index_kpool > 1``,
    GLM-5.3-Flash) the index-key buffer holds one record per pool, at the pool's
    position, and a raw buffer keeps each token's bf16 index key and
    compression gate, ``[num_pages, page_size, 2 * index_head_dim]``.

    Only the ops write these buffers, and they update them in place. That is
    why the accessors of the flat MLA pool, which address tokens by index, are
    not supported here.

    The core pool sizer counts the GPU DSA layout; hooks/dsa_kv_pool.py hands
    it ``bytes_per_token`` instead.
    """

    NOPE_BYTES = 512
    # One lane. A token's RoPE bytes are padded to this size.
    LANE_BYTES = 128
    # The RoPE and index-key buffers store this many tokens per row of a page.
    TOKENS_PER_ROW = 4

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        *,
        index_head_dim: int,
        index_key_layers: list[bool],
        index_kpool: int = 1,
    ):
        # Skip MLATokenToKVPool.__init__: it allocates flat [size, 1, 576]
        # buffers, which the kernels cannot use. TpuFusedMLATokenToKVPool does
        # the same.
        from sglang.srt.mem_cache.memory_pool import (
            KVCache,  # local import to avoid cycles
        )

        if len(index_key_layers) != layer_num:
            raise ValueError(
                f"index_key_layers has {len(index_key_layers)} entries for "
                f"{layer_num} layers."
            )

        KVCache.__init__(
            self,
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.store_dtype = torch.uint8
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.use_nsa = False
        self.nsa_kv_cache_store_fp8 = False
        self.kv_cache_dim = kv_lora_rank + qk_rope_head_dim
        self._index_head_dim = index_head_dim
        self._index_key_layers = index_key_layers
        self._index_kpool = index_kpool

        if self.size % self.page_size != 0:
            raise ValueError(
                f"size ({self.size}) must be divisible by page_size "
                f"({self.page_size}) for the TPU split MLA pool."
            )
        if self.page_size % self.TOKENS_PER_ROW != 0:
            raise ValueError(
                f"page_size ({self.page_size}) must be divisible by "
                f"{self.TOKENS_PER_ROW} for the TPU split MLA pool."
            )
        # Allocate one extra page. Token index 0 is the slot used for padded
        # tokens, and the TPU allocator hands out page IDs starting at 1, so
        # page 0 is never a real page.
        self.num_pages = self.size // self.page_size + 1

        self._create_buffers()
        self._finalize_allocation_log(size)

    @classmethod
    def bytes_per_token(
        cls, *, index_head_dim: int, index_key_layers: list[bool], index_kpool: int
    ) -> int:
        """Bytes one token takes across all layers' buffers, as allocated."""
        index_bytes = cls.index_key_record_bytes(index_head_dim)
        if index_kpool > 1:
            index_bytes += 2 * index_head_dim * torch.bfloat16.itemsize
        layer_bytes = cls.NOPE_BYTES + cls.LANE_BYTES
        return len(index_key_layers) * layer_bytes + sum(index_key_layers) * index_bytes

    @classmethod
    def index_key_record_bytes(cls, index_head_dim: int) -> int:
        """Bytes one token's index key takes in the cache.

        A record is the key's FP8 bytes in whole 128-byte lanes, then one scale
        byte per lane, rounded up to whole lanes again. A 128-wide key takes
        256 bytes. The indexer op checks the buffer's shape against what its
        kernel reads, so a mistake here fails at the first compile.
        """
        lane = cls.LANE_BYTES
        key_bytes = _cdiv(index_head_dim, lane) * lane
        return _cdiv(key_bytes + key_bytes // lane, lane) * lane

    def _create_buffers(self):
        rows_per_page = self.page_size // self.TOKENS_PER_ROW
        nope_shape = (
            self.num_pages,
            self.page_size,
            self.NOPE_BYTES // self.LANE_BYTES,
            self.LANE_BYTES,
        )
        rope_shape = (
            self.num_pages,
            rows_per_page,
            self.TOKENS_PER_ROW,
            self.LANE_BYTES,
        )
        index_key_shape = (
            self.num_pages,
            rows_per_page,
            self.TOKENS_PER_ROW,
            self.index_key_record_bytes(self._index_head_dim),
        )
        logger.info(
            "Allocating split TPU MLA KV buffers: layers=%d, nope=%s, rope=%s, "
            "index keys=%s on %d layers",
            self.layer_num,
            nope_shape,
            rope_shape,
            index_key_shape,
            sum(self._index_key_layers),
        )
        # One whole uint8 tensor per layer and buffer, not views of a shared
        # tensor: an op can only update a whole tensor in place.
        self.nope_buffer = [
            _zeros_kv_buffer(nope_shape, torch.uint8, self.device)
            for _ in range(self.layer_num)
        ]
        self.rope_buffer = [
            _zeros_kv_buffer(rope_shape, torch.uint8, self.device)
            for _ in range(self.layer_num)
        ]
        self.index_key_buffer = [
            (
                _zeros_kv_buffer(index_key_shape, torch.uint8, self.device)
                if has_indexer
                else None
            )
            for has_indexer in self._index_key_layers
        ]
        raw_shape = (self.num_pages, self.page_size, 2 * self._index_head_dim)
        self.kpool_raw_buffer = [
            (
                _zeros_kv_buffer(raw_shape, torch.bfloat16, self.device)
                if has_indexer and self._index_kpool > 1
                else None
            )
            for has_indexer in self._index_key_layers
        ]

    def _clear_buffers(self):
        del self.nope_buffer
        del self.rope_buffer
        del self.index_key_buffer
        del self.kpool_raw_buffer

    def get_nope_cache(self, layer_id: int) -> torch.Tensor:
        return self.nope_buffer[layer_id - self.start_layer]

    def get_rope_cache(self, layer_id: int) -> torch.Tensor:
        return self.rope_buffer[layer_id - self.start_layer]

    def get_index_key_cache(self, layer_id: int) -> torch.Tensor:
        buf = self.index_key_buffer[layer_id - self.start_layer]
        if buf is None:
            raise ValueError(
                f"layer {layer_id} reuses the previous layer's top-k result "
                "and has no index-key cache."
            )
        return buf

    def get_kpool_raw_cache(self, layer_id: int) -> torch.Tensor:
        buf = self.kpool_raw_buffer[layer_id - self.start_layer]
        if buf is None:
            raise ValueError(f"layer {layer_id} has no pooled-indexer raw cache.")
        return buf

    def get_kv_size_bytes(self):
        buffers = [
            *self.nope_buffer,
            *self.rope_buffer,
            *self.index_key_buffer,
            *self.kpool_raw_buffer,
        ]
        return sum(get_tensor_size_bytes(buf) for buf in buffers if buf is not None)

    def _unsupported(self, accessor: str):
        raise NotImplementedError(
            f"{accessor} is not supported on the TPU split MLA pool. Only the "
            "sparse MLA ops read and write this cache, and its layout cannot be "
            "addressed by token index the way the flat MLA pool is."
        )

    def get_key_buffer(self, *args, **kwargs):
        self._unsupported("get_key_buffer")

    def get_value_buffer(self, *args, **kwargs):
        self._unsupported("get_value_buffer")

    def get_kv_buffer(self, *args, **kwargs):
        self._unsupported("get_kv_buffer")

    def get_contiguous_buf_infos(self, *args, **kwargs):
        self._unsupported("get_contiguous_buf_infos")

    def get_cpu_copy(self, *args, **kwargs):
        self._unsupported("get_cpu_copy")

    def load_cpu_copy(self, *args, **kwargs):
        self._unsupported("load_cpu_copy")

    def get_mla_kv_buffer(self, *args, **kwargs):
        self._unsupported("get_mla_kv_buffer")

    def move_kv_cache(self, *args, **kwargs):
        self._unsupported("move_kv_cache")

    def set_kv_buffer(self, *args, **kwargs):
        self._unsupported("set_kv_buffer")

    def set_mla_kv_buffer(self, *args, **kwargs):
        self._unsupported("set_mla_kv_buffer")

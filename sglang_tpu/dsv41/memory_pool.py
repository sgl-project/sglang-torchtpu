# SPDX-License-Identifier: Apache-2.0
"""V4.1 word pools with SGLang page allocation and source-layer ownership."""

import numpy as np
import torch

from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool
from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool


class CSA2LatentPool(DeepSeekV4SingleKVPool):
    def _create_buffers(self):
        dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if dim != 512:
            raise ValueError("CSA2 storage requires 512-feature latents")
        if 1024 % self.page_size or self.page_size > 1024:
            raise ValueError("CSA2 page_size must divide 1024 FP32 tile elements")
        self.num_pages = (self.size + self.page_size + 1) // self.page_size
        self.kv_cache_total_dim = dim + (4096 // self.page_size)
        records = self.num_pages * self.page_size
        # 3-D page-major root owners keep dim[0]=num_pages outside XLA's (8,128)
        # minor tile dimensions for Raiden DMA while giving kernels zero-copy
        # views: k_pages uses [num_pages, page_size, 128] (page_size % 8 == 0)
        # and scale_pages uses [num_pages, 8, 128] (one whole (8,128) FP32 tile
        # per page, indexed at r * (1024 // page_size) so pages never share an
        # 8-sublane HBM tile across dim[0]).
        self.k_pages = [
            torch.zeros(
                (self.num_pages, self.page_size, 128),
                dtype=torch.int32,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.scale_pages = [
            torch.zeros(
                (self.num_pages, 8, 128),
                dtype=torch.float32,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.k_buffer = [t.view(records, 128) for t in self.k_pages]
        self.scale_buffer = [t.view(self.num_pages * 1024) for t in self.scale_pages]
        self.kv_buffer = [t.view(self.num_pages, -1) for t in self.k_pages]

    def get_kv_size_bytes(self):
        return sum(t.nbytes for t in self.k_pages + self.scale_pages), 0


class SWAWordPool(DeepSeekV4SingleKVPool):
    def _create_buffers(self):
        if self.qk_nope_head_dim + self.qk_rope_head_dim != 512:
            raise ValueError("V4.1 SWA storage requires 512-feature latents")
        self.num_pages = (self.size + self.page_size + 1) // self.page_size
        self.kv_cache_total_dim = 1024
        # 3-D page-major root owner [pages, 2*page_size, 128]; 2*page_size is a
        # multiple of 8 so the 2-D word view [pages*2*page_size, 128] is a
        # zero-copy bitcast on TPU.
        self.k_pages = [
            torch.zeros(
                (self.num_pages, 2 * self.page_size, 128),
                dtype=torch.int32,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.k_buffer = [
            t.view(self.num_pages * 2 * self.page_size, 128) for t in self.k_pages
        ]
        self.kv_buffer = [t.view(self.num_pages, -1) for t in self.k_pages]

    def get_kv_size_bytes(self):
        return sum(t.nbytes for t in self.k_pages), 0


class PackedFP8IndexPool:
    def __init__(
        self,
        size,
        page_size,
        head_dim,
        layer_num,
        device,
        *,
        record_alignment,
        full_pages=None,
    ):
        if head_dim != 128 or page_size < 1 or page_size & (page_size - 1):
            raise ValueError("V4.1 index storage requires D128 and power-of-two pages")
        if record_alignment < page_size or record_alignment % page_size:
            raise ValueError("Index allocation alignment must contain whole pages")
        self.size, self.page_size = size, page_size
        # Feature-major E4M3 bytes plus an in-page FP32 scale per record.
        self.page_rows = (33 * page_size + 127) // 128
        pages_per_alignment = record_alignment // page_size
        if full_pages is not None:
            # Align per-full-page rows to a multiple of 8 (TPU sublane height) so
            # [full_pages, rows_per_full, 128] <-> [num_pages * page_stride, 128]
            # is a zero-copy bitcast and Raiden transfers whole full pages.
            a, b = 8, pages_per_alignment
            while b:
                a, b = b, a % b
            stride_align = 8 // a
            self.page_stride = (
                (self.page_rows + stride_align - 1) // stride_align * stride_align
            )
            self.num_pages = full_pages * pages_per_alignment
            rows_per_full = pages_per_alignment * self.page_stride
            self.k_pages = [
                torch.zeros(
                    (full_pages, rows_per_full, head_dim),
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(layer_num)
            ]
            self.k_buffer = [
                t.view(self.num_pages * self.page_stride, head_dim)
                for t in self.k_pages
            ]
        else:
            self.page_stride = self.page_rows
            self.num_pages = (size + page_size + 1) // page_size
            # Core index pages may be smaller than the scorer's C1/C2 DMA pages.
            # Keep their logical geometry and round only the physical allocation.
            self.num_pages = (
                (self.num_pages + pages_per_alignment - 1)
                // pages_per_alignment
                * pages_per_alignment
            )
            self.k_pages = None
            self.k_buffer = [
                torch.zeros(
                    (self.num_pages * self.page_rows, head_dim),
                    dtype=torch.int32,
                    device=device,
                )
                for _ in range(layer_num)
            ]

    def get_kv_size_bytes(self):
        owners = self.k_pages if self.k_pages is not None else self.k_buffer
        return sum(t.nbytes for t in owners), 0

    def contiguous_page_row_buffers(self):
        return [buffer.view(self.num_pages, -1) for buffer in self.k_buffer]


class TPUPairStatePool(CompressStatePool):
    """3-D request-slot-major C2 pending-pair state for Raiden P/D transfer.

    ``state_buffer`` has shape ``(num_req_slots + 1, ring_stride, feat_stride)``
    with ``ring_stride % 8 == 0`` and ``feat_stride % 128 == 0`` so each request
    slot owns disjoint XLA ``(8, 128)`` HBM tiles (``state_max_blocks = 1``) and
    its 2-D view in ``pair_compress`` is a zero-copy bitcast.
    """

    def __init__(
        self,
        *,
        num_req_slots: int,
        ring_size: int,
        head_dim: int,
        device: str,
        enable_memory_saver: bool,
    ):
        self.num_req_slots = int(num_req_slots)
        self.ring_stride = max(8, (int(ring_size) + 7) // 8 * 8)
        self.feat_stride = ((2 * int(head_dim) + 127) // 128) * 128
        super().__init__(
            size=self.num_req_slots * ring_size,
            ring_size=ring_size,
            overlap=False,
            head_dim=head_dim,
            dtype=torch.float32,
            device=device,
            enable_memory_saver=enable_memory_saver,
            ratio=2,
            request_scoped=True,
            online=False,
        )
        self.state_buffer = torch.zeros(
            (self.num_req_slots + 1, self.ring_stride, self.feat_stride),
            dtype=torch.float32,
            device=device,
        )
        self.state_buffer[:, :, head_dim : 2 * head_dim] = float("-inf")
        self.kv_score_buffer = KVAndScore(self.state_buffer.view(-1, self.feat_stride))

    def _alloc_kv_score_buffer(self, *, dtype, device, enable_memory_saver):
        del dtype, device, enable_memory_saver
        self.memory_saver_adapter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        # Dummy 1-row CPU buffer for super().__init__()'s clear() call.
        self.kv_score_buffer = KVAndScore(
            torch.zeros((1, self.last_dim), dtype=torch.float32, device="cpu")
        )

    def get_kv_size_bytes(self):
        return int(self.state_buffer.nbytes), 0


class DSV41TPUTokenToKVPool(DSV4TPUTokenToKVPool):
    raiden_state_max_blocks = 1

    def _init_compressed_pools(self, **kwargs):
        self._building_compressed = True
        try:
            return super()._init_compressed_pools(**kwargs)
        finally:
            self._building_compressed = False

    def _make_kv_pool(self, **kwargs):
        ratio = kwargs["global_page_size"] // kwargs["page_size"]
        if getattr(self, "_building_compressed", False) and ratio in (1, 2):
            cls = CSA2LatentPool
        elif not getattr(self, "_building_compressed", False) and ratio == 1:
            cls = SWAWordPool
        else:
            return super()._make_kv_pool(**kwargs)
        return cls(
            kwargs["size"],
            kwargs["page_size"],
            kwargs["dtype"],
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            kwargs["layer_num"],
            kwargs["device"],
            kwargs["enable_memory_saver"],
        )

    def _make_indexer_pool(
        self,
        size,
        page_size,
        dtype,
        index_head_dim,
        layer_num,
        device,
        enable_memory_saver,
        force_fp4=False,
        global_page_size=None,
    ):
        # force_fp4 is the core's low-ratio index-pool selector, not our dtype.
        # Those calls omit global_page_size and use DSV41_INDEX_PAGE_SIZE.
        if (
            getattr(self, "_building_compressed", False)
            and getattr(self, "full_size", None) is not None
            and global_page_size is None
        ):
            full_tokens_with_dummy = self.full_size + self.page_size
            ratio = max(1, full_tokens_with_dummy // size)
            full_pages = (self.full_size + self.page_size + 1) // self.page_size
            return PackedFP8IndexPool(
                size,
                page_size,
                index_head_dim,
                layer_num,
                device,
                record_alignment=self.page_size // ratio,
                full_pages=full_pages,
            )
        return PackedFP8IndexPool(
            size,
            page_size,
            index_head_dim,
            layer_num,
            device,
            record_alignment=global_page_size or self.page_size,
        )

    def _make_pair_state_pool(self, enable_memory_saver: bool):
        return TPUPairStatePool(
            num_req_slots=self.num_req_slots,
            ring_size=self.get_ring_size(2),
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
        )

    def index_page_records(self, layer_id):
        item = self.layer_mapping[layer_id]
        return self.index_pools[item.compress_ratio].page_size

    def index_page_stride(self, layer_id):
        item = self.layer_mapping[layer_id]
        return self.index_pools[item.compress_ratio].page_stride

    def swa_root(self, layer_id):
        self.wait_layer_transfer(layer_id)
        idx = self._swa_local_layer_id(layer_id)
        if getattr(self.swa_kv_pool, "k_pages", None) is not None:
            return self.swa_kv_pool.k_pages[idx]
        return self.swa_kv_pool.k_buffer[idx]

    def get_swa_buffer(self, layer_id):
        return self.swa_root(layer_id).view(-1, 128)

    def low_ratio_root(self, layer_id, *, index=False):
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        pool = (self.index_pools if index else self.kv_pools)[item.compress_ratio]
        if getattr(pool, "k_pages", None) is not None:
            return pool.k_pages[item.compress_layer_id]
        return pool.k_buffer[item.compress_layer_id]

    def low_ratio_scale_root(self, layer_id):
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        pool = self.kv_pools[item.compress_ratio]
        if getattr(pool, "scale_pages", None) is not None:
            return pool.scale_pages[item.compress_layer_id]
        return pool.scale_buffer[item.compress_layer_id]

    def low_ratio_cache(self, layer_id, *, index=False):
        return self.low_ratio_root(layer_id, index=index).view(-1, 128)

    def low_ratio_scales(self, layer_id):
        return self.low_ratio_scale_root(layer_id).view(-1)

    def raiden_full_buffers(self):
        full = []
        for ratio in sorted(self.kv_pools):
            kv_pool = self.kv_pools.get(ratio)
            if isinstance(kv_pool, CSA2LatentPool):
                full.extend(kv_pool.k_pages)
                full.extend(kv_pool.scale_pages)
            index_pool = self.index_pools.get(ratio)
            if isinstance(index_pool, PackedFP8IndexPool) and index_pool.k_pages:
                full.extend(index_pool.k_pages)
        return full

    def raiden_swa_buffers(self):
        if isinstance(self.swa_kv_pool, SWAWordPool):
            return list(self.swa_kv_pool.k_pages)
        return []

    def raiden_state_buffers(self):
        return [
            p.state_buffer
            for p in self.compress_state_pools
            if isinstance(p, TPUPairStatePool)
        ]

    def get_contiguous_buf_infos(self):
        bufs = self.raiden_full_buffers()
        return (
            [b.data_ptr() for b in bufs],
            [b.nbytes for b in bufs],
            [b[0].nbytes for b in bufs],
        )

    def get_state_buf_infos(self):
        bufs = self.raiden_swa_buffers()
        return (
            [b.data_ptr() for b in bufs],
            [b.nbytes for b in bufs],
            [b[0].nbytes for b in bufs],
        )

    def get_request_state_buf_infos(self):
        bufs = self.raiden_state_buffers()
        return (
            [b.data_ptr() for b in bufs],
            [b.nbytes for b in bufs],
            [b[0].nbytes for b in bufs],
        )

    def request_state_transfer_indices(self, req_pool_idx: int, seq_len: int = 0):
        del seq_len
        req_idx = int(req_pool_idx)
        assert 0 <= req_idx < self.num_req_slots, (
            f"req_pool_idx={req_idx} out of range [0, {self.num_req_slots})"
        )
        return np.array([req_idx], dtype=np.int32)

    def get_kv_size_bytes(self):
        pools = [self.swa_kv_pool, *self.kv_pools.values(), *self.index_pools.values()]
        total = sum(p.get_kv_size_bytes()[0] for p in pools if p is not None)
        for pool in self.compress_state_pools + self.indexer_compress_state_pools:
            if pool is None:
                continue
            if hasattr(pool, "get_kv_size_bytes"):
                total += pool.get_kv_size_bytes()[0]
            else:
                tensor = pool.kv_score_buffer.kv_score
                total += tensor.numel() * tensor.element_size()
        return total, 0

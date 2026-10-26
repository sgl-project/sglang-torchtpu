# SPDX-License-Identifier: Apache-2.0
"""V4.1 word pools with SGLang page allocation and source-layer ownership."""

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4SingleKVPool
from sglang_tpu.dsv4.dsv4_memory_pool import DSV4TPUTokenToKVPool


class CSA2LatentPool(DeepSeekV4SingleKVPool):
    def _create_buffers(self):
        dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        if dim != 512:
            raise ValueError("CSA2 storage requires 512-feature latents")
        self.num_pages = (self.size + self.page_size + 1) // self.page_size
        self.kv_cache_total_dim = dim + 4
        records = self.num_pages * self.page_size
        self.k_buffer = [
            torch.zeros(
                (records, 128),
                dtype=torch.int32,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.scale_buffer = [
            torch.zeros(records, dtype=torch.float32, device=self.device)
            for _ in range(self.layer_num)
        ]
        self.kv_buffer = self.k_buffer

    def get_kv_size_bytes(self):
        return sum(t.nbytes for t in self.k_buffer + self.scale_buffer), 0


class SWAWordPool(DeepSeekV4SingleKVPool):
    def _create_buffers(self):
        if self.qk_nope_head_dim + self.qk_rope_head_dim != 512:
            raise ValueError("V4.1 SWA storage requires 512-feature latents")
        self.num_pages = (self.size + self.page_size + 1) // self.page_size
        self.kv_cache_total_dim = 1024
        # A genuine 2-D owner, viewed as [pages,2,page_size,128] by CSA2.
        # Each word holds low/high BF16 features in one of two planes.
        self.k_buffer = [
            torch.zeros(
                (self.num_pages * 2 * self.page_size, 128),
                dtype=torch.int32,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.kv_buffer = self.k_buffer

    def get_kv_size_bytes(self):
        return sum(t.nbytes for t in self.k_buffer), 0


class BF16IndexHalfplanePool:
    def __init__(
        self, size, page_size, head_dim, layer_num, device, *, record_alignment
    ):
        if head_dim != 128 or page_size % 2:
            raise ValueError("V4.1 index storage requires D128 and even pages")
        if record_alignment < page_size or record_alignment % page_size:
            raise ValueError("Index allocation alignment must contain whole pages")
        self.size, self.page_size = size, page_size
        self.num_pages = (size + page_size + 1) // page_size
        # Core index pages may be smaller than the scorer's C1/C2 DMA pages.
        # Keep their logical geometry and round only the physical allocation.
        pages_per_alignment = record_alignment // page_size
        self.num_pages = (
            (self.num_pages + pages_per_alignment - 1)
            // pages_per_alignment
            * pages_per_alignment
        )
        self.k_buffer = [
            torch.zeros(
                (self.num_pages * page_size // 2, head_dim),
                dtype=torch.int32,
                device=device,
            )
            for _ in range(layer_num)
        ]

    def get_kv_size_bytes(self):
        return sum(t.nbytes for t in self.k_buffer), 0

    def contiguous_page_row_buffers(self):
        return [buffer.view(self.num_pages, -1) for buffer in self.k_buffer]


class DSV41TPUTokenToKVPool(DSV4TPUTokenToKVPool):
    def clear_request_scoped_state(self, req_pool_idx):
        super().clear_request_scoped_state(req_pool_idx)
        # The V4 TPU parent resets only Raiden ring-shift metadata. V4.1 C2
        # instead owns a pending pair in HBM and must clear it on slot reuse.
        for pool in self.compress_state_pools:
            if pool is not None and pool.request_scoped and pool.ratio == 2:
                start = req_pool_idx * pool.ring_size
                pool.kv_score_buffer[start : start + pool.ring_size].clear()

    def clear_c4_req_states(self, req_pool_indices):
        super().clear_c4_req_states(req_pool_indices)
        # The core registers this callback at request-slot allocation for TPU.
        # C2 owns request state too, including when a cancelled slot is reused.
        for req in req_pool_indices:
            self.clear_request_scoped_state(int(req))

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
        if force_fp4:
            return BF16IndexHalfplanePool(
                size,
                page_size,
                index_head_dim,
                layer_num,
                device,
                record_alignment=self.page_size,
            )
        return super()._make_indexer_pool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            force_fp4,
            global_page_size,
        )

    def low_ratio_cache(self, layer_id, *, index=False):
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        pools = self.index_pools if index else self.kv_pools
        return pools[item.compress_ratio].k_buffer[item.compress_layer_id]

    def low_ratio_scales(self, layer_id):
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        return self.kv_pools[item.compress_ratio].scale_buffer[item.compress_layer_id]

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

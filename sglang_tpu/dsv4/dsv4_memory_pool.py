# SPDX-License-Identifier: Apache-2.0
"""TPU 3-tier KV cache and compressor state pools for DeepSeek-V4.

All DeepSeek-V4 Pallas kernels on TPU (`compressor_v1`, `streamindex_topk`,
`mla_swa`, `sparse_mla`, and `mla`) operate on 4D page-major `torch.uint8`
tensors with a global raw-token page size of 256 (`page_size = 256`):

1. **SWA KV Pool (:class:`TPUDeepSeekV4SWAKVPool`)**:
   - `swa_page_size = 256` raw tokens per page.
   - Stores raw `bfloat16` latents (`512 * 2 = 1024` bytes/token = 2 physical
     rows of `(4, 128)` `uint8`).
   - Shape per layer: `(num_swa_pages, 512, 4, 128)` `torch.uint8`.

2. **CSA (`compress_ratio = 4`) KV Pool (:class:`TPUDeepSeekV4CSAKVPool`)**:
   - `c4_page_size = 256 // 4 = 64` compressed tokens per page (1-to-1 with
     256-token full-attention pages).
   - Split into two companion 4D `torch.uint8` tensors per C4 layer:
     - `nope_buffer`: `(num_c4_pages, 64, 4, 128)` `torch.uint8`
       (`512` bytes/token: `448` FP8 values + `64` lane-periodic UE8M0 scales).
     - `rope_buffer`: `(num_c4_pages, 16, 4, 128)` `torch.uint8`
       (`128` bytes/token: `64` BF16 values packed 4 tokens per `(4, 128)` row).

3. **CSA Lightning Indexer K Pool (:class:`TPUDeepSeekV4IndexerPool`)**:
   - `c4_page_size = 64` compressed tokens per page.
   - Shape per C4 layer: `(num_c4_pages, 16, 4, 256)` `torch.uint8`
     (`256` bytes/token: `128` FP8 values + `1` UE8M0 scale padded to `256`
     bytes, 4 tokens packed per `(4, 256)` row).

4. **HCA (`compress_ratio = 128`) KV Pool (:class:`TPUDeepSeekV4HCAKVPool`)**:
   - `c128_page_size = 256 // 128 = 2` compressed tokens per page (1-to-1 with
     256-token full-attention pages).
   - Stores raw `bfloat16` latents (`512 * 2 = 1024` bytes/token = 2 physical
     rows of `(4, 128)` `uint8`).
   - Shape per C128 layer: `(num_c128_pages, 4, 4, 128)` `torch.uint8`.

5. **Compressor State Pool (:class:`TPUCompressStatePool`)**:
   - Stores byte-transposed `float32` `(kv, score)` rolling window states in 4D
     `torch.uint8` page layout for `proj_and_save_state` and
     `compress_norm_rope_store`:
     - C4 Attention (`head_dim=512, overlap=True, state_block_size=4`):
       `(num_state_pages, 64, 4, 128)` `torch.uint8` (`16` rows/token).
     - C4 Indexer (`head_dim=128, overlap=True, state_block_size=4`):
       `(num_state_pages, 8, 4, 256)` `torch.uint8` (`2` rows/token).
     - C128 Attention (`head_dim=512, overlap=False, state_block_size=8`):
       `(num_state_pages, 64, 4, 128)` `torch.uint8` (`8` rows/token).

Every buffer is allocated as a genuine `torch.uint8` base tensor (`_base is None`)
so `torch_tpu` honours `donate_argnums` / `input_output_aliases` without
materializing defensive copies during `torch.compile`.
"""

from __future__ import annotations

import logging
import struct
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# Cache of `[max_pos, rope_head_dim]` float32 cos_sin tables derived from
# complex64 `freqs_cis`, keyed by `(id(freqs_cis), freqs_cis.device)`.
_COS_SIN_CACHE: Dict[Tuple[int, torch.device], torch.Tensor] = {}


def get_or_build_cos_sin_cache(freqs_cis: torch.Tensor) -> torch.Tensor:
    """Convert complex64 ``freqs_cis`` ``[max_pos, rope_head_dim // 2]`` into the
    contiguous ``float32`` ``[max_pos, rope_head_dim]`` ``[cos, sin]`` table
    consumed by the TPU Pallas compressor and RoPE kernels."""
    key = (id(freqs_cis), freqs_cis.device)
    cached = _COS_SIN_CACHE.get(key)
    if cached is not None:
        return cached
    real_imag = torch.view_as_real(freqs_cis)
    cos_sin = (
        torch.cat([real_imag[..., 0], real_imag[..., 1]], dim=-1)
        .to(dtype=torch.float32, device=freqs_cis.device)
        .contiguous()
    )
    _COS_SIN_CACHE[key] = cos_sin
    return cos_sin


from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    ONLINE_C128,
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import get_tensor_size_bytes

logger = logging.getLogger(__name__)

_TPU_KV_PACKING = 4
_TPU_LANE_DIM = 128
_TPU_INDEXER_LANE_DIM = 256

# Minimum number of scratch token slots in TPUCompressStatePool so a large
# single-step prefill batch (up to 8192 tokens) never collides with persistent
# per-request ring pages even when --max-total-tokens is small.
_TPU_MIN_PREFILL_STATE_TOKENS = 8192

# Uniform per-request page stride (`32` pages) and scratch page count (`2048`
# pages) across ALL TPUCompressStatePool instances (C4, C4-Indexer, C128).
# Keeping `shape[0] = 1 + num_req_slots * 32 + 2048` identical across every
# state tensor allows Raiden's single state engine to pull all C4, C4-Indexer,
# and C128 persistent rings for a request in one 32-page transfer without
# changing `compressor_v1`'s internal modulo `8` / `32` ring arithmetic.
_TPU_STATE_SLOT_STRIDE_PAGES = 32
_TPU_STATE_SCRATCH_PAGES = _TPU_MIN_PREFILL_STATE_TOKENS // 4


def _zeros_kv_buffer(
    shape: Tuple[int, ...], dtype: torch.dtype, device: str
) -> torch.Tensor:
    """Allocate a genuine ``dtype`` base tensor (`_base is None`) for TPU Pallas
    buffer donation."""
    buf = torch.zeros(shape, dtype=dtype, device=device)
    assert buf._base is None and buf.dtype == dtype
    return buf


class TPUDeepSeekV4SWAKVPool(DeepSeekV4SingleKVPool):
    """TPU SWA KV pool (`page_size = 256` raw tokens, raw `bfloat16` latents).

    Physical shape per layer: `(num_swa_pages, 512, 4, 128)` `torch.uint8`
    (`2` rows of `(4, 128)` `uint8` = `1024` bytes per token).
    """

    ROWS_PER_TOKEN = 2

    def _create_buffers(self) -> None:
        self.kv_cache_total_dim = 1024
        self.num_pages = (self.size + self.page_size + 1) // self.page_size
        self.physical_page_size = self.page_size * self.ROWS_PER_TOKEN
        shape = (
            self.num_pages,
            self.physical_page_size,
            _TPU_KV_PACKING,
            _TPU_LANE_DIM,
        )
        logger.info(
            "Allocating TPU DSV4 SWA KV buffers: layers=%d, shape=%s, dtype=uint8",
            self.layer_num,
            shape,
        )
        self.k_buffer = [
            _zeros_kv_buffer(shape, torch.uint8, self.device)
            for _ in range(self.layer_num)
        ]
        self.kv_buffer = self.k_buffer

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return sum(get_tensor_size_bytes(t) for t in self.k_buffer), 0


class TPUDeepSeekV4CSAKVPool(DeepSeekV4SingleKVPool):
    """TPU CSA (`compress_ratio = 4`) KV pool (`page_size = 64` compressed tokens).

    Split into two companion 4D `torch.uint8` arrays per C4 layer:
    - `nope_buffer`: `(num_c4_pages, 64, 4, 128)` `torch.uint8` (`512` B/token)
    - `rope_buffer`: `(num_c4_pages, 16, 4, 128)` `torch.uint8` (`128` B/token)
    """

    def _create_buffers(self) -> None:
        assert self.page_size % _TPU_KV_PACKING == 0, (
            f"CSA compressed page_size ({self.page_size}) must be divisible by "
            f"{_TPU_KV_PACKING}."
        )
        self.kv_cache_total_dim = 640
        self.num_pages = (self.size + self.page_size + 1) // self.page_size
        nope_shape = (
            self.num_pages,
            self.page_size,
            _TPU_KV_PACKING,
            _TPU_LANE_DIM,
        )
        rope_shape = (
            self.num_pages,
            self.page_size // _TPU_KV_PACKING,
            _TPU_KV_PACKING,
            _TPU_LANE_DIM,
        )
        logger.info(
            "Allocating TPU DSV4 CSA KV buffers: layers=%d, nope_shape=%s, "
            "rope_shape=%s, dtype=uint8",
            self.layer_num,
            nope_shape,
            rope_shape,
        )
        self.nope_buffer = [
            _zeros_kv_buffer(nope_shape, torch.uint8, self.device)
            for _ in range(self.layer_num)
        ]
        self.rope_buffer = [
            _zeros_kv_buffer(rope_shape, torch.uint8, self.device)
            for _ in range(self.layer_num)
        ]
        self.k_buffer = self.nope_buffer
        self.kv_buffer = self.nope_buffer

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.nope_buffer[layer_id]

    def get_rope_buffer(self, layer_id: int) -> torch.Tensor:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.rope_buffer[layer_id]

    def get_nope_and_rope_buffer(
        self, layer_id: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layer_transfer_counter is not None:
            self.layer_transfer_counter.wait_until(layer_id)
        return self.nope_buffer[layer_id], self.rope_buffer[layer_id]

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        k_size_bytes = sum(get_tensor_size_bytes(t) for t in self.nope_buffer) + sum(
            get_tensor_size_bytes(t) for t in self.rope_buffer
        )
        return k_size_bytes, 0


class TPUDeepSeekV4HCAKVPool(DeepSeekV4SingleKVPool):
    """TPU HCA (`compress_ratio = 128`) KV pool (`page_size = 2` compressed tokens).

    Stores raw `bfloat16` latents (`1024` bytes per compressed token = `2` rows
    of `(4, 128)` `uint8`).
    Physical shape per C128 layer: `(num_c128_pages, 4, 4, 128)` `torch.uint8`.
    """

    ROWS_PER_TOKEN = 2

    def _create_buffers(self) -> None:
        self.kv_cache_total_dim = 1024
        self.num_pages = (self.size + self.page_size + 1) // self.page_size
        self.physical_page_size = self.page_size * self.ROWS_PER_TOKEN
        shape = (
            self.num_pages,
            self.physical_page_size,
            _TPU_KV_PACKING,
            _TPU_LANE_DIM,
        )
        logger.info(
            "Allocating TPU DSV4 HCA KV buffers: layers=%d, shape=%s, dtype=uint8",
            self.layer_num,
            shape,
        )
        self.k_buffer = [
            _zeros_kv_buffer(shape, torch.uint8, self.device)
            for _ in range(self.layer_num)
        ]
        self.kv_buffer = self.k_buffer

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return sum(get_tensor_size_bytes(t) for t in self.k_buffer), 0


class TPUDeepSeekV4IndexerPool(DeepSeekV4IndexerPool):
    """TPU CSA Lightning Indexer K pool (`page_size = 64` compressed tokens).

    Physical shape per C4 layer: `(num_c4_pages, 16, 4, 256)` `torch.uint8`
    (`256` bytes per compressed token, 4 tokens packed per `(4, 256)` row).
    """

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: Optional[int] = None,
    ):
        assert page_size % _TPU_KV_PACKING == 0, (
            f"Indexer compressed page_size ({page_size}) must be divisible by "
            f"{_TPU_KV_PACKING}."
        )
        self.num_pages = (size + page_size + 1) // page_size
        self.physical_page_size = page_size // _TPU_KV_PACKING
        super().__init__(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            global_page_size=global_page_size,
        )

    def _create_buffer(self) -> None:
        shape = (
            self.num_pages,
            self.physical_page_size,
            _TPU_KV_PACKING,
            _TPU_INDEXER_LANE_DIM,
        )
        logger.info(
            "Allocating TPU DSV4 Indexer K buffers: layers=%d, shape=%s, dtype=uint8",
            self.layer_num,
            shape,
        )
        self.index_k_with_scale_buffer = [
            _zeros_kv_buffer(shape, torch.uint8, self.device)
            for _ in range(self.layer_num)
        ]
        self.index_k_fp4_buffer = None
        self.index_k_fp4_scale_buffer = None

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        return self.index_k_with_scale_buffer[layer_id]

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return (
            sum(get_tensor_size_bytes(t) for t in self.index_k_with_scale_buffer),
            0,
        )


class TPUCompressStatePool(CompressStatePool):
    """TPU 4D `torch.uint8` paged compressor state pool for `compressor_v1`.

    Geometry by mode:
    - C4 Attention (`ratio=4, head_dim=512, overlap=True`):
      `state_block_size = 4` tokens/page, `16` rows/token ->
      `state_buffer` shape `(num_state_pages, 64, 4, 128)` `torch.uint8`.
    - C4 Indexer (`ratio=4, head_dim=128, overlap=True`):
      `state_block_size = 4` tokens/page, `2` rows/token ->
      `state_buffer` shape `(num_state_pages, 8, 4, 256)` `torch.uint8`.
    - C128 Attention (`ratio=128, head_dim=512, overlap=False`):
      `state_block_size = 8` tokens/page, `8` rows/token ->
      `state_buffer` shape `(num_state_pages, 64, 4, 128)` `torch.uint8`.

    Layout of `num_state_pages`:
    - Page `0`: reserved dummy padding page.
    - Pages `[1 .. 1 + num_req_slots * 32)`: persistent per-request ring slots
      strided uniformly at `slot_stride_pages = 32` across C4, C4-Indexer, and
      C128 (`compressor_v1` wraps within the first `ring_pages_per_req = 8`
      pages of each 32-page slot for C4 and all `32` pages for C128).
    - Pages `[1 + num_req_slots * 32 .. num_state_pages)`: uniform `2048`
      scratch pages for intra-prefill blocks (`>= 8192` tokens for C4,
      `>= 16384` tokens for C128).
    """

    def __init__(
        self,
        size: int,
        ring_size: int,
        overlap: bool,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        ratio: int,
        swa_page_size: int = 256,
        num_req_slots: int = 257,
    ):
        self.head_dim = head_dim
        self.overlap = overlap
        self.num_req_slots = num_req_slots
        self.is_indexer = head_dim == 128 and overlap

        if ratio == 4:
            self.state_block_size = 4
            self.window_blocks = 4
            self.ring_tokens_per_req = max(ring_size, 32)
        elif ratio == 128:
            self.state_block_size = 8
            self.window_blocks = 16
            self.ring_tokens_per_req = max(ring_size, 256)
        else:
            raise ValueError(f"Unsupported compress ratio for TPU state pool: {ratio}")

        self.ring_pages_per_req = self.ring_tokens_per_req // self.state_block_size
        self.slot_stride_pages = max(
            self.ring_pages_per_req, _TPU_STATE_SLOT_STRIDE_PAGES
        )
        self.last_dim_u8 = _TPU_INDEXER_LANE_DIM if self.is_indexer else _TPU_LANE_DIM
        coff = 1 + int(overlap)
        state_bytes_per_token = 2 * coff * head_dim * 4  # float32 bytes
        row_bytes = _TPU_KV_PACKING * self.last_dim_u8
        assert state_bytes_per_token % row_bytes == 0
        self.state_rows_per_token = state_bytes_per_token // row_bytes
        self.state_physical_page_size = (
            self.state_block_size * self.state_rows_per_token
        )

        # Uniform page count across C4, C4-Indexer, and C128 state pools:
        # 1) Page 0 dummy page (1 page)
        # 2) Persistent per-request slots (num_req_slots * slot_stride_pages)
        # 3) Prefill scratch pages (_TPU_STATE_SCRATCH_PAGES = 2048 pages)
        self.scratch_start_page = 1 + self.num_req_slots * self.slot_stride_pages
        self.num_scratch_pages = _TPU_STATE_SCRATCH_PAGES
        self.num_state_pages = self.scratch_start_page + self.num_scratch_pages
        exact_total_tokens = self.num_state_pages * self.state_block_size

        super().__init__(
            size=exact_total_tokens,
            ring_size=ring_size,
            overlap=overlap,
            head_dim=head_dim,
            dtype=torch.float32,
            device=device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            online=False,
            swa_page_size=swa_page_size,
            state_cache_page_size=self.state_block_size,
        )

        # Allocate the real 4D uint8 state_buffer after super().__init__() so
        # CompressStatePool.__init__'s `self.kv_score_buffer[-1].clear()` does
        # not mutate our uint8 buffer through a float32 view on XLA/TPU.
        self._size = exact_total_tokens
        self._logical_size = exact_total_tokens
        assert self._size % self.state_block_size == 0
        assert self.num_state_pages == self._size // self.state_block_size

        shape = (
            self.num_state_pages,
            self.state_physical_page_size,
            _TPU_KV_PACKING,
            self.last_dim_u8,
        )
        self.state_buffer = _zeros_kv_buffer(shape, torch.uint8, device)
        self.kv_score_buffer = KVAndScore(
            self.state_buffer.view(torch.float32).view(self._size, self.last_dim)
        )

    def _alloc_kv_score_buffer(
        self, *, dtype: torch.dtype, device: str, enable_memory_saver: bool
    ) -> None:
        del dtype, device, enable_memory_saver
        self.memory_saver_adapter = None
        self.enable_custom_mem_pool = False
        self.custom_mem_pool = None
        # Dummy 1-row CPU buffer for super().__init__()'s `[-1].clear()` call.
        self.kv_score_buffer = KVAndScore(
            torch.zeros((1, self.last_dim), dtype=torch.float32, device="cpu")
        )

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return int(get_tensor_size_bytes(self.state_buffer)), 0


class DSV4TPUTokenToKVPool(DeepSeekV4TokenToKVPool):
    """TPU 3-tier KV cache and compressor state pool for DeepSeek-V4."""

    def __init__(self, *args, **kwargs):
        if (
            kwargs.get("start_layer") is None
            and kwargs.get("end_layer") is None
            and kwargs.get("layer_num") is not None
        ):
            kwargs["start_layer"] = 0
            kwargs["end_layer"] = int(kwargs["layer_num"])
        super().__init__(*args, **kwargs)
        assert self.page_size == 256, (
            f"DSV4TPUTokenToKVPool requires page_size=256, got {self.page_size}"
        )
        assert self.swa_page_size == 256, (
            f"DSV4TPUTokenToKVPool requires swa_page_size=256, got {self.swa_page_size}"
        )
        self._req_state_shifts: dict[int, torch.Tensor] = {
            4: torch.zeros(self.num_req_slots, dtype=torch.int32, device="cpu"),
            128: torch.zeros(self.num_req_slots, dtype=torch.int32, device="cpu"),
        }
        self._state_pool_proto: dict[int, Optional[TPUCompressStatePool]] = {
            4: next(
                (
                    p
                    for p in self.compress_state_pools
                    if isinstance(p, TPUCompressStatePool) and p.ratio == 4
                ),
                None,
            ),
            128: next(
                (
                    p
                    for p in self.compress_state_pools
                    if isinstance(p, TPUCompressStatePool) and p.ratio == 128
                ),
                None,
            ),
        }

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        total_k = 0
        for pool in (
            self.swa_kv_pool,
            self.c4_kv_pool,
            self.c128_kv_pool,
            self.c4_indexer_kv_pool,
            *self.compress_state_pools,
            *self.indexer_compress_state_pools,
        ):
            if pool is not None and hasattr(pool, "get_kv_size_bytes"):
                total_k += int(pool.get_kv_size_bytes()[0])
        return total_k, 0

    def _make_kv_pool(
        self,
        *,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        global_page_size: int,
        cls: type = DeepSeekV4SingleKVPool,
        kv_layout: KVLayout = KVLayout.V4,
    ) -> DeepSeekV4SingleKVPool:
        assert cls is DeepSeekV4SingleKVPool, (
            "HiSparse / uniform_fp8 custom KV pool classes are not supported on TPU "
            f"(got {cls.__name__})."
        )
        assert kv_layout is KVLayout.V4, (
            f"TPU DSV4 pools do not support CUDA FlashMLA layout {kv_layout}."
        )
        if page_size * 4 == global_page_size:
            pool_cls: type = TPUDeepSeekV4CSAKVPool
        elif page_size * 128 == global_page_size:
            pool_cls = TPUDeepSeekV4HCAKVPool
        elif page_size == global_page_size:
            pool_cls = TPUDeepSeekV4SWAKVPool
        else:
            raise ValueError(
                f"Unexpected (page_size={page_size}, global_page_size={global_page_size}) "
                "for DSV4TPUTokenToKVPool."
            )
        return pool_cls(
            size,
            page_size,
            dtype,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
        )

    def _make_indexer_pool(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        force_fp4: bool = False,
        global_page_size: Optional[int] = None,
    ) -> TPUDeepSeekV4IndexerPool:
        assert not force_fp4, "Low-ratio FP4 indexer pool is not supported on TPU DSV4."
        return TPUDeepSeekV4IndexerPool(
            size,
            page_size,
            dtype,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            global_page_size=global_page_size,
        )

    def _make_compress_state_pool(
        self, ratio: int, *, head_dim: int, enable_memory_saver: bool
    ) -> TPUCompressStatePool:
        assert not (ratio == 128 and ONLINE_C128), (
            "SGLANG_OPT_USE_ONLINE_COMPRESS is not supported on TPU DSV4."
        )
        config = self.compressed_pool_configs[ratio]
        return TPUCompressStatePool(
            size=config.state_size,
            ring_size=self.get_ring_size(ratio),
            overlap=ratio == 4,
            head_dim=head_dim,
            dtype=config.state_dtype,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            swa_page_size=self.swa_page_size,
            num_req_slots=self.num_req_slots,
        )

    # ------------------------------------------------------------------ #
    # TPU Pallas 4D uint8 buffer accessors
    # ------------------------------------------------------------------ #

    def get_swa_buffer(self, layer_id: int) -> torch.Tensor:
        """Return `(num_swa_pages, 512, 4, 128)` `uint8` SWA KV buffer."""
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.k_buffer[self._swa_local_layer_id(layer_id)]

    def get_csa_nope_buffer(self, layer_id: int) -> torch.Tensor:
        """Return `(num_c4_pages, 64, 4, 128)` `uint8` CSA NoPE KV buffer."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None and item.compress_ratio == 4
        assert isinstance(self.c4_kv_pool, TPUDeepSeekV4CSAKVPool)
        return self.c4_kv_pool.nope_buffer[item.compress_layer_id]

    def get_csa_rope_buffer(self, layer_id: int) -> torch.Tensor:
        """Return `(num_c4_pages, 16, 4, 128)` `uint8` CSA RoPE KV buffer."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None and item.compress_ratio == 4
        assert isinstance(self.c4_kv_pool, TPUDeepSeekV4CSAKVPool)
        return self.c4_kv_pool.rope_buffer[item.compress_layer_id]

    def get_csa_kv_buffers(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return `(nope_buffer, rope_buffer)` for a C4 layer."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None and item.compress_ratio == 4
        assert isinstance(self.c4_kv_pool, TPUDeepSeekV4CSAKVPool)
        return (
            self.c4_kv_pool.nope_buffer[item.compress_layer_id],
            self.c4_kv_pool.rope_buffer[item.compress_layer_id],
        )

    def get_hca_buffer(self, layer_id: int) -> torch.Tensor:
        """Return `(num_c128_pages, 4, 4, 128)` `uint8` HCA KV buffer."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None and item.compress_ratio == 128
        assert isinstance(self.c128_kv_pool, TPUDeepSeekV4HCAKVPool)
        return self.c128_kv_pool.k_buffer[item.compress_layer_id]

    def get_indexer_k_buffer(self, layer_id: int) -> torch.Tensor:
        """Return `(num_c4_pages, 16, 4, 256)` `uint8` Indexer K buffer."""
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None and item.compress_ratio == 4
        indexer_pool = self._indexer_pool(4)
        return indexer_pool.get_index_k_with_scale_buffer(item.compress_layer_id)

    def get_state_buffer(
        self, layer_id: int, from_indexer: bool = False
    ) -> torch.Tensor:
        """Return the 4D `uint8` compressor state buffer for `layer_id`."""
        pool = (
            self.get_indexer_compress_states(layer_id)
            if from_indexer
            else self.get_attention_compress_states(layer_id)
        )
        assert isinstance(pool, TPUCompressStatePool)
        return pool.state_buffer

    def build_state_block_table(
        self,
        ratio: int,
        req_pool_indices_cpu: Sequence[int] | torch.Tensor,
        seq_lens_cpu: Optional[Sequence[int] | torch.Tensor],
        extend_seq_lens_cpu: Optional[Sequence[int] | torch.Tensor],
        is_decode: bool,
        bs: int,
        non_empty_bs: int,
        device: torch.device | str,
    ) -> Optional[torch.Tensor]:
        """Build the compact `[bs, 6]` `int32` state block table for `ratio`.

        Columns per request row `i`:
        `(ring_base, b_prev, old_shift, b_tail_start, new_shift, scratch_base)`.
        """
        proto = self._state_pool_proto.get(ratio)
        if proto is None:
            return None

        state_block_size = proto.state_block_size
        window_blocks = proto.window_blocks
        ring_pages = proto.ring_pages_per_req
        slot_stride = proto.slot_stride_pages
        scratch_start_page = proto.scratch_start_page
        num_state_pages = proto.num_state_pages
        shifts = self._req_state_shifts[ratio]

        table_cpu = torch.zeros((bs, 6), dtype=torch.int32, device="cpu")
        if non_empty_bs <= 0:
            return table_cpu.to(device=device)

        if is_decode:
            reqs = torch.as_tensor(
                req_pool_indices_cpu[:non_empty_bs], dtype=torch.long, device="cpu"
            )
            req_shifts = shifts[reqs]
            table_cpu[:non_empty_bs, 0] = 1 + reqs.to(torch.int32) * slot_stride
            table_cpu[:non_empty_bs, 2] = req_shifts
            table_cpu[:non_empty_bs, 4] = req_shifts
            return table_cpu.to(device=device)

        assert seq_lens_cpu is not None, (
            "seq_lens_cpu is required to build extend/prefill state_block_table."
        )
        scratch_cursor = scratch_start_page
        for i in range(non_empty_bs):
            req_idx = int(req_pool_indices_cpu[i])
            seq_len = int(seq_lens_cpu[i])
            q_len = (
                int(extend_seq_lens_cpu[i])
                if extend_seq_lens_cpu is not None
                else seq_len
            )
            if seq_len <= 0 or q_len <= 0:
                continue
            start_pos = max(0, seq_len - q_len)
            b_prev = (start_pos - 1) // state_block_size if start_pos > 0 else -1
            b_end = (seq_len - 1) // state_block_size
            old_shift = int(shifts[req_idx].item())
            ring_base = 1 + req_idx * slot_stride
            if b_end - b_prev <= window_blocks:
                table_cpu[i, 0] = ring_base
                table_cpu[i, 2] = old_shift
                table_cpu[i, 4] = old_shift
            else:
                b_tail_start = b_end - window_blocks + 1
                if start_pos == 0:
                    new_shift = old_shift
                else:
                    new_shift = (
                        old_shift + window_blocks - (b_end - b_prev)
                    ) % ring_pages
                num_scratch = b_tail_start - (b_prev + 1)
                if scratch_cursor + num_scratch > num_state_pages:
                    raise RuntimeError(
                        f"TPUCompressStatePool(ratio={ratio}) scratch page overflow: "
                        f"required={scratch_cursor + num_scratch}, "
                        f"capacity={num_state_pages}."
                    )
                table_cpu[i, 0] = ring_base
                table_cpu[i, 1] = b_prev
                table_cpu[i, 2] = old_shift
                table_cpu[i, 3] = b_tail_start
                table_cpu[i, 4] = new_shift
                table_cpu[i, 5] = scratch_cursor
                scratch_cursor += num_scratch
                shifts[req_idx] = new_shift

        return table_cpu.to(device=device)

    # ------------------------------------------------------------------ #
    # Prefill/Decode Disaggregation buffer descriptors & state metadata
    # ------------------------------------------------------------------ #

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Page-major descriptors for 256-token full-page compressed KV pools.

        Mirrors :meth:`TpuFusedMHATokenToKVPool.get_contiguous_buf_infos`: each
        4D ``torch.uint8`` buffer has ``buf.shape[0] = num_pages`` and
        ``item_len = buf[0].nbytes``. Ordering is deterministic across prefill
        and decode peers: C4 NoPE, C4 RoPE, C4 Indexer K, then C128 HCA.
        """
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        groups: List[Sequence[torch.Tensor]] = []
        if isinstance(self.c4_kv_pool, TPUDeepSeekV4CSAKVPool):
            groups.append(self.c4_kv_pool.nope_buffer)
            groups.append(self.c4_kv_pool.rope_buffer)
        if isinstance(self.c4_indexer_kv_pool, TPUDeepSeekV4IndexerPool):
            groups.append(self.c4_indexer_kv_pool.index_k_with_scale_buffer)
        if isinstance(self.c128_kv_pool, TPUDeepSeekV4HCAKVPool):
            groups.append(self.c128_kv_pool.k_buffer)
        for group in groups:
            for buf in group:
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Page-major descriptors for 256-token SWA sliding-window KV pages."""
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        if isinstance(self.swa_kv_pool, TPUDeepSeekV4SWAKVPool):
            for buf in self.swa_kv_pool.k_buffer:
                data_ptrs.append(buf.data_ptr())
                data_lens.append(buf.nbytes)
                item_lens.append(buf[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_request_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        """Page-major descriptors for all request-slot compressor state rings.

        On TPU, C4 Attention, C4 Indexer, and C128 Attention state pools are all
        request-slot-scoped with uniform ``num_state_pages`` and a 32-page slot
        stride ``[1 + req_pool_idx * 32 .. +32)``.
        """
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for pool in (*self.compress_state_pools, *self.indexer_compress_state_pools):
            if pool is None:
                continue
            assert isinstance(pool, TPUCompressStatePool)
            buf = pool.state_buffer
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def request_state_transfer_indices(
        self, req_pool_idx: int, seq_len: int = 0
    ) -> np.ndarray:
        """Return the 32 persistent state page indices owned by ``req_pool_idx``."""
        del seq_len
        req_idx = int(req_pool_idx)
        assert 0 <= req_idx < self.num_req_slots, (
            f"req_pool_idx={req_idx} out of range [0, {self.num_req_slots})"
        )
        base = 1 + req_idx * _TPU_STATE_SLOT_STRIDE_PAGES
        return np.arange(base, base + _TPU_STATE_SLOT_STRIDE_PAGES, dtype=np.int32)

    @staticmethod
    def _req_idx_from_state_rows(state_rows: Sequence[int]) -> int:
        if len(state_rows) != _TPU_STATE_SLOT_STRIDE_PAGES:
            raise ValueError(
                f"Expected {_TPU_STATE_SLOT_STRIDE_PAGES} state pages, "
                f"got {len(state_rows)}"
            )
        req_idx, rem = divmod(int(state_rows[0]) - 1, _TPU_STATE_SLOT_STRIDE_PAGES)
        if rem != 0 or req_idx < 0:
            raise ValueError(
                f"Invalid base state page {state_rows[0]} for stride "
                f"{_TPU_STATE_SLOT_STRIDE_PAGES}"
            )
        return req_idx

    def pack_req_state_meta(self, state_rows: Sequence[int]) -> bytes:
        """Pack post-prefill ``(shift_c4, shift_c128)`` for ``state_rows``."""
        req_idx = self._req_idx_from_state_rows(state_rows)
        return struct.pack(
            "<ii",
            int(self._req_state_shifts[4][req_idx].item()),
            int(self._req_state_shifts[128][req_idx].item()),
        )

    def unpack_req_state_meta(self, state_rows: Sequence[int], blob: bytes) -> None:
        """Apply incoming prefill ``(shift_c4, shift_c128)`` to ``state_rows``."""
        req_idx = self._req_idx_from_state_rows(state_rows)
        shift_4, shift_128 = struct.unpack("<ii", blob)
        self._req_state_shifts[4][req_idx] = int(shift_4)
        self._req_state_shifts[128][req_idx] = int(shift_128)

    def get_key_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        item = self.layer_mapping[layer_id]
        assert item is not None
        if item.compress_ratio == 0:
            return self.get_swa_buffer(layer_id)
        if item.compress_ratio == 4:
            return self.get_csa_nope_buffer(layer_id)
        if item.compress_ratio == 128:
            return self.get_hca_buffer(layer_id)
        raise ValueError(f"Unsupported compress_ratio={item.compress_ratio}")

    def get_value_buffer(self, layer_id: int) -> torch.Tensor:
        return self.get_key_buffer(layer_id)

    def get_kv_buffer(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self.get_key_buffer(layer_id)
        return buf, buf

    def tpu_commit_pending_loads(self) -> None:
        return

    # ------------------------------------------------------------------ #
    # State reset hooks (no-ops on TPU: `compressor_v1` stores byte-transposed
    # f32 across the 4 sublanes and masks `curr_pos < 0` in-kernel via
    # `positions_ref`, so `-inf` host-driven fills are unnecessary and would
    # write non-transposed bytes).
    # ------------------------------------------------------------------ #

    def translate_loc_from_full_to_swa(self, kv_indices: torch.Tensor) -> torch.Tensor:
        assert self.full_to_swa_index_mapping is not None
        mapping = self.full_to_swa_index_mapping
        if kv_indices.device == mapping.device:
            return mapping[kv_indices].to(torch.int32)
        return (
            mapping[kv_indices.to(mapping.device)].to(torch.int32).to(kv_indices.device)
        )

    def clear_c4_req_states(self, req_pool_indices: Sequence[int]) -> None:
        if req_pool_indices:
            reqs = torch.as_tensor(req_pool_indices, dtype=torch.long, device="cpu")
            self._req_state_shifts[4][reqs] = 0
            self._req_state_shifts[128][reqs] = 0

    def clear_c128_req_state(self, req_pool_idx: int) -> None:
        if 0 <= req_pool_idx < self.num_req_slots:
            self._req_state_shifts[128][req_pool_idx] = 0

    def clear_request_scoped_state(self, req_pool_idx: int) -> None:
        """Reset host ring-shift metadata without eager HBM writes.

        Must NEVER execute an eager in-place write (``clear()`` / ``fill_``) on
        ``state_buffer``: on ``torch_tpu``, eager mutation reallocates the
        underlying PJRT buffer and invalidates pointers registered with Raiden.
        """
        if 0 <= req_pool_idx < self.num_req_slots:
            self._req_state_shifts[4][req_pool_idx] = 0
            self._req_state_shifts[128][req_pool_idx] = 0

    def clear_unaccepted_c128_draft_states(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        accept_lens: torch.Tensor,
        num_draft_tokens: int,
    ) -> None:
        del req_pool_indices, seq_lens, accept_lens, num_draft_tokens
        return

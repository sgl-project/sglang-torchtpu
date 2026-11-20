# SPDX-License-Identifier: Apache-2.0
"""TPU KV pool for DeepSeek-V4.1 (compress ratios 0, 1 and 2).

Keeps the core DeepSeekV4TokenToKVPool control plane (full and SWA allocators,
full_to_swa_index_mapping, kv_source layer sharing, ratio-2 pair state) and swaps
only the physical buffers for TPU-tile-aligned 3D page roots (for Raiden P/D
page DMA without XLA (8, 128) tile interleaving) with zero-copy kernel views:

- SWA, every layer: root bf16 [num_pages256, 256, 512]; 64-token kernel view
  bf16 [num_pages256 * 4, 64, 512]; page = swa_loc // 64.
- Main KV, kv_source layers: root FP8 latent words int32
  [num_full_pages, 256 // r, 128] + root fp32 scales [num_full_pages, 8, 128]
  (one whole (8, 128) tile per full page, scale_step = 1024 // (256 // r));
  flat views int32 [R, 128] and fp32 [num_full_pages * 1024].
- Indexer K, kv_source layers: root packed FP8 int32
  [num_full_pages, rows_per_full, 128] with page_stride=36 (r=1, 72 rows/page)
  or page_stride=40 (r=2, 40 rows/page); flat view int32 [P * page_stride, 128]
  described by PackedFP8Layout(P, page_stride=page_stride, page_size=128).
- Ratio-2 pair state, ratio-2 kv_source layers: root fp32
  [num_req_slots + 1, ring_stride, feat_stride] (ring_stride=8, feat_stride=1024);
  spare slot ``num_req_slots`` holds the read-only ``-1`` sentinel ``(0, -inf)``.

The allocators reserve logical page 0, so SWA page 0 (64-token pages 0..3), main
page 0 and the indexer pages of full page 0 are never written. Every root buffer
is a base tensor (`_base is None`) so torch_tpu can donate it to Pallas calls
and Raiden can register its HBM address directly.
"""

from __future__ import annotations

import inspect
import logging
import math
from dataclasses import dataclass
from functools import partial
from typing import List, Mapping, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np
import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.layers.attention.dsv4.unified_kv_gate import is_unified_kv_triton
from sglang.srt.mem_cache.deepseek_v4_compress_state import (
    CompressStatePool,
    KVAndScore,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.kv.memory import request_window as core_request_window
from sglang.srt.mem_cache.memory_pool import get_tensor_size_bytes
from sglang.srt.runtime_context import get_disagg, get_exec, get_schedule
from sglang.srt.utils import ceil_div
from sglang_tpu.dsv4.dsv4_memory_pool import _zeros_kv_buffer
from sglang_tpu.models.deepseek_v41.request_window import (
    RESERVED_PAGES,
    RING_PAGES,
    RING_TOKENS,
    DSV41TPURequestWindow,
    ring_rows,
    window_buffer_pages,
)

try:
    from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
        PAGE_RECORDS,
        PAGE_ROWS,
        PackedFP8Layout,
    )
except ModuleNotFoundError as exc:
    if exc.name != "jax" and not (exc.name or "").startswith("jax."):
        raise
    PAGE_RECORDS = 128
    PAGE_ROWS = 33

    @dataclass(frozen=True)
    class PackedFP8Layout:  # type: ignore[no-redef]
        """Pure-Python layout fallback when JAX is absent (CPU unit tests)."""

        pages: int
        page_stride: int = PAGE_ROWS
        page_offset: int = 0
        page_size: int = PAGE_RECORDS

        @property
        def page_rows(self) -> int:
            return (PAGE_ROWS * self.page_size + PAGE_RECORDS - 1) // PAGE_RECORDS

        def validate(self, bank_rows: Tuple[int, ...]) -> None:
            if (
                self.pages < 1
                or self.page_stride < self.page_rows
                or self.page_offset < 0
                or len(bank_rows) != 1
                or self.page_offset + self.pages * self.page_stride > bank_rows[0]
            ):
                raise ValueError("Packed FP8 page addresses exceed their owners")


logger = logging.getLogger(__name__)

FULL_PAGE_TOKENS = 256
# CSA2 reads 64-token SWA views; the root buffer groups 4 views per 256-token
# logical SWA page so Raiden page DMA moves whole TPU-tile-aligned pages.
SWA_PAGE_TOKENS = 64
LATENT_DIM = 512
INDEX_HEAD_DIM = 128
WORD_LANES = 128
TPU_SUBLANE_ALIGN = 8
SCALE_TILE_ROWS = TPU_SUBLANE_ALIGN
SCALE_TILE_LANES = WORD_LANES
SCALE_TILE_ELEMENTS = SCALE_TILE_ROWS * SCALE_TILE_LANES

SWA_BYTES_PER_TOKEN = LATENT_DIM * torch.bfloat16.itemsize
# One 512 B FP8 record (448 e4m3 + 64 e8m0) in int32[128].
MAIN_WORD_BYTES_PER_RECORD = WORD_LANES * 4

LOW_RATIOS = (1, 2)
SUPPORTED_RATIOS = (0, *LOW_RATIOS)


def num_swa_pages256(swa_size: int) -> int:
    # The SWA allocator reserves one logical 256-token page, so the top swa_loc
    # is swa_size + 255.
    return ceil_div(swa_size + FULL_PAGE_TOKENS, FULL_PAGE_TOKENS)


def num_swa_pages64(swa_size: int) -> int:
    return num_swa_pages256(swa_size) * (FULL_PAGE_TOKENS // SWA_PAGE_TOKENS)


def num_main_pages(full_size: int) -> int:
    return ceil_div(full_size + FULL_PAGE_TOKENS, FULL_PAGE_TOKENS)


def num_main_records(full_size: int, ratio: int) -> int:
    return num_main_pages(full_size) * (FULL_PAGE_TOKENS // ratio)


def num_index_pages(full_size: int, ratio: int) -> int:
    return num_main_records(full_size, ratio) // PAGE_RECORDS


def main_bytes_per_record(ratio: int) -> int:
    """512 B FP8 latent word + share of one (8, 128) fp32 scale tile per page."""
    records_per_page = FULL_PAGE_TOKENS // ratio
    scale_bytes_per_page = SCALE_TILE_ELEMENTS * torch.float32.itemsize
    assert scale_bytes_per_page % records_per_page == 0
    return MAIN_WORD_BYTES_PER_RECORD + scale_bytes_per_page // records_per_page


def index_page_stride(ratio: int) -> int:
    """128-record subpage stride so each full page's rows are a multiple of 8."""
    subpages = (FULL_PAGE_TOKENS // ratio) // PAGE_RECORDS
    stride_align = TPU_SUBLANE_ALIGN // math.gcd(TPU_SUBLANE_ALIGN, subpages)
    return ceil_div(PAGE_ROWS, stride_align) * stride_align


def index_bytes_per_record(ratio: int) -> int:
    """Packed FP8 index-K bytes per record including 8-row page alignment."""
    return index_page_stride(ratio) * WORD_LANES * 4 // PAGE_RECORDS


def c2_state_fixed_bytes(
    *,
    num_req_slots: int,
    ring_size: int,
    head_dim: int,
    num_layers: int,
) -> int:
    """Total HBM bytes of ``TPUPairStatePool`` across ``num_layers`` ratio-2 sources."""
    ring_stride = max(
        TPU_SUBLANE_ALIGN,
        ceil_div(ring_size, TPU_SUBLANE_ALIGN) * TPU_SUBLANE_ALIGN,
    )
    feat_stride = ceil_div(2 * head_dim, WORD_LANES) * WORD_LANES
    return (
        (num_req_slots + 1)
        * ring_stride
        * feat_stride
        * torch.float32.itemsize
        * num_layers
    )


def uses_dsv41_pool(compress_ratios: Sequence[int]) -> bool:
    """Whether a DeepSeek-V4 family model needs this pool (ratio 1 or 2 layers)."""
    return any(ratio in LOW_RATIOS for ratio in compress_ratios)


class DSV41TPUSWAKVPool(DeepSeekV4SingleKVPool):
    """Per-layer bf16 SWA roots [num_pages256, 256, 512] with 64-token views."""

    def __init__(
        self,
        *,
        size: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
    ):
        self.num_pages256 = num_swa_pages256(size)
        self.num_pages = self.num_pages256 * (FULL_PAGE_TOKENS // SWA_PAGE_TOKENS)
        super().__init__(
            size,
            SWA_PAGE_TOKENS,
            torch.bfloat16,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            global_page_size=FULL_PAGE_TOKENS,
        )

    def _create_buffers(self) -> None:
        self.kv_cache_total_dim = SWA_BYTES_PER_TOKEN
        shape = (self.num_pages256, FULL_PAGE_TOKENS, LATENT_DIM)
        self.k_pages = [
            _zeros_kv_buffer(shape, torch.bfloat16, self.device)
            for _ in range(self.layer_num)
        ]
        self.kv_buffer = [
            p.view(self.num_pages, SWA_PAGE_TOKENS, LATENT_DIM) for p in self.k_pages
        ]

    def get_bytes_per_token(self) -> int:
        return SWA_BYTES_PER_TOKEN

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return int(get_tensor_size_bytes(self.k_pages)), 0


class DSV41TPUMainKVPool(DeepSeekV4SingleKVPool):
    """Per kv_source layer FP8 latent records: 3D page roots + flat kernel views.

    Root ``k_pages`` is ``int32 [num_pages, 256 // ratio, 128]`` and root
    ``scale_pages`` is ``fp32 [num_pages, 8, 128]`` (one whole TPU ``(8, 128)``
    tile per full page, ``scale_step = 1024 // (256 // ratio)``).
    """

    def __init__(
        self,
        *,
        full_size: int,
        ratio: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
    ):
        self.ratio = ratio
        self.num_pages = num_main_pages(full_size)
        self.records_per_page = FULL_PAGE_TOKENS // ratio
        self.num_records = self.num_pages * self.records_per_page
        self.scale_step = SCALE_TILE_ELEMENTS // self.records_per_page
        super().__init__(
            full_size // ratio,
            self.records_per_page,
            torch.int32,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            global_page_size=FULL_PAGE_TOKENS,
        )

    def _create_buffers(self) -> None:
        self.kv_cache_total_dim = main_bytes_per_record(self.ratio)
        self.k_pages = [
            _zeros_kv_buffer(
                (self.num_pages, self.records_per_page, WORD_LANES),
                torch.int32,
                self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.scale_pages = [
            _zeros_kv_buffer(
                (self.num_pages, SCALE_TILE_ROWS, SCALE_TILE_LANES),
                torch.float32,
                self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.kv_buffer = [p.view(self.num_records, WORD_LANES) for p in self.k_pages]
        self.scale_buffer = [
            p.view(self.num_pages * SCALE_TILE_ELEMENTS) for p in self.scale_pages
        ]

    def get_bytes_per_token(self) -> int:
        return main_bytes_per_record(self.ratio)

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return (
            int(get_tensor_size_bytes(self.k_pages))
            + int(get_tensor_size_bytes(self.scale_pages)),
            0,
        )


class DSV41TPUIndexerPool(DeepSeekV4IndexerPool):
    """Per kv_source layer packed FP8 index keys, 8-row aligned per full page.

    Root ``k_pages`` is ``int32 [num_full_pages, rows_per_full, 128]`` where
    ``rows_per_full = 72`` (r=1, ``page_stride=36``) or ``40`` (r=2,
    ``page_stride=40``).
    """

    def __init__(
        self,
        *,
        full_size: int,
        ratio: int,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
    ):
        assert index_head_dim == INDEX_HEAD_DIM, (
            f"V4.1 packed FP8 index keys have {INDEX_HEAD_DIM} channels, "
            f"got {index_head_dim}"
        )
        self.ratio = ratio
        self.num_full_pages = num_main_pages(full_size)
        self.subpages_per_full = (FULL_PAGE_TOKENS // ratio) // PAGE_RECORDS
        page_stride = index_page_stride(ratio)
        self.rows_per_full = self.subpages_per_full * page_stride
        self.layout = PackedFP8Layout(
            self.num_full_pages * self.subpages_per_full,
            page_stride=page_stride,
            page_offset=0,
            page_size=PAGE_RECORDS,
        )
        super().__init__(
            self.layout.pages * PAGE_RECORDS,
            PAGE_RECORDS,
            torch.int32,
            index_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            use_fp4_indexer=False,
            global_page_size=FULL_PAGE_TOKENS,
        )

    def _create_buffer(self) -> None:
        rows = self.layout.pages * self.layout.page_stride
        assert rows == self.num_full_pages * self.rows_per_full
        self.layout.validate((rows,))
        self.k_pages = [
            _zeros_kv_buffer(
                (self.num_full_pages, self.rows_per_full, WORD_LANES),
                torch.int32,
                self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.index_k_with_scale_buffer = [
            p.view(rows, WORD_LANES) for p in self.k_pages
        ]

    def get_bytes_per_token(self) -> int:
        return index_bytes_per_record(self.ratio)

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return int(get_tensor_size_bytes(self.k_pages)), 0


class TPUPairStatePool(CompressStatePool):
    """Ratio-2 pending-pair state with one TPU-tile-aligned 3D block per req slot.

    Root ``state_buffer`` has shape ``[num_req_slots + 1, ring_stride, feat_stride]``
    in fp32 (``ring_stride = 8``, ``feat_stride = 1024`` for ``head_dim = 512``).
    Slot ``num_req_slots`` is the read-only ``-1`` sentinel initialized once to
    ``(kv=0, score=-inf)``; ``write_rows`` drops negative slots, so it stays clean
    without ever issuing an eager in-place write after Raiden registration.
    """

    def __init__(
        self,
        *,
        num_req_slots: int,
        ring_size: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
        ratio: int = 2,
    ):
        self.num_req_slots = int(num_req_slots)
        self.head_dim = int(head_dim)
        self.ring_stride = max(
            TPU_SUBLANE_ALIGN,
            ceil_div(int(ring_size), TPU_SUBLANE_ALIGN) * TPU_SUBLANE_ALIGN,
        )
        self.feat_stride = ceil_div(2 * self.head_dim, WORD_LANES) * WORD_LANES
        super().__init__(
            size=self.num_req_slots * int(ring_size),
            ring_size=int(ring_size),
            overlap=False,
            head_dim=self.head_dim,
            dtype=dtype,
            device=device,
            enable_memory_saver=enable_memory_saver,
            ratio=ratio,
            request_scoped=True,
            online=False,
        )

    def _alloc_kv_score_buffer(
        self, *, dtype: torch.dtype, device: str, enable_memory_saver: bool
    ) -> None:
        del enable_memory_saver
        shape = (self.num_req_slots + 1, self.ring_stride, self.feat_stride)
        buf = _zeros_kv_buffer(shape, dtype, device)
        buf[:, :, self.head_dim : 2 * self.head_dim].fill_(float("-inf"))
        assert buf._base is None and buf.dtype == dtype
        self.state_buffer = buf
        # CompressStatePool.__init__ calls self.kv_score_buffer.clear(); keep a
        # 1-row CPU placeholder so the base constructor never touches HBM state.
        self.kv_score_buffer = KVAndScore(
            torch.zeros((1, 2 * self.head_dim), dtype=dtype)
        )

    def translate_from_req_position_to_state_loc(
        self, req_pool_indices: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        state_loc = req_pool_indices * self.ring_stride + positions % self.ring_size
        # Negative positions address the last row of the spare sentinel slot.
        return state_loc.masked_fill(positions < 0, -1)

    def get_state_by_state_loc(self, state_loc: torch.Tensor) -> KVAndScore:
        # Derive a fresh view from the root on every call so AOTAutograd tracks
        # reads against the donated root buffer.
        flat = self.state_buffer.view(-1, self.feat_stride)
        rows = flat[state_loc]
        if self.feat_stride > 2 * self.head_dim:
            rows = rows[..., : 2 * self.head_dim]
        return KVAndScore(rows)

    def set_state_by_state_loc(self, state_loc: torch.Tensor, value: KVAndScore):
        flat = self.state_buffer.view(-1, self.feat_stride)
        valid = state_loc >= 0
        if bool(valid.any()):
            flat[state_loc[valid], : 2 * self.head_dim] = value.kv_score[valid]

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return int(get_tensor_size_bytes(self.state_buffer)), 0


def _check_dsv41_pool_supported(
    *,
    compression_ratios: Sequence[int],
    layer_num: int,
    full_size: Optional[int],
    page_size: int,
    swa_page_size: int,
    start_layer: Optional[int],
    end_layer: Optional[int],
    enable_hisparse: bool,
    kv_layout: KVLayout,
    qk_head_dim: int,
    indexer_head_dim: int,
) -> None:
    unsupported = sorted(set(compression_ratios) - set(SUPPORTED_RATIOS))
    if unsupported:
        raise ValueError(
            f"DSV41TPUTokenToKVPool supports compress ratios {SUPPORTED_RATIOS}, "
            f"got {unsupported}"
        )
    # compress_ratios also lists the trailing MTP layers, which the target model
    # does not build: [0, layer_num) is the whole target, anything else a PP stage.
    is_stage_slice = start_layer not in (None, 0) or end_layer not in (None, layer_num)
    if layer_num > len(compression_ratios) or is_stage_slice:
        raise ValueError(
            "DSV41TPUTokenToKVPool does not support pipeline parallelism: "
            f"{layer_num=} {start_layer=} {end_layer=} of {len(compression_ratios)}"
        )
    if page_size != FULL_PAGE_TOKENS or swa_page_size != FULL_PAGE_TOKENS:
        raise ValueError(
            f"DSV41TPUTokenToKVPool requires page_size={FULL_PAGE_TOKENS} and "
            f"swa_page_size={FULL_PAGE_TOKENS}, got {page_size} and {swa_page_size}"
        )
    if full_size is not None and full_size % FULL_PAGE_TOKENS:
        raise ValueError(f"full_size={full_size} is not page aligned")
    if KVLayout.parse(kv_layout) is not KVLayout.V4:
        raise ValueError(
            f"SGLANG_DSV4_KV_LAYOUT={kv_layout} is a CUDA FlashMLA layout; "
            "the TPU V4.1 pool keeps its own record formats"
        )
    if qk_head_dim != LATENT_DIM or indexer_head_dim != INDEX_HEAD_DIM:
        raise ValueError(
            f"DSV41TPUTokenToKVPool needs a {LATENT_DIM}-dim latent and a "
            f"{INDEX_HEAD_DIM}-dim indexer, got {qk_head_dim} and {indexer_head_dim}"
        )
    exec_cfg = get_exec()
    for feature, enabled in (
        ("HiSparse", enable_hisparse),
        ("unified_kv", is_unified_kv_triton()),
        (
            "--dsv4-attn-backend trtllm",
            exec_cfg.kernel.dsv4_attn_backend == "trtllm",
        ),
    ):
        if enabled:
            raise ValueError(f"DSV41TPUTokenToKVPool does not support {feature}")
    disagg = get_disagg()
    if (
        disagg.disaggregation_mode != "null"
        and disagg.disaggregation_transfer_backend != "raiden"
    ):
        raise ValueError(
            "DSV41TPUTokenToKVPool supports P/D disaggregation only with "
            "--disaggregation-transfer-backend raiden, got "
            f"{disagg.disaggregation_transfer_backend!r}"
        )


class _DSV41TPURequestWindowMemory:
    """Built by the core pool constructor in place of ``RequestWindowMemory``:
    the window is a ``DSV41TPURequestWindow`` over the pool's own SWA buffers."""

    def __init__(self, pool, spec, *, dtype, device, enable_memory_saver=False):
        def make_window_pool(size, layers):
            return pool._make_kv_pool(
                size=size,
                page_size=spec.page_size,
                dtype=dtype,
                layer_num=layers,
                device=device,
                enable_memory_saver=enable_memory_saver,
                global_page_size=spec.page_size,
            )

        self.view = DSV41TPURequestWindow(
            make_window_pool,
            num_slots=spec.num_slots,
            layers=spec.layers,
            page_size=spec.page_size,
            capacity=spec.capacity,
            workspace_rows=spec.workspace_rows,
        )

    def reset_rows(self, rows) -> None:
        self.view.reset(rows)


class DSV41TPUTokenToKVPool(DeepSeekV4TokenToKVPool):
    """DeepSeek-V4.1 TPU pool: core control plane, Pallas-shaped buffers."""

    raiden_state_max_blocks: int = 1

    def __init__(
        self,
        *,
        compression_ratios: Sequence[int],
        layer_num: int,
        page_size: int,
        swa_page_size: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        indexer_head_dim: int,
        full_size: Optional[int] = None,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        enable_hisparse: bool = False,
        kv_layout: KVLayout | str = KVLayout.V4,
        **kwargs,
    ):
        _check_dsv41_pool_supported(
            compression_ratios=compression_ratios,
            layer_num=layer_num,
            full_size=full_size,
            page_size=page_size,
            swa_page_size=swa_page_size,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_hisparse=enable_hisparse,
            kv_layout=kv_layout,
            qk_head_dim=qk_nope_head_dim + qk_rope_head_dim,
            indexer_head_dim=indexer_head_dim,
        )
        # The core constructor builds the SWA pool first, then the main-KV pools,
        # both through _make_kv_pool.
        self._swa_pool_pending = True
        # Only register_mapping sets it; a request-window pool never gets one.
        self.full_to_swa_index_mapping = None
        # Under encoder replay the core constructor builds its request window,
        # looked up at call time, instead of a paged SWA pool.
        with patch.object(
            core_request_window,
            "RequestWindowMemory",
            partial(_DSV41TPURequestWindowMemory, self),
        ):
            super().__init__(
                compression_ratios=list(compression_ratios),
                layer_num=layer_num,
                page_size=page_size,
                swa_page_size=swa_page_size,
                qk_nope_head_dim=qk_nope_head_dim,
                qk_rope_head_dim=qk_rope_head_dim,
                indexer_head_dim=indexer_head_dim,
                full_size=full_size,
                start_layer=start_layer,
                end_layer=end_layer,
                enable_hisparse=False,
                kv_layout=kv_layout,
                **kwargs,
            )
        assert not self._swa_pool_pending, "the core pool built no SWA pool"
        if self.swa_req_ring_size is not None:
            # The attention backend builds the draft's SWA table on the host.
            self.swa_valid_from = torch.zeros(self.num_req_slots, dtype=torch.int64)
        if self.request_window is not None:
            swa_storage = "request window"
        elif self.swa_req_ring_size is not None:
            swa_storage = "draft ring"
        else:
            swa_storage = "paged"
        logger.info(
            "DSV41 TPU pool: swa_pages64=%d (%s), sources=%s, main_records=%s, "
            "index_pages=%s",
            self._swa_storage().num_pages,
            swa_storage,
            self.sources_by_ratio,
            {r: p.num_records for r, p in self.kv_pools.items() if p is not None},
            {r: p.layout.pages for r, p in self.index_pools.items()},
        )

    # ------------------------------------------------------------------ #
    # Byte sizing (configurator budget hooks)
    # ------------------------------------------------------------------ #

    @staticmethod
    def bytes_per_swa_token(*, num_layers: int) -> float:
        return float(SWA_BYTES_PER_TOKEN * num_layers)

    @staticmethod
    def bytes_per_full_token(
        *,
        swa_ratio: float,
        bytes_per_swa_token: float,
        sources_by_ratio: Mapping[int, Sequence[int]],
    ) -> float:
        """SWA share plus main KV and indexer K of every kv_source layer."""
        total = swa_ratio * bytes_per_swa_token
        for ratio, sources in sources_by_ratio.items():
            if ratio not in LOW_RATIOS:
                raise ValueError(f"no V4.1 TPU storage for compress ratio {ratio}")
            total += (
                (main_bytes_per_record(ratio) + index_bytes_per_record(ratio))
                / ratio
                * len(sources)
            )
        return total

    @staticmethod
    def c2_state_fixed_bytes(
        *,
        num_req_slots: int,
        ring_size: int,
        head_dim: int,
        num_layers: int,
    ) -> int:
        return c2_state_fixed_bytes(
            num_req_slots=num_req_slots,
            ring_size=ring_size,
            head_dim=head_dim,
            num_layers=num_layers,
        )

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        pools = [
            self._swa_storage(),
            *self.kv_pools.values(),
            *self.index_pools.values(),
        ]
        total = sum(pool.get_kv_size_bytes()[0] for pool in pools if pool is not None)
        for pool in self.compress_state_pools:
            if pool is None:
                continue
            if isinstance(pool, TPUPairStatePool):
                total += pool.get_kv_size_bytes()[0]
            else:
                total += int(get_tensor_size_bytes(pool.kv_score_buffer.kv_score))
        return total, 0

    # ------------------------------------------------------------------ #
    # Factories
    # ------------------------------------------------------------------ #

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
        del dtype, kv_layout
        assert cls is DeepSeekV4SingleKVPool, (
            f"custom KV pool class {cls.__name__} is not supported on TPU V4.1"
        )
        assert global_page_size == FULL_PAGE_TOKENS
        if self._swa_pool_pending:
            self._swa_pool_pending = False
            if self.swa_req_ring_size is not None:
                # The draft ring takes the request window's layout: RING_PAGES
                # pages per slot after the reserved ones (num_swa_pages64 adds them).
                assert self.swa_req_ring_size <= RING_TOKENS, self.swa_req_ring_size
                size = self.num_req_slots * RING_TOKENS
            return DSV41TPUSWAKVPool(
                size=size,
                layer_num=layer_num,
                device=device,
                enable_memory_saver=enable_memory_saver,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
            )
        ratio = global_page_size // page_size
        assert ratio in LOW_RATIOS and size == self.full_size // ratio, (
            f"unexpected main-KV pool {size=} {page_size=}"
        )
        return DSV41TPUMainKVPool(
            full_size=self.full_size,
            ratio=ratio,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
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
        index_k_page8: bool = False,
    ) -> DSV41TPUIndexerPool:
        # The core page size (DSV41_INDEX_PAGE_SIZE) and fp4 choice are CUDA
        # DeepGEMM limits; the TPU kernels read packed FP8 pages of 128 records.
        del page_size, dtype, global_page_size
        assert force_fp4, "only the ratio-1/2 indexer pool exists on TPU V4.1"
        # The page-8 index-K layout is a HIP (AITER) kernel option.
        assert not index_k_page8, "the page-8 index-K pool is HIP-only"
        # The core sizes it (full_size + page_size) // ratio slots.
        ratio = (self.full_size + self.page_size) // size
        assert ratio in LOW_RATIOS and size * ratio == self.full_size + self.page_size
        return DSV41TPUIndexerPool(
            full_size=self.full_size,
            ratio=ratio,
            index_head_dim=index_head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
        )

    def _make_pair_state_pool(self, enable_memory_saver: bool) -> TPUPairStatePool:
        """Ratio-2 pending-pair state: one tile-aligned 3D block per req slot."""
        return TPUPairStatePool(
            num_req_slots=self.num_req_slots,
            ring_size=self.get_ring_size(2),
            head_dim=self.qk_nope_head_dim + self.qk_rope_head_dim,
            dtype=torch.float32,
            device=self.device,
            enable_memory_saver=enable_memory_saver,
            ratio=2,
        )

    # ------------------------------------------------------------------ #
    # Request-slot lifecycle
    # ------------------------------------------------------------------ #

    def clear_request_scoped_state(self, req_pool_idx: int) -> None:
        """No-op on TPU: never eager-write ``state_buffer`` after allocation.

        On ``torch_tpu``, any eager in-place write (``zero_()`` / ``fill_()``)
        reallocates the underlying PJRT device buffer and invalidates the
        pointers registered with Raiden. ``DSV41TPUAttnBackend._pair_partners``
        masks even and padded positions to the read-only ``-1`` sentinel row,
        while every odd position reads ``pos - 1`` written by the same request
        (or transferred from prefill), so stale rows in a reused slot are never
        observed.
        """
        del req_pool_idx
        return

    def clear_request_slots(self, req_pool_indices: Sequence[int]) -> None:
        """ReqToTokenPool on_alloc_rows hook (no-op on TPU; see above)."""
        for req_pool_idx in req_pool_indices:
            self.clear_request_scoped_state(int(req_pool_idx))

    # ------------------------------------------------------------------ #
    # Kernel buffer accessors (absolute layer ids)
    # ------------------------------------------------------------------ #

    def set_swa_valid_from(self, slots: torch.Tensor, valid_from: torch.Tensor) -> None:
        self.swa_valid_from[slots.cpu()] = valid_from.cpu()

    def swa_ring_rows(
        self,
        *,
        slots: torch.Tensor,
        positions: torch.Tensor,
        ends: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """int64 draft ring rows of (slot, position), broadcast. With ``ends`` (each
        token's request end), -1 where a later token of one write takes the row."""
        positions = positions.to(torch.int64)
        rows = ring_rows(slots.to(torch.int64), positions)
        if ends is None:
            return rows
        return torch.where(positions >= ends - RING_TOKENS, rows, -1)

    def _swa_storage(self) -> DSV41TPUSWAKVPool:
        """The paged SWA pool, or the request window's buffers under encoder replay."""
        if self.request_window is not None:
            return self.request_window.state
        return self.swa_kv_pool

    def get_swa_root(self, layer_id: int) -> torch.Tensor:
        """Base bf16 root ``[num_swa_pages256, 256, 512]`` (``_base is None``)."""
        self.wait_layer_transfer(layer_id)
        return self._swa_storage().k_pages[self._swa_local_layer_id(layer_id)]

    def get_swa_pages(self, layer_id: int) -> torch.Tensor:
        """Fresh bf16 view ``[num_swa_pages64, 64, 512]`` derived from root."""
        return self.get_swa_root(layer_id).view(-1, SWA_PAGE_TOKENS, LATENT_DIM)

    def _low_ratio_item(self, layer_id: int) -> Tuple[int, int]:
        item = self.layer_mapping[layer_id]
        assert item is not None and item.compress_ratio in LOW_RATIOS, (
            f"layer {layer_id} has no compressed storage"
        )
        return item.compress_ratio, item.compress_layer_id

    def index_source(self, layer_id: int) -> Tuple[int, int]:
        """(compress ratio, kv_source index) of the layer's index keys."""
        return self._low_ratio_item(layer_id)

    def get_main_kv_roots(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Base ``(int32 [P, 256 // r, 128], fp32 [P, 8, 128])`` roots."""
        self.wait_layer_transfer(layer_id)
        ratio, source_id = self._low_ratio_item(layer_id)
        pool = self.kv_pools[ratio]
        return pool.k_pages[source_id], pool.scale_pages[source_id]

    def get_main_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fresh ``(int32 words [R, 128], fp32 scales [P * 1024])`` views."""
        words_root, scales_root = self.get_main_kv_roots(layer_id)
        return words_root.view(-1, WORD_LANES), scales_root.view(-1)

    def get_index_k_root(self, layer_id: int) -> torch.Tensor:
        """Base packed FP8 root ``int32 [P_full, rows_per_full, 128]``."""
        self.wait_layer_transfer(layer_id)
        ratio, source_id = self._low_ratio_item(layer_id)
        return self.index_pools[ratio].k_pages[source_id]

    def get_index_k(self, layer_id: int) -> torch.Tensor:
        """Fresh packed FP8 view ``int32 [P * page_stride, 128]``."""
        return self.get_index_k_root(layer_id).view(-1, WORD_LANES)

    def index_k_layout(self, layer_id: int) -> Tuple[int, int, int, int]:
        """PackedFP8Layout fields (pages, page_stride, page_offset, page_size)."""
        ratio, _ = self._low_ratio_item(layer_id)
        layout = self.index_pools[ratio].layout
        return layout.pages, layout.page_stride, layout.page_offset, layout.page_size

    def main_page_records(self, layer_id: int) -> int:
        ratio, _ = self._low_ratio_item(layer_id)
        return self.kv_pools[ratio].page_size

    # ------------------------------------------------------------------ #
    # direct_compile KVPool protocol (sglang_tpu/compilation/direct_compile.py)
    # ------------------------------------------------------------------ #

    def _state_slots(self) -> list[tuple[object, str, Optional[int]]]:
        """(owner, attribute, list index or None) of every root tensor the forward
        reads or writes, in a fixed order: SWA page roots, then per ratio main KV
        word roots, main KV scale roots and indexer K roots, then each compress-state
        ring root."""
        slots: list[tuple[object, str, Optional[int]]] = []

        def add_list(owner, attr):
            slots.extend((owner, attr, i) for i in range(len(getattr(owner, attr))))

        add_list(self._swa_storage(), "k_pages")
        for ratio in sorted(self.kv_pools):
            pool = self.kv_pools[ratio]
            if pool is not None:
                add_list(pool, "k_pages")
                add_list(pool, "scale_pages")
        for ratio in sorted(self.index_pools):
            add_list(self.index_pools[ratio], "k_pages")
        for pool in (*self.compress_state_pools, *self.indexer_compress_state_pools):
            if pool is None:
                continue
            if isinstance(pool, TPUPairStatePool):
                slots.append((pool, "state_buffer", None))
            else:
                slots.append((pool.kv_score_buffer, "kv_score", None))
        return slots

    def get_state(self) -> list[torch.Tensor]:
        return [
            getattr(owner, attr) if index is None else getattr(owner, attr)[index]
            for owner, attr, index in self._state_slots()
        ]

    def set_state(self, state: list[torch.Tensor]) -> None:
        slots = self._state_slots()
        if len(state) != len(slots):
            raise ValueError(
                f"DSV41TPUTokenToKVPool.set_state expects {len(slots)} tensors, "
                f"got {len(state)}"
            )
        for (owner, attr, index), tensor in zip(slots, state):
            if index is None:
                setattr(owner, attr, tensor)
            else:
                getattr(owner, attr)[index] = tensor

    # ------------------------------------------------------------------ #
    # Prefill/Decode Disaggregation (Raiden) buffer descriptors
    # ------------------------------------------------------------------ #

    def raiden_full_buffers(self) -> List[torch.Tensor]:
        """Deterministic list of 3D full-page roots (uniform ``dim[0]``)."""
        bufs: List[torch.Tensor] = []
        for ratio in LOW_RATIOS:
            kv_pool = self.kv_pools.get(ratio)
            if kv_pool is not None:
                bufs.extend(kv_pool.k_pages)
                bufs.extend(kv_pool.scale_pages)
            index_pool = self.index_pools.get(ratio)
            if index_pool is not None:
                bufs.extend(index_pool.k_pages)
        return bufs

    def raiden_swa_buffers(self) -> List[torch.Tensor]:
        """Per-layer 3D 256-token SWA page roots (paged pool or encoder replay window)."""
        swa_storage = self._swa_storage()
        if swa_storage is None:
            return []
        return list(swa_storage.k_pages)

    def raiden_state_buffers(self) -> List[torch.Tensor]:
        """Per ratio-2 kv_source 3D pair-state roots (one block per req slot)."""
        return [
            pool.state_buffer
            for pool in self.compress_state_pools
            if isinstance(pool, TPUPairStatePool)
        ]

    @staticmethod
    def _buf_infos(
        buffers: Sequence[torch.Tensor],
    ) -> Tuple[List[int], List[int], List[int]]:
        data_ptrs: List[int] = []
        data_lens: List[int] = []
        item_lens: List[int] = []
        for buf in buffers:
            data_ptrs.append(buf.data_ptr())
            data_lens.append(buf.nbytes)
            item_lens.append(buf[0].nbytes)
        return data_ptrs, data_lens, item_lens

    def get_contiguous_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        return self._buf_infos(self.raiden_full_buffers())

    def get_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        return self._buf_infos(self.raiden_swa_buffers())

    def get_request_state_buf_infos(self) -> Tuple[List[int], List[int], List[int]]:
        return self._buf_infos(self.raiden_state_buffers())

    def request_state_transfer_indices(
        self, req_pool_idx: int, seq_len: int = 0
    ) -> np.ndarray:
        """Always return ``[req_pool_idx]`` (1 block) for Raiden's fixed state engine."""
        del seq_len
        req_idx = int(req_pool_idx)
        assert 0 <= req_idx < self.num_req_slots, (
            f"req_pool_idx={req_idx} out of range [0, {self.num_req_slots})"
        )
        return np.array([req_idx], dtype=np.int32)


# ---------------------------------------------------------------------- #
# Configurator hook bodies (registered in sglang_tpu/models/deepseek_v41/hooks.py)
#
# sglang_tpu/hooks/dsv4.py wraps the same targets for V4. Register these after
# it so they are the outer wrapper: for a V4.1 model they bypass the V4 hooks,
# otherwise they delegate to them through ``original``.
# ---------------------------------------------------------------------- #


def build_tpu_dsv41_kv_pool(original, kvc, *args, **kwargs):
    """AROUND KVCacheConfigurator._build_dsv4_kv_pool: build DSV41TPUTokenToKVPool
    for a V4.1 model and reset ratio-2 pair state whenever a request slot is
    (re)allocated. Other DeepSeek-V4 models fall through to ``original``."""
    import sglang.srt.mem_cache.kv_cache_configurator as kvc_mod

    # Keyed on the target's ratios, so a V4.1 draft worker gets this pool too.
    if not uses_dsv41_pool(kvc.model_config.compress_ratios):
        return original(kvc, *args, **kwargs)
    # The V4 TPU hook would rebind the pool class and register its own slot reset.
    core_build = inspect.unwrap(original)
    with patch.object(kvc_mod, "DeepSeekV4TokenToKVPool", DSV41TPUTokenToKVPool):
        pool = core_build(kvc, *args, **kwargs)
    # The draft pool has only ratio-0 layers and shares this req pool.
    if not kvc.is_draft_worker:
        kwargs["req_to_token_pool"].register_on_alloc_rows(pool.clear_request_slots)
    return pool


def get_tpu_dsv41_bytes_per_swa_token(original, configurator) -> float:
    """AROUND DSV4PoolConfigurator._get_bytes_per_swa_token. Under encoder replay
    only the DSpark draft keeps paged SWA, priced at the TPU bf16 page row."""
    if not uses_dsv41_pool(configurator.stage_compress_ratios):
        return original(configurator)
    layers = (
        configurator.paged_draft_layers
        if configurator.encoder_replay
        else configurator.num_layers_total
    )
    return DSV41TPUTokenToKVPool.bytes_per_swa_token(num_layers=layers)


def get_tpu_dsv41_bytes_per_full_token(original, configurator) -> float:
    """AROUND DSV4PoolConfigurator._get_bytes_per_full_token."""
    if not uses_dsv41_pool(configurator.stage_compress_ratios):
        return original(configurator)
    swa_ratio = (
        0
        if configurator._unified or configurator.swa_cap_tokens is not None
        else configurator.swa_ratio
    )
    return DSV41TPUTokenToKVPool.bytes_per_full_token(
        swa_ratio=swa_ratio,
        bytes_per_swa_token=configurator.bytes_per_swa_token,
        sources_by_ratio=configurator.stage_owner_layers,
    )


def window_bytes(*, num_slots: int, chunk_tokens: int, num_layers: int) -> int:
    """SWA bytes of the TPU request window (reserved pages, rings, workspace)."""
    pages = window_buffer_pages(num_slots=num_slots, chunk_tokens=chunk_tokens)
    return pages * SWA_PAGE_TOKENS * SWA_BYTES_PER_TOKEN * num_layers


def init_tpu_dsv41_pool_configurator(original, configurator, kvc) -> None:
    """AROUND DSV4PoolConfigurator.__init__: keep the DSpark draft's ring layer
    count, which core computes as a local, for get_tpu_dsv41_swa_fixed_bytes."""
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import uses_draft_swa_ring

    original(configurator, kvc)
    draft_layers = 0
    if configurator.encoder_replay and kvc.spec_algorithm.is_dspark():
        draft_layers = int(kvc.spec_aux_config.dflash_draft_num_layers or 0)
    # Same rule as core: a ring-kept draft is all the draft layers not paged.
    configurator.draft_ring_layers = draft_layers if uses_draft_swa_ring() else 0
    assert (
        configurator.paged_draft_layers == draft_layers - configurator.draft_ring_layers
    )


def get_tpu_dsv41_swa_fixed_bytes(original, configurator) -> float:
    """AROUND DSV4PoolConfigurator._get_swa_fixed_bytes: price the request window
    as the TPU window's bf16 pages instead of core's FlashMLA-packed rows."""
    fixed = original(configurator)
    if (
        not uses_dsv41_pool(configurator.stage_compress_ratios)
        or not configurator.encoder_replay
    ):
        return fixed
    tpu_window = window_bytes(
        num_slots=configurator.requested_max_running_requests_per_worker + 1,
        chunk_tokens=get_schedule().chunked_prefill_size,
        num_layers=configurator.num_layers_total,
    )
    slots = configurator.requested_max_running_requests_per_worker + 1
    draft_ring = (
        (RESERVED_PAGES + RING_PAGES * slots)
        * SWA_PAGE_TOKENS
        * SWA_BYTES_PER_TOKEN
        * configurator.draft_ring_layers
    )
    return fixed - configurator.request_window_bytes + tpu_window + draft_ring


def get_tpu_dsv41_c2_state_fixed_bytes(
    original, configurator, max_running_requests: int
) -> int:
    """AROUND DSV4PoolConfigurator._get_c2_state_fixed_bytes."""
    if not uses_dsv41_pool(configurator.stage_compress_ratios):
        return original(configurator, max_running_requests)
    num_layers = configurator.num_layers(2)
    if num_layers == 0:
        return 0
    return DSV41TPUTokenToKVPool.c2_state_fixed_bytes(
        num_req_slots=configurator._get_num_req_slots(max_running_requests),
        ring_size=configurator.ring_sizes[2],
        head_dim=configurator.attn_head_dim,
        num_layers=num_layers,
    )


def derive_tpu_dsv41_pool_sizes(original, kvc, *, config):
    """AROUND KVCacheConfigurator._derive_pool_sizes. Under encoder replay the core
    sizes the paged SWA pool at 0 tokens (SWA lives in the request window), which
    the TPU 8-page SWA floor refuses; skip the floor for that case only."""
    import dataclasses

    import msgspec

    from sglang.srt.configs.model_config import is_deepseek_v4

    # Wraps every model's sizing; ModelConfig sets compress_ratios only for V4.
    if (
        kvc.device != "tpu"
        or not is_deepseek_v4(kvc.model_config.hf_config)
        or not uses_dsv41_pool(kvc.model_config.compress_ratios)
        or not get_exec().features.enable_encoder_swa_bounded_replay
        or config.swa_max_total_num_tokens != 0
    ):
        return original(kvc, config=config)
    sizes = original(
        kvc, config=dataclasses.replace(config, swa_max_total_num_tokens=None)
    )
    return msgspec.structs.replace(sizes, swa_max_total_num_tokens=0)


def build_tpu_dsv41_allocator(original, kvc, **kwargs):
    """AROUND KVCacheConfigurator._build_token_to_kv_pool_allocator. A V4.1 pool
    without a paged SWA side reaches the core PagedTokenToKVPoolAllocator arm,
    whose alloc_extend/alloc_decode launch Triton; take the TPU paged allocator."""
    import sglang.srt.mem_cache.kv_cache_configurator as kvc_mod
    from sglang.srt.platforms import current_platform

    pool = kwargs["token_to_kv_pool"]
    if (
        kvc.device != "tpu"
        or kwargs["token_to_kv_pool_allocator"] is not None
        or not isinstance(pool, DSV41TPUTokenToKVPool)
        or pool.needs_paged_swa_allocator
    ):
        return original(kvc, **kwargs)
    with patch.object(
        kvc_mod,
        "PagedTokenToKVPoolAllocator",
        current_platform.get_paged_allocator_cls(),
    ):
        return original(kvc, **kwargs)

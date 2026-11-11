# SPDX-License-Identifier: Apache-2.0
"""TPU KV pool for DeepSeek-V4.1 (compress ratios 0, 1 and 2).

Keeps the core DeepSeekV4TokenToKVPool control plane (full and SWA allocators,
full_to_swa_index_mapping, kv_source layer sharing, ratio-2 pair state) and swaps
only the physical buffers for the shapes the V4.1 Pallas kernels read:

- SWA, every layer: int32 [ceil((swa_size + 256) / 64) * 128, 128];
  two word rows per BF16 token, page = swa_loc // 64. Flat base owners are
  reshaped to physical pages only inside JAX, preserving TorchTPU donation.
- Main KV, kv_source layers: FP8 latent words int32 [R, 128] + fp32 scales [R];
  record = full_loc // r, R = ceil((full_size + 256) / 256) * (256 // r), so a main
  page holds 256 // r records and its id is the full page id.
- Indexer K, kv_source layers: packed FP8 int32 [P * 33, 128] with
  PackedFP8Layout(P, page_size=128); record = full_loc // r, P = R // 128.

The allocators reserve logical page 0, so SWA pages 0..3, main page 0 and the
indexer pages of full page 0 are never written. Every buffer is a base tensor
(`_base is None`) so torch_tpu can donate it to Pallas calls.
"""

from __future__ import annotations

import inspect
import logging
from typing import Mapping, Optional, Sequence, Tuple
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
    is_unified_kv_triton,
)
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)
from sglang.srt.mem_cache.memory_pool import get_tensor_size_bytes
from sglang.srt.runtime_context import get_exec
from sglang.srt.utils import ceil_div
from sglang_tpu.dsv4.dsv4_memory_pool import _zeros_kv_buffer
from sglang_tpu.kernels.deepseek_v41.index_selection.fp8.cache import (
    PAGE_RECORDS,
    PAGE_ROWS,
    PackedFP8Layout,
)

logger = logging.getLogger(__name__)

FULL_PAGE_TOKENS = 256
# CSA2 accepts SWA pages of 64; the fused SWA publish runs only at 64.
SWA_PAGE_TOKENS = 64
LATENT_DIM = 512
INDEX_HEAD_DIM = 128
WORD_LANES = 128

SWA_BYTES_PER_TOKEN = LATENT_DIM * torch.bfloat16.itemsize
# One 512 B FP8 record (448 e4m3 + 64 e8m0) plus its fp32 scale companion.
MAIN_BYTES_PER_RECORD = WORD_LANES * 4 + 4
# 128 FP8 values + one fp32 scale: 33 word rows per 128-record page.
INDEX_BYTES_PER_RECORD = PAGE_ROWS * WORD_LANES * 4 // PAGE_RECORDS

LOW_RATIOS = (1, 2)
SUPPORTED_RATIOS = (0, *LOW_RATIOS)


def num_swa_pages64(swa_size: int) -> int:
    # The SWA allocator reserves one logical page, so the top swa_loc is swa_size + 255.
    return ceil_div(swa_size + FULL_PAGE_TOKENS, SWA_PAGE_TOKENS)


def num_main_records(full_size: int, ratio: int) -> int:
    full_pages = ceil_div(full_size + FULL_PAGE_TOKENS, FULL_PAGE_TOKENS)
    return full_pages * (FULL_PAGE_TOKENS // ratio)


def num_index_pages(full_size: int, ratio: int) -> int:
    return num_main_records(full_size, ratio) // PAGE_RECORDS


def uses_dsv41_pool(compress_ratios: Sequence[int]) -> bool:
    """Whether a DeepSeek-V4 family model needs this pool (ratio 1 or 2 layers)."""
    return any(ratio in LOW_RATIOS for ratio in compress_ratios)


class DSV41TPUSWAKVPool(DeepSeekV4SingleKVPool):
    """Per-layer flat SWA word owners [num_pages * 128, 128].

    Matches epic/dsv41 dc1cd126e's paired-token BF16 packing. The logical
    allocator and byte budget are unchanged; the JAX bridge forms page views.
    """

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
        self.num_pages = num_swa_pages64(size)
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
        # A pair of 512-feature BF16 tokens occupies four INT32[128]
        # rows. Keep the owner flat: passing a Torch view to a donated op
        # would prevent TorchTPU from safely aliasing the original allocation.
        shape = (self.num_pages * self.page_size * 2, WORD_LANES)
        self.kv_buffer = [
            _zeros_kv_buffer(shape, torch.int32, self.device)
            for _ in range(self.layer_num)
        ]

    def get_bytes_per_token(self) -> int:
        return SWA_BYTES_PER_TOKEN

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return int(get_tensor_size_bytes(self.kv_buffer)), 0


class DSV41TPUMainKVPool(DeepSeekV4SingleKVPool):
    """Per kv_source layer FP8 latent records: int32 words [R, 128] + fp32 [R]."""

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
        self.num_records = num_main_records(full_size, ratio)
        super().__init__(
            full_size // ratio,
            FULL_PAGE_TOKENS // ratio,
            torch.int32,
            qk_nope_head_dim,
            qk_rope_head_dim,
            layer_num,
            device,
            enable_memory_saver,
            global_page_size=FULL_PAGE_TOKENS,
        )

    def _create_buffers(self) -> None:
        self.kv_cache_total_dim = MAIN_BYTES_PER_RECORD
        self.kv_buffer = [
            _zeros_kv_buffer((self.num_records, WORD_LANES), torch.int32, self.device)
            for _ in range(self.layer_num)
        ]
        self.scale_buffer = [
            _zeros_kv_buffer((self.num_records,), torch.float32, self.device)
            for _ in range(self.layer_num)
        ]

    def get_bytes_per_token(self) -> int:
        return MAIN_BYTES_PER_RECORD

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return (
            int(get_tensor_size_bytes(self.kv_buffer))
            + int(get_tensor_size_bytes(self.scale_buffer)),
            0,
        )


class DSV41TPUIndexerPool(DeepSeekV4IndexerPool):
    """Per kv_source layer packed FP8 index keys int32 [P * 33, 128]."""

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
        self.layout = PackedFP8Layout(
            num_index_pages(full_size, ratio), page_size=PAGE_RECORDS
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
        self.layout.validate((rows,))
        self.index_k_with_scale_buffer = [
            _zeros_kv_buffer((rows, WORD_LANES), torch.int32, self.device)
            for _ in range(self.layer_num)
        ]

    def get_bytes_per_token(self) -> int:
        return INDEX_BYTES_PER_RECORD

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        return int(get_tensor_size_bytes(self.index_k_with_scale_buffer)), 0


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
            "encoder SWA bounded replay (request window)",
            exec_cfg.features.enable_encoder_swa_bounded_replay,
        ),
        (
            "--dsv4-attn-backend trtllm",
            exec_cfg.kernel.dsv4_attn_backend == "trtllm",
        ),
    ):
        if enabled:
            raise ValueError(f"DSV41TPUTokenToKVPool does not support {feature}")


class DSV41TPUTokenToKVPool(DeepSeekV4TokenToKVPool):
    """DeepSeek-V4.1 TPU pool: core control plane, Pallas-shaped buffers."""

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
        logger.info(
            "DSV41 TPU pool: swa_pages64=%d, sources=%s, main_records=%s, index_pages=%s",
            self.swa_kv_pool.num_pages,
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
                (MAIN_BYTES_PER_RECORD + INDEX_BYTES_PER_RECORD) / ratio * len(sources)
            )
        return total

    def get_kv_size_bytes(self) -> Tuple[int, int]:
        pools = [self.swa_kv_pool, *self.kv_pools.values(), *self.index_pools.values()]
        total = sum(pool.get_kv_size_bytes()[0] for pool in pools if pool is not None)
        total += sum(
            int(get_tensor_size_bytes(pool.kv_score_buffer.kv_score))
            for pool in self.compress_state_pools
            if pool is not None
        )
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
    ) -> DSV41TPUIndexerPool:
        # The core page size (DSV41_INDEX_PAGE_SIZE) and fp4 choice are CUDA
        # DeepGEMM limits; the TPU kernels read packed FP8 pages of 128 records.
        del page_size, dtype, global_page_size
        assert force_fp4, "only the ratio-1/2 indexer pool exists on TPU V4.1"
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

    # ------------------------------------------------------------------ #
    # Request-slot lifecycle
    # ------------------------------------------------------------------ #

    def clear_request_slots(self, req_pool_indices: Sequence[int]) -> None:
        """ReqToTokenPool on_alloc_rows hook: reset each reused slot's ratio-2
        pending-pair ring through the core clear_request_scoped_state."""
        for req_pool_idx in req_pool_indices:
            self.clear_request_scoped_state(int(req_pool_idx))

    # ------------------------------------------------------------------ #
    # Kernel buffer accessors (absolute layer ids)
    # ------------------------------------------------------------------ #

    def get_swa_pages(self, layer_id: int) -> torch.Tensor:
        """Base INT32 [num_swa_pages64 * 128, 128] owner (not a page view)."""
        self.wait_layer_transfer(layer_id)
        return self.swa_kv_pool.kv_buffer[self._swa_local_layer_id(layer_id)]

    def _low_ratio_item(self, layer_id: int) -> Tuple[int, int]:
        item = self.layer_mapping[layer_id]
        assert item is not None and item.compress_ratio in LOW_RATIOS, (
            f"layer {layer_id} has no compressed storage"
        )
        return item.compress_ratio, item.compress_layer_id

    def get_main_kv(self, layer_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """(int32 words [R, 128], fp32 scales [R]) of the layer's kv_source."""
        self.wait_layer_transfer(layer_id)
        ratio, source_id = self._low_ratio_item(layer_id)
        pool = self.kv_pools[ratio]
        return pool.kv_buffer[source_id], pool.scale_buffer[source_id]

    def get_index_k(self, layer_id: int) -> torch.Tensor:
        """Packed FP8 int32 [P * 33, 128] of the layer's kv_source."""
        self.wait_layer_transfer(layer_id)
        ratio, source_id = self._low_ratio_item(layer_id)
        return self.index_pools[ratio].index_k_with_scale_buffer[source_id]

    def index_k_layout(self, layer_id: int) -> Tuple[int, int, int, int]:
        """PackedFP8Layout fields (pages, page_stride, page_offset, page_size)."""
        ratio, _ = self._low_ratio_item(layer_id)
        layout = self.index_pools[ratio].layout
        return layout.pages, layout.page_stride, layout.page_offset, layout.page_size

    def main_page_records(self, layer_id: int) -> int:
        ratio, _ = self._low_ratio_item(layer_id)
        return self.kv_pools[ratio].page_size


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
    """AROUND DSV4PoolConfigurator._get_bytes_per_swa_token."""
    if not uses_dsv41_pool(configurator.stage_compress_ratios):
        return original(configurator)
    return DSV41TPUTokenToKVPool.bytes_per_swa_token(
        num_layers=configurator.num_layers_total
    )


def get_tpu_dsv41_bytes_per_full_token(original, configurator) -> float:
    """AROUND DSV4PoolConfigurator._get_bytes_per_full_token. The core ratio-2
    pair-state budget (_get_c2_state_fixed_bytes) already matches the core pool
    this class keeps, and the V4 c4/c128 state hooks return 0 without those layers."""
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

# SPDX-License-Identifier: Apache-2.0
"""PCP-sharded variant of the TPU fused KV pool.

Under Prefill-Context-Parallel (see :mod:`sglang.srt.layers.cp.pcp_streaming`)
each CP rank owns only ``1/cp_group_size`` of the paged KV cache: rank ``r`` owns
page slot ``i`` iff ``i % cp_group_size == r``. The PCP streaming kernel's
striped writer handles the stride internally, so from
the pool's perspective the only difference from the non-CP fused pool is
that each rank's storage is ``1/cp_group_size`` the full size.

Design decisions (mirrors ``kv_pool.py``'s ``TpuFusedMHATokenToKVPool`` API
so callers can swap classes transparently):

* Subclass, not sibling. Only ``num_blocks`` and ``get_kv_size_bytes``
  materially change; all layout accessors (``get_key_buffer`` /
  ``get_value_buffer`` / ``get_fused_kv_buffer``) inherit unchanged.

* ``set_kv_buffer`` continues to raise. Under PCP the PCP kernel's strided
  writer performs all KV writes in place during the query phase — any
  out-of-band ``set_kv_buffer`` would race the kernel and silently
  scatter data into the wrong pages.

* Page ownership is driven by the PCP kernel, not the pool. The allocator
  hands out globally-consistent page ids across ranks (rank 0 coordinates);
  each rank's kernel-side write / read uses those global ids modulo
  ``cp_group_size`` to compute local-slot offsets. The pool's job is only
  to reserve ``floor(size / (page_size * cp_group_size))`` blocks per rank
  (+ the parent class's one pad block). Sizes that don't land on a
  ``page_size * cp_group_size`` boundary are auto-rounded DOWN with a
  warning: the caller-computed pool size can be a few tokens larger than
  a clean per-rank shard, and dropping those extra tokens is preferable to
  either failing init or over-provisioning some ranks.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool

logger = logging.getLogger(__name__)


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


class TpuPcpFusedMHATokenToKVPool(TpuFusedMHATokenToKVPool):
    """TpuFusedMHATokenToKVPool with per-rank strided page ownership.

    Constructor takes an extra ``cp_group_size`` — the PCP world size.
    Every other API mirrors the non-CP fused pool.

    Rank ``r`` (``cp_rank``, discovered via the CP process group at
    forward time) owns page slots ``[i for i in range(total_pages) if i %
    cp_group_size == r]``. The kernel's ``_update_kv_cache_partial``
    consumes the caller's ``cp_rank`` scalar and translates global
    ``page_indices`` into local-slot offsets internally — the pool does
    not need to know the rank identity.
    """

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
        cp_group_size: int = 1,
    ):
        if cp_group_size < 1:
            raise ValueError(
                f"cp_group_size must be >= 1 for TpuPcpFusedMHATokenToKVPool, "
                f"got {cp_group_size!r}"
            )
        # Caller contract: ``size`` is the GLOBAL logical capacity (across
        # all cp_group ranks). Per-rank size is derived below by dividing
        # by cp_group_size. If size < page_size * cp_group_size, the
        # per-rank allocation would round to 0 pages, so the caller either
        # (a) passed per-rank size by mistake or (b) allocated below the
        # PCP minimum. Fail loudly here rather than in the parent's
        # buffer-alloc path.
        min_size = page_size * cp_group_size
        if size < min_size:
            raise ValueError(
                f"TpuPcpFusedMHATokenToKVPool size={size} is below the PCP "
                f"minimum page_size*cp_group_size={min_size}. The caller "
                "must pass GLOBAL logical capacity (per-rank size is "
                "derived from cp_group_size internally). Check the "
                "get_mha_kv_pool_extra_kwargs path in tpu_platform.py + "
                "model_runner_kv_cache_mixin."
            )
        if page_size % cp_group_size != 0:
            # The CP kernel's strided writer requires ``bkv_sz % cp_group_size == 0``,
            # which in turn requires ``page_size % cp_group_size == 0`` because
            # ``bkv_sz`` is always a multiple of ``page_size``. Reject up front so
            # the failure is at pool-construction time (loud) instead of at first
            # attention (silent-wrong or a cryptic Mosaic assert).
            raise ValueError(
                f"page_size ({page_size}) must be divisible by cp_group_size "
                f"({cp_group_size}) for the PCP-sharded pool. Legal cp_group_size "
                f"values for page_size={page_size}: "
                f"{[c for c in (1, 2, 4, 8, 16, 32, 64) if page_size % c == 0]}."
            )

        self.cp_group_size = int(cp_group_size)

        # NB: parent ``__init__`` computes ``num_blocks = size / page_size + 1``,
        # then allocates the full-size buffer via ``_create_buffers``. Rather
        # than override its arithmetic mid-flight, we call it with a
        # per-rank ``size`` so the whole chain (``num_blocks``,
        # ``_create_buffers``, ``_finalize_allocation_log``) sees the reduced
        # storage naturally. ``size`` is required to be a multiple of
        # ``page_size`` by the parent, and we additionally require it to be a
        # multiple of ``page_size * cp_group_size`` so per-rank slots are whole.
        # Auto-round DOWN when the caller-computed size doesn't align:
        # sglang's memory-fraction pool sizing is unaware of ``cp_group_size``,
        # so at large context lengths (e.g. 1M on the 480B PCP-4 config) the
        # derived size lands on a page boundary but not on the cp-aligned one.
        # Losing a few tokens is preferable to a hard init failure; log a
        # warning so we notice if the drift is anomalously large.
        aligned_size = (size // (page_size * cp_group_size)) * (
            page_size * cp_group_size
        )
        if aligned_size != size:
            _dropped = size - aligned_size
            _pct = 100.0 * _dropped / max(1, size)
            logger.warning(
                "[TPU/PCP] rounded pool size %d -> %d "
                "(dropped %d tokens, %.4f%%) to satisfy "
                "size %% (page_size=%d * cp_group_size=%d) == 0",
                size,
                aligned_size,
                _dropped,
                _pct,
                page_size,
                cp_group_size,
            )
        per_rank_size = aligned_size // cp_group_size

        super().__init__(
            size=per_rank_size,
            page_size=page_size,
            dtype=dtype,
            head_num=head_num,
            head_dim=head_dim,
            layer_num=layer_num,
            device=device,
            enable_memory_saver=enable_memory_saver,
            v_head_dim=v_head_dim,
            swa_head_num=swa_head_num,
            swa_head_dim=swa_head_dim,
            swa_v_head_dim=swa_v_head_dim,
            start_layer=start_layer,
            end_layer=end_layer,
            enable_alt_stream=enable_alt_stream,
            enable_kv_cache_copy=enable_kv_cache_copy,
        )

        logger.info(
            "TpuPcpFusedMHATokenToKVPool: cp_group_size=%d, per-rank size=%d "
            "(global size=%d), num_blocks=%d, page_size=%d",
            self.cp_group_size,
            per_rank_size,
            size,
            self.num_blocks,
            self.page_size,
        )

    # `set_kv_buffer` continues to raise (inherited from parent). Documented
    # at :attr:`TpuFusedMHATokenToKVPool.set_kv_buffer`; under PCP the same
    # invariant holds — the CP kernel owns all writes.

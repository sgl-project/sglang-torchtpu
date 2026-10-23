"""Shared helpers for the raiden prefill/decode disaggregation backend.

Holds what both sides of the transport need: the package logger, the port and
chunk-size constants, the peer-endpoint formatter, the KV-cache grouping that
turns a device pool into the per-namespace tensor lists raiden registers, and
the cache layouts push-resharding is validated for.
"""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

import torch

from sglang.srt.disaggregation.base.conn import StateType
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, KVCache
from sglang.srt.utils.network import NetworkAddress
from sglang_tpu.disaggregation.transfer_layout import PageSharding
from sglang_tpu.dsv4.dsv4_memory_pool import (
    _TPU_STATE_SLOT_STRIDE_PAGES,
    DSV4TPUTokenToKVPool,
    TPUCompressStatePool,
    TPUDeepSeekV4CSAKVPool,
    TPUDeepSeekV4HCAKVPool,
    TPUDeepSeekV4IndexerPool,
    TPUDeepSeekV4SWAKVPool,
)
from sglang_tpu.mamba_pool import TpuMambaPool

# One logger for the whole package, deliberately keeping the name of the module
# this code used to live in: emitted record names and the existing test
# monkeypatch target "…raiden.conn.logger" both depend on it.
logger = logging.getLogger("sglang_tpu.disaggregation.raiden.conn")

# Pages are announced to the decode in pieces of at least this many tokens. A
# raiden read costs about the same whether it moves 16 pages or 200, so
# announcing every prefill step would spend more on rendezvous than it hides;
# pieces of this size keep the overlap while costing a few reads per request.
CHUNK_ANNOUNCE_TOKENS = 8192

# Ports each rank reserves: control+data for the full, SWA and GDN state engines.
PORTS_PER_RANK = 6
SWA_PORT_OFFSET = 2
STATE_PORT_OFFSET = 4

# Base of the per-rank control ports (`base + 6*local_rank`). Not 30000: that
# puts rank 0 on the default HTTP serving port, and the engine wins the bind --
# uvicorn then dies with EADDRINUSE and takes the server down before any
# transfer is attempted.
DEFAULT_KV_PORT = 31000


def _tcp_endpoint(ip: str, port: int) -> str:
    """``tcp://`` endpoint of a peer rank's rendezvous socket.

    Peer addresses travel as bare literals (``rank_ip`` from the bootstrap
    server, the ip frame of PULL_REQ / RESHARD_REQ); an IPv6 literal must be
    bracketed before a port is appended or the port merges into the address.
    The bracket is also what ``_send_multipart_locked`` keys the IPv6 socket
    option on.
    """
    return NetworkAddress(host=ip, port=int(port)).to_tcp()


@dataclass(frozen=True)
class RaidenKVCacheGroups:
    """Page-compatible cache groups registered with independent Raiden engines."""

    full: List[Any]
    swa: List[Any]
    full_layer_ids: List[int] = field(default_factory=list)
    swa_layer_ids: List[int] = field(default_factory=list)
    # Trailing tensors of full / swa that belong to a separate draft pool.
    num_draft_full: int = 0
    num_draft_swa: int = 0
    # Per-layer request-state tensors (GDN/Mamba rows or DSV4 compressor rings).
    state: List[torch.Tensor] = field(default_factory=list)
    state_type: StateType = StateType.MAMBA
    state_max_blocks: int = 1
    # Maps a request's mamba slot to the state row a transfer moves.
    committed_state_row: Optional[Callable[[int], int]] = None
    # Per state tensor: the per-row axis TP shards, and the full widths of the
    # sub-blocks sharded independently along it (None when it is one block);
    # what a decode at a narrower attention TP reassembles its rows from.
    state_slice_axes: List[int] = field(default_factory=list)
    state_conv_groups: List[Optional[List[int]]] = field(default_factory=list)
    # How the full and SWA pools shard their pages across attention-TP ranks;
    # get_raiden_kv_caches sets both from the pools' declarations, and groups
    # built without them cannot be push-resharded.
    full_sharding: PageSharding = PageSharding.UNSUPPORTED
    swa_sharding: PageSharding = PageSharding.UNSUPPORTED

    # Optional host-side metadata pack/unpack hooks accompanying state_blocks.
    read_state_meta: Optional[Callable[[Sequence[int]], bytes]] = None
    write_state_meta: Optional[Callable[[Sequence[int], bytes], None]] = None


def _declared_sharding(pool: Any) -> PageSharding:
    """The ``raiden_page_sharding`` a pool declares; one that declares none
    cannot be push-resharded."""
    declared = getattr(pool, "raiden_page_sharding", None)
    if not isinstance(declared, str):
        return PageSharding.UNSUPPORTED
    try:
        return PageSharding(declared)
    except ValueError:
        raise RuntimeError(
            f"{type(pool).__name__} declares raiden_page_sharding={declared!r}; "
            f"expected one of {[sharding.value for sharding in PageSharding]}"
        ) from None


def _assert_root_buffers(name: str, buffers: Sequence[torch.Tensor]) -> None:
    for idx, buf in enumerate(buffers):
        if isinstance(buf, torch.Tensor) and buf._is_view():
            raise RuntimeError(
                f"Raiden requires root tensors (not views) for {name}[{idx}] "
                "so PJRT buffer registration remains valid."
            )


def get_raiden_kv_caches(
    pool: KVCache, draft_pool: Optional[KVCache] = None
) -> RaidenKVCacheGroups:
    """Group per-layer device KV tensors by their page-index namespace.

    The TPU fused KV pool stores one [num_blocks, page_size, ...] tensor per
    layer in ``kv_buffer`` and exposes them via ``get_fused_kv_buffer``. Hybrid
    SWA pools allocate full-attention and sliding-window pages independently, so
    they must not share one Raiden block-id list. Speculative decoding uses the
    same page indices for target and draft KV, so append the draft tensors to
    the matching namespace and move them in the same Raiden transfer. Hybrid
    linear-attention pools contribute GDN state rows; ``DSV4TPUTokenToKVPool``
    contributes its compressed full-page sub-pools, SWA pages, and 32-page
    request compressor state rings.
    """

    def group_one(current_pool: KVCache) -> RaidenKVCacheGroups:
        if isinstance(current_pool, DSV4TPUTokenToKVPool):
            dsv4_full: List[torch.Tensor] = []
            if isinstance(current_pool.c4_kv_pool, TPUDeepSeekV4CSAKVPool):
                dsv4_full.extend(current_pool.c4_kv_pool.nope_buffer)
                dsv4_full.extend(current_pool.c4_kv_pool.rope_buffer)
            if isinstance(current_pool.c4_indexer_kv_pool, TPUDeepSeekV4IndexerPool):
                dsv4_full.extend(
                    current_pool.c4_indexer_kv_pool.index_k_with_scale_buffer
                )
            if isinstance(current_pool.c128_kv_pool, TPUDeepSeekV4HCAKVPool):
                dsv4_full.extend(current_pool.c128_kv_pool.k_buffer)
            dsv4_swa: List[torch.Tensor] = (
                list(current_pool.swa_kv_pool.k_buffer)
                if isinstance(current_pool.swa_kv_pool, TPUDeepSeekV4SWAKVPool)
                else []
            )
            _assert_root_buffers("DSV4 full KV", dsv4_full)
            _assert_root_buffers("DSV4 SWA KV", dsv4_swa)
            return RaidenKVCacheGroups(full=dsv4_full, swa=dsv4_swa)

        layers_mapping = getattr(current_pool, "layers_mapping", None)
        start = current_pool.start_layer
        if isinstance(current_pool, HybridLinearKVPool):
            # Only full-attention layers own KV; GDN layers ship as state.
            layer_ids = sorted(current_pool.full_attention_layer_id_mapping)
            buffers = [
                current_pool.get_fused_kv_buffer(layer_id) for layer_id in layer_ids
            ]
        elif hasattr(current_pool, "get_fused_kv_buffer"):
            layer_ids = (
                sorted(layers_mapping)
                if layers_mapping
                else list(range(start, start + current_pool.layer_num))
            )
            buffers = [
                current_pool.get_fused_kv_buffer(layer_id) for layer_id in layer_ids
            ]
        elif hasattr(current_pool, "kv_buffer"):
            buffers = list(current_pool.kv_buffer)
            layer_ids = (
                sorted(layers_mapping)
                if layers_mapping
                else list(range(start, start + len(buffers)))
            )
        else:
            raise RuntimeError(
                "Raiden backend requires a fused KV pool exposing kv_buffer; got "
                f"{type(current_pool).__name__}"
            )
        if layers_mapping:
            full = []
            swa = []
            full_ids = []
            swa_ids = []
            for layer_id, buf in zip(layer_ids, buffers, strict=True):
                (swa if layers_mapping[layer_id][1] else full).append(buf)
                (swa_ids if layers_mapping[layer_id][1] else full_ids).append(layer_id)
            return RaidenKVCacheGroups(
                full=full, swa=swa, full_layer_ids=full_ids, swa_layer_ids=swa_ids
            )
        return RaidenKVCacheGroups(
            full=buffers, swa=[], full_layer_ids=layer_ids, swa_layer_ids=[]
        )

    groups = group_one(pool)
    full, swa = list(groups.full), list(groups.swa)
    full_ids, swa_ids = list(groups.full_layer_ids), list(groups.swa_layer_ids)
    num_draft_full = num_draft_swa = 0
    if draft_pool is not None and draft_pool is not pool:
        draft_groups = group_one(draft_pool)
        num_draft_full, num_draft_swa = len(draft_groups.full), len(draft_groups.swa)
        full.extend(draft_groups.full)
        swa.extend(draft_groups.swa)
        full_ids.extend(draft_groups.full_layer_ids)
        swa_ids.extend(draft_groups.swa_layer_ids)

    state: List[torch.Tensor] = []
    state_type = StateType.MAMBA
    state_max_blocks = 1
    committed_state_row = None
    state_slice_axes: List[int] = []
    state_conv_groups: List[Optional[List[int]]] = []
    read_state_meta = None
    write_state_meta = None
    if isinstance(pool, DSV4TPUTokenToKVPool):
        state = [
            p.state_buffer
            for p in (*pool.compress_state_pools, *pool.indexer_compress_state_pools)
            if isinstance(p, TPUCompressStatePool)
        ]
        _assert_root_buffers("DSV4 compressor state", state)
        state_type = StateType.DSV4_REQUEST_STATE
        state_max_blocks = _TPU_STATE_SLOT_STRIDE_PAGES
        read_state_meta = pool.pack_req_state_meta
        write_state_meta = pool.unpack_req_state_meta
    elif isinstance(pool, HybridLinearKVPool) and isinstance(
        pool.mamba_pool, TpuMambaPool
    ):
        state = pool.mamba_pool.get_state()
        committed_state_row = pool.mamba_pool.committed_row
        state_slice_axes = pool.mamba_pool.get_state_slice_axes()
        state_conv_groups = pool.mamba_pool.get_state_conv_shard_groups()

    for name, group in (("full KV", full), ("SWA KV", swa), ("GDN state", state)):
        block_counts = {int(buf.shape[0]) for buf in group}
        if len(block_counts) > 1:
            raise RuntimeError(
                f"Raiden requires uniform {name} block counts; got "
                f"{sorted(block_counts)}"
            )
    return RaidenKVCacheGroups(
        full=full,
        swa=swa,
        full_layer_ids=full_ids,
        swa_layer_ids=swa_ids,
        num_draft_full=num_draft_full,
        num_draft_swa=num_draft_swa,
        state=state,
        state_type=state_type,
        state_max_blocks=state_max_blocks,
        committed_state_row=committed_state_row,
        state_slice_axes=state_slice_axes,
        state_conv_groups=state_conv_groups,
        # A composite pool's sub-pools declare how their own pages are sharded;
        # draft tensors move with the target's.
        full_sharding=_declared_sharding(getattr(pool, "full_kv_pool", pool)),
        swa_sharding=_declared_sharding(getattr(pool, "swa_kv_pool", pool)),
        read_state_meta=read_state_meta,
        write_state_meta=write_state_meta,
    )


class StateSharding(str, enum.Enum):
    """How a request state can be push-resharded, derived from what its pool
    describes rather than declared."""

    # One row per request, each tensor sharded on a per-row axis the pool names.
    ROW_SHARDED = "row_sharded"
    UNSUPPORTED = "unsupported"


# Cache layouts push-resharding is validated for, keyed by how the full-attention
# pages, the sliding-window pages and the request state shard (None for a model
# without them), each with the e2e that validates it. Any model whose layout
# matches is admitted; enabling another layout takes an entry here and its e2e.
# A layout with both sliding-window pages and request state first needs the SWA
# push to handle a full prefix hit: such a room still asks its sources for the
# state (register_decode_request), and the SWA push fails a room that has no
# window pages to move (_run_reshard_swa_push).
VALIDATED_RESHARD_LAYOUTS: Mapping[
    Tuple[PageSharding, Optional[PageSharding], Optional[StateSharding]], str
] = {
    (PageSharding.HEAD_SLICED, None, None): (
        "test_tpu_qwen3_coder_480b_pd_multihost_kv_offload_resharding.py"
    ),
    (PageSharding.HEAD_SLICED, PageSharding.HEAD_SLICED, None): (
        "test_tpu_gpt_oss_120b_pd_multihost_kv_offload_resharding.py, "
        "test_tpu_gpt_oss_120b_pd_multihost_prefix_cache_kv_offload_spec_reshard.py"
    ),
    (PageSharding.REPLICATED, None, StateSharding.ROW_SHARDED): (
        "test_tpu_kimi_k3_tiny_pd_reshard.py"
    ),
}


def _describe_layout(
    full: PageSharding, swa: Optional[PageSharding], state: Optional[StateSharding]
) -> str:
    parts = [f"{full.value} KV pages"]
    if swa is not None:
        parts.append(f"{swa.value} sliding-window pages")
    if state is not None:
        parts.append(f"{state.value} request state")
    return " + ".join(parts)


def reshard_refusal(
    full: PageSharding, swa: Optional[PageSharding], state: Optional[StateSharding]
) -> Optional[str]:
    """Why push-resharding a cache of this layout is refused, or None when the
    layout is validated. ``swa`` and ``state`` are None when the model has no
    sliding-window pages or no request state."""
    if (full, swa, state) in VALIDATED_RESHARD_LAYOUTS:
        return None
    validated = ", ".join(
        _describe_layout(*layout) for layout in VALIDATED_RESHARD_LAYOUTS
    )
    return (
        "Raiden push resharding is not validated for "
        f"{_describe_layout(full, swa, state)}; validated layouts: {validated}"
    )

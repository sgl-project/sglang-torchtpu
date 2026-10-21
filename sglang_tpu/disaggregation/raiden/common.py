"""Shared helpers for the raiden prefill/decode disaggregation backend.

Holds what both sides of the transport need: the package logger, the port and
chunk-size constants, the peer-endpoint formatter, and the KV-cache grouping
that turns a device pool into the per-namespace tensor lists raiden registers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import torch

from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, KVCache
from sglang.srt.utils.network import NetworkAddress
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
    # Per-layer GDN state tensors, dim 0 = state row; only the target pool's.
    state: List[torch.Tensor] = field(default_factory=list)
    # Maps a request's mamba slot to the state row a transfer moves.
    committed_state_row: Optional[Callable[[int], int]] = None
    # Set for the TPU unified pool, whose sliding-window layers share buffers:
    # the SWA engine registers each of them once (swa_buffers) and moves a
    # sliding-window page as swa_block_ids(pages), swa_groups page ids each.
    swa_buffers: Optional[List[Any]] = None
    swa_block_ids: Optional[Callable[[List[int]], List[int]]] = None
    swa_groups: int = 1


def get_raiden_kv_caches(
    pool: KVCache, draft_pool: Optional[KVCache] = None
) -> RaidenKVCacheGroups:
    """Group per-layer device KV tensors by their page-index namespace.

    The TPU fused KV pool stores one [num_blocks, page_size, ...] tensor per
    layer in ``kv_buffer`` and exposes them via ``get_fused_kv_buffer``. Hybrid
    SWA pools allocate full-attention and sliding-window pages independently, so
    they must not share one Raiden block-id list. Speculative decoding uses the
    same page indices for target and draft KV, so append the draft tensors to
    the matching namespace and move them in the same Raiden transfer. A hybrid
    linear-attention target pool also contributes its GDN state tensors.
    """

    def group_one(current_pool: KVCache) -> RaidenKVCacheGroups:
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
    committed_state_row = None
    if isinstance(pool, HybridLinearKVPool) and isinstance(
        pool.mamba_pool, TpuMambaPool
    ):
        state = pool.mamba_pool.get_state()
        committed_state_row = pool.mamba_pool.committed_row

    for name, group in (("full KV", full), ("SWA KV", swa), ("GDN state", state)):
        block_counts = {int(buf.shape[0]) for buf in group}
        if len(block_counts) > 1:
            raise RuntimeError(
                f"Raiden requires uniform {name} block counts; got "
                f"{sorted(block_counts)}"
            )
    # Deferred: the unified pool module pulls in the KV cache stack.
    from sglang_tpu.unified_kv_pool import TpuUnifiedSWAKVPool

    shared_swa = {}
    if isinstance(pool, TpuUnifiedSWAKVPool):
        # Its block ids would also index a draft's sliding-window pool.
        assert not num_draft_swa
        shared_swa = dict(
            swa_buffers=pool.swa_transfer_buffers(),
            swa_block_ids=pool.swa_transfer_pages,
            swa_groups=pool.num_swa_groups,
        )
    return RaidenKVCacheGroups(
        full=full,
        swa=swa,
        full_layer_ids=full_ids,
        swa_layer_ids=swa_ids,
        num_draft_full=num_draft_full,
        num_draft_swa=num_draft_swa,
        state=state,
        committed_state_row=committed_state_row,
        **shared_swa,
    )

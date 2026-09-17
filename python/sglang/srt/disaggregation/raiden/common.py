"""Shared helpers for the raiden prefill/decode disaggregation backend.

Holds what both sides of the transport need: the package logger, the port and
chunk-size constants, the peer-endpoint formatter, and the KV-cache grouping
that turns a device pool into the per-namespace tensor lists raiden registers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

from sglang.srt.utils.network import NetworkAddress

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import KVCache

# One logger for the whole package, deliberately keeping the name of the module
# this code used to live in: emitted record names and the existing test
# monkeypatch target "…raiden.conn.logger" both depend on it.
logger = logging.getLogger("sglang.srt.disaggregation.raiden.conn")

# Pages are announced to the decode in pieces of at least this many tokens. A
# raiden read costs about the same whether it moves 16 pages or 200, so
# announcing every prefill step would spend more on rendezvous than it hides;
# pieces of this size keep the overlap while costing a few reads per request.
CHUNK_ANNOUNCE_TOKENS = 8192

# Base of the per-rank control ports (`base + 4*local_rank`). Not 30000: that
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


def get_raiden_kv_caches(
    pool: KVCache, draft_pool: Optional[KVCache] = None
) -> RaidenKVCacheGroups:
    """Group per-layer device KV tensors by their page-index namespace.

    The TPU fused KV pool stores one [num_blocks, page_size, ...] tensor per
    layer in ``kv_buffer`` and exposes them via ``get_fused_kv_buffer``. Hybrid
    SWA pools allocate full-attention and sliding-window pages independently, so
    they must not share one Raiden block-id list. Speculative decoding uses the
    same page indices for target and draft KV, so append the draft tensors to
    the matching namespace and move them in the same Raiden transfer.
    """

    def group_one(current_pool: KVCache) -> RaidenKVCacheGroups:
        if hasattr(current_pool, "get_fused_kv_buffer"):
            start = current_pool.start_layer
            buffers = [
                current_pool.get_fused_kv_buffer(start + i)
                for i in range(current_pool.layer_num)
            ]
        elif hasattr(current_pool, "kv_buffer"):
            buffers = list(current_pool.kv_buffer)
        else:
            raise RuntimeError(
                "Raiden backend requires a fused KV pool exposing kv_buffer; got "
                f"{type(current_pool).__name__}"
            )
        layers_mapping = getattr(current_pool, "layers_mapping", None)
        if layers_mapping:
            full = []
            swa = []
            for offset, buf in enumerate(buffers):
                layer_id = current_pool.start_layer + offset
                (swa if layers_mapping[layer_id][1] else full).append(buf)
            return RaidenKVCacheGroups(full=full, swa=swa)
        return RaidenKVCacheGroups(full=buffers, swa=[])

    groups = group_one(pool)
    full, swa = list(groups.full), list(groups.swa)
    if draft_pool is not None and draft_pool is not pool:
        draft_groups = group_one(draft_pool)
        full.extend(draft_groups.full)
        swa.extend(draft_groups.swa)

    for name, group in (("full", full), ("SWA", swa)):
        block_counts = {int(buf.shape[0]) for buf in group}
        if len(block_counts) > 1:
            raise RuntimeError(
                f"Raiden requires uniform {name} KV block counts; got "
                f"{sorted(block_counts)}"
            )
    return RaidenKVCacheGroups(full=full, swa=swa)

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

# Parallelism of a resharding push: the blocks of one push are spread over this
# many TCP connections, the count raiden's same-layout reads use per rank. On a
# single connection a push fills only a fraction of the prefill rank's NIC.
RESHARD_PUSH_PARALLELISM = 4

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
    """Group registered device KV tensors by page-index namespace to prevent aliasing.

    Raiden block IDs index dim 0; SWA pools allocate pages independently,
    while speculative draft KV shares target page indices.
    """

    from sglang.srt.hardware_backend.tpu.kv_pool import TpuFusedMHATokenToKVPool
    from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool

    def group_one(current_pool: KVCache) -> RaidenKVCacheGroups:
        if isinstance(current_pool, SWAKVPool):
            return RaidenKVCacheGroups(
                full=group_one(current_pool.full_kv_pool).full,
                swa=group_one(current_pool.swa_kv_pool).full,
            )
        if isinstance(current_pool, TpuFusedMHATokenToKVPool):
            buffers = current_pool.page_buffers()
        elif hasattr(current_pool, "get_fused_kv_buffer"):
            # Wrappers such as HybridLinearKVPool proxy the underlying
            # full-attention pool layer by layer.
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

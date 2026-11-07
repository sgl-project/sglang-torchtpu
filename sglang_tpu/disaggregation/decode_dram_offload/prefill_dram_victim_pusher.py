"""Prefill side of the push: offload cold host-backed branches to decode DRAM.

The prefill caches every session as usual and records, at each request's
bootstrap commit, which decode serves the session. When its host pool runs low
it takes the tracked host-backed session with the oldest path end, host-locks
it and offers its host pages (VICTIM_PUSH). The decode pulls them into its
host pages, inserts the session and answers (VICTIM_ACK). On a stored session
the prefill makes its own copy the next eviction victim.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
import weakref
from array import array
from typing import Any, Dict, List, Optional, Tuple

import msgspec
import torch.distributed as dist

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    depth_tokens,
    host_pages,
    is_live_host_backed_leaf,
    leaf_of,
    lock_host_path,
    mark_next_victim,
    path_tokens,
    unlock_host_path,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    push_manifest,
)

logger = logging.getLogger(__name__)

# Arbitrary: bounds the host pages a prefill pins and a decode holds at once.
MAX_INFLIGHT = 4
# Arbitrary: push once the host pool could take only about two more sessions.
_PUSH_FULL_FREE_TOKENS = 65536
# Arbitrary, well past one RAM-to-RAM push of a session and the decode's
# agreement rounds.
_ACK_TIMEOUT_S = 60.0
# Arbitrary: often enough to follow a sweep, rare enough to stay quiet.
_LOG_INTERVAL_S = 30.0
# Arbitrary: how long a cache flush waits for in-flight pushes to settle.
DRAIN_TIMEOUT_S = 60.0

# Victim paths take access stamps counting up from here, so they stay older
# than any live node (the tree's counter starts at 0) yet never tie.
_VICTIM_STAMP_BASE = -1.0e15


def log_offload_stats(stats: Dict[str, int], role: str, last: List[float]) -> None:
    now = time.monotonic()
    if now - last[0] < _LOG_INTERVAL_S:
        return
    last[0] = now
    logger.info(
        "[victim-push] role=%s %s",
        role,
        " ".join(f"{k}={v}" for k, v in stats.items() if v),
    )


class _PushInFlight(msgspec.Struct):
    seq: int
    # The pushed leaf; ``locks`` host-lock its whole path until the push
    # settles, since the decode reads pages of every node of it.
    leaf: Any
    locks: List[Any]
    sent_at: float
    # The native registration of the leaf's pages; None if it was refused.
    read_id: Optional[str] = None
    # The decode's answer; False also once the offer could not be sent, timed
    # out or failed. The registration still drains before the push settles.
    stored: Optional[bool] = None
    # Set once the registration settled: whether the decode stored the push.
    ok: Optional[bool] = None


def victim_transfer_id(prefill: str, seq: int) -> str:
    return f"victim:{prefill}:{seq}"


def _transfer_uuid(prefill: str, seq: int) -> int:
    digest = hashlib.blake2b(f"{prefill}:{seq}".encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "little") & ((1 << 63) - 1)) or 1


class PrefillDramVictimPusher:
    """Pushes host-backed sessions near eviction to their decodes."""

    def __init__(self, ctx: DramOffloadWorkerContext) -> None:
        self._channel = ctx.channel
        self._cache = ctx.cache
        self._host_pool = ctx.host_pool
        self._page_size = ctx.page_size
        self._lockstep = ctx.lockstep
        # Incarnation shared by this prefill's ranks: independent prefills and
        # restarts cannot alias push labels on one decode.
        epoch = self._lockstep.reduce([[secrets.randbits(62)]], dist.ReduceOp.MIN)[0][0]
        self._channel.victim_namespace = f"{epoch:x}"
        # Exact cached leaf -> (decode peer, semantic fingerprint). Weak keys
        # disappear with eviction and retain independent shared-prefix branches.
        self._routes = weakref.WeakKeyDictionary()
        self._pushes: Dict[int, _PushInFlight] = {}
        # Path length last pushed, or found unpushable, per session.
        self._pushed_len = weakref.WeakKeyDictionary()
        self._next_seq = 1
        # Access stamps handed to victim paths: far below the tree's access
        # counter (which starts at 0 and only grows), increasing per node.
        self._victim_stamp = _VICTIM_STAMP_BASE
        self.stats = dict.fromkeys(("started", "stored", "declined"), 0)
        self._last_log = [0.0]

    def note_decode_of(self, req: Any, *, recalled: bool = False) -> None:
        """Carry the actual decode peer to the request's cache checkpoint.
        ``recalled``: the request's cached prefix was just read from that
        decode, which therefore holds it."""
        peer = self._channel.decode_peer(req.bootstrap_room)
        if peer is not None:
            req._pd_victim_decode_peer = peer
        self.track_host_leaf(req, on_decode=recalled)

    def track_host_leaf(self, req: Any, *, on_decode: bool = False) -> None:
        # Set by record(); a request that never reached its commit has none.
        peer = getattr(req, "_pd_victim_decode_peer", None)
        node_id = req.last_node
        if node_id is None:
            # Matching can split a node, so every rank matches, with a peer or
            # not: the listener thread can drop a room's peer on one rank.
            key = RadixKey(
                array("q", req.origin_input_ids),
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
            ).page_aligned(self._page_size)
            match = self._cache.match_prefix(MatchPrefixParams(key=key))
            node_id = match.best_match_node
        node = leaf_of(self._cache, node_id)
        if node is not None and peer is not None:
            tokens = path_tokens(self._cache, node)
            self._routes[node] = (
                peer,
                push_manifest(tokens, node.key.extra_key, node.key.cache_salt),
            )
            if on_decode:
                # Not pushed back at the length it was recalled at.
                self._pushed_len[node] = len(tokens)

    def clear_push_routes(self) -> None:
        self._routes.clear()
        self._pushed_len.clear()

    def advance(self, *, serving: bool) -> None:
        """Every bootstrap step, every rank; no tree decision uses a local
        clock. Pushes start only while the engine serves requests: an engine
        left alone settles the pushes in flight and goes idle, so that it can
        be flushed."""
        self.retire_settled_pushes()
        self._start_push_if_agreed(serving=serving)
        log_offload_stats(self.stats, "prefill", self._last_log)

    def has_pushes_in_flight(self) -> bool:
        return bool(self._pushes)

    def release_all(self) -> None:
        """Keep source pins until every native registration has drained."""
        for push in self._pushes.values():
            push.stored, push.ok = False, None
        self._lockstep.drain_until(
            done=lambda: not self._pushes,
            step=self.retire_settled_pushes,
            timeout_s=DRAIN_TIMEOUT_S,
            what="a victim read is in flight",
        )

    def retire_settled_pushes(self) -> None:
        """Retire the pushes whose outcome every rank has."""
        if not self._pushes:
            return
        self._channel.drain_host_reads()
        now = time.monotonic()
        pushes = list(self._pushes.values())
        rows = []
        for push in pushes:
            if push.stored is None:
                push.stored = self._channel.victim_reply(push.seq)
                if push.stored is None and now - push.sent_at > _ACK_TIMEOUT_S:
                    push.stored = False
            sent = (
                False
                if push.read_id is None
                else self._channel.poll_host_read(push.read_id)
            )
            if sent is False:
                push.stored = False
            if push.stored is not None and sent is not None and push.ok is None:
                push.ok = push.stored and sent
            rows.append([int(push.ok is not None), int(push.ok is True)])
        # A rank that failed makes every rank treat the push as declined, so
        # no tree diverges.
        agreed = self._lockstep.reduce(rows, dist.ReduceOp.MIN)
        for push, (settled, ok) in zip(pushes, agreed):
            if settled:
                self.stats["stored" if ok else "declined"] += 1
                self._finish_push(push, stored=bool(ok))

    def _finish_push(self, push: _PushInFlight, *, stored: bool) -> None:
        if push.read_id is not None:
            self._channel.forget_host_read(push.read_id)
        del self._pushes[push.seq]
        unlock_host_path(self._cache, push.locks)
        self._channel.forget_victim(push.seq)
        if stored:
            # The decode holds the session now: its path here goes first in
            # every eviction order.
            self._victim_stamp = mark_next_victim(
                self._cache, push.leaf.id, self._victim_stamp
            )

    def _start_push_if_agreed(self, *, serving: bool) -> None:
        """Start a push in a step where every rank serves requests, is under
        host pressure, picked the same leaf and holds it whole in host RAM."""
        host_free = self._cache.cache_controller.mem_pool_host.available_size()
        picked = (
            self._pick_coldest_host_leaf()
            if serving
            and len(self._pushes) < MAX_INFLIGHT
            and host_free < _PUSH_FULL_FREE_TOKENS
            else None
        )
        # [picked, the leaf's fingerprint (2 values), its tokens, its host pages]
        row = [0, 0, 0, 0, 0]
        if picked is not None:
            decode, leaf = picked
            path = path_tokens(self._cache, leaf)
            pages = host_pages(self._cache, self._host_pool, leaf)
            row = [
                1,
                *push_manifest(path, leaf.key.extra_key, leaf.key.cache_salt),
                len(path),
                -1 if pages is None else len(pages),
            ]
        low, high = self._lockstep.min_max(row)
        if not low[0] or low[:4] != high[:4]:
            return
        if low[4] != high[4] or pages is None:
            return
        # Not again at this length, stored or not.
        self._pushed_len[leaf] = len(path)
        seq = self._next_seq
        self._next_seq += 1
        namespace = self._channel.victim_namespace
        uuid = _transfer_uuid(namespace, seq)
        read_id = victim_transfer_id(namespace, seq)
        push = _PushInFlight(
            seq=seq,
            leaf=leaf,
            locks=lock_host_path(self._cache, leaf),
            sent_at=time.monotonic(),
        )
        self._pushes[seq] = push
        self.stats["started"] += 1
        if self._channel.register_host_read(read_id, uuid, pages):
            push.read_id = read_id
        if not self._lockstep.all(push.read_id is not None) or not (
            self._channel.send_victim_push(
                decode=decode,
                seq=seq,
                tokens=path,
                blocks=pages,
                uuid=uuid,
                extra_key=leaf.key.extra_key,
                cache_salt=leaf.key.cache_salt,
            )
        ):
            push.stored = False

    def _pick_coldest_host_leaf(self) -> Optional[Tuple[Tuple[str, int], Any]]:
        """(decode peer, leaf) of the oldest leaf owned by a known decode whose
        whole path is in host RAM now. A leaf whose backup is still pending is
        a candidate again once it completed."""
        busy = {push.leaf for push in self._pushes.values()}
        candidates = []
        for node, (peer, fingerprint) in list(self._routes.items()):
            pushable = is_live_host_backed_leaf(self._cache, node)
            if pushable is None:
                self._routes.pop(node, None)
            if not pushable or node in busy:
                continue
            if self._pushed_len.get(node) == depth_tokens(self._cache, node):
                continue
            if host_pages(self._cache, self._host_pool, node) is None:
                continue
            candidates.append((node.last_access_time, fingerprint, peer, node))
        if not candidates:
            return None
        _, _, peer, node = min(candidates, key=lambda x: x[:2])
        return peer, node

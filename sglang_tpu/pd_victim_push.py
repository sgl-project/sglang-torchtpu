"""Victim push: the decode's RAM is the cache tier below the prefill's RAM.

Why: core has no per-request disaggregation cache policy or queue extension.
How: backend puller subclass, plus small lifecycle wrappers listed below.
Upstream PR or issue: none yet; queue/cache lifecycle seams need upstreaming.
Remove when: core supplies equivalent checkpoint and decode-queue callbacks.

SGLANG_DISAGG_PREFIX_VICTIM_PUSH=1 on both roles, with the decode-held prefix
pull (SGLANG_DISAGG_PREFILL_PULL_DECODE_PREFIX) on.

The bytes move RAM to RAM: the prefill's HiCache host pool pushes straight
into pages the decode's HiCache host pool allocated, through the host pool's
own raiden manager's pull-serve API. No HBM, no load
back, no staging slot on either side.

- Decode: does not cache the requests it serves; its radix tree holds only the
  sessions prefills push to it.
- Prefill: caches every session as usual and records, at each request's
  bootstrap commit, which decode serves the session. When its host pool runs
  low it takes the tracked host-backed session with the oldest path end,
  host-locks it and offers its host pages (VICTIM_PUSH). The decode answers
  by pulling into its host pages (VICTIM_ACK); the prefill reports completion
  (VICTIM_DONE). On a stored session the prefill makes its own copy the next
  eviction victim.
- Decode: every ``_CADENCE`` scheduler iterations its ranks agree on the
  offers all of them received, allocate host pages and start host reads.
  Native receive completion and VICTIM_DONE gate host-only insertion.

Sticky routing sends the session's next turn to the same decode, whose prefix
offer lets the prefill pull it back. TP/CP ranks agree on identity, content
and outcome before changing a tree. Hybrid models transfer full-attention KV
and the page-aligned trailing SWA window through their separate host pools.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import secrets
import time
import weakref
from array import array
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import msgspec
import torch
import torch.distributed as dist

from sglang.srt.disaggregation.decode import DecodePreallocQueue, DecodeTransferQueue
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
    PoolTransferResult,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import CacheTransferPhase
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang_tpu.disaggregation.raiden.conn import RaidenKVManager
from sglang_tpu.disaggregation.raiden.prefix_pull import DecodePrefixPuller
from sglang_tpu.disaggregation.raiden.serde import VictimAck
from sglang_tpu.envs import envs as tpu_envs

logger = logging.getLogger(__name__)

_FULL = ComponentType.FULL
_SWA = ComponentType.SWA

# Arbitrary: bounds the host pages a prefill pins and a decode holds at once.
_MAX_INFLIGHT = 4
# Arbitrary: push once the host pool could take only about two more sessions.
_PUSH_FULL_FREE_TOKENS = 65536
_PUSH_SWA_FREE_WINDOWS = 4
# Arbitrary, well past one RAM-to-RAM push of a session.
_ACK_TIMEOUT_S = 60.0
# Arbitrary: report a granted landing with no VICTIM_DONE after this.
# A timeout cannot revoke remote writes: retain allocations until native reads settle.
_LANDING_TIMEOUT_S = 180.0
# Arbitrary: decode iterations between agreement rounds, and pushes a rank
# offers per round; a push not every rank received in 8 rounds is dropped.
_CADENCE = 16
_AGREE_SLOTS = 8
_MAX_UNAGREED_ROUNDS = 8
# Arbitrary: often enough to follow a sweep, rare enough to stay quiet.
_LOG_INTERVAL_S = 30.0
# Arbitrary: how long a cache flush waits for in-flight pushes to settle.
_DRAIN_TIMEOUT_S = 60.0

# Victim paths take access stamps counting up from here, so they stay older
# than any live node (the tree's counter starts at 0) yet never tie.
_VICTIM_STAMP_BASE = -1.0e15

_is_installed = False


def _all_ranks_reduce(
    groups: Sequence[Any], rows: List[List[int]], op
) -> List[List[int]]:
    if not groups or not rows:
        return [list(row) for row in rows]
    values = torch.tensor(rows, dtype=torch.int64)
    for group in groups:
        dist.all_reduce(values, op=op, group=group)
    return values.tolist()


def _all_ranks_common(groups: Sequence[Any], keys: list) -> list:
    """Agree on bounded manifests, including content, over the full TP/CP mesh.

    Feed each group's intersection into the next group. Intersecting separate
    local neighbour lists does not establish agreement over a Cartesian mesh.
    """
    common = set(sorted(keys)[:_AGREE_SLOTS])
    for group in groups:
        gathered = [None] * dist.get_world_size(group)
        dist.all_gather_object(gathered, sorted(common), group=group)
        common.intersection_update(*(set(row) for row in gathered))
    return sorted(common)


def _manifest(tokens, extra_key=None, cache_salt=None, is_eagle=False, *geometry):
    payload = msgspec.msgpack.encode(
        (tokens, extra_key, cache_salt, is_eagle, geometry)
    )
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return tuple(int.from_bytes(digest[i : i + 8], "little") >> 1 for i in (0, 8))


def _agree(groups, row):
    low = _all_ranks_reduce(groups, [row], dist.ReduceOp.MIN)
    high = _all_ranks_reduce(groups, [row], dist.ReduceOp.MAX)
    return low == high


def _cpu_groups(scheduler: Any) -> List[Any]:
    return [
        group
        for group in (scheduler.attn_tp_cpu_group, scheduler.attn_cp_cpu_group)
        if group is not None and dist.get_world_size(group) > 1
    ]


def _depth(node: Any, root: Any) -> int:
    depth = 0
    while node is not root:
        depth += len(node.key)
        node = node.parent
    return depth


def _path_tokens(root: Any, leaf: Any) -> List[int]:
    """Reconstruct raw tokens, joining overlapping EAGLE boundaries once."""
    parts = []
    node = leaf
    while node is not root:
        parts.append(node.key.token_ids)
        node = node.parent
    tokens = []
    for part in reversed(parts):
        tokens.extend(part[1:] if tokens and leaf.key.is_bigram else part)
    return tokens


def _host_pages(
    root: Any, host_pool: Any, leaf: Any, *, component=_FULL, tail_tokens=None
) -> Optional[List[int]]:
    """Read a host-backed path (or its trailing window), in token order.

    A window may start inside a radix node; only its intersecting suffix is
    exported. Pending backups cannot serve either component yet.
    """
    segments = []
    remaining = tail_tokens
    node = leaf
    while node is not root and (remaining is None or remaining > 0):
        data = node.component_data[component]
        if data.host_value is None or node.write_through_pending_id is not None:
            return None
        count = len(node.key) if remaining is None else min(remaining, len(node.key))
        segments.append(host_pool._to_page_ids(data.host_value[-count:]))
        if remaining is not None:
            remaining -= count
        node = node.parent
    if remaining:
        return None
    return [int(p) for segment in reversed(segments) for p in segment]


def _swa_host_pool(cache: Any) -> Any:
    if not cache.supports_swa():
        return None
    return cache.host_pool_group.get_pool(PoolName.SWA)


def _window_tokens(cache: Any, page_size: int) -> int:
    if not cache.supports_swa():
        return 0
    window = cache.components[_SWA].sliding_window_size
    return -(-window // page_size) * page_size


def _page_hashes(tokens: List[int], page_size: int) -> List[str]:
    try:
        from sglang.srt.mem_cache.utils import get_hash_str

        hashes = get_hash_str(tokens, None, page_size)
        return [hashes] if isinstance(hashes, str) else list(hashes)
    except Exception:  # noqa: BLE001 - hashes only feed storage and events
        return [""] * (len(tokens) // page_size)


def _anchor_host_pool(cache: Any) -> Any:
    """The HiCache host pool that holds full KV, and its raiden manager."""
    return cache.cache_controller.mem_pool_host.anchor_entry.host_pool


def _maybe_log(stats: Dict[str, int], role: str, last: List[float]) -> None:
    now = time.monotonic()
    if now - last[0] < _LOG_INTERVAL_S:
        return
    last[0] = now
    logger.info(
        "[victim-push] role=%s %s",
        role,
        " ".join(f"{k}={v}" for k, v in stats.items() if v),
    )


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------


class _Push(msgspec.Struct):
    seq: int
    decode: Tuple[str, int]
    tokens: List[int]
    pages: List[int]
    path_len: int
    anchor_id: Any
    host_receipt: Any
    source_node: Any = None
    extra_key: Optional[str] = None
    cache_salt: Optional[str] = None
    is_eagle: bool = False
    sent_at: float = 0.0
    # Refused or timed-out offers still drain their native registrations.
    state: str = "offered"
    swa_pages: List[int] = []
    uuid: int = 0
    ok: Optional[bool] = None
    offered: bool = False
    acknowledged: Optional[bool] = None
    transfers: List[Tuple[Any, str]] = []


def _transfer_id(prefill: str, seq: int, component: str) -> str:
    return f"victim:{prefill}:{seq}:{component}"


def _transfer_uuid(prefill: str, seq: int) -> int:
    digest = hashlib.blake2b(f"{prefill}:{seq}".encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "little") & ((1 << 63) - 1)) or 1


class PrefillPusher:
    """Pushes host-backed sessions near eviction to their decodes."""

    def __init__(self, puller: DecodePrefixPuller) -> None:
        self._mgr = puller._mgr
        self._cache = puller._tree_cache
        self._page_size = puller._page_size
        self._groups = list(puller._groups)
        self._host_pool = _anchor_host_pool(self._cache)
        self._swa_host_pool = _swa_host_pool(self._cache)
        self._window_tokens = _window_tokens(self._cache, self._page_size)
        # Incarnation shared by this prefill's ranks: independent prefills and
        # restarts cannot alias push labels on one decode.
        epoch = _all_ranks_reduce(
            self._groups, [[secrets.randbits(62)]], dist.ReduceOp.MIN
        )[0][0]
        self._mgr.victim_namespace = f"{epoch:x}"
        # Exact cached leaf -> (decode peer, semantic fingerprint). Weak keys
        # disappear with eviction and retain independent shared-prefix branches.
        self._routes = weakref.WeakKeyDictionary()
        _pushers_by_cache[self._cache] = weakref.ref(self)
        self._pushes: Dict[int, _Push] = {}
        # Path length last pushed, or found unpushable, per session.
        self._pushed_len = weakref.WeakKeyDictionary()
        self._next_seq = 1
        # Access stamps handed to victim paths: far below the tree's access
        # counter (which starts at 0 and only grows), increasing per node.
        self._victim_stamp = _VICTIM_STAMP_BASE
        self.stats = dict.fromkeys(
            ("started", "pushed", "stored", "declined", "aborted"), 0
        )
        self._last_log = [0.0]

    def record(self, req: Any) -> None:
        """Carry the actual decode peer to the request's cache checkpoint."""
        peer = self._mgr.decode_peer(req.bootstrap_room)
        if peer is not None:
            req._pd_victim_decode_peer = peer
            self.remember(req)

    def remember(self, req: Any) -> None:
        peer = getattr(req, "_pd_victim_decode_peer", None)
        if peer is None:
            return
        tree_core = self._cache.tree_core
        node_id = getattr(req, "last_node", None)
        if node_id is None:
            key = RadixKey(
                array("q", req.origin_input_ids),
                extra_key=getattr(req, "extra_key", None),
                cache_salt=getattr(req, "cache_salt", None),
                is_bigram=tree_core.is_eagle,
            ).page_aligned(self._page_size)
            match = self._cache.match_prefix(MatchPrefixParams(key=key))
            node_id = match.best_match_node
        node = tree_core.node_by_id(node_id)
        if node is not tree_core.root_node and not node.children:
            tokens = _path_tokens(tree_core.root_node, node)
            self._routes[node] = (
                peer,
                _manifest(
                    tokens, node.key.extra_key, node.key.cache_salt, tree_core.is_eagle
                ),
            )

    def step(self) -> None:
        """Every bootstrap step, every rank; no tree decision uses a local clock."""
        self._settle()
        eligible = int(
            len(self._pushes) < _MAX_INFLIGHT and self._under_host_pressure()
        )
        if _all_ranks_reduce(self._groups, [[eligible]], dist.ReduceOp.MIN)[0][0]:
            self._maybe_start()
        _maybe_log(self.stats, "prefill", self._last_log)

    def release_all(self) -> None:
        """Keep source pins until every native registration has drained."""
        deadline = time.monotonic() + _DRAIN_TIMEOUT_S
        for push in self._pushes.values():
            push.acknowledged = False
            push.state = "draining"
        while True:
            empty, expired = _all_ranks_reduce(
                self._groups,
                [[int(not self._pushes), int(time.monotonic() > deadline)]],
                dist.ReduceOp.MIN,
            )[0]
            if empty:
                return
            if expired:
                raise RuntimeError(
                    "Cannot release KV memory while a victim read is in flight; retry after it drains"
                )
            self._settle()
            time.sleep(0.05)

    def _settle(self) -> None:
        if not self._pushes:
            return
        now = time.monotonic()
        pushes = list(self._pushes.values())
        rows = []
        for push in pushes:
            if push.state == "offered":
                ack = self._mgr.take_victim_ack(push.seq)
                if ack is not None:
                    push.acknowledged = ack.stored
                    push.state = "draining"
                elif now - push.sent_at > _ACK_TIMEOUT_S:
                    push.acknowledged = False
                    push.state = "draining"
            outcomes = [
                transfer.poll_transfer(req_id) for transfer, req_id in push.transfers
            ]
            if "failed" in outcomes:
                push.acknowledged = False
                push.state = "draining"
            if push.state == "draining" and all(outcomes):
                push.ok = push.acknowledged is True and all(
                    outcome == "sent" for outcome in outcomes
                )
                push.state = "done"
            rows.append([int(push.state == "done"), int(push.ok is True)])
        agreed = _all_ranks_reduce(self._groups, rows, dist.ReduceOp.MIN)
        for push, (settled, ok) in zip(pushes, agreed):
            if not settled:
                continue
            # The decode inserts only what every rank pushed; a rank that
            # failed makes every rank report failure, so no tree diverges.
            if push.offered:
                self._mgr.send_victim_done(
                    decode=push.decode, seq=push.seq, ok=bool(ok)
                )
            self.stats["stored" if ok else "declined"] += 1
            self._finish(push, stored=bool(ok))

    def _register_push(self, push: _Push) -> bool:
        try:
            for component, pool, pages in (
                ("full", self._host_pool, push.pages),
                ("swa", self._swa_host_pool, push.swa_pages),
            ):
                if not pages:
                    continue
                req_id = _transfer_id(self._mgr.victim_namespace, push.seq, component)
                # False can mean an early pull already completed. The native
                # terminal report, rather than the return value, releases pins.
                pool.transfer.register_read(req_id, push.uuid, pages, source="host")
                push.transfers.append((pool.transfer, req_id))
            return True
        except Exception as exc:  # noqa: BLE001 - drain accepted registrations
            logger.warning("Cannot register victim %d: %s", push.seq, exc)
            return False

    def _finish(self, push: _Push, *, stored: bool) -> None:
        for transfer, req_id in push.transfers:
            transfer.forget_transfer(req_id)
        self._pushes.pop(push.seq, None)
        self._pushed_len[push.source_node] = push.path_len
        self._cache.dec_host_lock_ref(push.anchor_id, push.host_receipt)
        self._mgr.forget_victim(push.seq)
        if stored:
            self.stats["pushed"] += 1
            self._make_next_victim(push.anchor_id)

    def _make_next_victim(self, anchor_id: Any) -> None:
        """The decode holds the session now: make its path here the oldest in
        every eviction order, below any branch other sessions share."""
        tree_core = self._cache.tree_core
        anchor = tree_core.node_by_id(anchor_id)
        nodes = []
        stack = [anchor]
        while stack:
            node = stack.pop()
            nodes.append(node)
            stack.extend(node.children.values())
        node = anchor.parent
        while node is not tree_core.root_node and len(node.children) == 1:
            nodes.append(node)
            node = node.parent
        lrus = [*tree_core.lru_lists.values(), *tree_core.host_lru_lists.values()]
        for node in nodes:
            # Device eviction heaps order leaves by last_access_time and break
            # ties by set iteration order, which differs per rank; every victim
            # node gets a unique stamp below any live node's counter value.
            self._victim_stamp += 1.0
            node.last_access_time = self._victim_stamp
            for lru in lrus:
                if lru.in_list(node):
                    lru.reset_node_lru(node)

    def _under_host_pressure(self) -> bool:
        controller = self._cache.cache_controller
        if controller is None:
            return False
        return controller.mem_pool_host.available_size() < _PUSH_FULL_FREE_TOKENS or (
            self._swa_host_pool is not None
            and self._swa_host_pool.available_size()
            < _PUSH_SWA_FREE_WINDOWS * self._window_tokens
        )

    def _maybe_start(self) -> None:
        picked = self._pick()
        tree_core = self._cache.tree_core
        if picked is None:
            signature = [-1, -1]
        else:
            _, decode, leaf = picked
            path = _path_tokens(tree_core.root_node, leaf)
            signature = list(
                _manifest(
                    path, leaf.key.extra_key, leaf.key.cache_salt, tree_core.is_eagle
                )
            )
        if not _agree(self._groups, signature) or picked is None:
            return
        _, decode, leaf = picked
        num_tokens = _depth(leaf, tree_core.root_node)
        pages = _host_pages(tree_core.root_node, self._host_pool, leaf)
        swa_pages = []
        if self._window_tokens:
            swa_pages = (
                _host_pages(
                    tree_core.root_node,
                    self._swa_host_pool,
                    leaf,
                    component=_SWA,
                    tail_tokens=min(num_tokens, self._window_tokens),
                )
                if self._swa_host_pool is not None
                else None
            )
        shape = [
            num_tokens,
            -1 if pages is None else len(pages),
            -1 if swa_pages is None else len(swa_pages),
        ]
        if not _agree(self._groups, shape) or pages is None or swa_pages is None:
            # Not whole on host everywhere: not again at this length.
            self._pushed_len[leaf] = num_tokens
            return
        seq = self._next_seq
        self._next_seq += 1
        push = _Push(
            seq=seq,
            decode=decode,
            tokens=path[: num_tokens + int(tree_core.is_eagle)],
            pages=pages,
            swa_pages=swa_pages,
            path_len=num_tokens,
            anchor_id=leaf.id,
            host_receipt=self._cache.inc_host_lock_ref(leaf.id).to_dec_params(),
            source_node=leaf,
            extra_key=leaf.key.extra_key,
            cache_salt=leaf.key.cache_salt,
            is_eagle=tree_core.is_eagle,
            sent_at=time.monotonic(),
            uuid=_transfer_uuid(self._mgr.victim_namespace, seq),
        )
        self._pushes[seq] = push
        self.stats["started"] += 1
        registered = self._register_push(push)
        registered = _all_ranks_reduce(
            self._groups, [[int(registered)]], dist.ReduceOp.MIN
        )[0][0]
        if not registered:
            push.acknowledged = False
            push.state = "draining"
            return
        push.offered = self._mgr.send_victim_push(
            decode=decode,
            seq=seq,
            tokens=push.tokens,
            blocks=pages,
            swa_blocks=swa_pages,
            uuid=push.uuid,
            address=self._host_pool.transfer.host_control_address(self._mgr.local_ip),
            swa_address=(
                self._swa_host_pool.transfer.host_control_address(self._mgr.local_ip)
                if swa_pages
                else ""
            ),
            extra_key=push.extra_key,
            cache_salt=push.cache_salt,
            is_eagle=push.is_eagle,
        )
        if not push.offered:
            push.acknowledged = False
            push.state = "draining"

    def _pick(self):
        """Oldest host-backed leaf owned by a known decode."""
        tree_core = self._cache.tree_core
        busy = {push.source_node for push in self._pushes.values()}
        candidates = []
        for node, (peer, fingerprint) in list(self._routes.items()):
            try:
                if tree_core.node_by_id(node.id) is not node:
                    continue
            except KeyError:
                self._routes.pop(node, None)
                continue
            if node.children or node in busy:
                continue
            if node.component_data[_FULL].host_value is None:
                continue
            depth = _depth(node, tree_core.root_node)
            if self._pushed_len.get(node) == depth:
                continue
            candidates.append((node.last_access_time, fingerprint, peer, node))
        if not candidates:
            return None
        _, fingerprint, peer, node = min(candidates, key=lambda x: x[:2])
        return fingerprint[0], peer, node


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


class _Landing(msgspec.Struct):
    push: Any
    indices: Any
    granted_at: float
    swa_indices: Any = None
    timed_out: bool = False
    transfers: List[Tuple[Any, str]] = []
    failed: bool = False


class DecodeReceiver:
    """Lands the sessions prefills push into this decode's host pool."""

    def __init__(self, *, queue: DecodeTransferQueue, mgr: RaidenKVManager) -> None:
        scheduler = queue.scheduler
        self._mgr = mgr
        self._cache = queue.tree_cache
        self._page_size = int(scheduler.token_to_kv_pool_allocator.page_size)
        self._host_pool = _anchor_host_pool(self._cache)
        self._swa_host_pool = _swa_host_pool(self._cache)
        self._window_tokens = _window_tokens(self._cache, self._page_size)
        self._groups = _cpu_groups(scheduler)
        self._iterations = 0
        self._rounds = 0
        # Round a received push was first seen in, by (prefill, seq).
        self._first_seen: Dict[Tuple[str, int], int] = {}
        self._landings: Dict[Tuple[str, int], _Landing] = {}
        self.stats = dict.fromkeys(
            (
                "received",
                "granted",
                "no_room",
                "landed",
                "landed_tokens",
                "failed",
                "timed_out",
                "dropped",
            ),
            0,
        )
        self._last_log = [0.0]

    def holds_pages(self) -> bool:
        return bool(self._landings)

    def drain(self) -> None:
        """Settle every granted landing before a cache reset, for up to
        ``_DRAIN_TIMEOUT_S``: a prefill may still be pushing into the pages."""
        deadline = time.monotonic() + _DRAIN_TIMEOUT_S
        while True:
            row = _all_ranks_reduce(
                self._groups,
                [[int(not self._landings), int(time.monotonic() > deadline)]],
                dist.ReduceOp.MIN,
            )[0]
            if row[0]:
                return
            if row[1]:
                raise RuntimeError(
                    "Cannot reset KV memory while victim pushes are landing; retry after they drain"
                )
            self._finish_agreed()
            time.sleep(0.05)

    def step(self) -> None:
        """Every decode iteration, every rank; acts once per ``_CADENCE``."""
        self._iterations += 1
        if self._iterations % _CADENCE:
            return
        self._rounds += 1
        self._admit_agreed()
        self._finish_agreed()
        _maybe_log(self.stats, "decode", self._last_log)

    def _admit_agreed(self) -> None:
        received = {
            key: push
            for key, push in self._mgr.victim_pushes().items()
            if key not in self._landings
        }
        for key in received:
            if self._first_seen.setdefault(key, self._rounds) == self._rounds:
                self.stats["received"] += 1
        manifests = {
            (
                *key,
                *_manifest(
                    push.tokens,
                    push.extra_key,
                    push.cache_salt,
                    push.is_eagle,
                    len(push.blocks),
                    len(push.swa_blocks),
                ),
            ): key
            for key, push in received.items()
        }
        for manifest in _all_ranks_common(self._groups, list(manifests)):
            key = manifests[manifest]
            push = self._mgr.take_victim_push(key)
            self._first_seen.pop(key, None)
            if len(self._landings) >= _MAX_INFLIGHT or not self._grant(key, push):
                self.stats["no_room"] += 1
                self._mgr.send_victim_ack(
                    endpoint=push.ack_endpoint,
                    ack=VictimAck(seq=push.seq, stored=False),
                )
        for key, first in list(self._first_seen.items()):
            if self._rounds - first > _MAX_UNAGREED_ROUNDS:
                self._first_seen.pop(key)
                if self._mgr.take_victim_push(key) is not None:
                    self.stats["dropped"] += 1

    def _grant(self, key: Tuple[str, int], push: Any) -> bool:
        num_tokens = len(push.tokens) - int(push.is_eagle)
        valid = (
            num_tokens > 0
            and num_tokens == len(push.blocks) * self._page_size
            and push.is_eagle == self._cache.tree_core.is_eagle
            and len(push.swa_blocks) * self._page_size
            == min(num_tokens, self._window_tokens)
            and (not self._window_tokens or self._swa_host_pool is not None)
            and push.uuid > 0
            and bool(push.address)
            and (not push.swa_blocks or bool(push.swa_address))
            and push.source == "host"
        )
        valid = _all_ranks_reduce(self._groups, [[int(valid)]], dist.ReduceOp.MIN)[0][0]
        if not valid:
            return False
        indices = self._alloc_host(num_tokens)
        swa_indices = (
            self._alloc_host(len(push.swa_blocks) * self._page_size, swa=True)
            if push.swa_blocks
            else None
        )
        allocated = _all_ranks_reduce(
            self._groups,
            [
                [
                    int(
                        indices is not None
                        and (not push.swa_blocks or swa_indices is not None)
                    )
                ]
            ],
            dist.ReduceOp.MIN,
        )[0][0]
        if not allocated:
            if indices is not None:
                self._host_pool.free(indices)
            if swa_indices is not None:
                self._swa_host_pool.free(swa_indices)
            return False
        landing = _Landing(
            push=push,
            indices=indices,
            swa_indices=swa_indices,
            granted_at=time.monotonic(),
        )
        started = True
        try:
            for component, pool, blocks, local_indices, address in (
                ("full", self._host_pool, push.blocks, indices, push.address),
                (
                    "swa",
                    self._swa_host_pool,
                    push.swa_blocks,
                    swa_indices,
                    push.swa_address,
                ),
            ):
                if not blocks:
                    continue
                req_id = _transfer_id(push.prefill, push.seq, component)
                pool.transfer.start_read(
                    req_id,
                    push.uuid,
                    address,
                    list(blocks),
                    None,
                    local_host_block_ids=[
                        int(p) for p in pool._to_page_ids(local_indices)
                    ],
                    destination="host",
                )
                landing.transfers.append((pool.transfer, req_id))
        except Exception as exc:
            logger.warning("Cannot start victim landing %s: %s", key, exc)
            started = False
        started = _all_ranks_reduce(self._groups, [[int(started)]], dist.ReduceOp.MIN)[
            0
        ][0]
        # A failure after FULL started cannot release its live destination.
        # All ranks keep their allocations until their accepted reads settle.
        landing.failed = not bool(started)
        self._landings[key] = landing
        self.stats["granted"] += 1
        self._mgr.send_victim_ack(
            endpoint=push.ack_endpoint,
            ack=VictimAck(
                seq=push.seq,
                stored=bool(started),
            ),
        )
        return True

    def _alloc_host(
        self, num_tokens: int, *, swa: bool = False
    ) -> Optional[torch.Tensor]:
        pool = self._swa_host_pool if swa else self._host_pool
        short = _all_ranks_reduce(
            self._groups,
            [[max(0, num_tokens - pool.available_size())]],
            dist.ReduceOp.MAX,
        )[0][0]
        if short > 0:
            self._cache.evict_host(short, _SWA if swa else _FULL)
        return pool.alloc(num_tokens)

    def _finish_agreed(self) -> None:
        dones = self._mgr.victim_dones()
        for key in list(dones):
            if key not in self._landings and key not in self._mgr.victim_pushes():
                # No landing was granted here: nothing to insert.
                self._mgr.take_victim_done(key)
                dones.pop(key)
        if not self._landings:
            return
        now = time.monotonic()
        keys = sorted(self._landings)
        rows = []
        for key in keys:
            landing = self._landings[key]
            outcomes = [
                transfer.poll_transfer(req_id) for transfer, req_id in landing.transfers
            ]
            done = dones.get(key)
            rows.append(
                [
                    int(all(outcomes)),
                    int(
                        not landing.failed
                        and all(outcome == "received" for outcome in outcomes)
                    ),
                    int(done is not None),
                    int(done is not None and done.ok),
                ]
            )
        agreed = _all_ranks_reduce(self._groups, rows, dist.ReduceOp.MIN)
        expired = _all_ranks_reduce(
            self._groups,
            [
                [int(now - self._landings[k].granted_at > _LANDING_TIMEOUT_S)]
                for k in keys
            ],
            dist.ReduceOp.MAX,
        )
        for key, (settled, received, done, ok), (gone,) in zip(keys, agreed, expired):
            landing = self._landings[key]
            if settled and (not received or done or gone):
                self._mgr.take_victim_done(key)
                self._landings.pop(key)
                if received and done and ok:
                    self._insert(landing)
                else:
                    self._free_landing(landing)
                    self.stats["failed"] += 1
                continue
            if gone:
                if not landing.timed_out:
                    landing.timed_out = True
                    self.stats["timed_out"] += 1
                    logger.warning(
                        "Victim landing %s timed out; retaining pages until native "
                        "reads settle because remote writes may still be in flight",
                        key,
                    )

    def _forget_landing(self, landing: _Landing) -> None:
        for transfer, req_id in landing.transfers:
            transfer.forget_transfer(req_id)
        landing.transfers.clear()

    def _free_landing(self, landing: _Landing) -> None:
        self._forget_landing(landing)
        self._host_pool.free(landing.indices)
        if landing.swa_indices is not None:
            self._swa_host_pool.free(landing.swa_indices)

    def _insert(self, landing: _Landing) -> None:
        self._forget_landing(landing)
        push = landing.push
        num_tokens = len(push.tokens) - int(push.is_eagle)
        core = self._cache.tree_core
        result = core.insert_host(
            core.root_node.id,
            RadixKey(
                array("q", push.tokens),
                extra_key=push.extra_key,
                cache_salt=push.cache_salt,
                is_bigram=push.is_eagle,
            ),
            landing.indices,
            _page_hashes(list(push.tokens[:num_tokens]), self._page_size),
        )
        self._cache._apply_cache_actions(result.cache_actions)
        if result.host_insert_dropped or result.inserted_host_node is None:
            self._free_landing(landing)
            self.stats["dropped"] += 1
            return
        # Pages the tree already held for the shared prefix are duplicates.
        if result.prefix_len > 0:
            self._host_pool.free(landing.indices[: result.prefix_len])
        if landing.swa_indices is not None:
            # Use the existing prefetch commit: it splits the window boundary,
            # fills SWA tombstones and releases duplicate window slices.
            actions = []
            core.commit_hicache_transfers(
                core.root_node.id,
                CacheTransferPhase.PREFETCH,
                {
                    _SWA: [
                        PoolTransfer(
                            name=PoolName.SWA, host_indices=landing.swa_indices
                        )
                    ]
                },
                cache_actions=actions,
                insert_result=result,
                pool_storage_result=PoolTransferResult(
                    kv_hit_pages=len(push.blocks),
                    extra_pool_hit_pages={PoolName.SWA: len(push.swa_blocks)},
                ),
            )
            self._cache._apply_cache_actions(actions)
        self.stats["landed"] += 1
        self.stats["landed_tokens"] += num_tokens - result.prefix_len


class VictimPrefixPuller(DecodePrefixPuller):
    """Prefill orchestration selected by Raiden's existing backend factory."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pusher = PrefillPusher(self)

    def commit(self, req):
        super().commit(req)
        self.pusher.record(req)

    def advance(self, reqs, *, waiting_reqs=()):
        super().advance(reqs, waiting_reqs=waiting_reqs)
        self.pusher.step()

    def has_pending_deferred_releases(self) -> bool:
        return super().has_pending_deferred_releases() or bool(self.pusher._pushes)

    def resolve_deferred_releases(self) -> None:
        super().resolve_deferred_releases()
        # Paused and empty control ticks must retire existing exports too.
        self.pusher._settle()

    def release_all(self):
        self.pusher.release_all()
        super().release_all()


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

_pushers_by_cache = weakref.WeakKeyDictionary()

_receivers: weakref.WeakKeyDictionary[DecodeTransferQueue, DecodeReceiver] = (
    weakref.WeakKeyDictionary()
)


def _decode_add(
    upstream: Callable, queue: DecodePreallocQueue, req: Any, *args, **kwargs
):
    """Wraps ``DecodePreallocQueue.add``: the decode caches only pushed sessions."""
    if (
        isinstance(queue.kv_manager, RaidenKVManager)
        and queue.kv_manager.prefix_pull_active
        and isinstance(queue.tree_cache, UnifiedRadixCache)
    ):
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore

        if isinstance(queue.tree_cache.tree_core, UnifiedTreeCore):
            if queue.tree_cache.cache_controller is None:
                raise ValueError(
                    "Victim prefix caching requires --enable-hierarchical-cache on both roles"
                )
            if (
                queue.tree_cache.supports_swa()
                and not queue.tree_cache.tree_core.has_swa_host_pool
            ):
                raise ValueError(
                    "Victim SWA prefix caching requires SGLANG_TPU_PD_DECODE_SWA_HOST_WINDOWS > 0"
                )
            req.skip_radix_cache_insert = True
    return upstream(queue, req, *args, **kwargs)


def _receiver(queue: DecodeTransferQueue) -> Optional[DecodeReceiver]:
    receiver = _receivers.get(queue)
    if receiver is None:
        mgr = queue.scheduler.disagg_decode_prealloc_queue.kv_manager
        if (
            not isinstance(mgr, RaidenKVManager)
            or not mgr.prefix_pull_active
            or not isinstance(queue.tree_cache, UnifiedRadixCache)
        ):
            return None
        receiver = _receivers[queue] = DecodeReceiver(queue=queue, mgr=mgr)
    return receiver


def _resolve_deferred_releases(upstream: Callable, queue: DecodeTransferQueue) -> Any:
    """Wraps ``DecodeTransferQueue.resolve_deferred_releases``, which every rank
    runs each decode iteration: drive the receiver."""
    result = upstream(queue)
    receiver = _receiver(queue)
    if receiver is not None:
        receiver.step()
    return result


def _has_pending_deferred_releases(
    upstream: Callable, queue: DecodeTransferQueue
) -> bool:
    """Wraps ``DecodeTransferQueue.has_pending_deferred_releases``: granted
    landing pages sit outside the tree by design too, so the idle pool leak
    check waits for them as for deferred releases."""
    receiver = _receivers.get(queue)
    return upstream(queue) or (receiver is not None and receiver.holds_pages())


def _checkpoint(upstream, cache, req, **kwargs):
    result = upstream(cache, req, **kwargs)
    ref = _pushers_by_cache.get(cache)
    pusher = ref() if ref is not None else None
    if pusher is not None:
        pusher.remember(req)
    return result


def _reset_cache(upstream, cache):
    ref = _pushers_by_cache.get(cache)
    pusher = ref() if ref is not None else None
    if pusher is not None:
        pusher.release_all()
        pusher._routes.clear()
        pusher._pushed_len.clear()
    for receiver in list(_receivers.values()):
        if receiver._cache is cache:
            receiver.drain()
    return upstream(cache)


def _patch_targets() -> tuple[tuple[Any, str, Callable], ...]:
    return (
        (UnifiedRadixCache, "reset", _reset_cache),
        (UnifiedRadixCache, "checkpoint", _checkpoint),
        (DecodePreallocQueue, "add", _decode_add),
        (DecodeTransferQueue, "resolve_deferred_releases", _resolve_deferred_releases),
        (
            DecodeTransferQueue,
            "has_pending_deferred_releases",
            _has_pending_deferred_releases,
        ),
    )


def _wrap(owner: Any, name: str, impl: Callable) -> None:
    upstream = getattr(owner, name)

    @functools.wraps(upstream)
    def wrapper(*args, **kwargs):
        return impl(upstream, *args, **kwargs)

    setattr(owner, name, wrapper)


def install() -> None:
    global _is_installed
    if _is_installed or not tpu_envs.SGLANG_DISAGG_PREFIX_VICTIM_PUSH.get():
        return
    _is_installed = True
    for owner, name, impl in _patch_targets():
        _wrap(owner, name, impl)

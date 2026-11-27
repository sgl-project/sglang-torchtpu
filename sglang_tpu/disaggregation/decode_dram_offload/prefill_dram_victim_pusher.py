"""Prefill side of the push: offload cold host-backed branches to decode DRAM.

The prefill caches every session as usual and records, at each request's
bootstrap commit, which decode serves the session. When its host pool runs low
it takes the tracked host-backed session with the oldest path end, host-locks
it and offers its host pages (VICTIM_PUSH), with the window pages that end
where they end on a hybrid sliding-window model. The decode pulls them into
its host pages, inserts the session and answers (VICTIM_ACK). On a stored
session the prefill makes its own copy the next eviction victim.
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
from sglang.srt.mem_cache.common import page_align_floor
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    host_pages,
    is_live_host_backed_leaf,
    leaf_of,
    lock_host_path,
    mark_next_victim,
    path_token_count,
    path_tokens,
    unlock_host_path,
    window_host_pages,
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
# Arbitrary: on a hybrid model, also once the SWA host pool could take only
# this many more windows.
_PUSH_SWA_FREE_WINDOWS = 4
# Arbitrary, well past one RAM-to-RAM push of a session and the decode's
# agreement rounds.
_ACK_TIMEOUT_S = 60.0
# Arbitrary: often enough to follow a sweep, rare enough to stay quiet.
_LOG_INTERVAL_S = 30.0

# Victim paths take access stamps counting up from here, so they stay older
# than any live node (the tree's counter starts at 0) yet never tie.
_VICTIM_STAMP_BASE = -1.0e15


def log_offload_stats(stats: Dict[str, int], what: str, last: List[float]) -> None:
    now = time.monotonic()
    if now - last[0] < _LOG_INTERVAL_S:
        return
    last[0] = now
    logger.info(
        "[dram-offload] %s %s",
        what,
        " ".join(f"{k}={v}" for k, v in stats.items() if v),
    )


class _PushInFlight(msgspec.Struct):
    seq: int
    # The pushed leaf; ``locks`` host-lock its whole path until the push
    # settles, since the decode reads pages of every node of it.
    leaf: Any
    locks: List[Any]
    sent_at: float
    # The accepted native registrations of the leaf's pages (full, then the
    # window of a hybrid model); ``registered`` once all were accepted.
    read_ids: List[str] = msgspec.field(default_factory=list)
    registered: bool = False
    # The decode's answer; False also once the offer could not be sent, timed
    # out or failed. The registration still drains before the push settles.
    stored: Optional[bool] = None


def victim_transfer_id(prefill: str, seq: int, *, swa: bool = False) -> str:
    return f"victim:{prefill}:{seq}" + (":swa" if swa else "")


def _transfer_uuid(prefill: str, seq: int) -> int:
    digest = hashlib.blake2b(f"{prefill}:{seq}".encode(), digest_size=8).digest()
    return (int.from_bytes(digest, "little") & ((1 << 63) - 1)) or 1


class PrefillDramVictimPusher:
    """Pushes host-backed sessions near eviction to their decodes."""

    def __init__(self, ctx: DramOffloadWorkerContext) -> None:
        self._channel = ctx.channel
        self._cache = ctx.cache
        self._host_pool = ctx.host_pool
        self._swa_host_pool = ctx.swa_host_pool
        self._window_tokens = ctx.window_tokens
        self._page_size = ctx.page_size
        self._lockstep = ctx.lockstep
        # Incarnation shared by this prefill's ranks: independent prefills and
        # restarts cannot alias push labels on one decode.
        epoch = self._lockstep.reduce([[secrets.randbits(62)]], dist.ReduceOp.MIN)[0][0]
        self._namespace = f"{epoch:x}"
        # Exact cached leaf -> (decode peer, lazy semantic fingerprint). Weak
        # keys disappear with eviction and retain independent shared-prefix
        # branches.
        self._routes = weakref.WeakKeyDictionary()
        self._pushes: Dict[int, _PushInFlight] = {}
        # Path length last pushed, or found unpushable, per session.
        self._pushed_len = weakref.WeakKeyDictionary()
        # Strictly rank-replicated gate for active steps with an empty bootstrap
        # queue: cleared when victim selection does not exhaust MAX_INFLIGHT,
        # re-armed only by rank-synchronized cache and push transitions.
        self._selection_dirty = True
        self._was_serving = False
        self._next_seq = 1
        self._victim_stamp = _VICTIM_STAMP_BASE
        self.stats = dict.fromkeys(("started", "stored", "declined"), 0)
        self._last_log = [0.0]

    def mark_selection_dirty(self) -> None:
        """Re-arm victim selection on a rank-synchronized cache transition."""
        self._selection_dirty = True

    def can_select_victim(self) -> bool:
        """Rank-local vote piggybacked onto a non-empty bootstrap step."""
        return (
            len(self._pushes) < MAX_INFLIGHT
            and bool(self._routes)
            and self._under_host_pressure()
        )

    def _prompt_radix_key(self, req: Any) -> RadixKey:
        raw = req.origin_input_ids
        if isinstance(raw, array) and raw.typecode == "q":
            token_ids = raw
        else:
            fill_ids = getattr(req, "full_untruncated_fill_ids", None)
            if (
                isinstance(fill_ids, array)
                and fill_ids.typecode == "q"
                and len(fill_ids) >= len(raw)
            ):
                token_ids = fill_ids
            else:
                token_ids = array("q", raw)
        return RadixKey(
            token_ids,
            extra_key=req.extra_key,
            limit=page_align_floor(len(raw), self._page_size),
            cache_salt=req.cache_salt,
        )

    def _record_leaf_route(
        self, node_id: Any, peer: Any, *, on_decode: bool = False
    ) -> None:
        node = leaf_of(self._cache, node_id)
        if node is None or peer is None:
            return
        existing = self._routes.get(node)
        fingerprint = existing[1] if existing is not None else None
        self._routes[node] = (peer, fingerprint)
        if on_decode:
            # Not pushed back at the length it was recalled at.
            self._pushed_len[node] = path_token_count(self._cache, node)

    def note_decode_of(self, req: Any, *, recalled: bool) -> None:
        """Carry the actual decode peer to the request's cache checkpoint.
        ``recalled``: the request's cached prefix was just read from that
        decode, which therefore holds it."""
        peer = self._channel.decode_peer(req.bootstrap_room)
        if peer is not None:
            req._pd_victim_decode_peer = peer
        if (
            not recalled
            and getattr(req, "_pd_defer_leaf_match", False)
            and req.last_node is None
        ):
            return
        self.track_host_leaf(req, on_decode=recalled)

    def track_matched_leaf(self, req: Any, best_match_node: Any) -> None:
        """Route the leaf reached by admission's rank-consensus prefix match."""
        self._selection_dirty = True
        peer = getattr(req, "_pd_victim_decode_peer", None)
        if peer is None or best_match_node is None:
            return
        node = self._cache.tree_core.node_by_id(best_match_node)
        if node.children:
            prompt_len = len(req.origin_input_ids)
            page = self._page_size
            # Admission caps its match at len(prompt) - 1; when the prompt is an
            # exact page multiple and its final page is a cached leaf child, step
            # onto that leaf without mutating the tree.
            if (
                prompt_len >= page
                and prompt_len % page == 0
                and path_token_count(self._cache, node) == prompt_len - page
            ):
                key = self._prompt_radix_key(req)
                child = node.children.get(key.child_key_at(prompt_len - page, page))
                if (
                    child is not None
                    and not child.children
                    and len(child.key) == page
                    and child.key.match_at(
                        key, offset=prompt_len - page, page_size=page
                    )
                    == page
                ):
                    best_match_node = child.id
        self._record_leaf_route(best_match_node, peer)

    def track_host_leaf(self, req: Any, *, on_decode: bool = False) -> None:
        self._selection_dirty = True
        # Set by note_decode_of(); a request that never reached its commit has none.
        peer = getattr(req, "_pd_victim_decode_peer", None)
        recalled_node = getattr(req, "_pd_recalled_node", None) if on_decode else None
        if recalled_node is not None and hasattr(req, "__dict__"):
            delattr(req, "_pd_recalled_node")
        node_id = req.last_node if recalled_node is None else recalled_node
        if node_id is None:
            # Matching can split a node, so every rank matches, with a peer or
            # not: the listener thread can drop a room's peer on one rank.
            key = self._prompt_radix_key(req)
            match = self._cache.match_prefix(MatchPrefixParams(key=key))
            node_id = match.best_match_node
        self._record_leaf_route(node_id, peer, on_decode=on_decode)

    def advance(
        self, *, serving: bool, selection_agreed: Optional[bool] = True
    ) -> None:
        """Every bootstrap step, every rank; no tree decision uses a local
        clock. Pushes start only while the engine serves requests: an engine
        left alone settles the pushes in flight and goes idle, so that it can
        be flushed. ``serving`` comes from replicated scheduler queues;
        ``selection_agreed`` is True/False when piggybacked onto a non-empty
        bootstrap collective, or None on empty-bootstrap steps."""
        settled = self.retire_settled_pushes()
        if not serving:
            self._was_serving = False
        else:
            if not self._was_serving:
                self._was_serving = True
                self._selection_dirty = True
            if (
                selection_agreed is True
                or settled
                or (selection_agreed is None and self._selection_dirty)
            ):
                self._start_push_if_agreed(serving=serving)
            else:
                self._selection_dirty = False
        log_offload_stats(self.stats, "push", self._last_log)

    def _under_host_pressure(self) -> bool:
        host_free = self._cache.cache_controller.mem_pool_host.available_size()
        return host_free < _PUSH_FULL_FREE_TOKENS or (
            self._swa_host_pool is not None
            and self._swa_host_pool.available_size()
            < _PUSH_SWA_FREE_WINDOWS * self._window_tokens
        )

    def has_pushes_in_flight(self) -> bool:
        return bool(self._pushes)

    def release_all(self) -> None:
        """Before a cache reset or memory release, which core issues only on
        an idle engine: no push may be in flight."""
        if self._pushes:
            raise RuntimeError("victim pushes are in flight; the engine is not idle")
        self._routes.clear()
        self._pushed_len.clear()
        self._selection_dirty = True
        self._was_serving = False

    def retire_settled_pushes(self) -> bool:
        """Retire the pushes whose outcome every rank has; whether any settled."""
        if not self._pushes:
            return False
        self._channel.drain_host_reads()
        now = time.monotonic()
        pushes = list(self._pushes.values())
        rows = []
        for push in pushes:
            if push.stored is None:
                push.stored = self._channel.victim_reply(push.seq)
                if push.stored is None and now - push.sent_at > _ACK_TIMEOUT_S:
                    push.stored = False
            # Accepted registrations drain even when another one was refused.
            sent = self._channel.poll_host_reads(push.read_ids)
            if sent is not None and not push.registered:
                sent = False
            if sent is False:
                push.stored = False
            settled = push.stored is not None and sent is not None
            rows.append([int(settled), int(settled and push.stored and sent)])
        # A rank that failed makes every rank treat the push as declined, so
        # no tree diverges.
        agreed = self._lockstep.reduce(rows, dist.ReduceOp.MIN)
        any_settled = False
        for push, (settled, ok) in zip(pushes, agreed):
            if settled:
                any_settled = True
                self.stats["stored" if ok else "declined"] += 1
                self._finish_push(push, stored=bool(ok))
        if any_settled:
            self._selection_dirty = True
        return any_settled

    def _finish_push(self, push: _PushInFlight, *, stored: bool) -> None:
        self._channel.forget_host_reads(push.read_ids)
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
        picked = (
            self._pick_coldest_host_leaf()
            if serving
            and len(self._pushes) < MAX_INFLIGHT
            and bool(self._routes)
            and self._under_host_pressure()
            else None
        )
        # [picked, the leaf's fingerprint (2 values), its tokens, its host
        # pages, its window pages]
        row = [0, 0, 0, 0, 0, 0]
        if picked is not None:
            decode, leaf, path, pages, swa_pages = picked
            row = [
                1,
                *push_manifest(path, leaf.key.extra_key, leaf.key.cache_salt),
                len(path),
                len(pages),
                len(swa_pages),
            ]
        low, high = self._lockstep.min_max(row)
        if not low[0] or low != high:
            self._selection_dirty = False
            return
        # Not again at this length, stored or not.
        self._pushed_len[leaf] = len(path)
        seq = self._next_seq
        self._next_seq += 1
        uuid = _transfer_uuid(self._namespace, seq)
        push = _PushInFlight(
            seq=seq,
            leaf=leaf,
            locks=lock_host_path(self._cache, leaf),
            sent_at=time.monotonic(),
        )
        self._pushes[seq] = push
        self._selection_dirty = len(self._pushes) < MAX_INFLIGHT
        self.stats["started"] += 1
        # Snapshot under the same host pin as the KV read. A metadata owner may
        # need to resolve an outstanding seal before the page can leave.
        metadata = swa_metadata = None
        try:
            metadata = self._channel.export_host_metadata(pages)
            swa_metadata = self._channel.export_host_metadata(swa_pages, swa=True)
        except Exception:  # noqa: BLE001 - all ranks decline before registering
            logger.exception("Cannot export host metadata for victim push %s", seq)
        if not self._lockstep.all(metadata is not None and swa_metadata is not None):
            push.stored = False
            return
        push.registered = self._register_leaf_pages(push, uuid, pages, swa_pages)
        if not self._lockstep.all(push.registered) or not (
            self._channel.send_victim_push(
                decode=decode,
                prefill=self._namespace,
                seq=seq,
                tokens=path,
                blocks=pages,
                swa_blocks=swa_pages,
                uuid=uuid,
                extra_key=leaf.key.extra_key,
                cache_salt=leaf.key.cache_salt,
                page_metadata=metadata,
                swa_page_metadata=swa_metadata,
            )
        ):
            push.stored = False

    def _register_leaf_pages(
        self, push: _PushInFlight, uuid: int, pages: List[int], swa_pages: List[int]
    ) -> bool:
        """Register the leaf's host pages as the sources of the decode's reads:
        the full pages, and on a hybrid model the window pages under the same
        UUID on their own pool. An accepted registration stays in ``read_ids``
        when a later one is refused: it drains before the push settles."""
        for swa, part in ((False, pages), (True, swa_pages)):
            if not part:
                continue
            read_id = victim_transfer_id(self._namespace, push.seq, swa=swa)
            if not self._channel.register_host_read(read_id, uuid, part, swa=swa):
                return False
            push.read_ids.append(read_id)
        return True

    def _pick_coldest_host_leaf(
        self,
    ) -> Optional[Tuple[Tuple[str, int], Any, List[int], List[int], List[int]]]:
        """(decode peer, leaf, its path tokens, its host pages, its window
        pages) of the oldest leaf owned by a known decode whose whole path,
        and window, is in host RAM now. A leaf whose backup is still pending
        is a candidate again once it completed."""
        busy = {push.leaf for push in self._pushes.values()}
        candidates = []
        for node, (peer, fingerprint) in list(self._routes.items()):
            pushable = is_live_host_backed_leaf(self._cache, node)
            if pushable is None:
                self._routes.pop(node, None)
            if not pushable or node in busy:
                continue
            pushed_tokens = self._pushed_len.get(node)
            # Revisited routes commonly have no new tokens to push. Check
            # current ancestor metadata before copying their whole token path.
            if pushed_tokens is not None and pushed_tokens == path_token_count(
                self._cache, node
            ):
                continue
            path = path_tokens(self._cache, node)
            if pushed_tokens == len(path):
                continue
            pages = host_pages(self._cache, self._host_pool, node)
            if pages is None:
                continue
            swa_pages = window_host_pages(
                self._cache,
                self._swa_host_pool,
                node,
                tokens=len(path),
                window_tokens=self._window_tokens,
            )
            if swa_pages is None:
                continue
            if fingerprint is None:
                fingerprint = push_manifest(
                    path, node.key.extra_key, node.key.cache_salt
                )
                self._routes[node] = (peer, fingerprint)
            candidates.append(
                (node.last_access_time, fingerprint, peer, node, path, pages, swa_pages)
            )
        if not candidates:
            return None
        _, _, peer, node, path, pages, swa_pages = min(candidates, key=lambda x: x[:2])
        return peer, node, path, pages, swa_pages

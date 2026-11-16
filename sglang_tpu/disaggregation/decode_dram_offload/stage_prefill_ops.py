"""What a pipeline stage's push and recall decisions do to its tree, its host
pools and the wire (the ``ops`` of stage_prefill_workers.py).

Kept thin: every decision is in stage_prefill_workers.py, which runs without
torch. A method named there as "from the tree alone" must read nothing that
differs between stages at the same micro-batch slot.

Simpler than the tensor-parallel workers in three places, each a follow-up:
a recall reads the whole offered prefix even when the stage holds part of it;
a stored branch is not made the next eviction victim; a pair with the switch
on for one role only is not failed by name.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import time
import weakref
from array import array
from typing import Any, Callable, Dict, List, Optional, Tuple

import msgspec

from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang_tpu.disaggregation.decode_dram_offload import prefill_dram_victim_pusher
from sglang_tpu.disaggregation.decode_dram_offload.prefill_dram_victim_pusher import (
    victim_transfer_id,
)
from sglang_tpu.disaggregation.decode_dram_offload.radix_host_ops import (
    host_page_ids,
    host_pages,
    insert_host_prefix,
    is_live_host_backed_leaf,
    leaf_of,
    lock_host_path,
    offer_limit,
    path_tokens,
    prefix_radix_key,
    unlock_host_path,
    window_host_pages,
)
from sglang_tpu.disaggregation.decode_dram_offload.rank_lockstep import (
    DramOffloadWorkerContext,
    push_manifest,
)
from sglang_tpu.disaggregation.decode_dram_offload.stage_layout_store import (
    _page_digests,
)
from sglang_tpu.disaggregation.decode_dram_offload.stage_pool_builder import (
    stage_pool_layout,
)

logger = logging.getLogger(__name__)

# Arbitrary: page records the byte check keeps, oldest dropped first.
_VERIFY_RECORDS = 200_000


class PageByteCheck:
    """Optional end-to-end check (SGLANG_PD_OFFLOAD_VERIFY_PAGES): what a stage
    reads back from decode DRAM must be bytes this stage pushed for the same
    tokens. A page is named by its tokens and everything before them; several
    contents can be valid for one name, since the same prefix computed in
    other chunks gives other bytes and the decode keeps the first it stored."""

    def __init__(self, page_size: int) -> None:
        self._page_size = page_size
        self._pushed: collections.OrderedDict[bytes, set] = collections.OrderedDict()
        self.stats = dict.fromkeys(
            ("verified_pages", "mismatched_pages", "unknown_pages"), 0
        )

    @staticmethod
    def _content(pool: Any, page_ids: List[int]) -> Optional[List[bytes]]:
        """A digest of each host page over all its tensors; None when the
        pool's manager cannot hand out block bytes."""
        transfer = pool.transfer
        read = getattr(getattr(transfer, "_mgr", None), "read_block_bytes", None)
        if read is None:
            return None
        digests = []
        try:
            for page in page_ids:
                digest = hashlib.blake2b(digest_size=16)
                for layer in range(transfer.layer_num):
                    digest.update(read(layer, int(page)))
                digests.append(digest.digest())
        except Exception as exc:  # noqa: BLE001 - a debugging aid must not fail a push
            logger.warning("[dram-offload] cannot read host block bytes: %s", exc)
            return None
        return digests

    def _names(
        self, scope: Any, tokens: List[int], kind: str, count: int
    ) -> List[bytes]:
        """The names of the last ``count`` whole pages of ``tokens``."""
        names = _page_digests((scope, kind), tokens, self._page_size)
        return names[len(names) - count :] if count else []

    def record(
        self, scope: Any, tokens: List[int], pool: Any, pages: List[int], kind: str
    ) -> None:
        """At a push: ``pages`` hold the last ``len(pages)`` pages of ``tokens``."""
        contents = self._content(pool, pages)
        if contents is None:
            return
        for name, content in zip(
            self._names(scope, tokens, kind, len(pages)), contents
        ):
            self._pushed.setdefault(name, set()).add(content)
            self._pushed.move_to_end(name)
        while len(self._pushed) > _VERIFY_RECORDS:
            self._pushed.popitem(last=False)

    def verify(
        self, scope: Any, tokens: List[int], pool: Any, pages: List[int], kind: str
    ) -> int:
        """At a recall, before the pages join the tree; returns the pages
        whose bytes were never pushed under that name."""
        contents = self._content(pool, pages)
        if contents is None:
            return 0
        names = self._names(scope, tokens, kind, len(pages))
        mismatched = 0
        for index, (name, content) in enumerate(zip(names, contents)):
            pushed = self._pushed.get(name)
            if pushed is None:
                self.stats["unknown_pages"] += 1
            elif content in pushed:
                self.stats["verified_pages"] += 1
            else:
                mismatched += 1
                self.stats["mismatched_pages"] += 1
                logger.error(
                    "[dram-offload] recalled %s page %d of %d is not what this "
                    "stage pushed for those tokens",
                    kind,
                    index,
                    len(names),
                )
        return mismatched


# Arbitrary: how long a stage waits for the decode's READY before it gives a
# pull up; well past one control round trip.
_READY_TIMEOUT_S = 30.0


class _Candidate(msgspec.Struct):
    leaf: Any
    num_tokens: int


class _PushState(msgspec.Struct):
    leaf: Any
    locks: List[Any]
    sent_at: float
    read_ids: List[str]
    registered: bool


class StagePushOps:
    """Push side of one stage."""

    def __init__(
        self,
        ctx: DramOffloadWorkerContext,
        *,
        stage: int,
        stages: int,
        serving: Callable[[], bool],
        byte_check: Optional[PageByteCheck] = None,
        draft_pools: Tuple[Any, Any] = (None, None),
    ) -> None:
        self._ctx = ctx
        self._stage = stage
        self._stages = stages
        self._serving = serving
        self._byte_check = byte_check
        # This stage's draft sidecar pools (draft KV, draft window); their
        # pages have the ids of the full and window pages they belong to.
        self._draft_pool, self._draft_swa_pool = draft_pools
        self._draft_layout = (
            stage_pool_layout(self._draft_pool, stage=stage, stages=stages)
            if self._draft_pool is not None
            else None
        )
        self._draft_swa_layout = (
            stage_pool_layout(
                self._draft_swa_pool, stage=stage, stages=stages, window=True
            )
            if self._draft_swa_pool is not None
            else None
        )
        self._layout = stage_pool_layout(ctx.host_pool, stage=stage, stages=stages)
        self._swa_layout = (
            stage_pool_layout(
                ctx.swa_host_pool, stage=stage, stages=stages, window=True
            )
            if ctx.swa_host_pool is not None
            else None
        )
        # The same on every stage: the first room this prefill bootstrapped.
        # Independent prefills and restarts cannot alias push labels on a decode.
        self._namespace: Optional[str] = None
        # Leaf -> the decode ranks that served it: (ranks, {rank: endpoint}).
        self._routes = weakref.WeakKeyDictionary()
        self._pushed_len = weakref.WeakKeyDictionary()
        self._pushes: Dict[int, _PushState] = {}

    # ----- tracking, from the hooks ------------------------------------------------

    def note_decode_of(self, req: Any, *, recalled: bool) -> None:
        """At a request's bootstrap commit, in the same slot on every stage."""
        if self._namespace is None:
            self._namespace = f"{req.bootstrap_room:x}"
        peers = self._ctx.channel.stage_decode_peers(req.bootstrap_room)
        if peers is not None:
            req._pd_victim_decode_peer = peers
        self.track_host_leaf(req, on_decode=recalled)

    def track_host_leaf(self, req: Any, *, on_decode: bool = False) -> None:
        peers = getattr(req, "_pd_victim_decode_peer", None)
        node_id = req.last_node
        if node_id is None:
            # Matching can split a node, so every stage matches, peers or not.
            key = RadixKey(
                array("q", req.origin_input_ids),
                extra_key=req.extra_key,
                cache_salt=req.cache_salt,
            ).page_aligned(self._ctx.page_size)
            match = self._ctx.cache.match_prefix(MatchPrefixParams(key=key))
            node_id = match.best_match_node
        node = leaf_of(self._ctx.cache, node_id)
        if node is not None and peers is not None:
            self._routes[node] = peers
            if on_decode:
                self._pushed_len[node] = len(path_tokens(self._ctx.cache, node))

    def release_all(self) -> None:
        if self._pushes:
            raise RuntimeError("stage pushes are in flight; the engine is not idle")
        self._routes.clear()
        self._pushed_len.clear()

    # ----- ops ---------------------------------------------------------------------

    def _under_host_pressure(self) -> bool:
        ctx = self._ctx
        return (
            ctx.host_pool.available_size()
            < prefill_dram_victim_pusher._PUSH_FULL_FREE_TOKENS
            or (
                ctx.swa_host_pool is not None
                and ctx.swa_host_pool.available_size()
                < prefill_dram_victim_pusher._PUSH_SWA_FREE_WINDOWS * ctx.window_tokens
            )
        )

    def _pages_of(self, node: Any) -> Optional[Tuple[List[int], List[int], List[int]]]:
        """(path tokens, host pages, window pages) of a leaf held whole in
        host RAM, else None."""
        ctx = self._ctx
        path = path_tokens(ctx.cache, node)
        pages = host_pages(ctx.cache, ctx.host_pool, node)
        if pages is None:
            return None
        swa_pages = window_host_pages(
            ctx.cache,
            ctx.swa_host_pool,
            node,
            tokens=len(path),
            window_tokens=ctx.window_tokens,
        )
        if swa_pages is None:
            return None
        return path, pages, swa_pages

    def pick(self) -> Optional[Tuple[str, _Candidate]]:
        if self._namespace is None or not self._serving():
            return None
        if not self._under_host_pressure():
            return None
        busy = {push.leaf for push in self._pushes.values()}
        coldest: List[Tuple[Any, List[int]]] = []
        for node, (ranks, endpoints) in list(self._routes.items()):
            pushable = is_live_host_backed_leaf(self._ctx.cache, node)
            if pushable is None:
                self._routes.pop(node, None)
            if not pushable or node in busy or len(endpoints) < ranks:
                continue
            if coldest and node.last_access_time > coldest[0][0].last_access_time:
                continue
            held = self._pages_of(node)
            if held is None or self._pushed_len.get(node) == len(held[0]):
                continue
            if coldest and node.last_access_time < coldest[0][0].last_access_time:
                coldest = []
            coldest.append((node, held[0]))
        if not coldest:
            return None
        # The content names the branch across stages; hashed for the coldest
        # only, and it also orders leaves of one access time the same way.
        (low, high), node, path = min(
            (
                (
                    push_manifest(path, node.key.extra_key, node.key.cache_salt),
                    node,
                    path,
                )
                for node, path in coldest
            ),
            key=lambda entry: entry[0],
        )
        return f"{low:x}.{high:x}:{len(path)}", _Candidate(node, len(path))

    def still_pushable(self, candidate: _Candidate) -> bool:
        node = candidate.leaf
        if not is_live_host_backed_leaf(self._ctx.cache, node):
            return False
        held = self._pages_of(node)
        return held is not None and len(held[0]) == candidate.num_tokens

    def begin(self, candidate: _Candidate, seq: int) -> bool:
        ctx, node = self._ctx, candidate.leaf
        path, pages, swa_pages = self._pages_of(node)
        # Not again at this length, stored or not.
        self._pushed_len[node] = len(path)
        push = _PushState(
            leaf=node,
            locks=lock_host_path(ctx.cache, node),
            sent_at=time.monotonic(),
            read_ids=[],
            registered=False,
        )
        self._pushes[seq] = push
        uuid = prefill_dram_victim_pusher._transfer_uuid(self._namespace, seq)
        # (window, draft sidecar, page ids): a sidecar is registered at the
        # ids of the pool it follows, under the same UUID.
        kinds = [(False, False, pages), (True, False, swa_pages)]
        if self._draft_pool is not None:
            kinds.append((False, True, pages))
        if self._draft_swa_pool is not None:
            kinds.append((True, True, swa_pages))
        for swa, draft, part in kinds:
            if not part:
                continue
            read_id = victim_transfer_id(self._namespace, seq, swa=swa) + (
                ":draft" if draft else ""
            )
            if not ctx.channel.register_host_read(
                read_id, uuid, part, swa=swa, draft=draft
            ):
                return False
            push.read_ids.append(read_id)
        push.registered = True
        if self._byte_check is not None:
            scope = (node.key.extra_key, node.key.cache_salt)
            if self._draft_pool is not None:
                self._byte_check.record(scope, path, self._draft_pool, pages, "draft")
            if swa_pages and self._draft_swa_pool is not None:
                self._byte_check.record(
                    scope, path, self._draft_swa_pool, swa_pages, "draft window"
                )
            self._byte_check.record(scope, path, ctx.host_pool, pages, "full")
            if swa_pages:
                self._byte_check.record(
                    scope, path, ctx.swa_host_pool, swa_pages, "window"
                )
        return ctx.channel.send_stage_push(
            peers=self._routes[node],
            prefill=self._namespace,
            seq=seq,
            stage=self._stage,
            stages=self._stages,
            tokens=path,
            blocks=pages,
            swa_blocks=swa_pages,
            uuid=uuid,
            extra_key=node.key.extra_key,
            cache_salt=node.key.cache_salt,
            window_pages=ctx.window_tokens // ctx.page_size,
            layout=self._layout,
            swa_layout=self._swa_layout,
            draft_layout=self._draft_layout,
            draft_swa_layout=self._draft_swa_layout,
        )

    def outcome(self, seq: int) -> Optional[bool]:
        push = self._pushes[seq]
        channel = self._ctx.channel
        channel.drain_host_reads()
        stored = channel.victim_reply(seq)
        waited = time.monotonic() - push.sent_at
        if stored is None and waited > prefill_dram_victim_pusher._ACK_TIMEOUT_S:
            stored = False
        # An accepted registration is owned by its native read until it settles.
        sent = channel.poll_host_reads(push.read_ids)
        if stored is None or sent is None:
            return None
        # The decode's answer decides. It reads only the pages it does not
        # hold yet, so a registration it had no use for ends unread, at the
        # native deadline, on a branch that is stored all the same.
        return bool(stored and push.registered)

    def finish(self, seq: int, stored: bool) -> None:
        push = self._pushes.pop(seq)
        channel = self._ctx.channel
        channel.forget_host_reads(push.read_ids)
        channel.forget_victim(seq)
        unlock_host_path(self._ctx.cache, push.locks)


class _Pull(msgspec.Struct):
    req: Any
    blocks: List[int]
    swa_blocks: List[int]
    indices: Any = None
    swa_indices: Any = None
    asked_at: float = 0.0
    read_ids: List[str] = msgspec.field(default_factory=list)
    started: bool = False
    failed: bool = False


class StageRecallOps:
    """Recall side of one stage."""

    def __init__(
        self,
        ctx: DramOffloadWorkerContext,
        *,
        stage: int,
        queued: Callable[[], List[Any]],
        byte_check: Optional[PageByteCheck] = None,
        draft_pools: Tuple[Any, Any] = (None, None),
    ) -> None:
        self._ctx = ctx
        self._stage = stage
        self._byte_check = byte_check
        self._draft_pool, self._draft_swa_pool = draft_pools
        # The requests of the bootstrap queue; the same on every stage at a slot.
        self._queued = queued
        self._window_pages = ctx.window_tokens // ctx.page_size
        self._offered: Dict[int, Tuple[List[int], List[int]]] = {}
        self._pulls: Dict[int, _Pull] = {}
        # Room -> (request, node, receipt) of a committed pull.
        self._pins: Dict[int, Tuple[Any, Any, Any]] = {}
        self.recalled: set = set()
        self.stats = dict.fromkeys(
            ("adopted_backup_tokens", "no_memory", "refused", "ready_timeout"), 0
        )

    def max_inflight_pages(self) -> int:
        # As the tensor-parallel puller: half the host pool for landings.
        return int(0.5 * self._ctx.host_pool.size) // self._ctx.page_size

    def _req(self, room: int) -> Any:
        for req in self._queued():
            if req.bootstrap_room == room:
                return req
        return None

    def offer(self, room: int) -> int:
        if room not in self._offered:
            offer = self._ctx.channel.take_prefix_offer(room)
            if offer is None:
                return 0
            self._offered[room] = offer
        blocks, swa_blocks = self._offered[room]
        req = self._req(room)
        if req is None:
            # The offer can arrive before the request reaches this stage.
            return 0
        page = self._ctx.page_size
        if not 0 < len(blocks) * page <= offer_limit(req=req, page_size=page) or len(
            swa_blocks
        ) != min(len(blocks), self._window_pages):
            # Nothing to resume from: every stage sees the same offer and the
            # same request, so each lets the room leave bootstrap by itself.
            self.resolve(room)
            return 0
        return len(blocks)

    def lacks(self, room: int, pages: int) -> bool:
        req = self._req(room)
        if req is None:
            return False
        tokens = pages * self._ctx.page_size
        key = prefix_radix_key(req=req, num_tokens=tokens)
        match = self._ctx.cache.match_prefix(MatchPrefixParams(key=key))
        return len(match.device_indices) + match.host_hit_length < tokens

    def start(self, room: int, pages: int) -> bool:
        ctx = self._ctx
        req = self._req(room)
        offered = self._offered.pop(room, None)
        if req is None or offered is None or len(offered[0]) != pages:
            return False
        pull = _Pull(req=req, blocks=offered[0], swa_blocks=offered[1])
        self._pulls[room] = pull
        page = ctx.page_size
        for pool, tokens, component in (
            (ctx.host_pool, pages * page, ComponentType.FULL),
            (ctx.swa_host_pool, len(pull.swa_blocks) * page, ComponentType.SWA),
        ):
            if not tokens:
                continue
            short = max(0, tokens - pool.available_size())
            if short:
                ctx.cache.evict_host(short, component)
            indices = pool.alloc(tokens)
            if indices is None:
                self.stats["no_memory"] += 1
                return False
            if component == ComponentType.FULL:
                pull.indices = indices
            else:
                pull.swa_indices = indices
        pull.asked_at = time.monotonic()
        return ctx.channel.ask_stage_prefix_pull(room, 0, pages, stage=self._stage)

    def outcome(self, room: int) -> Optional[bool]:
        pull = self._pulls[room]
        channel = self._ctx.channel
        if not pull.started:
            ready = channel.prefix_ready(room)
            if ready is None:
                if time.monotonic() - pull.asked_at > _READY_TIMEOUT_S:
                    self.stats["ready_timeout"] += not pull.failed
                    pull.failed = True
                    return False
                return None
            pull.started = True
            self.stats["refused"] += bool(ready.refused)
            pull.failed = ready.refused or not self._start_reads(room, pull, ready)
        channel.drain_host_reads()
        done = channel.poll_host_reads(pull.read_ids)
        if done is None:
            return None
        return bool(done) and not pull.failed

    def _start_reads(self, room: int, pull: _Pull, ready: Any) -> bool:
        ctx = self._ctx
        if ready.has_window != bool(pull.swa_blocks):
            return False
        reads = [
            (
                False,
                ready.uuid,
                ready.endpoint,
                pull.blocks,
                ctx.host_pool,
                pull.indices,
            )
        ]
        if pull.swa_blocks:
            reads.append(
                (
                    True,
                    ready.swa_uuid,
                    ready.swa_endpoint,
                    pull.swa_blocks,
                    ctx.swa_host_pool,
                    pull.swa_indices,
                )
            )
        # A draft sidecar lands at the host page ids of the landing it
        # belongs to; the decode must have registered every sidecar this
        # stage has, or the draft would read pages nobody wrote.
        sidecars = (
            (False, self._draft_pool, ready.draft_endpoint),
            (True, self._draft_swa_pool, ready.draft_swa_endpoint),
        )
        for swa, pool, endpoint in sidecars:
            source = [r for r in reads if r[0] == swa]
            if pool is None or not source:
                continue
            if not endpoint:
                return False
            _, uuid, _, remote, source_pool, indices = source[0]
            reads.append((swa, uuid, endpoint, remote, source_pool, indices, True))
        for swa, uuid, endpoint, remote, pool, indices, *draft in reads:
            read_id = ctx.channel.prefix_req_id(room, swa=swa, draft=bool(draft))
            if not ctx.channel.start_host_read(
                read_id,
                uuid,
                endpoint,
                remote,
                host_page_ids(pool, indices),
                swa=swa,
                draft=bool(draft),
            ):
                return False
            pull.read_ids.append(read_id)
        return True

    def _free(self, pull: _Pull) -> None:
        if pull.indices is not None:
            self._ctx.host_pool.free(pull.indices)
        if pull.swa_indices is not None:
            self._ctx.swa_host_pool.free(pull.swa_indices)

    def commit(self, room: int) -> None:
        ctx = self._ctx
        pull = self._pulls.pop(room)
        ctx.channel.forget_host_reads(pull.read_ids)
        if self._req(room) is not pull.req:
            # The request left the queue: nothing resumes from these pages.
            self._free(pull)
            return
        if self._byte_check is not None:
            tokens = list(pull.req.origin_input_ids[: len(pull.blocks) * ctx.page_size])
            scope = (pull.req.extra_key, pull.req.cache_salt)
            self._byte_check.verify(
                scope,
                tokens,
                ctx.host_pool,
                host_page_ids(ctx.host_pool, pull.indices),
                "full",
            )
            if pull.swa_blocks:
                self._byte_check.verify(
                    scope,
                    tokens,
                    ctx.swa_host_pool,
                    host_page_ids(ctx.swa_host_pool, pull.swa_indices),
                    "window",
                )
            if self._draft_pool is not None:
                self._byte_check.verify(
                    scope,
                    tokens,
                    self._draft_pool,
                    host_page_ids(ctx.host_pool, pull.indices),
                    "draft",
                )
            if pull.swa_blocks and self._draft_swa_pool is not None:
                self._byte_check.verify(
                    scope,
                    tokens,
                    self._draft_swa_pool,
                    host_page_ids(ctx.swa_host_pool, pull.swa_indices),
                    "draft window",
                )
        node, _ = insert_host_prefix(
            ctx.cache,
            prefix_radix_key(req=pull.req, num_tokens=len(pull.blocks) * ctx.page_size),
            pull.indices,
            ctx.page_size,
            swa_indices=pull.swa_indices,
            stats=self.stats,
        )
        if node is not None:
            receipt = ctx.cache.inc_host_lock_ref(node).to_dec_params()
            self._pins[room] = (pull.req, node, receipt)
            self.recalled.add(room)

    def abandon(self, room: int) -> None:
        pull = self._pulls.pop(room, None)
        if pull is not None:
            # Every stage's read has settled: the verdict needed each outcome.
            self._ctx.channel.forget_host_reads(pull.read_ids)
            self._free(pull)

    def resolve(self, room: int) -> None:
        self._offered.pop(room, None)
        self._ctx.channel.resolve_prefix_pull(room)

    # ----- from the hooks --------------------------------------------------------

    def unpin_departed(self, queued_reqs: List[Any]) -> None:
        """Unlock the paths of requests no longer queued; queue membership is
        the same on every stage at a slot."""
        queued = {id(req) for req in queued_reqs}
        for room, (req, node, receipt) in list(self._pins.items()):
            if id(req) not in queued:
                del self._pins[room]
                self.recalled.discard(room)
                self._ctx.cache.dec_host_lock_ref(node, receipt)

    def drop_departed(self, queued_reqs: List[Any]) -> None:
        queued = {req.bootstrap_room for req in queued_reqs}
        for room in [room for room in self._offered if room not in queued]:
            del self._offered[room]

    def holds_host_landing_pages(self) -> bool:
        return bool(self._pulls)

    def release_all(self) -> None:
        if self._pulls:
            raise RuntimeError("stage recalls are in flight; the engine is not idle")
        self._offered.clear()
        for room, (_, node, receipt) in list(self._pins.items()):
            del self._pins[room]
            self._ctx.cache.dec_host_lock_ref(node, receipt)
        self.recalled.clear()

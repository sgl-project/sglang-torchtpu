"""Decode side of the offload for a pipeline-parallel prefill: land the
branches its stages push into stage-layout pools, and offer them back.

Why: a tensor-parallel decode cannot hold a pipeline stage's pages in its own
     host pool (stage_layout_store.py), so the two workers of the
     tensor-parallel pairing (decode_dram_victim_receiver.py,
     decode_dram_prefix_offerer.py), which insert into and match the decode's
     radix cache, do not apply.
How: every stage pushes its share of a branch to the decode rank that hosts
     its pool (``hosting_rank``); stage zero also sends the branch's tokens to
     every rank. The ranks agree on the pushes all of them hold whole, reserve
     the same page ids in the index and read each hosted stage's pages into
     that stage's pool. A request is matched against the index, not against
     the radix cache: the decode offers the page ids, each stage reads its own
     share back from its hosting rank, and the decode itself still receives
     the whole prompt from the prefill.

The decode's ranks are tensor-parallel, so they agree with the blocking
collectives of rank_lockstep.RankLockstep. This module holds no torch and no
transport: the lockstep, the channel and the pools are passed in.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from sglang_tpu.disaggregation.decode_dram_offload.stage_layout_store import (
    Landing,
    Match,
    StageLayoutStore,
    hosted_stages,
    size_store,
)

logger = logging.getLogger(__name__)

# Arbitrary, as in decode_dram_victim_receiver: decode iterations between
# agreement rounds and landings at once.
_CADENCE = 16
# Arbitrary: how long a push may wait for the parts of every stage. The stages
# apply a verdict one after the other along the ring, a forward apart, so
# their parts arrive seconds apart on a large model; counting agreement
# rounds dropped such pushes (seen live), since an idle decode runs many
# rounds a second.
_MAX_WAIT_S = 20.0
MAX_INFLIGHT = 4

PushKey = Tuple[str, int]


def stage_transfer_id(
    prefill: str, seq: int, stage: int, *, swa: bool = False, draft: bool = False
) -> str:
    return (
        f"victim:{prefill}:{seq}:s{stage}"
        + (":swa" if swa else "")
        + (":draft" if draft else "")
    )


def _layouts(part: Any) -> List[Tuple[bool, bool, Any]]:
    """(window, draft sidecar, layout) of every pool a stage's part brings.
    A draft sidecar is addressed by the page ids of the pool it follows."""
    kinds = [(False, False, part.layout)]
    if part.swa_blocks:
        kinds.append((True, False, part.swa_layout))
    if getattr(part, "draft_layout", None) is not None:
        kinds.append((False, True, part.draft_layout))
    if part.swa_blocks and getattr(part, "draft_swa_layout", None) is not None:
        kinds.append((True, True, part.draft_swa_layout))
    return kinds


def _manifest(header: Any) -> Tuple[int, int]:
    """Two non-negative int64 values naming a pushed branch across ranks."""
    text = repr(
        (
            tuple(header.tokens),
            header.extra_key,
            header.cache_salt,
            header.stages,
            header.window_pages,
        )
    )
    digest = hashlib.blake2b(text.encode(), digest_size=16).digest()
    return tuple(int.from_bytes(digest[i : i + 8], "little") >> 1 for i in (0, 8))


@dataclass
class StagePush:
    """What a decode rank holds of one pushed branch: ``header`` from stage
    zero (tokens, sampling scope, stage count, the model's window in pages)
    and one part per stage this rank hosts (page ids, read source, layout,
    where to answer). A part's window is the branch's last pages, a whole
    window or the whole branch when that is shorter."""

    header: Any = None
    parts: Dict[int, Any] = field(default_factory=dict)
    manifest: Optional[Tuple[int, int]] = None

    def window_len(self, page_size: int) -> int:
        return min(len(self.header.tokens) // page_size, self.header.window_pages)


class StagePoolSet:
    """This decode rank's host-only pools, one per hosted stage and one more
    per hosted stage for sliding windows, and with a draft the stage's draft
    sidecar of each. They are built at the first push, when the stages'
    layouts are known, and every pool of every rank gets the same number of
    pages so that one page id serves all of them; a sidecar has the pages of
    the pool it follows."""

    def __init__(
        self,
        *,
        rank: int,
        decode_ranks: int,
        budget_bytes: int,
        build_pool: Callable[[Any, int], Any],
    ) -> None:
        self.rank = rank
        self.decode_ranks = decode_ranks
        self._budget_bytes = budget_bytes
        # (layout of a part, pages) -> an object with RaidenKVTransfer's host
        # read methods, over host memory in that layout.
        self._build_pool = build_pool
        # By (stage, window, draft sidecar).
        self._pools: Dict[Tuple[int, bool, bool], Any] = {}
        self._fingerprints: Dict[Tuple[int, bool, bool], int] = {}
        self.sizes: Optional[Tuple[int, int]] = None
        # Why no pool can be built; every push is declined from then on.
        self.refusal: Optional[str] = None

    def hosted(self, stages: int) -> List[int]:
        return hosted_stages(self.rank, stages, self.decode_ranks)

    def transfer(self, stage: int, *, swa: bool = False, draft: bool = False) -> Any:
        return self._pools[(stage, swa, draft)]

    def has(self, stage: int, *, swa: bool = False, draft: bool = False) -> bool:
        return (stage, swa, draft) in self._pools

    def fits(self, parts: Dict[int, Any]) -> bool:
        """Whether the parts are in the layouts the pools were built for,
        draft sidecars included: a stage that brings a draft needs its pool,
        and a pool built with a draft needs the stage to bring it."""
        for stage, part in parts.items():
            brought = {(swa, draft): layout for swa, draft, layout in _layouts(part)}
            for (swa, draft), layout in brought.items():
                if self._fingerprints.get((stage, swa, draft)) != layout.fingerprint():
                    return False
            for swa in (False, True):
                if (swa, False) in brought and self.has(stage, swa=swa, draft=True):
                    if (swa, True) not in brought:
                        return False
        return True

    def build(
        self, parts: Dict[int, Any], *, window_pages: int, lockstep: Any
    ) -> Optional[Tuple[int, int]]:
        """Size and allocate the pools from the first agreed push: (pages,
        windows) of every pool, the same on all ranks, or None once any rank
        could not build its own."""

        # A page costs its bytes in the pool and in the draft sidecar that
        # follows it, on the stages that have one.
        def cost(swa: bool) -> int:
            return sum(
                layout.page_bytes
                for part in parts.values()
                for is_swa, _, layout in self._kinds(part, window_pages)
                if is_swa == swa
            )

        page_bytes = lockstep.max(cost(False))
        swa_bytes = lockstep.max(cost(True) if window_pages else 0)
        # A rank that hosts no stage reports zero; the largest rank sizes all.
        pages, windows = size_store(
            budget_bytes=self._budget_bytes,
            page_bytes=max(page_bytes, 1),
            window_page_bytes=swa_bytes,
            window_pages=window_pages if swa_bytes else 0,
        )
        error = None
        if pages < 1 or (window_pages and windows < 1):
            error = f"budget of {self._budget_bytes} bytes holds no page or no window"
        else:
            try:
                for stage, part in sorted(parts.items()):
                    for swa, draft, layout in self._kinds(part, window_pages):
                        count = windows * window_pages if swa else pages
                        self._add(stage, swa, draft, layout, count)
            except Exception as exc:  # noqa: BLE001 - the offload is declined
                error = f"cannot build a stage-layout pool: {exc}"
        if error is not None:
            logger.error("[dram-offload] stage store on rank %d: %s", self.rank, error)
        if not lockstep.all(error is None):
            self.refusal = error or "another decode rank could not build its pools"
            return None
        self.sizes = (pages, windows)
        logger.info(
            "[dram-offload] stage store ready on rank %d: stages %s, %d pages "
            "(%d bytes each on the largest rank), %d windows of %d pages",
            self.rank,
            sorted(parts),
            pages,
            page_bytes,
            windows,
            window_pages,
        )
        return self.sizes

    @staticmethod
    def _kinds(part: Any, window_pages: int) -> List[Tuple[bool, bool, Any]]:
        """The pools to build for a stage from its first part. The window
        pools are built whenever the model has a window, also when this
        branch is too short to bring window pages of its own."""
        kinds = [(False, False, part.layout)]
        if window_pages:
            kinds.append((True, False, part.swa_layout))
        if getattr(part, "draft_layout", None) is not None:
            kinds.append((False, True, part.draft_layout))
        if window_pages and getattr(part, "draft_swa_layout", None) is not None:
            kinds.append((True, True, part.draft_swa_layout))
        return kinds

    def _add(self, stage: int, swa: bool, draft: bool, layout: Any, pages: int) -> None:
        self._pools[(stage, swa, draft)] = self._build_pool(layout, pages)
        self._fingerprints[(stage, swa, draft)] = layout.fingerprint()


@dataclass
class _StageLanding:
    push: StagePush
    landing: Landing
    read_ids: List[str] = field(default_factory=list)
    failed: bool = False


class DecodeStageBranchReceiver:
    """Lands the branches a pipeline prefill's stages push to this decode."""

    def __init__(
        self,
        *,
        channel: Any,
        lockstep: Any,
        pools: StagePoolSet,
        page_size: int,
        log_stats: Callable[[Dict[str, int]], None] = lambda stats: None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._channel = channel
        self._lockstep = lockstep
        self._pools = pools
        self._page_size = page_size
        self._log_stats = log_stats
        self._clock = clock
        self.store: Optional[StageLayoutStore] = None
        self._iterations = 0
        self._rounds = 0
        # When a received push was first seen, on this rank's clock.
        self._first_seen: Dict[PushKey, float] = {}
        self._landings: Dict[PushKey, _StageLanding] = {}
        self.stats = dict.fromkeys(
            ("received", "granted", "declined", "held", "failed", "dropped", "layout"),
            0,
        )

    def has_host_landings(self) -> bool:
        return bool(self._landings)

    def release_all(self) -> None:
        """Before a cache reset, which core issues only on an idle engine and
        on every rank: forget every stored branch."""
        if self._landings:
            raise RuntimeError("stage pushes are landing; the engine is not idle")
        if self.store is not None:
            self.store.clear()

    def counters(self) -> Dict[str, int]:
        return dict(self.stats, **(self.store.counters() if self.store else {}))

    def advance(self) -> None:
        """Every decode iteration, every rank; acts once per ``_CADENCE``."""
        self._iterations += 1
        if self._iterations % _CADENCE:
            return
        self._rounds += 1
        self._admit_agreed_pushes()
        self.finish_agreed_landings()
        self._log_stats(self.counters())

    # ----- admission ---------------------------------------------------------------

    def _whole(self, push: StagePush) -> bool:
        """Whether this rank holds the header and the part of every stage it
        hosts, each naming as many pages as the header has."""
        header = push.header
        if header is None:
            return False
        pages = len(header.tokens) // self._page_size
        hosted = self._pools.hosted(header.stages)
        return all(
            stage in push.parts
            and len(push.parts[stage].blocks) == pages
            and len(push.parts[stage].swa_blocks) == push.window_len(self._page_size)
            for stage in hosted
        )

    def _admit_agreed_pushes(self) -> None:
        received = {
            key: push
            for key, push in self._channel.received_stage_pushes().items()
            if key not in self._landings
        }
        now = self._clock()
        for key in received:
            if key not in self._first_seen:
                self._first_seen[key] = now
                self.stats["received"] += 1
        manifests = {}
        for key, push in received.items():
            if self._whole(push):
                if push.manifest is None:
                    push.manifest = _manifest(push.header)
                manifests[(*key, *push.manifest)] = key
        for manifest in self._lockstep.common_to_all_ranks(list(manifests)):
            key = manifests[manifest]
            push = self._channel.take_stage_push(key)
            self._first_seen.pop(key, None)
            stored = self._admit(key, push)
            if stored is not None:
                self._ack(push, stored=stored)
        for key, first in list(self._first_seen.items()):
            if now - first > _MAX_WAIT_S:
                self._first_seen.pop(key)
                if self._channel.take_stage_push(key) is not None:
                    self.stats["dropped"] += 1

    def _admit(self, key: PushKey, push: StagePush) -> Optional[bool]:
        """Start landing an agreed push. The answer for the prefill when it
        does not land (True: the decode holds it already), None when it does."""
        header = push.header
        hosted = {s: push.parts[s] for s in self._pools.hosted(header.stages)}
        if not self._ensure_store(hosted, header.window_pages):
            self.stats["declined"] += 1
            return False
        if not self._lockstep.all(
            self._pools.fits(hosted)
            and header.window_pages in (0, self.store.window_pages)
        ):
            self.stats["layout"] += 1
            return False
        scope = (header.extra_key, header.cache_salt)
        with_window = bool(header.window_pages)
        if self.store.holds(scope, header.tokens, with_window=with_window):
            self.stats["held"] += 1
            return True
        if len(self._landings) >= MAX_INFLIGHT:
            self.stats["declined"] += 1
            return False
        landing = self.store.reserve(scope, header.tokens, with_window=with_window)
        if landing is None:
            self.stats["declined"] += 1
            return False
        entry = _StageLanding(push=push, landing=landing)
        started = all(
            self._start_reads(key, stage, part, entry)
            for stage, part in sorted(hosted.items())
        )
        # A rank whose read started cannot release its destination, so every
        # rank keeps its reservation until the started reads settle.
        entry.failed = not self._lockstep.all(started)
        self._landings[key] = entry
        self.stats["granted"] += 1
        return None

    def _ensure_store(self, hosted: Dict[int, Any], window_pages: int) -> bool:
        if self.store is not None:
            return True
        if self._pools.refusal is not None:
            return False
        sizes = self._pools.build(
            hosted, window_pages=window_pages, lockstep=self._lockstep
        )
        if sizes is None:
            return False
        self.store = StageLayoutStore(
            page_size=self._page_size,
            pages=sizes[0],
            window_pages=window_pages,
            windows=sizes[1],
        )
        return True

    def _start_reads(
        self, key: PushKey, stage: int, part: Any, entry: _StageLanding
    ) -> bool:
        """Read the stage's pages the index does not hold yet, and its window,
        into the reserved pages of the stage's pools."""
        landing = entry.landing
        # (window, draft sidecar, source address, remote ids, local ids). A
        # sidecar's pages have the ids of the pool it follows on both sides.
        reads = []
        full = (part.blocks[landing.first_new_page :], landing.new_pages)
        window = (
            (part.swa_blocks, landing.window[: len(part.swa_blocks)])
            if landing.window is not None
            else None
        )
        if landing.new_pages:
            reads.append((False, False, part.address, *full))
            if getattr(part, "draft_layout", None) is not None:
                reads.append((False, True, part.draft_address, *full))
        if window is not None:
            reads.append((True, False, part.swa_address, *window))
            if getattr(part, "draft_swa_layout", None) is not None:
                reads.append((True, True, part.draft_swa_address, *window))
        for swa, draft, address, remote, local in reads:
            read_id = stage_transfer_id(*key, stage, swa=swa, draft=draft)
            if not self._channel.start_stage_read(
                read_id,
                part.uuid,
                address,
                list(remote),
                list(local),
                stage=stage,
                swa=swa,
                draft=draft,
            ):
                return False
            entry.read_ids.append(read_id)
        return True

    # ----- completion --------------------------------------------------------------

    def finish_agreed_landings(self) -> None:
        """Commit, or give back, the landings whose reads settled on every
        rank, and tell every hosted stage; a paused engine's tick calls this."""
        if not self._landings:
            return
        self._channel.drain_host_reads()
        keys = sorted(self._landings)
        row = []
        for key in keys:
            entry = self._landings[key]
            received = self._channel.poll_host_reads(entry.read_ids)
            row += [
                int(received is not None),
                int(received is True and not entry.failed),
            ]
        low, _ = self._lockstep.min_max(row)
        for index, key in enumerate(keys):
            settled, received = low[2 * index], low[2 * index + 1]
            if not settled:
                continue
            entry = self._landings.pop(key)
            self._channel.forget_host_reads(entry.read_ids)
            if received:
                self.store.commit(entry.landing)
            else:
                self.store.abort(entry.landing)
                self.stats["failed"] += 1
            self._ack(entry.push, stored=bool(received))

    def _ack(self, push: StagePush, *, stored: bool) -> None:
        """Every stage hears the one agreed answer from the rank that hosts it."""
        for part in push.parts.values():
            self._channel.send_victim_ack(
                endpoint=part.ack_endpoint, seq=part.seq, stored=stored
            )


class DecodeStageBranchOfferer:
    """Offers the stored prefix of a request back to the pipeline prefill and
    keeps it pinned until the request left the queues and the stages' reads
    settled."""

    def __init__(
        self,
        *,
        channel: Any,
        lockstep: Any,
        receiver: DecodeStageBranchReceiver,
        owned_rooms: Callable[[], Iterable[int]],
    ) -> None:
        self._channel = channel
        self._lockstep = lockstep
        self._receiver = receiver
        # Rooms whose request a decode queue still owns.
        self._owned_rooms = owned_rooms
        self.offers: Dict[int, Match] = {}
        self._released: set = set()
        self.stats = dict.fromkeys(("offers", "offered_pages"), 0)

    def offer_stage_prefix(self, *, room: int, tokens: List[int], scope: Any) -> None:
        """Pin and offer what the index holds of a request's prompt, before
        its request for KV goes to the stages. The index is the same on every
        rank, so every rank offers the same page ids."""
        store = self._receiver.store
        if store is None or room in self.offers:
            return
        match = store.match(scope, tokens)
        if match is None:
            return
        store.pin(match)
        self.offers[room] = match
        self.stats["offers"] += 1
        self.stats["offered_pages"] += match.num_pages
        self._channel.offer_stage_prefix(room, match.pages, match.window)

    def settle_offers(self) -> None:
        """Every decode iteration, every rank."""
        self.close_departed_offers()
        self.unpin_settled_offers()

    def close_departed_offers(self) -> None:
        if len(self.offers) == len(self._released):
            return
        owned = set(self._owned_rooms())
        for room in self.offers:
            if room not in owned and room not in self._released:
                # Closed before the pin goes: a late pull must not register
                # pages the queue let go of.
                self._channel.release_prefix_offer(room)
                self._released.add(room)

    def unpin_settled_offers(self) -> None:
        """Unpin the closed offers no stage's read can still touch, on any rank."""
        if not self._released:
            return
        self._channel.drain_host_reads()
        rooms = sorted(self._released)
        low, _ = self._lockstep.min_max(
            [int(not self._channel.host_prefix_pending(room)) for room in rooms]
        )
        for room, done in zip(rooms, low):
            if done:
                self._receiver.store.unpin(self.offers.pop(room))
                self._released.remove(room)

    def release_all(self) -> None:
        if self.offers:
            raise RuntimeError("stage prefix offers are pinned; the engine is not idle")

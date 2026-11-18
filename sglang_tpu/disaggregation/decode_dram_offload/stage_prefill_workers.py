"""Prefill side of the offload under pipeline parallelism: what each stage
votes and what all stages do on a verdict (stage_agreement.py).

Each stage runs one ``StagePushCore`` and one ``StageRecallCore``. They hold
the decisions only; what a decision does to the tree, the host pool and the
wire is behind ``ops`` (stage_prefill_ops.py), so this module has no torch
and its test drives a simulated ring of stages.

Push, per branch:
    every stage proposes its coldest pushable leaf  ->  all proposed the same:
    each locks the path, registers its pages and sends its share to the decode
    rank that hosts its pool  ->  every stage knows its outcome: the branch is
    stored only if no stage failed.

Recall, per request in bootstrap:
    every stage holds the decode's offer of the same length  ->  each checks
    whether its tree lacks the prefix; if any does, all pull it  ->  every
    stage's read settled: all insert the prefix, or none does and the prefill
    computes the prompt.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Iterable

from sglang_tpu.disaggregation.decode_dram_offload.stage_agreement import (
    Verdict,
    Votes,
)

# Arbitrary, as in prefill_dram_victim_pusher: pushes in flight at once.
MAX_INFLIGHT = 4
# Verdicts repeat for about one loop; this many keys is far more than a loop.
_REMEMBERED = 256


class _Recent:
    """The last keys acted on, so a repeated verdict is applied once."""

    def __init__(self) -> None:
        self._keys: OrderedDict[str, None] = OrderedDict()

    def first_time(self, key: str) -> bool:
        if key in self._keys:
            return False
        self._keys[key] = None
        if len(self._keys) > _REMEMBERED:
            self._keys.popitem(last=False)
        return True


@dataclass
class _Push:
    # This stage could not register or send its share.
    failed_here: bool


class StagePushCore:
    """``ops``:
    pick() -> Optional[(key, leaf)]: this stage's coldest pushable leaf now,
        and a key that is equal on stages that picked the same branch at the
        same length; None when it would not push (not serving, no host
        pressure, nothing pushable).
    still_pushable(leaf) -> bool: from the tree alone.
    begin(leaf, seq) -> bool: lock the path, then register this stage's pages
        and send its share; False if the registration or the send failed.
    outcome(seq) -> Optional[bool]: None until this stage's registration is
        settled and the decode answered or the wait ran out; then whether
        this stage's share was stored.
    finish(seq, stored): unlock; ``stored`` is the same on every stage.
    """

    def __init__(self, ops: Any) -> None:
        self._ops = ops
        self._pushes: Dict[int, _Push] = {}
        self._proposed: OrderedDict[str, Any] = OrderedDict()
        self._begun = _Recent()
        self._next_seq = 1
        self.stats = dict.fromkeys(("started", "stored", "declined", "stale"), 0)

    def has_pushes_in_flight(self) -> bool:
        return bool(self._pushes)

    def votes(self, votes: Votes) -> None:
        if len(self._pushes) < MAX_INFLIGHT:
            picked = self._ops.pick()
            if picked is not None:
                key, leaf = picked
                self._proposed[key] = leaf
                self._proposed.move_to_end(key)
                if len(self._proposed) > _REMEMBERED:
                    self._proposed.popitem(last=False)
                votes.all_of.add(f"push?{key}")
        for seq, push in self._pushes.items():
            outcome = False if push.failed_here else self._ops.outcome(seq)
            if outcome is not None:
                votes.all_of.add(f"push!{seq}")
                if not outcome:
                    votes.any_of.add(f"push-{seq}")

    def settle(self, verdict: Verdict) -> None:
        for seq in sorted(self._pushes):
            if f"push!{seq}" in verdict.all_of:
                stored = f"push-{seq}" not in verdict.any_of
                del self._pushes[seq]
                self.stats["stored" if stored else "declined"] += 1
                self._ops.finish(seq, stored)
        for name in sorted(verdict.all_of):
            if not name.startswith("push?") or not self._begun.first_time(name):
                continue
            # Every stage cast this key, so every stage holds its leaf.
            leaf = self._proposed.pop(name[5:], None)
            if (
                leaf is None
                or len(self._pushes) >= MAX_INFLIGHT
                or not self._ops.still_pushable(leaf)
            ):
                self.stats["stale"] += 1
                continue
            seq = self._next_seq
            self._next_seq += 1
            self.stats["started"] += 1
            self._pushes[seq] = _Push(failed_here=not self._ops.begin(leaf, seq))


@dataclass
class _Recall:
    pages: int
    # "eval": each stage says whether it lacks the prefix. "pull": reading.
    phase: str
    lacks_here: bool = False
    failed_here: bool = False


class StageRecallCore:
    """``ops``:
    offer(room) -> int: pages the decode offered this stage for the room, 0
        if none arrived (yet).
    lacks(room, pages) -> bool: whether this stage's tree does not hold the
        offered prefix; from the tree alone.
    start(room, pages) -> bool: reserve host pages and ask the hosting decode
        rank for this stage's share; False if it could not.
    outcome(room) -> Optional[bool]: None until this stage's read settled or
        was refused; then whether the pages arrived. Drives the read.
    commit(room): insert the read prefix into the tree and pin it for the
        request. abandon(room): give the reserved pages back.
    resolve(room): the room may leave bootstrap.
    """

    def __init__(self, ops: Any, *, max_inflight_pages: int) -> None:
        self._ops = ops
        self._max_inflight_pages = max_inflight_pages
        self._recalls: Dict[int, _Recall] = {}
        self._done = _Recent()
        self.stats = dict.fromkeys(
            ("offers", "pulls", "pulled_pages", "held", "over_budget", "failed"), 0
        )

    def has_recalls_in_flight(self) -> bool:
        return any(recall.phase == "pull" for recall in self._recalls.values())

    def _inflight_pages(self) -> int:
        return sum(r.pages for r in self._recalls.values() if r.phase == "pull")

    def votes(self, votes: Votes, rooms: Iterable[int]) -> None:
        for room in rooms:
            if room not in self._recalls:
                pages = self._ops.offer(room)
                if pages:
                    votes.all_of.add(f"recall?{room}:{pages}")
        for room, recall in self._recalls.items():
            if recall.phase == "eval":
                votes.all_of.add(f"recall={room}")
                if recall.lacks_here:
                    votes.any_of.add(f"recall+{room}")
                continue
            outcome = False if recall.failed_here else self._ops.outcome(room)
            if outcome is not None:
                votes.all_of.add(f"recall!{room}")
                if not outcome:
                    votes.any_of.add(f"recall-{room}")

    def settle(self, verdict: Verdict) -> None:
        for room in sorted(self._recalls):
            recall = self._recalls[room]
            if recall.phase == "eval" and f"recall={room}" in verdict.all_of:
                self._decide_pull(room, recall, f"recall+{room}" in verdict.any_of)
            elif recall.phase == "pull" and f"recall!{room}" in verdict.all_of:
                del self._recalls[room]
                if f"recall-{room}" in verdict.any_of:
                    self.stats["failed"] += 1
                    self._ops.abandon(room)
                else:
                    self.stats["pulls"] += 1
                    self.stats["pulled_pages"] += recall.pages
                    self._ops.commit(room)
                self._ops.resolve(room)
        for name in sorted(verdict.all_of):
            if not name.startswith("recall?") or not self._done.first_time(name):
                continue
            room, pages = (int(x) for x in name[7:].split(":"))
            if room in self._recalls:
                continue
            self.stats["offers"] += 1
            self._recalls[room] = _Recall(
                pages=pages, phase="eval", lacks_here=bool(self._ops.lacks(room, pages))
            )

    def _decide_pull(self, room: int, recall: _Recall, lacking: bool) -> None:
        if not lacking:
            self.stats["held"] += 1
        elif self._inflight_pages() + recall.pages > self._max_inflight_pages:
            self.stats["over_budget"] += 1
        else:
            recall.phase = "pull"
            recall.failed_here = not self._ops.start(room, recall.pages)
            return
        del self._recalls[room]
        self._ops.resolve(room)

    def open_rooms(self) -> Iterable[int]:
        return list(self._recalls)


class StagePrefillWorkers:
    """One stage's two workers behind the two calls the hooks make."""

    def __init__(self, push: StagePushCore, recall: StageRecallCore) -> None:
        self.push = push
        self.recall = recall

    def cast(self, rooms: Iterable[int]) -> Votes:
        votes = Votes()
        self.recall.votes(votes, rooms)
        self.push.votes(votes)
        return votes

    def apply(self, verdict: Verdict) -> None:
        self.recall.settle(verdict)
        self.push.settle(verdict)

    def busy(self) -> bool:
        return self.push.has_pushes_in_flight() or self.recall.has_recalls_in_flight()

    def counters(self) -> Dict[str, int]:
        return {
            **{f"push_{k}": v for k, v in self.push.stats.items()},
            **{f"recall_{k}": v for k, v in self.recall.stats.items()},
        }

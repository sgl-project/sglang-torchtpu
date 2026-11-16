"""Agreement across the stages of a pipeline-parallel prefill, for the offload
to decode DRAM.

Why: every stage holds its own layers of a branch in its own host pool and its
     transfers settle on its own clock, but a tree may change only in a step
     every stage takes. The stages reach a micro-batch at different times, so
     a blocking collective (rank_lockstep.RankLockstep, the tensor-parallel
     case) would stall the pipeline.
How: the pipeline already agrees on which requests left bootstrap: in a
     micro-batch slot each stage intersects its "ready" request ids with the
     previous stage's and unites the "failed" ones, the last stage's lists
     travel the ring, and every stage applies them in the same later slot
     (scheduler_pp_mixin: _pp_pd_get_bootstrapped_ids, process_bootstrapped_queue).
     The offload adds marked entries to those two lists. A key in the first
     list comes back only if every stage cast it in that slot; a key in the
     second comes back if any stage did. Nothing else is sent.

Rules for the workers that vote this way (stage_prefill_workers.py):

- A verdict arrives a loop after its votes, and the tree changes in between.
  Vote only on local facts that cannot revert (a message arrived, a read
  settled, a registration was accepted) and keep casting them every slot.
- Whatever depends on the tree is decided when the verdict is applied, from
  the tree as it is then: every stage is at the same point of the same
  sequence of tree changes there. Nothing is locked before a verdict.
- A verdict repeats while its votes are still in the ring, so applying one is
  idempotent.

No transport and no torch here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Set

# No request id starts with a NUL.
MARK = "\x00dram-offload:"


def mark(keys: Iterable[str]) -> List[str]:
    """The entries that carry ``keys`` in a list of request ids."""
    return [MARK + key for key in sorted(keys)]


def unmark(rids: Iterable[str]) -> Set[str]:
    """The keys carried in a list of request ids."""
    return {
        rid[len(MARK) :]
        for rid in rids or ()
        if isinstance(rid, str) and rid.startswith(MARK)
    }


def marked(rids: Iterable[str]) -> List[str]:
    """The carrying entries of a list, to pass a verdict on unchanged."""
    return [rid for rid in rids or () if isinstance(rid, str) and rid.startswith(MARK)]


@dataclass
class Votes:
    """One stage's votes in one slot: ``all_of`` holds only if every stage
    casts it, ``any_of`` if any stage does."""

    all_of: Set[str] = field(default_factory=set)
    any_of: Set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Verdict:
    """What the ring returned for a slot."""

    all_of: frozenset = frozenset()
    any_of: frozenset = frozenset()

    @classmethod
    def of(cls, good_rids: Iterable[str], bad_rids: Iterable[str]) -> Verdict:
        return cls(frozenset(unmark(good_rids)), frozenset(unmark(bad_rids)))

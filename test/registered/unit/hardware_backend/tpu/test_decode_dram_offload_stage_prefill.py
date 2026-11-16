"""Prefill side of the offload under pipeline parallelism
(stage_agreement.py, stage_prefill_workers.py): the stages vote through the
pipeline's own ready/failed request lists, and every stage takes each step of
a push or a recall in the same micro-batch slot, whatever the timing of its
own transfers.

The ring is simulated as scheduler_pp_mixin runs it: in a slot each stage
intersects its ready list with the previous stage's and unites the failed
lists, and the last stage's lists are applied by every stage a loop later."""

import importlib.util
import sys
import types
import unittest
from collections import deque
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # run on its own, without the engine's dependencies

    def register_cpu_ci(**kwargs):
        pass


register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# The modules are transport- and torch-free; load them by path so the test also
# runs where the engine's dependencies are not installed.
_DIR = (
    Path(__file__).resolve().parents[5]
    / "sglang_tpu/disaggregation/decode_dram_offload"
)
_PACKAGE = "sglang_tpu.disaggregation.decode_dram_offload"


def _load(name):
    full = f"{_PACKAGE}.{name}"
    if full in sys.modules:
        return sys.modules[full]
    for depth in range(1, 4):
        parent = ".".join(_PACKAGE.split(".")[:depth])
        sys.modules.setdefault(parent, types.ModuleType(parent))
    spec = importlib.util.spec_from_file_location(full, _DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


try:
    from sglang_tpu.disaggregation.decode_dram_offload import (
        stage_agreement as agreement,
    )
    from sglang_tpu.disaggregation.decode_dram_offload import (
        stage_prefill_workers as workers,
    )
except ImportError:
    agreement = _load("stage_agreement")
    workers = _load("stage_prefill_workers")


class _Ops:
    """One stage's side effects, recorded with the slot they happened in."""

    def __init__(self, ring, stage):
        self.ring, self.stage = ring, stage
        self.log = []
        # Local facts the test sets.
        self.pick_result = None
        self.begin_ok = True
        self.outcomes = {}
        self.offers = {}
        self.lacking = set()
        self.start_ok = True
        self.read_outcomes = {}

    def _did(self, *what):
        self.log.append((self.ring.slot, *what))

    # push
    def pick(self):
        return self.pick_result

    def still_pushable(self, leaf):
        return leaf not in self.ring.evicted

    def begin(self, leaf, seq):
        self._did("begin", leaf, seq)
        return self.begin_ok

    def outcome(self, key):
        return (
            self.outcomes if isinstance(key, int) and key < 1000 else self.read_outcomes
        ).get(key)

    def finish(self, seq, stored):
        self._did("finish", seq, stored)

    # recall
    def offer(self, room):
        return self.offers.get(room, 0)

    def lacks(self, room, pages):
        self._did("lacks", room, pages)
        return room in self.lacking

    def start(self, room, pages):
        self._did("start", room, pages)
        return self.start_ok

    def commit(self, room):
        self._did("commit", room)

    def abandon(self, room):
        self._did("abandon", room)

    def resolve(self, room):
        self._did("resolve", room)


class _Ring:
    def __init__(self, stages=4, lag=3, max_inflight_pages=100):
        self.slot = 0
        self.evicted = set()
        self.rooms = [1000]
        self.ops = [_Ops(self, s) for s in range(stages)]
        self.stages = [
            workers.StagePrefillWorkers(
                workers.StagePushCore(ops),
                workers.StageRecallCore(ops, max_inflight_pages=max_inflight_pages),
            )
            for ops in self.ops
        ]
        self._in_flight = deque([None] * lag)

    def step(self, n=1):
        for _ in range(n):
            good = bad = None
            for stage in self.stages:
                votes = stage.cast(self.rooms)
                # A real request id rides along and must survive.
                mine_good = ["req-a", *agreement.mark(votes.all_of)]
                mine_bad = agreement.mark(votes.any_of)
                good = mine_good if good is None else list(set(good) & set(mine_good))
                bad = mine_bad if bad is None else list(set(bad) | set(mine_bad))
            assert "req-a" in good
            self._in_flight.append((good, bad))
            arrived = self._in_flight.popleft()
            if arrived is not None:
                verdict = agreement.Verdict.of(*arrived)
                for stage in self.stages:
                    stage.apply(verdict)
            self.slot += 1

    def logs(self, *kinds):
        return [[e for e in ops.log if e[1] in kinds] for ops in self.ops]

    def same_on_every_stage(self, test, *kinds):
        logs = self.logs(*kinds)
        for log in logs[1:]:
            test.assertEqual(log, logs[0])
        return logs[0]


class TestMarks(unittest.TestCase):
    def test_keys_ride_among_request_ids(self):
        rids = ["req-1", *agreement.mark({"b", "a"}), "req-2"]
        self.assertEqual(agreement.unmark(rids), {"a", "b"})
        self.assertEqual(len(agreement.marked(rids)), 2)
        self.assertEqual(agreement.unmark(None), set())
        self.assertEqual(agreement.unmark(["req-1", 7]), set())
        verdict = agreement.Verdict.of(rids, ["req-3", *agreement.mark({"c"})])
        self.assertEqual((verdict.all_of, verdict.any_of), ({"a", "b"}, {"c"}))


class TestPush(unittest.TestCase):
    def _propose(self, ring, key="k1", leaf="leaf1", stages=None):
        for ops in ring.ops if stages is None else [ring.ops[s] for s in stages]:
            ops.pick_result = (key, leaf)

    def _started(self, ring):
        # A stage that began a push no longer picks its leaf.
        for ops in ring.ops:
            ops.pick_result = None

    def test_all_stages_begin_the_same_push_in_the_same_slot_once(self):
        ring = _Ring()
        self._propose(ring)
        ring.step(12)
        begun = ring.same_on_every_stage(self, "begin")
        self.assertEqual([e[1:] for e in begun], [("begin", "leaf1", 1)])
        self.assertTrue(all(s.push.has_pushes_in_flight() for s in ring.stages))

    def test_nothing_begins_until_every_stage_proposes_the_same_branch(self):
        ring = _Ring()
        self._propose(ring, stages=(0, 1, 2))
        ring.ops[3].pick_result = ("other", "leaf9")
        ring.step(10)
        self.assertEqual(ring.logs("begin"), [[], [], [], []])
        self._propose(ring)
        ring.step(5)
        self.assertEqual(len(ring.same_on_every_stage(self, "begin")), 1)

    def test_a_leaf_evicted_before_the_verdict_is_not_pushed(self):
        ring = _Ring()
        self._propose(ring)
        ring.step(1)
        ring.evicted.add("leaf1")
        self._started(ring)
        ring.step(10)
        self.assertEqual(ring.logs("begin"), [[], [], [], []])
        self.assertEqual([s.push.stats["stale"] for s in ring.stages], [1] * 4)

    def test_stored_only_when_every_stage_stored_its_share(self):
        for failing in (None, 2):
            with self.subTest(failing=failing):
                ring = _Ring()
                self._propose(ring)
                ring.step(4)
                self._started(ring)
                # Outcomes arrive at different times.
                for stage, ops in enumerate(ring.ops):
                    ops.outcomes[1] = stage != failing
                    ring.step(2)
                    if stage < 3:
                        self.assertEqual(ring.logs("finish"), [[], [], [], []])
                ring.step(6)
                finished = ring.same_on_every_stage(self, "finish")
                self.assertEqual(
                    [e[1:] for e in finished], [("finish", 1, failing is None)]
                )
                self.assertFalse(
                    any(s.push.has_pushes_in_flight() for s in ring.stages)
                )

    def test_a_stage_that_cannot_send_its_share_declines_for_all(self):
        ring = _Ring()
        ring.ops[1].begin_ok = False
        self._propose(ring)
        ring.step(4)
        self._started(ring)
        for ops in ring.ops:
            ops.outcomes[1] = True
        ring.step(6)
        finished = ring.same_on_every_stage(self, "finish")
        self.assertEqual([e[1:] for e in finished], [("finish", 1, False)])

    def test_no_more_pushes_at_once_than_the_bound(self):
        ring = _Ring()
        for n in range(workers.MAX_INFLIGHT + 2):
            self._propose(ring, key=f"k{n}", leaf=f"leaf{n}")
            ring.step(5)
        begun = ring.same_on_every_stage(self, "begin")
        self.assertEqual(len(begun), workers.MAX_INFLIGHT)
        self.assertEqual([e[3] for e in begun], [1, 2, 3, 4])


class TestRecall(unittest.TestCase):
    ROOM = 1000

    def _offer(self, ring, pages=5, stages=None):
        for ops in ring.ops if stages is None else [ring.ops[s] for s in stages]:
            ops.offers[self.ROOM] = pages

    def test_every_stage_pulls_when_any_stage_lacks_the_prefix(self):
        ring = _Ring()
        ring.ops[2].lacking.add(self.ROOM)
        # The offer reaches the stages at different times.
        for stage in range(4):
            self._offer(ring, stages=(stage,))
            ring.step(2)
        ring.step(8)
        self.assertEqual(
            [e[1:] for e in ring.same_on_every_stage(self, "lacks", "start")],
            [("lacks", self.ROOM, 5), ("start", self.ROOM, 5)],
        )
        self.assertEqual(ring.logs("commit", "resolve"), [[], [], [], []])
        # Reads settle one stage at a time; the insert waits for the last.
        for stage, ops in enumerate(ring.ops):
            ops.read_outcomes[self.ROOM] = True
            ring.step(4)
            if stage < 3:
                self.assertEqual(ring.logs("commit"), [[], [], [], []])
        done = ring.same_on_every_stage(self, "commit", "abandon", "resolve")
        self.assertEqual([e[1] for e in done], ["commit", "resolve"])
        self.assertEqual([s.recall.stats["pulled_pages"] for s in ring.stages], [5] * 4)

    def test_no_pull_when_every_stage_holds_the_prefix(self):
        ring = _Ring()
        self._offer(ring)
        ring.step(12)
        done = ring.same_on_every_stage(self, "start", "commit", "abandon", "resolve")
        self.assertEqual([e[1] for e in done], ["resolve"])
        self.assertEqual([s.recall.stats["held"] for s in ring.stages], [1] * 4)

    def test_a_read_that_fails_on_one_stage_is_abandoned_on_all(self):
        for failure in ("read", "start"):
            with self.subTest(failure=failure):
                ring = _Ring()
                for ops in ring.ops:
                    ops.lacking.add(self.ROOM)
                    ops.read_outcomes[self.ROOM] = True
                if failure == "read":
                    ring.ops[3].read_outcomes[self.ROOM] = False
                else:
                    ring.ops[0].start_ok = False
                self._offer(ring)
                ring.step(16)
                done = ring.same_on_every_stage(self, "commit", "abandon", "resolve")
                self.assertEqual([e[1] for e in done], ["abandon", "resolve"])
                self.assertFalse(any(s.busy() for s in ring.stages))

    def test_stages_offered_different_lengths_do_not_pull(self):
        ring = _Ring()
        self._offer(ring, pages=5, stages=(0, 1, 2))
        self._offer(ring, pages=4, stages=(3,))
        ring.step(12)
        self.assertEqual(ring.logs("lacks", "start"), [[], [], [], []])

    def test_a_pull_over_the_in_flight_budget_is_skipped(self):
        ring = _Ring(max_inflight_pages=6)
        ring.rooms = [1000, 1001]
        for ops in ring.ops:
            ops.lacking.update((1000, 1001))
            ops.offers.update({1000: 5, 1001: 5})
        ring.step(12)
        started = ring.same_on_every_stage(self, "start")
        self.assertEqual([e[2] for e in started], [1000])
        resolved = ring.same_on_every_stage(self, "resolve")
        self.assertEqual([e[2] for e in resolved], [1001])
        self.assertEqual([s.recall.stats["over_budget"] for s in ring.stages], [1] * 4)


class TestLagAndStageCount(unittest.TestCase):
    def test_any_ring_depth_gives_the_same_steps(self):
        for stages, lag in ((1, 1), (2, 1), (8, 9), (8, 16)):
            with self.subTest(stages=stages, lag=lag):
                ring = _Ring(stages=stages, lag=lag)
                for ops in ring.ops:
                    ops.pick_result = ("k", "leaf")
                    ops.lacking.add(1000)
                    ops.offers[1000] = 3
                    ops.read_outcomes[1000] = True
                    ops.outcomes[1] = True
                ring.step(6 * lag + 6)
                log = ring.same_on_every_stage(
                    self, "begin", "finish", "start", "commit", "resolve"
                )
                self.assertEqual(
                    sorted(e[1] for e in log if e[1] != "begin" or e[3] == 1),
                    ["begin", "commit", "finish", "resolve", "start"],
                )


if __name__ == "__main__":
    unittest.main()

"""Decode side of the offload for a pipeline-parallel prefill
(sglang_tpu.disaggregation.decode_dram_offload.decode_stage_branches): every
decode rank lands a pushed branch at the same page ids whichever stages it
hosts, a failure on one rank is a failure on all, and a stored prefix is
offered back and stays pinned until no stage can still read it.

The ranks run as threads that meet in every collective, so a rank that issues
one collective more or less than the others fails the test."""

import importlib.util
import sys
import threading
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

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
        decode_stage_branches as branches,
    )
    from sglang_tpu.disaggregation.decode_dram_offload import (
        stage_layout_store as store_module,
    )
except ImportError:
    store_module = _load("stage_layout_store")
    branches = _load("decode_stage_branches")

PAGE = 4
CADENCE = branches._CADENCE


def _tokens(*pages):
    return [p for p in pages for _ in range(PAGE)]


class _Hub:
    """Where the rank threads meet: collective ``n`` of every rank sees the
    values all ranks passed to their collective ``n``."""

    def __init__(self, ranks):
        self.ranks = ranks
        self._barrier = threading.Barrier(ranks, timeout=10)
        self._values = [None] * ranks

    def exchange(self, rank, value):
        self._values[rank] = value
        self._barrier.wait()
        values = list(self._values)
        self._barrier.wait()
        return values


class _Lockstep:
    """rank_lockstep.RankLockstep over the hub."""

    def __init__(self, hub, rank):
        self._hub, self._rank = hub, rank

    def all(self, flag):
        return all(self._hub.exchange(self._rank, bool(flag)))

    def max(self, value):
        return max(self._hub.exchange(self._rank, int(value)))

    def min_max(self, row):
        rows = self._hub.exchange(self._rank, list(row))
        assert len({len(r) for r in rows}) == 1, (
            "ranks reduced rows of different length"
        )
        return [min(c) for c in zip(*rows)], [max(c) for c in zip(*rows)]

    def common_to_all_ranks(self, keys):
        rows = self._hub.exchange(self._rank, sorted(keys)[:8])
        return sorted(set(rows[0]).intersection(*map(set, rows)))


class _Channel:
    def __init__(self):
        self.pushes = {}
        self.reads = {}  # read id -> None (pending) | True | False
        self.started = []
        self.refuse_start = False
        self.acks = []
        self.offered = {}
        self.released = []
        self.pending_rooms = set()

    def received_stage_pushes(self):
        return dict(self.pushes)

    def take_stage_push(self, key):
        return self.pushes.pop(key, None)

    def start_stage_read(self, read_id, uuid, address, remote, local, *, stage, swa):
        if self.refuse_start:
            return False
        self.started.append((read_id, uuid, address, remote, local, stage, swa))
        self.reads[read_id] = None
        return True

    def drain_host_reads(self):
        pass

    def poll_host_reads(self, read_ids):
        results = [self.reads.get(r, False) for r in read_ids]
        return None if None in results else all(results)

    def forget_host_reads(self, read_ids):
        for read_id in read_ids:
            self.reads.pop(read_id, None)

    def send_victim_ack(self, *, endpoint, seq, stored):
        self.acks.append((endpoint, seq, stored))

    def offer_stage_prefix(self, room, pages, window):
        self.offered[room] = (list(pages), list(window))

    def release_prefix_offer(self, room):
        self.released.append(room)

    def host_prefix_pending(self, room):
        return room in self.pending_rooms


def _layout(stage, stages, heads=16, window=False):
    return store_module.StagePoolLayout(
        stage, stages, (("bfloat16", (PAGE, heads, 8)),), PAGE * heads * 16, window
    )


class _Decode:
    """A tensor-parallel decode of ``ranks`` ranks behind a pipeline prefill of
    ``stages`` stages."""

    def __init__(
        self, *, ranks, stages, budget_pages=16, window_pages=0, broken_rank=None
    ):
        self.ranks, self.stages, self.window_pages = ranks, stages, window_pages
        hub = _Hub(ranks)
        self.channels = [_Channel() for _ in range(ranks)]
        self.built = [[] for _ in range(ranks)]
        self.owned = set()
        self.now = 0.0
        self.receivers, self.offerers = [], []
        per_rank = max(
            len(store_module.hosted_stages(r, stages, ranks)) for r in range(ranks)
        )
        page_bytes = _layout(0, stages).page_bytes
        budget = budget_pages * per_rank * page_bytes
        if window_pages:
            # size_store buys whole branches: full pages plus their window.
            budget += budget_pages * per_rank * page_bytes
        for rank in range(ranks):

            def build_pool(layout, pages, rank=rank):
                if rank == broken_rank:
                    raise RuntimeError("no host memory")
                self.built[rank].append((layout.stage, layout.window, pages))
                return object()

            lockstep = _Lockstep(hub, rank)
            pools = branches.StagePoolSet(
                rank=rank,
                decode_ranks=ranks,
                budget_bytes=budget,
                build_pool=build_pool,
            )
            receiver = branches.DecodeStageBranchReceiver(
                channel=self.channels[rank],
                lockstep=lockstep,
                pools=pools,
                page_size=PAGE,
                clock=lambda: self.now,
            )
            self.receivers.append(receiver)
            self.offerers.append(
                branches.DecodeStageBranchOfferer(
                    channel=self.channels[rank],
                    lockstep=lockstep,
                    receiver=receiver,
                    owned_rooms=lambda: set(self.owned),
                )
            )

    def each(self, call):
        """Run ``call(rank)`` on every rank at once, as one scheduler step."""
        errors = []

        def run(rank):
            try:
                call(rank)
            except BaseException as exc:  # noqa: BLE001 - reported below
                errors.append((rank, exc))

        threads = [threading.Thread(target=run, args=(r,)) for r in range(self.ranks)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise AssertionError(f"rank {errors[0][0]} failed: {errors[0][1]!r}")

    def round(self):
        """Decode iterations up to the next agreement round."""
        for _ in range(CADENCE):
            self.each(lambda rank: self.receivers[rank].advance())

    def push(
        self, seq, tokens, *, prefill="p", skip=(), stages=None, heads=16, salt=None
    ):
        """Every stage pushes its share to its hosting rank; stage zero sends
        the header to every rank. ``skip``: stages whose message is lost."""
        stages = stages or self.stages
        pages = len(tokens) // PAGE
        window = min(pages, self.window_pages)
        header = SimpleNamespace(
            tokens=tokens,
            extra_key=None,
            cache_salt=salt,
            stages=stages,
            window_pages=self.window_pages,
        )
        for rank in range(self.ranks):
            entry = self.channels[rank].pushes.setdefault(
                (prefill, seq), branches.StagePush()
            )
            if 0 not in skip:
                entry.header = header
            for stage in store_module.hosted_stages(rank, stages, self.ranks):
                if stage in skip:
                    continue
                entry.parts[stage] = SimpleNamespace(
                    seq=seq,
                    stage=stage,
                    ack_endpoint=f"stage{stage}",
                    # Each stage's own host page ids of the branch.
                    blocks=[1000 * stage + i for i in range(pages)],
                    swa_blocks=[5000 * (stage + 1) + i for i in range(window)],
                    uuid=77,
                    address=f"host{stage}",
                    swa_address=f"swa{stage}",
                    layout=_layout(stage, stages, heads),
                    swa_layout=_layout(stage, stages, heads, window=True),
                )

    def settle_reads(self, ok=True, fail_rank=None):
        for rank, channel in enumerate(self.channels):
            for read_id in channel.reads:
                channel.reads[read_id] = ok and rank != fail_rank

    def land(self, seq, tokens, **kwargs):
        self.push(seq, tokens, **kwargs)
        self.round()
        self.settle_reads()
        self.round()

    def stores(self):
        return [receiver.store for receiver in self.receivers]

    def acks(self):
        """{stage endpoint: [stored, ...]} over all ranks."""
        out = {}
        for channel in self.channels:
            for endpoint, _, stored in channel.acks:
                out.setdefault(endpoint, []).append(stored)
        return out


class TestLanding(unittest.TestCase):
    def test_every_rank_lands_its_stage_at_the_same_page_ids(self):
        decode = _Decode(ranks=8, stages=8)
        decode.land(1, _tokens(1, 2, 3))
        for rank, channel in enumerate(decode.channels):
            (read,) = channel.started
            read_id, uuid, address, remote, local, stage, swa = read
            self.assertEqual((stage, swa, address), (rank, False, f"host{rank}"))
            self.assertEqual(remote, [1000 * rank + i for i in range(3)])
            self.assertEqual(local, [0, 1, 2])
            self.assertEqual(read_id, f"victim:p:1:s{rank}")
            self.assertEqual(decode.built[rank], [(rank, False, 16)])
        self.assertEqual(decode.acks(), {f"stage{s}": [True] for s in range(8)})
        for store in decode.stores():
            self.assertEqual(
                store.match((None, None), _tokens(1, 2, 3, 4)).pages, [0, 1, 2]
            )
            self.assertEqual(store.capacity_pages, 16)

    def test_a_rank_hosts_several_stages_or_none(self):
        for ranks, stages in ((4, 8), (2, 8), (4, 2), (1, 8)):
            with self.subTest(ranks=ranks, stages=stages):
                decode = _Decode(ranks=ranks, stages=stages)
                decode.land(1, _tokens(1, 2))
                read_stages = sorted(
                    read[5] for channel in decode.channels for read in channel.started
                )
                self.assertEqual(read_stages, list(range(stages)))
                self.assertEqual(
                    decode.acks(), {f"stage{s}": [True] for s in range(stages)}
                )
                sizes = {store.capacity_pages for store in decode.stores()}
                self.assertEqual(sizes, {16})
                for store in decode.stores():
                    self.assertEqual(store.used_pages, 2)

    def test_a_push_waits_for_every_part_and_is_dropped_without_them(self):
        decode = _Decode(ranks=4, stages=4)
        decode.push(1, _tokens(1), skip=(2,))
        decode.round()
        self.assertFalse(any(channel.started for channel in decode.channels))
        # The lost part arrives: the push lands.
        decode.push(1, _tokens(1))
        decode.round()
        self.assertTrue(all(channel.started for channel in decode.channels))
        decode.settle_reads()
        decode.round()
        self.assertEqual(decode.acks()["stage2"], [True])
        # A push whose header never comes is forgotten, by the clock: the
        # stages' parts arrive seconds apart on a large model, and an idle
        # decode runs many agreement rounds a second.
        decode.push(2, _tokens(7), skip=(0,))
        for _ in range(40):
            decode.round()
        self.assertEqual([len(c.pushes) for c in decode.channels], [1, 1, 1, 1])
        decode.now += branches._MAX_WAIT_S + 1
        decode.round()
        self.assertEqual([len(c.pushes) for c in decode.channels], [0, 0, 0, 0])
        self.assertEqual(decode.receivers[1].stats["dropped"], 1)
        self.assertNotIn(False, [s for a in decode.acks().values() for s in a])

    def test_a_read_that_fails_on_one_rank_fails_on_all(self):
        decode = _Decode(ranks=4, stages=4)
        decode.push(1, _tokens(1, 2))
        decode.round()
        decode.settle_reads(fail_rank=3)
        decode.round()
        self.assertEqual(decode.acks(), {f"stage{s}": [False] for s in range(4)})
        for store, receiver in zip(decode.stores(), decode.receivers):
            self.assertEqual(store.used_pages, 0)
            self.assertFalse(receiver.has_host_landings())
            self.assertEqual(receiver.stats["failed"], 1)

    def test_a_landing_waits_for_the_slowest_rank(self):
        decode = _Decode(ranks=2, stages=2)
        decode.push(1, _tokens(1))
        decode.round()
        for read_id in decode.channels[0].reads:
            decode.channels[0].reads[read_id] = True
        decode.round()
        self.assertEqual(decode.acks(), {})
        self.assertIsNone(decode.stores()[0].match((None, None), _tokens(1)))
        decode.settle_reads()
        decode.round()
        self.assertEqual(decode.acks(), {"stage0": [True], "stage1": [True]})

    def test_a_read_that_cannot_start_on_one_rank_keeps_the_pages_until_settled(self):
        decode = _Decode(ranks=2, stages=2)
        decode.channels[1].refuse_start = True
        decode.push(1, _tokens(1, 2))
        decode.round()
        # Rank 0's read runs: no rank gives its pages back yet.
        self.assertEqual([s.used_pages for s in decode.stores()], [2, 2])
        decode.settle_reads()
        decode.round()
        self.assertEqual(decode.acks(), {"stage0": [False], "stage1": [False]})
        self.assertEqual([s.used_pages for s in decode.stores()], [0, 0])

    def test_a_longer_push_reads_only_the_new_pages(self):
        decode = _Decode(ranks=2, stages=2)
        decode.land(1, _tokens(1, 2))
        decode.land(2, _tokens(1, 2, 3, 4))
        _, _, _, remote, local, stage, _ = decode.channels[1].started[-1]
        self.assertEqual((stage, remote, local), (1, [1002, 1003], [2, 3]))
        # Pushed again: held, answered at once, nothing read.
        reads = len(decode.channels[0].started)
        decode.push(3, _tokens(1, 2, 3))
        decode.round()
        self.assertEqual(len(decode.channels[0].started), reads)
        self.assertEqual(decode.acks()["stage0"], [True, True, True])
        self.assertEqual(decode.receivers[0].stats["held"], 1)

    def test_pages_in_another_layout_are_declined(self):
        decode = _Decode(ranks=2, stages=2)
        decode.land(1, _tokens(1))
        decode.push(2, _tokens(2), prefill="other", heads=8)
        decode.round()
        self.assertEqual(decode.acks()["stage1"], [True, False])
        self.assertEqual([r.stats["layout"] for r in decode.receivers], [1, 1])
        self.assertEqual([s.used_pages for s in decode.stores()], [1, 1])

    def test_a_rank_that_cannot_build_its_pools_declines_for_all(self):
        decode = _Decode(ranks=4, stages=4, broken_rank=2)
        decode.push(1, _tokens(1))
        decode.round()
        decode.push(2, _tokens(2))
        decode.round()
        self.assertEqual(decode.acks(), {f"stage{s}": [False, False] for s in range(4)})
        self.assertEqual(decode.stores(), [None] * 4)
        self.assertFalse(any(channel.started for channel in decode.channels))

    def test_no_more_landings_at_once_than_the_bound(self):
        decode = _Decode(ranks=2, stages=2, budget_pages=64)
        for seq in range(branches.MAX_INFLIGHT + 2):
            decode.push(seq, _tokens(10 + seq))
        decode.round()
        self.assertEqual(len(decode.receivers[0]._landings), branches.MAX_INFLIGHT)
        self.assertEqual(decode.acks()["stage0"], [False, False])

    def test_the_least_recently_used_branch_makes_room(self):
        decode = _Decode(ranks=2, stages=2, budget_pages=4)
        decode.land(1, _tokens(1, 2))
        decode.land(2, _tokens(3, 4))
        decode.land(3, _tokens(5, 6))
        for store in decode.stores():
            self.assertIsNone(store.match((None, None), _tokens(1, 2)))
            self.assertEqual(store.match((None, None), _tokens(5, 6)).pages, [0, 1])
        self.assertEqual(decode.receivers[0].counters()["evicted_pages"], 2)


class TestSlidingWindow(unittest.TestCase):
    def test_the_window_lands_in_the_window_pool(self):
        decode = _Decode(ranks=2, stages=2, window_pages=2)
        decode.land(1, _tokens(1, 2, 3))
        full, window = decode.channels[1].started
        self.assertEqual(
            (full[3], full[4], full[6]), ([1000, 1001, 1002], [0, 1, 2], False)
        )
        self.assertEqual(
            (window[2], window[3], window[4], window[6]),
            ("swa1", [10000, 10001], [0, 1], True),
        )
        self.assertEqual(window[0], "victim:p:1:s1:swa")
        self.assertEqual(
            sorted(decode.built[1]),
            [
                (1, False, decode.stores()[1].capacity_pages),
                (1, True, decode.stores()[1]._windows.count),
            ],
        )
        # Whole branches: four full pages per page of window.
        self.assertEqual(
            decode.stores()[1].capacity_pages, 4 * decode.stores()[1]._windows.count
        )
        match = decode.stores()[0].match((None, None), _tokens(1, 2, 3, 4))
        self.assertEqual((match.pages, match.window), ([0, 1, 2], [0, 1]))

    def test_a_branch_shorter_than_a_window_brings_its_own_pages_of_it(self):
        decode = _Decode(ranks=2, stages=2, window_pages=3, budget_pages=64)
        decode.land(1, _tokens(1))
        _, window = decode.channels[0].started
        self.assertEqual((window[3], window[4]), ([5000], [0]))
        match = decode.stores()[1].match((None, None), _tokens(1, 2))
        self.assertEqual((match.pages, match.window), ([0], [0]))


class TestOffers(unittest.TestCase):
    def _offer(self, decode, room, tokens):
        decode.owned.add(room)
        decode.each(
            lambda rank: decode.offerers[rank].offer_stage_prefix(
                room=room, tokens=tokens, scope=(None, None)
            )
        )

    def test_every_rank_offers_the_same_pages_and_pins_them(self):
        decode = _Decode(ranks=4, stages=8, budget_pages=4)
        decode.land(1, _tokens(1, 2))
        self._offer(decode, 42, _tokens(1, 2, 3))
        self.assertEqual(
            [channel.offered for channel in decode.channels], [{42: ([0, 1], [])}] * 4
        )
        # The offered branch is the oldest, and it stays through an eviction.
        decode.land(2, _tokens(3, 4))
        decode.land(3, _tokens(5, 6))
        for store in decode.stores():
            self.assertEqual(store.match((None, None), _tokens(1, 2)).pages, [0, 1])
            self.assertIsNone(store.match((None, None), _tokens(3, 4)))

    def test_nothing_stored_nothing_offered(self):
        decode = _Decode(ranks=2, stages=2)
        self._offer(decode, 1, _tokens(1))
        decode.land(1, _tokens(1, 2))
        self._offer(decode, 2, _tokens(9, 9))
        self.assertEqual([channel.offered for channel in decode.channels], [{}, {}])

    def test_the_pin_goes_once_the_request_left_and_no_stage_reads(self):
        decode = _Decode(ranks=2, stages=2)
        decode.land(1, _tokens(1, 2))
        self._offer(decode, 42, _tokens(1, 2, 3))
        settle = lambda: decode.each(lambda rank: decode.offerers[rank].settle_offers())
        settle()
        self.assertTrue(all(store.has_pins() for store in decode.stores()))
        self.assertEqual(decode.channels[0].released, [])
        # The request leaves the queues while stage 1 still reads from rank 1.
        decode.owned.clear()
        decode.channels[1].pending_rooms.add(42)
        settle()
        self.assertEqual(
            [channel.released for channel in decode.channels], [[42], [42]]
        )
        self.assertTrue(all(store.has_pins() for store in decode.stores()))
        with self.assertRaises(RuntimeError):
            decode.offerers[0].release_all()
        decode.channels[1].pending_rooms.clear()
        settle()
        self.assertFalse(any(store.has_pins() for store in decode.stores()))
        decode.offerers[0].release_all()
        self.assertEqual(decode.offerers[0].stats, {"offers": 1, "offered_pages": 2})


if __name__ == "__main__":
    unittest.main()

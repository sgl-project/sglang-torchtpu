"""Decode-side index of the branches kept in prefill-stage layout
(sglang_tpu.disaggregation.decode_dram_offload.stage_layout_store): what a
decode holds of a pipeline-parallel prefill's pushed branches, what it offers
back, what it evicts, and that ranks applying the same calls hold the same
page ids."""

import importlib.util
import random
import sys
import unittest
from pathlib import Path

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # run on its own, without the engine's dependencies

    def register_cpu_ci(**kwargs):
        pass


register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# The module is transport- and torch-free; load it by path so the test also
# runs where the engine's dependencies are not installed.
_PATH = (
    Path(__file__).resolve().parents[5]
    / "sglang_tpu/disaggregation/decode_dram_offload/stage_layout_store.py"
)
_spec = importlib.util.spec_from_file_location("stage_layout_store", _PATH)
store_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = store_module
_spec.loader.exec_module(store_module)

StageLayoutStore = store_module.StageLayoutStore
StagePoolLayout = store_module.StagePoolLayout

PAGE = 4


def _tokens(*pages):
    """Whole pages; page ``p`` is PAGE tokens of value ``p``."""
    return [p for p in pages for _ in range(PAGE)]


def _land(store, scope, tokens, **kwargs):
    landing = store.reserve(scope, tokens, **kwargs)
    if landing is not None:
        store.commit(landing)
    return landing


def _check(test, store, open_landings=()):
    """The index's books balance: every node holds one page, leaves are the
    childless nodes, and no page id is held twice. The windows of
    ``open_landings`` are reserved and not yet on a node."""
    nodes, stack = [], [n for level in store._roots.values() for n in level.values()]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children.values())
    ids = [n.page_id for n in nodes]
    test.assertEqual(len(ids), len(set(ids)))
    test.assertEqual(len(ids), store.used_pages)
    test.assertEqual(
        {id(n) for n in nodes if not n.children}, {id(n) for n in store._leaves}
    )
    window_ids = [i for n in nodes for i in (n.window or ())]
    window_ids += [i for landing in open_landings for i in (landing.window or ())]
    test.assertEqual(len(window_ids), len(set(window_ids)))
    if store.window_pages:
        test.assertEqual(len(window_ids), store.used_windows * store.window_pages)


class TestPlacementAndSizing(unittest.TestCase):
    def test_stages_spread_over_the_decode_ranks(self):
        """Eight stages on eight ranks: one each. On four ranks: two each. On
        sixteen: the first eight ranks hold one, the rest none."""
        for ranks in (1, 4, 8, 16):
            held = [store_module.hosted_stages(r, 8, ranks) for r in range(ranks)]
            self.assertEqual(sorted(s for h in held for s in h), list(range(8)))
            self.assertLessEqual(max(map(len, held)) - min(map(len, held)), 1)
        self.assertEqual(store_module.hosted_stages(3, 8, 4), [3, 7])

    def test_budget_buys_pages_and_windows(self):
        self.assertEqual(
            store_module.size_store(budget_bytes=1000, page_bytes=10), (100, 0)
        )
        # A branch is one window and four full pages per window page.
        pages, windows = store_module.size_store(
            budget_bytes=1000,
            page_bytes=10,
            window_page_bytes=5,
            window_pages=4,
        )
        self.assertEqual((pages, windows), (80, 5))
        self.assertLessEqual(pages * 10 + windows * 4 * 5, 1000)
        # Large window pages no longer starve the windows (Muse Spark's
        # layout: 2 MiB full pages, 5 MiB window pages, 16 pages a window).
        pages, windows = store_module.size_store(
            budget_bytes=4096, page_bytes=2, window_page_bytes=5, window_pages=16
        )
        self.assertEqual((pages, windows), (1216, 19))
        with self.assertRaises(ValueError):
            store_module.size_store(budget_bytes=1000, page_bytes=0)

    def test_fingerprint_names_the_layout(self):
        a = StagePoolLayout(0, 8, (("bfloat16", (256, 16, 128)),), 1 << 20)
        same = StagePoolLayout(0, 8, (("bfloat16", (256, 16, 128)),), 1 << 20)
        self.assertEqual(a.fingerprint(), same.fingerprint())
        self.assertGreaterEqual(a.fingerprint(), 0)
        for other in (
            StagePoolLayout(1, 8, a.tensors, a.page_bytes),
            StagePoolLayout(0, 4, a.tensors, a.page_bytes),
            StagePoolLayout(0, 8, (("bfloat16", (256, 8, 128)),), 1 << 19),
            StagePoolLayout(0, 8, a.tensors, a.page_bytes, window=True),
        ):
            self.assertNotEqual(a.fingerprint(), other.fingerprint())


class TestLandingAndRecall(unittest.TestCase):
    def test_a_landed_branch_is_offered_only_once_committed(self):
        store = StageLayoutStore(page_size=PAGE, pages=8)
        landing = store.reserve("s", _tokens(1, 2, 3))
        self.assertEqual((landing.first_new_page, landing.new_pages), (0, [0, 1, 2]))
        self.assertIsNone(store.match("s", _tokens(1, 2, 3, 4)))
        self.assertTrue(store.has_pins())
        store.commit(landing)
        self.assertFalse(store.has_pins())
        match = store.match("s", _tokens(1, 2, 3, 4) + [9])
        self.assertEqual(match.pages, [0, 1, 2])
        self.assertEqual(match.window, [])
        _check(self, store)

    def test_a_request_gets_the_longest_prefix_it_shares(self):
        store = StageLayoutStore(page_size=PAGE, pages=8)
        _land(store, "s", _tokens(1, 2, 3))
        self.assertEqual(store.match("s", _tokens(1, 2, 7)).pages, [0, 1])
        self.assertIsNone(store.match("s", _tokens(5, 2, 3)))
        self.assertIsNone(store.match("s", _tokens(1)[:-1]))
        # Another sampling scope shares nothing.
        self.assertIsNone(store.match("other", _tokens(1, 2, 3)))

    def test_a_longer_push_reads_only_the_pages_not_held(self):
        store = StageLayoutStore(page_size=PAGE, pages=8)
        _land(store, "s", _tokens(1, 2))
        longer = store.reserve("s", _tokens(1, 2, 3, 4))
        self.assertEqual((longer.first_new_page, longer.new_pages), (2, [2, 3]))
        store.commit(longer)
        fork = _land(store, "s", _tokens(1, 5))
        self.assertEqual((fork.first_new_page, fork.new_pages), (1, [4]))
        self.assertEqual(store.used_pages, 5)
        self.assertEqual(store.stats["reused_pages"], 3)
        # Nothing new: no landing, no page.
        self.assertIsNone(store.reserve("s", _tokens(1, 2, 3)))
        self.assertEqual(store.used_pages, 5)
        _check(self, store)

    def test_a_push_that_meets_a_landing_in_flight_is_declined(self):
        store = StageLayoutStore(page_size=PAGE, pages=8)
        first = store.reserve("s", _tokens(1, 2))
        self.assertIsNone(store.reserve("s", _tokens(1, 2, 3)))
        self.assertEqual(store.stats["busy"], 1)
        store.commit(first)
        self.assertIsNotNone(store.reserve("s", _tokens(1, 2, 3)))

    def test_a_failed_landing_gives_its_pages_back(self):
        store = StageLayoutStore(page_size=PAGE, pages=4, window_pages=1, windows=2)
        _land(store, "s", _tokens(1), with_window=True)
        landing = store.reserve("s", _tokens(1, 2, 3), with_window=True)
        self.assertEqual(landing.new_pages, [1, 2])
        store.abort(landing)
        self.assertEqual((store.used_pages, store.used_windows), (1, 1))
        self.assertFalse(store.has_pins())
        self.assertEqual(store.match("s", _tokens(1, 2, 3)).pages, [0])
        # The ids come back lowest first, as on a rank that never failed.
        again = store.reserve("s", _tokens(1, 2), with_window=True)
        self.assertEqual(again.new_pages, [1])
        _check(self, store, [again])

    def test_partial_pages_are_refused(self):
        store = StageLayoutStore(page_size=PAGE, pages=4)
        with self.assertRaises(ValueError):
            store.reserve("s", _tokens(1) + [2])
        with self.assertRaises(ValueError):
            store.reserve("s", [])
        with self.assertRaises(ValueError):
            store.reserve("s", _tokens(1), with_window=True)


class TestSlidingWindows(unittest.TestCase):
    def _store(self, **kwargs):
        sizes = dict(page_size=PAGE, pages=8, window_pages=2, windows=3)
        sizes.update(kwargs)
        return StageLayoutStore(**sizes)

    def test_a_branch_resumes_only_where_a_window_ends(self):
        store = self._store()
        landing = _land(store, "s", _tokens(1, 2, 3), with_window=True)
        self.assertEqual(landing.window, [0, 1])
        # A request that shares two pages has no window to resume from.
        self.assertIsNone(store.match("s", _tokens(1, 2, 9)))
        match = store.match("s", _tokens(1, 2, 3, 4))
        self.assertEqual((match.pages, match.window), ([0, 1, 2], [0, 1]))

    def test_successive_lengths_each_keep_their_window(self):
        store = self._store()
        _land(store, "s", _tokens(1, 2), with_window=True)
        longer = _land(store, "s", _tokens(1, 2, 3), with_window=True)
        self.assertEqual((longer.new_pages, longer.window), ([2], [2, 3]))
        self.assertEqual(store.match("s", _tokens(1, 2, 9)).window, [0, 1])
        self.assertEqual(store.match("s", _tokens(1, 2, 3)).window, [2, 3])
        # Pushed again at a held length: nothing to read.
        self.assertIsNone(store.reserve("s", _tokens(1, 2), with_window=True))
        _check(self, store)

    def test_a_held_path_without_a_window_takes_one(self):
        store = self._store()
        _land(store, "s", _tokens(1, 2, 3), with_window=True)
        landing = store.reserve("s", _tokens(1, 2), with_window=True)
        self.assertEqual((landing.new_pages, landing.first_new_page), ([], 2))
        self.assertEqual(landing.window, [2, 3])
        store.commit(landing)
        self.assertEqual(store.match("s", _tokens(1, 2, 8)).pages, [0, 1])

    def test_eviction_leaves_no_path_that_cannot_be_resumed(self):
        store = self._store(pages=4, windows=2)
        _land(store, "s", _tokens(1, 2, 3), with_window=True)
        # Needs three pages: the old branch goes whole, not just its last page.
        landing = _land(store, "s", _tokens(5, 6, 7), with_window=True)
        self.assertIsNotNone(landing)
        self.assertEqual(store.used_pages, 3)
        self.assertEqual(store.stats["evicted_pages"], 3)
        self.assertEqual(store.stats["evicted_windows"], 1)
        self.assertIsNone(store.match("s", _tokens(1, 2, 3)))
        _check(self, store)

    def test_eviction_stops_at_a_shorter_length_with_its_own_window(self):
        store = self._store(pages=4, windows=3)
        _land(store, "s", _tokens(1, 2), with_window=True)
        _land(store, "s", _tokens(1, 2, 3, 4), with_window=True)
        self.assertIsNotNone(_land(store, "s", _tokens(7, 8), with_window=True))
        self.assertEqual(store.match("s", _tokens(1, 2, 3, 4)).pages, [0, 1])
        _check(self, store)

    def test_windows_alone_can_run_out(self):
        store = self._store(pages=8, windows=1)
        _land(store, "s", _tokens(1), with_window=True)
        self.assertIsNotNone(_land(store, "s", _tokens(2), with_window=True))
        self.assertIsNone(store.match("s", _tokens(1)))
        self.assertEqual(store.used_windows, 1)


class TestEviction(unittest.TestCase):
    def test_the_least_recently_used_branch_goes_first(self):
        store = StageLayoutStore(page_size=PAGE, pages=4)
        _land(store, "s", _tokens(1, 2))
        _land(store, "s", _tokens(3, 4))
        store.match("s", _tokens(1, 2))
        _land(store, "s", _tokens(5, 6))
        self.assertIsNone(store.match("s", _tokens(3, 4)))
        self.assertEqual(store.match("s", _tokens(1, 2)).num_pages, 2)
        self.assertEqual(store.stats["evicted_pages"], 2)
        _check(self, store)

    def test_an_offered_prefix_is_not_evicted(self):
        store = StageLayoutStore(page_size=PAGE, pages=4)
        _land(store, "s", _tokens(1, 2))
        _land(store, "s", _tokens(3, 4))
        offered = store.match("s", _tokens(1, 2))
        store.pin(offered)
        store.match("s", _tokens(3, 4))
        # Room for two: the pinned branch is older but stays.
        self.assertIsNotNone(_land(store, "s", _tokens(5, 6)))
        self.assertEqual(store.match("s", _tokens(1, 2)).pages, offered.pages)
        # Everything left is pinned or new: a push for four pages is declined.
        store.pin(store.match("s", _tokens(5, 6)))
        self.assertIsNone(store.reserve("s", _tokens(7, 8)))
        self.assertEqual(store.stats["no_room"], 1)
        store.unpin(offered)
        self.assertIsNotNone(_land(store, "s", _tokens(7, 8)))
        with self.assertRaises(RuntimeError):
            store.unpin(offered)
        _check(self, store)

    def test_the_held_prefix_of_a_push_survives_its_own_room_making(self):
        store = StageLayoutStore(page_size=PAGE, pages=3)
        _land(store, "s", _tokens(1, 2))
        _land(store, "s", _tokens(9))
        store.match("s", _tokens(9))
        landing = store.reserve("s", _tokens(1, 2, 3))
        self.assertEqual((landing.first_new_page, len(landing.new_pages)), (2, 1))
        self.assertIsNone(store.match("s", _tokens(9)))

    def test_a_branch_larger_than_the_store_is_declined_and_evicts_nothing(self):
        store = StageLayoutStore(page_size=PAGE, pages=2)
        _land(store, "s", _tokens(1, 2))
        self.assertIsNone(store.reserve("s", _tokens(3, 4, 5)))
        self.assertEqual(store.stats["evicted_pages"], 0)
        self.assertEqual(store.match("s", _tokens(1, 2)).num_pages, 2)

    def test_clear_needs_an_idle_store(self):
        store = StageLayoutStore(page_size=PAGE, pages=4)
        landing = store.reserve("s", _tokens(1))
        with self.assertRaises(RuntimeError):
            store.clear()
        store.commit(landing)
        store.clear()
        self.assertEqual(store.used_pages, 0)
        self.assertEqual(store.reserve("s", _tokens(1)).new_pages, [0])


class TestRanksHoldTheSameIndex(unittest.TestCase):
    def test_same_calls_give_same_page_ids(self):
        """Every decode rank applies the agreed landings, aborts and offers in
        the same order; the page ids must then be equal on all of them, since
        one id names a page in every stage pool."""
        rng = random.Random(7)
        ranks = [
            StageLayoutStore(page_size=PAGE, pages=24, window_pages=2, windows=5)
            for _ in range(3)
        ]
        open_landings, pinned = [], []
        for step in range(600):
            action = rng.random()
            scope = rng.choice(("a", "b"))
            tokens = _tokens(*(rng.randrange(3) for _ in range(rng.randrange(1, 6))))
            if action < 0.45:
                got = [r.reserve(scope, tokens, with_window=True) for r in ranks]
                shapes = {
                    (
                        None
                        if g is None
                        else (
                            g.first_new_page,
                            tuple(g.new_pages),
                            tuple(g.window or ()),
                        )
                    )
                    for g in got
                }
                self.assertEqual(len(shapes), 1, f"ranks diverged at step {step}")
                if got[0] is not None:
                    open_landings.append(got)
            elif action < 0.75 and open_landings:
                got = open_landings.pop(rng.randrange(len(open_landings)))
                ok = rng.random() < 0.8
                for rank, landing in zip(ranks, got):
                    (rank.commit if ok else rank.abort)(landing)
            elif action < 0.9:
                got = [r.match(scope, tokens) for r in ranks]
                shapes = {
                    None if g is None else (tuple(g.pages), tuple(g.window))
                    for g in got
                }
                self.assertEqual(len(shapes), 1, f"ranks diverged at step {step}")
                if got[0] is not None and rng.random() < 0.5:
                    for rank, match in zip(ranks, got):
                        rank.pin(match)
                    pinned.append(got)
            elif pinned:
                for rank, match in zip(ranks, pinned.pop(rng.randrange(len(pinned)))):
                    rank.unpin(match)
            for index, rank in enumerate(ranks):
                _check(self, rank, [got[index] for got in open_landings])
        self.assertEqual(len({tuple(sorted(r.counters().items())) for r in ranks}), 1)
        self.assertGreater(ranks[0].stats["evicted_pages"], 0)
        self.assertGreater(ranks[0].stats["landed"], 0)


if __name__ == "__main__":
    unittest.main()

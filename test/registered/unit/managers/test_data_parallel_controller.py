"""DPBudget + DataParallelController dispatch tests.

The e2e counterpart, over the real scheduler load-report path, is
test/registered/disaggregation/test_disaggregation_dp_attention.py.

Fragility: scheduler tests bypass `DataParallelController.__init__` via
`__new__` and inject only the attrs the schedulers read (`workers`, `status`,
`_active_workers`, `round_robin_counter`, `dp_budget`). Update `_make_controller`
if a scheduler starts reading another attr. `maybe_external_dp_rank_routing`
is exercised as the real method, no mock. The refresh-throttle tests inject
`load_snapshot_reader` and `_last_refresh_time` on top of those.
"""

import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import msgspec.structs

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.data_parallel_controller import (
    DataParallelController,
    DPBudget,
    LoadBalanceMethod,
)
from sglang.srt.managers.dp_prefix_router import (
    PrefixAffinityIndex,
    PrefixAwareSelector,
)
from sglang.srt.managers.load_snapshot import LoadSnapshot

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


_BASE_LOAD = msgspec.structs.replace(
    LoadSnapshot(dp_rank=0),
    max_total_num_tokens=4096,
    max_running_requests=128,
)


def _load(**overrides) -> LoadSnapshot:
    return msgspec.structs.replace(_BASE_LOAD, **overrides)


def _make_controller(num_dp_ranks: int) -> DataParallelController:
    """Bypass __init__; inject only the attrs dispatch methods read."""
    ctl = DataParallelController.__new__(DataParallelController)
    ctl.workers = [MagicMock(name=f"worker_{i}") for i in range(num_dp_ranks)]
    ctl.status = [True] * num_dp_ranks
    ctl._active_workers = list(range(num_dp_ranks))
    ctl.round_robin_counter = 0
    ctl.dp_budget = DPBudget(num_dp_ranks=num_dp_ranks)
    return ctl


def _req(routed_dp_rank=None, bootstrap_room=None, input_ids=None):
    """Req stand-in; SimpleNamespace avoids pinning to the Req dataclass schema."""
    return SimpleNamespace(
        routed_dp_rank=routed_dp_rank,
        bootstrap_room=bootstrap_room,
        input_ids=input_ids or [],
    )


class TestDPBudgetUpdateBudget(CustomTestCase):
    def test_maps_running_plus_waiting_to_total_requests(self):
        budget = DPBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(dp_rank=0, timestamp=1.0, num_running_reqs=3, num_waiting_reqs=2),
                _load(dp_rank=1, timestamp=1.0, num_running_reqs=5, num_waiting_reqs=1),
            ]
        )
        self.assertEqual(budget.total_requests, [5, 6])

    def test_maps_num_total_tokens_not_num_used_tokens(self):
        budget = DPBudget(num_dp_ranks=2)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_used_tokens=100, num_total_tokens=150
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_used_tokens=80, num_total_tokens=80
                ),
            ]
        )
        self.assertEqual(budget.total_tokens, [150, 80])

    def test_partial_update_only_affects_reported_rank(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.update_budget(
            [
                _load(
                    dp_rank=0, timestamp=1.0, num_running_reqs=10, num_total_tokens=100
                ),
                _load(
                    dp_rank=1, timestamp=1.0, num_running_reqs=20, num_total_tokens=200
                ),
                _load(
                    dp_rank=2, timestamp=1.0, num_running_reqs=30, num_total_tokens=300
                ),
            ]
        )
        budget.update_budget(
            [
                _load(
                    dp_rank=1,
                    timestamp=2.0,
                    num_running_reqs=1,
                    num_waiting_reqs=1,
                    num_total_tokens=50,
                )
            ]
        )
        self.assertEqual(budget.total_requests, [10, 2, 30])
        self.assertEqual(budget.total_tokens, [100, 50, 300])


class TestDPBudgetDispatch(CustomTestCase):
    """DPBudget.dispatch picks a rank from current state and updates counters."""

    def test_total_requests_dispatch_picks_min_and_increments(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.total_requests = [4, 2, 7]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_REQUESTS)
        self.assertEqual(rank, 1)
        self.assertEqual(
            budget.total_requests[1],
            3,
            "dispatch should increment chosen worker's request count",
        )

    def test_total_tokens_dispatch_applies_estimated_tokens(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.total_tokens = [100, 50, 200]
        budget.total_requests = [0, 0, 0]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=30)
        self.assertEqual(rank, 1, "should pick worker with min total_tokens")
        self.assertEqual(
            budget.total_tokens[1],
            80,
            "dispatch should add estimated_tokens to chosen worker",
        )
        self.assertEqual(
            budget.total_requests[1],
            1,
            "dispatch should also increment request count",
        )

    def test_total_tokens_tie_breaks_on_total_requests(self):
        budget = DPBudget(num_dp_ranks=3)
        budget.total_tokens = [50, 50, 50]
        budget.total_requests = [4, 2, 7]
        rank = budget.dispatch(LoadBalanceMethod.TOTAL_TOKENS, estimated_tokens=10)
        self.assertEqual(
            rank, 1, "tie on total_tokens should fall back to min total_requests"
        )

    def test_dispatch_returns_none_for_methods_not_handled(self):
        """Round-robin and follow_bootstrap_room dispatch elsewhere; DPBudget
        only handles the load-aware variants."""
        budget = DPBudget(num_dp_ranks=3)
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.ROUND_ROBIN))
        self.assertIsNone(budget.dispatch(LoadBalanceMethod.FOLLOW_BOOTSTRAP_ROOM))


class TestRoundRobinScheduler(CustomTestCase):
    def test_cycles_through_active_workers_in_order(self):
        ctl = _make_controller(num_dp_ranks=4)
        for _ in range(8):
            ctl.round_robin_scheduler(_req())
        # 8 reqs across 4 active workers — 2 each, in round-robin order
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(worker.send_pyobj.call_count, 2, f"worker {i} call count")

    def test_first_dispatch_picks_worker_zero(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()
        for i in (1, 2, 3):
            ctl.workers[i].send_pyobj.assert_not_called()

    def test_skips_inactive_workers(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.status[1] = False
        ctl.status[3] = False
        for _ in range(6):
            ctl.round_robin_scheduler(_req())
        # Only workers 0 and 2 are active — should split 6 reqs evenly
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 3)
        ctl.workers[1].send_pyobj.assert_not_called()
        self.assertEqual(ctl.workers[2].send_pyobj.call_count, 3)
        ctl.workers[3].send_pyobj.assert_not_called()

    def test_routed_dp_rank_bypasses_counter(self):
        """External dp-rank routing must not advance the counter."""
        ctl = _make_controller(num_dp_ranks=4)
        ctl.round_robin_scheduler(_req(routed_dp_rank=2))
        ctl.workers[2].send_pyobj.assert_called_once()
        self.assertEqual(
            ctl.round_robin_counter,
            0,
            "external routing must not advance the round-robin counter",
        )
        # Subsequent round-robin req still lands on worker 0
        ctl.round_robin_scheduler(_req())
        ctl.workers[0].send_pyobj.assert_called_once()


class TestFollowBootstrapRoomScheduler(CustomTestCase):
    def test_dispatches_by_bootstrap_room_modulo(self):
        ctl = _make_controller(num_dp_ranks=4)
        for room, expected_rank in [
            (0, 0),
            (1, 1),
            (4, 0),
            (5, 1),
            (100, 0),
            (101, 1),
        ]:
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=room))
            ctl.workers[expected_rank].send_pyobj.assert_called()

    def test_requires_bootstrap_room(self):
        ctl = _make_controller(num_dp_ranks=4)
        with self.assertRaises(AssertionError):
            ctl.follow_bootstrap_room_scheduler(_req(bootstrap_room=None))

    def test_routed_dp_rank_bypasses_bootstrap_room(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.follow_bootstrap_room_scheduler(_req(routed_dp_rank=3, bootstrap_room=1))
        ctl.workers[3].send_pyobj.assert_called_once()
        ctl.workers[1].send_pyobj.assert_not_called()


class TestTotalRequestsScheduler(CustomTestCase):
    def test_dispatches_to_min_request_worker(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req())
        ctl.workers[2].send_pyobj.assert_called_once()
        for i in (0, 1, 3):
            ctl.workers[i].send_pyobj.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_requests[2],
            2,
            "DPBudget must record the dispatch by incrementing the counter",
        )

    def test_routed_dp_rank_bypasses_budget(self):
        ctl = _make_controller(num_dp_ranks=4)
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.total_requests_scheduler(_req(routed_dp_rank=0))
        ctl.workers[0].send_pyobj.assert_called_once()
        # DPBudget must not be touched when bypassed
        self.assertEqual(
            ctl.dp_budget.total_requests,
            [5, 3, 1, 4],
            "external routing must not mutate DPBudget state",
        )


class TestStatusAwarenessInconsistency(CustomTestCase):
    """Document a divergence: ``round_robin_scheduler`` skips workers whose
    ``status`` is False, but ``total_requests_scheduler`` /
    ``total_tokens_scheduler`` route purely by DPBudget — they do NOT
    consult ``self.status``. If a future change unifies this behaviour,
    this test will fail and force a reviewer to confirm intent."""

    def test_total_requests_ignores_status(self):
        ctl = _make_controller(num_dp_ranks=4)
        # Worker 2 is the global minimum AND marked inactive.
        ctl.dp_budget.total_requests = [5, 3, 1, 4]
        ctl.status[2] = False
        ctl.total_requests_scheduler(_req())
        # Current behaviour: still dispatches to the inactive worker.
        ctl.workers[2].send_pyobj.assert_called_once()


class TestRefreshLoadBudgetThrottle(CustomTestCase):
    @staticmethod
    def _controller_with_reader(num_dp_ranks, snapshots):
        ctl = _make_controller(num_dp_ranks)
        ctl.load_snapshot_reader = MagicMock()
        ctl.load_snapshot_reader.read_all.return_value = snapshots
        return ctl

    def test_throttled_refresh_spreads_a_burst_across_ranks(self):
        idle = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(4)]
        ctl = self._controller_with_reader(num_dp_ranks=4, snapshots=idle)
        # A refresh stamp in the future keeps every call inside the window, so
        # the burst runs entirely on speculative counters.
        ctl._last_refresh_time = time.perf_counter() + 3600.0

        for _ in range(8):
            ctl.refresh_load_budget()
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 100))

        ctl.load_snapshot_reader.read_all.assert_not_called()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [200, 200, 200, 200],
            "speculative increments should spread the burst evenly",
        )
        for i, worker in enumerate(ctl.workers):
            self.assertEqual(
                worker.send_pyobj.call_count, 2, f"worker {i} should get 2 of 8 reqs"
            )

    def test_refresh_outside_window_overwrites_speculative_increments(self):
        reported = [
            _load(dp_rank=0, timestamp=2.0, num_total_tokens=10),
            _load(dp_rank=1, timestamp=2.0, num_total_tokens=20),
        ]
        ctl = self._controller_with_reader(num_dp_ranks=2, snapshots=reported)
        ctl._last_refresh_time = 0.0  # window has long passed
        ctl.dp_budget.total_tokens = [999, 999]

        ctl.refresh_load_budget()

        ctl.load_snapshot_reader.read_all.assert_called_once()
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            [10, 20],
            "a fresh snapshot must replace the speculative state",
        )
        self.assertGreater(ctl._last_refresh_time, 0.0)

    def test_unchanged_snapshot_does_not_reset_the_burst(self):
        frozen = [_load(dp_rank=i, timestamp=1.0, num_total_tokens=0) for i in range(2)]
        ctl = self._controller_with_reader(num_dp_ranks=2, snapshots=frozen)
        ctl._last_refresh_time = 0.0
        ctl.refresh_load_budget()  # adopts timestamp 1.0

        for _ in range(4):
            ctl.total_tokens_scheduler(_req(input_ids=[0] * 50))
        after_burst = list(ctl.dp_budget.total_tokens)

        ctl._last_refresh_time = 0.0  # let the next refresh through the throttle
        ctl.refresh_load_budget()  # same timestamp -> update_budget skips it

        self.assertEqual(
            after_burst, [100, 100], "burst should have spread over both ranks"
        )
        self.assertEqual(
            ctl.dp_budget.total_tokens,
            after_burst,
            "a stale-timestamp snapshot must not wipe the speculative state",
        )


def _prompt(session: int, num_blocks: int, block_size: int = 4) -> list[int]:
    """A prompt whose blocks are unique to ``session``."""
    return [session * 10_000 + i for i in range(num_blocks * block_size)]


def _selector(num_ranks=4, block_size=4, capacity_blocks=1000, **overrides):
    kwargs = dict(
        cache_threshold=0.5,
        balance_abs_threshold=4,
        balance_rel_threshold=1.5,
    )
    kwargs.update(overrides)
    return PrefixAwareSelector(
        num_ranks=num_ranks,
        block_size=block_size,
        capacity_blocks=capacity_blocks,
        **kwargs,
    )


class TestPrefixAffinityIndex(CustomTestCase):
    def test_block_keys_chain_over_the_whole_prefix(self):
        index = PrefixAffinityIndex(num_ranks=1, block_size=2, capacity_blocks=10)
        a = index.block_keys([1, 2, 3, 4, 5])
        b = index.block_keys([9, 9, 3, 4])
        self.assertEqual(len(a), 2, "the partial tail block is ignored")
        self.assertNotEqual(a[1], b[1], "same block after a different prefix")
        self.assertEqual(a, index.block_keys([1, 2, 3, 4]))

    def test_match_counts_leading_blocks_per_rank(self):
        index = PrefixAffinityIndex(num_ranks=3, block_size=2, capacity_blocks=10)
        index.insert(0, index.block_keys([1, 2, 3, 4]))
        index.insert(2, index.block_keys([1, 2, 7, 7]))
        self.assertEqual(index.match(index.block_keys([1, 2, 3, 4, 5, 6])), [2, 0, 1])

    def test_lru_evicts_leaves_before_their_prefix(self):
        index = PrefixAffinityIndex(num_ranks=1, block_size=1, capacity_blocks=3)
        index.insert(0, index.block_keys([1, 2, 3]))
        index.insert(0, index.block_keys([5]))  # pushes out the oldest key
        self.assertEqual(index.num_blocks(0), 3)
        self.assertEqual(
            index.match(index.block_keys([1, 2, 3])),
            [2],
            "the leaf block goes first; the prefix [1, 2] still matches",
        )

    def test_reinsert_refreshes_the_prefix(self):
        index = PrefixAffinityIndex(num_ranks=1, block_size=1, capacity_blocks=3)
        index.insert(0, index.block_keys([1, 2]))
        index.insert(0, index.block_keys([7]))
        index.insert(0, index.block_keys([1, 2]))  # [7] is now the oldest
        index.insert(0, index.block_keys([8]))
        self.assertEqual(index.match(index.block_keys([1, 2])), [2])
        self.assertEqual(index.match(index.block_keys([7])), [0])


class TestPrefixAwareSelector(CustomTestCase):
    def test_new_prompts_go_to_least_loaded_rank(self):
        sel = _selector()
        self.assertEqual(sel.select(_prompt(1, 4), [3, 1, 2, 5], [0, 1, 2, 3]), 1)

    def test_follow_up_turn_returns_to_the_rank_holding_its_prefix(self):
        sel = _selector()
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0, 0, 0], [0, 1, 2, 3]), 0)
        turn2 = turn1 + _prompt(2, 4)  # turn 1 + its response + a new message
        self.assertEqual(sel.select(turn2, [1, 0, 0, 0], [0, 1, 2, 3]), 0)

    def test_sessions_spread_and_each_stays_on_its_rank(self):
        sel = _selector()
        loads = [0, 0, 0, 0]
        first = {}
        for s in range(4):
            first[s] = sel.select(_prompt(s, 4), loads, [0, 1, 2, 3])
            loads[first[s]] += 1
        self.assertEqual(sorted(first.values()), [0, 1, 2, 3])
        for s in range(4):
            turn2 = _prompt(s, 4) + _prompt(100 + s, 4)
            self.assertEqual(sel.select(turn2, loads, [0, 1, 2, 3]), first[s])

    def test_shared_system_prompt_does_not_pull_new_sessions(self):
        sel = _selector()
        system = _prompt(0, 2)
        loads = [0, 0, 0, 0]
        ranks = []
        for s in range(4):
            # 2 shared blocks of 8: below the 0.5 cache threshold.
            r = sel.select(system + _prompt(1 + s, 6), loads, [0, 1, 2, 3])
            loads[r] += 1
            ranks.append(r)
        self.assertEqual(sorted(ranks), [0, 1, 2, 3])

    def test_long_reply_keeps_the_session_on_its_rank(self):
        sel = _selector()
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0, 0, 0], [0, 1, 2, 3]), 0)
        # Turn 1 + an 8-block reply/tool result the index never saw: 4 of 12
        # blocks match, under the 0.5 ratio, but the match ends where turn 1 did.
        turn2 = turn1 + _prompt(2, 8)
        self.assertEqual(sel.select(turn2, [1, 0, 0, 0], [0, 1, 2, 3]), 0)

    def test_match_ending_mid_prompt_still_needs_the_ratio(self):
        sel = _selector()
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0, 0, 0], [0, 1, 2, 3]), 0)
        # Shares only turn 1's first 2 blocks: 2 of 10 match, no prompt ended there.
        other = turn1[:8] + _prompt(2, 8)
        self.assertEqual(sel.select(other, [1, 0, 0, 0], [0, 1, 2, 3]), 1)

    def test_long_shared_prefix_packs_until_the_overload_guard(self):
        """Known limit: a shared prefix over cache_threshold of the prompt pulls
        new sessions to the ranks holding it until they are overloaded."""
        sel = _selector(num_ranks=8)
        system = _prompt(0, 6)
        loads = [0] * 8
        for s in range(16):
            # 6 shared blocks of 8: over the 0.5 cache threshold.
            r = sel.select(system + _prompt(1 + s, 2), loads, list(range(8)))
            loads[r] += 1
        self.assertEqual(loads, [5, 5, 5, 1, 0, 0, 0, 0])
        # Once every rank holds the shared part, equal matches go by load.
        for r in range(8):
            sel.index.insert(r, sel.index.block_keys(system))
        for s in range(16, 32):
            r = sel.select(system + _prompt(1 + s, 2), loads, list(range(8)))
            loads[r] += 1
        self.assertEqual(loads, [5, 5, 5, 4, 4, 3, 3, 3])

    def test_overloaded_rank_loses_its_prefix_matches(self):
        sel = _selector(balance_abs_threshold=2, balance_rel_threshold=1.5)
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0], [0, 1]), 0)
        turn2 = turn1 + _prompt(2, 4)
        # Rank 0 has 2 more requests than rank 1: within the threshold.
        self.assertEqual(sel.select(turn2, [3, 1], [0, 1]), 0)
        # 4 more (and > 1.5x): rank 0 is skipped, and the turn re-caches on 1.
        turn3 = turn2 + _prompt(3, 4)
        self.assertEqual(sel.select(turn3, [5, 1], [0, 1]), 1)
        turn4 = turn3 + _prompt(4, 4)
        self.assertEqual(
            sel.select(turn4, [1, 1], [0, 1]), 1, "rank 1 now holds the longest prefix"
        )

    def test_equal_matches_tie_break_on_load(self):
        sel = _selector()
        prefix = _prompt(1, 4)
        sel.select(prefix, [0, 0, 0], [0, 1, 2])
        sel.index.insert(2, sel.index.block_keys(prefix))
        self.assertEqual(sel.select(prefix + [1], [2, 0, 1], [0, 1, 2]), 2)

    def test_only_active_ranks_are_chosen(self):
        sel = _selector()
        turn1 = _prompt(1, 4)
        self.assertEqual(sel.select(turn1, [0, 0, 0], [0, 1, 2]), 0)
        self.assertEqual(sel.select(turn1 + _prompt(2, 4), [0, 0, 0], [1, 2]), 1)

    def test_prompt_shorter_than_a_block_goes_by_load(self):
        sel = _selector(block_size=16)
        self.assertEqual(sel.select([1, 2, 3], [2, 0], [0, 1]), 1)
        self.assertEqual(sel.select([1, 2, 3], [0, 2], [0, 1]), 0)


class TestPrefixAwareScheduler(CustomTestCase):
    def _controller(self, num_dp_ranks=4):
        ctl = _make_controller(num_dp_ranks)
        ctl.prefix_selector = _selector(num_ranks=num_dp_ranks)
        return ctl

    def test_sticks_a_session_and_bumps_the_budget(self):
        ctl = self._controller()
        turn1 = _prompt(1, 4)
        ctl.prefix_aware_scheduler(_req(input_ids=turn1))
        ctl.prefix_aware_scheduler(_req(input_ids=_prompt(2, 4)))
        ctl.prefix_aware_scheduler(_req(input_ids=turn1 + _prompt(3, 4)))
        self.assertEqual(ctl.workers[0].send_pyobj.call_count, 2)
        self.assertEqual(ctl.workers[1].send_pyobj.call_count, 1)
        self.assertEqual(ctl.dp_budget.total_requests, [2, 1, 0, 0])
        self.assertEqual(ctl.dp_budget.total_tokens, [16 + 32, 16, 0, 0])

    def test_routed_dp_rank_bypasses_prefix_routing(self):
        ctl = self._controller()
        ctl.prefix_aware_scheduler(_req(routed_dp_rank=3, input_ids=_prompt(1, 4)))
        ctl.workers[3].send_pyobj.assert_called_once()
        self.assertEqual(ctl.dp_budget.total_requests, [0, 0, 0, 0])

    def test_skips_inactive_workers(self):
        ctl = self._controller()
        ctl._active_workers = [1, 3]
        ctl.prefix_aware_scheduler(_req(input_ids=_prompt(1, 4)))
        ctl.workers[1].send_pyobj.assert_called_once()

    def test_selector_derives_block_size_capacity_and_log_interval(self):
        ctl = _make_controller(4)
        ctl.max_dp_size = 4
        ctl.max_total_num_tokens = 1000
        ctl.page_size = 64
        with patch(
            "sglang.srt.managers.data_parallel_controller.get_observability",
            return_value=SimpleNamespace(decode_log_interval=40),
        ):
            sel = ctl._make_prefix_selector()
        self.assertEqual(sel.index.block_size, 64, "one index block per radix page")
        self.assertEqual(sel.index.capacity_blocks, 4 * 1000 // 64)
        self.assertEqual(sel.log_interval, 40)

    def test_selector_builds_within_dp_controller_namespaces(self):
        """Under SGLANG_ROLE_NAMESPACES=enforce the controller may only read its
        audited namespaces; building the selector runs during its startup, so an
        out-of-set read (e.g. `schedule`) would abort the server."""
        import sglang.srt.runtime_context as rc

        override = rc.get_context().override_server_args(
            attn_dp_size=4, load_balance_method="prefix_aware"
        )
        override.install()
        self.addCleanup(override.restore)
        ctl = _make_controller(4)
        ctl.max_dp_size = 4
        ctl.max_total_num_tokens = 1000
        ctl.page_size = 128
        with (
            patch.object(rc, "_ROLE_NS_MODE", "enforce"),
            patch.object(rc._CONTEXT, "_publish_role", "dp_controller"),
        ):
            sel = ctl._make_prefix_selector()
        self.assertEqual(sel.index.block_size, 128)

    def test_method_parses_from_server_arg(self):
        self.assertEqual(
            LoadBalanceMethod.from_str("prefix_aware"), LoadBalanceMethod.PREFIX_AWARE
        )


if __name__ == "__main__":
    unittest.main()

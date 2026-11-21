"""DP routing by pending uncached prompt tokens (SGLANG_TPU_DP_PROMPT_BACKLOG_MARGIN):
the router's decisions (sglang_tpu/dp_prompt_backlog.py) and the hooks that
install it on the DP controller (sglang_tpu/hooks/dp_prompt_backlog.py)."""

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
from sglang.srt.managers.dp_prefix_router import PrefixAwareSelector
from sglang.srt.managers.load_snapshot import LoadSnapshot
from sglang_tpu.dp_prompt_backlog import PromptBacklogRouter, backlog_routing_enabled

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

B = 16  # block tokens in these tests


def _load(rank, ts=1.0, uncached=0, running=0, waiting=0, hit=0.0):
    return msgspec.structs.replace(
        LoadSnapshot(dp_rank=rank),
        timestamp=ts,
        num_waiting_uncached_tokens=uncached,
        num_running_reqs=running,
        num_waiting_reqs=waiting,
        cache_hit_rate=hit,
    )


def _router(n=4, margin=1.0, routed=True, **kw):
    kw.setdefault("min_moved_tokens", 4 * B)
    return PromptBacklogRouter(
        num_ranks=n, margin=margin, override_routed=routed, block_tokens=B, **kw
    )


def _hold(r, rank, ids):
    """Record that `rank` caches `ids` (as an earlier routed turn would)."""
    r.index.insert(rank, r.index.block_keys(ids))


def _ids(seed, n):
    return [(seed * 1_000_003 + i) % 50_000 for i in range(n)]


class TestPromptBacklogRouter(CustomTestCase):
    def test_keeps_stock_until_gap_exceeds_margin_times_prompt(self):
        r = _router()
        r.update([_load(0, uncached=1000), _load(1, uncached=0)] + [_load(2), _load(3)])
        r.backlog[2] = r.backlog[3] = 500
        # gap 1000 == 1.0 x 1000: stays
        self.assertEqual(r.route(_ids(1, 1000), 0, [0, 1, 2, 3], routed=False), 0)
        # gap now 1000 + local estimate > 1.0 x 500: moves to rank 1
        self.assertEqual(r.route(_ids(2, 500), 0, [0, 1, 2, 3], routed=False), 1)
        self.assertEqual(r.stats.moved, 1)

    def test_margin_scales_the_threshold(self):
        r = _router(margin=3.0)
        r.update([_load(0, uncached=2500), _load(1), _load(2), _load(3)])
        self.assertEqual(r.route(_ids(1, 1000), 0, [0, 1, 2, 3], routed=False), 0)
        r = _router(margin=2.0)
        r.update([_load(0, uncached=2500), _load(1), _load(2), _load(3)])
        self.assertEqual(r.route(_ids(1, 1000), 0, [0, 1, 2, 3], routed=False), 1)

    def test_least_ties_break_on_requests_then_index(self):
        r = _router()
        r.update(
            [
                _load(0, uncached=10_000),
                _load(1, running=3),
                _load(2, running=1),
                _load(3, running=1),
            ]
        )
        self.assertEqual(r.route(_ids(1, 100), 0, [0, 1, 2, 3], routed=False), 2)

    def test_only_active_ranks_are_candidates(self):
        r = _router()
        r.update([_load(0, uncached=10_000), _load(1), _load(2, uncached=50), _load(3)])
        self.assertEqual(r.route(_ids(1, 100), 0, [0, 2], routed=False), 2)
        # A stock rank that is no longer active is always left.
        self.assertEqual(r.route(_ids(2, 100), 1, [0, 2], routed=False), 2)

    def test_routed_requests_are_hard_unless_enabled(self):
        r = _router(routed=False)
        r.update([_load(0, uncached=10_000), _load(1), _load(2), _load(3)])
        self.assertEqual(r.route(_ids(1, 100), 0, [0, 1, 2, 3], routed=True), 0)
        self.assertEqual(r.stats.requests, 0)
        # Its work still counts against rank 0 until the next snapshot.
        self.assertEqual((r.backlog[0], r.requests[0]), (10_100, 1))
        r = _router(routed=True)
        r.update([_load(0, uncached=10_000), _load(1), _load(2), _load(3)])
        self.assertEqual(r.route(_ids(1, 100), 0, [0, 1, 2, 3], routed=True), 1)
        self.assertEqual(r.stats.moved_new, 1)

    def test_stale_snapshots_keep_local_estimates(self):
        r = _router()
        r.update([_load(i, ts=1.0) for i in range(4)])
        r.route(_ids(1, 800), 1, [0, 1, 2, 3], routed=False)  # hit 0: +800 on 1
        self.assertEqual(r.backlog[1], 800)
        r.update([_load(1, ts=1.0)])  # same timestamp: ignored
        self.assertEqual(r.backlog[1], 800)
        r.update([_load(1, ts=2.0, uncached=5)])
        self.assertEqual(r.backlog[1], 5)

    def test_local_estimate_uses_hit_rate_unless_moved(self):
        r = _router()
        r.update([_load(0, hit=0.9), _load(1, hit=0.9), _load(2), _load(3)])
        r.route(_ids(1, 1000), 0, [0, 1, 2, 3], routed=False)
        self.assertAlmostEqual(r.backlog[0], 100.0)
        r.backlog[0] = 10_000
        chosen = r.route(_ids(2, 1000), 0, [0, 1, 2, 3], routed=False)
        self.assertNotEqual(chosen, 0)
        self.assertEqual(r.backlog[chosen], 1000)  # moved: no cache there

    def test_rejects_non_positive_margin(self):
        with self.assertRaises(ValueError):
            _router(margin=0.0)


class TestRouterInputs(CustomTestCase):
    def test_pinned_session_requests_never_move(self):
        r = _router()
        r.update([_load(0, uncached=100_000), _load(1), _load(2), _load(3)])
        self.assertEqual(
            r.route(_ids(1, 1000), 0, [0, 1, 2, 3], routed=False, pinned=True), 0
        )
        self.assertEqual(r.backlog[0], 101_000)

    def test_embedding_prompts_use_their_length(self):
        r = _router()
        r.update([_load(0, uncached=1500), _load(1), _load(2), _load(3)])
        self.assertEqual(
            r.route([], 0, [0, 1, 2, 3], routed=False, prompt_tokens=1000), 1
        )
        self.assertEqual(r.backlog[1], 1000)
        self.assertEqual(r.index.num_blocks(1), 0)  # nothing hashable to record

    def test_zero_length_prompt_keeps_stock(self):
        r = _router()
        r.update([_load(0, uncached=100_000), _load(1), _load(2), _load(3)])
        self.assertEqual(r.route([], 0, [0, 1, 2, 3], routed=True), 0)


class TestNoCacheLossRouting(CustomTestCase):
    """Routed requests under SGLANG_TPU_DP_PROMPT_BACKLOG_ROUTED=1."""

    def test_new_conversation_goes_to_least_backlogged_rank(self):
        r = _router()
        r.update(
            [_load(0, uncached=100_000), _load(1, uncached=50), _load(2), _load(3)]
        )
        self.assertEqual(r.route(_ids(1, 40 * B), 0, [0, 1, 2, 3], routed=True), 2)
        self.assertEqual(r.stats.moved_new, 1)
        self.assertEqual(r.backlog[2], 40 * B)  # full prefill there

    def test_continuation_stays_pinned_however_backlogged(self):
        r = _router()
        turn1 = _ids(1, 40 * B)
        _hold(r, 0, turn1)
        r.update([_load(0, uncached=10_000_000), _load(1), _load(2), _load(3)])
        turn2 = turn1 + _ids(2, 4 * B)
        self.assertEqual(r.route(turn2, 0, [0, 1, 2, 3], routed=True), 0)
        self.assertEqual(r.stats.moved, 0)
        self.assertEqual(r.backlog[0], 10_000_000 + 4 * B)  # only the new tail

    def test_moved_conversation_follows_its_cache(self):
        r = _router()
        r.update([_load(0, uncached=100_000), _load(1), _load(2), _load(3)])
        turn1 = _ids(7, 40 * B)
        self.assertEqual(r.route(turn1, 0, [0, 1, 2, 3], routed=True), 1)
        r.update([_load(i, ts=2.0) for i in range(4)])
        turn2 = turn1 + _ids(8, 3 * B)  # the front end still pins rank 0
        self.assertEqual(r.route(turn2, 0, [0, 1, 2, 3], routed=True), 1)
        turn3 = turn2 + _ids(9, 3 * B)
        self.assertEqual(r.route(turn3, 0, [0, 1, 2, 3], routed=True), 1)
        self.assertEqual(r.stats.followed, 2)

    def test_shared_system_prompt_is_not_a_continuation(self):
        r = _router()
        system = _ids(3, 10 * B)  # 20% of the prompts
        _hold(r, 1, system + _ids(4, 40 * B))
        r.update([_load(0), _load(1, uncached=100_000), _load(2), _load(3)])
        # A new session sharing the system prompt, hinted to the busy rank 1.
        self.assertEqual(r.route(system + _ids(5, 40 * B), 1, [0, 1, 2, 3], True), 0)
        self.assertEqual(r.stats.moved_new, 1)

    def test_new_conversation_moves_only_past_the_margin(self):
        r = _router(margin=2.0)
        r.update([_load(0, uncached=1500), _load(1), _load(2), _load(3)])
        self.assertEqual(r.route(_ids(1, 1000), 0, [0, 1, 2, 3], routed=True), 0)

    def test_continuation_margin_moves_continuations_when_set(self):
        r = _router(continuation_margin=4.0)
        turn1 = _ids(1, 40 * B)
        _hold(r, 0, turn1)
        r.update([_load(0, uncached=100_000), _load(1), _load(2), _load(3)])
        turn2 = turn1 + _ids(2, 4 * B)
        self.assertEqual(r.route(turn2, 0, [0, 1, 2, 3], routed=True), 1)
        self.assertEqual(r.stats.moved_continuations, 1)
        self.assertEqual(r.backlog[1], len(turn2))  # recomputed there

    def test_index_is_bounded(self):
        r = _router(capacity_blocks=8)
        r.update([_load(i) for i in range(4)])
        for k in range(10):
            r.route(_ids(k, 4 * B), 0, [0], routed=True)
        self.assertLessEqual(r.index.num_blocks(0), 8)


class TestContinuationBoundaries(CustomTestCase):
    def test_turn_extending_a_whole_prompt_by_more_than_it_had_stays(self):
        r = _router()
        turn1 = _ids(1, 4 * B)
        r.update([_load(i) for i in range(4)])
        self.assertEqual(r.route(turn1, 0, [0, 1, 2, 3], routed=True), 0)
        r.update([_load(0, ts=2.0, uncached=100_000)])
        turn2 = turn1 + _ids(2, 5 * B)  # 4/9 held, but it extends turn 1
        self.assertEqual(r.route(turn2, 0, [0, 1, 2, 3], routed=True), 0)
        self.assertEqual(r.stats.moved, 0)

    def test_short_prompts_keep_their_hint(self):
        r = _router()
        r.update([_load(0, uncached=100_000), _load(1), _load(2), _load(3)])
        self.assertEqual(r.route(_ids(1, 3 * B), 0, [0, 1, 2, 3], routed=True), 0)

    def test_hinted_holder_beats_a_deeper_index_entry_elsewhere(self):
        r = _router()
        prompt = _ids(1, 8 * B)
        _hold(r, 0, prompt)  # possibly evicted there since
        _hold(r, 1, prompt[: 6 * B])
        r.update([_load(0, uncached=1_000_000), _load(1), _load(2), _load(3)])
        self.assertEqual(r.route(prompt, 1, [0, 1, 2, 3], routed=True), 1)

    def test_siblings_sharing_a_parent_prefix_stay_with_the_parent(self):
        """Documented tradeoff: as the session-affinity proxy places them."""
        r = _router()
        parent = _ids(1, 30 * B)
        _hold(r, 0, parent + _ids(2, 10 * B))
        r.update([_load(0, uncached=10_000), _load(1), _load(2), _load(3)])
        child = parent + _ids(3, 20 * B)  # 60% shared
        self.assertEqual(r.route(child, 2, [0, 1, 2, 3], routed=True), 0)

    def test_predicate_includes_elastic(self):
        self.assertTrue(backlog_routing_enabled(1.0, "round_robin", "null"))
        self.assertFalse(
            backlog_routing_enabled(1.0, "round_robin", "null", elastic=True)
        )


class TestPromptGrowth(CustomTestCase):
    def test_short_conversation_keeps_its_rank_as_it_grows(self):
        r = _router()
        r.update([_load(i) for i in range(4)])
        turn1 = _ids(1, 3 * B)  # under the 4-block minimum
        self.assertEqual(r.route(turn1, 0, [0, 1, 2, 3], routed=True), 0)
        r.update([_load(0, ts=2.0, uncached=100_000)])
        turn2 = turn1 + _ids(2, B)
        self.assertEqual(r.route(turn2, 0, [0, 1, 2, 3], routed=True), 0)
        self.assertEqual(r.stats.moved, 0)

    def test_extension_needs_the_earlier_prompts_tail(self):
        r = _router()
        r.update([_load(i) for i in range(4)])
        first = _ids(1, 4 * B + 1)  # a partial block past 4 full ones
        self.assertEqual(r.route(first, 0, [0, 1, 2, 3], routed=True), 0)
        r.update([_load(0, ts=2.0, uncached=1_000_000)])
        # Shares only the 4 full blocks, not the tail token: a new prompt.
        other = first[: 4 * B] + [first[4 * B] + 1] + _ids(3, 11 * B)
        self.assertEqual(r.route(other, 1, [0, 1, 2, 3], routed=True), 1)
        # A real extension (tail included) still counts as a continuation.
        turn2 = first + _ids(4, 11 * B)
        self.assertEqual(r.route(turn2, 0, [0, 1, 2, 3], routed=True), 0)


# ---------------------------------------------------------------------------
# Hooks on the DP controller (sglang_tpu/hooks/dp_prompt_backlog.py)

from sglang_tpu.hooks import dp_prompt_backlog as hooks  # noqa: E402

_DPC = "sglang.srt.managers.data_parallel_controller"


def _controller(n=4, margin=1.0, routed=True, method=None, launch=None, max_dp=None):
    ctl = DataParallelController.__new__(DataParallelController)
    ctl.workers = [MagicMock(name=f"worker_{i}") for i in range(n)]
    ctl.status = [True] * n
    ctl._active_workers = list(range(n))
    ctl.round_robin_counter = 0
    ctl.dp_budget = DPBudget(num_dp_ranks=n)
    ctl.prefix_selector = None
    ctl.load_balance_method = method or LoadBalanceMethod.ROUND_ROBIN
    ctl.launch_dp_size = launch or n
    ctl.max_dp_size = max_dp or n
    ctl.refresh_load_budget_on_dispatch = False
    with (
        patch(
            "sglang.srt.runtime_context.get_disagg",
            return_value=SimpleNamespace(disaggregation_mode="null"),
        ),
        patch(
            "sglang.srt.runtime_context.get_observability",
            return_value=SimpleNamespace(decode_log_interval=40),
        ),
        patch.dict(
            "os.environ",
            {
                "SGLANG_TPU_DP_PROMPT_BACKLOG_MARGIN": str(margin),
                "SGLANG_TPU_DP_PROMPT_BACKLOG_ROUTED": "1" if routed else "0",
            },
        ),
    ):
        hooks.init_controller(lambda self: None, ctl)
    return ctl


def _req(ids, routed=None, **fields):
    req = SimpleNamespace(
        routed_dp_rank=routed, bootstrap_room=None, input_ids=ids, input_embeds=None
    )
    for k, v in fields.items():
        setattr(req, k, v)
    return req


def _dispatch(ctl, scheduler, req):
    """Run one dispatch through the hooks: dispatching_with_trace scopes it and
    the module's sock_send is wrapped as the registry would; returns the rank."""
    sent = []

    def original_send(sock, obj, *a, **k):
        sent.append(ctl.workers.index(sock))

    def wrapped_send(sock, obj, *a, **k):
        return hooks.sock_send(original_send, sock, obj, *a, **k)

    with patch(f"{_DPC}.sock_send", wrapped_send):
        hooks.dispatching_with_trace(lambda self, r: scheduler(r), ctl, req)
    assert len(sent) == 1
    return sent[0]


class TestControllerHooks(CustomTestCase):
    def test_off_by_default(self):
        ctl = _controller(margin=0.0)
        self.assertIsNone(ctl._tpu_backlog_router)
        self.assertFalse(ctl.refresh_load_budget_on_dispatch)
        # Pass-through: dispatch goes where the stock policy sends it.
        self.assertEqual(
            _dispatch(ctl, ctl.round_robin_scheduler, _req(_ids(1, 5000))), 0
        )

    def test_enabled_turns_on_load_refresh_and_feeds_the_router(self):
        ctl = _controller()
        self.assertTrue(ctl.refresh_load_budget_on_dispatch)
        loads = (x for x in [_load(0, uncached=7), _load(1), _load(2), _load(3)])
        hooks.update_budget(DPBudget.update_budget, ctl.dp_budget, loads)
        self.assertEqual(ctl._tpu_backlog_router.backlog[0], 7)

    def test_off_with_elastic_dp_or_pd(self):
        self.assertIsNone(_controller(launch=2, max_dp=4)._tpu_backlog_router)
        with (
            patch(
                "sglang.srt.runtime_context.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="prefill"),
            ),
            patch.dict("os.environ", {"SGLANG_TPU_DP_PROMPT_BACKLOG_MARGIN": "1"}),
        ):
            self.assertIsNone(hooks.make_router(_controller(margin=0.0)))

    def test_round_robin_choice_and_routed_hint_are_overridden(self):
        ctl = _controller()
        ctl._tpu_backlog_router.update(
            [_load(0, uncached=50_000), _load(1, uncached=40_000), _load(2), _load(3)]
        )
        self.assertEqual(
            _dispatch(ctl, ctl.round_robin_scheduler, _req(_ids(1, 1000))), 2
        )
        hinted = _req(_ids(2, 5000), routed=1)
        self.assertEqual(_dispatch(ctl, ctl.maybe_external_dp_rank_routing, hinted), 3)
        self.assertEqual(ctl.dp_budget.total_requests, [0, 0, 0, 0])

    def test_hard_routes_stay_when_routed_override_is_off(self):
        ctl = _controller(routed=False)
        ctl._tpu_backlog_router.update(
            [_load(0, uncached=50_000), _load(1), _load(2), _load(3)]
        )
        req = _req(_ids(1, 5000), routed=0)
        self.assertEqual(_dispatch(ctl, ctl.maybe_external_dp_rank_routing, req), 0)

    def test_budget_increments_move_with_the_request(self):
        ctl = _controller(method=LoadBalanceMethod.TOTAL_TOKENS)
        ctl._tpu_backlog_router.update(
            [_load(0, uncached=50_000), _load(1), _load(2), _load(3)]
        )
        ctl.dp_budget.total_tokens = [0, 10, 10, 10]
        self.assertEqual(
            _dispatch(ctl, ctl.total_tokens_scheduler, _req(_ids(1, 1000))), 1
        )
        self.assertEqual(ctl.dp_budget.total_tokens, [0, 1010, 10, 10])
        self.assertEqual(ctl.dp_budget.total_requests, [0, 1, 0, 0])

        ctl = _controller(method=LoadBalanceMethod.TOTAL_REQUESTS)
        ctl._tpu_backlog_router.update(
            [_load(0, uncached=50_000), _load(1), _load(2), _load(3)]
        )
        ctl.dp_budget.total_requests = [0, 5, 5, 5]
        self.assertEqual(
            _dispatch(ctl, ctl.total_requests_scheduler, _req(_ids(1, 1000))), 1
        )
        self.assertEqual(ctl.dp_budget.total_requests, [0, 6, 5, 5])
        self.assertEqual(ctl.dp_budget.total_tokens, [0, 0, 0, 0])

    def test_prefix_index_records_the_rank_that_got_the_prompt(self):
        ctl = _controller(method=LoadBalanceMethod.PREFIX_AWARE)
        ctl.prefix_selector = PrefixAwareSelector(
            num_ranks=4,
            block_size=B,
            capacity_blocks=1 << 12,
            cache_threshold=0.5,
            balance_abs_threshold=4,
            balance_rel_threshold=1.5,
        )
        ctl._tpu_backlog_router.update(
            [_load(0, uncached=50_000), _load(1), _load(2), _load(3)]
        )
        ids = _ids(1, 10 * B)
        self.assertEqual(_dispatch(ctl, ctl.prefix_aware_scheduler, _req(ids)), 1)
        index = ctl.prefix_selector.index
        self.assertEqual(index.match(index.block_keys(ids))[:2], [0, 10])

    def test_pinned_requests_keep_their_rank(self):
        ctl = _controller()
        ctl._tpu_backlog_router.update(
            [_load(0, uncached=50_000), _load(1), _load(2), _load(3)]
        )
        for field, value in (
            ("session_params", {"id": "s"}),
            ("session_id", "s"),
            ("lora_id", "adapter"),
            ("extra_key", "key"),
            ("cache_salt", "salt"),
        ):
            req = _req(_ids(1, 5000), routed=0, **{field: value})
            self.assertEqual(_dispatch(ctl, ctl.maybe_external_dp_rank_routing, req), 0)

    def test_embedding_request_is_sized_by_its_embeddings(self):
        ctl = _controller()
        ctl._tpu_backlog_router.update(
            [_load(0, uncached=1500), _load(1), _load(2), _load(3)]
        )
        req = _req(None, input_embeds=[[0.0]] * 1000)
        self.assertEqual(_dispatch(ctl, ctl.round_robin_scheduler, req), 1)

    def test_sends_outside_a_dispatch_pass_through(self):
        calls = []
        sock = object()
        hooks.sock_send(lambda s, o: calls.append((s, o)), sock, "obj")
        self.assertEqual(calls, [(sock, "obj")])


def _prefix_controller():
    ctl = _controller(method=LoadBalanceMethod.PREFIX_AWARE)
    ctl.prefix_selector = PrefixAwareSelector(
        num_ranks=4,
        block_size=B,
        capacity_blocks=1 << 12,
        cache_threshold=0.5,
        balance_abs_threshold=4,
        balance_rel_threshold=1.5,
    )
    ctl._tpu_backlog_router.update(
        [_load(0, uncached=50_000), _load(1), _load(2), _load(3)]
    )
    return ctl


class TestPrefixAwareEdges(CustomTestCase):
    def test_routed_requests_leave_the_prefix_index_alone(self):
        """External routing bypasses select(); a redirected hint must not
        create or move prefix-index claims."""
        ctl = _prefix_controller()
        ids = _ids(1, 5000)
        req = _req(ids, routed=0)
        self.assertEqual(_dispatch(ctl, ctl.maybe_external_dp_rank_routing, req), 1)
        index = ctl.prefix_selector.index
        self.assertEqual(index.match(index.block_keys(ids)), [0, 0, 0, 0])

    def test_embedding_redirect_moves_no_tokens(self):
        """prefix_aware adds len(input_ids) tokens (0 for embeddings); the
        transfer must move exactly that, not the embedding length."""
        ctl = _prefix_controller()
        ctl._tpu_backlog_router.update(
            [
                _load(0, ts=2.0, uncached=1500),
                _load(1, ts=2.0),
                _load(2, ts=2.0),
                _load(3, ts=2.0),
            ]
        )
        req = _req(None, input_embeds=[[0.0]] * 1000)
        self.assertEqual(_dispatch(ctl, ctl.prefix_aware_scheduler, req), 1)
        self.assertEqual(ctl.dp_budget.total_tokens, [0, 0, 0, 0])
        self.assertEqual(ctl.dp_budget.total_requests, [0, 1, 0, 0])

    def test_failed_send_leaves_the_prefix_index_on_the_stock_rank(self):
        ctl = _prefix_controller()
        ids = _ids(1, 10 * B)

        def failing_send(sock, obj, *a, **k):
            raise RuntimeError("socket closed")

        def wrapped_send(sock, obj, *a, **k):
            return hooks.sock_send(failing_send, sock, obj, *a, **k)

        with patch(f"{_DPC}.sock_send", wrapped_send):
            with self.assertRaises(RuntimeError):
                hooks.dispatching_with_trace(
                    lambda self, r: ctl.prefix_aware_scheduler(r), ctl, _req(ids)
                )
        index = ctl.prefix_selector.index
        self.assertEqual(index.match(index.block_keys(ids))[:2], [10, 0])


class TestZmqOwnershipHook(CustomTestCase):
    def _owner(
        self,
        method,
        margin,
        caller="DataParallelController",
        disagg="null",
        max_ep=None,
    ):
        from sglang.srt.managers import load_snapshot

        parallel = SimpleNamespace(
            node_rank=0, num_dp_ranks=8, load_balance_method=method, max_ep_size=max_ep
        )
        with (
            patch.object(load_snapshot, "should_use_zmq", return_value=True),
            patch.object(load_snapshot, "get_parallel", return_value=parallel),
            patch("sglang.srt.runtime_context.get_parallel", return_value=parallel),
            patch.object(
                load_snapshot,
                "get_serving",
                return_value=SimpleNamespace(tokenizer_worker_num=1),
            ),
            patch(
                "sglang.srt.runtime_context.get_disagg",
                return_value=SimpleNamespace(disaggregation_mode=disagg),
            ),
            patch.dict(
                "os.environ", {"SGLANG_TPU_DP_PROMPT_BACKLOG_MARGIN": str(margin)}
            ),
        ):
            return hooks.zmq_reader_owner(load_snapshot.zmq_reader_owner, caller)

    def test_controller_owns_loads_when_backlog_routing_round_robin(self):
        self.assertFalse(self._owner("round_robin", 0))
        self.assertTrue(self._owner("round_robin", 0, caller="TokenizerManager"))
        self.assertTrue(self._owner("round_robin", 1.0))
        self.assertFalse(self._owner("round_robin", 1.0, caller="TokenizerManager"))
        self.assertFalse(self._owner("round_robin", 1.0, disagg="prefill"))
        self.assertFalse(self._owner("round_robin", 1.0, max_ep=16))
        self.assertTrue(
            self._owner("round_robin", 1.0, caller="TokenizerManager", max_ep=16)
        )
        # Load-aware methods: the stock rule already gives the controller the socket.
        self.assertTrue(self._owner("total_requests", 1.0))


class TestHookRegistration(CustomTestCase):
    def test_hooks_are_registered(self):
        from sglang_tpu import hooks as tpu_hooks

        targets = {s[0] for s in tpu_hooks.HOOKS if s[1].__module__ == hooks.__name__}
        self.assertEqual(
            targets,
            {
                f"{_DPC}.DataParallelController.__init__",
                f"{_DPC}.DPBudget.update_budget",
                f"{_DPC}.DataParallelController.dispatching_with_trace",
                f"{_DPC}.sock_send",
                "sglang.srt.managers.load_snapshot.zmq_reader_owner",
            },
        )

    def test_flags_are_registered_default_off(self):
        from sglang_tpu.envs import envs

        self.assertEqual(envs.SGLANG_TPU_DP_PROMPT_BACKLOG_MARGIN.get(), 0.0)
        self.assertFalse(envs.SGLANG_TPU_DP_PROMPT_BACKLOG_ROUTED.get())
        self.assertEqual(
            envs.SGLANG_TPU_DP_PROMPT_BACKLOG_CONTINUATION_MARGIN.get(), 0.0
        )


if __name__ == "__main__":
    import unittest

    unittest.main()

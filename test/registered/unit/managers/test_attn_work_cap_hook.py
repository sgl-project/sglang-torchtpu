"""CPU checks of the prefill attention-work cap and SGLANG_PREFILL_SPF hooks."""

import pkgutil
import unittest
from array import array
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock

from test_prefill_adder import TestPrefillAdder

from sglang.srt.managers.schedule_batch import NextBatchPlan, Req
from sglang.srt.managers.schedule_policy import AddReqResult, SchedulePolicy
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.plugins.hook_registry import HookRegistry, _propagate_patch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import attn_work_cap as awc
from sglang_tpu.envs import envs
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import attn_work_cap as hooks
from sglang_tpu.hooks.attn_work_cap import hook_specs

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

W = 400_000_000
PAGE = 256
# A page-aligned context (radix prefixes are) where 400M pairs fit 3919 tokens.
DEEP = 391 * PAGE
SCRIPT_CAP = 200_000_000


def _apply(stack: ExitStack, specs) -> None:
    """Apply through the real registry, restoring every binding afterwards."""
    for target, hook, kind in specs:
        owner_path, name = target.rsplit(".", 1)
        owner = pkgutil.resolve_name(owner_path)
        raw = vars(owner).get(name)
        original = getattr(owner, name)
        HookRegistry._apply_target(target, [(kind, hook, None)])
        wrapped = getattr(owner, name)

        def restore(
            owner=owner, name=name, raw=raw, original=original, wrapped=wrapped
        ):
            setattr(owner, name, raw)
            _propagate_patch(wrapped, original, owner)

        stack.callback(restore)


def _running(decoding: bool) -> MagicMock:
    batch = MagicMock()
    batch.is_empty.return_value = not decoding
    batch.is_prefill_only = False
    return batch


class TestCapMath(CustomTestCase):
    def test_chunk_work(self):
        self.assertEqual(awc.chunk_attn_work(4096, 100_000), 4096 * (100_000 + 2048))
        self.assertEqual(awc.chunk_attn_work(0, 5), 0)

    def test_fit_is_the_largest_chunk_within_the_room(self):
        for ctx in (0, 1, 255, 4096, 100_000, 1_000_000):
            for room in (1, 1000, 25e6, W, 3.7e9):
                with self.subTest(ctx=ctx, room=room):
                    n = awc.attn_fit(ctx, room)
                    self.assertLessEqual(awc.chunk_attn_work(n, ctx), room)
                    self.assertGreater(awc.chunk_attn_work(n + 1, ctx), room)
        self.assertEqual(awc.attn_fit(10, 0), 0)
        self.assertEqual(awc.attn_fit(10, -5), 0)
        # 4096 tokens at ctx 100k is 418M pairs: 400M fits 3923.
        self.assertEqual(awc.attn_fit(100_000, W), 3923)

    def test_pass_work_sums_chunks(self):
        self.assertEqual(
            awc.pass_attn_work([(0, 1024), (100_000, 256)]),
            1024 * 512 + 256 * (100_000 + 128),
        )

    def test_capped_chunk_tokens(self):
        def capped(ctx, remaining, *, work=0.0, first=True, cap=W):
            return awc.capped_chunk_tokens(
                cap=cap,
                work=work,
                ctx=ctx,
                remaining=remaining,
                grid=PAGE,
                first=first,
            )

        # Fits whole: no cap.
        self.assertIsNone(capped(0, 4096))
        self.assertIsNone(capped(100_000, 3923))
        # Clipped to the largest page multiple that fits.
        self.assertEqual(capped(100_000, 4096), 3840)
        self.assertEqual(capped(100_000, 4096, first=False), 3840)
        # Less than a page of room: the first prefill still gets one page.
        self.assertEqual(capped(10_000_000, 4096), PAGE)
        self.assertEqual(capped(100_000, 4096, work=W), PAGE)
        # A later request waits.
        self.assertEqual(capped(10_000_000, 4096, first=False), 0)
        self.assertEqual(capped(0, 4096, work=W, first=False), 0)
        # A short rest that fits is never cut, whatever the page.
        self.assertIsNone(capped(0, 10, work=W - 100, first=False))

    def test_batch_shows_decode(self):
        def batch(mode, tokens, extend=False, plan=None):
            return SimpleNamespace(
                forward_mode=mode,
                global_num_tokens=tokens,
                is_extend_in_batch=extend,
                mixed_verify=plan,
            )

        self.assertFalse(awc.batch_shows_decode(None))
        self.assertTrue(awc.batch_shows_decode(batch(ForwardMode.IDLE, [0, 12, 0])))
        self.assertTrue(awc.batch_shows_decode(batch(ForwardMode.DECODE, [24, 0])))
        self.assertTrue(
            awc.batch_shows_decode(batch(ForwardMode.TARGET_VERIFY, [24, 0]))
        )
        self.assertFalse(awc.batch_shows_decode(batch(ForwardMode.IDLE, [0, 0])))
        # Extend steps (this rank prefilling or idle beside one) do not tell.
        self.assertIsNone(awc.batch_shows_decode(batch(ForwardMode.EXTEND, [512, 0])))
        self.assertIsNone(
            awc.batch_shows_decode(batch(ForwardMode.IDLE, [0, 512], extend=True))
        )
        self.assertIsNone(awc.batch_shows_decode(batch(ForwardMode.DECODE, None)))
        # A mixed prefill + verify step carries every rank's verify count.
        for verify, seen in (((0, 3), True), ((0, 0), False)):
            plan = SimpleNamespace(global_verify_bs=verify)
            self.assertIs(
                awc.batch_shows_decode(
                    batch(ForwardMode.EXTEND, [512, 9], extend=True, plan=plan)
                ),
                seen,
            )


class TestPassScope(CustomTestCase):
    def _scheduler(self, *, dp=True, seen=None):
        scheduler = SimpleNamespace(require_mlp_sync=dp)
        if seen is not None:
            setattr(scheduler, hooks._DECODE_SEEN, seen)
        return scheduler

    def _cap_seen_by_pass(self, scheduler, running_batch):
        seen = []

        def original(sched, prefill_delayer_single_pass=None, running_batch=None):
            seen.append(hooks._Pass.cap)
            return None, running_batch

        hooks.cap_prefill_pass(
            original,
            scheduler,
            prefill_delayer_single_pass=None,
            running_batch=running_batch,
        )
        self.assertEqual(hooks._Pass.cap, 0)
        return seen[0]

    def test_off_by_default(self):
        self.assertEqual(envs.SGLANG_DP_ATTN_WORK_CAP.get(), 0)
        self.assertFalse(envs.SGLANG_PREFILL_SPF.get())
        for decoding in (False, True):
            self.assertEqual(
                self._cap_seen_by_pass(self._scheduler(seen=True), _running(decoding)),
                0,
            )

    def test_cap_needs_a_decoder_on_some_rank(self):
        with envs.SGLANG_DP_ATTN_WORK_CAP.override(W):
            cases = (
                # (DP attention, some rank decoded last step, local decode, cap)
                (True, None, False, 0),
                (True, False, False, 0),
                (True, True, False, W),
                (True, False, True, W),
                (False, None, True, W),
                (False, True, False, 0),  # no DP: the flag is never consulted
            )
            for dp, seen, local, cap in cases:
                with self.subTest(dp=dp, seen=seen, local=local):
                    self.assertEqual(
                        self._cap_seen_by_pass(
                            self._scheduler(dp=dp, seen=seen), _running(local)
                        ),
                        cap,
                    )

    def test_prefill_only_running_batch_does_not_decode(self):
        batch = _running(True)
        batch.is_prefill_only = True
        with envs.SGLANG_DP_ATTN_WORK_CAP.override(W):
            self.assertEqual(self._cap_seen_by_pass(self._scheduler(), batch), 0)

    def test_pass_state_is_cleared_on_error(self):
        def original(sched, **kwargs):
            raise RuntimeError("boom")

        with envs.SGLANG_DP_ATTN_WORK_CAP.override(W):
            with self.assertRaises(RuntimeError):
                hooks.cap_prefill_pass(
                    original, self._scheduler(), running_batch=_running(True)
                )
        self.assertEqual(hooks._Pass.cap, 0)

    def test_tracks_decode_across_steps(self):
        scheduler = self._scheduler()

        def step(batch):
            plan = NextBatchPlan(batch_to_run=batch, running_batch=MagicMock())
            self.assertIsNone(hooks.track_dp_decode(plan, scheduler, batch, None))
            return getattr(scheduler, hooks._DECODE_SEEN, None)

        decode = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            global_num_tokens=[0, 4],
            is_extend_in_batch=False,
            mixed_verify=None,
        )
        extend = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            global_num_tokens=[2048, 0],
            is_extend_in_batch=True,
            mixed_verify=None,
        )
        # Off: nothing recorded.
        self.assertIsNone(step(decode))
        with envs.SGLANG_DP_ATTN_WORK_CAP.override(W):
            self.assertTrue(step(decode))
            self.assertTrue(step(extend))  # kept through an extend step
            self.assertFalse(step(None))  # every rank idle
            self.assertFalse(step(extend))


class TestCappedAdder(TestPrefillAdder):
    """Reuses TestPrefillAdder's fixtures; its tests are dropped below."""

    def setUp(self):
        super().setUp()
        self.mock_tree_cache.supports_mamba.return_value = False
        self.mock_tree_cache.supports_prefix_sharing.return_value = False
        self.mock_tree_cache.transfers_in_flight.return_value = False
        self.mock_token_allocator.available_size.return_value = 1 << 20
        self.mock_token_allocator.full_available_size.return_value = 1 << 20
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        _apply(self.stack, hook_specs())

    def _adder(self, rem_chunk_tokens=4096):
        return self.create_adder(
            self.create_running_batch(),
            page_size=PAGE,
            rem_chunk_tokens=rem_chunk_tokens,
            rem_input_tokens=1 << 20,
        )

    def _req(self, rid, new_tokens, prefix_len=0):
        req = self.create_shared_req(rid)
        req.prefix_len = prefix_len
        req.full_untruncated_fill_ids = list(range(prefix_len + new_tokens))
        return req

    def _add(self, adder, req, has_chunked_req=False, chunk_tokens_cap=None):
        return adder.add_one_req(
            req,
            has_chunked_req=has_chunked_req,
            truncation_align_size=None,
            chunk_tokens_cap=chunk_tokens_cap,
        )

    @staticmethod
    def _policy():
        return SchedulePolicy(
            policy="fcfs",
            tree_cache=RadixCache.create_simulated(),
            enable_hierarchical_cache=False,
            enable_priority_scheduling=False,
            schedule_low_priority_values_first=False,
        )

    def _resume(self, adder, req):
        adder.chunked_req_limit = self._policy().shortest_prefill_chunk_limit(
            req, [], adder.rem_chunk_tokens, adder.page_size, processed_tokens=0
        )
        return adder.add_chunked_req(req)

    def test_registered(self):
        for spec in hook_specs():
            self.assertIn(spec, HOOKS)

    def test_no_cap_outside_a_capped_pass(self):
        adder = self._adder()
        req = self._req("deep", 4096, prefix_len=DEEP)
        self._add(adder, req)
        self.assertEqual(req.extend_len, 4096)
        self.assertIsNone(adder.new_chunked_req)

    def test_clips_a_deep_chunk_and_ends_the_pass(self):
        with hooks.capped_pass(W):
            adder = self._adder()
            req = self._req("deep", 4096, prefix_len=DEEP)
            res = self._add(adder, req)
            behind = self._req("short", 16)
            later = self._add(adder, behind)
        self.assertEqual(req.extend_len, 3840)
        self.assertIs(adder.new_chunked_req, req)
        self.assertEqual(res, AddReqResult.OTHER)  # chunk budget closed
        self.assertNotEqual(later, AddReqResult.CONTINUE)
        self.assertEqual(adder.can_run_list, [req])

    def test_fresh_and_shallow_chunks_keep_the_chunk_size(self):
        """The cap only lowers chunks: chunked_prefill_size still bounds them."""
        with hooks.capped_pass(W):
            adder = self._adder()
            req = self._req("fresh", 8192)
            self._add(adder, req)
        self.assertEqual(req.extend_len, 4096)
        self.assertIs(adder.new_chunked_req, req)

    def test_keeps_a_tighter_request_cap(self):
        with hooks.capped_pass(W):
            adder = self._adder()
            req = self._req("deep", 4096, prefix_len=DEEP)
            self._add(adder, req, chunk_tokens_cap=1024)
        self.assertEqual(req.extend_len, 1024)

    def test_first_prefill_gets_a_page(self):
        with hooks.capped_pass(1_000_000):
            adder = self._adder()
            req = self._req("very-deep", 4096, prefix_len=DEEP)
            self._add(adder, req)
        self.assertEqual(req.extend_len, PAGE)
        self.assertEqual(adder.can_run_list, [req])

    def test_later_request_waits_without_a_page_of_room(self):
        with hooks.capped_pass(1_000_000):
            adder = self._adder()
            first = self._req("fresh", 1024)  # 524,288 pairs
            self.assertEqual(self._add(adder, first), AddReqResult.CONTINUE)
            fits = self._req("tiny", 8, prefix_len=1024)  # 8,224 pairs
            self.assertEqual(self._add(adder, fits), AddReqResult.CONTINUE)
            deep = self._req("deep", 1024, prefix_len=DEEP)
            self.assertEqual(self._add(adder, deep), AddReqResult.OTHER)
        self.assertEqual(adder.can_run_list, [first, fits])
        self.assertIsNone(deep.extend_end)
        self.assertEqual(first.extend_len, 1024)
        self.assertEqual(fits.extend_len, 8)

    def test_resumed_chunk_keeps_its_slot(self):
        """An in-progress chunked request runs first, clipped but at least one
        page; requests behind it run only if they fit whole in the room left."""
        with hooks.capped_pass(W):
            adder = self._adder(rem_chunk_tokens=8192)
            cont = self._req("continuation", 20_000, prefix_len=DEEP)
            self.assertIs(self._resume(adder, cont), cont)
            self.assertEqual(cont.extend_len, 3840)
            short = self._req("short", 1024)  # 524,288 pairs fit the 8.3M left
            self.assertEqual(
                self._add(adder, short, has_chunked_req=True), AddReqResult.CONTINUE
            )
            long = self._req("long", 4096, prefix_len=50_176)
            self._add(adder, long, has_chunked_req=True)
        self.assertEqual(adder.can_run_list, [cont, short])
        self.assertIsNone(long.extend_end)

        with hooks.capped_pass(1_000_000):
            adder = self._adder()
            cont = self._req("continuation", 20_000, prefix_len=DEEP)
            self._resume(adder, cont)
        self.assertEqual(cont.extend_len, PAGE)

    def test_resumed_chunk_that_fits_is_untouched(self):
        with hooks.capped_pass(W):
            adder = self._adder()
            cont = self._req("continuation", 2048, prefix_len=1024)
            self.assertIsNone(self._resume(adder, cont))
        self.assertEqual(cont.extend_len, 2048)
        self.assertIsNone(adder.chunked_req_limit)

    def test_cuts_land_on_the_truncation_alignment(self):
        """Page 128 with 256-token truncation alignment: the minimum and the
        clipped chunk are legal cuts, not rounded to nothing by core."""
        for cap, prefix, extend in ((1_000_000, DEEP, 256), (250_000, 0, 512)):
            with self.subTest(cap=cap), hooks.capped_pass(cap):
                adder = self.create_adder(
                    self.create_running_batch(),
                    page_size=128,
                    rem_chunk_tokens=4096,
                    rem_input_tokens=1 << 20,
                )
                req = self._req("aligned", 8192, prefix_len=prefix)
                adder.add_one_req(req, has_chunked_req=False, truncation_align_size=256)
                self.assertEqual(req.extend_len, extend)
                self.assertEqual(adder.can_run_list, [req])

    def test_ignore_eos_without_radix_cache_is_capped(self):
        """Core admits these through add_one_req_ignore_eos, which takes no
        chunk_tokens_cap: the cap goes through the chunk budget."""
        self.mock_tree_cache.disable = True

        def admit():
            adder = self._adder()
            req = self._req("ignore-eos", 4096)
            req.origin_input_ids = list(range(4096))
            req.sampling_params.ignore_eos = True
            res = self._add(adder, req)
            return adder, req, res

        _, req, _ = admit()
        self.assertEqual(req.extend_len, 4096)  # uncapped pass
        with hooks.capped_pass(1_000_000):
            adder, req, res = admit()
        self.assertEqual(req.extend_len, 1280)  # 1414 fit, page floored
        self.assertIs(adder.new_chunked_req, req)
        self.assertEqual(adder.rem_chunk_tokens, 0)
        self.assertEqual(res, AddReqResult.OTHER)
        self.assertLessEqual(awc.chunk_attn_work(1280, 0), 1_000_000)

    def test_declined_host_load_back_stays_capped(self):
        """The cap holds whether the host-cache load-back delivers the promised
        prefix or nothing (core then re-selects behind the resident prefix)."""
        for loaded, prefix, extend in ((0, DEEP, 3840), (50_176, DEEP + 50_176, 100)):
            with self.subTest(loaded=loaded), hooks.capped_pass(W):
                adder = self._adder()
                req = self._req("host-hit", 50_176 + 100, prefix_len=DEEP)
                req.host_hit_length = 50_176
                req.best_match_node = req.last_node
                req.needs_host_load_back.return_value = True
                self.mock_tree_cache.init_load_back.return_value = (
                    loaded,
                    req.last_node,
                )
                self._add(adder, req)
                self.assertEqual(adder.can_run_list, [req])
                self.assertEqual((req.prefix_len, req.extend_len), (prefix, extend))
                self.assertLessEqual(awc.chunk_attn_work(extend, prefix), W)

    def _scripted_pass(self, *, decoding: bool):
        """One scheduler-shaped pass (resumed chunk, then the queue) through the
        pass hook. Returns what it admitted and the budgets it left."""
        adder = self._adder()
        cont = self._req("continuation", 2000, prefix_len=120_064)
        queue = [
            self._req("hit", 512, prefix_len=60_160),
            self._req("fresh", 300),
            self._req("deep", 4096, prefix_len=200_192),
            self._req("after", 64),
        ]
        policy = self._policy()

        def raw(scheduler, prefill_delayer_single_pass=None, running_batch=None):
            adder.chunked_req_limit = policy.shortest_prefill_chunk_limit(
                cont, queue, adder.rem_chunk_tokens, adder.page_size, processed_tokens=0
            )
            adder.add_chunked_req(cont)
            for req in queue:
                if self._add(adder, req, has_chunked_req=True) != AddReqResult.CONTINUE:
                    break
            return None, running_batch

        hooks.cap_prefill_pass(
            raw,
            SimpleNamespace(require_mlp_sync=True),
            prefill_delayer_single_pass=None,
            running_batch=_running(decoding),
        )
        return (
            [(r.rid, r.prefix_len, r.extend_len) for r in adder.can_run_list],
            adder.rem_chunk_tokens,
            adder.rem_input_tokens,
            adder.rem_total_tokens,
        )

    def test_off_matches_core_on_a_scripted_queue(self):
        hooked_off = self._scripted_pass(decoding=True)
        with envs.SGLANG_DP_ATTN_WORK_CAP.override(SCRIPT_CAP):
            no_decoder = self._scripted_pass(decoding=False)
            capped = self._scripted_pass(decoding=True)
        self.stack.close()  # core alone
        core = self._scripted_pass(decoding=True)
        self.assertEqual(hooked_off, core)
        self.assertEqual(no_decoder, core)
        self.assertEqual(
            [rid for rid, _, _ in core[0]], ["continuation", "hit", "fresh"]
        )
        # Capped: the continuation's 2000 tokens at ctx 120k (242M pairs) are
        # cut to 1536, and the hit behind it has less than a page of room.
        self.assertEqual(capped[0], [("continuation", 120_064, 1536)])
        work = awc.pass_attn_work((ctx, n) for _, ctx, n in capped[0])
        self.assertLessEqual(work, SCRIPT_CAP)


class TestShortestPrefillFirstEnv(CustomTestCase):
    def _init(self, policy="fcfs", priority=False):
        scheduler = SimpleNamespace(
            schedule_policy=policy, enable_priority_scheduling=priority
        )
        built = []

        def original(sched):
            built.append(sched.schedule_policy)

        hooks.shortest_prefill_first(original, scheduler)
        return built[0]

    def test_env_turns_fcfs_into_shortest_prefill_first(self):
        self.assertEqual(self._init(), "fcfs")
        with envs.SGLANG_PREFILL_SPF.override(True):
            self.assertEqual(self._init(), "shortest-prefill-first")
            self.assertEqual(self._init("lpm"), "lpm")
            with self.assertRaises(ValueError):
                self._init(priority=True)

    def test_cap_refuses_the_hierarchical_cache(self):
        from unittest.mock import patch

        def memory(hicache):
            return patch(
                "sglang.srt.runtime_context.get_memory",
                return_value=SimpleNamespace(enable_hierarchical_cache=hicache),
            )

        # Cap off: the host cache is fine (the guard does not even look).
        with memory(True):
            self.assertEqual(self._init(), "fcfs")
        with envs.SGLANG_DP_ATTN_WORK_CAP.override(W):
            # Cap on without the host cache: boots.
            with memory(False):
                self.assertEqual(self._init(), "fcfs")
            # Cap on with the host cache: refused at boot, with the reason.
            with memory(True), self.assertRaisesRegex(ValueError, "hierarchical"):
                self._init()

    def _queue(self):
        def req(rid, uncached, cached=0, arrived=0):
            r = Req(rid, "", array("q", range(uncached + cached)), SamplingParams())
            r.full_untruncated_fill_ids = r.origin_input_ids[:]
            r.num_matched_prefix_tokens = cached
            r.prefix_len = cached
            r.time_stats.wait_queue_entry_time = arrived
            return r

        return [
            req("long", 8192, arrived=0),
            req("hit", 64, cached=50_000, arrived=1),
            req("short-late", 256, arrived=3),
            req("short-early", 256, arrived=2),
        ]

    def test_order(self):
        for policy, order in (
            ("fcfs", ["long", "hit", "short-late", "short-early"]),
            (
                "shortest-prefill-first",
                ["hit", "short-early", "short-late", "long"],
            ),
        ):
            with self.subTest(policy=policy):
                sp = SchedulePolicy(
                    policy=policy,
                    tree_cache=RadixCache.create_simulated(),
                    enable_hierarchical_cache=False,
                    enable_priority_scheduling=False,
                    schedule_low_priority_values_first=False,
                )
                queue = self._queue()
                # Keep the scripted cache hits (the simulated tree is empty).
                sp._compute_prefix_matches = lambda q, p: set()
                ranked = sp.calc_priority(queue)
                self.assertEqual([r.rid for r in ranked], order)


# Keep the inherited TestPrefillAdder tests from running a second time here.
for _name in list(vars(TestPrefillAdder)):
    if _name.startswith("test_") and _name not in vars(TestCappedAdder):
        setattr(TestCappedAdder, _name, None)
del TestPrefillAdder


if __name__ == "__main__":
    unittest.main()

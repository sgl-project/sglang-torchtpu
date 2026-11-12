"""The Kimi-K3 decode program as sglang's K3_PROGRAM speculative worker, over a
stub program: the rows each step hands it, finished requests, the result the
output processor reads, and the registered algorithm."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.k3_programs import sglang_bridge as bridge
from sglang_tpu.k3_programs import sglang_colocated as worker_colocated
from sglang_tpu.k3_programs import sglang_worker as worker
from sglang_tpu.k3_programs.kimi_k3 import pools
from sglang_tpu.k3_programs.program import StepOutput

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

SCHEDULE = ((1, 4, 7), (5, 8, 3))
STEP = 8192  # tokens per logical block (bf16)


class StubProgram:
    """Records every StepInput; request i accepts ``accepted[i]`` tokens."""

    def __init__(self):
        self.inputs = []
        self.accepted = {}

    def step(self, step_input):
        self.inputs.append(step_input)
        rows = torch.full((max(len(step_input.requests), 1), 8), -1, dtype=torch.int32)
        for i, req in enumerate(step_input.requests):
            n = self.accepted.get(req.req_id, 1)
            rows[i, :n] = torch.arange(100 * i, 100 * i + n)
        return StepOutput(rows)


def _req(rid, prompt, output, committed, pool_idx, slot):
    return SimpleNamespace(
        rid=rid,
        grammar=None,
        is_retracted=False,
        finished=lambda: False,
        spec_verify_ct=0,
        spec_num_correct_drafts=0,
        update_spec_correct_drafts_histogram=lambda n: None,
        origin_input_ids=list(range(prompt)),
        output_ids=list(output),
        sampling_params=SimpleNamespace(
            temperature=1.0, top_p=1.0, top_k=1, sampling_seed=None
        ),
        kv=SimpleNamespace(
            kv_committed_len=committed,
            req_pool_idx=pool_idx,
            mamba_pool_idx=torch.tensor([slot]),
        ),
    )


def _req_to_token(rows, pages_per_row, context=4 * STEP):
    """Host req_to_token: row r's tokens on its listed sglang pages."""
    table = np.zeros((rows, context), np.int32)
    for r, pages in pages_per_row.items():
        for k, page in enumerate(pages):
            table[r, k * STEP : (k + 1) * STEP] = np.arange(
                page * STEP, (page + 1) * STEP
            )
    return table


class TestColocatedDpSteps(CustomTestCase):
    """Colocated attention DP2: a request's B rows on its first decode step
    (the partner halves copied in), DCP16 tables, B rows for new pages, and
    its decode-written halves copied back when it leaves."""

    def test_b_rows_through_a_request(self):
        from sglang_tpu.k3_programs import sglang_dp as dp

        page = dp.PAGE_TOKENS
        plan = bridge.UnifiedPoolPlan(60, 4, pools.cache_layout("fp8_e4m3"), dp_rows=8)
        program = StubProgram()
        steps = worker.DecodeSteps(program, plan, table_width=8, schedule=SCHEDULE)
        steps.dp, steps.table_tokens = None, dp.GROUP_BLOCK_TOKENS
        steps.colo = dp.ColoRows(plan.num_blocks, plan.dp_row_base, plan.dp_rows)
        jobs = []
        steps.swap = jobs.append
        r2t = np.zeros((2, 4 * page), np.int32)
        for k, p in enumerate((3, 6, 9)):
            r2t[1, k * page : (k + 1) * page] = np.arange(p * page, (p + 1) * page)
        base = plan.dp_row_base
        req = _req(
            "a", prompt=page + 5, output=[7], committed=page + 5, pool_idx=1, slot=2
        )
        steps.step([req], r2t[:, ::page])
        self.assertEqual(jobs, [[([3, 6], [base, base + 1], 0)]])
        (row,) = program.inputs[-1].requests
        # page + 5 + 8 scheduled tokens touch three DCP16 blocks: A, B, A.
        self.assertEqual(row.block_tables[0].tolist(), [3, base, 6, 0, 0, 0, 0, 0])
        req.output_ids += [8] * page
        req.kv.kv_committed_len = 2 * page + 3
        steps.step([req], r2t[:, ::page])
        self.assertEqual(len(jobs), 1)  # a new page's B row needs no copy
        (row,) = program.inputs[-1].requests
        self.assertEqual(row.block_tables[0].tolist()[:5], [3, base, 6, base + 1, 9])
        steps.colo_release("a")
        # Decode-written pages (index 1 on) back to the partners' rows.
        self.assertEqual(jobs[-1], [([base + 1, base + 2], [6, 9], 1)])
        self.assertEqual(steps.colo.free_rows(0), 8)
        steps.step([], r2t[:, ::page])  # "a" finishes: already released
        self.assertEqual(len(jobs), 2)


class TestDecodeSteps(CustomTestCase):
    def setUp(self):
        self.plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("auto"))
        self.program = StubProgram()
        self.steps = worker.DecodeSteps(
            self.program, self.plan, table_width=4, schedule=SCHEDULE
        )

    def test_drafts_follow_the_schedule(self):
        self.assertEqual(
            [worker.drafts_for(SCHEDULE, n) for n in (1, 4, 5, 8, 9)], [7, 7, 3, 3, 0]
        )
        self.assertEqual(worker.drafts_for(None, 1), 0)

    def test_first_step_rows(self):
        # A transferred request: the prompt's KV committed, the prefill's token next.
        req = _req(
            "a", prompt=STEP + 5, output=[7], committed=STEP + 5, pool_idx=1, slot=2
        )
        r2t = _req_to_token(2, {1: [3, 6]})
        self.program.accepted = {"a": 4}
        flat, lens = self.steps.step([req], r2t[:, ::STEP])
        (row,) = self.program.inputs[0].requests
        self.assertEqual((row.computed, row.scheduled), (STEP + 5, 8))
        self.assertEqual(row.token_ids[-1], 7)
        self.assertEqual(row.num_tokens, STEP + 6)
        self.assertEqual(
            row.block_tables[0].tolist(),
            [self.plan.kv_block(3), self.plan.kv_block(6), 0, 0],
        )
        self.assertEqual(
            [t.tolist() for t in row.block_tables[1:]],
            [[self.plan.state_block(2, g)] for g in range(3)],
        )
        self.assertEqual(lens.tolist(), [4])
        self.assertEqual(flat.tolist(), [0, 1, 2, 3, -1, -1, -1, -1])

    def test_departed_requests_are_finished(self):
        r2t = _req_to_token(4, {1: [1], 2: [2], 3: [3]})
        a, b, c = (
            _req(rid, prompt=10, output=[1], committed=10, pool_idx=i + 1, slot=i + 1)
            for i, rid in enumerate("abc")
        )
        self.steps.step([a, b, c], r2t[:, ::STEP])
        self.assertEqual(set(self.program.inputs[0].finished), set())
        self.steps.step([a, c], r2t[:, ::STEP])
        self.assertEqual(set(self.program.inputs[1].finished), {"b"})
        self.steps.forget_all()
        self.assertEqual(set(self.program.inputs[2].finished), {"a", "c"})
        self.assertEqual(list(self.program.inputs[2].requests), [])

    def test_a_tracked_request_reuses_its_prompt(self):
        req = _req("a", prompt=10, output=[7], committed=10, pool_idx=1, slot=1)
        r2t = _req_to_token(2, {1: [1]})[:, ::STEP]
        self.steps.step([req], r2t)
        first = self.program.inputs[0].requests[0]
        self.assertEqual(first.token_ids.tolist(), list(range(10)) + [7])
        req.output_ids += [8, 9]
        req.kv.kv_committed_len = 12
        self.steps.step([req], r2t)
        second = self.program.inputs[1].requests[0]
        # The program holds its next input: the cached prompt array, no copy.
        self.assertIs(second.token_ids, self.steps._prompts.get(req))
        self.assertEqual(len(second.token_ids), 10)
        self.assertEqual(second.num_tokens, 13)

    def test_a_resumed_request_is_rejected(self):
        r2t = _req_to_token(3, {1: [1], 2: [2]})[:, ::STEP]
        a = _req("a", prompt=10, output=[1], committed=10, pool_idx=1, slot=1)
        b = _req("b", prompt=10, output=[1], committed=10, pool_idx=2, slot=2)
        self.steps.step([a, b], r2t)
        self.steps.step([a], r2t)  # b retracted
        with self.assertRaisesRegex(RuntimeError, "came back after leaving"):
            self.steps.step([a, b], r2t)

    def test_wide_batches_step_plain(self):
        reqs = [
            _req(str(i), prompt=10, output=[1], committed=10, pool_idx=1, slot=1)
            for i in range(9)
        ]
        self.steps.step(reqs, _req_to_token(2, {1: [1]})[:, ::STEP])
        self.assertEqual({r.scheduled for r in self.program.inputs[0].requests}, {1})

    def test_first_step_from_the_prefills_token(self):
        # Colocated under overlap: the request reaches its first decode step
        # before the scheduler processed its prefill (no output, the committed
        # length still the prefix): the prefill's token and the whole prompt.
        req = _req("a", prompt=STEP + 5, output=[], committed=STEP, pool_idx=1, slot=2)
        r2t = _req_to_token(2, {1: [3, 6]})[:, ::STEP]
        with self.assertRaisesRegex(RuntimeError, "no first token"):
            self.steps.launch([req], r2t, lagging=True)
        self.steps.first_tokens["a"] = 7
        self.steps.launch([req], r2t, lagging=True)
        (row,) = self.program.inputs[-1].requests
        self.assertEqual((row.computed, row.num_tokens), (STEP + 5, STEP + 6))
        self.assertEqual(row.token_ids[-1], 7)
        self.assertEqual(self.steps.first_tokens, {})
        colocated = worker_colocated.ColocatedSteps(
            mock.Mock(spec=worker.PrefillSteps), self.steps
        )
        self.assertIs(colocated.prefill.first_tokens, self.steps.first_tokens)

    def test_extents_stop_at_the_tables(self):
        # A request at the context limit (4 pages here) under overlap: its
        # lagging extent would pass the last page the tables hold.
        prompt = 4 * STEP - 20
        req = _req("a", prompt=prompt, output=[7], committed=prompt, pool_idx=1, slot=1)
        r2t = _req_to_token(2, {1: [1, 2, 3, 4]})[:, ::STEP]
        self.steps.launch([req], r2t, lagging=True)
        req.output_ids += list(range(18))
        req.kv.kv_committed_len = 4 * STEP - 3
        req.kv.kv_allocated_len = 5 * STEP
        self.steps.launch([req], r2t, lagging=True)
        table = self.program.inputs[-1].requests[0].block_tables[0].tolist()
        self.assertEqual(table, [self.plan.kv_block(p) for p in (1, 2, 3, 4)])

    def test_first_tokens_are_bounded_and_flushed(self):
        self.steps.first_tokens.update({str(i): i for i in range(3)})
        self.steps.forget_all()
        self.assertEqual(self.steps.first_tokens, {})

    def test_lagging_tables_reach_a_step_further(self):
        # Under overlap scheduling a running request's committed length trails
        # its device position by a step: its tables cover one step more, up
        # to the KV sglang allocated.
        req = _req(
            "a", prompt=STEP - 10, output=[7], committed=STEP - 10, pool_idx=1, slot=1
        )
        req.kv.kv_allocated_len = 2 * STEP
        r2t = _req_to_token(2, {1: [3, 6]})[:, ::STEP]
        self.steps.launch([req], r2t, lagging=True)  # admitted: exact
        first = self.program.inputs[0].requests[0].block_tables[0].tolist()
        self.assertEqual(first, [self.plan.kv_block(3), 0, 0, 0])
        self.steps.launch([req], r2t, lagging=True)
        second = self.program.inputs[1].requests[0].block_tables[0].tolist()
        self.assertEqual(second, [self.plan.kv_block(3), self.plan.kv_block(6), 0, 0])
        req.kv.kv_allocated_len = STEP  # nothing allocated past the page
        self.steps.launch([req], r2t, lagging=True)
        third = self.program.inputs[2].requests[0].block_tables[0].tolist()
        self.assertEqual(third, [self.plan.kv_block(3), 0, 0, 0])

    def test_a_request_without_a_token_is_an_error(self):
        req = _req("a", prompt=10, output=[1], committed=10, pool_idx=1, slot=1)
        self.program.accepted = {"a": 0}
        with self.assertRaisesRegex(RuntimeError, r"\['a'\] took no token"):
            self.steps.step([req], _req_to_token(2, {1: [1]})[:, ::STEP])


class StubPrefill:
    """First token 50 + i for each request that finishes its prompt."""

    def __init__(self):
        self.inputs = []

    def step(self, step_input):
        self.inputs.append(step_input)
        rows = torch.full((max(len(step_input.requests), 1), 1), -1, dtype=torch.int32)
        for i, req in enumerate(step_input.requests):
            if req.end >= req.num_prompt_tokens:
                rows[i, 0] = 50 + i
        return StepOutput(rows)


class TestPrefillSteps(CustomTestCase):
    def setUp(self):
        self.plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("auto"))
        self.program = StubPrefill()
        self.steps = worker.PrefillSteps(self.program, self.plan, table_width=4)

    def test_chunks_and_first_tokens(self):
        long = _req("long", prompt=3 * STEP, output=[], committed=0, pool_idx=1, slot=1)
        short = _req("short", prompt=100, output=[], committed=0, pool_idx=2, slot=2)
        r2t = _req_to_token(3, {1: [4, 5, 6], 2: [7]})
        first = self.steps.step([long, short], [STEP, 0], [STEP, 100], r2t[:, ::STEP])
        self.assertEqual(first.tolist(), [0, 51])
        a, b = self.program.inputs[0].requests
        self.assertEqual(
            (a.computed, a.scheduled, a.num_prompt_tokens), (STEP, STEP, 3 * STEP)
        )
        self.assertTrue(a.partial)
        self.assertEqual(
            a.block_tables[0].tolist(),
            [self.plan.kv_block(4), self.plan.kv_block(5), 0, 0],
        )
        self.assertEqual(b.block_tables[1].tolist(), [self.plan.state_block(2, 0)])
        # The short prompt is done; the long one goes on with its last chunk.
        self.steps.step([long], [2 * STEP], [STEP], r2t[:, ::STEP])
        self.assertEqual(set(self.program.inputs[1].finished), {"short"})

    def test_a_first_token_must_match_the_prompt_end(self):
        req = _req("a", prompt=100, output=[], committed=0, pool_idx=1, slot=1)
        self.program.step = lambda step_input: StepOutput(
            torch.full((1, 1), -1, dtype=torch.int32)
        )
        with self.assertRaisesRegex(RuntimeError, "finished its prompt without"):
            self.steps.step([req], [0], [100], _req_to_token(2, {1: [1]})[:, ::STEP])


class TestPrefillWait(CustomTestCase):
    """Colocated engines hold a prefill back while requests decode, for a
    number of scheduler rounds (the same on every TP rank)."""

    def _scheduler(self, decoding, waiting, chunked=False, mode=None):
        from sglang.srt.disaggregation.utils import DisaggregationMode

        running = SimpleNamespace(is_empty=lambda: not decoding, is_prefill_only=False)
        return SimpleNamespace(
            running_batch=running,
            waiting_queue=[object()] * waiting,
            chunked_req=object() if chunked else None,
            disaggregation_mode=mode or DisaggregationMode.NULL,
        )

    def test_defers_for_the_rounds_then_lets_prefill_run(self):
        from sglang_tpu.hooks import k3_program as hooks

        scheduler = self._scheduler(True, 1)
        held = [hooks.prefill_wait_defers(scheduler, 3) for _ in range(5)]
        self.assertEqual(held, [True, True, True, False, True])
        # Nothing decoding or nothing waiting: no hold, and the count restarts.
        self.assertFalse(hooks.prefill_wait_defers(self._scheduler(False, 1), 3))
        idle = self._scheduler(True, 0)
        self.assertFalse(hooks.prefill_wait_defers(idle, 3))
        self.assertEqual(idle._k3_prefill_wait_rounds, 0)
        # A chunked prompt in progress waits like a queued one.
        self.assertTrue(
            hooks.prefill_wait_defers(self._scheduler(True, 0, chunked=True), 3)
        )

    def test_hook_falls_back_to_the_interval(self):
        from sglang.srt.disaggregation.utils import DisaggregationMode
        from sglang_tpu.hooks import k3_program as hooks

        original = mock.Mock(return_value=False)
        with mock.patch.object(hooks, "k3_program_active", return_value=True):
            with tpu_envs.SGLANG_TPU_K3_PROGRAM_PREFILL_WAIT_ROUNDS.override(0):
                self.assertFalse(
                    hooks.k3_program_prefill_wait(original, self._scheduler(True, 1))
                )
            with tpu_envs.SGLANG_TPU_K3_PROGRAM_PREFILL_WAIT_ROUNDS.override(2):
                self.assertTrue(
                    hooks.k3_program_prefill_wait(original, self._scheduler(True, 1))
                )
                prefill = self._scheduler(True, 1, mode=DisaggregationMode.PREFILL)
                self.assertFalse(hooks.k3_program_prefill_wait(original, prefill))
        self.assertEqual(original.call_count, 2)


class TestPrefillShare(CustomTestCase):
    """Colocated engines defer prefill by rounds sized from the batch."""

    def test_rounds_follow_the_batch_cost(self):
        rounds = worker.prefill_deferral_rounds
        self.assertEqual(rounds(16384, 0, 0.25), 0)  # nothing decodes
        self.assertEqual(rounds(0, 4, 0.25), 0)
        cold = rounds(16384, 8, 0.25)
        turn = rounds(2048, 8, 0.25)
        self.assertGreater(cold, 4 * turn)
        self.assertGreaterEqual(turn, 1)
        # A larger share defers less; a larger batch decodes slower rounds.
        self.assertLess(rounds(16384, 8, 0.5), cold)
        self.assertLess(rounds(16384, 16, 0.25), cold)
        # Prefill takes about the share of the engine's time.
        prefill_s = worker.PREFILL_STEP_S + worker.PREFILL_ROW_S * 16384
        decode_s = worker.DECODE_STEP_S + worker.DECODE_REQUEST_S * 8
        share = prefill_s / (prefill_s + cold * decode_s)
        self.assertAlmostEqual(share, 0.25, delta=0.01)

    def test_hook_sets_the_deferral(self):
        from sglang_tpu.hooks import k3_program as hooks

        def arm(scheduler, batch):
            scheduler._prefill_decode_interval_remaining = 16

        scheduler = SimpleNamespace(
            prefill_decode_interval=16,
            _prefill_decode_interval_remaining=0,
            running_batch=SimpleNamespace(reqs=[object()] * 8),
        )
        batch = SimpleNamespace(
            extend_num_tokens=16384,
            forward_mode=SimpleNamespace(is_extend=lambda: True),
        )
        with mock.patch.object(hooks, "k3_program_active", return_value=True):
            with tpu_envs.SGLANG_TPU_K3_PROGRAM_PREFILL_SHARE.override(0.0):
                hooks.k3_program_prefill_share(arm, scheduler, batch)
                self.assertEqual(scheduler._prefill_decode_interval_remaining, 16)
            with tpu_envs.SGLANG_TPU_K3_PROGRAM_PREFILL_SHARE.override(0.25):
                hooks.k3_program_prefill_share(arm, scheduler, batch)
                self.assertEqual(
                    scheduler._prefill_decode_interval_remaining,
                    worker.prefill_deferral_rounds(16384, 8, 0.25),
                )


class TestNoSpecStateReserve(CustomTestCase):
    """The state pool is sized without sglang's speculative SSM scratch."""

    def test_spec_reads_none_during_the_call_only(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang_tpu.hooks import k3_program as hooks

        configurator = SimpleNamespace(spec_algorithm="K3_PROGRAM")
        seen = []

        def handle(c, rest):
            seen.append(c.spec_algorithm)
            return rest - 1

        with mock.patch.object(hooks, "k3_program_active", return_value=True):
            self.assertEqual(
                hooks.k3_program_no_spec_state_reserve(handle, configurator, 10), 9
            )
        self.assertEqual(seen, [SpeculativeAlgorithm.NONE])
        self.assertEqual(configurator.spec_algorithm, "K3_PROGRAM")
        with mock.patch.object(hooks, "k3_program_active", return_value=False):
            hooks.k3_program_no_spec_state_reserve(handle, configurator, 10)
        self.assertEqual(seen[-1], "K3_PROGRAM")


class TestRegisteredAlgorithm(CustomTestCase):
    def setUp(self):
        worker.register()
        self.algo = SpeculativeAlgorithm.from_string(worker.ALGORITHM)

    def test_algorithm(self):
        self.assertIsInstance(self.algo, worker.K3ProgramAlgo)
        self.assertIs(
            self.algo.create_worker(SimpleNamespace(disable_overlap_schedule=True)),
            worker.K3ProgramWorker,
        )
        self.assertFalse(self.algo.need_topk())
        args = SimpleNamespace(kv_cache_dtype="fp8_e4m3", disaggregation_mode="decode")
        self.algo.handle_server_args(args)
        self.assertEqual(args.page_size, 16384)
        self.assertTrue(args.disable_overlap_schedule and args.disable_radix_cache)
        # A prefill keeps the radix cache: extra_buffer, page-sized chunks.
        prefill = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            disaggregation_mode="prefill",
            disable_radix_cache=False,
        )
        self.algo.handle_server_args(prefill)
        self.assertFalse(prefill.disable_radix_cache)
        self.assertEqual(prefill.mamba_radix_cache_strategy, "extra_buffer")
        self.assertEqual(
            (prefill.mamba_track_interval, prefill.chunked_prefill_size), (16384, 16384)
        )
        # Colocated (no disaggregation): the radix cache stays, decode never tracked.
        colocated = SimpleNamespace(
            kv_cache_dtype="fp8_e4m3",
            disaggregation_mode="null",
            disable_radix_cache=False,
        )
        self.algo.handle_server_args(colocated)
        self.assertFalse(colocated.disable_radix_cache)
        self.assertEqual(colocated.mamba_radix_cache_strategy, "extra_buffer")
        self.assertGreaterEqual(colocated.mamba_track_interval, 1 << 30)
        self.assertEqual(colocated.mamba_track_interval % 16384, 0)
        self.assertEqual(
            (
                args.speculative_num_steps,
                args.speculative_eagle_topk,
                args.speculative_num_draft_tokens,
            ),
            (7, 1, 8),
        )
        spec = self.algo.build_disagg_draft_input(None, None, None)
        self.assertIsInstance(spec, worker.K3ProgramInput)
        self.assertIsNone(spec.accept_lens_device)
        worker.register()  # idempotent

    def test_decode_prefix_cache_settings(self):
        with tpu_envs.SGLANG_TPU_K3_PROGRAM_ENABLE_DECODE_PREFIX_CACHE.override(True):
            args = SimpleNamespace(
                kv_cache_dtype="fp8_e4m3",
                disaggregation_mode="decode",
                disable_radix_cache=False,
            )
            self.algo.handle_server_args(args)
        self.assertFalse(args.disable_radix_cache)
        self.assertTrue(args.disaggregation_decode_enable_radix_cache)
        self.assertEqual(args.mamba_radix_cache_strategy, "extra_buffer")
        self.assertEqual(args.mamba_track_interval, 16384 << 16)  # never tracked

    def test_overlap_relays_through_the_future_map(self):
        published, stashed = [], []
        future_map = SimpleNamespace(
            publish=lambda i, lens: published.append((i.tolist(), lens.tolist())),
            stash=lambda i, payload: stashed.append(payload.bonus_tokens.tolist()),
        )
        batch = SimpleNamespace(
            enable_overlap=True,
            req_pool_indices=torch.tensor([3, 5]),
            seq_lens=torch.tensor([100, 200]),
        )
        spec = self.algo.build_disagg_draft_input(
            batch, torch.tensor([7, 8]), future_map
        )
        self.assertEqual(spec.future_indices.tolist(), [3, 5])
        self.assertEqual(published, [([3, 5], [100, 200])])
        self.assertEqual(stashed, [[7, 8]])
        # The indices follow the batch's requests.
        other = worker.K3ProgramInput()
        other.future_indices = torch.tensor([9])
        spec.merge_batch(other)
        self.assertEqual(spec.future_indices.tolist(), [3, 5, 9])
        spec.filter_batch(torch.tensor([0, 2]))
        self.assertEqual(spec.future_indices.tolist(), [3, 9])
        synchronous = worker.K3ProgramInput()
        synchronous.merge_batch(worker.K3ProgramInput())
        self.assertIsNone(synchronous.future_indices)
        with self.assertRaisesRegex(AssertionError, "without future indices"):
            spec.merge_batch(worker.K3ProgramInput())
        # The fence's tensor: the copied tokens follow the step's last KV write.
        self.assertTrue(self.algo.supports_kv_free_fence(copies_draft_tail=False))

    def test_overlap_on_the_decoding_roles(self):
        with tpu_envs.SGLANG_TPU_K3_PROGRAM_OPT_OVERLAP.override(True):
            for role, overlapped in (
                ("decode", True),
                ("prefill", False),
                ("null", True),  # colocated
            ):
                args = SimpleNamespace(
                    kv_cache_dtype="fp8_e4m3",
                    disaggregation_mode=role,
                    disable_radix_cache=role == "decode",
                )
                self.algo.handle_server_args(args)
                self.assertEqual(args.disable_overlap_schedule, not overlapped, role)
        self.assertTrue(self.algo.supports_overlap)

    def test_settings_through_sglangs_speculative_hook(self):
        server_args = SimpleNamespace(
            speculative_draft_model_path=None,
            speculative_draft_model_revision=None,
            speculative_moe_runner_backend=None,
            moe_runner_backend="auto",
            speculative_algorithm="k3_program",
            decrypted_draft_config_file=None,
            trust_remote_code=False,
            speculative_draft_window_size=None,
            speculative_draft_sink_size=None,
            speculative_skip_dp_mlp_sync=False,
            speculative_adaptive=False,
            kv_cache_dtype="auto",
            disaggregation_mode="decode",
        )
        handle_speculative_decoding(server_args)
        self.assertEqual(
            (
                server_args.page_size,
                server_args.speculative_num_draft_tokens,
                server_args.disable_overlap_schedule,
                server_args.disable_radix_cache,
            ),
            (8192, 8, True, True),
        )

    def test_worker_result(self):
        plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("auto"))
        program = StubProgram()
        program.accepted = {"a": 3, "b": 1}
        r2t = torch.from_numpy(_req_to_token(3, {1: [1], 2: [2]}))
        target = SimpleNamespace(
            model_runner=None,
            get_memory_pool=lambda: (SimpleNamespace(req_to_token=r2t), None),
        )
        worker.K3ProgramWorker.build_steps = lambda target_worker: worker.DecodeSteps(
            program, plan, table_width=4, schedule=SCHEDULE
        )
        self.addCleanup(setattr, worker.K3ProgramWorker, "build_steps", None)
        w = worker.K3ProgramWorker(None, 0, 0, target)
        w.alloc_memory_pool()
        self.assertIsNone(w.draft_worker)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            reqs=[
                _req("a", prompt=10, output=[1], committed=10, pool_idx=1, slot=1),
                _req("b", prompt=20, output=[1], committed=20, pool_idx=2, slot=2),
            ],
            seq_lens=torch.tensor([10, 20]),
            return_logprob=False,
        )
        result = w.forward_batch_generation(batch)
        self.assertEqual(result.accept_lens.tolist(), [3, 1])
        self.assertEqual(result.new_seq_lens.tolist(), [13, 21])
        self.assertEqual(result.speculative_output_stride, 8)
        tokens = result.next_token_ids.view(2, 8)
        self.assertEqual(tokens[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(tokens[1, :1].tolist(), [100])
        self.assertIsInstance(result.next_draft_input, worker.K3ProgramInput)
        # sglang's output processor reads it as each request's accepted run.
        processor = SimpleNamespace(
            model_worker=w, advance_grammar_fsm=lambda result, batch: None
        )
        runs = SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
            processor, result, batch
        )
        self.assertEqual(runs, [[0, 1, 2], [100]])
        self.assertEqual([r.kv.kv_committed_len for r in batch.reqs], [13, 21])
        self.assertEqual(result.num_correct_drafts_per_req_cpu, [2, 0])
        batch.reqs[0].grammar = object()
        with self.assertRaisesRegex(NotImplementedError, "structured outputs"):
            w.forward_batch_generation(batch)
        batch.reqs[0].grammar = None
        batch.forward_mode = ForwardMode.EXTEND
        with self.assertRaisesRegex(NotImplementedError, "decode role runs no"):
            w.forward_batch_generation(batch)

    def test_overlapped_worker_result(self):
        plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("auto"))
        program = StubProgram()
        program.accepted = {"a": 3, "b": 1}
        r2t = torch.from_numpy(_req_to_token(3, {1: [1], 2: [2]}))
        target = SimpleNamespace(
            model_runner=None,
            get_memory_pool=lambda: (SimpleNamespace(req_to_token=r2t), None),
        )
        worker.K3ProgramWorker.build_steps = lambda target_worker: worker.DecodeSteps(
            program, plan, table_width=4, schedule=SCHEDULE
        )
        self.addCleanup(setattr, worker.K3ProgramWorker, "build_steps", None)
        w = worker.K3ProgramWorker(None, 0, 0, target)
        w.alloc_memory_pool()
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            reqs=[
                _req("a", prompt=10, output=[1], committed=10, pool_idx=1, slot=1),
                _req("b", prompt=20, output=[1], committed=20, pool_idx=2, slot=2),
            ],
            seq_lens=torch.tensor([10, 20]),
            return_logprob=False,
        )
        published = []
        result = w.forward_batch_generation(batch, on_publish=published.append)
        self.assertEqual(result.accept_lens.tolist(), [3, 1])
        self.assertEqual(result.new_seq_lens.tolist(), [13, 21])
        self.assertEqual(published[0].tolist(), [13, 21])
        self.assertEqual(result.next_token_ids.view(2, 8)[0, :3].tolist(), [0, 1, 2])
        # The future map relays each request's last token.
        self.assertEqual(result.next_draft_input.bonus_tokens.tolist(), [2, 100])

    def test_prefill_worker_result(self):
        plan = bridge.UnifiedPoolPlan(40, 4, pools.cache_layout("auto"))
        r2t = torch.from_numpy(_req_to_token(3, {1: [1], 2: [2]}))
        target = SimpleNamespace(
            model_runner=None,
            get_memory_pool=lambda: (
                SimpleNamespace(req_to_token=r2t, mamba_pool=None),
                None,
            ),
        )
        worker.K3ProgramWorker.build_steps = lambda target_worker: worker.PrefillSteps(
            StubPrefill(), plan, table_width=4
        )
        self.addCleanup(setattr, worker.K3ProgramWorker, "build_steps", None)
        w = worker.K3ProgramWorker(None, 0, 0, target)
        w.alloc_memory_pool()
        seq_lens = torch.tensor([100, 4000])
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            reqs=[
                _req("a", prompt=100, output=[], committed=0, pool_idx=1, slot=1),
                _req("b", prompt=9000, output=[], committed=0, pool_idx=2, slot=2),
            ],
            prefix_lens=[0, 0],
            extend_lens=[100, 4000],
            seq_lens=seq_lens,
            return_logprob=False,
        )
        result = w.forward_batch_generation(batch)
        self.assertEqual(result.next_token_ids.tolist(), [50, 0])
        self.assertIs(result.new_seq_lens, seq_lens)
        self.assertIsInstance(result.next_draft_input, worker.K3ProgramInput)
        batch.forward_mode = ForwardMode.DECODE
        with self.assertRaisesRegex(NotImplementedError, "prefill role runs no"):
            w.forward_batch_generation(batch)


if __name__ == "__main__":
    unittest.main()

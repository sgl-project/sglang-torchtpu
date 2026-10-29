"""The Kimi-K3 decode program as sglang's K3_PROGRAM speculative worker, over a
stub program: the rows each step hands it, finished requests, the result the
output processor reads, and the registered algorithm."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.k3_programs import sglang_bridge as bridge
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
        flat, lens = self.steps.step([req], r2t)
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
        self.steps.step([a, b, c], r2t)
        self.assertEqual(set(self.program.inputs[0].finished), set())
        self.steps.step([a, c], r2t)
        self.assertEqual(set(self.program.inputs[1].finished), {"b"})
        self.steps.forget_all()
        self.assertEqual(set(self.program.inputs[2].finished), {"a", "c"})
        self.assertEqual(list(self.program.inputs[2].requests), [])

    def test_wide_batches_step_plain(self):
        reqs = [
            _req(str(i), prompt=10, output=[1], committed=10, pool_idx=1, slot=1)
            for i in range(9)
        ]
        self.steps.step(reqs, _req_to_token(2, {1: [1]}))
        self.assertEqual({r.scheduled for r in self.program.inputs[0].requests}, {1})

    def test_a_request_without_a_token_is_an_error(self):
        req = _req("a", prompt=10, output=[1], committed=10, pool_idx=1, slot=1)
        self.program.accepted = {"a": 0}
        with self.assertRaisesRegex(RuntimeError, r"\['a'\] took no token"):
            self.steps.step([req], _req_to_token(2, {1: [1]}))


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
        args = SimpleNamespace()
        self.algo.handle_server_args(args)
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
        with self.assertRaisesRegex(NotImplementedError, "decode role"):
            w.forward_batch_generation(batch)


if __name__ == "__main__":
    unittest.main()

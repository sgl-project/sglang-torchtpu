"""Draft steps chosen per decode step from the running-request count (adaptive_steps)."""

import collections
import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.attention_gdn_pallas import TpuGDNAttnBackend
from sglang_tpu.attention_pallas_rpa_base import BaseTpuPallasRpaAttnBackend
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.experimental.compilation.chain_compile_policy import (
    TpuChainSpecDecodeCompilePolicyMixin,
)
from sglang_tpu.experimental.speculative import (
    base_spec_worker,
    spec_ops,
)
from sglang_tpu.experimental.speculative.adaptive_steps import (
    check_draft_step_schedule,
    parse_acc_len_by_steps,
    parse_steps_by_running,
)
from sglang_tpu.experimental.speculative.chain_draft_input import (
    chain_draft_forward_input,
)
from sglang_tpu.experimental.speculative.chain_draft_worker import (
    TpuChainDraftWorker,
)
from sglang_tpu.experimental.speculative.chain_spec_worker import (
    TpuChainSpecWorker,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

# The schedule qinghan asked for: K=7 up to 4 running, K=3 above.
SCHEDULE = "1:7,5:3"
ACC_LENS = "3:3.00,7:3.93"
SPARSE_BUCKETS = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32]


def _scheduled_worker(buckets, *, acc_lens=None) -> TpuChainSpecWorker:
    """A chain worker launched at K=7 with SCHEDULE, carrying only what the tested methods read."""
    worker = object.__new__(TpuChainSpecWorker)
    worker.device = "cpu"
    worker.num_drafts, worker.num_verify_rows = 7, 8
    worker._max_drafts = 7
    worker._step_schedule = parse_steps_by_running(SCHEDULE)
    worker._simulate_acc_len = 3.93 if acc_lens else -1.0
    worker._simulate_acc_len_by_steps = acc_lens or {}
    worker._draft_steps_ct = collections.Counter()
    worker._logs_draft_steps = False
    worker._dp_attention = False
    bucketer = SimpleNamespace(
        buckets=buckets,
        max_val=max(buckets),
        next_bucket=lambda n: min(b for b in buckets if b >= n),
    )
    worker._b_bucketer = lambda: bucketer
    return worker


class TestDraftStepSchedule(CustomTestCase):
    def test_steps_follow_the_running_count(self):
        schedule = parse_steps_by_running(" 5:3 , 1:7 ")
        self.assertEqual([schedule.steps_for(n) for n in (1, 4, 5, 32)], [7, 7, 3, 3])
        self.assertEqual(schedule.widths(), (4, 8))
        self.assertEqual(schedule.max_steps, 7)
        self.assertIsNone(parse_steps_by_running(None))
        self.assertIsNone(parse_steps_by_running(""))

    def test_rejects_schedules_without_a_first_tier(self):
        for text in ("2:7,5:3", "1:7,1:3", "1:0", "1-7", ","):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_steps_by_running(text)

    def test_bucket_gets_the_widths_of_the_sizes_it_pads(self):
        schedule = parse_steps_by_running(SCHEDULE)
        widths = {
            b: schedule.widths_for_bucket(b, SPARSE_BUCKETS) for b in SPARSE_BUCKETS
        }
        self.assertEqual(widths[1], (8,))
        self.assertEqual(widths[4], (8,))
        # bs 5..8 pad to 8: K=3 only, so the threshold adds no program.
        self.assertEqual(widths[8], (4,))
        self.assertEqual(widths[32], (4,))
        # A tier boundary inside a bucket's range warms both widths there.
        straddle = parse_steps_by_running("1:7,6:3")
        self.assertEqual(straddle.widths_for_bucket(8, [4, 8]), (4, 8))
        self.assertEqual(straddle.widths_for_bucket(4, [4, 8]), (8,))

    def test_acceptance_length_per_step_count(self):
        schedule = parse_steps_by_running(SCHEDULE)
        self.assertEqual(parse_acc_len_by_steps(ACC_LENS), {3: 3.0, 7: 3.93})
        check = lambda **kw: check_draft_step_schedule(  # noqa: E731
            schedule,
            **{
                "launch_steps": 7,
                "simulate_acc_len": 3.93,
                "dp_attention": False,
                **kw,
            },
        )
        with tpu_envs.SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS.override(ACC_LENS):
            self.assertEqual(check(), {7: 3.93, 3: 3.0})
            self.assertEqual(check(simulate_acc_len=-1.0), {})
            # The launch count must be the largest scheduled one: it sizes the window.
            with self.assertRaisesRegex(ValueError, "speculative-num-steps 7"):
                check(launch_steps=3)
            with self.assertRaisesRegex(ValueError, "differs"):
                check(simulate_acc_len=3.5)
            with self.assertRaises(NotImplementedError):
                check(dp_attention=True)
        with tpu_envs.SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS.override("7:3.93"):
            with self.assertRaisesRegex(ValueError, r"steps \[3\]"):
                check()
        with tpu_envs.SGLANG_TPU_SPEC_SIMULATE_ACC_LEN_BY_STEPS.override(
            "3:4.5,7:3.93"
        ):
            with self.assertRaisesRegex(ValueError, "Acceptance length"):
                check()


class TestWorkerSwitchesDraftSteps(CustomTestCase):
    def test_each_step_takes_its_draft_count_and_length(self):
        worker = _scheduled_worker(SPARSE_BUCKETS, acc_lens={7: 3.93, 3: 3.0})
        for bs, steps, acc_len in (
            (1, 7, 3.93),
            (9, 3, 3.0),
            (4, 7, 3.93),
            (5, 3, 3.0),
        ):
            worker._activate_draft_steps(bs)
            self.assertEqual(
                (worker.num_drafts, worker.num_verify_rows), (steps, steps + 1)
            )
            self.assertEqual(worker._simulate_acc_len, acc_len)
        self.assertEqual(worker._draft_steps_ct, {7: 2, 3: 2})

    def test_simulated_accept_follows_the_active_length(self):
        worker = _scheduled_worker([1, 2], acc_lens={7: 3.93, 3: 3.0})
        worker._simulated_correct = {
            (pbs, n): torch.full((pbs,), n) for pbs in (1, 2) for n in range(8)
        }
        worker._simulated_bonus = {pbs: torch.zeros(pbs) for pbs in (1, 2)}
        worker._simulated_drafts = {
            (pbs, n): torch.zeros((pbs, n)) for pbs in (1, 2) for n in (3, 7)
        }
        torch.manual_seed(0)
        for bs, acc_len in ((1, 3.93), (5, 3.0)):
            worker._activate_draft_steps(bs)
            lens = []
            for _ in range(2000):
                correct, _, drafts = worker._simulated_accept(2)
                self.assertEqual(drafts.shape, (2, worker.num_drafts))
                lens.append(int(correct[0]) + 1)
            self.assertAlmostEqual(sum(lens) / len(lens), acc_len, delta=0.03)
            self.assertLessEqual(max(lens), worker.num_verify_rows)

    def test_warmup_pairs_are_the_reachable_ones(self):
        worker = _scheduled_worker(SPARSE_BUCKETS)
        pairs = worker._bucket_widths(SPARSE_BUCKETS)
        self.assertEqual(
            set(pairs),
            {(8, b) for b in (1, 2, 4)} | {(4, b) for b in SPARSE_BUCKETS[3:]},
        )
        self.assertEqual(worker._live_bucket_widths(3), [(8, 4)])
        self.assertEqual(worker._live_bucket_widths(7), [(4, 8)])
        self.assertEqual(worker.verify_widths(), (4, 8))

    def test_verify_drafts_tensor_per_count(self):
        worker = _scheduled_worker([1, 2])
        worker._all_drafts = {pbs: torch.full((pbs,), 7) for pbs in (1, 2)}
        worker._short_drafts = {(2, 3): torch.full((2,), 3)}
        self.assertEqual(worker._num_verify_drafts(2, 7).tolist(), [7, 7])
        self.assertEqual(worker._num_verify_drafts(2, 3).tolist(), [3, 3])

    def test_selector_warmup_covers_each_bucket_at_its_width(self):
        worker = _scheduled_worker([1, 2, 4, 8])
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(max_running_requests=8)
        )
        worker._chain_draft_worker = SimpleNamespace(carry_size=3)
        worker._draft_runner = SimpleNamespace(dtype=torch.float32)
        worker._rng_step = torch.zeros(1, dtype=torch.int64)
        selections = []

        def warmup(op, *args):
            if op is spec_ops.select_verify_rows:
                selections.append((tuple(args[0].shape), args[2]))
            return op(*args)

        with (
            patch.object(
                base_spec_worker,
                "eager_warmup",
                lambda: contextlib.nullcontext(lambda t: None),
            ),
            patch.object(
                base_spec_worker, "max_speculative_num_draft_tokens", return_value=8
            ),
            patch.object(base_spec_worker, "warmup_compiled_op", side_effect=warmup),
        ):
            worker._warmup_logical_bs_ops()
        self.assertEqual(
            set(selections), {((b, 17), 8) for b in (1, 2, 4)} | {((8, 17), 4)}
        )


class _IdentityDrafter(TpuChainDraftWorker):
    def first_draft_logits(self, carry):
        return carry

    def target_tokens(self, draft_ids):
        return draft_ids

    def target_logits(self, draft_logits):
        return draft_logits


class TestFusedDraftLength(CustomTestCase):
    def test_draft_count_comes_from_the_verify_rows(self):
        calls = []

        def forward(input_ids, positions, forward_batch):
            calls.append(tuple(positions.shape))
            logits = torch.zeros((input_ids.shape[0], 4))
            return SimpleNamespace(next_token_logits=logits, hidden_states=logits)

        drafter = object.__new__(_IdentityDrafter)
        drafter.num_drafts = 7
        drafter.model_runner = SimpleNamespace(model=SimpleNamespace(forward=forward))
        for width, sampled in ((8, False), (4, False), (4, True)):
            calls.clear()
            forward_batch = SimpleNamespace(
                spec_info=chain_draft_forward_input(
                    hidden_states=torch.zeros((2, 4)), width=1
                ),
                reuse_dsa_topk_indices=False,
            )
            tokens, dense, _ = drafter._fused_draft_steps(
                forward_batch,
                torch.zeros((3, 4)),
                torch.tensor([1, 2]),
                torch.zeros((2, width), dtype=torch.int64),
                torch.zeros((2, width), dtype=torch.int64),
                torch.zeros(2, dtype=torch.int64),
                torch.ones(2) if sampled else None,
                torch.full((2,), 4, dtype=torch.int32) if sampled else None,
                torch.zeros(1, dtype=torch.int64) if sampled else None,
                None,
            )
            self.assertEqual(tokens.shape, (2, width - 1))
            # Sampled drafts draw one uniform per (row, draft) at the active count.
            self.assertEqual(
                None if dense is None else dense.shape,
                (2, width - 1, 4) if sampled else None,
            )
            self.assertEqual(len(calls), width - 2)


class _Policy(TpuChainSpecDecodeCompilePolicyMixin):
    pass


class TestScheduledPrecompile(CustomTestCase):
    def _policy(self, buckets):
        policy = object.__new__(_Policy)
        policy.b_bucketer = SimpleNamespace(buckets=buckets)
        policy.draft_token_num = 8
        policy.model_runner = SimpleNamespace(
            device="cpu", spec_algorithm=SimpleNamespace(name="EAGLE")
        )
        return policy

    def test_target_and_draft_programs_per_reachable_pair(self):
        policy = self._policy([1, 2, 4, 8, 12])
        with tpu_envs.SGLANG_TPU_SPEC_STEPS_BY_RUNNING.override(SCHEDULE):
            shapes = policy._target_verify_shapes()
            fused = []
            with patch.object(
                _Policy,
                "_precompile_shapes",
                lambda self, desc, items, run: [run(i) for i in items],
            ):
                policy._precompile_fused_draft(
                    lambda bs, sampled, width: fused.append((bs, sampled, width))
                )
        expected = [(8, 1), (8, 2), (8, 4), (4, 8), (4, 12)]
        self.assertEqual(shapes, expected)
        self.assertEqual(
            sorted(fused),
            sorted((bs, s, w) for w, bs in expected for s in (False, True)),
        )
        spec_info = policy._verify_spec_info(4)
        self.assertEqual(
            (spec_info.draft_token_num, spec_info.num_tokens_per_req), (4, 4)
        )
        # Without a schedule every bucket keeps the launch width.
        self.assertEqual(policy._target_verify_shapes()[-1], (8, 12))

    def test_fused_draft_specialization_keys_on_draft_count(self):
        policy = self._policy([4])
        policy.tracer = "direct"
        captured = {}

        def direct_compile(trace, specialization_key, model_extra_tensor_attrs):
            captured["key"] = specialization_key
            return trace

        with patch.object(
            _Policy, "_direct_compile_draft", side_effect=direct_compile, autospec=False
        ):
            policy.compile_fused_draft(lambda *a: None)
        key = captured["key"]
        forward_batch = SimpleNamespace(batch_size=4)

        def args(width):
            # (carry, req_pool_indices, positions, cache_locs, seq_lens, temperatures, ...)
            return (None, None, torch.zeros((4, width)), None, None, None, None, None)

        self.assertNotEqual(
            key(None, forward_batch, *args(8)), key(None, forward_batch, *args(4))
        )


class TestBackendsTakeScheduledWidths(CustomTestCase):
    def test_rpa_verify_runs_static_at_every_scheduled_width(self):
        backend = object.__new__(BaseTpuPallasRpaAttnBackend)
        backend.decode_query_size = 8

        def verify(width):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=SimpleNamespace(num_tokens_per_req=width),
            )
            return backend._forward_decode_query_size(batch)

        self.assertEqual([verify(8), verify(4)], [8, 1])
        backend._extra_query_sizes = frozenset({4, 8})
        self.assertEqual([verify(8), verify(4), verify(5)], [8, 4, 1])

    def test_linear_attention_verify_writes_its_own_width(self):
        backend = object.__new__(TpuGDNAttnBackend)
        backend._spec_window = 8
        backend._read_offsets_pool = torch.zeros(4, dtype=torch.int32)
        for width in (4, 8):
            batch = SimpleNamespace(
                forward_mode=ForwardMode.TARGET_VERIFY,
                spec_info=SimpleNamespace(num_tokens_per_req=width),
                req_pool_indices=torch.tensor([1, 2]),
            )
            backend._init_spec_metadata(batch)
            self.assertEqual(backend._num_spec_tokens, width - 1)
        batch.spec_info.num_tokens_per_req = 9
        with self.assertRaises(AssertionError):
            backend._init_spec_metadata(batch)


class TestSpecOpPrecompile(CustomTestCase):
    def test_spec_ops_warm_each_bucket_at_its_widths(self):
        from sglang_tpu import spec_op_precompile

        contexts = []
        op = SimpleNamespace(
            make_warmup_args=lambda ctx: (
                contexts.append((ctx.physical_bs, ctx.num_verify_tokens)) or ()
            ),
            op=lambda *a: None,
        )
        with (
            tpu_envs.SGLANG_TPU_SPEC_STEPS_BY_RUNNING.override(SCHEDULE),
            patch.object(spec_op_precompile, "get_compiled_ops", return_value=(op,)),
            patch.object(spec_op_precompile, "snapshot_recompilation_baseline"),
        ):
            spec_op_precompile.precompile_spec_v2_ops(
                device=torch.device("cpu"),
                physical_batch_buckets=[1, 2, 4, 8],
                num_verify_tokens=8,
                vocab_size=8,
                draft_vocab_size=8,
                hidden_size=4,
                hidden_dtype=torch.float32,
            )
        self.assertEqual(contexts, [(1, 8), (2, 8), (4, 8), (8, 4)])


if __name__ == "__main__":
    unittest.main()

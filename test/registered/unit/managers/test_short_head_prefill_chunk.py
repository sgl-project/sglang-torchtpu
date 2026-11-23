import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.disaggregation.mode import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.runtime_context import get_context

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SHORT, FULL, LIMIT = 4096, 8192, 6144


def _req(tokens, prefilled=0):
    return SimpleNamespace(
        origin_input_ids=[0] * tokens, output_ids=[], prefix_len=prefilled
    )


def _envs(short, limit):
    return (
        envs.SGLANG_SHORT_HEAD_PREFILL_CHUNK_SIZE.override(short),
        envs.SGLANG_SHORT_HEAD_PREFILL_MAX_TOKENS.override(limit),
    )


class TestShortHeadPrefillChunk(unittest.TestCase):
    def _size(self, chunked_req=None, waiting=(), short=SHORT, full=FULL, limit=0):
        s = Scheduler.__new__(Scheduler)
        s.chunked_reqs = [] if chunked_req is None else [chunked_req]
        s.waiting_queue = list(waiting)
        a, b = _envs(short, limit)
        with a, b:
            return s._short_head_chunk_size(full)

    # Guards: the cap only ever shrinks the chunk, and only when it is on.
    def test_off_by_default(self):
        self.assertEqual(self._size(waiting=[_req(1024)], short=0), FULL)

    def test_short_chunk_not_below_full_chunk_is_ignored(self):
        self.assertEqual(self._size(waiting=[_req(1024)], full=4096), 4096)
        self.assertEqual(self._size(waiting=[_req(1024)], full=2048), 2048)

    def test_no_chunking_is_left_alone(self):
        self.assertIsNone(self._size(waiting=[_req(1024)], full=None))

    def test_no_head_request_keeps_the_full_chunk(self):
        self.assertEqual(self._size(waiting=[]), FULL)

    def test_limit_below_short_chunk_acts_as_the_short_chunk(self):
        for limit in (0, 1024, SHORT):
            with self.subTest(limit=limit):
                self.assertEqual(self._size(waiting=[_req(SHORT)], limit=limit), SHORT)
                self.assertEqual(
                    self._size(waiting=[_req(SHORT + 1)], limit=limit), FULL
                )

    # Boundaries around the short chunk size (another request waiting).
    def test_short_chunk_boundary_with_others_waiting(self):
        for tokens, want in ((SHORT - 1, SHORT), (SHORT, SHORT), (SHORT + 1, FULL)):
            with self.subTest(tokens=tokens):
                got = self._size(waiting=[_req(tokens), _req(1024)], limit=LIMIT)
                self.assertEqual(got, want)

    # Boundaries around the lone-request limit (nothing else waiting).
    def test_lone_request_limit_boundary(self):
        for tokens, want in ((LIMIT - 1, SHORT), (LIMIT, SHORT), (LIMIT + 1, FULL)):
            with self.subTest(tokens=tokens):
                self.assertEqual(self._size(waiting=[_req(tokens)], limit=LIMIT), want)

    def test_lone_request_limit_ignored_while_others_wait(self):
        for tokens in (SHORT + 1, LIMIT):
            with self.subTest(tokens=tokens):
                got = self._size(waiting=[_req(tokens), _req(1024)], limit=LIMIT)
                self.assertEqual(got, FULL)

    # Mixed short/long batches: the head request alone decides.
    def test_short_head_before_long_prompts(self):
        self.assertEqual(
            self._size(waiting=[_req(1024), _req(8192), _req(8192)], limit=LIMIT),
            SHORT,
        )

    def test_long_head_before_short_prompts(self):
        self.assertEqual(
            self._size(waiting=[_req(8192)] + [_req(1024)] * 7, limit=LIMIT), FULL
        )

    def test_continuing_chunk_is_the_head(self):
        # 3072 of 8192 done: 5120 left, so the full chunk behind a short prompt.
        self.assertEqual(self._size(_req(8192, prefilled=3072), [_req(1024)]), FULL)
        # 6144 done: 2048 left, so the short chunk even behind a long prompt.
        self.assertEqual(self._size(_req(8192, prefilled=6144), [_req(8192)]), SHORT)
        # A lone continuing request gets the lone limit: 12288 - 8192 = 4096,
        # 8192 - 3072 = 5120 <= 6144, 8192 - 2047 = 6145 > 6144.
        self.assertEqual(self._size(_req(12288, prefilled=8192), limit=LIMIT), SHORT)
        self.assertEqual(self._size(_req(8192, prefilled=3072), limit=LIMIT), SHORT)
        self.assertEqual(self._size(_req(8192, prefilled=2047), limit=LIMIT), FULL)

    def test_cached_prefix_counts_as_done(self):
        # An 8192-token prompt with a 5120-token cached prefix has 3072 left.
        self.assertEqual(
            self._size(waiting=[_req(8192, prefilled=5120), _req(8192)]), SHORT
        )


class _Captured(Exception):
    pass


class _Adder:
    """Stops the batch build at the adder and reports the chunk it was given."""

    def __init__(self, *args, **kwargs):
        raise _Captured(args[6])


class _Queue(list):
    """The waiting queue, keeping the order it was given."""

    def reorder(self, items):
        pass


class TestShortHeadWiring(unittest.TestCase):
    """The batch builder hands the capped chunk to the prefill adder."""

    def _adder_chunk(self, waiting):
        s = Scheduler.__new__(Scheduler)
        s.grammar_manager = SimpleNamespace(has_waiting_grammars=lambda: False)
        s.enable_priority_preemption = False
        s.is_hybrid_swa = False
        s.waiting_queue = _Queue(waiting)
        s.disagg_prefill_bootstrap_queue = None
        s.chunked_reqs = []
        s.min_free_slots_delayer = None
        s.policy = Mock()
        s.policy.shortest_prefill_chunk_limit.return_value = None
        s.processed_tokens_counter = 0
        s.chunked_prefill_size = FULL
        s.dynamic_chunk_sizer = None
        s.tp_worker = SimpleNamespace(
            model_runner=SimpleNamespace(attn_backend=SimpleNamespace())
        )
        s._tpu_pp_chunk_cost = None
        s.page_size = 1
        s.tree_cache = SimpleNamespace(
            buffer_pipeline=None, storage_prefetch_retries=None
        )
        s.token_to_kv_pool_allocator = None
        s.new_token_ratio_tracker = SimpleNamespace(current=0.0)
        s.max_prefill_tokens = 1 << 20
        s.is_mixed_chunk = False
        s.priority_scheduling_preemption_threshold = 0
        s.max_prefill_bs = 0
        s.max_running_requests = 64
        s.dllm_config = None
        s.enable_lora = False
        s.disaggregation_mode = DisaggregationMode.PREFILL
        s.req_to_token_pool = SimpleNamespace(available_size=lambda: 64)
        s.beam_coordinator = SimpleNamespace(
            pending_member_rows=lambda batch: 0, pending_member_rows_of=lambda reqs: 0
        )
        s.request_retirer = None
        s.disagg_prefill_transfer_queue = None
        s.get_prefill_adder_class = lambda: _Adder
        a, b = _envs(SHORT, LIMIT)
        with (
            a,
            b,
            get_context().override_server_args(pp_max_micro_batch_size=64),
        ):
            with self.assertRaises(_Captured) as captured:
                s._get_new_batch_prefill_raw(
                    prefill_delayer_single_pass=None,
                    running_batch=ScheduleBatch(reqs=[]),
                )
        return captured.exception.args[0]

    def test_short_head_reaches_the_adder(self):
        self.assertEqual(self._adder_chunk([_req(1024), _req(8192)]), SHORT)

    def test_long_head_reaches_the_adder(self):
        self.assertEqual(self._adder_chunk([_req(8192), _req(1024)]), FULL)


if __name__ == "__main__":
    unittest.main()

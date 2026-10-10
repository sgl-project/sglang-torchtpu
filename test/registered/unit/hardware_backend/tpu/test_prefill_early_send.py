"""TPU prefill early send never reads KV pages the in-flight forward writes.

Under overlap scheduling, ``run_batch`` for step n+1 runs while step n's
forward may still be executing on device. A chunked request's step-n chunk is
already in the radix tree by then (and any new request may have matched it),
so the cached-prefix early send would read pages that are still being written.
On TPU that forward has to finish before its KV cache can be read from HBM, so
the early send is clipped to the pages step n does not write, and the rest
goes out with the request's next send.

A HiCache load-back has the same problem within one batch, with or without
overlap: its pages enter the radix tree when it is queued, and a request
admitted later in the same pass matches them as device-resident. The batch's
forward waits for the load, but the early send runs before the forward, so it
stops at those pages too.
"""

import contextlib
import unittest
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.disaggregation import prefill
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.prefill_early_send import (
    _collect_prefill_write_page_ids,
    _find_safe_prefix_send_end,
    tpu_clip_cached_prefix_send_end,
    tpu_clip_hicache_load_send_end,
    tpu_hicache_load_page_ids,
    tpu_update_inflight_kv_write_pages,
)

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PAGE_SIZE = 4


@contextlib.contextmanager
def _platform(*, is_tpu: bool, early_send: bool = True, update_write_pages=None):
    """prefill.py as it is imported on TPU (or not): it binds the TPU helpers
    only under ``_is_tpu``, which is False on the CPU runners."""
    with (
        patch.object(prefill, "_is_tpu", is_tpu),
        patch.object(prefill, "_is_cuda", False),
        patch.object(
            prefill,
            "tpu_clip_cached_prefix_send_end",
            tpu_clip_cached_prefix_send_end,
            create=True,
        ),
        patch.object(
            prefill,
            "tpu_clip_hicache_load_send_end",
            tpu_clip_hicache_load_send_end,
            create=True,
        ),
        patch.object(
            prefill,
            "tpu_update_inflight_kv_write_pages",
            update_write_pages or tpu_update_inflight_kv_write_pages,
            create=True,
        ),
        envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.override(early_send),
    ):
        yield


class _StopLoop(Exception):
    pass


class _Range:
    def __init__(self, start: int, end: int):
        self.start, self.end = start, end


class _Device:
    """Models the TPU: pages written by the launched forward stay in flight
    until the scheduler processes that batch's result."""

    def __init__(self):
        self.inflight: Set[int] = set()

    def launch(self, pages: Set[int]) -> None:
        self.inflight = set(pages)

    def land(self) -> None:
        self.inflight = set()


class _Sender:
    """Records every KV send with the pages that were in flight at read time."""

    def __init__(
        self, device: _Device, rid: str, log: List[Tuple[str, List[int], Set[int]]]
    ):
        self.device, self.rid, self.log = device, rid, log

    def get_max_transfer_tokens(self) -> Optional[int]:
        return None

    def should_send_kv_chunk(self, num_pages: int, last_chunk: bool) -> bool:
        return num_pages > 0

    def send(self, page_indices, state_indices=None, num_kv_tokens=None):
        self.log.append(
            (self.rid, [int(p) for p in page_indices], set(self.device.inflight))
        )


class _Scheduler(SchedulerDisaggregationPrefillMixin):
    def __init__(self, *, enable_overlap: bool, num_slots: int = 64):
        self.enable_overlap = enable_overlap
        self.enable_staging = False
        self.gracefully_exit = False
        self.kv_checksum_computer = None
        self.kv_free_fence = None
        self._deferred_input_requests = []
        self.token_to_kv_pool_allocator = SimpleNamespace(
            page_size=PAGE_SIZE, translate_kv_indices_for_transfer=lambda kv: kv
        )
        self.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((8, num_slots), dtype=torch.int32)
        )
        self.tpu_inflight_kv_write_page_ids: Optional[torch.Tensor] = None


def _req(rid: str, pool_idx: int, sender, input_len: int, prefix_indices=None):
    return SimpleNamespace(
        rid=rid,
        kv=SimpleNamespace(req_pool_idx=pool_idx),
        origin_input_ids=list(range(input_len)),
        prefix_indices=(
            torch.empty(0, dtype=torch.int64)
            if prefix_indices is None
            else prefix_indices
        ),
        host_hit_length=0,
        start_send_idx=0,
        pending_bootstrap=False,
        early_send_prefix_end=None,
        extend_range=None,
        disagg_kv_sender=sender,
    )


def _slots(pages: List[int]) -> torch.Tensor:
    return torch.tensor(
        [p * PAGE_SIZE + o for p in pages for o in range(PAGE_SIZE)], dtype=torch.int64
    )


class _Harness:
    """Drives the pieces of the overlap prefill loop that touch KV sends, in
    the loop's order:  run_batch (early send, forward launch)  ->
    process_batch_result of the previous batch (its regular chunk send)  ->
    tpu_update_inflight_kv_write_pages (TPU only)."""

    def __init__(self, scheduler: _Scheduler):
        self.s = scheduler
        self.device = _Device()
        self.log: List[Tuple[str, List[int], Set[int]]] = []
        # Requests of the launched batch and the end of the chunk each one is
        # computing: process_prefill_chunk records it as tmp_end_idx before the
        # next batch moves extend_range on.
        self.pending: List[Tuple[object, int]] = []

    def sender(self, rid: str) -> _Sender:
        return _Sender(self.device, rid, self.log)

    def write_chunk(self, req, start: int, end: int, pages: List[int]) -> None:
        """prepare_for_extend: allocate ``pages`` for positions [start, end)."""
        req.extend_range = _Range(start, end)
        # A trailing partial page holds only the range's tokens.
        self.s.req_to_token_pool.req_to_token[req.kv.req_pool_idx, start:end] = _slots(
            pages
        )[: end - start].to(torch.int32)

    def step(self, batch_reqs: Optional[list]) -> None:
        batch = (
            None
            if batch_reqs is None
            else SimpleNamespace(reqs=batch_reqs, hicache_load_page_ids=None)
        )
        if batch is not None:
            # run_batch: early send while the previous forward is still
            # running, then this forward starts writing its own pages.
            for req in batch.reqs:
                self.s.maybe_send_cached_prefix_chunk(req, hicache_load_page_ids=None)
            pages = {
                int(p)
                for req in batch.reqs
                for p in (
                    self.s.req_to_token_pool.req_to_token[
                        req.kv.req_pool_idx,
                        req.extend_range.start : req.extend_range.end,
                    ]
                    // PAGE_SIZE
                ).tolist()
            }
            self.device.launch(pages)
        else:
            self.device.land()
        # process_batch_result of the previous batch: it has completed, and the
        # overlap loop sends each request's chunk from here. A final chunk is
        # modelled the same way, up to its last full page.
        for req, chunk_end in self.pending:
            self.s.send_kv_chunk(req, last_chunk=False, end_idx=chunk_end)
        self.pending = (
            [] if batch is None else [(r, r.extend_range.end) for r in batch.reqs]
        )
        if (
            prefill._is_tpu
            and envs.SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX.get()
        ):
            prefill.tpu_update_inflight_kv_write_pages(self.s, batch)

    def cache_chunk(self, req, upto: int) -> None:
        """process_prefill_chunk: the chunk just launched enters the radix tree
        (before its forward has finished)."""
        req.prefix_indices = self.s.req_to_token_pool.req_to_token[
            req.kv.req_pool_idx, :upto
        ].to(torch.int64)

    def sends_of(self, rid: str) -> List[List[int]]:
        return [pages for r, pages, _ in self.log if r == rid]

    def inflight_reads(self) -> List[Tuple[str, List[int]]]:
        return [
            (r, sorted(set(pages) & inflight))
            for r, pages, inflight in self.log
            if set(pages) & inflight
        ]


def _scenario(harness: _Harness) -> Dict[str, object]:
    """Chunked request A shares its prefix with later arrivals B and C.

    step 0: A chunk0 [0,8)   -> pages 1,2
    step 1: A chunk1 [8,16)  -> pages 3,4 , B (prefix = A[0:8]) [8,12) -> page 6
    step 2: A chunk2 [16,20) -> page 7   , C (prefix = A[0:16]) [16,18) -> page 8
    step 3: nothing launched, the last results are processed
    """
    s = harness.s
    a = _req("A", 1, harness.sender("A"), input_len=20)
    harness.write_chunk(a, 0, 8, [1, 2])
    harness.step([a])

    harness.cache_chunk(a, 8)
    b = _req(
        "B",
        2,
        harness.sender("B"),
        input_len=12,
        prefix_indices=a.prefix_indices.clone(),
    )
    harness.write_chunk(a, 8, 16, [3, 4])
    harness.write_chunk(b, 8, 12, [6])
    s.req_to_token_pool.req_to_token[2, :8] = a.prefix_indices.to(torch.int32)
    n_before = len(harness.log)
    harness.step([a, b])
    step1 = [(r, p) for r, p, _ in harness.log[n_before:]]

    harness.cache_chunk(a, 16)
    c = _req(
        "C",
        3,
        harness.sender("C"),
        input_len=18,
        prefix_indices=a.prefix_indices.clone(),
    )
    harness.write_chunk(a, 16, 20, [7])
    harness.write_chunk(c, 16, 18, [8])
    s.req_to_token_pool.req_to_token[3, :16] = a.prefix_indices.to(torch.int32)
    n_before = len(harness.log)
    harness.step([a, c])
    step2 = [(r, p) for r, p, _ in harness.log[n_before:]]

    n_before = len(harness.log)
    harness.step(None)
    step3 = [(r, p) for r, p, _ in harness.log[n_before:]]
    return {"step1": step1, "step2": step2, "step3": step3}


class TestEarlySendAvoidsInflightWrites(CustomTestCase):
    def _run(self, *, is_tpu: bool):
        scheduler = _Scheduler(enable_overlap=True)
        harness = _Harness(scheduler)
        with _platform(is_tpu=is_tpu):
            out = _scenario(harness)
        return harness, out

    def test_unclipped_early_send_reads_pages_the_forward_is_writing(self):
        # Negative control: without the TPU clip (the CUDA-style path, which
        # relies on a stream event) step 1's early send reads pages 1,2 while
        # step 0's forward is still writing them. This is the bug.
        harness, _ = self._run(is_tpu=False)
        self.assertEqual(harness.inflight_reads()[:2], [("A", [1, 2]), ("B", [1, 2])])

    def test_tpu_early_send_never_reads_inflight_pages(self):
        harness, _ = self._run(is_tpu=True)
        self.assertEqual(harness.inflight_reads(), [])

    def test_tpu_sends_the_ready_prefix_before_the_forward_starts(self):
        # Step 2: A's chunk1 (pages 3,4) is in flight, but pages 1,2 were
        # written at step 0. C's early send goes out for those before forward 2
        # launches, and stops there.
        _, out = self._run(is_tpu=True)
        self.assertEqual(out["step2"][0], ("C", [1, 2]))
        self.assertNotIn(("C", [3, 4]), out["step2"])

    def test_tpu_held_back_prefix_goes_with_the_next_send(self):
        # Step 1: everything B could send is still in flight, so nothing of B
        # goes out. Its prefix follows with its own chunk once that completes.
        # C's held-back pages 3,4 follow the same way after step 2.
        _, out = self._run(is_tpu=True)
        self.assertEqual(out["step1"], [("A", [1, 2])])
        self.assertIn(("B", [1, 2, 6]), out["step2"])
        self.assertIn(("C", [3, 4]), out["step3"])

    def test_every_page_is_sent_exactly_once_per_request(self):
        harness, _ = self._run(is_tpu=True)
        for rid, expected in (
            ("A", [1, 2, 3, 4, 7]),
            ("B", [1, 2, 6]),
            ("C", [1, 2, 3, 4]),
        ):
            sent = [p for pages in harness.sends_of(rid) for p in pages]
            self.assertEqual(sent, expected, rid)


class TestFindSafePrefixSendEnd(CustomTestCase):
    def _safe_end(self, pages, candidate_send_end, *, inflight, start_send_idx=0):
        return _find_safe_prefix_send_end(
            prefix_indices=_slots(pages),
            start_send_idx=start_send_idx,
            candidate_send_end=candidate_send_end,
            inflight_write_page_ids=(
                None if inflight is None else torch.tensor(inflight)
            ),
            page_size=PAGE_SIZE,
        )

    def test_nothing_in_flight_keeps_the_end(self):
        self.assertEqual(self._safe_end([1, 2, 3], 12, inflight=None), 12)
        self.assertEqual(self._safe_end([1, 2, 3], 12, inflight=[]), 12)

    def test_stops_at_the_first_inflight_page(self):
        self.assertEqual(self._safe_end([1, 2, 3, 4], 16, inflight=[3, 9]), 8)
        self.assertEqual(self._safe_end([3, 1, 2], 12, inflight=[3, 9]), 0)
        self.assertEqual(self._safe_end([1, 2, 4], 12, inflight=[3, 9]), 12)

    def test_scans_from_the_page_holding_start_send_idx(self):
        # Page 1 lies before start_send_idx, so it is never read: no clip.
        self.assertEqual(
            self._safe_end([1, 2, 3], 12, inflight=[1], start_send_idx=4), 12
        )
        self.assertEqual(
            self._safe_end([1, 2, 3], 12, inflight=[2], start_send_idx=4), 4
        )


class TestCollectPrefillWritePageIds(CustomTestCase):
    def _write_page_ids(self, scheduler, reqs):
        return _collect_prefill_write_page_ids(
            reqs=reqs,
            req_to_token_table=scheduler.req_to_token_pool.req_to_token,
            page_size=PAGE_SIZE,
        )

    def test_covers_every_extend_range(self):
        s = _Scheduler(enable_overlap=True)
        r1 = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=1), extend_range=_Range(0, 8)
        )
        r2 = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=2), extend_range=_Range(4, 6)
        )
        r3 = SimpleNamespace(kv=SimpleNamespace(req_pool_idx=3), extend_range=None)
        s.req_to_token_pool.req_to_token[1, :8] = _slots([5, 7]).to(torch.int32)
        s.req_to_token_pool.req_to_token[2, 4:6] = torch.tensor(
            [9, 10], dtype=torch.int32
        )
        page_ids = self._write_page_ids(s, [r1, r2, r3])
        self.assertEqual(sorted(page_ids.tolist()), [2, 5, 7])
        self.assertEqual(self._write_page_ids(s, [r3]).numel(), 0)

    def test_covers_an_unaligned_first_page(self):
        # A chunk starting mid-page still writes that page.
        s = _Scheduler(enable_overlap=True)
        req = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=1), extend_range=_Range(2, 9)
        )
        s.req_to_token_pool.req_to_token[1, :12] = _slots([3, 6, 8]).to(torch.int32)
        self.assertEqual(self._write_page_ids(s, [req]).tolist(), [3, 6, 8])


class TestHiCacheLoadPageIds(CustomTestCase):
    def test_uses_the_allocator_page_size(self):
        # DCP widens the allocator page beyond the radix page. Slots 0-15 are
        # pages 0-3 at the radix size 4 but pages 0-1 at the allocator size 8,
        # the unit the early-send clips compare against.
        scheduler = SimpleNamespace(
            page_size=4, token_to_kv_pool_allocator=SimpleNamespace(page_size=8)
        )
        page_ids = tpu_hicache_load_page_ids(
            scheduler=scheduler,
            load_indices=[torch.arange(0, 8), torch.arange(8, 16)],
        )
        self.assertEqual(sorted(page_ids.tolist()), [0, 1])

    def test_deduplicates_page_ids(self):
        """Repeated page IDs must not enlarge every request's scan."""
        scheduler = SimpleNamespace(
            token_to_kv_pool_allocator=SimpleNamespace(page_size=64)
        )
        page_ids = tpu_hicache_load_page_ids(
            scheduler=scheduler,
            load_indices=[torch.arange(64, 192), torch.arange(192, 256)],
        )
        # Every slot of a page maps to the same page ID, so each ID repeats 64 times;
        # each physical page needs only one entry.
        self.assertEqual(sorted(page_ids.tolist()), [1, 2, 3])
        # The IDs stay unique even if a page's slots are not adjacent in a load.
        page_ids = tpu_hicache_load_page_ids(
            scheduler=scheduler, load_indices=[torch.tensor([0, 130, 1])]
        )
        self.assertEqual(sorted(page_ids.tolist()), [0, 2])


class TestUpdateInflightKvWritePages(CustomTestCase):
    def _batch(self, s, hicache_load_page_ids=None):
        req = SimpleNamespace(
            kv=SimpleNamespace(req_pool_idx=1), extend_range=_Range(0, 8)
        )
        s.req_to_token_pool.req_to_token[1, :8] = _slots([5, 7]).to(torch.int32)
        return SimpleNamespace(reqs=[req], hicache_load_page_ids=hicache_load_page_ids)

    def test_tracks_the_launched_batch(self):
        s = _Scheduler(enable_overlap=True)
        s.tpu_inflight_kv_write_page_ids = torch.tensor([1])
        with _platform(is_tpu=True):
            tpu_update_inflight_kv_write_pages(s, self._batch(s))
        self.assertEqual(s.tpu_inflight_kv_write_page_ids.tolist(), [5, 7])

    def test_tracks_the_launched_batch_hicache_load_pages(self):
        # A torch_tpu load scatters on device with the forward, so a later
        # batch must not send its pages before that forward finishes either.
        s = _Scheduler(enable_overlap=True)
        batch = self._batch(s, hicache_load_page_ids=torch.tensor([3, 4]))
        with _platform(is_tpu=True):
            tpu_update_inflight_kv_write_pages(s, batch)
        self.assertEqual(set(s.tpu_inflight_kv_write_page_ids.tolist()), {3, 4, 5, 7})

    def test_no_inflight_batch_clears_the_write_page_ids(self):
        s = _Scheduler(enable_overlap=True)
        s.tpu_inflight_kv_write_page_ids = torch.tensor([1])
        with _platform(is_tpu=True):
            tpu_update_inflight_kv_write_pages(s, None)
        self.assertIsNone(s.tpu_inflight_kv_write_page_ids)


class TestMaybeSendCachedPrefixChunk(CustomTestCase):
    def _send(
        self,
        *,
        is_tpu,
        enable_overlap,
        inflight_write_page_ids,
        hicache_load_page_ids=None,
    ):
        scheduler = _Scheduler(enable_overlap=enable_overlap)
        scheduler.tpu_inflight_kv_write_page_ids = inflight_write_page_ids
        scheduler.send_kv_chunk = Mock()
        req = SimpleNamespace(
            rid="r0",
            prefix_indices=_slots([1, 2]),
            host_hit_length=0,
            start_send_idx=0,
            pending_bootstrap=False,
            early_send_prefix_end=None,
        )
        with _platform(is_tpu=is_tpu):
            scheduler.maybe_send_cached_prefix_chunk(
                req, hicache_load_page_ids=hicache_load_page_ids
            )
        return scheduler.send_kv_chunk, req

    def test_tpu_overlap_skips_when_the_whole_prefix_is_in_flight(self):
        send, _ = self._send(
            is_tpu=True, enable_overlap=True, inflight_write_page_ids=torch.tensor([1])
        )
        send.assert_not_called()

    def test_tpu_overlap_sends_the_ready_part(self):
        send, req = self._send(
            is_tpu=True, enable_overlap=True, inflight_write_page_ids=torch.tensor([2])
        )
        send.assert_called_once_with(req, last_chunk=False, end_idx=PAGE_SIZE)

    def test_tpu_overlap_sends_everything_with_nothing_in_flight(self):
        send, req = self._send(
            is_tpu=True, enable_overlap=True, inflight_write_page_ids=None
        )
        send.assert_called_once_with(req, last_chunk=False, end_idx=2 * PAGE_SIZE)

    def test_tpu_without_overlap_does_not_clip(self):
        send, req = self._send(
            is_tpu=True, enable_overlap=False, inflight_write_page_ids=torch.tensor([1])
        )
        send.assert_called_once_with(req, last_chunk=False, end_idx=2 * PAGE_SIZE)

    def test_other_backends_do_not_clip(self):
        send, req = self._send(
            is_tpu=False,
            enable_overlap=True,
            inflight_write_page_ids=torch.tensor([1]),
            hicache_load_page_ids=torch.tensor([2]),
        )
        send.assert_called_once_with(req, last_chunk=False, end_idx=2 * PAGE_SIZE)


class TestOverlapLoopOrder(CustomTestCase):
    def _loop(self, batches, *, last_batch, early_send=True):
        """Run one loop iteration per entry of ``batches`` (None = nothing to
        launch) and stop at the end of the last one."""
        order = Mock()
        scheduler = _Scheduler(enable_overlap=True)
        scheduler._engine_paused = False
        scheduler.waiting_queue = []
        scheduler.running_batch = None
        scheduler.last_batch = last_batch
        scheduler.chunked_req = None
        scheduler.ingest_requests = Mock()
        scheduler.disagg_prefill_bootstrap_queue = Mock(
            pop_bootstrapped=Mock(return_value=[])
        )
        scheduler.get_next_disagg_prefill_batch_to_run = Mock(
            side_effect=[
                SimpleNamespace(running_batch=None, batch_to_run=b) for b in batches
            ]
        )
        scheduler.ngram_embedding_manager = Mock(
            prepare_for_forward=Mock(side_effect=lambda b, **kw: b)
        )
        scheduler._apply_war_barrier = Mock()
        scheduler.on_idle = Mock()
        scheduler.launch_batch_sample_if_needed = Mock()
        scheduler.run_batch = order.run_batch
        scheduler.process_batch_result = order.process_batch_result
        scheduler.process_disagg_prefill_inflight_queue = Mock(
            side_effect=[None] * (len(batches) - 1) + [_StopLoop]
        )
        with (
            _platform(
                is_tpu=True,
                early_send=early_send,
                update_write_pages=order.update_write_pages,
            ),
            self.assertRaises(_StopLoop),
        ):
            scheduler.event_loop_overlap_disagg_prefill()
        return order

    def test_tpu_tracks_the_batch_after_launching_it(self):
        batch = Mock(reqs=[])
        batch.copy.return_value = batch
        order = self._loop([batch], last_batch=Mock())
        calls = [c[0] for c in order.mock_calls]
        self.assertLess(calls.index("run_batch"), calls.index("update_write_pages"))
        self.assertIs(order.update_write_pages.call_args[0][1], batch)

    def test_tpu_tracks_nothing_with_early_send_off(self):
        batch = Mock(reqs=[])
        batch.copy.return_value = batch
        order = self._loop([batch], last_batch=Mock(), early_send=False)
        order.update_write_pages.assert_not_called()

    def test_tpu_clears_the_tracking_when_nothing_is_launched(self):
        batch = Mock(reqs=[])
        batch.copy.return_value = batch
        order = self._loop([batch, None], last_batch=None)
        self.assertEqual(order.update_write_pages.call_count, 2)
        self.assertIsNone(order.update_write_pages.call_args[0][1])


class _StopRunBatch(Exception):
    pass


class _RunBatchScheduler(_Scheduler):
    """Enough of a Scheduler to run ``run_batch`` up to the forward launch."""

    disaggregation_mode = DisaggregationMode.PREFILL
    forward_ct = 0
    processed_tokens_counter = 0
    scripted_scheduler_hook = None
    forward_sleep_time = None
    _sched_idled = False

    def __init__(self, *, enable_overlap: bool):
        super().__init__(enable_overlap=enable_overlap)
        self.metrics_reporter = Mock()
        self.profiler_manager = Mock()
        self.scheduler_stage_metrics = MagicMock()

    @property
    def is_generation(self):
        # run_batch reads this right after the early sends, to pick the forward.
        raise _StopRunBatch


class TestEarlySendSkipsHiCacheLoadPages(CustomTestCase):
    """One batch admits A, then B:

    - A matched pages 1,2 on device and pages 3,4 on host only. Admitting A
      queues a load-back into device pages 3,4 and puts them in the radix
      tree. A computes [16, 20) into page 5.
    - B is matched after that, so it finds pages 1-4 all on device. B computes
      [16, 20) into page 6.

    run_batch early-sends before the forward; the load fills pages 3,4 once the
    forward starts waiting for it.
    """

    def _run(self, *, enable_overlap: bool, hicache_load_page_ids):
        s = _RunBatchScheduler(enable_overlap=enable_overlap)
        harness = _Harness(s)
        a = _req("A", 1, harness.sender("A"), input_len=20)
        b = _req("B", 2, harness.sender("B"), input_len=20)
        for req, own_page in ((a, 5), (b, 6)):
            harness.write_chunk(req, 16, 20, [own_page])
            s.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :16] = _slots(
                [1, 2, 3, 4]
            ).to(torch.int32)
            req.prefix_indices = _slots([1, 2, 3, 4])
        a.host_hit_length = 2 * PAGE_SIZE
        batch = SimpleNamespace(
            reqs=[a, b],
            hicache_load_page_ids=hicache_load_page_ids,
            extend_num_tokens=0,
            forward_mode=Mock(
                is_split_prefill=Mock(return_value=False),
                is_prebuilt=Mock(return_value=False),
            ),
        )
        harness.device.launch({3, 4})

        with _platform(is_tpu=True), self.assertRaises(_StopRunBatch):
            Scheduler.run_batch(s, batch)
        early = [(r, p) for r, p, _ in harness.log]
        harness.device.land()
        with _platform(is_tpu=True):
            # Each request's send after its forward.
            for req in (a, b):
                s.send_kv_chunk(req, last_chunk=False, end_idx=20)
        return harness, early

    def test_early_send_never_reads_pages_a_same_batch_load_fills(self):
        for enable_overlap in (False, True):
            harness, _ = self._run(
                enable_overlap=enable_overlap,
                hicache_load_page_ids=torch.tensor([3, 4]),
            )
            self.assertEqual(harness.inflight_reads(), [], enable_overlap)

    def test_device_resident_prefix_still_goes_early(self):
        for enable_overlap in (False, True):
            _, early = self._run(
                enable_overlap=enable_overlap,
                hicache_load_page_ids=torch.tensor([3, 4]),
            )
            self.assertEqual(early, [("A", [1, 2]), ("B", [1, 2])], enable_overlap)

    def test_every_page_is_sent_exactly_once_per_request(self):
        for enable_overlap in (False, True):
            harness, _ = self._run(
                enable_overlap=enable_overlap,
                hicache_load_page_ids=torch.tensor([3, 4]),
            )
            for rid, expected in (("A", [1, 2, 3, 4, 5]), ("B", [1, 2, 3, 4, 6])):
                sent = [p for pages in harness.sends_of(rid) for p in pages]
                self.assertEqual(sent, expected, (rid, enable_overlap))


if __name__ == "__main__":
    unittest.main()

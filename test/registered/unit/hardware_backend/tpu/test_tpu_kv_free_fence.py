"""KV pages freed under overlap stay out of non-forward allocations until the
forwards in flight at free time are processed."""

import ast
import functools
import subprocess
import sys
import textwrap
import unittest
import warnings
from array import array
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple
from unittest.mock import Mock

import torch

import sglang
from sglang.srt.hardware_backend.tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)
from sglang.srt.hardware_backend.tpu.kv_free_fence import install_kv_free_fence
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.allocation import (
    alloc_paged_token_slots_decode,
    alloc_paged_token_slots_extend,
    alloc_token_slots,
)
from sglang.srt.mem_cache.allocator import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=45, suite="base-a-test-cpu")

PAGE = 4

_install = functools.partial(
    install_kv_free_fence,
    spec_algorithm=SpeculativeAlgorithm.NONE,
    copies_draft_tail=False,
)


class _CopyDone:
    def __init__(self, log):
        self.log = log

    def synchronize(self):
        self.log.append("copy_done")


class TestKVFreeFence(CustomTestCase):
    def _setup(self, num_pages=4, need_sort=False):
        self.log = []
        self.allocator = TpuPagedTokenToKVPoolAllocator(
            size=num_pages * PAGE,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=need_sort,
        )
        self.fence = _install(
            self.allocator, synchronize_device=lambda: self.log.append("device")
        )

    def _launch(self):
        result = SimpleNamespace(copy_done=_CopyDone(self.log))
        self.fence.on_batch_formed()
        self.fence.on_forward_launched(result)
        return result

    def _pages(self, locs):
        return set((locs[::PAGE] // PAGE).tolist())

    def _alloc_pages(self, num_pages):
        return self.allocator.alloc(num_pages * PAGE)

    def test_unscoped_allocation_never_gets_a_page_in_flight_forwards_write(self):
        """A page freed while forward N runs can be overwritten by forward N, so
        a host-to-device load into it must not be issued before N finishes."""
        for need_sort in (False, True):
            with self.subTest(need_sort=need_sort):
                self._setup(need_sort=need_sort)
                busy = self.allocator.alloc(2 * PAGE)
                spare = self.allocator.alloc(2 * PAGE)
                self._launch()
                self.allocator.free(busy)
                self.allocator.free(spare[:PAGE])

                # Free pages exist only on the held side, so the DMA waits.
                self.assertEqual(self.allocator.available_size(), 3 * PAGE)
                loaded = self._alloc_pages(3)
                self.assertEqual(self.log, ["device"])
                self.assertEqual(self.fence.stats.pages_held, 3)
                self.assertEqual(self.fence.stats.device_syncs, 1)
                self.assertEqual(
                    self._pages(loaded), self._pages(busy) | self._pages(spare[:PAGE])
                )

    def test_unscoped_allocation_uses_free_pages_without_waiting(self):
        self._setup()
        busy = self.allocator.alloc(2 * PAGE)
        self._launch()
        self.allocator.free(busy)

        loaded = self._alloc_pages(2)
        self.assertEqual(self.log, [])
        self.assertTrue(self._pages(loaded).isdisjoint(self._pages(busy)))
        self.assertIsNone(self._alloc_pages(3))
        self.assertEqual(self.log, [])

    def test_forward_allocations_borrow_held_pages_without_waiting(self):
        """Device compute runs in launch order, so a later forward's write to a
        held page lands after the in-flight forward's write."""
        self._setup()
        busy = self.allocator.alloc(4 * PAGE)
        self._launch()
        self.allocator.free(busy)

        tree_cache = SimpleNamespace(
            token_to_kv_pool_allocator=self.allocator,
            supports_prefix_sharing=lambda: False,
        )
        alloc_token_slots(tree_cache, num_tokens=PAGE)
        alloc_paged_token_slots_extend(
            tree_cache,
            prefix_lens=torch.tensor([0]),
            prefix_lens_cpu=torch.tensor([0]),
            seq_lens=torch.tensor([PAGE + 1]),
            seq_lens_cpu=torch.tensor([PAGE + 1]),
            last_loc=torch.tensor([-1]),
            extend_num_tokens=PAGE + 1,
        )
        alloc_paged_token_slots_decode(
            tree_cache,
            seq_lens=torch.tensor([1]),
            seq_lens_cpu=torch.tensor([1]),
            last_loc=torch.tensor([-1]),
        )
        self.assertEqual(self.log, [])
        self.assertEqual(self.allocator.available_size(), 0)
        self.assertEqual(self.fence.stats.pages_borrowed, 4)

    def test_pages_release_once_every_forward_in_flight_at_free_is_processed(self):
        self._setup()
        early, late = self.allocator.alloc(2 * PAGE).split(PAGE)
        first = self._launch()
        self.allocator.free(early)
        second = self._launch()
        self.allocator.free(late)

        self.fence.on_forward_done(first)
        self.assertEqual(self.log, ["copy_done"])
        self.assertEqual(self.allocator.held_pages.num_pages, 1)
        self.fence.on_forward_done(second)
        self.assertEqual(self.allocator.held_pages.num_pages, 0)
        self.assertEqual(len(self.allocator.free_pages), 4)

        # With nothing in flight, frees go straight to the free list.
        self.allocator.free(self.allocator.alloc(PAGE))
        self.assertEqual(self.allocator.held_pages.num_pages, 0)

    def test_unqueued_result_is_ignored_and_a_skipped_result_asserts(self):
        """Loops without overlap process results they never queued; a queued
        result processed out of order would hold its pages forever."""
        self._setup()
        busy = self.allocator.alloc(PAGE)
        first = self._launch()
        second = self._launch()
        self.allocator.free(busy)

        self.fence.on_forward_done(SimpleNamespace(copy_done=None))
        self.assertEqual(self.allocator.held_pages.num_pages, 1)
        with self.assertRaises(AssertionError):
            self.fence.on_forward_done(second)
        self.fence.on_forward_done(first)
        self.fence.on_forward_done(second)
        self.assertEqual(self.allocator.held_pages.num_pages, 0)

    def test_frees_after_a_device_sync_are_not_held(self):
        """A device sync finishes every launched forward, so frees made before
        their results are processed have nothing left to wait for."""
        self._setup()
        early, late = self.allocator.alloc(2 * PAGE).split(PAGE)
        self._launch()
        self.allocator.free(early)
        self.fence.wait_for_in_flight()
        self.allocator.free(late)
        self.assertEqual(self.allocator.held_pages.num_pages, 0)

        # A forward launched after the sync is running again.
        self._launch()
        self.allocator.free(self.allocator.alloc(PAGE))
        self.assertEqual(self.allocator.held_pages.num_pages, 1)

    def test_frees_before_a_formed_batch_launches_wait_for_it(self):
        """With overlap off for a batch, the previous result is processed after
        the batch formed but before it launches; a request it finishes is
        still in the batch, which writes the request's freed pages."""
        self._setup()
        busy = self.allocator.alloc(2 * PAGE)
        prev = self._launch()
        self.fence.on_batch_formed()
        self.allocator.free(busy[:PAGE])
        self.fence.on_forward_done(prev)
        # A sync before launch does not cover the formed batch's writes.
        self.assertIsNone(self._alloc_pages(3))
        self.assertEqual(self.log, ["copy_done", "device"])
        self.assertEqual(self.allocator.held_pages.num_pages, 1)

        nxt = SimpleNamespace(copy_done=None)
        self.fence.on_forward_launched(nxt)
        self.allocator.free(busy[PAGE:])
        self.fence.on_forward_done(nxt)
        self.assertEqual(self.allocator.held_pages.num_pages, 0)

    def test_after_in_flight_runs_once_the_forwards_formed_so_far_finish(self):
        """A DMA read registered while forward N runs must not run before N
        finishes, and reads registered later never overtake earlier ones."""
        self._setup()
        ran = []
        first = self._launch()
        self.fence.after_in_flight(lambda: ran.append("a"))
        second = self._launch()
        self.fence.after_in_flight(lambda: ran.append("b"))
        self.assertEqual(ran, [])
        self.fence.on_forward_done(first)
        self.assertEqual(ran, ["a"])
        self.fence.on_forward_done(second)
        self.assertEqual(ran, ["a", "b"])
        self.fence.after_in_flight(lambda: ran.append("c"))
        self.assertEqual(ran, ["a", "b", "c"])

    def test_after_in_flight_waits_for_a_formed_batch_not_yet_launched(self):
        """With overlap off for a batch, the previous result is processed after
        the batch forms; a read registered then must wait for that batch."""
        self._setup()
        ran = []
        previous = self._launch()
        self.fence.on_batch_formed()
        self.fence.after_in_flight(lambda: ran.append("read"))
        self.fence.on_forward_done(previous)
        self.assertEqual(ran, [])
        current = SimpleNamespace(copy_done=None)
        self.fence.on_forward_launched(current)
        self.fence.on_forward_done(current)
        self.assertEqual(ran, ["read"])

    def test_install_covers_swa_sub_pools_and_rejects_other_allocators(self):
        full, swa = (
            TpuPagedTokenToKVPoolAllocator(
                size=4 * PAGE,
                page_size=PAGE,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=None,
                need_sort=False,
            )
            for _ in range(2)
        )
        wrapper = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
        wrapper.full_attn_allocator = full
        wrapper.swa_attn_allocator = swa
        fence = _install(wrapper, synchronize_device=lambda: None)
        self.assertIs(full.kv_free_fence, fence)
        self.assertIs(swa.kv_free_fence, fence)

        generic = PagedTokenToKVPoolAllocator(
            size=4 * PAGE,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        with self.assertRaises(ValueError):
            _install(generic, synchronize_device=lambda: None)

    def test_install_allows_only_spec_algorithms_whose_result_copy_follows_kv(self):
        """copy_done covers a KV write only if a copied tensor is computed after
        it; EAGLE's draft extend writes KV after every result tensor but topk,
        and DFLASH, DSPARK and NGRAM write KV after all of them."""
        allowed = [
            (SpeculativeAlgorithm.NONE, False),
            (SpeculativeAlgorithm.FROZEN_KV_MTP, False),
            (SpeculativeAlgorithm.UNO, False),
            (SpeculativeAlgorithm.EAGLE, True),
            (SpeculativeAlgorithm.EAGLE3, True),
            (SpeculativeAlgorithm.STANDALONE, True),
        ]
        rejected = [
            (SpeculativeAlgorithm.EAGLE, False),
            (SpeculativeAlgorithm.DFLASH, True),
            (SpeculativeAlgorithm.DSPARK, True),
            (SpeculativeAlgorithm.NGRAM, True),
            (CustomSpecAlgo("PLUGIN", factory=lambda server_args: None), True),
        ]
        for spec_algorithm, copies_draft_tail in allowed + rejected:
            with self.subTest(spec=spec_algorithm, copies_draft_tail=copies_draft_tail):
                allocator = TpuPagedTokenToKVPoolAllocator(
                    size=4 * PAGE,
                    page_size=PAGE,
                    dtype=torch.bfloat16,
                    device="cpu",
                    kvcache=None,
                    need_sort=False,
                )
                install = functools.partial(
                    install_kv_free_fence,
                    allocator,
                    synchronize_device=lambda: None,
                    spec_algorithm=spec_algorithm,
                    copies_draft_tail=copies_draft_tail,
                )
                if (spec_algorithm, copies_draft_tail) in allowed:
                    install()
                else:
                    with self.assertRaises(ValueError):
                        install()


class _StopLoop(Exception):
    pass


class TestOverlapLoopWiring(CustomTestCase):
    """Each overlap loop must form the next batch before processing the last
    result, and release a result's held pages only once it is processed."""

    def _run(self, loop, *, disable_overlap: bool):
        """Run batches A, B, then none; processing A frees two pages that a
        request still in B writes."""
        log = []
        allocator = TpuPagedTokenToKVPoolAllocator(
            size=4 * PAGE,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=None,
            need_sort=False,
        )
        finished = allocator.alloc(2 * PAGE)
        held_at_processing = []

        def process_result(batch, result):
            held_at_processing.append(allocator.held_pages.num_pages)
            if batch is batch_a:
                allocator.free(finished)

        batch_a, batch_b = Mock(reqs=[]), Mock(reqs=[])
        for batch in (batch_a, batch_b):
            batch.copy.return_value = batch
            batch.forward_mode.is_decode.return_value = True
        plans = [
            SimpleNamespace(running_batch=None, batch_to_run=b)
            for b in (batch_a, batch_b, None)
        ]
        scheduler = Mock(
            gracefully_exit=False,
            _engine_paused=False,
            enable_unified_memory=False,
            is_generation=True,
            running_batch=None,
            last_batch=None,
            kv_free_fence=_install(allocator, synchronize_device=lambda: None),
            # Upstream's loop skips batch creation while input is deferred.
            _deferred_input_requests=[],
        )
        # ... and waits for the oldest result when continuous polling is on.
        scheduler._is_continuous_input_polling_enabled.return_value = False
        scheduler.ngram_embedding_manager.prepare_for_forward.side_effect = (
            lambda batch, **kwargs: batch
        )
        for name in (
            "get_next_batch_to_run",
            "get_next_disagg_prefill_batch_to_run",
            "get_next_disagg_decode_batch_to_run",
        ):
            getattr(scheduler, name).side_effect = list(plans)
        scheduler.is_disable_overlap_for_batch.side_effect = [
            False,
            disable_overlap,
            False,
        ]
        scheduler.run_batch.side_effect = lambda batch: SimpleNamespace(
            copy_done=_CopyDone(log)
        )
        scheduler.batch_result_processor.process_batch_result_decode.side_effect = (
            process_result
        )
        scheduler.process_batch_result = functools.partial(
            Scheduler.process_batch_result, scheduler
        )
        scheduler.launch_batch_sample_if_needed.side_effect = [None, None, _StopLoop]
        with self.assertRaises(_StopLoop):
            loop(scheduler)
        self.assertEqual(held_at_processing, [0, 2])
        self.assertEqual(allocator.held_pages.num_pages, 0)
        self.assertEqual(log, ["copy_done", "copy_done"])

    def test_pages_freed_by_a_result_wait_for_the_next_forward(self):
        loops = [
            (Scheduler.event_loop_overlap, False),
            (Scheduler.event_loop_overlap, True),
            (Scheduler.event_loop_overlap_disagg_prefill, False),
            (Scheduler.event_loop_overlap_disagg_decode, False),
            (Scheduler.event_loop_overlap_disagg_decode, True),
        ]
        for loop, disable_overlap in loops:
            with self.subTest(loop=loop.__name__, disable_overlap=disable_overlap):
                self._run(loop, disable_overlap=disable_overlap)


class TestRadixEvictionUnderFence(CustomTestCase):
    """Radix eviction frees skip the hold only for nodes no running forward can
    write: last inserted or unlocked before the oldest running batch formed."""

    def setUp(self):
        size = 16 * PAGE
        kv_pool = MHATokenToKVPool(
            size=size,
            page_size=PAGE,
            dtype=torch.bfloat16,
            head_num=1,
            head_dim=8,
            layer_num=1,
            device="cpu",
            enable_memory_saver=False,
        )
        self.allocator = TpuPagedTokenToKVPoolAllocator(
            size=size,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device="cpu",
            kvcache=kv_pool,
            need_sort=False,
        )
        self.fence = _install(self.allocator, synchronize_device=lambda: None)
        self.cache = UnifiedRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4,
                    max_context_len=64,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                token_to_kv_pool_allocator=self.allocator,
                page_size=PAGE,
                eviction_policy="lru",
                tree_components=(ComponentType.FULL,),
                # The fence's release stamps live in the Python core, which a TPU
                # allocator always resolves to; this CPU one would pick Rust.
                tree_core_backend="python",
            )
        )

    def _insert(self, token_ids):
        self.cache.insert(
            InsertParams(
                key=RadixKey(array("q", token_ids)),
                value=self.allocator.alloc(len(token_ids)).to(torch.int64),
            )
        )

    def _launch(self):
        result = SimpleNamespace(copy_done=None)
        self.fence.on_batch_formed()
        self.fence.on_forward_launched(result)
        return result

    def _evict_all(self):
        self.cache.evict(EvictParams(num_tokens=self.cache.evictable_size()))
        self.assertEqual(self.cache.evictable_size(), 0)

    def test_node_released_before_batch_formed_is_freed_at_once(self):
        self._insert(list(range(2 * PAGE)))
        self._launch()
        # Splitting after the batch formed keeps the stamp of the node it split.
        self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", list(range(PAGE)) + [99] * PAGE)))
        )
        self._evict_all()
        self.assertEqual(self.allocator.held_pages.num_pages, 0)
        self.assertEqual(self.fence.stats.pages_freed_past_write_horizon, 2)

    def test_node_inserted_or_unlocked_after_batch_formed_is_held(self):
        """A request in a running forward's batch can insert or unlock its
        pages once the batch formed, even before launch; evicting them early
        lets a load land under that forward's write."""
        self._insert(list(range(PAGE)))
        locked = self.cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", list(range(PAGE)))))
        ).last_device_node
        receipt = self.cache.inc_lock_ref(locked)
        self.fence.on_batch_formed()
        self.cache.dec_lock_ref(locked, receipt.to_dec_params())
        first = SimpleNamespace(copy_done=None)
        self.fence.on_forward_launched(first)
        self._insert([100 + i for i in range(PAGE)])
        self._evict_all()
        self.assertEqual(self.allocator.held_pages.num_pages, 2)

        self.fence.on_forward_done(first)
        self.assertEqual(self.allocator.held_pages.num_pages, 0)

    def test_every_tree_device_free_records_its_node_stamp(self):
        """A device value freed without recording its node's stamp is released
        early whenever its eviction also frees an older node's value."""
        stray = []
        for location, func in _srt_functions():
            for call in _attribute_calls(func):
                target = call.func.value
                if (
                    call.func.attr == "evict_component"
                    and func.name != "_evict_component_and_detach_lru"
                ) or (
                    call.func.attr in ("append", "extend")
                    and isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "device_frees"
                    and func.name != "evict_component"
                ):
                    stray.append(f"{location}:{call.lineno}")
        self.assertEqual(stray, [])

    def test_every_device_lock_release_stamps_its_node(self):
        """A lock release must stamp before it can evict; a stale stamp frees
        pages the releasing request's in-flight forward still writes."""
        releases = {
            "release_component_lock",
            "release_window_lock",
            "_release_components",
        }
        exempt = (
            # The shared helper; its callers stamp.
            "_release_components",
            # Host locks keep host values, which no forward writes.
            "dec_host_lock_ref",
        )
        stray = []
        for location, func in _srt_functions():
            if func.name in exempt:
                continue
            lines = {}
            for call in _attribute_calls(func):
                lines.setdefault(call.func.attr, []).append(call.lineno)
            release_lines = [n for name in releases for n in lines.get(name, ())]
            stamp_lines = lines.get("_stamp_write_horizon", [])
            if release_lines and min(stamp_lines, default=float("inf")) > min(
                release_lines
            ):
                stray.append(f"{location}:{func.name}")
        self.assertEqual(stray, [])


@functools.cache
def _srt_functions() -> List[Tuple[Path, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """(path, function) for every function under sglang/srt."""
    srt = Path(sglang.__file__).parent / "srt"
    functions = []
    for path in srt.rglob("*.py"):
        # The Rust core reports no write horizon, so all its frees are held.
        if "rust_tree_core" in path.parts:
            continue
        with warnings.catch_warnings():
            # Some modules carry invalid escapes in non-raw strings.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text())
        functions.extend(
            (path.relative_to(srt), func)
            for func in ast.walk(tree)
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    return functions


def _attribute_calls(func: ast.FunctionDef | ast.AsyncFunctionDef) -> List[ast.Call]:
    return [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


class TestTpuImportOrder(CustomTestCase):
    def test_swa_allocator_imports_first_on_tpu(self):
        # On TPU the SWA allocator imports the TPU allocator, which imports the
        # fence; the fence must not import the unified cache components then.
        script = textwrap.dedent("""
            import sglang.srt.utils
            sglang.srt.utils.is_tpu = lambda: True
            import sglang.srt.mem_cache.allocator.swa
            import sys
            assert "sglang.srt.hardware_backend.tpu.kv_free_fence" in sys.modules
            """)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(sglang.__file__).parents[1],
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])


if __name__ == "__main__":
    unittest.main()

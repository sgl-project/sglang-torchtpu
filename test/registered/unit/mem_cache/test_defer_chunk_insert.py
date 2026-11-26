"""SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE defers only mid-prefill chunk inserts.

The insert at the end of a prefill must still run: a follow-up turn diverges right
after the prompt, and its match needs the SWA window that insert keeps.
"""

import random
import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from swa_cache_world import PAGE, WINDOW, KVWorld

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.environ import envs
from sglang.srt.managers.req_types import FINISH_LENGTH
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.allocation import retire_window
from sglang.srt.mem_cache.common import checkpoint_kv_cache, release_kv_cache
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _Tree:
    def __init__(self, defers=True):
        self.defers = defers
        self.calls = []

    def defer_chunk_checkpoint(self, req):
        self.calls.append(("defer", req.extend_end))
        return self.defers

    def checkpoint(self, req, *, up_to):
        self.calls.append(("checkpoint", up_to))


def _run(stride, mid_prefill, start=4096, end=8192, tree=None):
    req = SimpleNamespace(
        rid="r",
        finished=lambda: False,
        skip_radix_cache_insert=False,
        prefix_len=start,
        extend_end=end,
    )
    tree = tree or _Tree()
    with envs.SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE.override(stride):
        checkpoint_kv_cache(req, tree, mid_prefill=mid_prefill)
    return tree.calls


class TestDeferChunkInsert(unittest.TestCase):
    def test_mid_prefill_chunk_is_deferred(self):
        self.assertEqual(_run(16384, mid_prefill=True), [("defer", 8192)])

    def test_end_of_prefill_insert_runs(self):
        self.assertEqual(_run(16384, mid_prefill=False), [("checkpoint", 8192)])

    def test_zero_stride_inserts_every_chunk(self):
        self.assertEqual(_run(0, mid_prefill=True), [("checkpoint", 8192)])

    def test_stride_inserts_the_chunk_that_reaches_a_multiple(self):
        # 16K stride, 4K chunks: [12K, 16K) and [30K, 34K) reach a multiple,
        # [16K, 20K) does not.
        self.assertEqual(_run(16384, True, 12288, 16384), [("checkpoint", 16384)])
        self.assertEqual(_run(16384, True, 30720, 34816), [("checkpoint", 34816)])
        self.assertEqual(_run(16384, True, 16384, 20480), [("defer", 20480)])

    def test_cache_that_declines_still_inserts(self):
        calls = _run(16384, mid_prefill=True, tree=_Tree(defers=False))
        self.assertEqual(calls, [("defer", 8192), ("checkpoint", 8192)])


def _run_disagg(stride, start, end, prompt_len):
    req = SimpleNamespace(
        rid="r",
        finished=lambda: False,
        skip_radix_cache_insert=False,
        pending_bootstrap=False,
        disagg_kv_sender=None,
        origin_input_ids=list(range(prompt_len)),
        prefix_len=start,
        extend_end=end,
    )
    scheduler = SimpleNamespace(tree_cache=_Tree())
    with envs.SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE.override(stride):
        SchedulerDisaggregationPrefillMixin.checkpoint_disagg_prefill(scheduler, req)
    return scheduler.tree_cache.calls, req.prefix_len


class TestDisaggPrefillChunkInsert(unittest.TestCase):
    """A disaggregated prefill checkpoints each chunk through
    checkpoint_disagg_prefill, which marks every chunk before the prompt end
    mid-prefill."""

    def test_chunk_before_prompt_end_is_deferred(self):
        calls, prefix_len = _run_disagg(16384, 4096, 8192, prompt_len=20000)
        self.assertEqual(calls, [("defer", 8192)])
        self.assertEqual(prefix_len, 8192)

    def test_last_chunk_inserts(self):
        # [16K, 20000) reaches no 16K multiple but ends the prompt.
        calls, _ = _run_disagg(16384, 16384, 20000, prompt_len=20000)
        self.assertEqual(calls, [("checkpoint", 20000)])

    def test_bootstrap_finalize_publishes_held_chunks(self):
        # Chunks advanced unpublished while bootstrap was pending leave
        # prefix_len == extend_end; the finalize call must insert them.
        calls, _ = _run_disagg(16384, 8192, 8192, prompt_len=20000)
        self.assertEqual(calls, [("checkpoint", 8192)])


def _swa_cache(**overrides):
    attrs = dict(
        disable=False,
        supports_swa=lambda: True,
        supports_mamba=lambda: False,
        page_size=4,
        _components_tuple=(Mock(),),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(64, dtype=torch.int32).reshape(2, 32)
        ),
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


def _req(**overrides):
    attrs = dict(
        session_id=None,
        finished=lambda: False,
        full_untruncated_fill_ids=list(range(30)),
        kv=SimpleNamespace(cache_protected_len=0, cache_inserted_len=0, req_pool_idx=1),
        priority=None,
        kv_rotation_base=None,
        prefix_len=0,
    )
    attrs.update(overrides)
    return SimpleNamespace(**attrs)


class TestUnifiedDeferChunkCheckpoint(unittest.TestCase):
    def test_falls_back_outside_swa_only_trees(self):
        for cache, req in (
            (_swa_cache(supports_swa=lambda: False), _req()),
            (_swa_cache(supports_mamba=lambda: True), _req()),
            (_swa_cache(disable=True), _req()),
            (_swa_cache(), _req(session_id="s")),
        ):
            self.assertFalse(UnifiedRadixCache.defer_chunk_checkpoint(cache, req))
            cache._components_tuple[0].free_out_of_window_slots.assert_not_called()
            self.assertEqual(req.prefix_len, 0)

    def test_falls_back_when_the_request_cannot_free_its_swa(self):
        for flag, value in (
            (envs.SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS, False),
            (envs.SGLANG_OPT_SWA_KEEP_PREFILL_CHUNK_WINDOWS, True),
        ):
            cache, req = _swa_cache(), _req()
            with self.subTest(flag=flag.name), flag.override(value):
                self.assertFalse(UnifiedRadixCache.defer_chunk_checkpoint(cache, req))
            cache._components_tuple[0].free_out_of_window_slots.assert_not_called()


def _tokens(seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1, 50000) for _ in range(n)]


class _TurnWorld(KVWorld):
    """A real hybrid-SWA UnifiedRadixCache driven as the scheduler drives a turn:
    admission match + lock, chunked prefill with checkpoint_kv_cache after every
    chunk (mid_prefill before the last one), decode that retires the request's
    own window, and release_kv_cache at the finish."""

    def turn(self, prompt: list[int], *, chunk: int) -> None:
        req = Req(
            rid=f"r{self.rid}",
            origin_input_text="",
            origin_input_ids=array("q", prompt),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1 << 20),
        )
        self.rid += 1
        self.req_to_token_pool.alloc([req])
        req.init_next_round_input(tree_cache=self.cache)
        req.lock = self.cache.lock(req.last_node)
        row, pos = req.kv.req_pool_idx, req.prefix_len
        if pos:
            self.req_to_token_pool.write(
                (row, slice(0, pos)), self.cache.prefix_device_indices(req)
            )
        while pos < len(prompt):
            end = min(pos + chunk, len(prompt))
            req.prefix_len, req.extend_end = pos, end
            self.req_to_token_pool.write(
                (row, slice(pos, end)), self._alloc(prompt[pos:end], pos)
            )
            req.kv.kv_committed_len = req.kv.kv_allocated_len = end
            pos = end
            if pos < len(prompt):
                checkpoint_kv_cache(req, self.cache, mid_prefill=True)
                req.init_next_round_input()
        req.output_ids.append(self._next_token(req))
        checkpoint_kv_cache(req, self.cache)
        for _ in range(2 * WINDOW):
            self.decode_step(req)
        retire_window(req, req.seqlen - 1, self.cache)
        req.finished_reason = FINISH_LENGTH(length=0)
        release_kv_cache(req, self.cache, checkpoint=True)


class TestSwaBranchCheckpoint(unittest.TestCase):
    CHUNK = 8 * PAGE

    def test_branch_and_prompt_end_stay_matchable(self):
        """A turn whose match stops at an SWA branch (Full KV cached past the last
        live window) must leave both the branch and its own prompt end matchable
        once decode has retired its window, wherever the branch falls among its
        chunks and whether or not mid-prefill inserts are deferred. Otherwise the
        next turn matches only the branch and recomputes, and so does every turn
        after it."""
        earlier = _tokens(1, 17 * PAGE)
        branch = len(earlier)
        # Chunks end at 128, 256, ...: the branch at 272 falls in the last chunk
        # of a 332-token prompt and in a mid chunk of a 472-token one.
        for stride in (0, 1 << 20):
            for new_tokens in (60, 200):
                with (
                    self.subTest(stride=stride, new_tokens=new_tokens),
                    envs.SGLANG_OPT_UNIFIED_CACHE_CHUNK_INSERT_STRIDE.override(stride),
                ):
                    w = _TurnWorld(full_tokens=8192, swa_tokens=4096)
                    w.turn(earlier, chunk=self.CHUNK)
                    w.evict(swa_tokens=10**6)  # the earlier turn's windows are gone
                    prompt = earlier + _tokens(2, new_tokens)
                    self.assertEqual(w.match(prompt).swa_branching_seqlen, branch)
                    w.turn(prompt, chunk=self.CHUNK)
                    follow_up = w.match(prompt + _tokens(3, PAGE))
                    self.assertEqual(
                        follow_up.device_prefix_len, len(prompt) // PAGE * PAGE
                    )
                    sibling = w.match(earlier + _tokens(4, PAGE))
                    self.assertEqual(sibling.device_prefix_len, branch)
                    w.cache.sanity_check()


if __name__ == "__main__":
    unittest.main()

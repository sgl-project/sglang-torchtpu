"""CPU check of the DSpark decode preparation path with Triton unavailable.

The first speculative decode prepare must not reach Triton:
spec_utils.allocate_and_assign_decode_cache -> get_last_loc picks
the Triton kernel because V4.1's attention backend ("dsv4") is not on its
non-Triton allowlist. This runs HostPaddedBlockDraftInput.prepare_for_decode
(the scheduler's spec decode prepare for our worker) on real Req /
ScheduleBatch / ReqToTokenPool / TPU paged allocator objects, with the
backend reported as "dsv4", the TPU torch-native switch on, and every Triton
entry point on that path replaced by a raiser. With the plugin's get_last_loc
hook applied it must complete; without it, it must reach Triton (control).
"""

from __future__ import annotations

import pkgutil
import unittest
from array import array
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.mem_cache.allocation as allocation
import sglang.srt.speculative.spec_utils as spec_utils
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.plugins.hook_registry import HookRegistry, HookType, _propagate_patch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import TpuPagedTokenToKVPoolAllocator
from sglang_tpu.experimental.speculative import draft_info
from sglang_tpu.experimental.speculative.block_info import HostPaddedBlockDraftInput
from sglang_tpu.hooks import dsv41 as dsv41_hooks

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

PAGE, K = 4, 5


class TritonDispatched(RuntimeError):
    pass


def _no_triton(*args, **kwargs):
    raise TritonDispatched("Triton kernel dispatched on the TPU decode path")


class _TreeCache:
    """The tree-cache surface decode allocation reads (no prefix sharing, no SWA)."""

    def __init__(self, allocator):
        self.token_to_kv_pool_allocator = allocator

    def supports_prefix_sharing(self):
        return False

    def supports_swa(self):
        return False


def _apply_hook(stack: ExitStack, target: str, hook) -> None:
    owner_path, name = target.rsplit(".", 1)
    owner = pkgutil.resolve_name(owner_path)
    raw, original = vars(owner).get(name), getattr(owner, name)
    HookRegistry._apply_target(target, [(HookType.AROUND, hook, None)])
    wrapped = getattr(owner, name)

    def restore():
        setattr(owner, name, raw)
        _propagate_patch(wrapped, original, owner)

    stack.callback(restore)


class TestDsparkDecodePrep(CustomTestCase):
    def _run(self, *, hooked: bool):
        reqs = [
            Req(f"r{i}", "", array("q", range(n)), SamplingParams())
            for i, n in enumerate((9, 30))
        ]
        pool = ReqToTokenPool(
            size=4, max_context_len=64, device="cpu", enable_memory_saver=False
        )
        pool.host_resident = True  # what a TPU process sets (table on host)
        allocator = TpuPagedTokenToKVPoolAllocator(
            256, PAGE, torch.int32, "cpu", kvcache=SimpleNamespace(), need_sort=False
        )
        slots = [1, 2]
        for req, slot in zip(reqs, slots):
            n = len(req.origin_input_ids)
            locs = allocator.alloc(-(-n // PAGE) * PAGE)
            pool.req_to_token[slot, : len(locs)] = locs
            req.kv.req_pool_idx = slot
            req.kv.kv_committed_len = n
            req.kv.kv_allocated_len = len(locs)
        batch = ScheduleBatch(reqs=reqs)
        batch.req_to_token_pool = pool
        batch.token_to_kv_pool_allocator = allocator
        batch.tree_cache = _TreeCache(allocator)
        batch.req_pool_indices = torch.tensor(slots)
        batch.req_pool_indices_cpu = torch.tensor(slots)
        batch.device = "cpu"
        batch.enable_overlap = True
        batch.sampling_info = SimpleNamespace(
            penalizer_orchestrator=SimpleNamespace(is_required=False)
        )
        draft = HostPaddedBlockDraftInput(
            bonus_tokens=torch.zeros(2, dtype=torch.int64),
            new_seq_lens=torch.zeros(2, dtype=torch.int64),
        )
        spec = SimpleNamespace(
            speculative_num_draft_tokens=K + 1, speculative_draft_window_size=None
        )
        with ExitStack() as stack:
            if hooked:
                _apply_hook(
                    stack,
                    "sglang.srt.mem_cache.allocation.get_last_loc",
                    dsv41_hooks.last_loc_on_host,
                )
            stack.enter_context(
                patch.object(allocation, "attention_backends", lambda: ("dsv4", "dsv4"))
            )
            for name in ("get_last_loc_triton", "get_last_loc_triton_safe"):
                stack.enter_context(patch.object(allocation, name, _no_triton))
            stack.enter_context(
                patch.object(spec_utils, "_assign_req_to_token_pool_func", _no_triton)
            )
            stack.enter_context(patch.object(spec_utils, "_use_torch_native", True))
            stack.enter_context(patch.object(draft_info, "get_spec", return_value=spec))
            stack.enter_context(
                patch.object(
                    draft_info, "max_speculative_num_draft_tokens", return_value=K + 1
                )
            )
            draft.prepare_for_decode(batch)
        return reqs, pool, draft

    def test_prepare_for_decode_stays_off_triton(self):
        reqs, pool, draft = self._run(hooked=True)
        # Committed + 2 (K + 1) rows, page-aligned: 9 + 12 -> 24, 30 + 12 -> 44.
        self.assertEqual([r.kv.kv_allocated_len for r in reqs], [24, 44])
        for req, slot in zip(reqs, (1, 2)):
            row = pool.req_to_token[slot, : req.kv.kv_allocated_len]
            self.assertTrue(bool((row > 0).all()), row)
            self.assertEqual(len(set(row.tolist())), len(row))
        self.assertEqual(draft.tpu_committed_lens.tolist(), [9, 30])

    def test_without_the_hook_it_reaches_triton(self):
        with self.assertRaises(TritonDispatched):
            self._run(hooked=False)


if __name__ == "__main__":
    unittest.main()

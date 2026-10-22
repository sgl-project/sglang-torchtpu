"""Paged decode allocation against a host-resident req_to_token pool."""

import importlib.util
import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache import allocation
from sglang.srt.mem_cache.allocator import swa
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.allocator_tpu import (
    TpuPagedTokenToKVPoolAllocator,
)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_tpu_ci(est_time=30, suite="stage-a-test-tpu-unit", chips=1)

PAGE = 4
# Requests A and B: req_to_token rows and prefill lengths.
ROWS = [3, 1]
PREFILL_LENS = [6, 3]
# Prefill takes full pages 1-2 for A and 3 for B. Decode writes columns
# (6, 3), (7, 4), (8, 5); B takes page 4 at step 2, A takes page 5 at step 3.
DECODE_LOCS = [[10, 15], [11, 16], [20, 17]]
EXPECTED_ROWS = {
    3: [4, 5, 6, 7, 8, 9, 10, 11, 20],
    1: [12, 13, 14, 15, 16, 17],
}
# SWA pages 1-2 are taken before prefill, so each SWA page runs two pages
# ahead of its full page.
EXPECTED_FULL_TO_SWA = {
    loc: loc + 2 * PAGE for locs in EXPECTED_ROWS.values() for loc in locs
}


def _tpu_available() -> bool:
    if importlib.util.find_spec("torch_tpu") is None:
        return False
    try:
        import torch_tpu  # noqa: F401

        torch.zeros(1).to("tpu")
        return True
    except Exception:
        return False


_TPU_AVAILABLE = _tpu_available()


def _paged_allocator(device):
    return TpuPagedTokenToKVPoolAllocator(
        size=64,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device=device,
        kvcache=None,
        need_sort=False,
    )


def _swa_allocator(device):
    kvcache = SWAKVPool(
        size=64,
        size_swa=32,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cpu",
    )
    allocator = swa.SWATokenToKVPoolAllocator(
        size=64,
        size_swa=32,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kvcache,
        need_sort=False,
    )
    allocator.swa_attn_allocator.alloc(2 * PAGE)
    return allocator


def _page_allocators(allocator):
    if isinstance(allocator, swa.SWATokenToKVPoolAllocator):
        return [allocator.full_attn_allocator, allocator.swa_attn_allocator]
    return [allocator]


def _batch(allocator, pool, seq_lens, rows, device):
    return SimpleNamespace(
        device=device,
        maybe_evict_swa=lambda: None,
        seq_lens_cpu=seq_lens,
        seq_lens=seq_lens.to(device),
        req_pool_indices_cpu=rows,
        req_pool_indices=rows.to(device),
        tree_cache=SimpleNamespace(
            page_size=PAGE,
            token_to_kv_pool_allocator=allocator,
            supports_prefix_sharing=lambda: False,
        ),
        req_to_token_pool=pool,
        model_config=SimpleNamespace(is_encoder_decoder=False),
        reqs=[
            SimpleNamespace(kv=SimpleNamespace(kv_allocated_len=n, kv_committed_len=n))
            for n in seq_lens.tolist()
        ],
    )


class TestAllocForDecodeHostResidentPool(CustomTestCase):
    def setUp(self):
        for target, name, value in (
            (allocation, "_is_tpu", True),
            (swa, "_is_tpu", True),
            (swa, "TpuPagedTokenToKVPoolAllocator", TpuPagedTokenToKVPoolAllocator),
        ):
            patcher = patch.object(target, name, value, create=True)
            patcher.start()
            self.addCleanup(patcher.stop)
        # On a TPU host, _alloc_page_size reads the parallel config for DCP.
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(get_parallel().override(dcp_enabled=False))
        with envs.SGLANG_TPU_REQ_TO_TOKEN_ON_HOST.override(True):
            self.pool = ReqToTokenPool(
                size=9, max_context_len=16, device="tpu", enable_memory_saver=False
            )
        self.assertTrue(self.pool.host_resident)

    def _prefill(self, allocator):
        lens = torch.tensor(PREFILL_LENS, dtype=torch.int64)
        zeros = torch.zeros_like(lens)
        locs = allocator.alloc_extend(
            zeros, zeros, lens, lens, zeros - 1, int(lens.sum())
        ).cpu()
        for row, run in zip(ROWS, locs.split(PREFILL_LENS)):
            self.pool.req_to_token[row, : len(run)] = run

    def _decode(self, allocator, device):
        """Prefill, then one decode step per DECODE_LOCS entry. On ``meta``,
        device-side tensors hold no data, so any host readback raises."""
        self._prefill(allocator)
        if device == "meta":
            for page_allocator in _page_allocators(allocator):
                page_allocator.device = "meta"
        batch = _batch(
            allocator,
            self.pool,
            torch.tensor(PREFILL_LENS, dtype=torch.int64),
            torch.tensor(ROWS, dtype=torch.int64),
            device,
        )

        outs = []
        for _ in DECODE_LOCS:
            outs.append(allocation.alloc_for_decode(batch, token_per_req=1))
            # ScheduleBatch.prepare_for_decode advances lengths after allocation.
            batch.seq_lens_cpu = batch.seq_lens_cpu + 1
            batch.seq_lens = batch.seq_lens + 1

        for out in outs:
            self.assertEqual(out.device.type, device)
            self.assertEqual(out.dtype, torch.int32)
        if device != "meta":
            self.assertEqual([out.cpu().tolist() for out in outs], DECODE_LOCS)
        expected = torch.zeros_like(self.pool.req_to_token)
        for row, locs in EXPECTED_ROWS.items():
            expected[row, : len(locs)] = torch.tensor(locs, dtype=expected.dtype)
        self.assertTrue(torch.equal(self.pool.req_to_token, expected))
        full = _page_allocators(allocator)[0]
        self.assertEqual(full.free_pages.tolist(), list(range(6, 17)))

    def _check_full_to_swa(self, allocator):
        mapping = allocator.full_to_swa_index_mapping
        expected = torch.zeros_like(mapping)
        expected[-1] = -1
        for full_loc, swa_loc in EXPECTED_FULL_TO_SWA.items():
            expected[full_loc] = swa_loc
        self.assertEqual(mapping.device.type, "cpu")
        self.assertTrue(torch.equal(mapping, expected))
        self.assertEqual(allocator.swa_attn_allocator.free_pages.tolist(), [8])

    def test_paged_writes_host_copy(self):
        self._decode(_paged_allocator("cpu"), "meta")

    def test_swa_writes_host_copies(self):
        allocator = _swa_allocator("cpu")
        self._decode(allocator, "meta")
        self._check_full_to_swa(allocator)

    @unittest.skipUnless(_TPU_AVAILABLE, "requires torch_tpu and a TPU device")
    def test_paged_on_tpu(self):
        self._decode(_paged_allocator("tpu"), "tpu")

    @unittest.skipUnless(_TPU_AVAILABLE, "requires torch_tpu and a TPU device")
    def test_swa_on_tpu(self):
        allocator = _swa_allocator("tpu")
        self._decode(allocator, "tpu")
        self._check_full_to_swa(allocator)

    def test_device_pool_keeps_device_result(self):
        pool = ReqToTokenPool(
            size=9, max_context_len=16, device="cpu", enable_memory_saver=False
        )
        self.assertFalse(pool.host_resident)
        pool.req_to_token[3, :6] = torch.arange(4, 10)
        pool.req_to_token[1, :3] = torch.arange(12, 15)
        result = torch.tensor(DECODE_LOCS[0], dtype=torch.int64)
        allocator = Mock(
            spec=["page_size", "alloc_decode", "free", "forward_write_target"],
            page_size=PAGE,
            forward_write_target=nullcontext,
        )
        allocator.alloc_decode.return_value = result
        batch = _batch(
            allocator,
            pool,
            torch.tensor(PREFILL_LENS, dtype=torch.int64),
            torch.tensor(ROWS, dtype=torch.int64),
            "cpu",
        )

        with patch.object(allocation, "_is_tpu", False):
            out = allocation.alloc_for_decode(batch, token_per_req=1)

        self.assertIs(out, result)
        args, kwargs = allocator.alloc_decode.call_args
        self.assertEqual(kwargs, {})
        self.assertEqual(args[2].tolist(), [9, 14])
        self.assertEqual(pool.req_to_token[3, 6].item(), 10)
        self.assertEqual(pool.req_to_token[1, 3].item(), 15)


class TestTpuAllocDecodeHostOutput(CustomTestCase):
    def test_rejects_device_last_loc_before_taking_pages(self):
        allocator = _paged_allocator("cpu")
        free_pages = allocator.free_pages.clone()
        seq_lens = torch.tensor([5], dtype=torch.int32)

        with self.assertRaises(ValueError):
            allocator.alloc_decode(
                seq_lens,
                seq_lens,
                torch.tensor([7], dtype=torch.int32, device="meta"),
                out_indices_cpu=torch.empty(1, dtype=torch.int32),
            )

        self.assertTrue(torch.equal(allocator.free_pages, free_pages))


if __name__ == "__main__":
    unittest.main()

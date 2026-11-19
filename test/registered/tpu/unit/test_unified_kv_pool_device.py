# SPDX-License-Identifier: Apache-2.0
"""Model-free BF16 device proof for shared Full/SWA pages and ordered reuse."""

import math
import unittest
from types import SimpleNamespace

import torch
import torch_tpu  # noqa: F401

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_tpu_ci
from sglang_tpu.attention_pallas_rpa_base import _build_pallas_kernel_op
from sglang_tpu.attention_pallas_rpa_v3 import RPA_V3_SPEC
from sglang_tpu.kv_free_fence import install_kv_free_fence
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
)

register_tpu_ci(
    est_time=180,
    suite="stage-a-test-tpu-unit",
    labels=["singlehost", "infra", "unified-kv"],
    chips=1,
)

PAGE = 128
HEADS = 2
DIM = 128


def _make(full_count, swa_count):
    groups = math.ceil(swa_count / full_count)
    pages = 2 * (1 + groups)
    pool = TpuUnifiedSWAKVPool(
        size=pages * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        full=AttnLayerSpec(HEADS, DIM, range(full_count)),
        swa=AttnLayerSpec(HEADS, DIM, range(full_count, full_count + swa_count)),
        device="tpu",
    )
    alloc = TpuUnifiedSWATokenToKVPoolAllocator(
        size=pages * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="tpu",
        kvcache=pool,
        need_sort=False,
    )
    assert all(buf.device.type == "tpu" for buf in pool.get_state())
    return pool, alloc, pages


def _allocate(alloc):
    lengths = torch.tensor([PAGE + 1], dtype=torch.int64)
    return alloc.alloc_extend(
        prefix_lens=torch.zeros_like(lengths, device="tpu"),
        prefix_lens_cpu=torch.zeros_like(lengths),
        seq_lens=lengths.to("tpu"),
        seq_lens_cpu=lengths,
        last_loc=torch.tensor([-1], dtype=torch.int64, device="tpu"),
        extend_num_tokens=PAGE + 1,
    )


def _pages(pool, indices, layer):
    locations = indices.cpu().to(torch.int64)
    _, is_swa = pool.layers_mapping[layer]
    if is_swa:
        locations = pool.translate_loc_from_full_to_swa(indices).cpu().long()
        locations = pool._map_swa_group_indices(pool.swa_page_group(layer), locations)
    return locations, (locations[::PAGE] // PAGE).int()


def _write(pool, indices, layer, request):
    locations, pages = _pages(pool, indices, layer)
    tokens = indices.numel()
    padded = math.ceil(tokens / PAGE) * PAGE
    positions = torch.arange(padded).reshape(-1, 1, 1)
    k = ((positions % 16) / 128 + (layer + 1) / 8).expand(-1, HEADS, DIM).bfloat16()
    v = ((positions % 8) / 64 + (request + 1) / 4).expand(-1, HEADS, DIM).bfloat16()
    q = torch.zeros((padded, 2 * HEADS, DIM), dtype=torch.bfloat16, device="tpu")
    op = _build_pallas_kernel_op(RPA_V3_SPEC)
    output = op(
        q,
        k.contiguous().to("tpu"),
        v.contiguous().to("tpu"),
        pool.get_fused_kv_buffer(layer),
        torch.tensor([tokens], dtype=torch.int32, device="tpu"),
        pages.to("tpu"),
        torch.tensor([0, tokens], dtype=torch.int32, device="tpu"),
        torch.tensor([0, padded], dtype=torch.int32, device="tpu"),
        torch.tensor([0, 1, 1], dtype=torch.int32, device="tpu"),
        None,
        1 / math.sqrt(DIM),
    )
    return output, locations, k[:tokens], v[:tokens]


def _ownership(test, alloc, total, live):
    store = alloc._page_store
    buckets = [list(live), store.free_pages.tolist()]
    buckets += [pages.tolist() for pages in store.staged_pages]
    buckets += [pages.tolist() for _, pages in store.held_pages._entries]
    flat = [page for bucket in buckets for page in bucket]
    test.assertEqual(len(flat), len(set(flat)), "page has multiple owners")
    test.assertEqual(set(flat), set(range(1, total + 1)))


class TestUnifiedKVDevice(unittest.TestCase):
    def test_layer_storage_and_swa_reclamation(self):
        self.assertTrue(torch.accelerator.is_available())
        for full, swa in ((1, 1), (2, 1), (2, 3)):
            with self.subTest(full=full, swa=swa):
                pool, alloc, total = _make(full, swa)
                indices = _allocate(alloc)
                self.assertIsNotNone(indices)
                expected = {}
                for layer in range(full + swa):
                    output, locations, k, v = _write(pool, indices, layer, 0)
                    self.assertTrue(torch.isfinite(output.cpu()).all())
                    expected[layer] = (locations, k, v)
                for layer, (locations, k, v) in expected.items():
                    self.assertTrue(
                        torch.equal(pool.get_key_buffer(layer).cpu()[locations], k)
                    )
                    self.assertTrue(
                        torch.equal(pool.get_value_buffer(layer).cpu()[locations], v)
                    )
                full_pages = set((indices.cpu()[::PAGE] // PAGE).tolist())
                primary = (
                    pool.translate_loc_from_full_to_swa(indices[::PAGE]).cpu() // PAGE
                )
                swa_pages = set(pool.expand_swa_pages(primary.tolist()))
                _ownership(self, alloc, total, full_pages | swa_pages)
                # Reclaim only the first out-of-window SWA page, including companions.
                released = set(pool.expand_swa_pages(primary[:1].tolist()))
                alloc.free_swa(indices[:PAGE])
                recovered = alloc.full_attn_allocator.alloc(len(released) * PAGE)
                self.assertIsNotNone(recovered)
                self.assertEqual(
                    set((recovered.cpu()[::PAGE] // PAGE).tolist()), released
                )
                # Force a different request to overwrite every recovered physical page.
                for layer in range(full):
                    for page in recovered.cpu()[::PAGE] // PAGE:
                        pool.get_fused_kv_buffer(layer)[int(page)].fill_(17)
                    locations, k, v = expected[layer]
                    self.assertTrue(
                        torch.equal(pool.get_key_buffer(layer).cpu()[locations], k)
                    )
                    self.assertTrue(
                        torch.equal(pool.get_value_buffer(layer).cpu()[locations], v)
                    )
                _ownership(
                    self, alloc, total, full_pages | (swa_pages - released) | released
                )
                alloc.full_attn_allocator.free(recovered)
                alloc.free(indices)
                _ownership(self, alloc, total, set())
                for layer in range(full + swa):
                    self.assertTrue(
                        (pool.get_fused_kv_buffer(layer)[0].cpu() == 0).all()
                    )

    def test_cancellation_held_pages_wait_for_real_device_completion(self):
        pool, alloc, total = _make(1, 3)
        fence = install_kv_free_fence(
            alloc,
            synchronize_device=torch.accelerator.synchronize,
            spec_algorithm=SpeculativeAlgorithm.NONE,
            copies_draft_tail=False,
        )
        indices = _allocate(alloc)
        fence.on_batch_formed()
        _write(pool, indices, 0, 0)
        event = torch.tpu.Event()
        event.record()
        result = SimpleNamespace(copy_done=event)
        fence.on_forward_launched(result)
        alloc.free(indices)
        self.assertEqual(alloc._page_store.held_pages.num_pages, total)
        _ownership(self, alloc, total, set())
        reused = _allocate(alloc)
        self.assertIsNotNone(reused)
        self.assertEqual(fence.stats.device_syncs, 1)
        self.assertEqual(alloc._page_store.held_pages.num_pages, 0)
        _, locations, k, v = _write(pool, reused, 0, 1)
        self.assertTrue(torch.equal(pool.get_key_buffer(0).cpu()[locations], k))
        self.assertTrue(torch.equal(pool.get_value_buffer(0).cpu()[locations], v))
        fence.on_forward_done(result)
        alloc.free(reused)
        _ownership(self, alloc, total, set())


if __name__ == "__main__":
    unittest.main()

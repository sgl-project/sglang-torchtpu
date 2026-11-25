"""The SWA allocator places its full-to-SWA mapping where the platform says.

``SRTPlatform.keeps_swa_index_mapping_on_host()`` is the only input: True keeps
the mapping on the host whatever the allocator's device, and the default keeps
it on the device. The allocator reads it once at construction, so every later
read and write of the mapping must follow that placement.
"""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.allocator import swa
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.platforms.interface import SRTPlatform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

PAGE = 4
SIZE = 8 * PAGE


class _HostMappingPlatform(SRTPlatform):
    def keeps_swa_index_mapping_on_host(self) -> bool:
        return True


def _build(platform, *, device, kvcache):
    with patch.object(swa, "current_platform", platform):
        return swa.SWATokenToKVPoolAllocator(
            size=SIZE,
            size_swa=SIZE,
            page_size=PAGE,
            dtype=torch.bfloat16,
            device=device,
            kvcache=kvcache,
            need_sort=False,
        )


def _cpu_kv_pool():
    return SWAKVPool(
        size=SIZE,
        size_swa=SIZE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cpu",
    )


class TestSWAMappingPlacement(CustomTestCase):
    PLATFORMS = (("default", SRTPlatform()), ("host", _HostMappingPlatform()))

    def test_mapping_device_follows_the_platform(self):
        # "meta" stands in for an accelerator: only a host answer moves the
        # mapping off the allocator's device.
        for name, platform, expected in (
            ("default", SRTPlatform(), "meta"),
            ("host", _HostMappingPlatform(), "cpu"),
        ):
            with self.subTest(platform=name):
                kvcache = Mock(
                    spec=BaseSWAKVPool,
                    full_kv_pool=None,
                    swa_kv_pool=None,
                    swa_req_ring_size=None,
                )
                allocator = _build(platform, device="meta", kvcache=kvcache)
                mapping = allocator.full_to_swa_index_mapping
                self.assertEqual(mapping.device.type, expected)
                kvcache.register_mapping.assert_called_once_with(mapping)

    def test_alloc_and_free_round_trip_on_either_placement(self):
        """Extend, decode, then free through the plain, grouped and segment
        paths: every slot's SWA peer is released and the mapping cleared, on
        both placements, after the platform patch is gone."""
        for name, platform in self.PLATFORMS:
            with self.subTest(platform=name):
                allocator = _build(platform, device="cpu", kvcache=_cpu_kv_pool())
                mapping = allocator.full_to_swa_index_mapping
                full0, swa0 = (
                    allocator.full_available_size(),
                    allocator.swa_available_size(),
                )

                zero = torch.zeros(3, dtype=torch.int64)
                lens = torch.tensor([5, 3, 4], dtype=torch.int64)
                extend = allocator.alloc_extend(
                    zero, zero, lens, lens, zero - 1, int(lens.sum())
                )
                rows = list(extend.split(lens.tolist()))
                decode = allocator.alloc_decode(
                    lens + 1, lens + 1, torch.stack([r[-1] for r in rows])
                )
                rows = [torch.cat((r, decode[i : i + 1])) for i, r in enumerate(rows)]

                live = torch.cat(rows).long()
                peers = mapping[live]
                self.assertTrue(bool((peers > 0).all()))
                self.assertEqual(peers.unique().numel(), live.numel())

                allocator.free(rows[0].long())
                allocator.free_group_begin()
                allocator.free(rows[1].long())
                allocator.free_group_end()
                allocator.free_segment(rows[2].long(), start_pos=0)

                self.assertFalse(mapping[:-1].any())
                self.assertEqual(int(mapping[-1]), -1)
                self.assertEqual(
                    (allocator.full_available_size(), allocator.swa_available_size()),
                    (full0, swa0),
                )


if __name__ == "__main__":
    unittest.main()

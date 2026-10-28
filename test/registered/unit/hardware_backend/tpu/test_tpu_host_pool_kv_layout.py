"""The TPU fused host pool must mirror the device's page shape, not rebuild it.

Regression guard for CI run 35639502143, where KV offloading crashed at
startup under the seq-along-lane (head-major) KV layout:

    RuntimeError: index_put_(): value tensor of shape [256, 4, 4, 128]
    cannot be broadcast to indexing result of shape [1, 16, 32, 4, 256]

The host pool built its page from token-major geometry
(page_size, kv_heads_packed, kv_packing, padded_head_dim) regardless of what
the device pool actually allocated. The two layouts have the SAME byte count,
so the pool sized correctly and only failed at the first scatter -- which is
why a size-based check would not have caught it and these cases assert shape.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.kv_pool_host import TpuFusedMHATokenToKVPoolHost

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

PAGE_SIZE = 256
LAYER_NUM = 4
# Same element count either way -- that equality is what hid the bug.
TOKEN_MAJOR_PAGE = (PAGE_SIZE, 4, 4, 128)  # page, heads_packed, packing, head_dim
SEQ_ALONG_LANE_PAGE = (16, 32, 4, PAGE_SIZE)  # 2H, head_dim_lanes/p, p, page


class _FakeDevicePool:
    def __init__(self, page_shape):
        self._fused = torch.empty((2,) + page_shape, dtype=torch.float32)
        self.layer_num = LAYER_NUM
        self.start_layer = 0
        self.device = "cpu"

    def get_fused_kv_buffer(self, layer_id):
        return self._fused


def _host_pool(page_shape, page_num=8):
    pool = TpuFusedMHATokenToKVPoolHost.__new__(TpuFusedMHATokenToKVPoolHost)
    pool.device_pool = _FakeDevicePool(page_shape)
    pool.page_size = PAGE_SIZE
    pool.dtype = torch.float32
    pool.page_num = page_num
    pool.pin_memory = False
    pool.device = "cpu"
    pool.allocator = None
    pool.size_per_token = pool.get_size_per_token()
    return pool


def _buffer_dims(pool):
    """Dims init_kv_buffer would allocate, without allocating them."""
    seen = {}

    def _record(dims, **kwargs):
        seen["dims"] = tuple(dims)
        return None

    with patch.dict(
        "sglang_tpu.kv_pool_host.ALLOC_MEMORY_FUNCS",
        {"cpu": _record},
    ):
        pool.init_kv_buffer()
    return seen["dims"]


class TestTpuHostPoolKvLayout(unittest.TestCase):
    def test_host_page_mirrors_device_page_in_both_layouts(self):
        # The invariant load_commit relies on: the host page IS the device
        # page, so `device[layer][dst] = src[:, layer]` broadcasts. Under the
        # old code the seq-along-lane case produced the token-major 6-tuple.
        for name, page_shape in (
            ("token_major", TOKEN_MAJOR_PAGE),
            ("seq_along_lane", SEQ_ALONG_LANE_PAGE),
        ):
            with self.subTest(layout=name):
                pool = _host_pool(page_shape)
                self.assertEqual(
                    _buffer_dims(pool), (pool.page_num, LAYER_NUM) + page_shape
                )

    def test_size_per_token_is_layout_independent(self):
        # Equal page bytes must give equal per-token bytes. This is the
        # coincidence that let the old, wrong shape size the pool correctly;
        # pinning it keeps a future change from "fixing" sizing by reaching
        # back for layout-specific attributes.
        token_major = _host_pool(TOKEN_MAJOR_PAGE).size_per_token
        seq_along_lane = _host_pool(SEQ_ALONG_LANE_PAGE).size_per_token
        self.assertEqual(token_major, seq_along_lane)
        page_elems = PAGE_SIZE * 4 * 4 * 128
        self.assertEqual(
            token_major,
            page_elems * torch.finfo(torch.float32).bits // 8 // PAGE_SIZE * LAYER_NUM,
        )


if __name__ == "__main__":
    unittest.main()

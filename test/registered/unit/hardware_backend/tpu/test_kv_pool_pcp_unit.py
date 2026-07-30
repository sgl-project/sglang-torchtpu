# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Pure-Python unit tests for TpuPcpFusedMHATokenToKVPool.

Cover the ``__init__`` validation and per-rank size arithmetic without
allocating any real KV buffers — the parent ``TpuFusedMHATokenToKVPool``
constructor is patched so tests run on any host (no TPU required).

The wire we care about here is the ``size`` value the child hands to the
parent: a regression that forgets ``// cp_group_size`` or under-rounds the
alignment would fit in HBM at small models but silently OOM at 480B.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.tpu.kv_pool_pcp import TpuPcpFusedMHATokenToKVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_DEFAULT_KWARGS = dict(
    page_size=128,
    dtype=torch.bfloat16,
    head_num=8,
    head_dim=128,
    layer_num=2,
    device="cpu",
    enable_memory_saver=False,
)


class _Recorder:
    """Records the kwargs the parent __init__ was called with."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, instance, *args, **kwargs) -> None:
        # Fake the tiny amount of parent state the child logs after super().
        self.calls.append({"args": args, "kwargs": kwargs})
        instance.num_blocks = kwargs["size"] // kwargs["page_size"] + 1
        instance.page_size = kwargs["page_size"]


class TpuPcpFusedMHATokenToKVPoolInitTest(CustomTestCase):
    def _construct(self, *, size: int, cp_group_size: int, **overrides):
        recorder = _Recorder()
        kwargs = {**_DEFAULT_KWARGS, **overrides}
        with patch(
            "sglang.srt.hardware_backend.tpu.kv_pool_pcp.TpuFusedMHATokenToKVPool.__init__",
            new=lambda self, **kw: recorder(self, **kw),
        ):
            pool = TpuPcpFusedMHATokenToKVPool(
                size=size, cp_group_size=cp_group_size, **kwargs
            )
        return pool, recorder

    def test_cp_group_size_zero_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "cp_group_size must be >= 1"):
            self._construct(size=1024, cp_group_size=0)

    def test_cp_group_size_negative_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "cp_group_size must be >= 1"):
            self._construct(size=1024, cp_group_size=-1)

    def test_page_size_not_divisible_raises(self) -> None:
        with self.assertRaisesRegex(
            ValueError, r"page_size \(128\) must be divisible by cp_group_size \(3\)"
        ):
            self._construct(size=1024, cp_group_size=3, page_size=128)

    def test_per_rank_size_math(self) -> None:
        # cp_group_size=4, page_size=128, size = 32 pages worth of tokens.
        # aligned_size = 4096 (already aligned to page_size * cp_group_size).
        # per_rank_size = 4096 / 4 = 1024.
        pool, recorder = self._construct(size=4096, cp_group_size=4)
        self.assertEqual(len(recorder.calls), 1)
        parent_kwargs = recorder.calls[0]["kwargs"]
        self.assertEqual(parent_kwargs["size"], 1024)
        self.assertEqual(pool.cp_group_size, 4)

    def test_size_auto_rounds_down_when_unaligned(self) -> None:
        # size=4200 with cp_group_size=4, page_size=128 -> alignment is
        # page_size * cp_group_size = 512. Rounded down to 4096; per-rank
        # 4096 / 4 = 1024. The extra 104 tokens are dropped.
        pool, recorder = self._construct(size=4200, cp_group_size=4)
        parent_kwargs = recorder.calls[0]["kwargs"]
        self.assertEqual(parent_kwargs["size"], 1024)
        self.assertEqual(pool.cp_group_size, 4)

    def test_cp_group_size_one_is_noop_on_size(self) -> None:
        # cp_group_size=1 leaves aligned_size==size; per-rank == full size.
        pool, recorder = self._construct(size=4096, cp_group_size=1)
        parent_kwargs = recorder.calls[0]["kwargs"]
        self.assertEqual(parent_kwargs["size"], 4096)
        self.assertEqual(pool.cp_group_size, 1)

    def test_passes_through_construction_kwargs(self) -> None:
        pool, recorder = self._construct(
            size=4096,
            cp_group_size=2,
            head_num=16,
            head_dim=256,
            layer_num=4,
            v_head_dim=128,
        )
        parent_kwargs = recorder.calls[0]["kwargs"]
        self.assertEqual(parent_kwargs["head_num"], 16)
        self.assertEqual(parent_kwargs["head_dim"], 256)
        self.assertEqual(parent_kwargs["layer_num"], 4)
        self.assertEqual(parent_kwargs["v_head_dim"], 128)


if __name__ == "__main__":
    unittest.main()

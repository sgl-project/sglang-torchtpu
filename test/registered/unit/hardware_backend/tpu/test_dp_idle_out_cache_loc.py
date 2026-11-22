"""An idle DP batch relabelled as decode gets its empty locations in the
allocator dtype by allocation, not by an on-device cast (which compiled an s32[0]
program at a server's first idle relabel). CPU only."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.compilation.decoder_compile_policy import DecoderTpuCompilePolicy

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _policy(dtype):
    policy = object.__new__(DecoderTpuCompilePolicy)
    policy._dummy_out_cache_loc_dtype = lambda mode: dtype
    policy._match_busiest_dp_decode_rows = lambda fb: None
    return policy


class TestIdleOutCacheLoc(CustomTestCase):
    def test_empty_locations_are_allocated_not_cast(self):
        fb = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            out_cache_loc=torch.empty(0, dtype=torch.int64),
        )
        calls = []
        real_to = torch.Tensor.to

        def to(self, *args, **kwargs):
            calls.append(kwargs)
            return real_to(self, *args, **kwargs)

        from unittest.mock import patch

        with patch.object(torch.Tensor, "to", to):
            _policy(torch.int32)._normalize_dp_idle_forward(fb)
        self.assertEqual(fb.forward_mode, ForwardMode.DECODE)
        self.assertEqual(fb.out_cache_loc.dtype, torch.int32)
        self.assertEqual(fb.out_cache_loc.shape, (0,))
        self.assertEqual(calls, [])

    def test_nonempty_locations_still_cast_and_matching_dtype_untouched(self):
        fb = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            out_cache_loc=torch.tensor([3, 4], dtype=torch.int64),
        )
        _policy(torch.int32)._normalize_dp_idle_forward(fb)
        self.assertTrue(
            torch.equal(fb.out_cache_loc, torch.tensor([3, 4], dtype=torch.int32))
        )
        same = torch.empty(0, dtype=torch.int32)
        fb = SimpleNamespace(forward_mode=ForwardMode.IDLE, out_cache_loc=same)
        _policy(torch.int32)._normalize_dp_idle_forward(fb)
        self.assertIs(fb.out_cache_loc, same)


if __name__ == "__main__":
    unittest.main()

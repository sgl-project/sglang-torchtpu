"""CPU checks of the GLM-5.3-Flash megakernel's serving-side bookkeeping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.glm_flash_megakernel import ops

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestGlmFlashMegakernelBuckets(CustomTestCase):
    def test_buckets_cover_every_batch_up_to_max(self):
        """Every live batch size maps to a compiled bucket; the largest is max_req itself."""
        for max_req in (1, 3, 8, 16):
            bs = ops._buckets(max_req)
            self.assertEqual(bs[-1], max_req)
            for n in range(1, max_req + 1):
                self.assertTrue(any(b >= n for b in bs), (max_req, n))


class TestServingGuards(CustomTestCase):
    def test_expert_parallel_is_rejected_at_startup(self):
        """EP>1 ranks hold whole experts, which the kernel's per-rank expert DMA cannot
        stream; the hook must refuse at startup, not fail Mosaic verification mid-serve."""
        ops.require_expert_tp(1)
        with self.assertRaises(ValueError):
            ops.require_expert_tp(8)


class TestPoolBookkeeping(CustomTestCase):
    def test_record_bytes_match_the_pool_layout(self):
        """The kernel's record stride is the pool's: a mismatch reads the wrong pooled key."""
        from sglang_tpu.glm_flash_megakernel import layout as PL
        from sglang_tpu.kv_pool import TpuSplitMLATokenToKVPool

        for d in (64, 128, 256):
            self.assertEqual(
                PL.record_bytes(d), TpuSplitMLATokenToKVPool.index_key_record_bytes(d)
            )

    def test_closing_pools_are_exactly_those_ending_in_the_step(self):
        """A pool closes when its last token is one of the step's rows; the step keys
        and stores those and no others."""
        import numpy as np

        from sglang_tpu.glm_flash_megakernel import layout as PL

        kp = 4
        for Q in (1, 2, 4):
            for pos in range(0, 40):
                first, count = PL.closing_pools(np.int32(pos), Q, kp)
                want = [
                    p for p in range(0, 20) if pos <= p * kp + kp - 1 <= pos + Q - 1
                ]
                got = list(range(int(first), int(first) + int(count)))
                self.assertEqual(got, want, (Q, pos))


class TestDecodeSnapshot(CustomTestCase):
    def test_rows_on_a_track_boundary_land_in_their_track_slots(self):
        """A plain-decode step copies the state row of each request on a track boundary
        into its track slot; the index rows reach the copy op padded to its bucket."""
        from sglang_tpu.mamba_state_copy import copy_bucket_size

        states = [
            torch.arange(16.0).reshape(16, 1).repeat(1, 2),
            torch.arange(16.0).reshape(16, 1, 1).repeat(1, 2, 2) + 100,
        ]
        before = [s.clone() for s in states]
        calls = []

        # The state-copy op needs a TPU, so its contract runs on host tensors here:
        # pairs [0, count) copy slot src[i] to slot dst[i], later pairs are padding.
        def copy(pools, src, dst, count):
            calls.append((src, dst, count))
            for pool in pools:
                for i in range(int(count[0])):
                    pool[int(dst[i])] = pool[int(src[i])].clone()

        mk = object.__new__(ops.GlmFlashMegakernel)
        # Request rows 1..3 own state slots 5..7; track ids 10..12 are slots 11..13.
        mk._lin = SimpleNamespace(
            _slot_indices=lambda rows: ((rows + 4).long(), (rows + 4).to(torch.int32)),
            _translate_mamba_indices=lambda ids: ids + 1,
        )
        mk._mamba = SimpleNamespace(_copy_states=lambda: states)
        batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True, False, True]),
            req_pool_indices=torch.tensor([1, 2, 3]),
            mamba_track_indices=torch.tensor([10, 11, 12]),
        )
        with patch(
            "sglang_tpu.mamba_state_copy.compiled_state_copy", return_value=copy
        ):
            mk._decode_snapshot(batch)

        self.assertEqual(len(calls), 1)
        src, dst, count = calls[0]
        self.assertEqual(tuple(src.shape), (copy_bucket_size(3),))
        self.assertEqual(tuple(dst.shape), (copy_bucket_size(3),))
        self.assertEqual((src.dtype, dst.dtype), (torch.int32, torch.int32))
        self.assertEqual(count.tolist(), [3])
        for was, now in zip(before, states):
            want = was.clone()
            want[11], want[13] = was[5], was[7]
            # Slot 0 is the reserved null slot; the row off the boundary lands there.
            self.assertTrue(torch.equal(now[1:], want[1:]))


if __name__ == "__main__":
    unittest.main()

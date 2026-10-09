"""CPU coverage for building a host-padded EXTEND forward's metadata on TPU."""

import unittest
from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.hardware_backend.tpu import host_extend_padding as hep
from sglang.srt.hardware_backend.tpu import patches
from sglang.srt.model_executor import forward_batch_info, host_token_padding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@contextmanager
def extend_buckets(token_bucket=16, batch_cap=4):
    """Register an extend bucket the way the TPU compile policy does."""

    def bucket(num_tokens, num_reqs):
        return None if num_reqs > batch_cap else max(token_bucket, num_tokens)

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(host_token_padding, "_extend_token_bucket", bucket)
        )
        stack.enter_context(
            patch.object(host_token_padding, "_extend_batch_cap", batch_cap)
        )
        yield


def make_batch(mode=ForwardMode.EXTEND):
    return SimpleNamespace(
        forward_mode=mode,
        extend_lens=[3, 2],
        prefix_lens=[11, 0],
        extend_num_tokens=5,
        seq_lens=torch.tensor([14, 2]),
        seq_lens_cpu=torch.tensor([14, 2]),
        req_pool_indices=torch.tensor([5, 7]),
        req_pool_indices_cpu=torch.tensor([5, 7]),
        orig_seq_lens=torch.tensor([14, 2], dtype=torch.int32),
        orig_seq_lens_cpu=[14, 2],
        out_cache_loc=torch.arange(100, 105),
        out_cache_loc_cpu=torch.arange(100, 105),
    )


class NoDeviceRead(torch.Tensor):
    """A tensor whose values must not be read back to the host."""

    def tolist(self):
        raise AssertionError("device read")


class TestPlan(CustomTestCase):
    def test_plain_extend_with_bucket(self):
        with extend_buckets():
            plan = hep.plan_extend_padding(make_batch())
        self.assertEqual(plan.prefix_lens, [11, 0])
        self.assertEqual(plan.seq_lens, [3, 2])
        self.assertEqual(plan.num_tokens, 5)
        self.assertEqual(plan.padded_num_tokens, 16)
        self.assertEqual(plan.padded_batch_size, 4)

    def test_no_bucket_or_not_plain_extend(self):
        # Other tests install a compile policy, which registers a bucket.
        with (
            patch.object(host_token_padding, "_extend_token_bucket", None),
            patch.object(host_token_padding, "_extend_batch_cap", None),
        ):
            self.assertIsNone(hep.plan_extend_padding(make_batch()))
        with extend_buckets():
            for mode in (ForwardMode.MIXED, ForwardMode.DECODE):
                self.assertIsNone(hep.plan_extend_padding(make_batch(mode)))
            over_cap = make_batch()
            over_cap.extend_lens = [1] * 5
            over_cap.prefix_lens = [0] * 5
            self.assertIsNone(hep.plan_extend_padding(over_cap))


class TestPositions(CustomTestCase):
    def test_host_positions_pad_tokens_and_requests(self):
        positions, start_loc = hep.host_positions([11, 0], [3, 2], 8, 4, torch.int32)
        self.assertEqual(positions.tolist(), [11, 12, 13, 0, 1, 0, 0, 0])
        self.assertEqual(start_loc.tolist(), [0, 3, 0, 0])
        self.assertEqual(start_loc.dtype, torch.int32)

    def test_patched_position_uses_the_plan_without_device_reads(self):
        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    forward_batch_info,
                    "compute_position_torch",
                    forward_batch_info.compute_position_torch,
                )
            )
            stack.enter_context(
                patch.object(patches, "_compute_position_torch_patched", False)
            )
            patches._patch_compute_position_torch()
            prefix = torch.tensor([11, 0], dtype=torch.int32).as_subclass(NoDeviceRead)
            seq = torch.tensor([3, 2], dtype=torch.int32).as_subclass(NoDeviceRead)
            with extend_buckets():
                token = hep.set_extend_padding(hep.plan_extend_padding(make_batch()))
            try:
                positions, start_loc = forward_batch_info.compute_position_torch(
                    prefix, seq
                )
            finally:
                hep.reset_extend_padding(token)
            self.assertEqual(positions.shape[0], 16)
            self.assertEqual(positions[:5].tolist(), [11, 12, 13, 0, 1])
            self.assertEqual(torch.Tensor(start_loc).tolist(), [0, 3, 0, 0])

            # Without a plan it reads the lengths and leaves them unpadded.
            positions, start_loc = forward_batch_info.compute_position_torch(
                torch.tensor([11, 0]), torch.tensor([3, 2], dtype=torch.int32)
            )
            self.assertEqual(positions.tolist(), [11, 12, 13, 0, 1])
            self.assertEqual(start_loc.tolist(), [0, 3])


class TestApply(CustomTestCase):
    def test_reuploads_real_sized_fields_padded(self):
        batch = make_batch()
        with extend_buckets():
            plan = hep.plan_extend_padding(batch)
        forward_batch = SimpleNamespace(
            input_ids=torch.zeros(16, dtype=torch.int64),
            batch_size=2,
            global_num_token_non_padded=torch.tensor(16, dtype=torch.int32),
            global_num_token_non_padded_cpu=16,
        )
        hep.apply_extend_padding(forward_batch, batch, plan)
        self.assertEqual(forward_batch.global_num_token_non_padded_cpu, 5)
        self.assertEqual(int(forward_batch.global_num_token_non_padded), 5)
        self.assertEqual(
            forward_batch.out_cache_loc.tolist(), list(range(100, 105)) + [0] * 11
        )
        self.assertEqual(forward_batch.extend_seq_lens.tolist(), [3, 2, 0, 0])
        self.assertEqual(forward_batch.extend_prefix_lens.tolist(), [11, 0, 0, 0])
        self.assertEqual(forward_batch.seq_lens.tolist(), [14, 2, 0, 0])
        self.assertEqual(forward_batch.req_pool_indices.tolist(), [5, 7, 0, 0])
        self.assertEqual(forward_batch.orig_seq_lens.tolist(), [14, 2, 0, 0])
        self.assertEqual(forward_batch.orig_seq_lens.dtype, torch.int32)


class TestInitNewPatch(CustomTestCase):
    def test_plan_is_current_inside_init_new_and_applied_after(self):
        seen = []

        def fake_init_new(cls, batch, model_runner, **kwargs):
            seen.append(hep.current_extend_padding())
            return SimpleNamespace(
                input_ids=torch.zeros(16, dtype=torch.int64),
                batch_size=2,
                global_num_token_non_padded=None,
                global_num_token_non_padded_cpu=16,
            )

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(ForwardBatch, "init_new", classmethod(fake_init_new))
            )
            stack.enter_context(
                patch.object(
                    patches, "_forward_batch_host_extend_padding_patched", False
                )
            )
            patches._patch_forward_batch_host_extend_padding()
            with extend_buckets():
                forward_batch = ForwardBatch.init_new(
                    make_batch(), None, return_hidden_states_before_norm=False
                )
                ForwardBatch.init_new(
                    make_batch(ForwardMode.DECODE),
                    None,
                    return_hidden_states_before_norm=False,
                )
        self.assertEqual(seen[0].padded_num_tokens, 16)
        self.assertIsNone(seen[1])
        self.assertIsNone(hep.current_extend_padding())
        self.assertEqual(forward_batch.global_num_token_non_padded_cpu, 5)
        self.assertEqual(forward_batch.seq_lens.tolist(), [14, 2, 0, 0])


if __name__ == "__main__":
    unittest.main()

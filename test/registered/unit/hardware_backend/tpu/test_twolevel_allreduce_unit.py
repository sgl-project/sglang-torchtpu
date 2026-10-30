"""CPU-side logic of the two-level Pallas all-reduce and the K3 merged MoE reduce.

The kernel itself runs on TPU in test_twolevel_allreduce.py; these cover the
eligibility gate, the fallbacks to the original collective and
the default (concatenating) merged_moe_all_reduce.
"""

import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu import twolevel_allreduce as tl
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestEligibility(CustomTestCase):
    def test_single_token_bf16_lane_aligned(self):
        self.assertTrue(tl.eligible(torch.zeros(1, 7168, dtype=torch.bfloat16)))

    def test_rejects_others(self):
        for t in (
            torch.zeros(4, 7168, dtype=torch.bfloat16),  # more than one token
            torch.zeros(1, 7168, dtype=torch.float32),
            torch.zeros(1, 7100, dtype=torch.bfloat16),  # not a 128 multiple
            torch.zeros(7168, dtype=torch.bfloat16),
            torch.zeros(1, tl.MAX_ELEMS + 128, dtype=torch.bfloat16),
        ):
            with self.subTest(shape=tuple(t.shape), dtype=t.dtype):
                self.assertFalse(tl.eligible(t))


class TestFallbacks(CustomTestCase):
    def test_all_reduce_calls_original_when_not_registered(self):
        original = mock.Mock(return_value="orig")
        group = mock.Mock(world_size=32)
        x = torch.zeros(1, 7168, dtype=torch.bfloat16)
        with mock.patch.object(tl, "_READY", False):
            self.assertEqual(tl.all_reduce_around(original, group, x), "orig")
        original.assert_called_once_with(group, x)

    def test_merged_calls_original_when_not_registered(self):
        original = mock.Mock(return_value=("a", "b"))
        latent = torch.zeros(1, 3584, dtype=torch.bfloat16)
        shared = torch.zeros(1, 7168, dtype=torch.bfloat16)
        with mock.patch.object(tl, "_READY", False):
            self.assertEqual(
                tl.merged_moe_all_reduce_around(original, latent, shared), ("a", "b")
            )

    def test_merged_falls_back_on_mixed_dtypes(self):
        """A non-bf16 shared output is not cast into the packed exchange."""
        original = mock.Mock(return_value=("a", "b"))
        latent = torch.zeros(1, 3584, dtype=torch.bfloat16)
        shared = torch.zeros(1, 7168, dtype=torch.float32)
        with mock.patch.object(tl, "_READY", True):
            self.assertEqual(
                tl.merged_moe_all_reduce_around(original, latent, shared), ("a", "b")
            )
        original.assert_called_once_with(latent, shared)


class TestRegistration(CustomTestCase):
    def _register(self, enabled, world=32, nnodes=4, seen=None, compile_bs=(1, 4)):
        def gather(out, value, group=None):
            out[:] = seen if seen is not None else [value] * len(out)

        with (
            tpu_envs.SGLANG_TPU_ENABLE_TWOLEVEL_ALLREDUCE.override(enabled),
            mock.patch.object(tl, "register") as reg,
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_world_group",
                return_value=mock.Mock(world_size=world, cpu_group="cpu"),
            ),
            mock.patch("torch.distributed.all_gather_object", side_effect=gather),
            mock.patch("torch.distributed.get_world_size", return_value=world),
            mock.patch(
                "sglang.srt.distributed.parallel_state."
                "get_tensor_model_parallel_world_size",
                return_value=world,
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_server_args",
                return_value=mock.Mock(nnodes=nnodes, tpu_compile_bs=list(compile_bs)),
            ),
        ):
            tl.register_from_parallel_state()
        return reg

    def test_off_by_default(self):
        self._register(False).assert_not_called()

    def test_measured_topologies_register(self):
        self._register(True).assert_called_once_with(32, 8)
        self._register(True, world=8, nnodes=1).assert_called_once_with(8, 8)

    def test_unmeasured_topologies_keep_xla(self):
        # (33, 4) would map to 4 x 8 without the divisibility check.
        for world, nnodes in ((64, 8), (16, 2), (4, 1), (33, 4), (30, 4)):
            with self.subTest(world=world, nnodes=nnodes):
                with self.assertLogs(tl.logger, "WARNING"):
                    reg = self._register(True, world=world, nnodes=nnodes)
                reg.assert_not_called()

    def test_ranks_must_agree(self):
        seen = [True] * 31 + [False]
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                with self.assertRaisesRegex(RuntimeError, "same on every rank"):
                    self._register(enabled, seen=seen)

    def test_warns_without_a_bs1_compile_bucket(self):
        with self.assertLogs(tl.logger, "WARNING") as logs:
            self._register(True, compile_bs=(4, 8))
        self.assertIn("no batch size 1", "\n".join(logs.output))


class TestMergedMoeAllReduceDefault(CustomTestCase):
    def test_concat_reduce_split(self):
        from sglang.srt.models import kimi_k3

        latent = torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)
        shared = torch.arange(10, 16, dtype=torch.float32).reshape(1, 6)
        with mock.patch.object(
            kimi_k3, "tensor_model_parallel_all_reduce", side_effect=lambda t: t * 2
        ) as ar:
            got_latent, got_shared = kimi_k3.merged_moe_all_reduce(latent, shared)
        ar.assert_called_once()  # one collective for both
        torch.testing.assert_close(got_latent, latent * 2)
        torch.testing.assert_close(got_shared, (shared * 2).to(torch.bfloat16))


if __name__ == "__main__":
    unittest.main()

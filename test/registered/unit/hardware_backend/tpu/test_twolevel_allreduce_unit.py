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
from sglang_tpu import rdma_all_gather as rdma_ag
from sglang_tpu import twolevel_allreduce as tl
from sglang_tpu.envs import envs as tpu_envs
from sglang_tpu.hooks import HOOKS
from sglang_tpu.hooks import rdma_all_gather as rdma_ag_hook

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

    def test_merged_passes_norm_to_original_when_not_registered(self):
        original = mock.Mock(return_value=("a", "b"))
        latent = torch.zeros(1, 3584, dtype=torch.bfloat16)
        shared = torch.zeros(1, 7168, dtype=torch.bfloat16)
        norm = mock.Mock()
        with mock.patch.object(tl, "_READY", False):
            self.assertEqual(
                tl.merged_moe_all_reduce_around(original, latent, shared, norm),
                ("a", "b"),
            )
        original.assert_called_once_with(latent, shared, norm)

    def test_merged_norm_unfused_without_2n_op(self):
        """Registered pair op but no fused-norm op: reduce, then the norm."""
        latent = torch.zeros(1, 3584, dtype=torch.bfloat16)
        shared = torch.zeros(1, 7168, dtype=torch.bfloat16)
        norm = mock.Mock(return_value="normed")
        pair = mock.Mock(return_value=("l", "s"))
        with (
            mock.patch.object(tl, "_READY", True),
            mock.patch.dict(tl._OPS, {2: pair}, clear=True),
        ):
            got = tl.merged_moe_all_reduce_around(mock.Mock(), latent, shared, norm)
        self.assertEqual(got, ("normed", "s"))
        norm.assert_called_once_with("l")

    def test_row_parallel_linear_calls_original_without_ar1r_op(self):
        """No fused attention-residual op: o_proj keeps its own all-reduce."""
        original = mock.Mock(return_value=("y", None))
        layer = mock.Mock()
        x = torch.zeros(1, 384, dtype=torch.bfloat16)
        with mock.patch.dict(tl._OPS, {}, clear=True):
            self.assertEqual(
                tl.row_parallel_linear_forward_around(original, layer, x),
                ("y", None),
            )
        original.assert_called_once_with(
            layer, x, skip_all_reduce=False, forward_batch=None, output_tensor=None
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


class TestTwoShotAndRdmaAllGather(CustomTestCase):
    def test_eligible_twoshot(self):
        self.assertTrue(
            tl.eligible_twoshot(torch.zeros(16, 4096, dtype=torch.bfloat16), 8)
        )
        for bad in (
            torch.zeros(4096, dtype=torch.bfloat16),
            torch.zeros(0, 4096, dtype=torch.bfloat16),
            torch.zeros(16, 4096, dtype=torch.float32),
            torch.zeros(16, 1152, dtype=torch.bfloat16),
            torch.zeros(64, 4096, dtype=torch.bfloat16),
        ):
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype):
                self.assertFalse(tl.eligible_twoshot(bad, 8))

    def test_all_reduce_around_dispatches_twoshot_and_falls_back(self):
        x = torch.zeros(8, 4096, dtype=torch.bfloat16)
        tp_group = mock.Mock()
        attn_tp_group = mock.Mock()
        other_group = mock.Mock()
        par = mock.Mock(attn_tp_size=8, attn_tp_group=attn_tp_group)
        original = mock.Mock(return_value="orig")
        with (
            mock.patch.object(tl, "_READY", False),
            mock.patch.object(tl, "_TWOSHOT_WORLD", 8),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=tp_group,
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_parallel",
                return_value=par,
            ),
            mock.patch(
                "sglang_tpu.twoshot_all_reduce.twoshot_all_reduce_tensor",
                side_effect=lambda t, w: f"twoshot:{w}:{tuple(t.shape)}",
            ),
        ):
            self.assertEqual(
                tl.all_reduce_around(original, tp_group, x), "twoshot:8:(8, 4096)"
            )
            self.assertEqual(
                tl.all_reduce_around(original, attn_tp_group, x),
                "twoshot:8:(8, 4096)",
            )
            self.assertEqual(tl.all_reduce_around(original, other_group, x), "orig")
        original.assert_called_once_with(other_group, x)

    def test_check_ranks_agree_custom_env_skips_split_barrier_check(self):
        calls = []

        def gather(out, value, group=None):
            calls.append(value)
            out[:] = [value] * len(out)

        with (
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_world_group",
                return_value=mock.Mock(world_size=8, cpu_group="cpu"),
            ),
            mock.patch("torch.distributed.all_gather_object", side_effect=gather),
        ):
            tl._check_ranks_agree(True, "SGLANG_TPU_ENABLE_RDMA_ALL_GATHER")
        self.assertEqual(calls, [True])

        def mismatch_gather(out, value, group=None):
            out[:] = [True] * 7 + [False]

        with (
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_world_group",
                return_value=mock.Mock(world_size=8, cpu_group="cpu"),
            ),
            mock.patch(
                "torch.distributed.all_gather_object", side_effect=mismatch_gather
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "SGLANG_TPU_ENABLE_RDMA_ALL_GATHER"
            ):
                tl._check_ranks_agree(True, "SGLANG_TPU_ENABLE_RDMA_ALL_GATHER")

    def test_can_use_rdma_all_gather_checks_device_and_chunk_limits(self):
        tp_group = mock.Mock(world_size=8)
        other_group = mock.Mock(world_size=8)

        def fake_block(shape, *, dtype=torch.bfloat16, device_type="tpu"):
            numel = 1
            for dim in shape:
                numel *= dim
            elem = 2 if dtype == torch.bfloat16 else 4
            return mock.Mock(
                shape=shape,
                dtype=dtype,
                device=mock.Mock(type=device_type),
                dim=lambda: len(shape),
                numel=lambda: numel,
                element_size=lambda: elem,
            )

        with (
            mock.patch.object(rdma_ag, "_RDMA_AG_WORLD", 8),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=tp_group,
            ),
        ):
            # Rejects CPU tensor.
            self.assertFalse(
                rdma_ag.can_use_rdma_all_gather(
                    tp_group, torch.zeros(1024, 8192, dtype=torch.bfloat16)
                )
            )
            # Rejects non-TP group.
            self.assertFalse(
                rdma_ag.can_use_rdma_all_gather(other_group, fake_block((1024, 8192)))
            )
            # Accepts 16 MiB single chunk and 32 MiB / 128 MiB multi-chunk aligned rows.
            self.assertTrue(
                rdma_ag.can_use_rdma_all_gather(tp_group, fake_block((1024, 8192)))
            )
            self.assertTrue(
                rdma_ag.can_use_rdma_all_gather(tp_group, fake_block((2048, 8192)))
            )
            self.assertTrue(
                rdma_ag.can_use_rdma_all_gather(tp_group, fake_block((8192, 8192)))
            )
            # Rejects odd rows across >1 chunk, unaligned cols, and >128 MiB payload.
            self.assertFalse(
                rdma_ag.can_use_rdma_all_gather(tp_group, fake_block((1025, 8192)))
            )
            self.assertFalse(
                rdma_ag.can_use_rdma_all_gather(tp_group, fake_block((16, 100)))
            )
            self.assertFalse(
                rdma_ag.can_use_rdma_all_gather(tp_group, fake_block((8193, 8192)))
            )

    def test_rdma_all_gather_into_tensor_single_and_multichunk_slicing(self):
        calls = []

        def fake_op(x2d):
            calls.append(tuple(x2d.shape))
            return torch.stack([x2d + r for r in range(8)], dim=0)

        with mock.patch.dict(rdma_ag._OPS, {8: fake_op}, clear=True):
            block_32m = torch.arange(2048 * 8192, dtype=torch.bfloat16).reshape(
                2048, 8192
            )
            out_32m = torch.empty((8 * 2048, 8192), dtype=torch.bfloat16)
            rdma_ag.rdma_all_gather_into_tensor(out_32m, block_32m, 8)
            self.assertEqual(calls, [(1024, 8192), (1024, 8192)])
            torch.testing.assert_close(out_32m[:2048], block_32m)
            torch.testing.assert_close(out_32m[2048:4096], block_32m + 1)

    def test_rdma_all_gather_hook_registration_and_dispatch(self):
        from sglang.srt.plugins.hook_registry import HookType

        self.assertIn(
            (
                "sglang.srt.distributed.parallel_state.GroupCoordinator.all_gather_into_tensor",
                rdma_ag_hook.all_gather_into_tensor,
                HookType.AROUND,
            ),
            HOOKS,
        )
        group = mock.Mock(world_size=8)
        output = torch.zeros(8, 128, dtype=torch.bfloat16)
        inp = torch.zeros(1, 128, dtype=torch.bfloat16)
        original = mock.Mock(return_value="fallback")

        with mock.patch("torch.compiler.is_dynamo_compiling", return_value=False):
            self.assertEqual(
                rdma_ag_hook.all_gather_into_tensor(original, group, output, inp),
                "fallback",
            )
        original.assert_called_once_with(group, output, inp)

        with (
            mock.patch("torch.compiler.is_dynamo_compiling", return_value=True),
            mock.patch.object(rdma_ag, "can_use_rdma_all_gather", return_value=True),
            mock.patch.object(rdma_ag, "rdma_all_gather_into_tensor") as run_rdma,
        ):
            self.assertIsNone(
                rdma_ag_hook.all_gather_into_tensor(mock.Mock(), group, output, inp)
            )
        run_rdma.assert_called_once_with(output, inp, 8)


if __name__ == "__main__":
    unittest.main()

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
            mock.patch.object(tl, "_register_twoshot_from_parallel_state"),
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


class TestTwoShotRoute(CustomTestCase):
    """Single-host TP: decode-sized bf16 TP all-reduces go to the two-shot op."""

    def test_eligibility_cap_and_layout(self):
        cap_rows = tl.TWOSHOT_TP_ALLREDUCE_MAX_BYTES // (4096 * 2)
        for shape, dtype, want in (
            ((1, 4096), torch.bfloat16, True),
            ((cap_rows, 4096), torch.bfloat16, True),
            ((cap_rows + 1, 4096), torch.bfloat16, False),  # over the cap
            ((32, 4096), torch.float32, False),
            ((32, 4000), torch.bfloat16, False),  # columns not 128 per rank
            ((0, 4096), torch.bfloat16, False),
            ((4096,), torch.bfloat16, False),
        ):
            with self.subTest(shape=shape, dtype=dtype):
                t = torch.zeros(shape, dtype=dtype)
                self.assertEqual(tl.twoshot_eligible(t, 8), want)

    def _dispatch(self, x, *, group_is_tp=True, ready=False):
        tp_group, other = mock.Mock(), mock.Mock()
        original = mock.Mock(return_value="xla")
        twoshot, twolevel = (
            mock.Mock(return_value="twoshot"),
            mock.Mock(return_value="twolevel"),
        )
        with (
            mock.patch.object(tl, "_TWOSHOT_WORLD", 8),
            mock.patch.object(tl, "_READY", ready),
            mock.patch.dict(tl._OPS, {1: twolevel}),
            mock.patch(
                "sglang.srt.distributed.parallel_state.get_tp_group",
                return_value=tp_group,
            ),
            mock.patch(
                "sglang_tpu.twoshot_all_reduce.get_twoshot_all_reduce_op",
                return_value=twoshot,
            ) as get_op,
        ):
            out = tl.all_reduce_around(original, tp_group if group_is_tp else other, x)
        return out, get_op

    def test_dispatch(self):
        small = torch.zeros(32, 4096, dtype=torch.bfloat16)
        out, get_op = self._dispatch(small)
        self.assertEqual(out, "twoshot")
        get_op.assert_called_once_with(8)
        self.assertEqual(self._dispatch(small, group_is_tp=False)[0], "xla")
        large = torch.zeros(4096, 4096, dtype=torch.bfloat16)
        self.assertEqual(self._dispatch(large)[0], "xla")

    def test_single_row_prefers_the_two_level_op(self):
        row = torch.zeros(1, 4096, dtype=torch.bfloat16)
        self.assertEqual(self._dispatch(row, ready=True)[0], "twolevel")

    def _register(self, *, world=8, tp=None, nnodes=1):
        with (
            mock.patch.object(tl, "_TWOSHOT_WORLD", 0),
            mock.patch("torch.distributed.get_world_size", return_value=world),
            mock.patch(
                "sglang.srt.distributed.parallel_state."
                "get_tensor_model_parallel_world_size",
                return_value=world if tp is None else tp,
            ),
            mock.patch(
                "sglang.srt.runtime_context.get_server_args",
                return_value=mock.Mock(nnodes=nnodes),
            ),
            mock.patch(
                "sglang_tpu.twoshot_all_reduce.get_twoshot_all_reduce_op"
            ) as build,
        ):
            tl._register_twoshot_from_parallel_state()
            return tl._TWOSHOT_WORLD, build

    def test_registration_needs_tp_on_one_host(self):
        world, build = self._register()
        self.assertEqual(world, 8)
        build.assert_called_once_with(8)
        for kwargs in (
            dict(world=32, nnodes=4),  # the exchange is host-local
            dict(world=8, tp=4),  # the kernel's mesh is the whole world
            dict(world=1),
        ):
            with self.subTest(**kwargs):
                world, build = self._register(**kwargs)
                self.assertEqual(world, 0)
                build.assert_not_called()


if __name__ == "__main__":
    unittest.main()

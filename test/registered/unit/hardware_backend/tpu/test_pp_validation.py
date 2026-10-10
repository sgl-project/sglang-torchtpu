"""Unit tests for the TPU platform's pipeline-parallel launch validation.

Mocks the slicebuilder env setup, so it runs without TPU hardware.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.arg_groups.overrides import declare_resolution, resolution_result
from sglang.srt.model_executor.cuda_graph_config import default_cuda_graph_config
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.tpu_platform import TpuSRTPlatform


def _args(**overrides):
    defaults = dict(
        dtype=None,
        attention_backend=None,
        sampling_backend=None,
        disable_cuda_graph=False,
        cuda_graph_config=default_cuda_graph_config(),
        enable_torch_compile=False,
        disable_overlap_schedule=False,
        mem_fraction_static=None,
        tp_size=1,
        pp_size=1,
        dp_size=1,
        enable_dp_attention=False,
        attn_dp_size=1,
        enable_dp_lm_head=False,
        enable_dynamic_chunking=False,
        page_size=None,
        kv_cache_dtype="auto",
        use_ray=False,
        hicache_io_backend=None,
        max_total_tokens=None,
        speculative_algorithm=None,
        speculative_eagle_topk=None,
        speculative_use_rejection_sampling=False,
        pp_async_batch_depth=0,
        disaggregation_mode="null",
        disaggregation_transfer_backend="mooncake",
        optimistic_prefill_attempts=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _apply(args):
    with patch("sglang_tpu.patches._prepare_tpu_env") as prepare_tpu_env:
        TpuSRTPlatform().apply_server_args_defaults(args)
    return prepare_tpu_env


class TestPipelineParallelValidation(CustomTestCase):
    def test_pp8_tp1_is_accepted_and_sizes_the_slice_to_eight_devices(self):
        args = _args(pp_size=8, tp_size=1, enable_torch_compile=True)
        prepare_tpu_env = _apply(args)
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)
        self.assertIsNone(args.attention_backend)
        self.assertEqual(
            resolution_result(args, "attention_backend"), "pallas_tpu_rpa_v3"
        )
        self.assertTrue(args.enable_torch_compile)

    def test_pipeline_runs_at_async_batch_depth_one(self):
        """The ICI hand-off needs the last stage's outputs received before
        each forward; depth 0 exchanges them after the forward and deadlocks."""
        args = _args(pp_size=8, tp_size=1, enable_torch_compile=True)
        _apply(args)
        self.assertEqual(args.pp_async_batch_depth, 0)
        self.assertEqual(resolution_result(args, "pp_async_batch_depth"), 1)
        args = _args(
            pp_size=8, tp_size=1, enable_torch_compile=True, pp_async_batch_depth=2
        )
        with self.assertRaisesRegex(NotImplementedError, "pp-async-batch-depth 1"):
            _apply(args)
        args = _args(tp_size=8)
        _apply(args)
        self.assertEqual(args.pp_async_batch_depth, 0)

    def test_pipeline_validation_honors_prior_resolution(self):
        args = _args(enable_torch_compile=True)
        declare_resolution(args, "test.pipeline", pp_size=8)
        prepare_tpu_env = _apply(args)
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)
        self.assertEqual(args.pp_size, 1)
        self.assertEqual(args.pp_async_batch_depth, 0)
        self.assertEqual(resolution_result(args, "pp_async_batch_depth"), 1)

    def test_resolved_unsupported_depth_is_rejected_before_environment_setup(self):
        args = _args(pp_size=8)
        declare_resolution(args, "test.pipeline", pp_async_batch_depth=2)
        with patch("sglang_tpu.patches._prepare_tpu_env") as prepare:
            with self.assertRaisesRegex(NotImplementedError, "pp-async-batch-depth 1"):
                TpuSRTPlatform().apply_server_args_defaults(args)
        prepare.assert_not_called()
        self.assertEqual(args.pp_async_batch_depth, 0)

    def test_tp_only_slice_size_is_unchanged(self):
        prepare_tpu_env = _apply(_args(tp_size=8))
        prepare_tpu_env.assert_called_once_with(8, setup_slicebuilder=True)

    def test_rejects_tensor_parallel_stages(self):
        with self.assertRaisesRegex(NotImplementedError, r"requires --tp-size 1"):
            _apply(_args(pp_size=2, tp_size=2))

    def test_rejects_speculative_decoding(self):
        with self.assertRaisesRegex(
            NotImplementedError, r"Speculative decoding.*pipeline parallelism"
        ):
            _apply(_args(pp_size=2, speculative_algorithm="EAGLE3"))

    def test_rejects_dynamic_chunking(self):
        with self.assertRaisesRegex(
            NotImplementedError, r"--enable-dynamic-chunking.*pipeline"
        ):
            _apply(_args(pp_size=2, enable_dynamic_chunking=True))

    def test_rejects_eager_forward(self):
        """An eager forward gets the received activations at the hand-off
        buffer's row count, not the batch's token count."""
        with self.assertRaisesRegex(NotImplementedError, "--enable-torch-compile"):
            _apply(_args(pp_size=2))

    def test_rejection_happens_before_the_slice_env_is_prepared(self):
        with patch("sglang_tpu.patches._prepare_tpu_env") as prepare_tpu_env:
            with self.assertRaises(NotImplementedError):
                TpuSRTPlatform().apply_server_args_defaults(_args(pp_size=2, tp_size=2))
        prepare_tpu_env.assert_not_called()

    def test_single_stage_keeps_every_flag(self):
        args = _args(pp_size=1, tp_size=8, enable_dynamic_chunking=True)
        _apply(args)
        self.assertTrue(args.enable_dynamic_chunking)


if __name__ == "__main__":
    unittest.main()

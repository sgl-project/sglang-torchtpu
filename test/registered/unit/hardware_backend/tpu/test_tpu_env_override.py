"""Unit tests for apply_tpu_env_overrides."""

import os
import unittest

from sglang.srt.hardware_backend.tpu.env_override import (
    _LIBTPU_INIT_ARGS,
    apply_tpu_env_overrides,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTpuEnvOverride(CustomTestCase):
    """Test TPU LIBTPU_INIT_ARGS overrides under different TP/DP configurations."""

    def setUp(self):
        self.orig_env = os.environ.get(_LIBTPU_INIT_ARGS)

    def tearDown(self):
        if self.orig_env is not None:
            os.environ[_LIBTPU_INIT_ARGS] = self.orig_env
        else:
            os.environ.pop(_LIBTPU_INIT_ARGS, None)

    def test_default_env_overrides(self):
        os.environ.pop(_LIBTPU_INIT_ARGS, None)
        apply_tpu_env_overrides(enable_dp_attention=False, tp_size=1)
        flags = os.environ.get(_LIBTPU_INIT_ARGS, "").split()
        self.assertIn(
            "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false", flags
        )
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false", flags
        )
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
            flags,
        )

    def test_dp_attention_tp1_no_sc_offload_flags(self):
        os.environ.pop(_LIBTPU_INIT_ARGS, None)
        apply_tpu_env_overrides(enable_dp_attention=True, tp_size=1)
        flags = os.environ.get(_LIBTPU_INIT_ARGS, "").split()
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false", flags
        )
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
            flags,
        )

    def test_no_dp_attention_tp8_no_sc_offload_flags(self):
        os.environ.pop(_LIBTPU_INIT_ARGS, None)
        apply_tpu_env_overrides(enable_dp_attention=False, tp_size=8)
        flags = os.environ.get(_LIBTPU_INIT_ARGS, "").split()
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false", flags
        )
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
            flags,
        )

    def test_dp_attention_attn_tp1_no_sc_offload_flags(self):
        # tp_size=8, dp_size=8 -> attn_tp_size = 1
        os.environ.pop(_LIBTPU_INIT_ARGS, None)
        apply_tpu_env_overrides(enable_dp_attention=True, tp_size=8, dp_size=8)
        flags = os.environ.get(_LIBTPU_INIT_ARGS, "").split()
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false", flags
        )
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
            flags,
        )

    def test_dp_attention_attn_tp1_with_cp_no_sc_offload_flags(self):
        # tp_size=8, dp_size=2, attn_cp_size=4 -> attn_tp_size = 8 // (2 * 4) = 1
        os.environ.pop(_LIBTPU_INIT_ARGS, None)
        apply_tpu_env_overrides(
            enable_dp_attention=True, tp_size=8, dp_size=2, attn_cp_size=4
        )
        flags = os.environ.get(_LIBTPU_INIT_ARGS, "").split()
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false", flags
        )
        self.assertNotIn(
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
            flags,
        )

    def test_dp_attention_attn_tp_gt_1_appends_sc_offload_flags(self):
        # tp_size=8, dp_size=2 -> attn_tp_size = 4 (> 1)
        os.environ.pop(_LIBTPU_INIT_ARGS, None)
        apply_tpu_env_overrides(enable_dp_attention=True, tp_size=8, dp_size=2)
        flags = os.environ.get(_LIBTPU_INIT_ARGS, "").split()
        self.assertIn(
            "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false", flags
        )
        self.assertIn(
            "--xla_tpu_enable_sparse_core_collective_offload_all_gather=false", flags
        )
        self.assertIn(
            "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
            flags,
        )


if __name__ == "__main__":
    unittest.main()

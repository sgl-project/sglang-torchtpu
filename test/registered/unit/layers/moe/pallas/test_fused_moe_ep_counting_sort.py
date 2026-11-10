# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for EP block counting-sort prefix gather and SC combine v3."""

import contextlib
import importlib.util
import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.envs import envs as tpu_envs

register_cpu_ci(est_time=15, suite="base-a-test-cpu")
register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

_HAS_JAX = importlib.util.find_spec("jax") is not None


@contextlib.contextmanager
def _stub_torch_tpu_if_missing():
    if importlib.util.find_spec("torch_tpu") is not None:
        yield
        return
    torch_tpu = ModuleType("torch_tpu")
    torch_tpu_internal = ModuleType("torch_tpu._internal")
    torch_tpu_pallas = ModuleType("torch_tpu._internal.pallas")
    torch_tpu_pallas.jax_op = MagicMock()
    torch_tpu._internal = torch_tpu_internal
    torch_tpu_internal.pallas = torch_tpu_pallas
    with patch.dict(
        sys.modules,
        {
            "torch_tpu": torch_tpu,
            "torch_tpu._internal": torch_tpu_internal,
            "torch_tpu._internal.pallas": torch_tpu_pallas,
        },
    ):
        yield


class TestFusedMoeEpCountingSortEnvs(CustomTestCase):
    def test_env_defaults_and_overrides(self):
        self.assertFalse(bool(tpu_envs.SGLANG_TPU_FUSED_MOE_EP_COUNTING_SORT.get()))
        self.assertEqual(int(tpu_envs.SGLANG_TPU_RAGGED_GATHER_REDUCE_VERSION.get()), 2)
        with (
            tpu_envs.SGLANG_TPU_FUSED_MOE_EP_COUNTING_SORT.override(True),
            tpu_envs.SGLANG_TPU_RAGGED_GATHER_REDUCE_VERSION.override(3),
        ):
            self.assertTrue(bool(tpu_envs.SGLANG_TPU_FUSED_MOE_EP_COUNTING_SORT.get()))
            self.assertEqual(
                int(tpu_envs.SGLANG_TPU_RAGGED_GATHER_REDUCE_VERSION.get()), 3
            )


@unittest.skipUnless(_HAS_JAX, "Requires JAX")
class TestFusedMoeEpCountingSort(CustomTestCase):
    def test_counting_sort_positions_matches_stable_argsort_and_counts(self):
        import jax.numpy as jnp

        with _stub_torch_tpu_if_missing():
            from sglang_tpu.moe.pallas.fused_moe_op import _counting_sort_positions

        rng = np.random.default_rng(17)
        for n_rows, num_buckets in ((130, 17), (512, 65), (770, 65)):
            with self.subTest(n_rows=n_rows, num_buckets=num_buckets):
                keys_np = rng.integers(0, num_buckets, size=n_rows, dtype=np.int32)
                unsort_idx, bucket_counts = _counting_sort_positions(
                    jnp.asarray(keys_np), num_buckets=num_buckets
                )
                self.assertIsNotNone(unsort_idx)
                self.assertIsNotNone(bucket_counts)

                ref_argsort = np.argsort(keys_np, kind="stable")
                ref_unsort = np.argsort(ref_argsort).astype(np.int32)
                ref_counts = np.bincount(keys_np, minlength=num_buckets).astype(
                    np.int32
                )
                np.testing.assert_array_equal(np.asarray(unsort_idx), ref_unsort)
                np.testing.assert_array_equal(np.asarray(bucket_counts), ref_counts)

    def test_counting_sort_returns_none_when_buckets_exceed_block(self):
        import jax.numpy as jnp

        with _stub_torch_tpu_if_missing():
            from sglang_tpu.moe.pallas.fused_moe_op import _counting_sort_positions

        unsort_idx, bucket_counts = _counting_sort_positions(
            jnp.arange(128, dtype=jnp.int32), num_buckets=256
        )
        self.assertIsNone(unsort_idx)
        self.assertIsNone(bucket_counts)

        unsort_idx_large, bucket_counts_large = _counting_sort_positions(
            jnp.arange(1024, dtype=jnp.int32) % 513, num_buckets=513
        )
        self.assertIsNone(unsort_idx_large)
        self.assertIsNone(bucket_counts_large)

    def test_token_block_alignment_and_partition_counts(self):
        with _stub_torch_tpu_if_missing():
            from sglang_tpu.moe.pallas.sparse_core.ragged_gather_reduce_v3 import (
                partition_counts,
                token_block_alignment,
            )

        tpu_info = SimpleNamespace(
            num_lanes=128,
            sparse_core=SimpleNamespace(num_cores=4, num_subcores=8),
        )
        self.assertEqual(partition_counts(4096, tpu_info), (4, 8))
        self.assertEqual(token_block_alignment(4096, tpu_info), 512)
        self.assertEqual(partition_counts(1024, tpu_info), (1, 32))
        self.assertEqual(token_block_alignment(1024, tpu_info), 2048)


if __name__ == "__main__":
    unittest.main()

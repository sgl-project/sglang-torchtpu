# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for :mod:`sglang.srt.hardware_backend.tpu.pcp_rank_binding`
that do not need a TPU device.

Covers ``compute_pcp_local_rank_remap`` (pure function) and the small
public gate helpers. The ``ensure_pcp_local_rank_remap`` cache reset is
exercised so future edits to the cache logic don't quietly leak state
between test cases.
"""

from __future__ import annotations

import unittest

from sglang.srt.hardware_backend.tpu.pcp_rank_binding import (
    _validate_remap,
    clear_pcp_local_rank_remaps,
    compute_pcp_local_rank_remap,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ComputeRemapTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_pcp_local_rank_remaps()

    def test_identity_mapping(self) -> None:
        # local_rank k already binds to device k, and jax orders devices
        # in the same order — remap is identity.
        remap = compute_pcp_local_rank_remap(
            local_rank_to_device_id=[0, 1, 2, 3],
            jax_device_ids=[0, 1, 2, 3],
        )
        self.assertEqual(remap, (0, 1, 2, 3))

    def test_permutation(self) -> None:
        # local_rank 0 -> device 2, local_rank 1 -> device 0, etc.
        # jax presents devices in natural id order.
        # Expected remap[k] = local_rank that binds to jax.devices()[k].
        remap = compute_pcp_local_rank_remap(
            local_rank_to_device_id=[2, 0, 3, 1],
            jax_device_ids=[0, 1, 2, 3],
        )
        # jax device 0 -> local_rank 1; device 1 -> 3; device 2 -> 0;
        # device 3 -> 2.
        self.assertEqual(remap, (1, 3, 0, 2))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "mismatched result lengths"):
            compute_pcp_local_rank_remap([0, 1, 2], [0, 1, 2, 3])

    def test_duplicate_local_ids_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicate TorchTPU device ids"):
            compute_pcp_local_rank_remap([0, 0, 2, 3], [0, 1, 2, 3])

    def test_duplicate_jax_ids_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "duplicate JAX device ids"):
            compute_pcp_local_rank_remap([0, 1, 2, 3], [0, 0, 2, 3])

    def test_missing_jax_id_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "missing from TorchTPU bindings"):
            compute_pcp_local_rank_remap([0, 1, 2, 3], [0, 1, 2, 99])


class ValidateRemapTest(unittest.TestCase):
    def test_valid_permutation_returns_tuple(self) -> None:
        self.assertEqual(_validate_remap(4, [3, 1, 0, 2]), (3, 1, 0, 2))

    def test_wrong_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_remap(4, [0, 1, 2])

    def test_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_remap(4, [0, 1, 2, 5])

    def test_duplicate_raises(self) -> None:
        with self.assertRaises(ValueError):
            _validate_remap(4, [0, 1, 1, 3])


if __name__ == "__main__":
    unittest.main()

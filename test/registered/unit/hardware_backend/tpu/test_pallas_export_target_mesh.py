"""CPU export must use physical TPU metadata without leaking target state.

Regression: a CPU-hosted fused-MoE export queried Pallas hardware information
without a TPU target and failed before StableHLO lowering. Only physical-device
metadata discovery is mocked; JAX mesh scoping and Pallas queries are real.
"""

import unittest
from unittest.mock import patch

import jax
from jax.experimental.pallas import tpu as pltpu
from jax.sharding import AbstractDevice, AbstractMesh, use_abstract_mesh
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from sglang_tpu.pallas_export import physical_tpu_mesh_for_cpu_export

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPallasExportTargetMesh(CustomTestCase):
    def test_cpu_export_target_restores_after_failure(self):
        original = jax.sharding.get_abstract_mesh()
        cpu_mesh = AbstractMesh(
            (2,), ("worker",), abstract_device=AbstractDevice("cpu", 1, "cpu")
        )
        with (
            use_abstract_mesh(cpu_mesh),
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                return_value="TPU v7",
            ),
        ):
            with (
                self.assertRaisesRegex(RuntimeError, "export interrupted"),
                physical_tpu_mesh_for_cpu_export(),
            ):
                target = jax.sharding.get_abstract_mesh()
                self.assertEqual(target.shape, cpu_mesh.shape)
                self.assertEqual(target.axis_types, cpu_mesh.axis_types)
                self.assertEqual(target.abstract_device.platform, "tpu")
                self.assertEqual(target.abstract_device.device_kind, "TPU7x")
                self.assertEqual(pltpu.get_tpu_info().num_lanes, 128)
                raise RuntimeError("export interrupted")
            self.assertEqual(jax.sharding.get_abstract_mesh(), cpu_mesh)
        self.assertEqual(jax.sharding.get_abstract_mesh(), original)

    def test_existing_tpu_target_keeps_its_hardware(self):
        mesh = AbstractMesh(
            (), (), (), abstract_device=AbstractDevice("TPU v5 lite", 1, "tpu")
        )
        with (
            use_abstract_mesh(mesh),
            patch(
                "sglang_tpu.pallas_export.get_torch_tpu_device_kind",
                side_effect=RuntimeError("physical discovery must not replace target"),
            ),
            physical_tpu_mesh_for_cpu_export(),
        ):
            self.assertEqual(jax.sharding.get_abstract_mesh(), mesh)
            self.assertEqual(pltpu.get_tpu_info().num_lanes, 128)


if __name__ == "__main__":
    unittest.main()

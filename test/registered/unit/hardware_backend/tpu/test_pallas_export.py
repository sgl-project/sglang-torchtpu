# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TorchTPU CPU-only JAX export environment."""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.hardware_backend.tpu import pallas_export
from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit")


class TestPallasExportEnvironment(CustomTestCase):
    def setUp(self):
        env_patcher = patch.dict(os.environ, {}, clear=False)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)
        os.environ.pop("JAX_PLATFORMS", None)
        os.environ.pop("XLA_FLAGS", None)
        self.fake_sys = SimpleNamespace(modules={})
        sys_patcher = patch.object(pallas_export, "sys", self.fake_sys)
        sys_patcher.start()
        self.addCleanup(sys_patcher.stop)

    def test_configures_cpu_platform_and_distinct_device_count(self):
        pallas_export.configure_jax_export_env(8)

        self.assertEqual(os.environ["JAX_PLATFORMS"], "cpu")
        self.assertIn(
            "--xla_force_host_platform_device_count=8", os.environ["XLA_FLAGS"]
        )

    def test_preserves_unrelated_xla_flags(self):
        os.environ["XLA_FLAGS"] = "--xla_tpu_enable_async_collective_fusion=true"

        pallas_export.configure_jax_export_env(4)

        self.assertEqual(
            os.environ["XLA_FLAGS"],
            "--xla_tpu_enable_async_collective_fusion=true "
            "--xla_force_host_platform_device_count=4",
        )

    def test_rejects_conflicting_platform_or_device_count(self):
        os.environ["JAX_PLATFORMS"] = "tpu"
        with self.assertRaisesRegex(RuntimeError, "requires JAX_PLATFORMS=cpu"):
            pallas_export.configure_jax_export_env(8)

        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        with self.assertRaisesRegex(RuntimeError, "must equal WORLD_SIZE=8"):
            pallas_export.configure_jax_export_env(8)

    def test_rejects_late_jax_import_or_backend_initialization(self):
        self.fake_sys.modules["jax"] = object()
        with self.assertRaisesRegex(RuntimeError, "JAX was imported before"):
            pallas_export.configure_jax_export_platform()

        os.environ["JAX_PLATFORMS"] = "cpu"
        self.fake_sys.modules["jax._src.xla_bridge"] = SimpleNamespace(
            _backends={"cpu": object()}
        )
        with self.assertRaisesRegex(RuntimeError, "JAX backend was initialized"):
            pallas_export.configure_jax_export_env(8)

    def test_worker_reuses_driver_world_size_configuration(self):
        from sglang.srt.hardware_backend.tpu.patches import _prepare_tpu_env

        os.environ["WORLD_SIZE"] = "8"
        with patch(
            "sglang.srt.hardware_backend.tpu.env_override.apply_tpu_env_overrides"
        ):
            _prepare_tpu_env(8, setup_slicebuilder=False)
            _prepare_tpu_env(setup_slicebuilder=False)

        self.assertIn(
            "--xla_force_host_platform_device_count=8", os.environ["XLA_FLAGS"]
        )

    @unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
    def test_cpu_export_context_reports_and_restores_tpu_target(self):
        from jax._src.pallas.mosaic import tpu_info
        from jax.experimental.pallas import tpu as pltpu

        original_public = pltpu.get_tpu_info
        original_private = tpu_info.get_tpu_info
        original_device_kind = tpu_info.get_device_kind
        original_is_tpu_device = tpu_info.is_tpu_device

        with patch.object(
            pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
        ):
            with pallas_export.tpu_info_for_cpu_export():
                self.assertEqual(tpu_info.get_device_kind(), "TPU7x")
                self.assertTrue(tpu_info.is_tpu_device())
                self.assertEqual(pltpu.get_tpu_info().chip_version.value, "7x")
                self.assertIs(pltpu.get_tpu_info(), tpu_info.get_tpu_info())

        self.assertIs(pltpu.get_tpu_info, original_public)
        self.assertIs(tpu_info.get_tpu_info, original_private)
        self.assertIs(tpu_info.get_device_kind, original_device_kind)
        self.assertIs(tpu_info.is_tpu_device, original_is_tpu_device)

    @unittest.skipUnless(importlib.util.find_spec("jax"), "requires JAX")
    def test_distinct_mesh_exports_collective_with_world_size_partitions(self):
        script = textwrap.dedent("""
            import base64
            import inspect
            import json
            import numpy as np
            from unittest.mock import patch

            from sglang.srt.hardware_backend.tpu import pallas_export

            devices = pallas_export.get_jax_export_devices(4)

            import jax
            import jax.numpy as jnp
            from jax import lax
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            from jax.sharding import Mesh, PartitionSpec as P
            from jaxlib.mlir import ir
            from jaxlib.mosaic.python import tpu as mosaic_tpu

            mesh = Mesh(np.asarray(devices).reshape(1, 4), ("data", "model"))
            if hasattr(jax, "shard_map"):
                shard_map = jax.shard_map
            else:
                from jax.experimental.shard_map import shard_map
            check_arg = (
                {"check_vma": False}
                if "check_vma" in inspect.signature(shard_map).parameters
                else {"check_rep": False}
            )
            mapped = jax.jit(
                shard_map(
                    lambda x: lax.psum(x, "model"),
                    mesh=mesh,
                    in_specs=P("model"),
                    out_specs=P("model"),
                    **check_arg,
                )
            )
            x = jnp.arange(4, dtype=jnp.float32)
            exported = jax.export.export(mapped)(
                jax.ShapeDtypeStruct((4,), jnp.float32)
            )

            def remote_copy_kernel(x_ref, y_ref, send_sem, recv_sem):
                rank = lax.axis_index("model")
                destination = (rank + 1) % lax.axis_size("model")
                copy = pltpu.make_async_remote_copy(
                    src_ref=x_ref,
                    dst_ref=y_ref,
                    send_sem=send_sem,
                    recv_sem=recv_sem,
                    device_id={"model": destination},
                    device_id_type=pl.DeviceIdType.MESH,
                )
                copy.start()
                copy.wait()

            def remote_copy_shard(x_local):
                block_spec = pl.BlockSpec(x_local.shape)
                return pl.pallas_call(
                    remote_copy_kernel,
                    out_shape=jax.ShapeDtypeStruct(x_local.shape, x_local.dtype),
                    grid_spec=pltpu.PrefetchScalarGridSpec(
                        num_scalar_prefetch=0,
                        in_specs=[block_spec],
                        out_specs=block_spec,
                        scratch_shapes=(
                            pltpu.SemaphoreType.DMA,
                            pltpu.SemaphoreType.DMA,
                        ),
                    ),
                    compiler_params=pltpu.CompilerParams(
                        vmem_limit_bytes=1024 * 1024
                    ),
                )(x_local)

            remote_copy = jax.jit(
                shard_map(
                    remote_copy_shard,
                    mesh=mesh,
                    in_specs=P("model"),
                    out_specs=P("model"),
                    **check_arg,
                )
            )
            exporter = jax.export.export(
                remote_copy,
                platforms=["tpu"],
                disabled_checks=[
                    jax.export.DisabledSafetyCheck.custom_call("ALL")
                ],
            )
            with patch.object(
                pallas_export,
                "get_pallas_tpu_device_kind",
                return_value="TPU7x",
            ), pallas_export.tpu_info_for_cpu_export():
                pallas_exported = exporter(
                    jax.ShapeDtypeStruct((32,), jnp.float32)
                )

            pallas_module = pallas_exported.mlir_module()
            body_start = pallas_module.index("TUzvUg")
            body_end = pallas_module.index(chr(92), body_start)
            mosaic_bytecode = base64.b64decode(
                pallas_module[body_start:body_end]
            )
            context = ir.Context()
            context.allow_unregistered_dialects = True
            mosaic_tpu.register_dialect(context)
            mosaic_module = str(ir.Module.parse(mosaic_bytecode, context)).lower()

            print(
                json.dumps(
                    {
                        "device_ids": [device.id for device in devices],
                        "mesh_shape": dict(mesh.shape),
                        "nr_devices": exported.nr_devices,
                        "has_all_reduce": (
                            "all_reduce" in exported.mlir_module().lower()
                        ),
                        "serialized_size": len(exported.mlir_module_serialized),
                        "result": np.asarray(mapped(x)).tolist(),
                        "pallas_nr_devices": pallas_exported.nr_devices,
                        "has_enqueue_dma": (
                            "stable_mosaic.tpu.enqueue_dma" in mosaic_module
                        ),
                        "has_wait_dma": (
                            "stable_mosaic.tpu.wait_dma2" in mosaic_module
                        ),
                    }
                )
            )
            """)
        env = os.environ.copy()
        env.pop("JAX_PLATFORMS", None)
        env.pop("XLA_FLAGS", None)
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"subprocess stderr:\n{completed.stderr}",
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])

        self.assertEqual(result["device_ids"], [0, 1, 2, 3])
        self.assertEqual(result["mesh_shape"], {"data": 1, "model": 4})
        self.assertEqual(result["nr_devices"], 4)
        self.assertTrue(result["has_all_reduce"])
        self.assertGreater(result["serialized_size"], 0)
        self.assertEqual(result["result"], [6.0, 6.0, 6.0, 6.0])
        self.assertEqual(result["pallas_nr_devices"], 4)
        self.assertTrue(result["has_enqueue_dma"])
        self.assertTrue(result["has_wait_dma"])


if __name__ == "__main__":
    unittest.main()

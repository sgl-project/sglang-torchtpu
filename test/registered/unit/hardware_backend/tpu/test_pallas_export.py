# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TorchTPU CPU-only JAX export environment."""

import importlib.util
import inspect
import json
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.test.ci.ci_register import register_cpu_ci, register_tpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.utils import pallas_export

register_cpu_ci(est_time=20, suite="base-a-test-cpu")
register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)


class TestPallasExportEnvironment(CustomTestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_pipeline_wave_uses_logical_ranks_and_native_assignment(self):
        import torch
        from jax.sharding import PartitionSpec as P

        from sglang.test.cpu_test_utils import get_cpu_export_mesh
        from sglang_tpu.distributed import pp_wave

        mesh = get_cpu_export_mesh(("pp",), (4,), (3, 1, 0, 2))
        group = SimpleNamespace(
            rank_in_group=0, world_size=4, ranks=(2, 0, 3, 1), cpu_group=object()
        )

        def gather_verdicts(values, verdict, group):
            values[:] = [verdict] * 4

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "4"}),
            patch.object(pp_wave, "_OPS", {}),
            patch.object(pp_wave.PPWave, "probe", return_value=True),
            patch.object(
                torch.distributed, "all_gather_object", side_effect=gather_verdicts
            ),
            patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            patch("torch_tpu._internal.pallas.jax_op") as register,
        ):
            pp_wave.build_pp_wave(
                group,
                stage_group=SimpleNamespace(rank_in_group=0, world_size=1, ranks=(2,)),
                world_group=group,
                rows=8,
                hidden=16,
                dtype=torch.float32,
                device="cpu",
            )
            self.assertEqual(list(pp_wave._OPS), [(2, 0, 0, 3, 3, 1, 1, 2)])
            register.assert_called_once()
            self.assertIs(register.call_args.kwargs["mesh"], mesh)
            self.assertEqual(register.call_args.kwargs["input_partition_specs"], (P(),))
            self.assertEqual(
                register.call_args.kwargs["output_partition_specs"], (P(),)
            )

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_eagle3_wave_registers_only_the_pair_permute(self):
        """The Eagle3 slots ride their own two-operand program; the single-operand
        permute of the wave without them must stay unregistered."""
        import torch
        from jax.sharding import PartitionSpec as P

        from sglang.test.cpu_test_utils import get_cpu_export_mesh
        from sglang_tpu.distributed import pp_wave

        mesh = get_cpu_export_mesh(("pp",), (4,), (3, 1, 0, 2))
        group = SimpleNamespace(
            rank_in_group=0, world_size=4, ranks=(2, 0, 3, 1), cpu_group=object()
        )

        def gather_verdicts(values, verdict, group):
            values[:] = [verdict] * 4

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "4"}),
            patch.object(pp_wave, "_OPS", {}),
            patch.object(pp_wave, "_PAIR_OPS", {}),
            patch.object(pp_wave.PPWave, "probe", return_value=True),
            patch.object(
                torch.distributed, "all_gather_object", side_effect=gather_verdicts
            ),
            patch("torch_tpu._internal.pallas.get_pallas_mesh", return_value=mesh),
            patch("torch_tpu._internal.pallas.jax_op") as register,
        ):
            wave = pp_wave.build_pp_wave(
                group,
                stage_group=SimpleNamespace(rank_in_group=0, world_size=1, ranks=(2,)),
                world_group=group,
                rows=8,
                hidden=16,
                num_aux=3,
                dtype=torch.float32,
                device="cpu",
            )
            self.assertIsInstance(wave, pp_wave.PPAuxWave)
            self.assertEqual(pp_wave._OPS, {})
            self.assertEqual(list(pp_wave._PAIR_OPS), [(2, 0, 0, 3, 3, 1, 1, 2)])
            register.assert_called_once()
            self.assertEqual(
                register.call_args.kwargs["input_partition_specs"], (P(), P())
            )
            self.assertEqual(
                register.call_args.kwargs["output_partition_specs"], (P(), P())
            )

    @unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires TorchTPU")
    def test_mesh_builders_use_native_mesh_unchanged(self):
        from sglang_tpu.layers import distributed_topk_pallas
        from sglang_tpu.layers.moe.pallas import fused_moe_rs_op

        for builder, args, axes, shape in (
            (distributed_topk_pallas._get_mesh, (8,), ("model",), (8,)),
            (fused_moe_rs_op._get_mesh, (), ("data", "model"), (1, 8)),
        ):
            mesh = object()
            get_mesh = MagicMock(return_value=mesh)
            with (
                self.subTest(axes=axes),
                patch.dict(os.environ, {"WORLD_SIZE": "8"}),
                patch.dict(
                    sys.modules,
                    {
                        "torch_tpu._internal.pallas": SimpleNamespace(
                            get_pallas_mesh=get_mesh
                        )
                    },
                ),
            ):
                builder.cache_clear()
                try:
                    self.assertIs(builder(*args), mesh)
                    get_mesh.assert_called_once_with(axes, shape)
                    builder.cache_clear()
                    get_mesh.side_effect = ValueError("mesh shape")
                    with self.assertRaisesRegex(ValueError, "mesh shape"):
                        builder(*args)
                finally:
                    builder.cache_clear()

    def setUp(self):
        env_patcher = patch.dict(os.environ, {}, clear=False)
        env_patcher.start()
        self.addCleanup(env_patcher.stop)

    def test_maps_torchtpu_device_kind_to_pallas_spelling(self):
        cases = (
            ("TPU v6e", "TPU v6 lite"),
            ("TPU v7", "TPU7x"),
            ("TPU v5e", "TPU v5e"),
        )
        for torch_device_kind, pallas_device_kind in cases:
            with (
                self.subTest(torch_device_kind=torch_device_kind),
                patch.object(
                    pallas_export,
                    "get_torch_tpu_device_kind",
                    return_value=torch_device_kind,
                ),
            ):
                self.assertEqual(
                    pallas_export.get_pallas_tpu_device_kind(), pallas_device_kind
                )

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_cpu_export_custom_op_uses_process_stable_trace_identity(self):
        from sglang_tpu.utils import pallas_jax_op

        def identity(value: pallas_jax_op.jax.Array) -> pallas_jax_op.jax.Array:
            return value

        fake_custom_op = MagicMock()
        with (
            patch.object(pallas_jax_op.jax, "jit", side_effect=lambda fn, **_: fn),
            patch.object(
                pallas_jax_op.pallas_impl,
                "_get_kernel_invocation_key",
                return_value="stable-key",
            ) as get_key,
            patch.object(pallas_jax_op, "_CpuExportJaxCallable"),
            patch.object(
                pallas_jax_op.torch.library,
                "custom_op",
                return_value=fake_custom_op,
            ),
        ):
            pallas_jax_op.cpu_export_jax_op(
                "sglang_tpu_test::stable_identity", identity
            )

        get_key.assert_called_once_with(
            "sglang_tpu_test::stable_identity",
            [],
            {
                "static_argnums": (),
                "donate_argnums": (),
                "output_partition_specs": (),
            },
        )
        fake_custom_op.register_fake.assert_called_once()

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_kv_write_kernels_are_registered_as_functions_of_tensors(self):
        """custom_jax_kernel binds a kernel's keyword defaults (`interpret`) and
        a partial's keywords (`max_span_len`) and traces them inside its jit,
        where the Pallas call rejects them; the KV-write kernels must reach it
        through functions that take tensors only."""
        import torch

        from sglang_tpu.mem_cache import (
            kv_fused_rope_write,
            kv_page_write,
            kv_row_write,
        )

        with (
            patch("torch_tpu._internal.pallas.custom_jax_kernel") as register,
            patch.object(torch.library, "custom_op"),
            patch.object(kv_page_write, "_OP", None),
            patch.object(kv_row_write, "_OP", None),
        ):
            kv_page_write.get_write_pages_op()
            kv_row_write.get_write_rows_op()
            kv_fused_rope_write.get_fused_rope_write_op.__wrapped__(2, 256)

        positional = (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        )
        self.assertEqual(register.call_count, 3)
        for call in register.call_args_list:
            for param in inspect.signature(call.args[0]).parameters.values():
                with self.subTest(kernel=call.kwargs["name"], param=param.name):
                    self.assertIn(param.kind, positional)
                    self.assertIs(param.default, inspect.Parameter.empty)

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_native_cache_key_tracks_invocation_and_pallas_body_and_reuses_export(self):
        import jax
        import torch
        from jax.experimental import pallas as pl
        from jax.sharding import PartitionSpec as P

        from sglang.test.cpu_test_utils import get_cpu_export_mesh
        from sglang_tpu.utils import pallas_jax_op

        device_ids = (3, 1, 0, 2)
        mesh = get_cpu_export_mesh(("model",), (4,), device_ids)
        registered = set()
        fingerprints = []
        native_keys = []
        value = torch.ones((8, 128), dtype=torch.float32)
        native = pallas_jax_op.pallas_impl.tpu_torch_pallas
        with (
            patch.object(
                pallas_export, "get_pallas_tpu_device_kind", return_value="TPU7x"
            ),
            patch.object(pallas_jax_op, "_torch_placeholder", return_value=value),
            patch.object(
                native,
                "lookup_custom_kernel",
                side_effect=lambda name, key: (name, key) in registered,
            ) as lookup,
            patch.object(
                native,
                "register_custom_kernel",
                side_effect=lambda name, key, **_: registered.add((name, key)),
            ) as register,
            patch.object(native, "call_custom_kernel", return_value=[value]) as call,
        ):
            for increment, trace_key in (
                (1, "signature-a"),
                (1, "signature-a"),
                (2, "signature-a"),
                (2, "signature-a"),
                (2, "signature-b"),
            ):

                def kernel(value_ref, output_ref):
                    output_ref[...] = value_ref[...] + increment

                def add(value: jax.Array) -> jax.Array:
                    return pl.pallas_call(
                        kernel,
                        out_shape=jax.ShapeDtypeStruct(value.shape, value.dtype),
                    )(value)

                wrapped = pallas_jax_op._CpuExportJaxCallable(
                    name="sglang_tpu_test::body_identity",
                    jit_fn=jax.jit(
                        jax.shard_map(
                            add,
                            mesh=mesh,
                            in_specs=(P(),),
                            out_specs=P(),
                            check_vma=False,
                        )
                    ),
                    trace_key=trace_key,
                    mesh=mesh,
                    input_partition_specs=(P(),),
                )
                wrapped.sglang_output_shardings = ()
                with patch.object(
                    wrapped, "exported", wraps=wrapped.exported
                ) as export:
                    wrapped(value)
                    kernel_key = pallas_jax_op.pallas_impl._get_kernel_invocation_key(
                        trace_key, (value,), {}, ()
                    )
                    fingerprint = wrapped.kernel_key_to_mlir_fingerprint[kernel_key]
                    native_key = f"{kernel_key};mlir={fingerprint}"
                    self.assertEqual(register.call_args.args[1], native_key)
                    self.assertEqual(call.call_args.args[1], native_key)
                    self.assertEqual(
                        call.call_args.kwargs["device_assignment"], list(device_ids)
                    )
                    wrapped(value)
                    lookup.assert_called_with(wrapped.name, native_key)
                    self.assertEqual(call.call_args.args[1], native_key)
                    self.assertEqual(
                        call.call_args.kwargs["device_assignment"], list(device_ids)
                    )
                    export.assert_called_once()
                fingerprints.append(fingerprint)
                native_keys.append(native_key)

        self.assertEqual(fingerprints[0], fingerprints[1])
        self.assertNotEqual(fingerprints[1], fingerprints[2])
        self.assertEqual(fingerprints[2], fingerprints[3])
        self.assertEqual(fingerprints[3], fingerprints[4])
        self.assertEqual(native_keys[0], native_keys[1])
        self.assertNotEqual(native_keys[1], native_keys[2])
        self.assertEqual(native_keys[2], native_keys[3])
        self.assertNotEqual(native_keys[3], native_keys[4])
        self.assertEqual(len(registered), 3)

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

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_distinct_mesh_exports_collective_with_world_size_partitions(self):
        script = textwrap.dedent("""
            import base64
            import inspect
            import json
            from unittest.mock import patch

            from sglang_tpu.utils import pallas_export

            from sglang.test.cpu_test_utils import get_cpu_export_mesh

            import jax
            import jax.numpy as jnp
            from jax import lax
            from jax._src.interpreters import mlir as jax_mlir
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            from jax.sharding import PartitionSpec as P
            from jaxlib.mlir import ir
            from jaxlib.mlir.dialects import sdy, stablehlo
            from jaxlib.mosaic.python import tpu as mosaic_tpu

            mesh = get_cpu_export_mesh(("data", "model"), (1, 4))
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

            outer_context = ir.Context()
            outer_context.allow_unregistered_dialects = True
            outer_context.append_dialect_registry(jax_mlir.upstream_dialects)
            sdy.register_dialect(outer_context)
            stablehlo.register_dialect(outer_context)
            outer_context.load_all_available_dialects()
            outer_module = ir.Module.parse(
                pallas_exported.mlir_module(), outer_context
            )

            def walk_operations(operation):
                yield operation
                for region in operation.regions:
                    for block in region.blocks:
                        for child in block.operations:
                            yield from walk_operations(child.operation)

            mosaic_bodies = []
            for operation in walk_operations(outer_module.operation):
                if operation.name != "stablehlo.custom_call":
                    continue
                try:
                    backend_config = ir.StringAttr(
                        operation.attributes["backend_config"]
                    ).value
                except KeyError:
                    continue
                config = json.loads(backend_config)
                body = config.get("custom_call_config", {}).get("body")
                if body is not None:
                    mosaic_bodies.append(body)

            if len(mosaic_bodies) != 1:
                raise AssertionError(
                    f"Expected one serialized Mosaic body, got {len(mosaic_bodies)}"
                )
            mosaic_bytecode = base64.b64decode(mosaic_bodies[0], validate=True)
            context = ir.Context()
            context.allow_unregistered_dialects = True
            mosaic_tpu.register_dialect(context)
            mosaic_module = str(ir.Module.parse(mosaic_bytecode, context)).lower()

            print(
                json.dumps(
                    {
                        "device_ids": [device.id for device in mesh.devices.flat],
                        "mesh_shape": dict(mesh.shape),
                        "nr_devices": exported.nr_devices,
                        "has_all_reduce": (
                            "all_reduce" in exported.mlir_module().lower()
                        ),
                        "serialized_size": len(exported.mlir_module_serialized),
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
        env["JAX_PLATFORMS"] = "cpu"
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
        self.assertEqual(result["pallas_nr_devices"], 4)
        self.assertTrue(result["has_enqueue_dma"])
        self.assertTrue(result["has_wait_dma"])

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_cpu_export_jax_op_preserves_unused_input(self):
        script = textwrap.dedent("""
            import json
            import os

            import jax
            import torch
            import torch_tpu  # noqa: F401

            from sglang_tpu.utils.pallas_jax_op import (
                cpu_export_jax_op,
            )

            def add_one(x: jax.Array, unused: jax.Array) -> jax.Array:
                return x + 1

            op = cpu_export_jax_op(
                "sglang_tpu_test::cpu_export_add_one",
                add_one,
            )
            x = torch.arange(8, dtype=torch.float32, device="tpu")
            # Torch keeps both operands, so JAX must not remove this one.
            unused = torch.zeros(8, dtype=torch.float32, device="tpu")
            y = op(x, unused)
            torch.accelerator.synchronize()
            print(
                json.dumps(
                    {
                        "result": y.cpu().tolist(),
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
            timeout=60,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"subprocess stderr:\n{completed.stderr}",
        )
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(result["result"], [float(value) for value in range(1, 9)])

    @unittest.skipUnless(
        importlib.util.find_spec("jax") and importlib.util.find_spec("torch_tpu"),
        "requires JAX and TorchTPU",
    )
    def test_partitioned_shape_supports_multiple_mesh_axes(self):
        from types import SimpleNamespace

        from jax.sharding import PartitionSpec as P

        from sglang_tpu.utils.pallas_jax_op import (
            _convert_partitioned_shape,
        )

        mesh = SimpleNamespace(shape={"chip": 4, "core": 2})
        partition_spec = P(("chip", "core"), None)

        global_shape = _convert_partitioned_shape(
            (16, 32),
            mesh,
            partition_spec,
            lambda size, axis_size: size * axis_size,
        )
        self.assertEqual(global_shape, (128, 32))
        self.assertEqual(
            _convert_partitioned_shape(
                global_shape,
                mesh,
                partition_spec,
                lambda size, axis_size: size // axis_size,
            ),
            (16, 32),
        )


if __name__ == "__main__":
    unittest.main()

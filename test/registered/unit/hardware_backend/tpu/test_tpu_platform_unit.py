"""Unit tests for TpuSRTPlatform.

These tests mock torch_tpu so they can run on any machine without TPU hardware.
"""

import builtins
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch.compiler.config as compiler_config

from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# CPU suite: mocks torch_tpu, so it runs on the CPU runners -- never a TPU node.
register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestTpuPlatformIdentity(CustomTestCase):
    """Platform identity and capability flags."""

    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_enum_is_tpu(self):
        self.assertEqual(self.platform._enum, PlatformEnum.TPU)

    def test_device_name(self):
        self.assertEqual(self.platform.device_name, "tpu")
        self.assertEqual(self.platform.device_type, "tpu")

    def test_is_tpu(self):
        self.assertTrue(self.platform.is_tpu())
        self.assertFalse(self.platform.is_cuda())
        self.assertFalse(self.platform.is_npu())

    def test_supports_fp8_false(self):
        self.assertFalse(self.platform.supports_fp8())

    def test_pin_memory_unavailable(self):
        self.assertFalse(self.platform.is_pin_memory_available())

    def test_cuda_graph_disabled(self):
        self.assertFalse(self.platform.support_cuda_graph())
        self.assertFalse(self.platform.support_piecewise_cuda_graph())

    def test_dispatch_key(self):
        self.assertEqual(self.platform.get_dispatch_key_name(), "tpu")

    def test_compile_backend(self):
        self.assertEqual(self.platform.get_compile_backend(), "tpu")

    def test_distributed_backend(self):
        self.assertEqual(self.platform.get_torch_distributed_backend_str(), "tpu_dist")

    def test_default_attention_backend(self):
        self.assertEqual(
            self.platform.get_default_attention_backend(), "pallas_tpu_rpa_v3"
        )


class TestTpuPlatformInitialization(CustomTestCase):
    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_rejects_backend_initialization_without_worker_world_size(self):
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"
            ) as prepare_tpu_env,
            patch(
                "sglang.srt.hardware_backend.tpu.patches.apply_tpu_patches"
            ) as apply_tpu_patches,
        ):
            os.environ.pop("WORLD_SIZE", None)
            with self.assertRaisesRegex(RuntimeError, "requires worker WORLD_SIZE"):
                self.platform.init_backend()

        prepare_tpu_env.assert_not_called()
        apply_tpu_patches.assert_not_called()

    def test_worker_world_size_prepares_environment_before_runtime_patches(self):
        calls = []
        fake_torch_tpu = ModuleType("torch_tpu")

        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16"}, clear=False),
            patch.dict(sys.modules, {"torch_tpu": fake_torch_tpu}),
            compiler_config.patch(cache_key_tag=""),
            patch(
                "sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env",
                side_effect=lambda world_size, **kwargs: calls.append(
                    ("prepare", world_size, kwargs)
                ),
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.patches.apply_tpu_patches",
                side_effect=lambda: calls.append(("patch", None)),
            ),
        ):
            self.platform.init_backend()

        self.assertEqual(
            calls,
            [
                (
                    "prepare",
                    16,
                    {"resolve_single_host_topology": True},
                ),
                ("patch", None),
            ],
        )


class TestTpuCompileCache(CustomTestCase):
    def test_pallas_source_change_updates_compile_cache_key(self):
        import sglang.srt.hardware_backend.tpu.compile_cache as compile_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            srt_root = Path(temp_dir)
            source_files = (
                srt_root / "layers/attention/pallas/kernel.py",
                srt_root / "layers/moe/pallas/kernel.py",
                srt_root / "hardware_backend/tpu/attention_pallas.py",
                srt_root / "hardware_backend/tpu/attention_pallas_rpa_base.py",
                srt_root / "hardware_backend/tpu/attention_pallas_rpa_batched.py",
                srt_root / "hardware_backend/tpu/attention_pallas_rpa_v3.py",
                srt_root / "hardware_backend/tpu/kernels/kernel.py",
                srt_root / "hardware_backend/tpu/pallas_export.py",
                srt_root / "hardware_backend/tpu/pallas_jax_op.py",
                srt_root / "hardware_backend/tpu/quantized_matmul.py",
            )
            for source_file in source_files:
                source_file.parent.mkdir(parents=True, exist_ok=True)
                source_file.write_text("original source\n")

            module_path = srt_root / "hardware_backend/tpu/compile_cache.py"
            with (
                patch.object(compile_cache, "__file__", str(module_path)),
                compiler_config.patch(cache_key_tag="user-tag"),
            ):
                compile_cache.add_pallas_kernel_sources_to_compile_cache_key()
                original_key = compiler_config.cache_key_tag

                source_files[0].write_text("changed source\n")
                compile_cache.add_pallas_kernel_sources_to_compile_cache_key()
                changed_key = compiler_config.cache_key_tag

                self.assertTrue(original_key.startswith("user-tag|"))
                self.assertNotEqual(changed_key, original_key)


class TestTpuPlatformMemory(CustomTestCase):
    """Device memory methods delegate to torch.accelerator.get_memory_info.

    The per-process HBM limit is read from the runtime (``(free, limit)``), not
    a static per-device-name table, so these mock get_memory_info + get_device
    to run without a real TPU.
    """

    def setUp(self):
        self.platform = TpuSRTPlatform()

    def _patch_mem(self, free, limit):
        # get_device() builds torch.device("tpu", ...) which is invalid off-TPU;
        # stub it, and stub the runtime memory query to (free, limit).
        return (
            patch.object(self.platform, "get_device", return_value=None),
            patch("torch.accelerator.get_memory_info", return_value=(free, limit)),
        )

    def test_total_memory_returns_reported_limit(self):
        dev, mem = self._patch_mem(10 << 30, 96 << 30)
        with dev, mem:
            self.assertEqual(self.platform.get_device_total_memory(), 96 << 30)

    def test_current_memory_usage_is_total_minus_free(self):
        dev, mem = self._patch_mem(10 << 30, 96 << 30)
        with dev, mem:
            self.assertEqual(self.platform.get_current_memory_usage(), float(86 << 30))

    def test_available_memory_returns_free_and_total(self):
        dev, mem = self._patch_mem(10 << 30, 96 << 30)
        with dev, mem:
            free, total = self.platform.get_available_memory()
            self.assertEqual(total, 96 << 30)
            self.assertEqual(free, 10 << 30)


class TestTpuPlatformServerDefaults(CustomTestCase):
    """apply_server_args_defaults sets correct values."""

    def _args(self, **overrides):
        # Full set of fields apply_server_args_defaults reads, plus an
        # override() shim mirroring ServerArgs.override (its audited mutation
        # point, which the defaults now flow through).
        defaults = dict(
            dtype=None,
            attention_backend=None,
            disable_cuda_graph=False,
            disable_piecewise_cuda_graph=False,
            enable_torch_compile=False,
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            tp_size=1,
            dp_size=1,
            enable_dp_attention=False,
            enable_dp_lm_head=False,
            page_size=None,
            kv_cache_dtype="auto",
            use_ray=False,
            hicache_io_backend=None,
            speculative_algorithm=None,
            speculative_eagle_topk=None,
            speculative_use_rejection_sampling=False,
        )
        defaults.update(overrides)
        args = SimpleNamespace(**defaults)

        def _override(source, **fields):
            for k, v in fields.items():
                setattr(args, k, v)

        args.override = _override
        return args

    def _apply(self, args):
        platform = TpuSRTPlatform()
        with patch("sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"):
            platform.apply_server_args_defaults(args)

    def test_defaults_applied(self):
        args = self._args(enable_torch_compile=False)
        self._apply(args)

        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.attention_backend, "pallas_tpu_rpa_v3")
        self.assertTrue(args.disable_cuda_graph)
        self.assertTrue(args.disable_piecewise_cuda_graph)
        # enable_torch_compile is opt-in on TPU and must not be forced off.
        self.assertFalse(args.enable_torch_compile)
        # Overlap is enabled on TPU via the thread-based path in
        # scheduler_overlap_mixin; the platform must NOT force-disable it.
        self.assertFalse(args.disable_overlap_schedule)
        self.assertEqual(args.mem_fraction_static, 0.40)

    def test_rpa_backends_default_to_fp8_kv_cache(self):
        for backend in ("pallas_tpu_rpa_v3", "pallas_tpu_rpa_batched"):
            with self.subTest(backend=backend):
                args = self._args(attention_backend=backend)
                self._apply(args)
                self.assertEqual(args.kv_cache_dtype, "fp8_e4m3")

    def test_explicit_kv_cache_dtype_not_overridden(self):
        args = self._args(
            attention_backend="pallas_tpu_rpa_batched", kv_cache_dtype="bfloat16"
        )
        self._apply(args)
        self.assertEqual(args.kv_cache_dtype, "bfloat16")

    def test_non_rpa_backend_keeps_auto_kv_cache_dtype(self):
        # pallas_tpu_mla rejects an fp8 KV cache, so the fp8 default is scoped
        # to the RPA backends and must not leak to it.
        args = self._args(attention_backend="pallas_tpu_mla")
        self._apply(args)
        self.assertEqual(args.kv_cache_dtype, "auto")
        self.assertEqual(args.page_size, 64)

    def test_batched_rpa_page_size_default_meets_the_kernel_floor(self):
        args = self._args(attention_backend="pallas_tpu_rpa_batched")
        self._apply(args)
        self.assertEqual(args.page_size, 256)

    def test_batched_rpa_rejects_a_page_size_below_the_floor(self):
        args = self._args(attention_backend="pallas_tpu_rpa_batched", page_size=64)
        with self.assertRaisesRegex(ValueError, "page-size >= 256"):
            self._apply(args)

    def test_explicit_page_size_above_the_floor_is_kept(self):
        args = self._args(attention_backend="pallas_tpu_rpa_batched", page_size=512)
        self._apply(args)
        self.assertEqual(args.page_size, 512)

    def test_enable_torch_compile_preserved_when_opted_in(self):
        args = self._args(enable_torch_compile=True)
        self._apply(args)
        self.assertTrue(args.enable_torch_compile)

    def test_dp_attention_requires_dp_lm_head_when_attn_tp_gt_1(self):
        args = self._args(
            enable_dp_attention=True,
            tp_size=8,
            dp_size=2,
            enable_dp_lm_head=False,
        )
        with self.assertRaisesRegex(
            ValueError, "When attention TP > 1 under DP attention on TPU"
        ):
            self._apply(args)

    def test_dp_attention_allows_valid_dp_lm_head_combos(self):
        # attn_tp_size > 1 with enable_dp_lm_head=True
        args = self._args(
            enable_dp_attention=True,
            tp_size=8,
            dp_size=2,
            enable_dp_lm_head=True,
        )
        self._apply(args)

        # attn_tp_size == 1 with enable_dp_lm_head=False
        args = self._args(
            enable_dp_attention=True,
            tp_size=8,
            dp_size=8,
            enable_dp_lm_head=False,
        )
        self._apply(args)

    def test_configures_export_mesh_from_tp_size(self):
        args = self._args(tp_size=16)
        platform = TpuSRTPlatform()

        with patch(
            "sglang.srt.hardware_backend.tpu.patches._prepare_tpu_env"
        ) as prepare_tpu_env:
            platform.apply_server_args_defaults(args)

        prepare_tpu_env.assert_called_once_with(16, setup_slicebuilder=True)

    def test_parent_defaults_do_not_import_torch_tpu(self):
        args = self._args(tp_size=8)
        platform = TpuSRTPlatform()
        imported_modules = []
        real_import = builtins.__import__

        def track_import(name, *args, **kwargs):
            imported_modules.append(name)
            return real_import(name, *args, **kwargs)

        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "sglang.srt.hardware_backend.tpu.pallas_export.configure_jax_export_env"
            ),
            patch(
                "sglang.srt.hardware_backend.tpu.env_override.apply_tpu_env_overrides"
            ),
            patch("builtins.__import__", side_effect=track_import),
        ):
            platform.apply_server_args_defaults(args)

        self.assertFalse(
            any(
                name == "torch_tpu" or name.startswith("torch_tpu.")
                for name in imported_modules
            )
        )

    def test_explicit_dtype_not_overridden(self):
        args = self._args(dtype="float32")
        self._apply(args)
        self.assertEqual(args.dtype, "float32")

    def test_explicit_mem_fraction_not_overridden(self):
        args = self._args(mem_fraction_static=0.60)
        self._apply(args)
        self.assertEqual(args.mem_fraction_static, 0.60)


class TestTpuPlatformFactoryMethods(CustomTestCase):
    """Factory methods return the expected classes."""

    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_graph_runner_cls(self):
        cls = self.platform.get_graph_runner_cls()
        self.assertEqual(cls.__name__, "TpuGraphRunner")

    def test_paged_allocator_cls(self):
        cls = self.platform.get_paged_allocator_cls()
        self.assertEqual(cls.__name__, "TpuPagedTokenToKVPoolAllocator")

    def test_mha_kv_pool_cls(self):
        # get_mha_kv_pool_cls picks the pool by the resolved attention backend
        # read off the global server args; a non-RPA backend yields the flat
        # MHATokenToKVPool.
        with patch(
            "sglang.srt.server_args.get_global_server_args",
            return_value=SimpleNamespace(attention_backend="torch_native"),
        ):
            cls = self.platform.get_mha_kv_pool_cls()
        self.assertEqual(cls.__name__, "MHATokenToKVPool")

    def test_nsa_raises(self):
        with self.assertRaises(NotImplementedError):
            self.platform.get_nsa_kv_pool_cls()

    def test_piecewise_backend_raises(self):
        with self.assertRaises(NotImplementedError):
            self.platform.get_piecewise_backend_cls()


if __name__ == "__main__":
    unittest.main()

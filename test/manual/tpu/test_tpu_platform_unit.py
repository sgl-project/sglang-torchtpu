"""Unit tests for TpuSRTPlatform.

These tests mock torch_tpu so they can run on any machine without TPU hardware.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.hardware_backend.tpu.tpu_platform import TpuSRTPlatform
from sglang.srt.platforms.device_mixin import PlatformEnum
from sglang.test.test_utils import CustomTestCase


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
        self.assertEqual(self.platform.get_default_attention_backend(), "pallas_tpu")


class TestTpuPlatformMemory(CustomTestCase):
    """Device memory methods."""

    def setUp(self):
        self.platform = TpuSRTPlatform()

    def test_hbm_lookup_known_devices(self):
        known = {
            "TPU v4": 32 * (1 << 30),
            "TPU v5p": 95 * (1 << 30),
            "TPU v5e": 16 * (1 << 30),
            "TPU v6e": 32 * (1 << 30),
            "TPU v7": 192 * (1 << 30),
        }
        for name, expected_bytes in known.items():
            with self.subTest(device=name):
                with patch.object(self.platform, "get_device_name", return_value=name):
                    self.assertEqual(
                        self.platform.get_device_total_memory(), expected_bytes
                    )

    def test_hbm_lookup_unknown_device_defaults(self):
        with patch.object(self.platform, "get_device_name", return_value="TPU v99"):
            self.assertEqual(
                self.platform.get_device_total_memory(), 32 * (1 << 30)
            )

    def test_current_memory_usage_returns_zero(self):
        self.assertEqual(self.platform.get_current_memory_usage(), 0.0)

    def test_available_memory(self):
        with patch.object(self.platform, "get_device_name", return_value="TPU v7"):
            free, total = self.platform.get_available_memory()
            self.assertEqual(total, 192 * (1 << 30))
            self.assertEqual(free, total)


class TestTpuPlatformServerDefaults(CustomTestCase):
    """apply_server_args_defaults sets correct values."""

    @patch("sglang.srt.hardware_backend.tpu.tpu_platform.TpuSRTPlatform.apply_server_args_defaults")
    def _make_args(self, mock_apply):
        return SimpleNamespace(
            dtype=None,
            attention_backend=None,
            disable_cuda_graph=False,
            disable_piecewise_cuda_graph=False,
            enable_torch_compile=True,
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            tp_size=1,
        )

    def test_defaults_applied(self):
        args = SimpleNamespace(
            dtype=None,
            attention_backend=None,
            disable_cuda_graph=False,
            disable_piecewise_cuda_graph=False,
            enable_torch_compile=True,
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            tp_size=1,
        )
        platform = TpuSRTPlatform()
        with patch(
            "sglang.srt.hardware_backend.tpu.tpu_platform._prepare_tpu_env"
        ):
            platform.apply_server_args_defaults(args)

        self.assertEqual(args.dtype, "bfloat16")
        self.assertEqual(args.attention_backend, "pallas_tpu")
        self.assertTrue(args.disable_cuda_graph)
        self.assertTrue(args.disable_piecewise_cuda_graph)
        self.assertFalse(args.enable_torch_compile)
        self.assertTrue(args.disable_overlap_schedule)
        self.assertEqual(args.mem_fraction_static, 0.40)

    def test_explicit_dtype_not_overridden(self):
        args = SimpleNamespace(
            dtype="float32",
            attention_backend=None,
            disable_cuda_graph=False,
            disable_piecewise_cuda_graph=False,
            enable_torch_compile=True,
            disable_overlap_schedule=False,
            mem_fraction_static=None,
            tp_size=1,
        )
        platform = TpuSRTPlatform()
        with patch(
            "sglang.srt.hardware_backend.tpu.tpu_platform._prepare_tpu_env"
        ):
            platform.apply_server_args_defaults(args)

        self.assertEqual(args.dtype, "float32")

    def test_explicit_mem_fraction_not_overridden(self):
        args = SimpleNamespace(
            dtype=None,
            attention_backend=None,
            disable_cuda_graph=False,
            disable_piecewise_cuda_graph=False,
            enable_torch_compile=True,
            disable_overlap_schedule=False,
            mem_fraction_static=0.60,
            tp_size=1,
        )
        platform = TpuSRTPlatform()
        with patch(
            "sglang.srt.hardware_backend.tpu.tpu_platform._prepare_tpu_env"
        ):
            platform.apply_server_args_defaults(args)

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

# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPU platform. Cases needing torch_tpu self-skip."""

import os
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.platforms import tpu_platform_plugin
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform

_HAS_TORCH_TPU = hasattr(torch, "tpu")


class TestTpuPlatformPlugin(unittest.TestCase):
    def test_selected_when_torch_tpu_is_registered(self):
        with patch.object(torch, "tpu", object(), create=True):
            self.assertEqual(
                tpu_platform_plugin(),
                "sglang.multimodal_gen.runtime.platforms.tpu.TpuPlatform",
            )

    def test_not_selected_without_torch_tpu(self):
        # The wheel being installed is not enough; ``torch.tpu`` must exist.
        with patch.object(torch, "tpu", None, create=True):
            self.assertIsNone(tpu_platform_plugin())


class TestTpuPlatformSurface(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(TpuPlatform.device_type, "tpu")
        self.assertTrue(TpuPlatform.get_device_name().lower().startswith("tpu"))
        self.assertTrue(TpuPlatform().is_tpu())

    def test_float64_unsupported(self):
        self.assertFalse(TpuPlatform.is_float64_supported())

    def test_inference_mode_is_no_grad(self):
        self.assertIsInstance(TpuPlatform.inference_mode(), torch.no_grad)

    def test_attn_backend_default_is_sdpa(self):
        for selected in (None, AttentionBackendEnum.TORCH_SDPA):
            self.assertIn(
                "sdpa.SDPABackend",
                TpuPlatform.get_attn_backend_cls_str(selected, 128, torch.bfloat16),
            )

    def test_attn_backend_unsupported_raises(self):
        with self.assertRaises(ValueError):
            TpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.FA, 128, torch.bfloat16
            )

    def test_single_chip_distributed_backend_is_gloo(self):
        self.assertEqual(TpuPlatform().get_torch_distributed_backend_str(), "gloo")

    def test_layerwise_offload_disabled(self):
        self.assertFalse(TpuPlatform.enable_dit_layerwise_offload_by_default())


@unittest.skipUnless(_HAS_TORCH_TPU, "torch.device('tpu') needs torch_tpu")
class TestWorkerInitOnce(unittest.TestCase):
    def setUp(self):
        self.addCleanup(setattr, TpuPlatform, "_worker_initialized", False)
        TpuPlatform._worker_initialized = False

    def test_init_runs_once_across_call_sites(self):
        with (
            patch.object(TpuPlatform, "_init_worker_process") as init,
            patch.object(torch, "tpu", MagicMock(), create=True),
        ):
            TpuPlatform.set_device(torch.device("tpu", 0))
            TpuPlatform.set_device(torch.device("tpu", 0))
        self.assertEqual(init.call_count, 1)

    def test_failed_init_is_retryable(self):
        with (
            patch.object(
                TpuPlatform, "_init_worker_process", side_effect=RuntimeError("boom")
            ),
            patch.object(torch, "tpu", MagicMock(), create=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                TpuPlatform.set_device(torch.device("tpu", 0))
        self.assertFalse(TpuPlatform._worker_initialized)

    def test_multi_rank_env_is_rejected(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "8"}):
            with self.assertRaisesRegex(RuntimeError, "one worker"):
                TpuPlatform._init_worker_process()


@unittest.skipUnless(_HAS_TORCH_TPU, "torch_tpu not installed")
class TestTorchTpuIntegration(unittest.TestCase):
    def test_pre_init_memory_queries_do_not_initialize_pjrt(self):
        if torch.tpu.is_initialized():
            self.skipTest("PJRT already initialized in this process")
        self.assertGreater(TpuPlatform.get_device_total_memory(), 0)
        self.assertGreater(TpuPlatform.get_available_gpu_memory(), 0.0)
        self.assertEqual(TpuPlatform.get_current_memory_usage(), 0.0)
        self.assertTrue(TpuPlatform.get_device_name().startswith("TPU "))
        self.assertFalse(torch.tpu.is_initialized())

    def test_srt_symbols_the_platform_imports_exist(self):
        # tpu.py imports this lazily; a rename on the srt side must fail here,
        # not at worker bootstrap.
        from sglang.srt.hardware_backend.tpu.patches import _prepare_tpu_env

        self.assertTrue(callable(_prepare_tpu_env))

    def test_device_module_shims_installed_once(self):
        from sglang.multimodal_gen.runtime.platforms.tpu import (
            _install_device_module_shims,
        )

        _install_device_module_shims()
        for name in (
            "empty_cache",
            "reset_peak_memory_stats",
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
        ):
            self.assertTrue(hasattr(torch.tpu, name))
        torch.tpu.empty_cache()
        if not torch.tpu.is_initialized():
            self.assertEqual(torch.tpu.memory_allocated(), 0)


if __name__ == "__main__":
    unittest.main()

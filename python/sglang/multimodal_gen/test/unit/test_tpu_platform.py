# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPU platform. Cases needing torch_tpu self-skip."""

import os
import subprocess
import sys
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

    def test_float64_supported(self):
        self.assertTrue(TpuPlatform.is_float64_supported())

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

    def test_lone_worker_distributed_backend_is_gloo(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            self.assertEqual(TpuPlatform().get_torch_distributed_backend_str(), "gloo")

    def test_multi_worker_distributed_backend_is_tpu_dist(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "16"}):
            self.assertEqual(
                TpuPlatform().get_torch_distributed_backend_str(), "tpu_dist"
            )

    def test_p2p_is_unsupported(self):
        self.assertFalse(TpuPlatform.supports_distributed_p2p())

    def test_auto_spatial_decode_without_p2p_only_for_opted_in_vaes(self):
        from sglang.multimodal_gen.configs.models.vaes import base as vae_base
        from sglang.multimodal_gen.configs.models.vaes.ltx_video import (
            LTXVideoVAEConfig,
        )
        from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig

        explicit = LTXVideoVAEConfig()
        explicit.parallel_decode_mode = "spatial_shard"
        cases = (
            # Halos travel by all-gather where the VAE opts in.
            (False, WanVAEConfig(), True),
            (False, LTXVideoVAEConfig(), False),
            (False, explicit, True),
            (True, LTXVideoVAEConfig(), True),
        )
        for p2p, config, expected in cases:
            platform = MagicMock()
            platform.supports_distributed_p2p.return_value = p2p
            with (
                self.subTest(
                    p2p=p2p, vae=type(config).__name__, mode=config.parallel_decode_mode
                ),
                patch.object(vae_base, "current_platform", platform),
            ):
                self.assertEqual(
                    vae_base.should_use_spatial_shard_parallel_decode(
                        config, world_size=2
                    ),
                    expected,
                )

    def test_platform_defaults_leave_other_backends_alone(self):
        from sglang.multimodal_gen.runtime.platforms.interface import Platform

        self.assertTrue(Platform.supports_distributed_p2p())
        self.assertTrue(Platform.supports_device_tensor_pickle())

    def test_device_tensor_pickle_is_unsupported(self):
        self.assertFalse(TpuPlatform.supports_device_tensor_pickle())

    def test_supporting_platforms_get_plain_pickle(self):
        import pickle

        from sglang.multimodal_gen.runtime.distributed import utils as dist_utils

        platform = MagicMock()
        platform.supports_device_tensor_pickle.return_value = True
        payload = [torch.arange(4, dtype=torch.float32), {"k": 1}]
        with patch.object(dist_utils.platforms, "current_platform", platform):
            self.assertEqual(dist_utils._dumps(payload), pickle.dumps(payload))

    def test_cpu_tensors_are_pickled_untouched(self):
        import pickle

        from sglang.multimodal_gen.runtime.distributed import utils as dist_utils

        platform = MagicMock()
        platform.supports_device_tensor_pickle.return_value = False
        payload = [torch.arange(6, dtype=torch.float32), "text", 7]
        with patch.object(dist_utils.platforms, "current_platform", platform):
            restored = pickle.loads(dist_utils._dumps(payload))
        self.assertTrue(torch.equal(restored[0], payload[0]))
        self.assertEqual(restored[1:], ["text", 7])

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
        from sglang_tpu.patches import _prepare_tpu_env

        self.assertTrue(callable(_prepare_tpu_env))

    def test_importing_the_platform_package_does_not_initialize_pjrt(self):
        # The parent resolves current_platform before spawning the worker;
        # PJRT is one client per process and the worker must be the one to own it.
        code = (
            "import torch, sglang.multimodal_gen.runtime.platforms as p; "
            "p.current_platform; import sglang.multimodal_gen.runtime.platforms.tpu; "
            "raise SystemExit(1 if torch.tpu.is_initialized() else 0)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])

    def test_worker_init_selects_cpu_only_jax_before_calling_srt(self):
        # A worker started without srt's entry-point activation in the parent
        # has no JAX_PLATFORMS; _prepare_tpu_env then raises once JAX is imported.
        from sglang_tpu import patches

        seen = {}

        def record(**kwargs):
            seen["JAX_PLATFORMS"] = os.environ.get("JAX_PLATFORMS")

        with (
            patch.dict(os.environ),
            patch.object(patches, "_prepare_tpu_env", side_effect=record),
        ):
            os.environ.pop("JAX_PLATFORMS", None)
            TpuPlatform._init_worker_process()
        self.assertEqual(seen.get("JAX_PLATFORMS"), "cpu")

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

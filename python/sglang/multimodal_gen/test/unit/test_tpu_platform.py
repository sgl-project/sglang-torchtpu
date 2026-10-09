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


class TestSplashBackendSelection(unittest.TestCase):
    def setUp(self):
        from sglang.multimodal_gen.runtime.layers.attention import selector

        selector._cached_get_attn_backend.cache_clear()
        self.addCleanup(selector._cached_get_attn_backend.cache_clear)

    def test_explicit_selection_overrides_the_layer_preference_set(self):
        from sglang.multimodal_gen.runtime.layers.attention.backends.tpu_splash_attn import (
            TpuSplashAttentionBackend,
        )
        from sglang.multimodal_gen.runtime.layers.attention.selector import (
            get_attn_backend,
        )

        with patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform", TpuPlatform()
        ):
            backend = get_attn_backend(
                128,
                torch.bfloat16,
                supported_attention_backends={
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.TORCH_SDPA,
                },
                selected_attention_backend=AttentionBackendEnum.TPU_SPLASH_ATTN,
            )
        self.assertIs(backend, TpuSplashAttentionBackend)

    def test_selectable_by_cli_name(self):
        # server_args resolves --attention-backend via the enum name.
        self.assertIs(
            AttentionBackendEnum["TPU_SPLASH_ATTN"],
            AttentionBackendEnum.TPU_SPLASH_ATTN,
        )
        self.assertEqual(str(AttentionBackendEnum.TPU_SPLASH_ATTN), "tpu_splash_attn")


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

    def test_attn_backend_splash_is_opt_in(self):
        self.assertIn(
            "tpu_splash_attn.TpuSplashAttentionBackend",
            TpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.TPU_SPLASH_ATTN, 128, torch.bfloat16
            ),
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

    def test_spatial_shard_decode_is_refused_without_p2p(self):
        from sglang.multimodal_gen.configs.models.vaes import base as vae_base
        from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig

        platform = MagicMock()
        platform.supports_distributed_p2p.return_value = False
        with patch.object(vae_base, "current_platform", platform):
            self.assertFalse(
                vae_base.should_use_spatial_shard_parallel_decode(
                    WanVAEConfig(), world_size=2
                )
            )

    def test_spatial_shard_decode_survives_where_p2p_works(self):
        from sglang.multimodal_gen.configs.models.vaes import base as vae_base
        from sglang.multimodal_gen.configs.models.vaes.wanvae import WanVAEConfig

        platform = MagicMock()
        platform.supports_distributed_p2p.return_value = True
        with patch.object(vae_base, "current_platform", platform):
            self.assertTrue(
                vae_base.should_use_spatial_shard_parallel_decode(
                    WanVAEConfig(), world_size=2
                )
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

    def test_compile_warmup_keeps_the_dit_resident(self):
        from sglang.multimodal_gen.configs.pipeline_configs.base import (
            PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.platforms import current_platform
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        def server_args(**kwargs):
            # ServerArgs.__post_init__ applies the platform defaults.
            with (
                patch.object(
                    PipelineConfig, "from_kwargs", return_value=PipelineConfig()
                ),
                patch.object(
                    current_platform,
                    "apply_server_args_defaults",
                    TpuPlatform().apply_server_args_defaults,
                ),
            ):
                return ServerArgs.from_kwargs(model_path="/fake", **kwargs)

        self.assertFalse(server_args().offload_during_compile)
        self.assertTrue(server_args(offload_during_compile=True).offload_during_compile)


class TestSdpaQueryBroadcastMask(unittest.TestCase):
    """torch_tpu SDPA returns NaN for masks that broadcast over queries."""

    @staticmethod
    def _tpu_query(*shape):
        query = MagicMock()
        query.device.type = "tpu"
        query.shape = torch.Size(shape)
        return query

    def test_key_padding_mask_is_expanded_over_queries(self):
        from sglang.multimodal_gen.runtime.platforms.tpu import (
            _expand_query_broadcast_mask,
        )

        mask = torch.zeros(2, 1, 1, 8)
        out = _expand_query_broadcast_mask(self._tpu_query(2, 4, 5, 64), mask)
        self.assertEqual(out.shape, (2, 1, 5, 8))
        # expand() is a view: no copy of the mask.
        self.assertEqual(out.data_ptr(), mask.data_ptr())

    def test_other_masks_are_untouched(self):
        from sglang.multimodal_gen.runtime.platforms.tpu import (
            _expand_query_broadcast_mask,
        )

        full = torch.zeros(2, 1, 5, 8)
        self.assertIs(
            _expand_query_broadcast_mask(self._tpu_query(2, 4, 5, 64), full), full
        )
        # Single-query attention needs no expansion.
        row = torch.zeros(2, 1, 1, 8)
        self.assertIs(
            _expand_query_broadcast_mask(self._tpu_query(2, 4, 1, 64), row), row
        )
        self.assertIsNone(
            _expand_query_broadcast_mask(self._tpu_query(2, 4, 5, 64), None)
        )
        # Non-TPU tensors are never touched.
        cpu_query = torch.zeros(2, 4, 5, 64)
        self.assertIs(_expand_query_broadcast_mask(cpu_query, row), row)

    def test_patch_is_idempotent_and_preserves_results(self):
        from sglang.multimodal_gen.runtime.platforms import tpu as tpu_mod

        current = torch.nn.functional.scaled_dot_product_attention
        # A TPU worker in this process may already have installed the patch.
        original = getattr(current, "__wrapped__", current)
        try:
            torch.nn.functional.scaled_dot_product_attention = original
            tpu_mod._patch_sdpa_query_broadcast_mask()
            patched = torch.nn.functional.scaled_dot_product_attention
            tpu_mod._patch_sdpa_query_broadcast_mask()
            self.assertIs(torch.nn.functional.scaled_dot_product_attention, patched)
            self.assertIs(patched.__wrapped__, original)

            q, k, v = (torch.randn(1, 2, 5, 8) for _ in range(3))
            mask = torch.zeros(1, 1, 1, 5)
            mask[..., :2] = float("-inf")
            torch.testing.assert_close(
                patched(q, k, v, attn_mask=mask), original(q, k, v, attn_mask=mask)
            )
            torch.testing.assert_close(
                patched(q, k, v, mask, 0.0, False), original(q, k, v, mask, 0.0, False)
            )
        finally:
            torch.nn.functional.scaled_dot_product_attention = current


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
        from sglang.srt.hardware_backend.tpu.patches import _prepare_tpu_env

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
        from sglang.srt.hardware_backend.tpu import patches

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

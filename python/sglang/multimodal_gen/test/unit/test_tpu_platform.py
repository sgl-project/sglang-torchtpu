# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the TPU platform. Cases needing torch_tpu self-skip."""

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.platforms import tpu_platform_plugin
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
)
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
        self.assertEqual(
            f"{backend.__module__}.{backend.__qualname__}",
            TpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.TPU_SPLASH_ATTN, 128, torch.bfloat16
            ),
        )

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

    def test_only_tpu_exchanges_attention_output_in_head_slabs(self):
        self.assertTrue(TpuPlatform.prefers_head_slab_output_exchange())
        self.assertFalse(Platform.prefers_head_slab_output_exchange())

    def test_attn_backend_default_is_sdpa(self):
        for selected in (None, AttentionBackendEnum.TORCH_SDPA):
            self.assertIn(
                "sdpa.SDPABackend",
                TpuPlatform.get_attn_backend_cls_str(selected, 128, torch.bfloat16),
            )

    def test_attn_backend_splash_class_path(self):
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
        # An explicit request is overridden: the warmup offload crashes on TPU.
        self.assertFalse(
            server_args(offload_during_compile=True).offload_during_compile
        )

    def test_teacache_is_dropped_under_torch_compile(self):
        from sglang.multimodal_gen.configs.sample.sampling_params import (
            SamplingParams,
        )
        from sglang.multimodal_gen.runtime.platforms import current_platform

        self.assertFalse(TpuPlatform().supports_teacache_with_torch_compile())
        for compile_on, expected in ((True, False), (False, True)):
            with (
                self.subTest(enable_torch_compile=compile_on),
                patch.object(
                    current_platform,
                    "supports_teacache_with_torch_compile",
                    TpuPlatform().supports_teacache_with_torch_compile,
                ),
            ):
                params = SamplingParams(enable_teacache=True)
                params._drop_teacache_if_it_recompiles(
                    SimpleNamespace(enable_torch_compile=compile_on)
                )
                self.assertEqual(params.enable_teacache, expected)


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
        # The plugin's worker hooks import these lazily; a rename on the srt
        # side must fail here, not at worker bootstrap.
        from sglang_tpu.patches import _prepare_tpu_env, _register_aten_fallbacks

        self.assertTrue(callable(_prepare_tpu_env))
        self.assertTrue(callable(_register_aten_fallbacks))

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


class TestFusionBarrier(unittest.TestCase):
    # CPU-only torch rejects torch.device("tpu"); only .device.type is read.
    _TPU_TENSOR = SimpleNamespace(device=SimpleNamespace(type="tpu"))

    def test_default_platform_returns_the_input(self):
        x = torch.arange(4.0)
        self.assertIs(Platform.fusion_barrier(x), x)

    def test_host_tensors_pass_through(self):
        x = torch.arange(4.0)
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertIs(TpuPlatform.fusion_barrier(x), x)

    def test_eager_tpu_tensors_pass_through(self):
        x = self._TPU_TENSOR
        self.assertIs(TpuPlatform.fusion_barrier(x), x)

    def test_compiled_tpu_tensors_take_the_torch_tpu_barrier(self):
        x = self._TPU_TENSOR
        public = MagicMock(return_value=["public"])
        raw = MagicMock(return_value=["raw"])
        for tpu_module, want in (
            (
                SimpleNamespace(ops=SimpleNamespace(optimization_barrier=public)),
                "public",
            ),
            (SimpleNamespace(), "raw"),
        ):
            with (
                self.subTest(want=want),
                patch.object(torch, "tpu", tpu_module, create=True),
                patch.object(
                    torch.ops, "tpu", SimpleNamespace(optimization_barrier=raw)
                ),
                patch.object(torch.compiler, "is_compiling", return_value=True),
            ):
                self.assertEqual(TpuPlatform.fusion_barrier(x), want)
        public.assert_called_once_with([x])
        raw.assert_called_once_with([x])

    def test_wan_ffn_runs_the_barrier_between_the_activation_and_fc_out(self):
        from sglang.multimodal_gen.runtime.models.dits import wanvideo

        calls = []

        def step(name, value):
            calls.append(name)
            return value

        mlp = SimpleNamespace(
            fc_in=lambda x: (step("fc_in", x), None),
            fc_out=lambda x: (step("fc_out", x), None),
            _apply_activation=lambda x, bias, use_fused_bias_gelu: step("act", x),
        )
        with (
            patch.object(wanvideo, "fused_gelu_active", return_value=False),
            patch.object(wanvideo, "nvfp4_bias_gelu_active", return_value=False),
            patch.object(
                wanvideo.current_platform,
                "fusion_barrier",
                side_effect=lambda x: step("barrier", x),
            ),
        ):
            wanvideo._WanGELUMLP.forward(mlp, torch.ones(2))
        self.assertEqual(calls, ["fc_in", "act", "barrier", "fc_out"])


if __name__ == "__main__":
    unittest.main()

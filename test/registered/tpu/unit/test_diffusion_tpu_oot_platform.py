"""TPU CI entry for the diffusion TPU plugin platform (sglang_tpu.multimodal_gen)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=15, suite="stage-a-test-tpu-unit", chips=1)

import unittest
from importlib.metadata import entry_points
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.layers.attention import selector
from sglang.multimodal_gen.runtime.platforms import (
    PLATFORM_PLUGINS_GROUP,
    _load_platform_class,
)
from sglang.multimodal_gen.runtime.platforms.interface import (
    AttentionBackendEnum,
    Platform,
)
from sglang.multimodal_gen.runtime.server_args import server_args as server_args_module
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen import PLATFORM_CLASS_PATH, activate
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

_SPLASH_BACKEND_CLS = (
    "sglang_tpu.multimodal_gen.tpu_splash_attn.TpuSplashAttentionBackend"
)


class TestActivation(CustomTestCase):
    def test_answers_only_when_torch_tpu_is_registered(self):
        with patch.object(torch, "tpu", object(), create=True):
            self.assertEqual(activate(), PLATFORM_CLASS_PATH)
        with patch.object(torch, "tpu", None, create=True):
            self.assertIsNone(activate())

    def test_entry_point_is_installed(self):
        values = {
            entry_point.name: entry_point.value
            for entry_point in entry_points(group=PLATFORM_PLUGINS_GROUP)
        }
        self.assertEqual(values.get("tpu_oot"), "sglang_tpu.multimodal_gen:activate")

    def test_core_loads_it_as_an_external_platform(self):
        self.assertIs(
            _load_platform_class(PLATFORM_CLASS_PATH, external=True), TpuOOTPlatform
        )


class TestOutOfTreeContract(CustomTestCase):
    def test_identity(self):
        platform = TpuOOTPlatform()
        self.assertTrue(platform.is_out_of_tree())
        self.assertTrue(platform.is_tpu())
        self.assertEqual(platform.device_type, "tpu")
        self.assertEqual(platform.get_dispatch_key_name(), "tpu")
        self.assertEqual(platform.get_torch_library_dispatch_key(), "PrivateUse1")

    def test_all_to_all_keeps_the_base_communicator(self):
        self.assertEqual(
            TpuOOTPlatform.get_all_to_all_communicator_cls(),
            Platform.get_device_communicator_cls(),
        )


class TestSplashBackend(CustomTestCase):
    def setUp(self):
        selector._cached_get_attn_backend.cache_clear()
        self.addCleanup(selector._cached_get_attn_backend.cache_clear)

    def test_the_platform_names_and_resolves_it(self):
        self.assertEqual(TpuOOTPlatform.get_oot_attn_backend_name(), "tpu_splash_attn")
        self.assertEqual(
            TpuOOTPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.OOT, 128, torch.bfloat16
            ),
            _SPLASH_BACKEND_CLS,
        )
        for selected in (None, AttentionBackendEnum.TORCH_SDPA):
            self.assertIn(
                "sdpa.SDPABackend",
                TpuOOTPlatform.get_attn_backend_cls_str(selected, 128, torch.bfloat16),
            )

    def test_explicit_selection_overrides_the_layer_preference_set(self):
        from sglang_tpu.multimodal_gen.tpu_splash_attn import (
            TpuSplashAttentionBackend,
        )

        with patch(
            "sglang.multimodal_gen.runtime.platforms.current_platform",
            TpuOOTPlatform(),
        ):
            backend = selector.get_attn_backend(
                128,
                torch.bfloat16,
                supported_attention_backends={
                    AttentionBackendEnum.FA,
                    AttentionBackendEnum.TORCH_SDPA,
                },
                selected_attention_backend=AttentionBackendEnum.OOT,
            )
        self.assertIs(backend, TpuSplashAttentionBackend)
        self.assertIs(backend.get_enum(), AttentionBackendEnum.OOT)

    def test_cli_name_normalizes_to_oot(self):
        normalize = server_args_module.ServerArgs._normalize_attention_backend_name
        with patch.object(server_args_module, "current_platform", TpuOOTPlatform()):
            self.assertEqual(normalize("tpu_splash_attn"), "oot")
            self.assertEqual(normalize("torch_sdpa"), "torch_sdpa")
        # The name is the platform's: elsewhere it is not a backend.
        no_oot_backend = SimpleNamespace(get_oot_attn_backend_name=lambda: None)
        with patch.object(server_args_module, "current_platform", no_oot_backend):
            with self.assertRaises(ValueError):
                normalize("tpu_splash_attn")


class TestCompileDefaults(CustomTestCase):
    def test_compile_warmup_keeps_the_dit_resident(self):
        from sglang.multimodal_gen.configs.pipeline_configs.base import (
            PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.platforms import current_platform

        def server_args(**kwargs):
            # ServerArgs.__post_init__ applies the platform defaults.
            with (
                patch.object(
                    PipelineConfig, "from_kwargs", return_value=PipelineConfig()
                ),
                patch.object(
                    current_platform,
                    "apply_server_args_defaults",
                    TpuOOTPlatform().apply_server_args_defaults,
                ),
            ):
                return server_args_module.ServerArgs.from_kwargs(
                    model_path="/fake", **kwargs
                )

        self.assertFalse(server_args().offload_during_compile)
        # An explicit request is overridden: the warmup offload crashes on TPU.
        self.assertFalse(
            server_args(offload_during_compile=True).offload_during_compile
        )

    def test_teacache_and_vae_decode_stay_out_of_compile(self):
        # What the core does with each False: TestPlatformCompileGates.
        self.assertFalse(TpuOOTPlatform().supports_teacache_with_torch_compile())
        self.assertFalse(TpuOOTPlatform.supports_vae_torch_compile())

    def test_worker_init_keeps_recompiles_static_and_patches_the_worker(self):
        import torch._dynamo

        from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform
        from sglang_tpu.multimodal_gen import worker

        with (
            torch._dynamo.config.patch(automatic_dynamic_shapes=True),
            patch.object(TpuPlatform, "_init_worker_process") as base_init,
            patch.object(worker, "install_runtime_patches") as install,
        ):
            TpuOOTPlatform._init_worker_process()
            self.assertFalse(torch._dynamo.config.automatic_dynamic_shapes)
        base_init.assert_called_once_with()
        install.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

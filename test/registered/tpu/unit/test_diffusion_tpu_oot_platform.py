"""TPU CI entry for the diffusion TPU plugin platform (sglang_tpu.multimodal_gen)."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=20, suite="stage-a-test-tpu-unit", chips=1)

import importlib.util
import sys
import unittest
from importlib.metadata import entry_points
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
import torch.compiler.config as compiler_config

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

    def test_attention_output_is_exchanged_in_head_slabs(self):
        self.assertTrue(TpuOOTPlatform.prefers_head_slab_output_exchange())
        self.assertFalse(Platform.prefers_head_slab_output_exchange())

    def test_safetensors_are_read_on_the_host(self):
        self.assertFalse(TpuOOTPlatform.supports_safetensors_device_load())
        self.assertTrue(Platform.supports_safetensors_device_load())

    def test_every_request_drains_the_device_queue(self):
        # Without it Wan2.2-A14B's second warmup request hangs; see the hook.
        self.assertTrue(TpuOOTPlatform.drain_device_queue_after_request())
        self.assertFalse(Platform.drain_device_queue_after_request())


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

    def test_it_serves_layers_that_take_sdpa(self):
        fa, sdpa = AttentionBackendEnum.FA, AttentionBackendEnum.TORCH_SDPA
        self.assertTrue(TpuOOTPlatform.oot_attn_backend_serves({fa, sdpa}))
        self.assertFalse(TpuOOTPlatform.oot_attn_backend_serves({fa}))
        self.assertFalse(Platform.oot_attn_backend_serves({fa, sdpa}))

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


_CACHE_TAG = "spmd-safe-regions-v1"


def _torch_tpu_stubs(spmd_safe=None):
    names = (
        "torch_tpu",
        "torch_tpu._internal",
        "torch_tpu._internal.distributed",
        "torch_tpu._internal.distributed.spmd_util",
    )
    stubs = {name: ModuleType(name) for name in names}
    if spmd_safe is not None:
        stubs[names[-1]].spmd_safe = spmd_safe
    return stubs


class TestTpuSpmdRegionWrapper(CustomTestCase):
    def test_wrapper_tags_the_compile_cache_once(self):
        spmd_safe = object()
        with (
            patch.dict(sys.modules, _torch_tpu_stubs(spmd_safe)),
            compiler_config.patch(cache_key_tag="user"),
        ):
            # Once per compiled model: Wan2.2-A14B compiles two transformers.
            for _ in range(2):
                self.assertIs(TpuOOTPlatform().prepare_spmd_region_wrapper(), spmd_safe)
            self.assertEqual(compiler_config.cache_key_tag, f"user|{_CACHE_TAG}")

    def test_torch_tpu_without_spmd_safe_leaves_regions_unwrapped(self):
        # spmd_safe is a private torch_tpu API: without it, regions compile
        # with split collectives and the cache key keeps its tags.
        with (
            patch.dict(sys.modules, _torch_tpu_stubs()),
            compiler_config.patch(cache_key_tag="user"),
        ):
            self.assertIsNone(TpuOOTPlatform().prepare_spmd_region_wrapper())
            self.assertEqual(compiler_config.cache_key_tag, "user")

    @unittest.skipUnless(importlib.util.find_spec("torch_tpu"), "requires torch_tpu")
    def test_wrapper_is_torch_tpu_spmd_safe(self):
        from torch_tpu._internal.distributed.spmd_util import spmd_safe

        with compiler_config.patch(cache_key_tag=""):
            self.assertIs(TpuOOTPlatform().prepare_spmd_region_wrapper(), spmd_safe)


class TestTpuFusionBarrier(CustomTestCase):
    # Only .device.type is read, so a stand-in needs no TPU device.
    _TPU_TENSOR = SimpleNamespace(device=SimpleNamespace(type="tpu"))

    def test_host_tensors_pass_through(self):
        x = torch.arange(4.0)
        with patch.object(torch.compiler, "is_compiling", return_value=True):
            self.assertIs(TpuOOTPlatform.fusion_barrier(x), x)

    def test_compiled_tpu_tensors_take_the_torch_tpu_barrier(self):
        x = self._TPU_TENSOR
        barrier = MagicMock(return_value=["barrier"])
        tpu_module = SimpleNamespace(ops=SimpleNamespace(optimization_barrier=barrier))
        with (
            patch.object(torch, "tpu", tpu_module, create=True),
            patch.object(torch.compiler, "is_compiling", return_value=True),
        ):
            self.assertEqual(TpuOOTPlatform.fusion_barrier(x), "barrier")
        barrier.assert_called_once_with([x])


class TestFusionBarrierOnDevice(CustomTestCase):
    def test_values_pass_through_eager_and_a_compiled_graph(self):
        import torch_tpu._internal.compile  # noqa: F401
        from torch._dynamo.backends.registry import lookup_backend

        TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())
        x = (torch.randn(256, 512) * 3).bfloat16().to("tpu")
        # Eager runs op by op: there is no fusion to block, so no copy either.
        self.assertIs(TpuOOTPlatform.fusion_barrier(x), x)

        targets = []
        tpu_backend = lookup_backend("tpu")

        def recording_backend(gm, example_inputs):
            targets.extend(str(node.target) for node in gm.graph.nodes)
            return tpu_backend(gm, example_inputs)

        def mlp_act(t):
            h = torch.nn.functional.gelu(t, approximate="tanh")
            return TpuOOTPlatform.fusion_barrier(h) * 2

        compiled = torch.compile(
            mlp_act, backend=recording_backend, dynamic=False, fullgraph=True
        )
        torch.testing.assert_close(
            compiled(x).cpu(), mlp_act(x).cpu(), rtol=1.6e-2, atol=1e-2
        )
        self.assertTrue(any("optimization_barrier" in t for t in targets), targets)


class TestWorkerRuntimePatches(CustomTestCase):
    def test_worker_registers_the_aten_fallbacks_and_patches(self):
        # Ring attention merges replicated-token partials with logaddexp, which
        # torch_tpu has no kernel for (test_aten_fallbacks_tpu checks the kernel).
        from sglang_tpu import patches
        from sglang_tpu.multimodal_gen import hunyuan3d, minimax_h3, rope, worker

        with (
            patch.object(patches, "_register_aten_fallbacks") as aten_fallbacks,
            patch.object(rope, "install") as rope_install,
            patch.object(minimax_h3, "install") as minimax_h3_install,
            patch.object(hunyuan3d, "install") as hunyuan3d_install,
        ):
            worker.install_runtime_patches()
        aten_fallbacks.assert_called_once_with()
        rope_install.assert_called_once_with()
        minimax_h3_install.assert_called_once_with()
        hunyuan3d_install.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

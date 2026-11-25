"""Unit tests for the sglang-diffusion (multimodal_gen) TPU platform of the plugin,
and TPU CI entry for the core platform-detection, attention-selector and
server-args cases it relies on. Cases needing torch_tpu self-skip or need a TPU."""

from sglang.test.ci.ci_register import register_tpu_ci

register_tpu_ci(est_time=40, suite="stage-a-test-tpu-unit", chips=1)

import json
import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.test.unit.test_attention_backend_selector import (  # noqa: F401
    TestAttentionBackendFallback,
)
from sglang.multimodal_gen.test.unit.test_platform_detection import (  # noqa: F401
    TestDiffusionPlatformPlugins,
)
from sglang.multimodal_gen.test.unit.test_server_args import (  # noqa: F401
    TestRingCapableAttentionBackends,
    TestTpuDistributedArgs,
)
from sglang.test.test_utils import CustomTestCase
from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform, _init_torch_tpu_runtime

_HAS_TORCH_TPU = hasattr(torch, "tpu")


class TestTpuPlatformSurface(CustomTestCase):
    def test_identity(self):
        self.assertEqual(TpuOOTPlatform.device_type, "tpu")
        self.assertTrue(TpuOOTPlatform.get_device_name().lower().startswith("tpu"))
        self.assertTrue(TpuOOTPlatform().is_tpu())

    def test_float64_supported(self):
        self.assertTrue(TpuOOTPlatform.is_float64_supported())

    def test_inference_mode_is_no_grad(self):
        self.assertIsInstance(TpuOOTPlatform.inference_mode(), torch.no_grad)

    def test_attn_backend_unsupported_raises(self):
        with self.assertRaises(ValueError):
            TpuOOTPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.FA, 128, torch.bfloat16
            )

    def test_lone_worker_distributed_backend_is_gloo(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            self.assertEqual(
                TpuOOTPlatform().get_torch_distributed_backend_str(), "gloo"
            )

    def test_multi_worker_distributed_backend_is_tpu_dist(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "16"}):
            self.assertEqual(
                TpuOOTPlatform().get_torch_distributed_backend_str(), "tpu_dist"
            )

    def test_p2p_is_unsupported(self):
        self.assertFalse(TpuOOTPlatform.supports_distributed_p2p())

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
        self.assertFalse(TpuOOTPlatform.supports_device_tensor_pickle())

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
        self.assertFalse(TpuOOTPlatform.enable_dit_layerwise_offload_by_default())

    def test_memory_shims_report_allocated_bytes_as_reserved(self):
        # PJRT has no caching allocator: reserved is the allocated bytes, and
        # gpu_worker sizes residency off reserved and its peak.
        from sglang_tpu.multimodal_gen.platform import _DEVICE_MODULE_SHIMS

        stats = {"allocated_bytes.all.current": 5, "allocated_bytes.all.peak": 9}
        for initialized, current, peak in ((True, 5, 9), (False, 0, 0)):
            fake_tpu = SimpleNamespace(is_initialized=lambda up=initialized: up)
            with (
                self.subTest(initialized=initialized),
                patch.object(torch, "tpu", fake_tpu, create=True),
                patch.object(
                    torch.accelerator, "memory_stats", return_value=stats
                ) as memory_stats,
            ):
                self.assertEqual(
                    [
                        _DEVICE_MODULE_SHIMS[name]()
                        for name in (
                            "memory_allocated",
                            "memory_reserved",
                            "max_memory_allocated",
                            "max_memory_reserved",
                        )
                    ],
                    [current, current, peak, peak],
                )
                # Before PJRT is up the launcher may call them: no device query.
                self.assertEqual(memory_stats.called, initialized)


@unittest.skipUnless(_HAS_TORCH_TPU, "torch.device('tpu') needs torch_tpu")
class TestWorkerInitOnce(CustomTestCase):
    def setUp(self):
        self.addCleanup(setattr, TpuOOTPlatform, "_worker_initialized", False)
        TpuOOTPlatform._worker_initialized = False

    def test_init_runs_once_across_call_sites(self):
        with (
            patch.object(TpuOOTPlatform, "_init_worker_process") as init,
            patch.object(torch, "tpu", MagicMock(), create=True),
        ):
            TpuOOTPlatform.set_device(torch.device("tpu", 0))
            TpuOOTPlatform.set_device(torch.device("tpu", 0))
        self.assertEqual(init.call_count, 1)

    def test_failed_init_is_retryable(self):
        with (
            patch.object(
                TpuOOTPlatform, "_init_worker_process", side_effect=RuntimeError("boom")
            ),
            patch.object(torch, "tpu", MagicMock(), create=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                TpuOOTPlatform.set_device(torch.device("tpu", 0))
        self.assertFalse(TpuOOTPlatform._worker_initialized)


@unittest.skipUnless(_HAS_TORCH_TPU, "torch_tpu not installed")
class TestTorchTpuIntegration(CustomTestCase):
    def test_pre_init_memory_queries_do_not_initialize_pjrt(self):
        # The launcher sizes offload with these before any worker exists, and
        # the worker must be the one to own PJRT. A fresh interpreter: an
        # earlier case may have initialized PJRT in this one.
        code = textwrap.dedent(
            """
            import json, torch
            from sglang_tpu.multimodal_gen.platform import TpuOOTPlatform

            queries = {
                "total": TpuOOTPlatform.get_device_total_memory(),
                "available": TpuOOTPlatform.get_available_gpu_memory(),
                "used": TpuOOTPlatform.get_current_memory_usage(),
                "name": TpuOOTPlatform.get_device_name(),
            }
            queries["pjrt_initialized"] = torch.tpu.is_initialized()
            print("QUERIES", json.dumps(queries))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
        )
        self.assertEqual(completed.returncode, 0, completed.stderr[-2000:])
        [line] = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("QUERIES ")
        ]
        queries = json.loads(line.removeprefix("QUERIES "))
        self.assertFalse(queries["pjrt_initialized"], queries)
        self.assertGreater(queries["total"], 0)
        self.assertGreater(queries["available"], 0.0)
        self.assertEqual(queries["used"], 0.0)
        self.assertTrue(queries["name"].startswith("TPU "), queries["name"])

    def test_srt_symbols_the_platform_imports_exist(self):
        # The platform imports this lazily; a rename on the srt side must fail here,
        # not at worker bootstrap.
        from sglang_tpu.patches import _prepare_tpu_env

        self.assertTrue(callable(_prepare_tpu_env))

    def test_importing_the_platform_package_does_not_initialize_pjrt(self):
        # The parent resolves current_platform before spawning the worker;
        # PJRT is one client per process and the worker must be the one to own it.
        code = (
            "import torch, sglang.multimodal_gen.runtime.platforms as p; "
            "p.current_platform; import sglang_tpu.multimodal_gen.platform; "
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
            _init_torch_tpu_runtime()
        self.assertEqual(seen.get("JAX_PLATFORMS"), "cpu")

    def test_device_module_shims_installed_once(self):
        from sglang_tpu.multimodal_gen.platform import _install_device_module_shims

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


class TestWorkerBootstrap(CustomTestCase):
    """Real-hardware worker initialization smoke test."""

    def test_set_device_brings_up_the_runtime(self):
        TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())
        self.assertTrue(torch.tpu.is_initialized())

        x = torch.ones((8, 128), device="tpu", dtype=torch.bfloat16)
        y = (x + x).sum()
        torch.accelerator.synchronize()
        self.assertEqual(y.item(), 2 * 8 * 128)

        self.assertGreater(TpuOOTPlatform.get_device_total_memory(), 0)
        self.assertGreater(TpuOOTPlatform.get_current_memory_usage(), 0.0)

    def test_device_tensors_survive_a_pickle_round_trip(self):
        import pickle

        from sglang.multimodal_gen.runtime.distributed import utils as dist_utils

        TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())
        contiguous = torch.arange(12, dtype=torch.float32, device="tpu") * 2.0
        strided = contiguous.view(3, 4).transpose(0, 1)
        for original in (contiguous, strided):
            restored = pickle.loads(dist_utils._dumps([original]))[0]
            self.assertEqual(restored.device.type, "tpu")
            self.assertEqual(restored.shape, original.shape)
            self.assertEqual(restored.stride(), original.stride())
            self.assertEqual(restored.is_contiguous(), original.is_contiguous())
            self.assertTrue(torch.equal(restored.cpu(), original.cpu()))

    def test_requires_grad_survives_the_round_trip(self):
        import pickle

        from sglang.multimodal_gen.runtime.distributed import utils as dist_utils

        TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())
        for requires_grad in (False, True):
            original = torch.ones(4, dtype=torch.float32, device="tpu")
            original.requires_grad_(requires_grad)
            restored = pickle.loads(dist_utils._dumps([original]))[0]
            self.assertEqual(restored.requires_grad, requires_grad)


class TestFloat64IsRealOnDevice(CustomTestCase):
    """The measurements behind `TpuOOTPlatform.is_float64_supported() -> True`."""

    def test_elementwise_float64_matches_the_host(self):
        TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())
        self.assertTrue(TpuOOTPlatform.is_float64_supported())
        inv = 1.0 / (
            10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64, device="tpu") / 64)
        )
        pos = torch.arange(128, dtype=torch.float64, device="tpu")
        got = torch.outer(pos, inv).cos().cpu()

        ref_inv = 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float64) / 64))
        ref = torch.outer(torch.arange(128, dtype=torch.float64), ref_inv).cos()
        self.assertEqual(got.dtype, torch.float64)
        self.assertLess((got - ref).abs().max().item(), 1e-12)

    def test_float64_matmul_still_downcasts(self):
        TpuOOTPlatform.set_device(TpuOOTPlatform.get_local_torch_device())
        a = torch.linspace(0, 1, 64, device="tpu").double().reshape(8, 8)
        got = (a @ a.t()).cpu()
        ref_a = torch.linspace(0, 1, 64).double().reshape(8, 8)
        err = (got - ref_a @ ref_a.t()).abs().max().item()
        self.assertGreater(
            err,
            1e-12,
            "float64 matmul now matches the host: widen "
            "TpuOOTPlatform.is_float64_supported's comment and re-check minimax_h3",
        )


if __name__ == "__main__":
    unittest.main()

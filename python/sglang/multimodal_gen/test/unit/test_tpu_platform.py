"""CPU-safe tests for the TPU platform plugin and capability surface.

These run without torch_tpu installed: the platform module defers all
torch_tpu imports to call time.
"""

import importlib.util
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime import platforms
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum
from sglang.multimodal_gen.runtime.platforms.tpu import TpuPlatform

_TPU_QUALNAME = "sglang.multimodal_gen.runtime.platforms.tpu.TpuPlatform"


class TestTpuPlatformPlugin(unittest.TestCase):
    def test_plugin_detects_installed_torch_tpu(self):
        with patch.object(importlib.util, "find_spec", return_value=object()):
            self.assertEqual(platforms.tpu_platform_plugin(), _TPU_QUALNAME)

    def test_plugin_returns_none_without_torch_tpu(self):
        with patch.object(importlib.util, "find_spec", return_value=None):
            self.assertIsNone(platforms.tpu_platform_plugin())

    def test_forced_override_resolves_tpu(self):
        with patch.dict(
            "os.environ", {"SGLANG_DIFFUSION_PLATFORM_OVERRIDE": "tpu"}
        ):
            self.assertEqual(
                platforms.resolve_current_platform_cls_qualname(), _TPU_QUALNAME
            )


class TestTpuPlatformSurface(unittest.TestCase):
    def test_identity(self):
        self.assertEqual(TpuPlatform.device_type, "tpu")
        self.assertEqual(TpuPlatform.dispatch_key, "PrivateUse1")
        self.assertEqual(TpuPlatform.simple_compile_backend, "tpu")

    def test_capabilities(self):
        self.assertFalse(TpuPlatform.supports_op_level_torch_compile())
        self.assertFalse(TpuPlatform.supports_p2p_communication())
        self.assertFalse(TpuPlatform.supports_vae_decode_torch_compile())
        self.assertFalse(TpuPlatform.is_float64_supported())
        self.assertTrue(TpuPlatform.use_functional_seq_all_to_all())

    def test_attention_backend_selection(self):
        default = TpuPlatform.get_attn_backend_cls_str(None, 128, torch.bfloat16)
        self.assertTrue(default.endswith("TpuPallasAttentionBackend"))
        explicit = TpuPlatform.get_attn_backend_cls_str(
            AttentionBackendEnum.TPU_PALLAS_FLASH, 128, torch.bfloat16
        )
        self.assertEqual(explicit, default)
        sdpa = TpuPlatform.get_attn_backend_cls_str(
            AttentionBackendEnum.TORCH_SDPA, 128, torch.bfloat16
        )
        self.assertTrue(sdpa.endswith("SDPABackend"))
        with self.assertRaises(ValueError):
            TpuPlatform.get_attn_backend_cls_str(
                AttentionBackendEnum.FA, 128, torch.bfloat16
            )

    def test_worker_spawn_env(self):
        env = TpuPlatform.worker_spawn_env(rank=3, world_size=8, master_port=1234)
        self.assertEqual(
            env,
            {
                "LOCAL_RANK": "3",
                "RANK": "3",
                "WORLD_SIZE": "8",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "1234",
            },
        )

    def test_inference_mode_is_no_grad(self):
        with TpuPlatform.inference_mode():
            self.assertFalse(torch.is_grad_enabled())


if __name__ == "__main__":
    unittest.main()
